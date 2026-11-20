"""Tiny CPU fixtures for a download-only, pinned MoE setup inspector."""
import copy
import hashlib
import io
import struct

import pytest

from tools import quality_research_moe_model as setup


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def fixture_gguf(path, architecture="qwen35moe"):
    metadata = [("general.architecture", 8, string(architecture)),
                ("qwen35moe.block_count", 4, struct.pack("<I", 40)),
                ("qwen35moe.embedding_length", 4, struct.pack("<I", 2048))]
    raw = struct.pack("<4sIQQ", b"GGUF", 3, 40, len(metadata))
    for key, kind, value in metadata:
        raw += string(key) + struct.pack("<I", kind) + value
    for number in range(40):
        raw += string(f"blk.{number}.ffn_up_exps.weight") + struct.pack("<IQIQ", 1, 16, 0, number * 64)
    raw += b"\0" * ((-len(raw)) % 32)
    raw += b"\0" * (40 * 64)
    path.write_bytes(raw)
    return path


def remote():
    return {"id": setup.REPO, "sha": setup.REVISION, "siblings": [{"rfilename": setup.FILENAME,
        "size": setup.MODEL_BYTES, "lfs": {"size": setup.MODEL_BYTES, "sha256": setup.MODEL_SHA}}]}


def test_pinned_remote_metadata_matches():
    setup.validate_remote(remote())


@pytest.mark.parametrize("field", ["repo", "revision", "size", "sha", "missing", "duplicate"])
def test_changed_remote_metadata_rejected(field):
    value = copy.deepcopy(remote())
    if field == "repo":
        value["id"] = "different/model"
    elif field == "revision":
        value["sha"] = "0" * 40
    elif field == "size":
        value["siblings"][0]["size"] -= 1
    elif field == "sha":
        value["siblings"][0]["lfs"]["sha256"] = "0" * 64
    elif field == "missing":
        value["siblings"] = []
    else:
        value["siblings"].append(value["siblings"][0])
    with pytest.raises(ValueError):
        setup.validate_remote(value)


def test_original_file_size_and_sha_required(tmp_path):
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"GGUF")
    assert setup.verified_model(path, 4, hashlib.sha256(b"GGUF").hexdigest())["bytes"] == 4
    with pytest.raises(ValueError, match="size"):
        setup.verified_model(path, 5, "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        setup.verified_model(path, 4, "0" * 64)


def test_directory_only_roundtrip_and_explicit_offload_estimate(tmp_path):
    path = fixture_gguf(tmp_path / "fixture.gguf")
    inventory = setup.inspect_gguf(path)
    plan = setup.offload_plan(inventory)
    assert inventory["tensor_count"] == 40 and inventory["tensor_storage_bytes"] == 2560
    assert inventory["data_start"] % 32 == 0
    assert plan["proposed_cpu_expert_storage_bytes"] == 16 * 64
    assert plan["remaining_tensor_storage_upper_bound_bytes"] == 24 * 64
    assert plan["attention_kv_bytes_at_16k_f16"] == 335544320
    assert plan["status"] == "planned_not_run"


def test_wrong_gguf_architecture_never_generates_a_plan(tmp_path):
    inventory = setup.inspect_gguf(fixture_gguf(tmp_path / "wrong.gguf", "qwen35"))
    with pytest.raises(ValueError, match="architecture"):
        setup.offload_plan(inventory)


def test_missing_expert_layer_cannot_be_counted_as_offloaded(tmp_path):
    inventory = setup.inspect_gguf(fixture_gguf(tmp_path / "fixture.gguf"))
    inventory["tensors"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        setup.offload_plan(inventory)


@pytest.mark.parametrize("raw", [b"GGUF", struct.pack("<4sIQQ", b"GGUF", 2, 1, 1), b"not gguf" * 5])
def test_truncated_or_unsupported_header_rejected(tmp_path, raw):
    path = tmp_path / "bad.gguf"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        setup.inspect_gguf(path)


def test_metadata_arrays_can_be_skipped_without_decoding_token_bytes():
    raw = struct.pack("<IQ", 8, 2) + struct.pack("<Q", 1) + b"\xff" + string("fine")
    reader = setup.HeaderReader(io.BytesIO(raw))
    assert reader.value(9, keep=False) is None
    assert reader.stream.tell() == len(raw)


def test_truncated_metadata_array_is_rejected():
    reader = setup.HeaderReader(io.BytesIO(struct.pack("<IQ", 4, 4) + b"\0" * 4))
    with pytest.raises(ValueError, match="Truncated"):
        reader.value(9, keep=False)
