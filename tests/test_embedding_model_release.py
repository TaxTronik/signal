from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from risk_layer.embedding import encoder_oeffnen
from risk_layer.embedding_model import (
    BGE_M3_REQUIRED_FILES,
    BGE_M3_REVISION,
    EmbeddingModelError,
    MODEL_MANIFEST_NAME,
    validate_local_bge_m3,
    validate_local_bge_m3_cached,
    write_model_manifest,
)
from tools.snapshot_embedding_model import snapshot


def _model(tmp_path: Path) -> Path:
    root = tmp_path / "bge-m3"
    root.mkdir(parents=True)
    for relative in BGE_M3_REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"test:{relative}".encode())
    write_model_manifest(root)
    return root


def test_bge_m3_manifest_bindet_revision_und_exaktes_inventar(tmp_path: Path) -> None:
    root = _model(tmp_path)
    assert validate_local_bge_m3(root) == root.resolve()

    manifest = json.loads((root / MODEL_MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    (root / MODEL_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EmbeddingModelError, match="falsche Revision"):
        validate_local_bge_m3(root)


def test_bge_m3_verwirft_fehlende_und_alternative_gewichte(tmp_path: Path) -> None:
    root = _model(tmp_path)
    (root / "pytorch_model.bin").unlink()
    with pytest.raises(EmbeddingModelError, match="fehlt|falsche Größe"):
        validate_local_bge_m3(root)

    root = _model(tmp_path / "zweites")
    onnx = root / "onnx/model.onnx"
    onnx.parent.mkdir()
    onnx.write_bytes(b"alternative")
    with pytest.raises(EmbeddingModelError, match="unerlaubte Dateien"):
        validate_local_bge_m3(root)


def test_offline_encoder_nutzt_nur_manifestierten_lokalen_pfad(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _model(tmp_path)
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        @staticmethod
        def get_embedding_dimension() -> int:
            return 1024

    def fake_sentence_transformer(name: str, **kwargs):
        calls.append((name, kwargs))
        return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )
    monkeypatch.setenv("RISK_LAYER_EMBEDDING_MODEL", str(root))
    monkeypatch.setenv("RISK_LAYER_EMBEDDING_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    validate_local_bge_m3_cached.cache_clear()

    encoder = encoder_oeffnen("bge-m3", device="cpu")
    encoder._laden()

    assert encoder.revision == BGE_M3_REVISION
    assert calls == [(str(root.resolve()), {
        "device": "cpu",
        "trust_remote_code": False,
        "revision": None,
        "local_files_only": True,
    })]


def test_offline_encoder_hat_keinen_remote_fallback(monkeypatch) -> None:
    monkeypatch.delenv("RISK_LAYER_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("RISK_LAYER_EMBEDDING_OFFLINE", "1")
    with pytest.raises(EmbeddingModelError, match="verlangt.*MODEL|--model"):
        encoder_oeffnen("bge-m3")


def test_snapshot_pinnt_commit_und_kopiert_keine_onnx_alternative(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeApi:
        @staticmethod
        def model_info(repo_id: str, *, revision: str):
            calls["info"] = (repo_id, revision)
            return SimpleNamespace(sha=BGE_M3_REVISION)

    def fake_download(**kwargs):
        calls["download"] = kwargs
        local = Path(kwargs["local_dir"])
        for relative in BGE_M3_REQUIRED_FILES:
            path = local / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"snapshot:{relative}".encode())
        alternative = local / "onnx/model.onnx"
        alternative.parent.mkdir()
        alternative.write_bytes(b"darf nicht ins Release")
        return str(local)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_download),
    )
    output = tmp_path / "snapshot"
    snapshot(output)

    assert validate_local_bge_m3(output) == output.resolve()
    assert not (output / "onnx").exists()
    assert calls["info"] == ("BAAI/bge-m3", BGE_M3_REVISION)
    download = calls["download"]
    assert download["revision"] == BGE_M3_REVISION
    assert download["max_workers"] == 1
    assert "onnx/*" in download["ignore_patterns"]
