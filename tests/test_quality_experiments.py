"""Cross-run cache reuse must retain verified model and artifact provenance."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import quality_experiments as experiments


PACKAGES = {"numpy": "2.2.6", "torch": "2.9.1+rocm7.2.1",
            "transformers": "5.15.1", "sentence-transformers": "5.6.0"}
REVISIONS = {"encoder_revision": "fixture-bge-m3", "reranker_revision": "fixture-bge-reranker"}
ARTIFACTS = [f"{stem}{suffix}" for stem in ("queries", "passages", "anchors")
             for suffix in (".npy", ".texts.json")] + ["new-scores.jsonl"]


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    for stem in ("queries", "passages", "anchors"):
        np.save(directory / f"{stem}.npy", np.array([[1., 0.], [0., 1.]], dtype=np.float32))
        experiments.save_json(directory / f"{stem}.texts.json", ["first exact input", "second exact input"])
    (directory / "new-scores.jsonl").write_text(
        json.dumps(["query", "document", .75]) + "\n", encoding="utf-8")
    manifest = {"format": "quality_experiments/2", "packages": dict(PACKAGES)}
    experiments.save_json(directory / "manifest.json", manifest)
    experiments.save_json(directory / "report.json", {
        "manifest": manifest, "runtime": {**REVISIONS, "torch": PACKAGES["torch"]},
        "rows": [], "summaries": {},
        "artifact_hashes": {name: experiments.sha256(directory / name)
                            for name in [*ARTIFACTS, "manifest.json"]},
    })
    return directory


def validate(source, **kwargs):
    return experiments.validiere_reuse(source, **{**REVISIONS, "packages": PACKAGES, **kwargs})


def test_missing_archived_index_has_clear_error_before_graph_or_model_load(tmp_path, monkeypatch):
    evidence = tmp_path / "baseline"
    evidence.mkdir()
    experiments.save_json(evidence / "report.json", {"source_hashes": {}, "rows": []})
    # Reach the artifact check in the CPU-only CI without loading model weights.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(experiments.importlib.metadata, "version", PACKAGES.__getitem__)
    monkeypatch.setattr(os, "environ", os.environ.copy())

    with pytest.raises(ValueError, match="Missing archived embedding index") as error:
        experiments.main(["--baseline-evidence", str(evidence), "--out", str(tmp_path / "run")])
    assert str(evidence / "artifacts" / "index") in str(error.value)
    assert not (evidence / "artifacts").exists()


def forbidden_encoder():
    def encode(_):
        pytest.fail("Matching verified cache must not invoke a model")

    return SimpleNamespace(dim=2, encode=encode)


def test_completed_legacy_report_reuses_exact_vectors_texts_and_scores(source, tmp_path):
    reuse = validate(source)
    target = tmp_path / "target"
    target.mkdir()
    expected = np.load(source / "queries.npy")
    result = experiments.encode_cached(["first exact input", "second exact input"],
                                       target / "queries.npy", forbidden_encoder(), reuse_from=reuse)
    assert np.array_equal(result, expected)
    reuse.copy_files(["new-scores.jsonl"], target)
    assert set(reuse.copied) == {"queries.npy", "queries.texts.json", "new-scores.jsonl"}
    for name, checksum in reuse.copied.items():
        assert checksum == experiments.sha256(source / name) == experiments.sha256(target / name)
    assert reuse.report_sha256 == experiments.sha256(source / "report.json")


@pytest.mark.parametrize("field", ["encoder_revision", "reranker_revision"])
def test_foreign_model_revision_cannot_reuse_identical_texts(source, field):
    with pytest.raises(ValueError, match="model revision"):
        validate(source, **{field: "different-weights"})


@pytest.mark.parametrize("package", experiments.REUSE_PACKAGES)
def test_all_inference_package_versions_must_match_exactly(source, package):
    with pytest.raises(ValueError, match="package versions"):
        validate(source, packages={**PACKAGES, package: PACKAGES[package] + "-other"})


def test_runtime_torch_must_agree_with_the_source_manifest(source):
    report = experiments.read_json(source / "report.json")
    report["runtime"]["torch"] = "same-version-but-cpu"
    experiments.save_json(source / "report.json", report)
    with pytest.raises(ValueError, match="package versions"):
        validate(source)


@pytest.mark.parametrize("name", ["report.json", "manifest.json"])
def test_incomplete_run_is_not_a_reuse_source(source, name):
    (source / name).unlink()
    with pytest.raises(ValueError, match="completed report"):
        validate(source)


def test_standalone_and_embedded_manifest_must_agree(source):
    manifest = experiments.read_json(source / "manifest.json")
    manifest["batch_size"] = 99
    experiments.save_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="report/manifest"):
        validate(source)


@pytest.mark.parametrize("change", [{"status": "running"}, {"error": "interrupted"}, {"rows": None}])
def test_report_with_incomplete_or_failed_results_is_rejected(source, change):
    report = experiments.read_json(source / "report.json")
    report.update(change)
    experiments.save_json(source / "report.json", report)
    with pytest.raises(ValueError, match="completed experiment results"):
        validate(source)


def test_manifest_itself_must_match_its_recorded_checksum(source):
    with (source / "manifest.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")  # Still valid JSON with identical metadata.
    with pytest.raises(ValueError, match="checksum"):
        validate(source)


def test_new_manifest_model_identity_must_also_agree_with_runtime(source):
    report = experiments.read_json(source / "report.json")
    report["manifest"]["model_revisions"] = {**REVISIONS, "reranker_revision": "foreign"}
    experiments.save_json(source / "manifest.json", report["manifest"])
    report["artifact_hashes"]["manifest.json"] = experiments.sha256(source / "manifest.json")
    experiments.save_json(source / "report.json", report)
    with pytest.raises(ValueError, match="manifest and runtime"):
        validate(source)


@pytest.mark.parametrize("name", ARTIFACTS)
def test_every_copied_artifact_is_rechecked_after_initial_validation(source, tmp_path, name):
    reuse = validate(source)
    with (source / name).open("ab") as stream:
        stream.write(b"\n")
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="checksum"):
        reuse.copy_files([name], target)
    assert list(target.iterdir()) == []
    assert reuse.copied == {}


@pytest.mark.parametrize("name", ["queries.npy", "queries.texts.json", "new-scores.jsonl"])
def test_missing_artifact_hash_is_never_treated_as_a_cache_hit(source, tmp_path, name):
    report = experiments.read_json(source / "report.json")
    del report["artifact_hashes"][name]
    experiments.save_json(source / "report.json", report)
    reuse = validate(source)
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="recorded hash"):
        reuse.copy_files([name], target)
    assert list(target.iterdir()) == []


def test_changed_copy_cannot_be_loaded_by_a_later_resume(source, tmp_path, monkeypatch):
    reuse = validate(source)
    target = tmp_path / "target"
    target.mkdir()
    original_copy = experiments.shutil.copy2

    def changed_copy(src, dst):
        original_copy(src, dst)
        with dst.open("ab") as stream:
            stream.write(b"changed during copy")

    monkeypatch.setattr(experiments.shutil, "copy2", changed_copy)
    with pytest.raises(ValueError, match="while copying"):
        experiments.encode_cached(["first exact input", "second exact input"],
                                  target / "queries.npy", forbidden_encoder(), reuse_from=reuse)
    assert list(target.iterdir()) == []


def test_different_ordered_text_inputs_are_recomputed(source, tmp_path):
    calls = []
    reuse = validate(source)
    target = tmp_path / "target"
    target.mkdir()

    def encode(texts):
        calls.append(texts)
        return np.full((len(texts), 2), 3, dtype=np.float32)

    texts = ["second exact input", "first exact input"]
    result = experiments.encode_cached(texts, target / "queries.npy",
                                       SimpleNamespace(dim=2, encode=encode), reuse_from=reuse)
    assert calls == [texts] and np.array_equal(result, np.full((2, 2), 3, dtype=np.float32))
    assert reuse.copied == {}


def test_unvalidated_reuse_path_cannot_copy_even_matching_inputs(source, tmp_path):
    with pytest.raises(ValueError, match="validiere_reuse"):
        experiments.encode_cached(["first exact input", "second exact input"],
                                  tmp_path / "queries.npy", forbidden_encoder(), reuse_from=source)
