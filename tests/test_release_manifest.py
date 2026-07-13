from __future__ import annotations

import hashlib
import json

import pytest

from risk_layer.release_manifest import (
    MANIFEST_FILENAME,
    MAX_ARTIFACT_BYTES,
    ArtifactDigest,
    ManifestError,
    ReleaseManifest,
    validate_artifact_path,
    validate_release_id,
)


SHA = "1" * 40
DIGEST = hashlib.sha256(b"payload").hexdigest()


def _data(**overrides):
    data = {
        "format_version": 1,
        "release_id": "2026.07.13-abc1234",
        "git_sha": SHA,
        "schema_version": "1",
        "api_version": "v1",
        "features": ["festwissen", "embedding"],
        "artifacts": {
            "app/package.whl": {
                "sha256": DIGEST,
                "size": len(b"payload"),
                "executable": False,
            }
        },
    }
    data.update(overrides)
    return data


def test_manifest_roundtrip_ist_kanonisch_und_fingerprint_stabil(tmp_path):
    manifest = ReleaseManifest.from_dict(_data())
    assert manifest.release_id == "2026.07.13-abc1234"
    assert manifest.features == ("embedding", "festwissen")
    assert manifest.artifacts[0] == ArtifactDigest(
        "app/package.whl", DIGEST, 7, False
    )

    path = manifest.write(tmp_path / MANIFEST_FILENAME)
    loaded = ReleaseManifest.load(path)
    assert loaded == manifest
    assert loaded.fingerprint() == manifest.fingerprint()
    assert path.read_bytes() == manifest.to_json_bytes()


@pytest.mark.parametrize("release_id", [
    "../escape", ".", "-leading", "trailing-", "has space", "CON", "x" * 129,
])
def test_release_id_blockt_traversal_und_unportable_namen(release_id):
    with pytest.raises(ManifestError):
        validate_release_id(release_id)


@pytest.mark.parametrize("path", [
    "../escape", "a/../escape", "/absolute", "C:/windows", "a\\windows",
    "a//nicht-kanonisch", "a/CON.txt", "a/trailing.", "release.json", "",
    "/".join(["a"] * 65),
])
def test_artefaktpfad_ist_relativ_kanonisch_und_portabel(path):
    with pytest.raises(ManifestError):
        validate_artifact_path(path)


def test_manifest_verlangt_exakte_top_level_felder():
    missing = _data()
    del missing["api_version"]
    with pytest.raises(ManifestError, match="fehlend"):
        ReleaseManifest.from_dict(missing)

    with pytest.raises(ManifestError, match="unbekannt"):
        ReleaseManifest.from_dict(_data(transport_url="https://example.invalid"))


@pytest.mark.parametrize(("field", "value"), [
    ("git_sha", "abc123"),
    ("git_sha", "A" * 40),
    ("format_version", True),
    ("format_version", 2),
    ("schema_version", "eins zwei"),
    ("api_version", ""),
])
def test_manifest_blockt_ungueltige_kernfelder(field, value):
    with pytest.raises(ManifestError):
        ReleaseManifest.from_dict(_data(**{field: value}))


@pytest.mark.parametrize("entry", [
    {"sha256": "0" * 63, "size": 7},
    {"sha256": "A" * 64, "size": 7},
    {"sha256": DIGEST, "size": True},
    {"sha256": DIGEST, "size": -1},
    {"sha256": DIGEST, "size": 7, "executable": "yes"},
    {"sha256": DIGEST, "size": 7, "url": "https://example.invalid"},
])
def test_manifest_blockt_ungueltige_artefaktbeschreibung(entry):
    with pytest.raises(ManifestError):
        ReleaseManifest.from_dict(_data(artifacts={"app/package.whl": entry}))


def test_manifest_begrenzt_einzelartefakt_und_release_gesamtgroesse():
    with pytest.raises(ManifestError, match="Artefaktgroesse"):
        ReleaseManifest.from_dict(_data(artifacts={
            "model.gguf": {"sha256": DIGEST, "size": MAX_ARTIFACT_BYTES + 1}
        }))

    artifacts = {
        f"models/model-{index}.gguf": {
            "sha256": DIGEST,
            "size": MAX_ARTIFACT_BYTES,
        }
        for index in range(5)
    }
    with pytest.raises(ManifestError, match="Gesamtgroesse"):
        ReleaseManifest.from_dict(_data(artifacts=artifacts))


def test_manifest_blockt_casefold_und_datei_verzeichnis_kollisionen():
    entry = {"sha256": DIGEST, "size": 7}
    with pytest.raises(ManifestError, match="Grossschreibung"):
        ReleaseManifest.from_dict(_data(artifacts={"A.bin": entry, "a.BIN": entry}))
    with pytest.raises(ManifestError, match="Datei und Verzeichnis"):
        ReleaseManifest.from_dict(_data(artifacts={"app": entry, "app/file": entry}))


def test_manifest_blockt_doppelte_features_ohne_beachtung_der_schreibung():
    with pytest.raises(ManifestError, match="Doppeltes Feature"):
        ReleaseManifest.from_dict(_data(features=["Embedding", "embedding"]))


def test_load_blockt_doppelte_json_keys_und_nichtstandardwerte(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"format_version":1,"format_version":1}', encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="Doppelter"):
        ReleaseManifest.load(duplicate)

    nan = tmp_path / "nan.json"
    nan.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ManifestError, match="Nicht-standardkonstanter"):
        ReleaseManifest.load(nan)


def test_load_blockt_symlink_manifest(tmp_path):
    target = tmp_path / "target.json"
    target.write_text(json.dumps(_data()), encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks stehen auf diesem Testsystem nicht zur Verfuegung")
    with pytest.raises(ManifestError, match="Symlink"):
        ReleaseManifest.load(link)
