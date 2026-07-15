"""Checkout-freier, reproduzierbarer Bootstrap fuer Release-Verifikation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from risk_layer.release_manifest import ArtifactDigest, ReleaseManifest
from tools.build_release_tools_zipapp import build


def _native_archive(tmp_path: Path) -> tuple[Path, ReleaseManifest]:
    bundle = tmp_path / "bundle"
    artifact = bundle / "app" / "demo.whl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"wheel")
    manifest = ReleaseManifest(
        release_id="v1.2.3",
        git_sha="a" * 40,
        schema_version="1",
        api_version="1.3.2",
        artifacts=(
            ArtifactDigest(
                path="app/demo.whl",
                sha256=hashlib.sha256(b"wheel").hexdigest(),
                size=5,
            ),
        ),
    )
    manifest.write(bundle / "release.json")
    archive = tmp_path / "native.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle / "release.json", arcname="release.json")
        tar.add(artifact, arcname="app/demo.whl")
    return archive, manifest


def test_release_tools_zipapp_ist_reproduzierbar_und_checkout_frei(tmp_path):
    first = tmp_path / "first.pyz"
    second = tmp_path / "second.pyz"
    build(first, 1_700_000_000)
    build(second, 1_700_000_000)
    assert first.read_bytes() == second.read_bytes()
    assert Path(f"{first}.sha256").read_text(encoding="ascii").endswith(
        "  first.pyz\n"
    )

    archive, manifest = _native_archive(tmp_path)
    destination = tmp_path / "extracted"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(first),
            "extract-native",
            str(archive),
            str(destination),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert ReleaseManifest.load(destination / "release.json") == manifest
    assert (destination / "app" / "demo.whl").read_bytes() == b"wheel"


def test_release_tools_zipapp_verifier_faellt_ohne_manifest_laut(tmp_path):
    app = tmp_path / "tools.pyz"
    build(app, 1_700_000_000)
    result = subprocess.run(
        [sys.executable, "-I", str(app), "verify-container", str(tmp_path / "fehlt")],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "Deployment-Manifest abgelehnt" in result.stderr


def test_release_tools_zipapp_enthaelt_container_deployer(tmp_path):
    app = tmp_path / "tools.pyz"
    build(app, 1_700_000_000)
    result = subprocess.run(
        [sys.executable, "-I", str(app), "deploy-container", "--help"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "{deploy,rollback,status}" in result.stdout


def test_release_tools_zipapp_enthaelt_checkout_freie_operator_fassade(tmp_path):
    app = tmp_path / "tools.pyz"
    build(app, 1_700_000_000)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", str(app), "ops", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "bootstrap" in result.stdout
    assert "backup-verify" in result.stdout
