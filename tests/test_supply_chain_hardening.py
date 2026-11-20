"""Statische Invarianten fuer reproduzierbare Produktions-Beschaffung."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _text(relativ: str) -> str:
    return (ROOT / relativ).read_text(encoding="utf-8")


def test_produktionsinstaller_loesen_keine_ml_ranges_mehr_auf() -> None:
    docker = _text("Dockerfile")
    linux = _text("scripts/setup.sh")
    windows = _text("scripts/setup.ps1")

    assert "requirements-runtime-lock.txt" in docker
    assert "requirements-lock.txt" not in docker
    assert "requirements-embedding-linux-cpu-py312-lock.txt" in docker
    assert "requirements-quanten-linux-py312-lock.txt" in docker
    assert "ARG WITH_QUANTEN=0" in docker

    for skript in (linux, windows):
        assert 'pip install "sentence-transformers' not in skript
        assert "hf_transfer" not in skript
        assert "qiskit>=" not in skript
        assert "requirements-runtime-lock.txt" in skript
        assert "--no-build-isolation --no-deps -e ." not in skript
        assert "pip wheel --no-build-isolation --no-deps" in skript
        assert "--no-index --no-deps --force-reinstall" in skript

    assert "requirements-embedding-linux-cpu-py312-lock.txt" in linux
    assert "requirements-quanten-linux-py312-lock.txt" in linux
    assert "requirements-embedding-windows-cpu-py312-lock.txt" in windows
    assert "requirements-embedding-windows-amd-py312-lock.txt" in windows
    assert "requirements-quanten-windows-py312-lock.txt" in windows


def test_optionale_locks_sind_zielsystemspezifisch_und_hashgepinnt() -> None:
    embedding_locks = (
        "requirements-embedding-linux-cpu-py312-lock.txt",
        "requirements-embedding-windows-cpu-py312-lock.txt",
    )
    quanten_locks = (
        "requirements-quanten-linux-py312-lock.txt",
        "requirements-quanten-windows-py312-lock.txt",
    )

    for name in embedding_locks:
        lock = _text(name).lower()
        assert "sentence-transformers==5.6.0" in lock
        assert "torch==2.13.0+cpu" in lock
        assert "--hash=sha256:" in lock
        assert "nvidia-" not in lock
        assert "cuda-toolkit==" not in lock
        assert "triton==" not in lock

    amd_lock = _text("requirements-embedding-windows-amd-py312-lock.txt").lower()
    assert "sentence-transformers==5.6.0" in amd_lock
    assert "https://stable.repo.amd.com/rocm/whl-next/torch/torch-2.15.0%2brocm10.2.0-cp312-cp312-win_amd64.whl" in amd_lock
    for arch in ("gfx1100", "gfx1101", "gfx1200", "gfx1201"):
        assert f"amd-torch-device-{arch} @" in amd_lock
        assert f"rocm-sdk-device-{arch} @" in amd_lock
    assert "--hash=sha256:" in amd_lock
    assert "nvidia-" not in amd_lock

    for name in quanten_locks:
        lock = _text(name).lower()
        assert "qiskit==2.5.0" in lock
        assert "qiskit-aer==0.17.2" in lock
        assert "qiskit-ibm-runtime==0.47.0" in lock
        assert "--hash=sha256:" in lock

    runtime = _text("requirements-runtime-lock.txt").lower()
    assert "pytest==" not in runtime
    assert "numpy==2.2.6" in runtime
    assert "setuptools==84.1.0" in runtime


def test_setuptools_vertrag_ist_in_quellen_und_locks_identisch() -> None:
    erwartet = "setuptools==84.1.0"
    projekt = tomllib.loads(_text("pyproject.toml"))
    build_anforderungen = projekt["build-system"]["requires"]
    assert [
        eintrag
        for eintrag in build_anforderungen
        if eintrag.lower().startswith("setuptools")
    ] == [erwartet]

    for name in (
        "requirements.in",
        "requirements-runtime.in",
        "requirements-lock.txt",
        "requirements-runtime-lock.txt",
        "requirements-ner-lock.txt",
        "requirements-embedding-linux-cpu-py312-lock.txt",
        "requirements-embedding-windows-cpu-py312-lock.txt",
        "requirements-embedding-windows-amd-py312-lock.txt",
    ):
        setuptools_anforderungen = [
            zeile.split()[0]
            for zeile in _text(name).splitlines()
            if zeile.strip().lower().startswith("setuptools")
        ]
        assert setuptools_anforderungen == [erwartet], name


def test_ci_actions_und_containerbasis_sind_digestgepinnt() -> None:
    workflows = "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted((ROOT / ".forgejo" / "workflows").glob("*.yml"))
    )
    refs = re.findall(r"(?m)^\s*- uses:\s*[^\s@]+@([^\s#]+)", workflows)
    assert refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs)

    docker = _text("Dockerfile")
    assert re.search(r"(?m)^FROM\s+\S+@sha256:[0-9a-f]{64}(?:\s|$)", docker)


@pytest.mark.parametrize("filename", ["security.yml", "build.yml"])
def test_security_workflow_auditiert_alle_produktionslocks(filename: str) -> None:
    workflow = _text(f".forgejo/workflows/{filename}")
    assert "python tools/audit_locks.py" in workflow


def test_audit_prueft_upstreamversionen_von_vendor_wheels(tmp_path):
    from tools.audit_locks import audit_pins

    lock = tmp_path / "requirements-test-lock.txt"
    pin = "--hash=sha256:" + "a" * 64
    lock.write_text(
        f"torch @ https://repo.radeon.com/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl {pin}\n"
        f"numpy==2.2.6 {pin}\n", encoding="utf-8")
    assert audit_pins(lock) == {"torch==2.9.1", "numpy==2.2.6"}
    lock.write_text(f"torch==2.13.0+cpu {pin}\n", encoding="utf-8")
    assert audit_pins(lock) == {"torch==2.13.0"}


def test_herstellerartefakt_benoetigt_hash_und_passenden_wheelnamen(tmp_path):
    from tools.audit_locks import audit_pins

    lock = tmp_path / "requirements-test-lock.txt"
    pin = "--hash=sha256:" + "a" * 64
    lock.write_text(
        "rocm @ https://repo.radeon.com/rocm-1.0-py3-none-any.whl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        audit_pins(lock)
    lock.write_text(
        f"rocm @ https://repo.radeon.com/other-1.0-py3-none-any.whl {pin}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact name"):
        audit_pins(lock)
    lock.write_text(
        f"rocm @ https://repo.radeon.com/rocm-10.2.0.tar.gz {pin}\n"
        f"numpy==2.2.6 {pin}\n", encoding="utf-8")
    assert audit_pins(lock) == {"numpy==2.2.6"}
