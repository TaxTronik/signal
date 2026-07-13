"""Statische Invarianten fuer reproduzierbare Produktions-Beschaffung."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relativ: str) -> str:
    return (ROOT / relativ).read_text(encoding="utf-8")


def test_produktionsinstaller_loesen_keine_ml_ranges_mehr_auf() -> None:
    docker = _text("Dockerfile")
    linux = _text("deploy.sh")
    windows = _text("deploy.ps1")

    assert "requirements-runtime-lock.txt" in docker
    assert "requirements-lock.txt" not in docker
    assert "requirements-embedding-linux-cpu-py312-lock.txt" in docker

    for skript in (linux, windows):
        assert 'pip install "sentence-transformers' not in skript
        assert "hf_transfer" not in skript
        assert "qiskit>=" not in skript
        assert "requirements-runtime-lock.txt" in skript

    assert "requirements-embedding-linux-cpu-py312-lock.txt" in linux
    assert "requirements-quanten-linux-py312-lock.txt" in linux
    assert "requirements-embedding-windows-cpu-py312-lock.txt" in windows
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
        assert "torch==2.11.0+cpu" in lock
        assert "--hash=sha256:" in lock
        assert "nvidia-" not in lock
        assert "cuda-toolkit==" not in lock
        assert "triton==" not in lock

    for name in quanten_locks:
        lock = _text(name).lower()
        assert "qiskit==2.5.0" in lock
        assert "qiskit-aer==0.17.2" in lock
        assert "qiskit-ibm-runtime==0.47.0" in lock
        assert "--hash=sha256:" in lock

    runtime = _text("requirements-runtime-lock.txt").lower()
    assert "pytest==" not in runtime
    assert "numpy==2.2.6" in runtime
    assert "setuptools==81.0.0" in runtime


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


def test_security_workflow_auditiert_alle_produktionslocks() -> None:
    workflow = _text(".forgejo/workflows/security.yml")
    for lock in (
        "requirements-runtime-lock.txt",
        "requirements-ner-lock.txt",
        "requirements-embedding-linux-cpu-py312-lock.txt",
        "requirements-quanten-linux-py312-lock.txt",
    ):
        assert f"pip_audit -r {lock}" in workflow

    for lock in (
        "requirements-embedding-windows-cpu-py312-lock.txt",
        "requirements-quanten-windows-py312-lock.txt",
    ):
        assert f"pip_audit --no-deps --disable-pip -r {lock}" in workflow
