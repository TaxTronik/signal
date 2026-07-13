"""Der stabile stdlib-Launcher startet ausschliesslich die current-Runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "risk_layer_release_launcher", ROOT / "deploy" / "release-launcher.py"
)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def _runtime(home: Path) -> tuple[Path, str]:
    release_id = "v1.2.3"
    manifest = "a" * 64
    runtime = home / "runtimes" / f"{release_id}-{manifest[:16]}"
    python_relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    python = runtime / python_relative
    python.parent.mkdir(parents=True)
    shutil.copyfile(sys.executable, python)
    digest = hashlib.sha256(python.read_bytes()).hexdigest()
    (runtime / ".risk-layer-runtime.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "release_id": release_id,
                "manifest_sha256": manifest,
                "platform_tag": LAUNCHER._platform_tag(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "python_path": python_relative.as_posix(),
                "python_sha256": digest,
                "python_size": python.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    home.mkdir(exist_ok=True)
    (home / "state.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "generation": 1,
                "current": release_id,
                "current_manifest_sha256": manifest,
                "previous": None,
                "previous_manifest_sha256": None,
                "updated_at": "2026-07-13T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return python, manifest


def test_launcher_bindet_state_marker_und_interpreter(tmp_path):
    python, _ = _runtime(tmp_path / "home")
    assert LAUNCHER.current_python(tmp_path / "home") == python
    with python.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(LAUNCHER.LauncherError, match="Interpreter"):
        LAUNCHER.current_python(tmp_path / "home")


def test_launcher_execve_nutzt_current_ops_cli_isoliert(tmp_path, monkeypatch):
    python, _ = _runtime(tmp_path / "home")
    call: dict[str, object] = {}

    def fake_execve(path, arguments, environment):
        call.update(path=path, arguments=arguments, environment=environment)
        raise RuntimeError("abgefangen")

    monkeypatch.setattr(LAUNCHER.os, "execve", fake_execve)
    monkeypatch.setenv("PYTHONPATH", "boese")
    with pytest.raises(RuntimeError, match="abgefangen"):
        LAUNCHER.main(
            [
                "--home", str(tmp_path / "home"), "--", "service", "run",
                "--home", str(tmp_path / "home"), "--berater", str(tmp_path / "berater"),
            ]
        )
    assert call["path"] == str(python)
    assert call["arguments"][:4] == [  # type: ignore[index]
        str(python), "-I", "-m", "risk_layer.ops_cli"
    ]
    assert "PYTHONPATH" not in call["environment"]  # type: ignore[operator]
