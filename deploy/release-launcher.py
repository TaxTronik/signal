#!/usr/bin/env python3
"""Minimaler stdlib-Launcher in die manifestgebundene current-Runtime.

Die Datei bleibt stabil ausserhalb der Release-Generationen. Sie waehlt anhand
des atomaren State-v2-Pointers genau die Runtime des aktuellen Manifests,
prueft deren Marker und Interpreterhash und startet erst dann das im Release
installierte ``risk_layer.ops_cli``. Dadurch laeuft der Dienst nicht dauerhaft
mit einer alten Control-Venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import sysconfig
from typing import Any, Sequence


MAX_JSON = 64 * 1024
MAX_PYTHON = 1024 * 1024 * 1024
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RELEASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class LauncherError(RuntimeError):
    pass


def _is_reparse(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(info, "st_file_attributes", 0) & flag)


def _check_chain(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            raise LauncherError(f"Symlink-/Reparse-Pfad abgelehnt: {component}")
        if not stat.S_ISDIR(info.st_mode):
            raise LauncherError(f"Pfadkomponente ist kein Verzeichnis: {component}")


def _read_regular(path: Path, maximum: int, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(path)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LauncherError(f"{description} nicht sicher lesbar: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _is_reparse(named)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
            or opened.st_size > maximum
        ):
            raise LauncherError(f"{description} wurde ausgetauscht/ist unsicher: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum:
                raise LauncherError(f"{description} ueberschreitet das Limit")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LauncherError(f"Doppelter JSON-Schluessel: {key}")
        result[key] = value
    return result


def _json(path: Path, description: str) -> dict[str, Any]:
    raw = _read_regular(path, MAX_JSON, description)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LauncherError(f"Nichtstandard-JSON: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherError(f"{description} ist kein gueltiges UTF-8-JSON") from exc
    if not isinstance(value, dict):
        raise LauncherError(f"{description} muss ein JSON-Objekt sein")
    return value


def _platform_tag() -> str:
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_").lower()
    if not re.fullmatch(r"[a-z0-9_]{1,96}", platform):
        raise LauncherError(f"Ungueltige Plattformkennung: {platform!r}")
    return f"{platform}-cp{sys.version_info.major}{sys.version_info.minor}"


def _sha256_file(path: Path, expected_size: int) -> str:
    if expected_size <= 0 or expected_size > MAX_PYTHON:
        raise LauncherError("Runtime-Marker enthaelt ungueltige Interpretergroesse")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(path)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LauncherError(f"Runtime-Interpreter nicht sicher lesbar: {path}") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _is_reparse(named)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
            or opened.st_size != expected_size
        ):
            raise LauncherError("Runtime-Interpreter wurde ausgetauscht/hat falsche Groesse")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > expected_size:
                raise LauncherError("Runtime-Interpreter wuchs beim Lesen")
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            total != expected_size
            or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise LauncherError("Runtime-Interpreter wurde waehrend der Pruefung veraendert")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def current_python(home_value: str | Path) -> Path:
    home = Path(os.path.abspath(home_value))
    _check_chain(home)
    state = _json(home / "state.json", "Release-State")
    expected_state = {
        "format_version", "generation", "current", "current_manifest_sha256",
        "previous", "previous_manifest_sha256", "updated_at",
    }
    if set(state) != expected_state or state.get("format_version") != 2:
        raise LauncherError("Release-State hat nicht das erwartete v2-Schema")
    release_id = state.get("current")
    manifest_sha = state.get("current_manifest_sha256")
    if not isinstance(release_id, str) or not RELEASE_RE.fullmatch(release_id):
        raise LauncherError("Release-State enthaelt kein gueltiges current")
    if not isinstance(manifest_sha, str) or not SHA_RE.fullmatch(manifest_sha):
        raise LauncherError("Release-State enthaelt keinen Manifest-SHA")

    runtime = home / "runtimes" / f"{release_id}-{manifest_sha[:16]}"
    _check_chain(runtime)
    marker = _json(runtime / ".risk-layer-runtime.json", "Runtime-Marker")
    expected_marker = {
        "format_version", "release_id", "manifest_sha256", "platform_tag",
        "python_version", "python_path", "python_sha256", "python_size",
    }
    python_relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    if set(marker) != expected_marker or marker.get("format_version") != 1:
        raise LauncherError("Runtime-Marker hat nicht das erwartete Schema")
    if (
        marker.get("release_id") != release_id
        or marker.get("manifest_sha256") != manifest_sha
        or marker.get("platform_tag") != _platform_tag()
        or marker.get("python_version")
        != f"{sys.version_info.major}.{sys.version_info.minor}"
        or marker.get("python_path") != python_relative
    ):
        raise LauncherError("Runtime-Marker passt nicht zum current-State/Host")
    python_hash = marker.get("python_sha256")
    python_size = marker.get("python_size")
    if (
        not isinstance(python_hash, str)
        or not SHA_RE.fullmatch(python_hash)
        or isinstance(python_size, bool)
        or not isinstance(python_size, int)
    ):
        raise LauncherError("Runtime-Marker bindet keinen gueltigen Interpreter")
    python = runtime.joinpath(*python_relative.split("/"))
    if _sha256_file(python, python_size) != python_hash:
        raise LauncherError("Runtime-Interpreterhash stimmt nicht")
    return python


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("nach -- fehlt der risk_layer.ops_cli-Befehl")
    try:
        python = current_python(args.home)
    except LauncherError as exc:
        raise SystemExit(f"Release-Launcher abgelehnt: {exc}") from exc
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    arguments = [str(python), "-I", "-m", "risk_layer.ops_cli", *command]
    os.execve(str(python), arguments, environment)
    raise AssertionError("os.execve kehrte unerwartet zurueck")


if __name__ == "__main__":
    raise SystemExit(main())
