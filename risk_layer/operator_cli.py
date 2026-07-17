#!/usr/bin/env python3
"""Einheitliche Operator-Fassade fuer Checkout und Release-Instanzen.

Ein Checkout besitzt einen lokalen Zero-Config-Pfad. Explizit profilierte
Release-Instanzen behalten die manifestgebundenen, gehaerteten Transaktionen.
Alle Ausfuehrungen delegieren als argv ohne Shell an bestehende Werkzeuge.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit


PROFILE_SCHEMA = "taxtronik.risk-layer-ops/v1"
PROFILE_KEYS = frozenset(
    {
        "schema",
        "mode",
        "python",
        "bootstrap_python",
        "docker_bin",
        "release_home",
        "state_dir",
        "berater_dir",
        "backup_dir",
        "backup_target_id",
        "launcher",
        "release_tools",
        "compose_file",
        "secrets_env",
        "token_file",
        "health_url",
        "status_url",
        "activation_command",
        "required_layers",
        "retention",
    }
)
RETENTION_KEYS = frozenset({"keep_last", "keep_days"})
COMMON_PATH_FIELDS = (
    "python",
    "release_home",
    "berater_dir",
    "backup_dir",
    "launcher",
    "token_file",
)
CONTAINER_PATH_FIELDS = (
    "state_dir",
    "release_tools",
    "compose_file",
    "secrets_env",
)
MANAGED_CONTAINER_ENV = frozenset(
    {
        "RISK_LAYER_ENGINE_REPOSITORY",
        "RISK_LAYER_ENGINE_DIGEST",
        "RISK_LAYER_RELEASE_ID",
        "RISK_LAYER_RELEASE_GIT_SHA",
        "RISK_LAYER_RELEASE_MANIFEST_SHA256",
        "RISK_LAYER_FESTWISSEN_DIR",
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
RELEASE_ID_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?\Z"
)
LAYER_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})?\Z")
BACKUP_TARGET_ID_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?\Z"
)
ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
WINDOWS_FORBIDDEN = frozenset('<>:"\\|?*')
WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
MAX_PROFILE_BYTES = 128 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_SECRET_BYTES = 1024 * 1024
MAX_BACKUP_TARGET_BYTES = 16 * 1024
BACKUP_TARGET_SCHEMA = "taxtronik.risk-layer-backup-target/v1"
BACKUP_TARGET_FILENAME = ".risk-layer-backup-target.json"
DEFAULT_PROFILE = (
    "C:/ProgramData/TaxTronik/RiskLayer/config/ops.json"
    if os.name == "nt"
    else "/etc/risk-layer/ops.json"
)
RELEASE_INBOX_NAME = "incoming"
CONTAINER_MANIFEST_PATTERN = "taxtronik-container-*.json"
NATIVE_MANIFEST_NAME = "release.json"
MAX_CHECKSUM_BYTES = 512
CHECKOUT_ROOT_ENV = "RISK_LAYER_CHECKOUT_ROOT"
CHECKOUT_STATE_DIR = ".signal"
CHECKOUT_BACKUP_TARGET_PREFIX = "checkout-"
CHECKOUT_SYSTEMD_UNIT = "risk-layer.service"
CHECKOUT_SYSTEMD_SYSTEM_DIR = Path("/etc/systemd/system")
CHECKOUT_SYSTEMD_USER = "risk-layer"
CHECKOUT_WINDOWS_TASK = "RiskLayer"
CHECKOUT_SERVICE_SCHEMA = "risk-layer-checkout-service/v1"
CHECKOUT_SERVICE_STATE_NAME = "service.json"
MINIMUM_PYTHON = (3, 11, 4)
INTERPRETER_CHECK_TIMEOUT = 10.0
WINDOWS_POWERSHELL_TIMEOUT = 15.0
MAX_OPERATION_TIMEOUT_SECONDS = 3600.0
WINDOWS_SYSTEM_SID = "S-1-5-18"
WINDOWS_ADMINISTRATORS_SID = "S-1-5-32-544"
WINDOWS_OWNER_RIGHTS_SID = "S-1-3-4"
WINDOWS_TRUSTED_INSTALLER_SID = (
    "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
)
WINDOWS_FULL_CONTROL = 0x1F01FF
WINDOWS_MUTATING_RIGHTS = 0x0D0156
WINDOWS_MODIFY = 0x0301BF

_WINDOWS_ACL_QUERY = r"""
$ErrorActionPreference = 'Stop'
$path = [System.IO.Path]::GetFullPath($env:RISK_LAYER_ACL_PATH)
$attributes = [System.IO.File]::GetAttributes($path)
$isDirectory = ($attributes -band [System.IO.FileAttributes]::Directory) -ne 0
if ($isDirectory) {
    $acl = [System.IO.Directory]::GetAccessControl($path)
} else {
    $acl = [System.IO.File]::GetAccessControl($path)
}
$current = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
if ($null -eq $current) { throw 'Aktuelle Benutzer-SID fehlt.' }
$rules = $acl.GetAccessRules(
    $true,
    $true,
    [System.Security.Principal.SecurityIdentifier]
)
$parts = [System.Collections.Generic.List[string]]::new()
foreach ($rule in $rules) {
    $inherited = $rule.IsInherited.ToString().ToLowerInvariant()
    $entry = '{"sid":"' + $rule.IdentityReference.Value + `
        '","rights":' + ([int64]$rule.FileSystemRights).ToString() + `
        ',"type":"' + $rule.AccessControlType.ToString() + `
        '","inherited":' + $inherited + '}'
    $parts.Add($entry)
}
$owner = $acl.GetOwner(
    [System.Security.Principal.SecurityIdentifier]
).Value
$protected = $acl.AreAccessRulesProtected.ToString().ToLowerInvariant()
$json = '{"current":"' + $current.Value + '","owner":"' + $owner + `
    '","protected":' + $protected + ',"rules":[' + `
    [string]::Join(',', $parts) + ']}'
[Console]::Out.WriteLine($json)
"""

_WINDOWS_ACL_HARDEN = r"""
$ErrorActionPreference = 'Stop'
$path = [System.IO.Path]::GetFullPath($env:RISK_LAYER_ACL_PATH)
$current = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
if ($null -eq $current) { throw 'Aktuelle Benutzer-SID fehlt.' }
$acl = [System.IO.File]::GetAccessControl($path)
[void]$acl.SetAccessRuleProtection($true, $false)
foreach ($existing in @($acl.Access)) {
    [void]$acl.RemoveAccessRuleAll($existing)
}
$allow = [System.Security.AccessControl.AccessControlType]::Allow
$entries = @(
    [pscustomobject]@{
        Sid = $current
        Rights = [System.Security.AccessControl.FileSystemRights]::FullControl
    },
    [pscustomobject]@{
        Sid = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-18')
        Rights = [System.Security.AccessControl.FileSystemRights]::Read
    },
    [pscustomobject]@{
        Sid = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
        Rights = [System.Security.AccessControl.FileSystemRights]::FullControl
    }
)
foreach ($entry in $entries) {
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $entry.Sid,
        $entry.Rights,
        $allow
    )
    [void]$acl.AddAccessRule($rule)
}
[System.IO.File]::SetAccessControl($path, $acl)
"""

_WINDOWS_DIRECTORY_ACL_HARDEN = r"""
$ErrorActionPreference = 'Stop'
$path = [System.IO.Path]::GetFullPath($env:RISK_LAYER_ACL_PATH)
$current = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
if ($null -eq $current) { throw 'Aktuelle Benutzer-SID fehlt.' }
$acl = [System.IO.Directory]::GetAccessControl($path)
[void]$acl.SetAccessRuleProtection($true, $false)
foreach ($existing in @($acl.Access)) {
    [void]$acl.RemoveAccessRuleAll($existing)
}
$allow = [System.Security.AccessControl.AccessControlType]::Allow
$inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit `
    -bor [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
$propagation = [System.Security.AccessControl.PropagationFlags]::None
$entries = @(
    [pscustomobject]@{
        Sid = $current
        Rights = [System.Security.AccessControl.FileSystemRights]::FullControl
    },
    [pscustomobject]@{
        Sid = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-18')
        Rights = [System.Security.AccessControl.FileSystemRights]::Read
    },
    [pscustomobject]@{
        Sid = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
        Rights = [System.Security.AccessControl.FileSystemRights]::FullControl
    }
)
foreach ($entry in $entries) {
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $entry.Sid,
        $entry.Rights,
        $inheritance,
        $propagation,
        $allow
    )
    [void]$acl.AddAccessRule($rule)
}
[System.IO.Directory]::SetAccessControl($path, $acl)
"""


class OperatorError(RuntimeError):
    """Sicher behandelbarer Profil-, Bedienungs- oder Delegate-Fehler."""


def _positive_timeout(value: Any, name: str) -> float:
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        normalized = math.nan
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(normalized)
        or normalized <= 0
        or normalized > MAX_OPERATION_TIMEOUT_SECONDS
    ):
        raise OperatorError(
            f"{name} muss endlich und > 0 bis {MAX_OPERATION_TIMEOUT_SECONDS:g}s sein"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class Profile:
    source: Path
    mode: str
    python: Path
    bootstrap_python: Path | None
    docker_bin: Path | None
    release_home: Path
    state_dir: Path | None
    berater_dir: Path
    backup_dir: Path
    backup_target_id: str
    launcher: Path
    release_tools: Path | None
    compose_file: Path | None
    secrets_env: Path | None
    token_file: Path
    health_url: str
    status_url: str
    activation_command: tuple[str, ...] | None
    required_layers: tuple[str, ...]
    keep_last: int
    keep_days: int


def _is_reparse(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(info, "st_file_attributes", 0) & flag)


def _is_link_like(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise OperatorError(f"Pfad kann nicht sicher geprueft werden: {path}") from exc
    return stat.S_ISLNK(info.st_mode) or _is_reparse(info)


def _check_directory_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for component in reversed((absolute, *absolute.parents)):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise OperatorError(
                f"Pfadkomponente kann nicht sicher geprueft werden: {component}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            raise OperatorError(f"Symlink-/Reparse-Komponente abgelehnt: {component}")
        if not stat.S_ISDIR(info.st_mode):
            raise OperatorError(f"Pfadkomponente ist kein Verzeichnis: {component}")


def _read_regular(path: Path, maximum: int, description: str) -> bytes:
    absolute = Path(os.path.abspath(path))
    _check_directory_chain(absolute.parent)
    if _is_link_like(absolute):
        raise OperatorError(f"{description} darf kein Link/Reparse-Point sein: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise OperatorError(f"{description} fehlt oder ist nicht lesbar: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        named = os.lstat(absolute)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _is_reparse(named)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
            or before.st_size > maximum
        ):
            raise OperatorError(f"{description} ist keine sichere regulaere Datei: {absolute}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum:
                raise OperatorError(f"{description} ueberschreitet {maximum} Bytes")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise OperatorError(f"{description} wurde beim Lesen veraendert: {absolute}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _path_identity(path: Path) -> tuple[int, int, int, int]:
    info = os.lstat(path)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OperatorError(f"Doppelter JSON-Schluessel: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise OperatorError(f"Nichtstandard-JSON ist unzulaessig: {token}")


def _decode_object(payload: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except OperatorError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OperatorError(f"{description} ist kein gueltiges UTF-8-JSON") from exc
    if not isinstance(value, dict):
        raise OperatorError(f"{description} muss ein JSON-Objekt sein")
    return value


def _clean_string(value: Any, field: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise OperatorError(f"Profilfeld {field} muss ein sauberer nichtleerer String sein")
    return value


def _absolute_profile_path(value: Any, field: str) -> Path:
    raw = _clean_string(value, field)
    path = Path(raw)
    if not path.is_absolute():
        raise OperatorError(f"Profilfeld {field} muss ein absoluter Pfad sein: {raw}")
    return Path(os.path.abspath(path))


def _canonical_path_key(path: Path) -> str:
    value = os.path.normpath(os.path.abspath(path))
    return os.path.normcase(value) if os.name == "nt" else value


def _paths_overlap(first: Path, second: Path) -> bool:
    first_key = _canonical_path_key(first)
    second_key = _canonical_path_key(second)
    try:
        common = os.path.commonpath((first_key, second_key))
    except ValueError:
        return False
    return common in {first_key, second_key}


def _validate_profile_path_roles(profile: Profile) -> None:
    directories = {
        "release_home": profile.release_home,
        "berater_dir": profile.berater_dir,
        "backup_dir": profile.backup_dir,
    }
    if profile.mode == "container":
        directories["state_dir"] = _required_container_path(
            profile.state_dir, "state_dir"
        )
    directory_items = list(directories.items())
    for index, (first_name, first) in enumerate(directory_items):
        for second_name, second in directory_items[index + 1 :]:
            if _paths_overlap(first, second):
                raise OperatorError(
                    f"Profilpfade {first_name} und {second_name} duerfen sich nicht ueberlappen"
                )

    files = {
        "profile": profile.source,
        "launcher": profile.launcher,
        "token_file": profile.token_file,
    }
    private_parents = {"token_file": profile.token_file.parent}
    if profile.mode == "container":
        files.update(
            {
                "release_tools": _required_container_path(
                    profile.release_tools, "release_tools"
                ),
                "compose_file": _required_container_path(
                    profile.compose_file, "compose_file"
                ),
                "secrets_env": _required_container_path(
                    profile.secrets_env, "secrets_env"
                ),
            }
        )
        private_parents["secrets_env"] = files["secrets_env"].parent
    file_items = list(files.items())
    for index, (first_name, first) in enumerate(file_items):
        for second_name, second in file_items[index + 1 :]:
            if _paths_overlap(first, second):
                raise OperatorError(
                    f"Dateirollen {first_name} und {second_name} kollidieren"
                )
        for directory_name, directory in directory_items:
            if _canonical_path_key(first) == _canonical_path_key(directory):
                raise OperatorError(
                    f"Dateirolle {first_name} kollidiert mit {directory_name}"
                )
    for private_name, parent in private_parents.items():
        for directory_name, directory in directory_items:
            if _paths_overlap(parent, directory):
                raise OperatorError(
                    f"Privater Pfad {private_name} kollidiert mit {directory_name}"
                )


def _validate_url(value: Any, field: str, *, status: bool) -> str:
    raw = _clean_string(value, field)
    parsed = urlsplit(raw)
    expected_path = "/v1/status" if status else ""
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != expected_path
    ):
        suffix = "/v1/status" if status else "ohne Pfad"
        raise OperatorError(f"Profilfeld {field} muss eine saubere HTTP(S)-URL {suffix} sein")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.casefold() == "localhost"
        if not loopback:
            raise OperatorError(f"Profilfeld {field}: HTTP ist nur auf Loopback erlaubt")
    return raw.rstrip("/") if not status else raw


def _require_profile_file(path: Path) -> None:
    _require_control_file(path, "Ops-Profil")


def _require_control_directory(path: Path, description: str) -> None:
    _require_directory(path, f"{description}-Elternverzeichnis")
    info = os.lstat(path)
    if os.name == "nt":
        _require_windows_acl(path, f"{description}-Elternverzeichnis", private=False)
        return
    geteuid = getattr(os, "geteuid", None)
    current_uid = geteuid() if callable(geteuid) else info.st_uid
    if info.st_uid not in {0, current_uid}:
        raise OperatorError(
            f"{description}-Elternverzeichnis gehoert weder root noch dem aktuellen Benutzer"
        )


def _require_private_directory(path: Path, description: str) -> None:
    _require_directory(path, f"{description}-Elternverzeichnis")
    info = os.lstat(path)
    if os.name == "nt":
        _require_windows_acl(path, f"{description}-Elternverzeichnis", private=True)
        return
    geteuid = getattr(os, "geteuid", None)
    current_uid = geteuid() if callable(geteuid) else info.st_uid
    if info.st_uid != current_uid:
        raise OperatorError(
            f"{description}-Elternverzeichnis gehoert einem anderen Benutzer"
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise OperatorError(
            f"{description}-Elternverzeichnis muss Modus 0700 besitzen"
        )


def _require_managed_directory(path: Path, description: str) -> None:
    _require_directory(path, description)
    info = os.lstat(path)
    if os.name == "nt":
        _require_windows_acl(
            path,
            description,
            private=False,
            require_current_write=True,
        )
        return
    geteuid = getattr(os, "geteuid", None)
    current_uid = geteuid() if callable(geteuid) else info.st_uid
    if info.st_uid not in {0, current_uid}:
        raise OperatorError(f"{description} gehoert weder root noch dem aktuellen Benutzer")
    if not os.access(path, os.W_OK | os.X_OK):
        raise OperatorError(f"{description} ist fuer den aktuellen Benutzer nicht schreibbar")


def _require_control_file(
    path: Path, description: str, *, strict_windows: bool = False
) -> None:
    _require_control_directory(Path(os.path.abspath(path)).parent, description)
    _require_file(path, description)
    info = os.lstat(path)
    if info.st_nlink != 1:
        raise OperatorError(f"{description} darf keine Hardlinks besitzen")
    if os.name == "nt":
        _require_windows_acl(
            path,
            description,
            private=False,
            strict_control=strict_windows,
        )
        return
    geteuid = getattr(os, "geteuid", None)
    current_uid = geteuid() if callable(geteuid) else info.st_uid
    if info.st_uid not in {0, current_uid}:
        raise OperatorError(
            f"{description} muss dem aktuellen Benutzer oder root gehoeren"
        )


def _require_executable_control(path: Path, description: str) -> None:
    _require_control_file(path, description)
    if not os.access(path, os.X_OK):
        raise OperatorError(f"{description} ist nicht ausfuehrbar: {path}")


def _required_container_path(value: Path | None, field: str) -> Path:
    if value is None:
        raise OperatorError(
            f"Interner Profilvertrag verletzt: Container-Pfad {field} fehlt"
        )
    return value


def load_profile(path: str | Path) -> Profile:
    source = Path(os.path.abspath(path))
    _require_profile_file(source)
    before = _path_identity(source)
    payload = _read_regular(source, MAX_PROFILE_BYTES, "Ops-Profil")
    _require_profile_file(source)
    if _path_identity(source) != before:
        raise OperatorError("Ops-Profil wurde waehrend der Sicherheitspruefung veraendert")
    document = _decode_object(payload, "Ops-Profil")
    if set(document) != PROFILE_KEYS:
        missing = sorted(PROFILE_KEYS - set(document))
        unknown = sorted(set(document) - PROFILE_KEYS)
        raise OperatorError(
            f"Ops-Profil hat fehlende/unbekannte Felder: missing={missing}, unknown={unknown}"
        )
    if document["schema"] != PROFILE_SCHEMA:
        raise OperatorError(f"Ops-Profil hat nicht Schema {PROFILE_SCHEMA}")
    mode = document["mode"]
    if mode not in {"native", "container"}:
        raise OperatorError("Profilfeld mode muss native oder container sein")
    paths = {
        field: _absolute_profile_path(document[field], field)
        for field in COMMON_PATH_FIELDS
    }
    container_paths: dict[str, Path | None] = {}
    for field in CONTAINER_PATH_FIELDS:
        value = document[field]
        if mode == "container":
            container_paths[field] = _absolute_profile_path(value, field)
        elif value is not None:
            raise OperatorError(f"Native Profile muessen {field} auf null setzen")
        else:
            container_paths[field] = None
    bootstrap_value = document["bootstrap_python"]
    bootstrap_python = (
        None
        if bootstrap_value is None
        else _absolute_profile_path(bootstrap_value, "bootstrap_python")
    )
    if mode == "container" and bootstrap_python is not None:
        raise OperatorError("Container-Profile muessen bootstrap_python auf null setzen")
    docker_value = document["docker_bin"]
    if mode == "container":
        docker_bin = _absolute_profile_path(docker_value, "docker_bin")
    elif docker_value is not None:
        raise OperatorError("Native Profile muessen docker_bin auf null setzen")
    else:
        docker_bin = None

    activation_value = document["activation_command"]
    activation: tuple[str, ...] | None
    if activation_value is None:
        activation = None
    elif (
        isinstance(activation_value, list)
        and 1 <= len(activation_value) <= 64
        and all(
            isinstance(item, str)
            and item
            and len(item) <= 4096
            and not any(ord(character) < 32 for character in item)
            for item in activation_value
        )
    ):
        activation = tuple(activation_value)
        if not Path(activation[0]).is_absolute():
            raise OperatorError(
                "activation_command[0] muss ein absoluter Executable-Pfad sein"
            )
    else:
        raise OperatorError("activation_command muss null oder ein JSON-String-Array sein")
    if mode == "container" and activation is not None:
        raise OperatorError("Container-Profile muessen activation_command auf null setzen")

    layers_value = document["required_layers"]
    if (
        not isinstance(layers_value, list)
        or len(layers_value) > 64
        or any(not isinstance(item, str) or not LAYER_RE.fullmatch(item) for item in layers_value)
        or len({item.casefold() for item in layers_value}) != len(layers_value)
    ):
        raise OperatorError("required_layers muss eine eindeutige Liste sicherer Layer-Namen sein")

    retention = document["retention"]
    if not isinstance(retention, dict) or set(retention) != RETENTION_KEYS:
        raise OperatorError("retention muss exakt keep_last und keep_days enthalten")
    keep_last = retention["keep_last"]
    keep_days = retention["keep_days"]
    if (
        isinstance(keep_last, bool)
        or not isinstance(keep_last, int)
        or not 1 <= keep_last <= 100_000
        or isinstance(keep_days, bool)
        or not isinstance(keep_days, int)
        or not 1 <= keep_days <= 100_000
    ):
        raise OperatorError("Retention-Werte muessen Ganzzahlen zwischen 1 und 100000 sein")

    health_url = _validate_url(document["health_url"], "health_url", status=False)
    status_url = _validate_url(document["status_url"], "status_url", status=True)
    if status_url != f"{health_url}/v1/status":
        raise OperatorError("status_url muss exakt health_url + /v1/status sein")
    backup_target_id = document["backup_target_id"]
    if (
        not isinstance(backup_target_id, str)
        or not BACKUP_TARGET_ID_RE.fullmatch(backup_target_id)
    ):
        raise OperatorError(
            "backup_target_id muss ein sicherer Identifier mit 1-64 Zeichen sein"
        )

    profile = Profile(
        source=source,
        mode=mode,
        python=paths["python"],
        bootstrap_python=bootstrap_python,
        docker_bin=docker_bin,
        release_home=paths["release_home"],
        state_dir=container_paths["state_dir"],
        berater_dir=paths["berater_dir"],
        backup_dir=paths["backup_dir"],
        backup_target_id=backup_target_id,
        launcher=paths["launcher"],
        release_tools=container_paths["release_tools"],
        compose_file=container_paths["compose_file"],
        secrets_env=container_paths["secrets_env"],
        token_file=paths["token_file"],
        health_url=health_url,
        status_url=status_url,
        activation_command=activation,
        required_layers=tuple(layers_value),
        keep_last=keep_last,
        keep_days=keep_days,
    )
    _validate_profile_path_roles(profile)
    return profile


def _state_path(profile: Profile) -> Path:
    if profile.mode == "native":
        return profile.release_home / "state.json"
    state_dir = _required_container_path(profile.state_dir, "state_dir")
    return state_dir / "state.json"


def _state_file_exists(path: Path) -> bool:
    _check_directory_chain(path.parent)
    if _is_link_like(path):
        raise OperatorError(f"Deployment-State darf kein Link/Reparse-Point sein: {path}")
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise OperatorError(f"Deployment-State kann nicht geprueft werden: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise OperatorError(f"Deployment-State ist keine regulaere Datei: {path}")
    _require_control_file(path, "Deployment-State")
    return True


def _state_exists(profile: Profile) -> bool:
    return _state_file_exists(_state_path(profile))


def _native_state_exists(profile: Profile) -> bool:
    return _state_file_exists(profile.release_home / "state.json")


def _require_backup_target(profile: Profile) -> Path:
    _require_control_directory(profile.backup_dir, "Backup-Ziel")
    sentinel = profile.backup_dir / BACKUP_TARGET_FILENAME
    _require_control_file(sentinel, "Backup-Ziel-Sentinel")
    before = _path_identity(sentinel)
    document = _decode_object(
        _read_regular(sentinel, MAX_BACKUP_TARGET_BYTES, "Backup-Ziel-Sentinel"),
        "Backup-Ziel-Sentinel",
    )
    _require_control_file(sentinel, "Backup-Ziel-Sentinel")
    if _path_identity(sentinel) != before:
        raise OperatorError("Backup-Ziel-Sentinel wurde bei der Pruefung veraendert")
    if set(document) != {"schema", "target_id"}:
        raise OperatorError("Backup-Ziel-Sentinel hat falsche Schluessel")
    if document["schema"] != BACKUP_TARGET_SCHEMA:
        raise OperatorError("Backup-Ziel-Sentinel hat ein unbekanntes Schema")
    if document["target_id"] != profile.backup_target_id:
        raise OperatorError("Backup-Ziel-Sentinel hat eine andere target_id")
    return sentinel


def _require_state(profile: Profile, operation: str) -> None:
    exists = _state_exists(profile)
    if operation == "bootstrap" and exists:
        raise OperatorError("bootstrap verlangt eine Instanz ohne vorhandenen Deployment-State")
    if operation in {"update", "rollback"} and not exists:
        raise OperatorError(f"{operation} verlangt einen vorhandenen Deployment-State")


def _require_file(path: Path, description: str, *, private: bool = False) -> None:
    absolute = Path(os.path.abspath(path))
    if private:
        _require_private_directory(absolute.parent, description)
    _check_directory_chain(absolute.parent)
    if _is_link_like(absolute):
        raise OperatorError(f"{description} darf kein Link/Reparse-Point sein: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise OperatorError(f"{description} fehlt oder ist nicht lesbar: {absolute}") from exc
    try:
        info = os.fstat(descriptor)
        named = os.lstat(absolute)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _is_reparse(named)
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise OperatorError(f"{description} ist keine sichere regulaere Datei: {absolute}")
    finally:
        os.close(descriptor)
    if private and info.st_nlink != 1:
        raise OperatorError(f"{description} darf keine Hardlinks besitzen: {absolute}")
    if os.name == "nt":
        if private:
            _require_windows_acl(absolute, description, private=True)
    else:
        mode = stat.S_IMODE(info.st_mode)
        if private and mode & 0o077:
            raise OperatorError(f"{description} muss Modus 0600 besitzen: {path}")
        if private:
            geteuid = getattr(os, "geteuid", None)
            current_uid = geteuid() if callable(geteuid) else info.st_uid
            if info.st_uid != current_uid:
                raise OperatorError(f"{description} gehoert einem anderen Benutzer")
        if not private and mode & 0o022:
            raise OperatorError(f"{description} darf nicht gruppen-/welt-schreibbar sein: {path}")


def _require_directory(path: Path, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    _check_directory_chain(absolute)
    if _is_link_like(absolute) or not absolute.is_dir():
        raise OperatorError(f"{description} fehlt oder ist kein sicheres Verzeichnis: {absolute}")
    if os.name != "nt" and stat.S_IMODE(os.lstat(absolute).st_mode) & 0o022:
        raise OperatorError(f"{description} darf nicht gruppen-/welt-schreibbar sein: {absolute}")


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _windows_powershell() -> Path:
    if os.name != "nt":
        raise OperatorError("Windows-ACL-Pruefung wurde auf einem anderen OS aufgerufen")
    try:
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:
            raise OSError("ctypes.WinDLL fehlt")
        kernel32 = win_dll("kernel32", use_last_error=True)
        get_system_directory = kernel32.GetSystemDirectoryW
        get_system_directory.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
        get_system_directory.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32_768)
        length = get_system_directory(buffer, len(buffer))
    except (AttributeError, OSError) as exc:
        raise OperatorError("Windows-Systemverzeichnis kann nicht ermittelt werden") from exc
    if length == 0 or length >= len(buffer):
        raise OperatorError("Windows-Systemverzeichnis kann nicht sicher ermittelt werden")
    executable = Path(buffer.value) / "WindowsPowerShell/v1.0/powershell.exe"
    if not executable.is_absolute() or _is_link_like(executable) or not executable.is_file():
        raise OperatorError("Vertrauenswuerdiges Windows PowerShell wurde nicht gefunden")
    return executable


def _run(
    command: Sequence[str],
    *,
    capture: bool = False,
    timeout: float | None = None,
    extra_environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if not command or any(not isinstance(item, str) or not item or "\x00" in item for item in command):
        raise OperatorError("Interner Delegate-Befehl ist ungueltig")
    if timeout is not None:
        timeout = _positive_timeout(timeout, "Delegate-Timeout")
    environment = _child_environment()
    if extra_environment is not None:
        environment.update(extra_environment)
    try:
        return subprocess.run(
            list(command),
            shell=False,
            check=False,
            capture_output=capture,
            text=True,
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise OperatorError(f"Delegate hat das Zeitlimit ueberschritten: {command[0]}") from exc
    except OSError as exc:
        raise OperatorError(f"Delegate konnte nicht gestartet werden: {command[0]}") from exc


def _windows_script(script: str, path: Path) -> subprocess.CompletedProcess[str]:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return _run(
        [
            str(_windows_powershell()),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        capture=True,
        timeout=WINDOWS_POWERSHELL_TIMEOUT,
        extra_environment={"RISK_LAYER_ACL_PATH": str(Path(os.path.abspath(path)))},
    )


def _windows_acl_document(path: Path) -> dict[str, Any]:
    result = _windows_script(_WINDOWS_ACL_QUERY, path)
    if result.returncode != 0:
        raise OperatorError(
            f"Windows-DACL kann nicht gelesen werden: {path}: "
            f"{result.stderr.strip() or 'PowerShell-Fehler'}"
        )
    try:
        document = json.loads(result.stdout)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise OperatorError("Windows-DACL-Pruefung lieferte kein striktes JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "current",
        "owner",
        "protected",
        "rules",
    }:
        raise OperatorError("Windows-DACL-Pruefung lieferte ein unerwartetes Format")
    if (
        not isinstance(document["current"], str)
        or not isinstance(document["owner"], str)
        or not isinstance(document["protected"], bool)
        or not isinstance(document["rules"], list)
    ):
        raise OperatorError("Windows-DACL-Pruefung lieferte ungueltige Typen")
    for rule in document["rules"]:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"sid", "rights", "type", "inherited"}
            or not isinstance(rule["sid"], str)
            or type(rule["rights"]) is not int
            or rule["type"] not in {"Allow", "Deny"}
            or not isinstance(rule["inherited"], bool)
        ):
            raise OperatorError("Windows-DACL-Pruefung lieferte eine ungueltige Regel")
    return document


def _require_windows_acl(
    path: Path,
    description: str,
    *,
    private: bool,
    strict_control: bool = False,
    require_current_write: bool = False,
) -> None:
    document = _windows_acl_document(path)
    current = document["current"]
    strict = private or strict_control
    owner_trusted = {
        current,
        WINDOWS_SYSTEM_SID,
        WINDOWS_ADMINISTRATORS_SID,
        WINDOWS_TRUSTED_INSTALLER_SID,
    }
    trusted = owner_trusted if strict else owner_trusted | {WINDOWS_OWNER_RIGHTS_SID}
    allowed_owner = {current} if private else owner_trusted
    if document["owner"] not in allowed_owner:
        raise OperatorError(f"{description} hat einen nicht vertrauenswuerdigen Windows-Owner")
    if strict and document["protected"] is not True:
        raise OperatorError(f"{description} muss eine geschuetzte Windows-DACL besitzen")
    current_full_control = False
    current_can_modify = False
    for rule in document["rules"]:
        if strict and rule["sid"] not in trusted:
            raise OperatorError(
                f"{description} enthaelt eine Windows-DACL-Regel fuer eine fremde SID"
            )
        if strict and rule["inherited"] is True:
            raise OperatorError(f"{description} darf keine geerbte Windows-DACL-Regel besitzen")
        if strict and rule["type"] != "Allow":
            raise OperatorError(f"{description} darf keine Windows-Deny-Regel besitzen")
        if (
            not strict
            and rule["type"] == "Allow"
            and rule["sid"] not in trusted
            and rule["rights"] & WINDOWS_MUTATING_RIGHTS
        ):
            raise OperatorError(
                f"{description} ist fuer eine fremde Windows-SID schreibbar"
            )
        if (
            rule["sid"] == current
            and rule["rights"] & WINDOWS_FULL_CONTROL == WINDOWS_FULL_CONTROL
        ):
            current_full_control = True
        if (
            rule["type"] == "Allow"
            and (
                rule["sid"] == current
                or (
                    rule["sid"] == WINDOWS_OWNER_RIGHTS_SID
                    and document["owner"] == current
                )
            )
            and rule["rights"] & WINDOWS_MODIFY == WINDOWS_MODIFY
        ):
            current_can_modify = True
    if private and not current_full_control:
        raise OperatorError(f"{description} gibt dem aktuellen Benutzer kein FullControl")
    if require_current_write and not current_can_modify:
        raise OperatorError(f"{description} gibt dem aktuellen Benutzer kein Modify-Recht")


def _harden_windows_acl(path: Path) -> None:
    result = _windows_script(_WINDOWS_ACL_HARDEN, path)
    if result.returncode != 0:
        raise OperatorError(
            f"Windows-DACL konnte nicht gehaertet werden: {path}: "
            f"{result.stderr.strip() or 'PowerShell-Fehler'}"
        )
    _require_windows_acl(path, "Gehaertete Datei", private=True)


def _harden_windows_directory_acl(path: Path, *, private_owner: bool = True) -> None:
    result = _windows_script(_WINDOWS_DIRECTORY_ACL_HARDEN, path)
    if result.returncode != 0:
        raise OperatorError(
            f"Windows-Verzeichnis-DACL konnte nicht gehaertet werden: {path}: "
            f"{result.stderr.strip() or 'PowerShell-Fehler'}"
        )
    _require_windows_acl(
        path,
        "Gehaertetes Verzeichnis",
        private=private_owner,
        strict_control=not private_owner,
    )


def _passthrough(command: Sequence[str]) -> int:
    return int(_run(command).returncode)


def _interpreter_contract(path: Path, description: str) -> str:
    """Pruefe den profilierten CPython als echten isolierten Interpreter."""

    _require_executable_control(path, description)
    probe = (
        "import json,sys;"
        "print(json.dumps({"
        "'implementation':sys.implementation.name,"
        "'isolated':bool(sys.flags.isolated),"
        "'no_user_site':bool(sys.flags.no_user_site),"
        "'version':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro]"
        "},sort_keys=True,separators=(',',':')))"
    )
    result = _run(
        [str(path), "-I", "-c", probe],
        capture=True,
        timeout=INTERPRETER_CHECK_TIMEOUT,
    )
    if result.returncode != 0:
        raise OperatorError(
            f"{description} erfuellt den isolierten Interpretervertrag nicht"
        )
    try:
        document = json.loads(result.stdout)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise OperatorError(f"{description} lieferte kein striktes JSON") from exc
    expected_keys = {"implementation", "isolated", "no_user_site", "version"}
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise OperatorError(f"{description} lieferte ein unerwartetes Probeformat")
    version = document["version"]
    if (
        document["implementation"] != "cpython"
        or document["isolated"] is not True
        or document["no_user_site"] is not True
        or not isinstance(version, list)
        or len(version) != 3
        or any(type(item) is not int for item in version)
    ):
        raise OperatorError(f"{description} ist kein unterstuetzter isolierter CPython")
    version_tuple = tuple(version)
    if version_tuple < MINIMUM_PYTHON:
        required = ".".join(str(item) for item in MINIMUM_PYTHON)
        actual = ".".join(str(item) for item in version_tuple)
        raise OperatorError(
            f"{description} ist CPython {actual}; mindestens {required} ist erforderlich"
        )
    actual = ".".join(str(item) for item in version_tuple)
    recommendation = "" if version_tuple[:2] == (3, 12) else "; CPython 3.12 empfohlen"
    return f"CPython {actual}, isoliert{recommendation}"


def _ops_command(
    profile: Profile, arguments: Sequence[str], *, force_launcher: bool = False
) -> list[str]:
    _require_control_directory(profile.release_home, "Release-Home")
    native_state = _native_state_exists(profile)
    if force_launcher and not native_state:
        raise OperatorError(
            "Der manifestgebundene Launcher braucht einen vorhandenen nativen current-State"
        )
    if native_state:
        _require_executable_control(profile.python, "Profil-Python")
        _require_control_file(profile.launcher, "Release-Launcher")
        return [
            str(profile.python),
            "-I",
            str(profile.launcher),
            "--home",
            str(profile.release_home),
            "--",
            *arguments,
        ]
    if profile.bootstrap_python is None:
        raise OperatorError(
            "Nativer Bootstrap ohne state.json braucht bootstrap_python aus der kurzlebigen Control-Venv"
        )
    _require_executable_control(profile.bootstrap_python, "Bootstrap-Python")
    return [
        str(profile.bootstrap_python),
        "-I",
        "-m",
        "risk_layer.ops_cli",
        *arguments,
    ]


def _release_tools_command(profile: Profile, arguments: Sequence[str]) -> list[str]:
    release_tools = _required_container_path(profile.release_tools, "release_tools")
    _require_executable_control(profile.python, "Profil-Python")
    _require_control_file(release_tools, "Release-Zipapp")
    return [
        str(profile.python),
        "-I",
        str(release_tools),
        *arguments,
    ]


def _activation_arguments(profile: Profile) -> list[str]:
    result: list[str] = []
    if profile.activation_command is None:
        return ["--no-activate", "--no-health-check"]
    else:
        _require_executable_control(
            Path(profile.activation_command[0]), "Aktivierungs-Executable"
        )
        result.extend(
            (
                "--activation-command",
                json.dumps(
                    list(profile.activation_command),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )
    _require_file(profile.token_file, "Token-Datei", private=True)
    result.extend(
        (
            "--health-url",
            profile.health_url,
            "--token-file",
            str(profile.token_file),
        )
    )
    for layer in profile.required_layers:
        result.extend(("--require-layer", layer))
    return result


def _identity_arguments(identity: Mapping[str, str] | None) -> list[str]:
    if identity is None:
        return []
    if set(identity) != {"release", "git", "manifest"}:
        raise OperatorError("Interne Release-Identitaet ist unvollstaendig")
    release = identity["release"]
    git_sha = identity["git"]
    manifest = identity["manifest"]
    _validate_release_id(release)
    if not GIT_SHA_RE.fullmatch(git_sha) or not SHA256_RE.fullmatch(manifest):
        raise OperatorError("Interne Release-Identitaet ist ungueltig")
    return [
        "--release-id",
        release,
        "--git-sha",
        git_sha,
        "--release-manifest-sha256",
        manifest,
    ]


def _backup_arguments(
    profile: Profile, identity: Mapping[str, str] | None = None
) -> list[str]:
    return [
        "backup",
        "create",
        "--berater",
        str(profile.berater_dir),
        "--ziel",
        str(profile.backup_dir),
        "--backup-target-id",
        profile.backup_target_id,
        "--retention-keep-last",
        str(profile.keep_last),
        "--retention-keep-days",
        str(profile.keep_days),
        *_identity_arguments(identity),
        "--json",
    ]


def _trust_arguments(profile: Profile, args: argparse.Namespace) -> list[str]:
    expected = args.manifest_sha256
    trust_local = bool(args.trust_local_manifest)
    if expected is not None and not SHA256_RE.fullmatch(expected):
        raise OperatorError("--manifest-sha256 muss 64 lowercase Hex-Zeichen enthalten")
    if profile.mode == "container":
        if trust_local:
            raise OperatorError("Container-Releases erlauben kein --trust-local-manifest")
        if expected is None:
            raise OperatorError("Container-Releases brauchen explizit --manifest-sha256")
        return ["--manifest-sha256", expected]
    if expected is not None:
        return ["--manifest-sha256", expected]
    if trust_local:
        return ["--trust-local-manifest"]
    raise OperatorError(
        "Native Releases brauchen --manifest-sha256 oder explizit --trust-local-manifest"
    )


def _validate_release_id(value: Any) -> str:
    if not isinstance(value, str) or not RELEASE_ID_RE.fullmatch(value):
        raise OperatorError("Container-Manifest enthaelt keine sichere release_id")
    if value.endswith((".", " ")) or any(
        ord(character) < 32 or character in WINDOWS_FORBIDDEN for character in value
    ):
        raise OperatorError("Container-Manifest enthaelt eine unportable release_id")
    if value.split(".", 1)[0].upper() in WINDOWS_RESERVED:
        raise OperatorError("Container-Manifest verwendet einen reservierten release_id-Namen")
    return value


def _container_manifest_contract(
    manifest: Path,
    expected_hash: str,
) -> tuple[str, str, str]:
    """Validiert den aeusseren Release-Vertrauensanker einmal zentral."""
    payload = _read_regular(manifest, MAX_MANIFEST_BYTES, "Container-Manifest")
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected_hash):
        raise OperatorError(
            f"Container-Manifest-Hash stimmt nicht: erwartet {expected_hash}, ist {actual}"
        )
    document = _decode_object(payload, "Container-Manifest")
    if document.get("schema") != "taxtronik-release/v1":
        raise OperatorError("Container-Manifest hat ein unerwartetes Schema")
    release_id = _validate_release_id(document.get("release_id"))
    git_sha = document.get("git_sha")
    native_bundle = document.get("native_bundle")
    if not isinstance(git_sha, str) or not GIT_SHA_RE.fullmatch(git_sha):
        raise OperatorError("Container-Manifest enthaelt keinen gueltigen git_sha")
    if not isinstance(native_bundle, dict) or set(native_bundle) != {
        "artifact",
        "sha256",
        "manifest_sha256",
    }:
        raise OperatorError("Container-Manifest hat keinen strikten native_bundle-Block")
    native_manifest = native_bundle.get("manifest_sha256")
    native_artifact = native_bundle.get("artifact")
    native_sha = native_bundle.get("sha256")
    if (
        not isinstance(native_artifact, str)
        or PurePosixPath(native_artifact).name != native_artifact
        or native_artifact in {".", ".."}
        or "\\" in native_artifact
    ):
        raise OperatorError("Container-Manifest hat keinen sicheren nativen Artefaktnamen")
    if not isinstance(native_sha, str) or not SHA256_RE.fullmatch(native_sha):
        raise OperatorError("Container-Manifest hat keinen gueltigen nativen Artefakt-SHA")
    if not isinstance(native_manifest, str) or not SHA256_RE.fullmatch(native_manifest):
        raise OperatorError("Container-Manifest hat keinen gueltigen nativen Manifest-SHA")
    return release_id, git_sha, native_manifest


def _container_release_path(
    profile: Profile,
    manifest: Path,
    expected_hash: str,
) -> tuple[Path, dict[str, str]]:
    release_id, git_sha, native_manifest = _container_manifest_contract(
        manifest, expected_hash
    )
    releases = profile.release_home / "releases"
    target = releases / release_id
    if target.parent != releases:
        raise OperatorError("release_id konnte nicht sicher in einen Release-Pfad abgeleitet werden")
    _require_directory(target, "Manifestgebundener Release-Pfad")
    return target, {
        "release": release_id,
        "git": git_sha,
        "manifest": native_manifest,
    }


def _release_inbox(profile: Profile) -> Path:
    inbox = profile.release_home / RELEASE_INBOX_NAME
    _require_control_directory(inbox, "Release-Eingang")
    return inbox


def _trusted_sidecar_digest(path: Path) -> str:
    """Bindet ein Inbox-Artefakt an seine publizierte sha256sum-Sidecar."""
    _require_control_file(path, "Release-Manifest")
    sidecar = Path(f"{path}.sha256")
    _require_control_file(sidecar, "Release-Pruefsumme")
    payload = _read_regular(sidecar, MAX_CHECKSUM_BYTES, "Release-Pruefsumme")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise OperatorError("Release-Pruefsumme ist nicht ASCII") from exc
    match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\\r\n]+)\r?\n", text)
    if match is None or match.group(2) != path.name:
        raise OperatorError(
            "Release-Pruefsumme muss exakt '<sha256>  <Manifestname>' enthalten"
        )
    expected = match.group(1)
    actual = hashlib.sha256(
        _read_regular(path, MAX_MANIFEST_BYTES, "Release-Manifest")
    ).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise OperatorError(
            f"Release-Pruefsumme stimmt nicht: erwartet {expected}, ist {actual}"
        )
    return expected


def _automatic_deployment_artifact(profile: Profile) -> tuple[Path, str]:
    """Waehlt fail-closed genau ein publiziertes Release aus dem festen Eingang."""
    inbox = _release_inbox(profile)
    try:
        manifests = sorted(inbox.glob(CONTAINER_MANIFEST_PATTERN), key=lambda item: item.name)
    except OSError as exc:
        raise OperatorError("Release-Eingang kann nicht aufgelistet werden") from exc
    safe_manifests: list[Path] = []
    for manifest in manifests:
        if _is_link_like(manifest):
            raise OperatorError(
                f"Release-Eingang enthaelt einen Link/Reparse-Point: {manifest}"
            )
        _require_control_file(manifest, "Release-Manifest")
        safe_manifests.append(manifest)
    if len(safe_manifests) != 1:
        raise OperatorError(
            "Release-Eingang muss genau ein taxtronik-container-*.json enthalten: "
            f"{inbox} (gefunden: {len(safe_manifests)})"
        )

    container_manifest = safe_manifests[0]
    container_digest = _trusted_sidecar_digest(container_manifest)
    release_id, _git_sha, native_manifest_digest = _container_manifest_contract(
        container_manifest, container_digest
    )
    if profile.mode == "container":
        return container_manifest, container_digest

    matches: list[Path] = []
    try:
        children = list(inbox.iterdir())
    except OSError as exc:
        raise OperatorError("Release-Eingang kann nicht aufgelistet werden") from exc
    for candidate in children:
        if _is_link_like(candidate):
            raise OperatorError(
                f"Release-Eingang enthaelt einen Link/Reparse-Point: {candidate}"
            )
        if not candidate.is_dir():
            continue
        manifest = candidate / NATIVE_MANIFEST_NAME
        if not manifest.exists():
            continue
        _require_directory(candidate, "Natives Release-Bundle im Eingang")
        _require_control_file(manifest, "Natives Release-Manifest")
        payload = _read_regular(
            manifest, MAX_MANIFEST_BYTES, "Natives Release-Manifest"
        )
        if not hmac.compare_digest(
            hashlib.sha256(payload).hexdigest(), native_manifest_digest
        ):
            continue
        document = _decode_object(payload, "Natives Release-Manifest")
        if document.get("release_id") != release_id:
            raise OperatorError(
                "Natives Release-Manifest und Container-Vertrauensanker haben "
                "verschiedene release_id"
            )
        matches.append(candidate)
    if len(matches) != 1:
        raise OperatorError(
            "Release-Eingang muss genau ein zum Vertrauensanker passendes entpacktes "
            f"natives Bundle enthalten: {inbox} (gefunden: {len(matches)})"
        )
    return matches[0], native_manifest_digest


def _resolve_deployment_arguments(
    profile: Profile, args: argparse.Namespace
) -> argparse.Namespace:
    if args.artifact is not None:
        return args
    if args.manifest_sha256 is not None or args.trust_local_manifest:
        raise OperatorError(
            "Ohne explizites ARTIFAKT duerfen keine manuellen Trust-Optionen gesetzt werden"
        )
    artifact, digest = _automatic_deployment_artifact(profile)
    args.artifact = str(artifact)
    args.manifest_sha256 = digest
    return args


def _deployment_command(profile: Profile, args: argparse.Namespace) -> list[str]:
    args = _resolve_deployment_arguments(profile, args)
    operation = args.command
    _require_state(profile, operation)
    artifact = Path(os.path.abspath(args.artifact))
    trust = _trust_arguments(profile, args)
    if profile.mode == "native":
        _require_managed_directory(profile.release_home, "Release-Home")
        _require_managed_directory(profile.berater_dir, "Berater-Verzeichnis")
        _require_managed_directory(profile.backup_dir, "Backup-Verzeichnis")
        _require_backup_target(profile)
        if args.dry_run:
            raise OperatorError("Native Deploys besitzen keinen transaktionalen Dry-run")
        _require_directory(artifact, "Natives Release-Bundle")
        command = _ops_command(
            profile,
            [
                "release",
                "install",
                str(artifact),
                "--home",
                str(profile.release_home),
                "--berater",
                str(profile.berater_dir),
                "--backup-ziel",
                str(profile.backup_dir),
                "--backup-target-id",
                profile.backup_target_id,
                "--retention-keep-last",
                str(profile.keep_last),
                "--retention-keep-days",
                str(profile.keep_days),
                *trust,
                *_activation_arguments(profile),
            ],
        )
        return command

    expected = args.manifest_sha256
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise OperatorError(
            "Interner Container-Deploy-Vertrag braucht einen validierten Manifest-SHA"
        )
    state_dir = _required_container_path(profile.state_dir, "state_dir")
    compose_file = _required_container_path(profile.compose_file, "compose_file")
    secrets_env = _required_container_path(profile.secrets_env, "secrets_env")
    docker_bin = _required_container_path(profile.docker_bin, "docker_bin")
    _require_managed_directory(profile.release_home, "Release-Home")
    _require_managed_directory(state_dir, "Container-State")
    _require_managed_directory(profile.berater_dir, "Berater-Verzeichnis")
    _require_managed_directory(profile.backup_dir, "Backup-Verzeichnis")
    _require_backup_target(profile)
    _require_executable_control(docker_bin, "Docker-Executable")
    _require_control_file(compose_file, "Compose-Datei")
    _require_file(secrets_env, "Secret-Env", private=True)
    if not _native_state_exists(profile):
        raise OperatorError(
            "Container-Deploy braucht einen nativen current-State fuer den Pflichtbackup-Launcher"
        )
    release_path, expected_native = _container_release_path(profile, artifact, expected)
    actual_native = _native_current_identity(profile, "Container-Deploy")
    if any(
        not hmac.compare_digest(actual_native[key], expected_native[key])
        for key in ("release", "git", "manifest")
    ):
        raise OperatorError(
            "Native current-Identitaet stimmt nicht mit dem Container-Manifest ueberein"
        )
    backup_hook = _ops_command(profile, _backup_arguments(profile), force_launcher=True)
    command = _release_tools_command(
        profile,
        [
            "deploy-container",
            "deploy",
            str(artifact),
            *trust,
            "--release-path",
            str(release_path),
            "--state-dir",
            str(state_dir),
            "--compose-file",
            str(compose_file),
            "--secrets-env",
            str(secrets_env),
            "--docker-bin",
            str(docker_bin),
            "--backup-hook-json",
            json.dumps(backup_hook, ensure_ascii=False, separators=(",", ":")),
            "--status-url",
            profile.status_url,
        ],
    )
    if args.dry_run:
        command.append("--dry-run")
    return command


def _cmd_deployment(profile: Profile, args: argparse.Namespace) -> int:
    return _passthrough(_deployment_command(profile, args))


def _rollback_command(profile: Profile, args: argparse.Namespace) -> list[str]:
    _require_state(profile, "rollback")
    if profile.mode == "native":
        _require_managed_directory(profile.release_home, "Release-Home")
        if args.dry_run:
            raise OperatorError("Native Rollbacks besitzen keinen transaktionalen Dry-run")
        return _ops_command(
            profile,
            [
                "release",
                "rollback",
                "--home",
                str(profile.release_home),
                *_activation_arguments(profile),
            ],
            force_launcher=True,
        )
    state_dir = _required_container_path(profile.state_dir, "state_dir")
    secrets_env = _required_container_path(profile.secrets_env, "secrets_env")
    docker_bin = _required_container_path(profile.docker_bin, "docker_bin")
    _require_managed_directory(profile.release_home, "Release-Home")
    _require_managed_directory(state_dir, "Container-State")
    _require_managed_directory(profile.berater_dir, "Berater-Verzeichnis")
    _require_managed_directory(profile.backup_dir, "Backup-Verzeichnis")
    _require_backup_target(profile)
    if not _native_state_exists(profile):
        raise OperatorError(
            "Container-Rollback braucht einen nativen current-State fuer den Pflichtbackup-Launcher"
        )
    backup_hook = _ops_command(profile, _backup_arguments(profile), force_launcher=True)
    _require_executable_control(docker_bin, "Docker-Executable")
    _require_file(secrets_env, "Secret-Env", private=True)
    command = _release_tools_command(
        profile,
        [
            "deploy-container",
            "rollback",
            "--state-dir",
            str(state_dir),
            "--secrets-env",
            str(secrets_env),
            "--docker-bin",
            str(docker_bin),
            "--backup-hook-json",
            json.dumps(backup_hook, ensure_ascii=False, separators=(",", ":")),
            "--status-url",
            profile.status_url,
        ],
    )
    if args.dry_run:
        command.append("--dry-run")
    return command


def _cmd_rollback(profile: Profile, args: argparse.Namespace) -> int:
    return _passthrough(_rollback_command(profile, args))


def _decode_delegate(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if result.returncode != 0:
        return {
            "ok": False,
            "exit_code": result.returncode,
            "error": result.stderr.strip() or "Delegate ist fehlgeschlagen",
        }
    try:
        value = json.loads(result.stdout)
    except (json.JSONDecodeError, RecursionError):
        return {
            "ok": False,
            "exit_code": 1,
            "error": "Delegate lieferte kein gueltiges JSON",
        }
    if not isinstance(value, dict):
        return {"ok": False, "exit_code": 1, "error": "Delegate-JSON ist kein Objekt"}
    return {"ok": True, "exit_code": 0, "data": value}


def _status_state_command(profile: Profile) -> list[str]:
    if profile.mode == "native":
        return _ops_command(
            profile,
            ["release", "status", "--home", str(profile.release_home)],
            force_launcher=_state_exists(profile),
        )
    state_dir = _required_container_path(profile.state_dir, "state_dir")
    _require_control_directory(state_dir, "Container-State")
    return _release_tools_command(
        profile,
        ["deploy-container", "status", "--state-dir", str(state_dir)],
    )


def _state_identity(
    profile: Profile,
    wrapped: dict[str, Any],
    *,
    native: bool | None = None,
) -> dict[str, str]:
    if not _delegate_contract_ok(wrapped):
        return {}
    data = wrapped.get("data")
    if not isinstance(data, dict):
        return {}
    state = data.get("state")
    if not isinstance(state, dict):
        return {}
    current = state.get("current")
    if not isinstance(current, dict):
        return {}
    native_mode = profile.mode == "native" if native is None else native
    if native_mode:
        manifest = current.get("manifest_sha256")
    else:
        manifest = current.get("release_manifest_sha256")
    candidates = {
        "release": current.get("release_id"),
        "git": current.get("git_sha"),
        "manifest": manifest,
    }
    release = candidates["release"]
    git_sha = candidates["git"]
    if (
        not isinstance(release, str)
        or not isinstance(git_sha, str)
        or not isinstance(manifest, str)
        or not GIT_SHA_RE.fullmatch(git_sha)
        or not SHA256_RE.fullmatch(manifest)
    ):
        return {}
    try:
        _validate_release_id(release)
    except OperatorError:
        return {}
    return {"release": release, "git": git_sha, "manifest": manifest}


def _delegate_contract_ok(wrapped: dict[str, Any]) -> bool:
    data = wrapped.get("data")
    return (
        wrapped.get("ok") is True
        and isinstance(data, dict)
        and data.get("ok") is True
    )


def _current_identity(profile: Profile, operation: str) -> dict[str, str] | None:
    if not _state_exists(profile):
        if profile.mode == "container":
            raise OperatorError(
                f"{operation} braucht einen vorhandenen Container-current-State"
            )
        return None
    wrapped = _decode_delegate(_run(_status_state_command(profile), capture=True))
    identity = _state_identity(profile, wrapped)
    if not _delegate_contract_ok(wrapped) or set(identity) != {
        "release",
        "git",
        "manifest",
    }:
        raise OperatorError(
            f"{operation} braucht eine vollstaendige validierte current-Identitaet"
        )
    return identity


def _native_current_identity(profile: Profile, operation: str) -> dict[str, str]:
    if not _native_state_exists(profile):
        raise OperatorError(f"{operation} braucht einen nativen current-State")
    wrapped = _decode_delegate(
        _run(
            _ops_command(
                profile,
                ["release", "status", "--home", str(profile.release_home)],
                force_launcher=True,
            ),
            capture=True,
        )
    )
    identity = _state_identity(profile, wrapped, native=True)
    if not _delegate_contract_ok(wrapped) or set(identity) != {
        "release",
        "git",
        "manifest",
    }:
        raise OperatorError(
            f"{operation} braucht eine vollstaendige validierte native current-Identitaet"
        )
    return identity


def _health_command(
    profile: Profile, identity: dict[str, str], timeout: float
) -> list[str]:
    _require_file(profile.token_file, "Token-Datei", private=True)
    arguments = [
        "doctor",
        "--url",
        profile.health_url,
        "--token-file",
        str(profile.token_file),
        "--timeout",
        str(timeout),
        "--json",
    ]
    if "release" in identity:
        arguments.extend(("--expected-release", identity["release"]))
    if "git" in identity:
        arguments.extend(("--expected-git-sha", identity["git"]))
    if "manifest" in identity:
        arguments.extend(("--expected-manifest-sha256", identity["manifest"]))
    for layer in profile.required_layers:
        arguments.extend(("--require-layer", layer))
    return _ops_command(profile, arguments, force_launcher=_state_exists(profile))


def _cmd_status(profile: Profile, args: argparse.Namespace) -> int:
    state = _decode_delegate(_run(_status_state_command(profile), capture=True))
    identity = _state_identity(profile, state)
    identity_complete = set(identity) == {"release", "git", "manifest"}
    health: dict[str, Any]
    if args.offline:
        health = {"ok": None, "skipped": "offline"}
    elif not identity_complete:
        health = {"ok": False, "skipped": "deployment identity incomplete"}
    else:
        health = _decode_delegate(
            _run(
                _health_command(profile, identity, args.timeout),
                capture=True,
            )
        )
    ok = (
        _delegate_contract_ok(state)
        and identity_complete
        and (args.offline or _delegate_contract_ok(health))
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "mode": profile.mode,
                "profile": str(profile.source),
                "deployment": state,
                "identity": {"complete": identity_complete, **identity},
                "health": health,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if ok else 1


def _create_secure_directory(path: Path) -> list[Path]:
    absolute = Path(os.path.abspath(path))
    missing: list[Path] = []
    cursor = absolute
    while not cursor.exists():
        if _is_link_like(cursor):
            raise OperatorError(f"Zu erzeugendes Verzeichnis ist umgelenkt: {cursor}")
        missing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    _check_directory_chain(cursor)
    if not cursor.is_dir():
        raise OperatorError(f"Elternpfad ist kein Verzeichnis: {cursor}")
    created: list[Path] = []
    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
        if os.name != "nt":
            os.chmod(directory, 0o700)
        created.append(directory)
    _check_directory_chain(absolute)
    return created


def _create_private_file(
    path: Path, payload: bytes, *, private_parent: bool = True
) -> bool:
    if path.exists() or _is_link_like(path):
        return False
    new_directories = _create_secure_directory(path.parent)
    for directory in new_directories:
        if os.name == "nt":
            _harden_windows_directory_acl(directory)
        else:
            os.chmod(directory, 0o700)
    if private_parent:
        _require_private_directory(path.parent, "Private Datei")
    else:
        _require_control_directory(path.parent, "Lokale Steuerdatei")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return False
    created = os.fstat(descriptor)
    os.close(descriptor)
    descriptor = -1
    try:
        if os.name == "nt":
            # Die Datei bleibt leer, bis die geerbte DACL durch einen
            # geschuetzten, SID-basierten Descriptor ersetzt und geprueft ist.
            _harden_windows_acl(path)
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_TRUNC
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        named = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _is_reparse(named)
            or (opened.st_dev, opened.st_ino) != (created.st_dev, created.st_ino)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise OperatorError(f"Private Datei wurde vor dem Schreiben ausgetauscht: {path}")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            os.chmod(path, 0o600)
    except BaseException:
        try:
            named = os.lstat(path)
            if (named.st_dev, named.st_ino) == (created.st_dev, created.st_ino):
                path.unlink()
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return True


def _read_token(path: Path) -> str:
    _require_file(path, "Token-Datei", private=True)
    before = _path_identity(path)
    payload = _read_regular(path, 4096, "Token-Datei")
    _require_file(path, "Token-Datei", private=True)
    if _path_identity(path) != before:
        raise OperatorError("Token-Datei wurde waehrend der Sicherheitspruefung veraendert")
    try:
        token = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise OperatorError("Token-Datei ist nicht gueltiges UTF-8") from exc
    if len(token.encode("utf-8")) < 32 or any(character.isspace() for character in token):
        raise OperatorError("Token-Datei muss mindestens 32 Bytes ohne Whitespace enthalten")
    return token


def _dotenv_value(value: str) -> str:
    if not value or any(ord(character) < 32 for character in value):
        raise OperatorError("Nicht darstellbarer Wert fuer generierte Secret-Env")
    if re.fullmatch(r"[A-Za-z0-9_./:@\\-]+", value):
        return value
    if "'" in value:
        raise OperatorError("Secret-Env-Wert mit Apostroph ist nicht sicher darstellbar")
    return f"'{value}'"


def _parse_secret_env(path: Path) -> dict[str, str]:
    _require_file(path, "Secret-Env", private=True)
    before = _path_identity(path)
    payload = _read_regular(path, MAX_SECRET_BYTES, "Secret-Env")
    _require_file(path, "Secret-Env", private=True)
    if _path_identity(path) != before:
        raise OperatorError("Secret-Env wurde waehrend der Sicherheitspruefung veraendert")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperatorError("Secret-Env ist nicht gueltiges UTF-8") from exc
    values: dict[str, str] = {}
    folded_keys: set[str] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise OperatorError(f"Secret-Env-Zeile {number} ist ungueltig")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        folded_key = key.casefold()
        if (
            not ENV_KEY_RE.fullmatch(key)
            or folded_key in folded_keys
        ):
            raise OperatorError(f"Secret-Env-Schluessel ist ungueltig/doppelt: {key!r}")
        if not key.startswith("RISK_LAYER_") or key.casefold().startswith("compose_"):
            raise OperatorError(f"Secret-Env darf nur RISK_LAYER_*-Werte enthalten: {key}")
        if key.casefold() in {item.casefold() for item in MANAGED_CONTAINER_ENV}:
            raise OperatorError(f"Secret-Env darf verwalteten Wert nicht setzen: {key}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = value.replace(r'\"', '"').replace(r"\\", "\\")
        if "\x00" in value or "\r" in value or "\n" in value:
            raise OperatorError(f"Secret-Env-Wert ist mehrzeilig: {key}")
        values[key] = value
        folded_keys.add(folded_key)
    token = values.get("RISK_LAYER_TOKEN", "")
    if len(token.encode("utf-8")) < 32:
        raise OperatorError("Secret-Env enthaelt kein RISK_LAYER_TOKEN mit mindestens 32 Bytes")
    return values


def _doctor_fix(profile: Profile) -> list[str]:
    # Ein fehlendes Off-host-Ziel darf niemals lokal nachgebaut werden: Ein
    # ausgefallener Mount wuerde sonst scheinbar erfolgreiche Backups erzeugen.
    _require_managed_directory(profile.backup_dir, "Explizites Backup-Ziel")
    _require_backup_target(profile)
    directories = [
        profile.release_home,
        profile.release_home / RELEASE_INBOX_NAME,
        profile.berater_dir,
    ]
    secrets_env: Path | None = None
    if profile.mode == "container":
        directories.append(_required_container_path(profile.state_dir, "state_dir"))
        secrets_env = _required_container_path(profile.secrets_env, "secrets_env")
    created: list[str] = []
    for directory in directories:
        new_directories = _create_secure_directory(directory)
        for item in new_directories:
            created.append(str(item))
            if os.name == "nt":
                _harden_windows_directory_acl(item, private_owner=False)
    private_parents = {profile.token_file.parent}
    if secrets_env is not None:
        private_parents.add(secrets_env.parent)
    for parent in sorted(private_parents, key=str):
        new_directories = _create_secure_directory(parent)
        for item in new_directories:
            created.append(str(item))
            if os.name == "nt":
                _harden_windows_directory_acl(item)
            else:
                os.chmod(item, 0o700)
        _require_private_directory(parent, "Private Konfiguration")
    if not profile.token_file.exists():
        token = (
            _parse_secret_env(secrets_env)["RISK_LAYER_TOKEN"]
            if secrets_env is not None and secrets_env.exists()
            else secrets.token_urlsafe(48)
        )
        if _create_private_file(profile.token_file, f"{token}\n".encode("utf-8")):
            created.append(str(profile.token_file))
    _require_file(profile.token_file, "Token-Datei", private=True)
    token = _read_token(profile.token_file)
    if secrets_env is not None and not secrets_env.exists():
        payload = (
            f"RISK_LAYER_TOKEN={_dotenv_value(token)}\n"
            f"RISK_LAYER_DEFINITIONEN_DIR={_dotenv_value(str(profile.berater_dir))}\n"
        ).encode("utf-8")
        if _create_private_file(secrets_env, payload):
            created.append(str(secrets_env))
    return created


def _doctor_checks(profile: Profile) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(name: str, operation: Callable[[], str | None]) -> None:
        try:
            detail = operation()
        except (OperatorError, OSError) as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})
        else:
            item: dict[str, Any] = {"name": name, "ok": True}
            if detail is not None:
                item["detail"] = detail
            checks.append(item)

    check("python", lambda: _interpreter_contract(profile.python, "Profil-Python"))
    check("release_home", lambda: _require_managed_directory(profile.release_home, "Release-Home"))
    check("release_inbox", lambda: str(_release_inbox(profile)))
    check("berater_dir", lambda: _require_managed_directory(profile.berater_dir, "Berater-Verzeichnis"))
    check("backup_dir", lambda: _require_managed_directory(profile.backup_dir, "Backup-Verzeichnis"))
    check("backup_target", lambda: str(_require_backup_target(profile)))
    check("launcher", lambda: _require_control_file(profile.launcher, "Release-Launcher"))
    check("token_file", lambda: _require_file(profile.token_file, "Token-Datei", private=True))

    def valid_token() -> None:
        _read_token(profile.token_file)

    check("token", valid_token)
    if profile.mode == "container":
        docker_bin = _required_container_path(profile.docker_bin, "docker_bin")
        state_dir = _required_container_path(profile.state_dir, "state_dir")
        release_tools = _required_container_path(profile.release_tools, "release_tools")
        compose_file = _required_container_path(profile.compose_file, "compose_file")
        secrets_env = _required_container_path(profile.secrets_env, "secrets_env")
        check("state_dir", lambda: _require_managed_directory(state_dir, "Container-State"))
        check(
            "docker_bin",
            lambda: _require_executable_control(docker_bin, "Docker-Executable"),
        )
        check("release_tools", lambda: _require_control_file(release_tools, "Release-Zipapp"))
        check("compose_file", lambda: _require_control_file(compose_file, "Compose-Datei"))
        check("secrets_env", lambda: _require_file(secrets_env, "Secret-Env", private=True))

        def secret_matches() -> None:
            values = _parse_secret_env(secrets_env)
            if not hmac.compare_digest(values["RISK_LAYER_TOKEN"], _read_token(profile.token_file)):
                raise OperatorError("Token-Datei und Secret-Env enthalten verschiedene Tokens")
            if values.get("RISK_LAYER_DEFINITIONEN_DIR") != str(profile.berater_dir):
                raise OperatorError("Secret-Env zeigt nicht auf das profilierte Berater-Verzeichnis")

        check("secrets_contract", secret_matches)

        def native_release_state() -> None:
            if not _native_state_exists(profile):
                raise OperatorError(
                    "Container-Betrieb braucht einen nativen current-State fuer den stabilen Launcher"
                )

        check("native_release_state", native_release_state)
    if profile.activation_command is not None:
        activation_executable = Path(profile.activation_command[0])
        check(
            "activation_executable",
            lambda: _require_executable_control(
                activation_executable, "Aktivierungs-Executable"
            ),
        )
    try:
        state_exists: bool | None = _state_exists(profile)
        state_error: str | None = None
    except (OperatorError, OSError) as exc:
        state_exists = None
        state_error = str(exc)

    if profile.mode == "native" and state_exists is False:
        # Nur der allererste native Bootstrap benoetigt eine vorgelagerte,
        # kurzlebige Control-Venv. Nach state.json ist allein der Launcher Pflicht.
        def bootstrap_import() -> str:
            if profile.bootstrap_python is None:
                raise OperatorError(
                    "Nativer Bootstrap ohne State braucht bootstrap_python im Profil"
                )
            interpreter = _interpreter_contract(
                profile.bootstrap_python, "Bootstrap-Python"
            )
            result = _run(
                [str(profile.bootstrap_python), "-I", "-c", "import risk_layer.ops_cli"],
                capture=True,
                timeout=INTERPRETER_CHECK_TIMEOUT,
            )
            if result.returncode != 0:
                raise OperatorError(
                    "Native Bootstrap-Control-Venv enthaelt risk_layer.ops_cli nicht"
                )
            return interpreter

        check("bootstrap_ops_cli", bootstrap_import)
    checks.append({
        "name": "deployment_state",
        "ok": state_error is None,
        "detail": state_error if state_error is not None else (
            "present" if state_exists else "absent"
        ),
    })
    return checks


def _cmd_doctor(profile: Profile, args: argparse.Namespace) -> int:
    created = _doctor_fix(profile) if args.fix else []
    checks = _doctor_checks(profile)
    ok = all(item["ok"] is True for item in checks)
    print(
        json.dumps(
            {
                "ok": ok,
                "mode": profile.mode,
                "profile": str(profile.source),
                "created": created,
                "checks": checks,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if ok else 1


def _cmd_backup(profile: Profile, args: argparse.Namespace) -> int:
    _require_operator_berater_directory(profile, args)
    _require_managed_directory(profile.backup_dir, "Backup-Verzeichnis")
    _require_backup_target(profile)
    identity = _current_identity(profile, "Backup")
    return _passthrough(
        _operator_ops_command(profile, args, _backup_arguments(profile, identity))
    )


def _cmd_backup_verify(profile: Profile, args: argparse.Namespace) -> int:
    _require_backup_target(profile)
    if args.file is None:
        entries = _backup_entries(profile)
        if not entries:
            raise OperatorError("Kein Backup zum Verifizieren vorhanden")
        archive = Path(entries[0]["path"])
    else:
        archive = Path(os.path.abspath(args.file))
    _require_file(archive, "Backup-Archiv")
    return _passthrough(
        _operator_ops_command(profile, args, ["backup", "verify", str(archive)])
    )


def _cmd_backup_prune(profile: Profile, args: argparse.Namespace) -> int:
    if args.apply:
        _require_managed_directory(profile.backup_dir, "Backup-Verzeichnis")
    else:
        _require_control_directory(profile.backup_dir, "Backup-Verzeichnis")
    _require_backup_target(profile)
    command = [
        "backup",
        "prune",
        "--ziel",
        str(profile.backup_dir),
        "--backup-target-id",
        profile.backup_target_id,
        "--keep-last",
        str(profile.keep_last),
        "--keep-days",
        str(profile.keep_days),
    ]
    if args.apply:
        command.append("--apply")
    return _passthrough(_operator_ops_command(profile, args, command))


def _backup_entries(profile: Profile) -> list[dict[str, Any]]:
    _require_control_directory(profile.backup_dir, "Backup-Verzeichnis")
    _require_backup_target(profile)
    entries: list[dict[str, Any]] = []
    try:
        candidates = list(profile.backup_dir.glob("festwissen_*.zip"))
    except OSError as exc:
        raise OperatorError("Backup-Verzeichnis kann nicht aufgelistet werden") from exc
    for path in candidates:
        if any(ord(character) < 32 for character in path.name):
            raise OperatorError(f"Backup-Dateiname enthaelt Steuerzeichen: {path.name!r}")
        if _is_link_like(path):
            raise OperatorError(f"Backup-Liste enthaelt einen Link/Reparse-Point: {path}")
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            raise OperatorError(f"Backup-Liste enthaelt keine regulaere Datei: {path}")
        digest, identity = _hash_backup_archive(path)
        entries.append(
            {
                "name": path.name,
                "path": str(path),
                "size": identity[2],
                "sha256": digest,
                "modified_at": datetime.fromtimestamp(
                    info.st_mtime, timezone.utc
                ).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "_mtime_ns": info.st_mtime_ns,
                "_identity": identity,
            }
        )
    entries.sort(key=lambda item: (item["_mtime_ns"], item["name"]), reverse=True)
    return entries


def _hash_backup_archive(path: Path) -> tuple[str, tuple[int, int, int, int]]:
    _require_file(path, "Backup-Archiv")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OperatorError(f"Backup-Archiv kann nicht sicher geoeffnet werden: {path}") from exc
    try:
        before = os.fstat(descriptor)
        named = os.lstat(path)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not stat.S_ISREG(named.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _is_reparse(named)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise OperatorError(f"Backup-Archiv wurde beim Oeffnen ausgetauscht: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
            raise OperatorError(f"Backup-Archiv wurde beim Hashen veraendert: {path}")
        final_named = os.lstat(path)
        if (
            not stat.S_ISREG(final_named.st_mode)
            or stat.S_ISLNK(final_named.st_mode)
            or _is_reparse(final_named)
            or final_named.st_nlink != 1
            or (
                final_named.st_dev,
                final_named.st_ino,
                final_named.st_size,
                final_named.st_mtime_ns,
            )
            != identity
        ):
            raise OperatorError(f"Backup-Archiv wurde nach dem Hashen ausgetauscht: {path}")
        return digest.hexdigest(), identity
    finally:
        os.close(descriptor)


def _cmd_restore(profile: Profile, args: argparse.Namespace) -> int:
    if args.list or (not args.latest and args.file is None):
        entries = _backup_entries(profile)
        for item in entries:
            item.pop("_mtime_ns", None)
            item.pop("_identity", None)
        print(json.dumps({"ok": True, "backups": entries}, ensure_ascii=False, indent=2))
        return 0
    if args.latest:
        entries = _backup_entries(profile)
        if not entries:
            raise OperatorError("Kein Backup fuer --latest vorhanden")
        archive = Path(entries[0]["path"])
        expected_hash = entries[0]["sha256"]
        expected_identity = entries[0]["_identity"]
    else:
        archive = Path(os.path.abspath(args.file))
        expected_hash, expected_identity = _hash_backup_archive(archive)
    confirmation = (
        f"RESTORE {archive.name} SHA256 {expected_hash} TO {profile.berater_dir}"
    )
    if args.confirm != confirmation:
        raise OperatorError(
            "Restore-Bestaetigung fehlt oder stimmt nicht. Exakt angeben: "
            f"--confirm {confirmation!r}"
        )
    actual_hash, actual_identity = _hash_backup_archive(archive)
    if actual_identity != expected_identity or not hmac.compare_digest(
        actual_hash, expected_hash
    ):
        raise OperatorError("Restore-Archiv wurde nach der Auswahl ausgetauscht")
    _require_operator_berater_directory(profile, args)
    _require_control_directory(profile.backup_dir, "Backup-Verzeichnis")
    _require_backup_target(profile)
    identity = _current_identity(profile, "Restore-Vorher-Sicherung")
    result = _passthrough(
        _operator_ops_command(
            profile,
            args,
            [
                "backup",
                "restore",
                str(archive),
                "--berater",
                str(profile.berater_dir),
                "--backup-ziel",
                str(profile.backup_dir),
                "--backup-target-id",
                profile.backup_target_id,
                "--expected-sha256",
                expected_hash,
                *_identity_arguments(identity),
                "--force",
                "--json",
            ],
        )
    )
    if result == 0 and isinstance(getattr(args, "checkout_root", None), Path):
        _normalize_checkout_berater_ownership(profile.berater_dir)
    return result


def _explicit_profile_requested(arguments: Sequence[str]) -> bool:
    return any(
        value in {"--config", "--profile"}
        or value.startswith("--config=")
        or value.startswith("--profile=")
        for value in arguments
    )


def _checkout_root(arguments: Sequence[str]) -> Path | None:
    raw = os.environ.get(CHECKOUT_ROOT_ENV, "").strip()
    if (
        not raw
        or os.environ.get("RISK_LAYER_OPS_PROFILE", "").strip()
        or _explicit_profile_requested(arguments)
    ):
        return None
    root = Path(os.path.abspath(raw))
    _require_control_directory(root, "Lokaler Checkout")
    for relative, description in (
        ("pyproject.toml", "Checkout-Projektvertrag"),
        ("risk_layer/operator_cli.py", "Checkout-Operator"),
    ):
        _require_control_file(root / relative, description)
    setup = root / ("scripts/setup.ps1" if os.name == "nt" else "scripts/setup.sh")
    _require_control_file(setup, "Checkout-Setup")
    return root


def _checkout_target_id(root: Path) -> str:
    canonical = os.path.normcase(os.path.abspath(root)).encode(
        "utf-8", errors="surrogatepass"
    )
    return CHECKOUT_BACKUP_TARGET_PREFIX + hashlib.sha256(canonical).hexdigest()[:32]


def _checkout_health_urls(root: Path) -> tuple[str, str]:
    state = root / CHECKOUT_STATE_DIR / CHECKOUT_SERVICE_STATE_NAME
    if not state.exists() and not _is_link_like(state):
        return "http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/status"
    _require_control_file(state, "Lokaler Service-State")
    document = _decode_object(
        _read_regular(state, MAX_BACKUP_TARGET_BYTES, "Lokaler Service-State"),
        "Lokaler Service-State",
    )
    if set(document) != {"schema", "host", "port", "scope"}:
        raise OperatorError("Lokaler Service-State hat unerwartete Felder")
    if document["schema"] != CHECKOUT_SERVICE_SCHEMA:
        raise OperatorError("Lokaler Service-State hat ein unerwartetes Schema")
    host = document["host"]
    port = document["port"]
    scope = document["scope"]
    if not isinstance(host, str) or not host or any(ord(char) < 32 for char in host):
        raise OperatorError("Lokaler Service-State hat keinen gueltigen Host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host != "localhost":
            raise OperatorError("Lokaler Service-State hat keinen gueltigen Host") from None
        address = None
    if type(port) is not int or not 1 <= port <= 65_535:
        raise OperatorError("Lokaler Service-State hat keinen gueltigen Port")
    if scope not in {"system", "user", "windows-task"}:
        raise OperatorError("Lokaler Service-State hat keinen gueltigen Scope")
    health_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if address is not None and address.version == 6 and ":" in health_host:
        health_host = f"[{health_host}]"
    base = f"http://{health_host}:{port}"
    return base, f"{base}/v1/status"


def _checkout_profile(root: Path) -> Profile:
    state_root = root / CHECKOUT_STATE_DIR
    health_url, status_url = _checkout_health_urls(root)
    return Profile(
        source=state_root / "checkout.json",
        mode="native",
        python=Path(os.path.abspath(sys.executable)),
        bootstrap_python=Path(os.path.abspath(sys.executable)),
        docker_bin=None,
        release_home=state_root / "runtime",
        state_dir=None,
        berater_dir=root / "definitionen",
        backup_dir=state_root / "backups",
        backup_target_id=_checkout_target_id(root),
        launcher=root / "deploy/release-launcher.py",
        release_tools=None,
        compose_file=None,
        secrets_env=None,
        token_file=root / ".engine_token",
        health_url=health_url,
        status_url=status_url,
        activation_command=None,
        required_layers=("matcher",),
        keep_last=30,
        keep_days=90,
    )


def _checkout_service_identity() -> tuple[int, int] | None:
    if os.name == "nt":
        return None
    geteuid = getattr(os, "geteuid", None)
    if not callable(geteuid) or geteuid() != 0:
        return None
    try:
        import pwd

        lookup = getattr(pwd, "getpwnam", None)
        if not callable(lookup):
            return None
        account = lookup(CHECKOUT_SYSTEMD_USER)
    except (ImportError, KeyError):
        return None
    if account.pw_uid == 0:
        raise OperatorError(
            f"Dienstkonto {CHECKOUT_SYSTEMD_USER} darf nicht UID 0 besitzen"
        )
    return account.pw_uid, account.pw_gid


def _require_checkout_data_directory(path: Path, description: str) -> None:
    if os.name == "nt":
        _require_private_directory(path, description)
        return
    _require_directory(path, f"{description}-Elternverzeichnis")
    info = os.lstat(path)
    geteuid = getattr(os, "geteuid", None)
    current_uid = geteuid() if callable(geteuid) else info.st_uid
    allowed = {current_uid}
    identity = _checkout_service_identity()
    if identity is not None:
        allowed.add(identity[0])
    if info.st_uid not in allowed:
        raise OperatorError(
            f"{description}-Elternverzeichnis gehoert keinem erlaubten Checkout-Konto"
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise OperatorError(f"{description}-Elternverzeichnis muss Modus 0700 besitzen")


def _require_operator_berater_directory(
    profile: Profile,
    args: argparse.Namespace,
) -> None:
    if isinstance(getattr(args, "checkout_root", None), Path):
        _require_checkout_data_directory(profile.berater_dir, "Berater-Verzeichnis")
    else:
        _require_managed_directory(profile.berater_dir, "Berater-Verzeichnis")


def _normalize_checkout_berater_ownership(path: Path) -> None:
    identity = _checkout_service_identity()
    if identity is None:
        return
    uid, gid = identity
    chown = getattr(os, "chown", None)
    if not callable(chown):
        raise OperatorError("Checkout-Dienstkonto kann auf diesem System nicht gesetzt werden")
    _require_directory(path, "Lokale Definitionen")
    for directory, child_directories, filenames in os.walk(path, followlinks=False):
        current = Path(directory)
        for name in [*child_directories, *filenames]:
            candidate = current / name
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode):
                raise OperatorError(
                    f"Lokale Definitionen duerfen keine Links enthalten: {candidate}"
                )
            if stat.S_ISDIR(info.st_mode):
                mode = 0o700
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise OperatorError(
                        f"Lokale Definition darf keine Hardlinks besitzen: {candidate}"
                    )
                mode = 0o600
            else:
                raise OperatorError(
                    f"Lokale Definition hat einen ungueltigen Dateityp: {candidate}"
                )
            chown(candidate, uid, gid, follow_symlinks=False)
            os.chmod(candidate, mode, follow_symlinks=False)
    chown(path, uid, gid, follow_symlinks=False)
    os.chmod(path, 0o700, follow_symlinks=False)


def _secure_checkout_directory(
    path: Path,
    *,
    allow_service_owner: bool = False,
) -> list[str]:
    if path.exists() or _is_link_like(path):
        identity = _checkout_service_identity() if allow_service_owner else None
        owner = os.lstat(path).st_uid if os.name != "nt" and path.exists() else None
        if identity is not None and owner == identity[0]:
            _require_checkout_data_directory(path, "Vorhandenes Checkout-Verzeichnis")
        else:
            _require_managed_directory(path, "Vorhandenes Checkout-Verzeichnis")
    created = _create_secure_directory(path)
    for directory in created:
        if os.name == "nt":
            _harden_windows_directory_acl(directory)
        else:
            os.chmod(directory, 0o700)
    if os.name == "nt":
        _harden_windows_directory_acl(path)
    else:
        os.chmod(path, 0o700)
    if allow_service_owner:
        _require_checkout_data_directory(path, "Lokales Checkout-Verzeichnis")
    else:
        _require_private_directory(path, "Lokales Checkout-Verzeichnis")
    return [str(item) for item in created]


def _require_checkout_token(path: Path) -> str:
    _require_control_file(path, "Lokales Engine-Token")
    payload = _read_regular(path, 4096, "Lokales Engine-Token")
    try:
        token = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise OperatorError("Lokales Engine-Token ist nicht gueltiges UTF-8") from exc
    if len(token.encode("utf-8")) < 32 or any(character.isspace() for character in token):
        raise OperatorError("Lokales Engine-Token muss mindestens 32 Bytes ohne Whitespace enthalten")
    return str(path)


def _checkout_token(path: Path) -> bool:
    if path.exists() or _is_link_like(path):
        _require_control_file(path, "Vorhandenes lokales Engine-Token")
    created = _create_private_file(
        path,
        f"{secrets.token_urlsafe(48)}\n".encode("utf-8"),
        private_parent=False,
    )
    if os.name == "nt":
        _harden_windows_acl(path)
    else:
        os.chmod(path, 0o600)
    _require_checkout_token(path)
    return created


def _ensure_checkout_layout(profile: Profile) -> list[str]:
    created: list[str] = []
    for directory in (
        profile.release_home,
        profile.release_home / RELEASE_INBOX_NAME,
        profile.backup_dir,
    ):
        created.extend(_secure_checkout_directory(directory))
    created.extend(
        _secure_checkout_directory(profile.berater_dir, allow_service_owner=True)
    )
    sentinel = profile.backup_dir / BACKUP_TARGET_FILENAME
    if _create_private_file(
        sentinel,
        (
            json.dumps(
                {
                    "schema": BACKUP_TARGET_SCHEMA,
                    "target_id": profile.backup_target_id,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    ):
        created.append(str(sentinel))
    if _checkout_token(profile.token_file):
        created.append(str(profile.token_file))
    _require_backup_target(profile)
    return created


def _checkout_setup_command(root: Path) -> list[str]:
    if os.name == "nt":
        script = root / "scripts/setup.ps1"
        _require_control_file(script, "Lokales Windows-Setup")
        return [
            str(_windows_powershell()),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Service",
        ]
    executable = shutil.which("bash")
    if executable is None:
        raise OperatorError("Lokaler Bootstrap braucht bash")
    bash = Path(os.path.abspath(executable))
    _require_executable_control(bash, "Bash")
    script = root / "scripts/setup.sh"
    _require_control_file(script, "Lokales Linux-Setup")
    return [str(bash), str(script), "--service"]


def _checkout_service_installed(root: Path) -> bool:
    if os.name == "nt":
        definition = root / "service-start.ps1"
    else:
        definition, _scope, _description = _checkout_systemd_scope()
    return definition.is_file() and not _is_link_like(definition)


def _checkout_service_ready(root: Path, profile: Profile) -> None:
    if not _checkout_is_bootstrapped(root, profile):
        raise OperatorError("Dienststeuerung braucht zuerst ./signal bootstrap")
    _require_checkout_token(profile.token_file)
    _require_control_file(root / "corpus/graph.sqlite", "Lokaler Normgraph")
    _checkout_python(root)


def _checkout_systemd_scope() -> tuple[Path, list[str], str]:
    geteuid = getattr(os, "geteuid", None)
    if callable(geteuid) and geteuid() == 0:
        return (
            CHECKOUT_SYSTEMD_SYSTEM_DIR / CHECKOUT_SYSTEMD_UNIT,
            [],
            "Lokale systemd-System-Unit",
        )
    return (
        Path.home() / ".config/systemd/user" / CHECKOUT_SYSTEMD_UNIT,
        ["--user"],
        "Lokale systemd-User-Unit",
    )


def _checkout_service_command(root: Path, operation: str) -> list[str]:
    if operation not in {"start", "stop", "restart"}:
        raise OperatorError(f"Unbekannte Dienstaktion: {operation}")
    if os.name == "nt":
        wrapper = root / "service-start.ps1"
        _require_control_file(wrapper, "Windows-Autostart-Wrapper")
        powershell = _windows_powershell()
        if operation == "restart":
            script = root / "deploy/restart-risk-layer-task.ps1"
            _require_control_file(script, "Windows-Neustart-Hook")
            return [
                str(powershell),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-TaskName",
                CHECKOUT_WINDOWS_TASK,
            ]
        verb = "Start-ScheduledTask" if operation == "start" else "Stop-ScheduledTask"
        guard = (
            "$task.State -ne 'Running'"
            if operation == "start"
            else "$task.State -eq 'Running'"
        )
        command = (
            "$ErrorActionPreference='Stop';"
            f"$task=Get-ScheduledTask -TaskName '{CHECKOUT_WINDOWS_TASK}' "
            "-TaskPath '\\' -ErrorAction Stop;"
            f"if ({guard}) {{{verb} -InputObject $task -ErrorAction Stop}}"
        )
        return [
            str(powershell),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ]
    executable = shutil.which("systemctl")
    if executable is None:
        raise OperatorError("Dienststeuerung braucht systemctl")
    systemctl = Path(os.path.abspath(executable))
    _require_executable_control(systemctl, "systemctl")
    unit, scope, description = _checkout_systemd_scope()
    _require_control_file(unit, description)
    non_blocking = ["--no-block"] if operation in {"start", "restart"} else []
    return [str(systemctl), *scope, *non_blocking, operation, CHECKOUT_SYSTEMD_UNIT]


def _checkout_service_control(
    root: Path,
    profile: Profile,
    operation: str,
) -> int:
    _checkout_service_ready(root, profile)
    result = _passthrough(_checkout_service_command(root, operation))
    if result == 0:
        labels = {"start": "gestartet", "stop": "gestoppt", "restart": "neu gestartet"}
        print(f"Risk-Layer-Dienst {labels[operation]}.")
    return result


def _cmd_profiled_service_control(profile: Profile, args: argparse.Namespace) -> int:
    del profile
    raise OperatorError(
        f"signal {args.command} ist fuer den lokalen Checkout-Service bestimmt; "
        "profilierte Instanzen werden durch ihren konfigurierten Service-Manager gesteuert"
    )


def _checkout_python(root: Path) -> Path:
    python = Path(os.path.abspath(sys.executable))
    venv = Path(os.path.abspath(root / ".venv"))
    try:
        inside_venv = os.path.commonpath((str(python), str(venv))) == str(venv)
    except ValueError:
        inside_venv = False
    if not inside_venv:
        raise OperatorError(
            "Lokale Betriebsbefehle brauchen zuerst ./signal bootstrap und den "
            "Interpreter aus .venv"
        )
    _require_control_directory(python.parent, "Checkout-Venv")
    if not _is_link_like(python):
        _require_executable_control(python, "Checkout-Python")
        return python
    if os.name == "nt":
        raise OperatorError("Checkout-Python darf unter Windows kein Reparse-Point sein")
    info = os.lstat(python)
    geteuid = getattr(os, "geteuid", None)
    current_uid = geteuid() if callable(geteuid) else info.st_uid
    if info.st_uid not in {0, current_uid}:
        raise OperatorError("Checkout-Python-Symlink gehoert einem anderen Benutzer")
    target = Path(os.path.realpath(python))
    _require_executable_control(target, "Ziel des Checkout-Python-Symlinks")
    if not os.access(python, os.X_OK):
        raise OperatorError("Checkout-Python ist nicht ausfuehrbar")
    return python


def _checkout_ops_command(root: Path, arguments: Sequence[str]) -> list[str]:
    return [str(_checkout_python(root)), "-I", "-m", "risk_layer.ops_cli", *arguments]


def _operator_ops_command(
    profile: Profile,
    args: argparse.Namespace,
    arguments: Sequence[str],
) -> list[str]:
    checkout = getattr(args, "checkout_root", None)
    if isinstance(checkout, Path):
        return _checkout_ops_command(checkout, arguments)
    return _ops_command(profile, arguments)


def _checkout_git(root: Path, arguments: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("git")
    if executable is None:
        raise OperatorError("Lokales Update braucht Git")
    git = Path(os.path.abspath(executable))
    _require_executable_control(git, "Git")
    return _run([str(git), "-C", str(root), *arguments], capture=capture)


def _checkout_is_bootstrapped(root: Path, profile: Profile) -> bool:
    return (
        profile.token_file.is_file()
        and (root / ".venv").is_dir()
        and (root / "corpus/graph.sqlite").is_file()
    )


def _checkout_deployment(root: Path, profile: Profile, args: argparse.Namespace) -> int:
    if (
        args.artifact is not None
        or args.manifest_sha256 is not None
        or args.trust_local_manifest
        or args.dry_run
    ):
        raise OperatorError(
            "Der lokale Checkout-Pfad ist parameterlos; explizite Release-Artefakte "
            "brauchen --config PROFIL"
        )
    if args.command == "deploy":
        raise OperatorError(
            "Lokales deploy gibt es nicht: ./signal bootstrap installiert einmalig; "
            "./signal update zieht spaetere Git-Aenderungen. Deploy ist fuer "
            "profilierte Release-Artefakte bestimmt"
        )
    if args.command == "bootstrap":
        _ensure_checkout_layout(profile)
        if _checkout_is_bootstrapped(root, profile) and _checkout_service_installed(root):
            raise OperatorError(
                "Checkout ist bereits eingerichtet; fuer Quellcode-Aenderungen "
                "./signal update, fuer den Dienst ./signal restart verwenden"
            )
        return _passthrough(_checkout_setup_command(root))
    if args.command != "update":
        raise OperatorError(f"Unbekannter lokaler Deployment-Befehl: {args.command}")
    if not _checkout_is_bootstrapped(root, profile):
        raise OperatorError("Lokales update braucht zuerst ./signal bootstrap")
    status = _checkout_git(root, ["status", "--porcelain"], capture=True)
    if status.returncode != 0:
        raise OperatorError("Git-Arbeitsbaum konnte vor dem Update nicht geprueft werden")
    if status.stdout.strip():
        entries = [line[:512] for line in status.stdout.splitlines() if line][:20]
        remaining = len(status.stdout.splitlines()) - len(entries)
        details = "\n".join(f"  {line}" for line in entries)
        if remaining > 0:
            details += f"\n  ... und {remaining} weitere"
        raise OperatorError(
            "Lokales update braucht einen sauberen Git-Arbeitsbaum. "
            f"Blockierende Eintraege:\n{details}"
        )
    pulled = _checkout_git(root, ["pull", "--ff-only"])
    if pulled.returncode != 0:
        return int(pulled.returncode)
    return _checkout_service_control(root, profile, "restart")


def _checkout_status(root: Path, profile: Profile, args: argparse.Namespace) -> int:
    checks = {
        "venv": (root / ".venv").is_dir(),
        "token": profile.token_file.is_file(),
        "graph": (root / "corpus/graph.sqlite").is_file(),
        "berater": profile.berater_dir.is_dir(),
        "backup": profile.backup_dir.is_dir(),
    }
    revision = _checkout_git(root, ["rev-parse", "HEAD"], capture=True)
    git_sha = revision.stdout.strip() if revision.returncode == 0 else None
    health: dict[str, Any]
    if args.offline or not all(checks.values()):
        health = {"ok": None, "skipped": "offline" if args.offline else "bootstrap incomplete"}
    else:
        doctor_arguments = [
            "doctor", "--url", profile.health_url,
            "--token-file", str(profile.token_file),
            "--timeout", str(args.timeout),
            "--require-layer", "matcher", "--json",
        ]
        if urlsplit(profile.health_url).scheme == "http":
            # Der Checkout-State stammt aus dem lokalen Service-Installer. Eine
            # konkrete Docker-Bridge-Adresse ist daher ein bewusst lokales Ziel.
            doctor_arguments.append("--allow-insecure-http")
        health = _decode_delegate(
            _run(
                _checkout_ops_command(root, doctor_arguments),
                capture=True,
            )
        )
    ok = all(checks.values()) and (args.offline or _delegate_contract_ok(health))
    print(
        json.dumps(
            {
                "ok": ok,
                "mode": "checkout",
                "root": str(root),
                "git_sha": git_sha,
                "checks": checks,
                "health": health,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if ok else 1


def _checkout_doctor(root: Path, profile: Profile, args: argparse.Namespace) -> int:
    created = _ensure_checkout_layout(profile) if args.fix else []
    checks: list[dict[str, Any]] = []

    def check(name: str, operation: Callable[[], str | None]) -> None:
        try:
            detail = operation()
        except (OperatorError, OSError) as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})
        else:
            item: dict[str, Any] = {"name": name, "ok": True}
            if detail is not None:
                item["detail"] = detail
            checks.append(item)

    check("checkout", lambda: _require_control_directory(root, "Lokaler Checkout"))
    check("python", lambda: _interpreter_contract(profile.python, "Checkout-Python"))
    check("setup", lambda: _require_control_file(root / ("scripts/setup.ps1" if os.name == "nt" else "scripts/setup.sh"), "Checkout-Setup"))
    check("runtime", lambda: _require_private_directory(profile.release_home, "Lokale Runtime"))
    check(
        "berater",
        lambda: _require_checkout_data_directory(
            profile.berater_dir, "Lokale Definitionen"
        ),
    )
    check("backup", lambda: _require_private_directory(profile.backup_dir, "Lokales Backup"))
    check("backup_target", lambda: str(_require_backup_target(profile)))
    check("token", lambda: _require_checkout_token(profile.token_file))
    check("graph", lambda: _require_control_file(root / "corpus/graph.sqlite", "Lokaler Normgraph"))
    ok = all(item["ok"] for item in checks)
    print(
        json.dumps(
            {
                "ok": ok,
                "mode": "checkout",
                "root": str(root),
                "created": created,
                "checks": checks,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if ok else 1


def _dispatch_checkout(root: Path, args: argparse.Namespace) -> int:
    profile = _checkout_profile(root)
    args.checkout_root = root
    if args.command in {"bootstrap", "deploy", "update"}:
        return _checkout_deployment(root, profile, args)
    if args.command == "status":
        return _checkout_status(root, profile, args)
    if args.command == "doctor":
        return _checkout_doctor(root, profile, args)
    if args.command in {"start", "stop", "restart"}:
        return _checkout_service_control(root, profile, args.command)
    if args.command == "backup":
        _ensure_checkout_layout(profile)
        return _cmd_backup(profile, args)
    if args.command == "backup-verify":
        return _cmd_backup_verify(profile, args)
    if args.command == "backup-prune":
        return _cmd_backup_prune(profile, args)
    if args.command == "restore":
        return _cmd_restore(profile, args)
    if args.command == "rollback":
        raise OperatorError(
            "Ein lokaler Git-Checkout besitzt kein transaktionales Release-Rollback; "
            "dafuer signal mit einem expliziten Release-Profil verwenden"
        )
    raise OperatorError(f"Unbekanntes lokales Kommando: {args.command}")


def _add_deployment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "artifact",
        nargs="?",
        metavar="ARTIFAKT",
        help=(
            "Pfad zum nativen Bundle-Verzeichnis oder, im Container-Modus, "
            "zum Release-Manifest; ohne Angabe wird der kontrollierte Eingang "
            "<release_home>/incoming verwendet"
        ),
    )
    trust = parser.add_mutually_exclusive_group()
    trust.add_argument(
        "--manifest-sha256",
        metavar="HASH",
        help="erwarteter SHA-256 (bei explizitem Container-Artefakt Pflicht)",
    )
    trust.add_argument(
        "--trust-local-manifest",
        action="store_true",
        help="lokales natives Manifest bewusst vertrauen (nie fuer Container)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Container-Transaktion nur simulieren"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal",
        description=(
            "Zero-Config im Checkout; profilierte, transaktionale Releases im Betrieb."
        ),
        epilog=(
            "Im Checkout erzeugt bootstrap alle lokalen Pfade selbst. Mit einem "
            "expliziten Profil wird ohne ARTIFAKT genau ein gehashtes Release aus "
            "<release_home>/incoming gewaehlt."
        ),
    )
    parser.add_argument(
        "--config",
        dest="config",
        default=os.environ.get("RISK_LAYER_OPS_PROFILE", DEFAULT_PROFILE),
        metavar="PFAD",
        help="striktes nicht geheimes Instanzprofil (Standard: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        dest="config",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    sub = parser.add_subparsers(dest="command")
    deployment_help = {
        "bootstrap": "Erstinstallation ohne vorhandenen Deployment-State",
        "deploy": "Release passend zum profilierten Modus anwenden",
        "update": "vorhandene Instanz transaktional aktualisieren",
    }
    for name, help_text in deployment_help.items():
        command = sub.add_parser(name, help=help_text, description=help_text)
        _add_deployment_arguments(command)
        command.set_defaults(handler=_cmd_deployment)

    status = sub.add_parser(
        "status", help="Deployment-State und Health aggregieren"
    )
    status.add_argument("--offline", action="store_true", help="Health-Abfrage auslassen")
    status.add_argument(
        "--timeout", type=float, default=30.0, help="Health-Zeitlimit in Sekunden"
    )
    status.set_defaults(handler=_cmd_status)

    doctor = sub.add_parser("doctor", help="Profil, Pfade, Rechte und Vertrag pruefen")
    doctor.add_argument(
        "--fix",
        action="store_true",
        help="nur sichere Verzeichnisse, Release-Eingang, Token und Secret-Env anlegen",
    )
    doctor.set_defaults(handler=_cmd_doctor)

    service_help = {
        "start": "lokalen Engine- und LLM-Dienst starten",
        "stop": "lokalen Engine- und LLM-Dienst stoppen",
        "restart": "lokalen Engine- und LLM-Dienst neu starten",
    }
    for name, help_text in service_help.items():
        service = sub.add_parser(name, help=help_text, description=help_text)
        service.set_defaults(handler=_cmd_profiled_service_control)

    backup = sub.add_parser("backup", help="manifestiertes Festwissen-Backup anlegen")
    backup.set_defaults(handler=_cmd_backup)

    verify = sub.add_parser("backup-verify", help="Backup ohne Mutation voll pruefen")
    verify.add_argument(
        "file", nargs="?", metavar="ARCHIV", help="Standard: neuestes Backup"
    )
    verify.set_defaults(handler=_cmd_backup_verify)

    prune = sub.add_parser("backup-prune", help="profilierte Retention auswerten")
    prune.add_argument("--apply", action="store_true", help="ermittelte Altbackups loeschen")
    prune.set_defaults(handler=_cmd_backup_prune)

    restore = sub.add_parser("restore", help="Backups auflisten oder sicher wiederherstellen")
    selection = restore.add_mutually_exclusive_group()
    selection.add_argument("--list", action="store_true", help="Backups rein lesend auflisten")
    selection.add_argument("--latest", action="store_true", help="neuestes Backup waehlen")
    selection.add_argument("--file", metavar="ARCHIV", help="bestimmtes Backup waehlen")
    restore.add_argument("--confirm", metavar="TEXT", help="an Archiv und Ziel gebundene Freigabe")
    restore.set_defaults(handler=_cmd_restore)

    rollback = sub.add_parser("rollback", help="atomar zum vorherigen Release wechseln")
    rollback.add_argument("--dry-run", action="store_true", help="nur fuer Container")
    rollback.set_defaults(handler=_cmd_rollback)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if hasattr(args, "timeout"):
            args.timeout = _positive_timeout(args.timeout, "--timeout")
        checkout = _checkout_root(arguments)
        if checkout is not None:
            return _dispatch_checkout(checkout, args)
        profile = load_profile(args.config)
        return int(args.handler(profile, args))
    except OperatorError as exc:
        print(f"Operator-Fehler: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Operator-Fehler: Betriebssystemzugriff fehlgeschlagen: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
