#!/usr/bin/env python3
"""Transaktionales Deploy und Rollback eines verifizierten Compose-Releases.

Das Werkzeug verwaltet ausschliesslich nicht-geheime Release-Metadaten. Das
separate Secret-Env wird direkt an Docker Compose weitergereicht und niemals in
State, generierte Env-Dateien oder Ausgaben kopiert.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence
from urllib import request
from urllib.parse import urlsplit

from risk_layer.dateisperre import DateisperrenFehler, festwissen_sperre
from risk_layer.release_lifecycle import IntegrityError, verify_bundle
from risk_layer.release_manifest import (
    ManifestError,
    ReleaseManifest,
    validate_release_id,
)


STATE_FORMAT_VERSION = 2
JOURNAL_FORMAT_VERSION = 1
CONTAINER_SCHEMA = "taxtronik-release/v1"
COMPOSE_PROJECT = "risk-layer"
JOURNAL_FILENAME = "transaction.json"
MAX_JSON_BYTES = 1024 * 1024
MAX_SECRET_ENV_BYTES = 1024 * 1024
MAX_STATUS_BYTES = 1024 * 1024
MAX_COMMAND_TIMEOUT = 3600.0
MAX_VERIFY_TIMEOUT = 300.0
MAX_LOCK_TIMEOUT = 3600.0
WINDOWS_POWERSHELL_TIMEOUT = 15.0
WINDOWS_SYSTEM_SID = "S-1-5-18"
WINDOWS_ADMINISTRATORS_SID = "S-1-5-32-544"
WINDOWS_TRUSTED_INSTALLER_SID = (
    "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
)
WINDOWS_FULL_CONTROL = 0x1F01FF
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REPOSITORY_RE = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,511}$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

BOOTSTRAP_KEYS = frozenset(
    {
        "release_tools",
        "verify_container",
        "extract_native",
        "deploy_container",
        "install_release_service",
        "release_systemd_service_template",
        "release_launcher",
        "windows_install_service",
        "windows_restart_task",
        "windows_install_backup_task",
        "operator_profile_container_linux",
        "operator_profile_native_linux",
        "operator_profile_native_windows",
        "operator_wrapper_posix",
        "operator_wrapper_windows",
        "backup_systemd_service",
        "backup_systemd_timer",
        "backup_systemd_env",
        "container_secrets_env_example",
    }
)

MANAGED_ENV_KEYS = frozenset(
    {
        "RISK_LAYER_ENGINE_REPOSITORY",
        "RISK_LAYER_ENGINE_DIGEST",
        "RISK_LAYER_RELEASE_ID",
        "RISK_LAYER_RELEASE_GIT_SHA",
        "RISK_LAYER_RELEASE_MANIFEST_SHA256",
        "RISK_LAYER_FESTWISSEN_DIR",
    }
)


_WINDOWS_ACL_QUERY = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
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
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$path = [System.IO.Path]::GetFullPath($env:RISK_LAYER_ACL_PATH)
$current = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
if ($null -eq $current) { throw 'Aktuelle Benutzer-SID fehlt.' }
$acl = [System.IO.File]::GetAccessControl($path)
[void]$acl.SetOwner($current)
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
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$path = [System.IO.Path]::GetFullPath($env:RISK_LAYER_ACL_PATH)
$current = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
if ($null -eq $current) { throw 'Aktuelle Benutzer-SID fehlt.' }
$acl = [System.IO.Directory]::GetAccessControl($path)
[void]$acl.SetOwner($current)
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


class _NoRedirects(request.HTTPRedirectHandler):
    """Gibt den Bearer-Token niemals an ein Redirect-Ziel weiter."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


_STATUS_OPENER = request.build_opener(_NoRedirects())


class DeployError(RuntimeError):
    """Ein Deployment wurde abgelehnt oder konnte nicht aktiviert werden."""

    def __init__(self, message: str, *, rolled_back: bool | None = None) -> None:
        super().__init__(message)
        self.rolled_back = rolled_back


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _is_reparse(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and getattr(info, "st_file_attributes", 0) & flag)


def _is_link_like(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DeployError(f"Pfad kann nicht sicher geprueft werden: {path}") from exc
    return stat.S_ISLNK(info.st_mode) or _is_reparse(info)


def _check_directory_chain(path: Path, description: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in reversed((absolute, *absolute.parents)):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise DeployError(
                f"{description} kann nicht sicher geprueft werden: {component}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            raise DeployError(
                f"Symlink-/Reparse-Komponente in {description}: {component}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise DeployError(
                f"Nicht-Verzeichnis-Komponente in {description}: {component}"
            )


def _ensure_directory(path: Path, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    _check_directory_chain(absolute, description)
    absolute.mkdir(parents=True, exist_ok=True)
    _check_directory_chain(absolute, description)
    if _is_link_like(absolute) or not absolute.is_dir():
        raise DeployError(f"{description} ist kein sicheres Verzeichnis: {absolute}")
    if os.name != "nt":
        os.chmod(absolute, 0o700)
    return absolute


def _secure_file_bytes(
    path: Path,
    *,
    maximum: int,
    description: str,
    private: bool = False,
    windows_private_acl: bool = False,
) -> bytes:
    absolute = Path(os.path.abspath(path))
    _check_directory_chain(absolute.parent, f"{description}-Elternpfad")
    if _is_link_like(absolute):
        raise DeployError(
            f"{description} darf kein Symlink/Reparse-Point sein: {absolute}"
        )
    try:
        named = os.lstat(absolute)
    except OSError as exc:
        raise DeployError(f"{description} fehlt oder ist unlesbar: {absolute}") from exc
    if not stat.S_ISREG(named.st_mode):
        raise DeployError(f"{description} ist keine regulaere Datei: {absolute}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise DeployError(
            f"{description} kann nicht sicher geoeffnet werden: {absolute}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            named.st_dev,
            named.st_ino,
        ):
            raise DeployError(
                f"{description} wurde beim Oeffnen ausgetauscht: {absolute}"
            )
        if private and opened.st_nlink != 1:
            raise DeployError(f"{description} darf keine Hardlinks besitzen")
        if private and os.name != "nt":
            current_uid = getattr(os, "geteuid")()
            if opened.st_uid != current_uid:
                raise DeployError(f"{description} gehoert einem anderen Benutzer")
            if stat.S_IMODE(opened.st_mode) & 0o077:
                raise DeployError(
                    f"{description} hat zu weite Rechte; erwartet wird 0600"
                )
        if windows_private_acl:
            _require_windows_private_file_acl(absolute, description)
            try:
                acl_checked = os.lstat(absolute)
            except OSError as exc:
                raise DeployError(
                    f"{description} wurde waehrend der Windows-DACL-Pruefung ausgetauscht"
                ) from exc
            if (acl_checked.st_dev, acl_checked.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise DeployError(
                    f"{description} wurde waehrend der Windows-DACL-Pruefung ausgetauscht"
                )
        if opened.st_size > maximum:
            raise DeployError(f"{description} ueberschreitet das Groessenlimit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(maximum + 1)
        final = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            raise DeployError(f"{description} wurde waehrend des Lesens veraendert")
        try:
            final_named = os.lstat(absolute)
        except OSError as exc:
            raise DeployError(
                f"{description} wurde waehrend des Lesens ausgetauscht"
            ) from exc
        if (final_named.st_dev, final_named.st_ino) != (opened.st_dev, opened.st_ino):
            raise DeployError(f"{description} wurde waehrend des Lesens ausgetauscht")
    finally:
        os.close(descriptor)
    if len(payload) > maximum:
        raise DeployError(f"{description} ueberschreitet das Groessenlimit")
    return payload


def _secure_optional_file_bytes(
    path: Path,
    *,
    maximum: int,
    description: str,
    private: bool = False,
) -> bytes | None:
    absolute = Path(os.path.abspath(path))
    _check_directory_chain(absolute.parent, f"{description}-Elternpfad")
    try:
        os.lstat(absolute)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DeployError(
            f"{description} kann nicht sicher geprueft werden: {absolute}"
        ) from exc
    return _secure_file_bytes(
        absolute,
        maximum=maximum,
        description=description,
        private=private,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, description: str, *, private: bool = False) -> str:
    return _sha256_bytes(
        _secure_file_bytes(
            path,
            maximum=MAX_JSON_BYTES * 64,
            description=description,
            private=private,
        )
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeployError(f"Doppelter JSON-Schluessel: {key!r}")
        result[key] = value
    return result


def _decode_json(payload: bytes, description: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DeployError(f"Ungueltige JSON-Konstante in {description}: {value}")
            ),
        )
    except DeployError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DeployError(f"{description} ist kein gueltiges UTF-8-JSON") from exc


def _windows_system_powershell() -> tuple[Path, Path]:
    if os.name != "nt":
        raise DeployError("Windows-DACL-Pruefung wurde auf einem anderen OS aufgerufen")
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
        raise DeployError(
            "Windows-Systemverzeichnis kann nicht ermittelt werden"
        ) from exc
    if length == 0 or length >= len(buffer):
        raise DeployError(
            "Windows-Systemverzeichnis kann nicht sicher ermittelt werden"
        )
    system_directory = Path(buffer.value)
    executable = system_directory / "WindowsPowerShell/v1.0/powershell.exe"
    _check_directory_chain(executable.parent, "Windows-PowerShell-Systempfad")
    if (
        not executable.is_absolute()
        or _is_link_like(executable)
        or not executable.is_file()
    ):
        raise DeployError("Vertrauenswuerdiges Windows PowerShell wurde nicht gefunden")
    return executable, system_directory


def _windows_acl_script(script: str, path: Path) -> subprocess.CompletedProcess[str]:
    executable, system_directory = _windows_system_powershell()
    system_root = system_directory.parent
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    environment = {
        "COMSPEC": str(system_directory / "cmd.exe"),
        "PATH": str(system_directory),
        "RISK_LAYER_ACL_PATH": str(Path(os.path.abspath(path))),
        "SystemRoot": str(system_root),
        "WINDIR": str(system_root),
    }
    try:
        return subprocess.run(
            [
                str(executable),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=WINDOWS_POWERSHELL_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise DeployError(
            "Windows-DACL-Pruefung hat das Zeitlimit ueberschritten"
        ) from exc
    except OSError as exc:
        raise DeployError(
            "Windows-DACL-Pruefung konnte nicht gestartet werden"
        ) from exc


def _windows_acl_document(path: Path) -> dict[str, Any]:
    result = _windows_acl_script(_WINDOWS_ACL_QUERY, path)
    if result.returncode != 0:
        raise DeployError(
            f"Windows-DACL kann nicht gelesen werden: {path}: "
            f"{result.stderr.strip() or 'PowerShell-Fehler'}"
        )
    document = _decode_json(result.stdout.encode("utf-8"), "Windows-DACL-Pruefung")
    if not isinstance(document, dict) or set(document) != {
        "current",
        "owner",
        "protected",
        "rules",
    }:
        raise DeployError("Windows-DACL-Pruefung lieferte ein unerwartetes Format")
    if (
        not isinstance(document["current"], str)
        or not isinstance(document["owner"], str)
        or not isinstance(document["protected"], bool)
        or not isinstance(document["rules"], list)
    ):
        raise DeployError("Windows-DACL-Pruefung lieferte ungueltige Typen")
    for rule in document["rules"]:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"sid", "rights", "type", "inherited"}
            or not isinstance(rule["sid"], str)
            or type(rule["rights"]) is not int
            or rule["type"] not in {"Allow", "Deny"}
            or not isinstance(rule["inherited"], bool)
        ):
            raise DeployError("Windows-DACL-Pruefung lieferte eine ungueltige Regel")
    return document


def _validate_windows_private_acl(
    document: Mapping[str, Any], description: str
) -> None:
    current = document["current"]
    trusted = {
        current,
        WINDOWS_SYSTEM_SID,
        WINDOWS_ADMINISTRATORS_SID,
        WINDOWS_TRUSTED_INSTALLER_SID,
    }
    if document["owner"] != current:
        raise DeployError(f"{description} hat einen fremden Windows-Owner")
    if document["protected"] is not True:
        raise DeployError(f"{description} muss eine geschuetzte Windows-DACL besitzen")
    current_full_control = False
    for rule in document["rules"]:
        if rule["sid"] not in trusted:
            raise DeployError(
                f"{description} enthaelt eine Windows-DACL-Regel fuer eine fremde SID"
            )
        if rule["inherited"] is True:
            raise DeployError(
                f"{description} darf keine geerbte Windows-DACL-Regel besitzen"
            )
        if rule["type"] != "Allow":
            raise DeployError(f"{description} darf keine Windows-Deny-Regel besitzen")
        if (
            rule["sid"] == current
            and rule["rights"] & WINDOWS_FULL_CONTROL == WINDOWS_FULL_CONTROL
        ):
            current_full_control = True
    if not current_full_control:
        raise DeployError(
            f"{description} gibt dem aktuellen Benutzer kein Windows-FullControl"
        )


def _require_windows_private_file_acl(path: Path, description: str) -> None:
    if os.name != "nt":
        return
    _validate_windows_private_acl(_windows_acl_document(path), description)


def _harden_windows_private_file_acl(path: Path, description: str) -> None:
    if os.name != "nt":
        return
    result = _windows_acl_script(_WINDOWS_ACL_HARDEN, path)
    if result.returncode != 0:
        raise DeployError(
            f"{description}-DACL konnte nicht gehaertet werden: "
            f"{result.stderr.strip() or 'PowerShell-Fehler'}"
        )
    _require_windows_private_file_acl(path, description)


def _harden_windows_private_directory_acl(path: Path, description: str) -> None:
    if os.name != "nt":
        return
    result = _windows_acl_script(_WINDOWS_DIRECTORY_ACL_HARDEN, path)
    if result.returncode != 0:
        raise DeployError(
            f"{description}-DACL konnte nicht gehaertet werden: "
            f"{result.stderr.strip() or 'PowerShell-Fehler'}"
        )
    _require_windows_private_file_acl(path, description)


def _require_keys(value: Any, keys: set[str], description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DeployError(f"{description} hat fehlende oder unbekannte Felder")
    return value


def _artifact_block(value: Any, keys: set[str], description: str) -> dict[str, Any]:
    block = _require_keys(value, keys, description)
    artifact = block.get("artifact")
    if (
        not isinstance(artifact, str)
        or not artifact
        or artifact in {".", ".."}
        or PurePosixPath(artifact).name != artifact
        or "/" in artifact
        or "\\" in artifact
    ):
        raise DeployError(f"Ungueltiger Artefaktname in {description}")
    sha = block.get("sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        raise DeployError(f"Ungueltiger SHA-256 in {description}")
    return block


@dataclass(frozen=True, slots=True)
class ManifestIdentity:
    release_id: str
    git_sha: str
    engine_repository: str
    engine_digest: str
    engine_image: str
    container_manifest_sha256: str
    release_manifest_sha256: str
    compose_artifact: str
    compose_sha256: str


def _load_container_manifest(path: Path, expected_sha256: str) -> ManifestIdentity:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise DeployError("--manifest-sha256 muss ein kleingeschriebener SHA-256 sein")
    payload = _secure_file_bytes(
        path, maximum=MAX_JSON_BYTES, description="Container-Manifest"
    )
    actual_sha256 = _sha256_bytes(payload)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise DeployError(
            "Container-Manifest stimmt nicht mit dem erwarteten SHA-256 ueberein"
        )
    decoded = _decode_json(payload, "Container-Manifest")
    base_keys = {
        "schema",
        "release_id",
        "git_sha",
        "created_at",
        "engine",
        "festwissen",
        "native_bundle",
        "compose",
        "bootstrap",
    }
    if not isinstance(decoded, dict):
        raise DeployError("Container-Manifest muss ein JSON-Objekt sein")
    actual_keys = frozenset(decoded)
    if actual_keys not in {frozenset(base_keys), frozenset(base_keys | {"llm_model"})}:
        raise DeployError("Container-Manifest hat fehlende oder unbekannte Felder")
    document = decoded
    if document["schema"] != CONTAINER_SCHEMA:
        raise DeployError("Nicht unterstuetztes Container-Manifest-Schema")
    try:
        release_id = validate_release_id(document["release_id"])
    except (ManifestError, TypeError) as exc:
        raise DeployError("Ungueltige Release-ID im Container-Manifest") from exc
    git_sha = document["git_sha"]
    if not isinstance(git_sha, str) or not GIT_SHA_RE.fullmatch(git_sha):
        raise DeployError("Ungueltiger Git-SHA im Container-Manifest")
    if not isinstance(document["created_at"], str) or not document["created_at"]:
        raise DeployError("created_at fehlt im Container-Manifest")

    engine = _require_keys(
        document["engine"], {"repository", "digest", "image"}, "Engine-Block"
    )
    repository = engine["repository"]
    digest = engine["digest"]
    if not isinstance(repository, str) or not REPOSITORY_RE.fullmatch(repository):
        raise DeployError("Ungueltiges Engine-Repository")
    if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
        raise DeployError("Ungueltiger Engine-Digest")
    image = f"{repository}@{digest}"
    if engine["image"] != image:
        raise DeployError("Engine-Image widerspricht Repository/Digest")

    festwissen = _artifact_block(
        document["festwissen"],
        {"artifact", "sha256", "manifest", "manifest_sha256", "embedding"},
        "Festwissen-Block",
    )
    if not isinstance(festwissen["embedding"], bool):
        raise DeployError("festwissen.embedding muss boolesch sein")
    if (
        not isinstance(festwissen["manifest"], str)
        or festwissen["manifest"] in {".", ".."}
        or PurePosixPath(festwissen["manifest"]).name != festwissen["manifest"]
        or "/" in festwissen["manifest"]
        or "\\" in festwissen["manifest"]
    ):
        raise DeployError("Ungueltiger Festwissen-Manifestname")
    if not isinstance(festwissen["manifest_sha256"], str) or not SHA256_RE.fullmatch(
        festwissen["manifest_sha256"]
    ):
        raise DeployError("Ungueltiger Festwissen-Manifest-SHA")

    native = _artifact_block(
        document["native_bundle"],
        {"artifact", "sha256", "manifest_sha256"},
        "Native-Bundle-Block",
    )
    release_manifest_sha256 = native["manifest_sha256"]
    if not isinstance(release_manifest_sha256, str) or not SHA256_RE.fullmatch(
        release_manifest_sha256
    ):
        raise DeployError("Ungueltiger nativer Release-Manifest-SHA")
    compose = _artifact_block(
        document["compose"], {"artifact", "sha256"}, "Compose-Block"
    )
    bootstrap = _require_keys(
        document["bootstrap"], set(BOOTSTRAP_KEYS), "Bootstrap-Block"
    )
    bootstrap_artifacts: set[str] = set()
    for key in sorted(BOOTSTRAP_KEYS):
        block = _artifact_block(
            bootstrap[key], {"artifact", "sha256"}, f"Bootstrap-Block {key}"
        )
        folded = block["artifact"].casefold()
        if folded in bootstrap_artifacts:
            raise DeployError("Bootstrap-Artefaktnamen kollidieren")
        bootstrap_artifacts.add(folded)
    if "llm_model" in document:
        # Das transaktionale Werkzeug aktiviert bewusst nur den Engine-Service.
        # Ein optional gebundenes Modell wird trotzdem streng geparst, damit ein
        # unbekanntes/mehrdeutiges erweitertes Manifest nicht still akzeptiert
        # wird. Die lokale Modelldatei prueft der vorgeschaltete Manifest-Verify.
        llm_model = _artifact_block(
            document["llm_model"],
            {"artifact", "sha256", "revision", "provenance"},
            "LLM-Modell-Block",
        )
        if not llm_model["artifact"].casefold().endswith(".gguf"):
            raise DeployError("LLM-Modell-Artefakt muss eine GGUF-Datei sein")
        for field in ("revision", "provenance"):
            value = llm_model[field]
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 4096
                or any(ord(character) < 32 for character in value)
            ):
                raise DeployError(f"Ungueltiges llm_model.{field}")
    return ManifestIdentity(
        release_id=release_id,
        git_sha=git_sha,
        engine_repository=repository,
        engine_digest=digest,
        engine_image=image,
        container_manifest_sha256=actual_sha256,
        release_manifest_sha256=release_manifest_sha256,
        compose_artifact=compose["artifact"],
        compose_sha256=compose["sha256"],
    )


@dataclass(frozen=True, slots=True)
class DeploymentRecord:
    release_id: str
    git_sha: str
    schema_version: str
    api_version: str
    features: tuple[str, ...]
    engine_repository: str
    engine_digest: str
    engine_image: str
    container_manifest_sha256: str
    release_manifest_sha256: str
    compose_artifact: str
    compose_sha256: str
    compose_file: str
    release_path: str
    env_generation: int
    env_file: str
    env_sha256: str
    activated_at: str

    @classmethod
    def from_dict(cls, value: Any) -> "DeploymentRecord":
        data = _require_keys(
            value,
            {
                "release_id",
                "git_sha",
                "schema_version",
                "api_version",
                "features",
                "engine_repository",
                "engine_digest",
                "engine_image",
                "container_manifest_sha256",
                "release_manifest_sha256",
                "compose_artifact",
                "compose_sha256",
                "compose_file",
                "release_path",
                "env_generation",
                "env_file",
                "env_sha256",
                "activated_at",
            },
            "Deployment-Record",
        )
        features = data.get("features")
        if not isinstance(features, list):
            raise DeployError("features im Deployment-State muss eine JSON-Liste sein")
        try:
            record = cls(**{**data, "features": tuple(features)})
        except TypeError as exc:
            raise DeployError("Deployment-Record hat ungueltige Feldtypen") from exc
        record.validate()
        return record

    def validate(self) -> None:
        try:
            validate_release_id(self.release_id)
        except (ManifestError, TypeError) as exc:
            raise DeployError("Ungueltige Release-ID im Deployment-State") from exc
        if not isinstance(self.git_sha, str) or not GIT_SHA_RE.fullmatch(self.git_sha):
            raise DeployError("Ungueltiger Git-SHA im Deployment-State")
        for name, value in (
            ("schema_version", self.schema_version),
            ("api_version", self.api_version),
        ):
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", value
            ):
                raise DeployError(f"{name} im Deployment-State ist ungueltig")
        if (
            not isinstance(self.features, tuple)
            or len(self.features) > 256
            or tuple(sorted(self.features, key=str.casefold)) != self.features
            or len({feature.casefold() for feature in self.features})
            != len(self.features)
            or any(
                not isinstance(feature, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", feature)
                for feature in self.features
            )
        ):
            raise DeployError("features im Deployment-State ist nicht kanonisch")
        if not isinstance(self.engine_repository, str) or not REPOSITORY_RE.fullmatch(
            self.engine_repository
        ):
            raise DeployError("Ungueltiges Engine-Repository im Deployment-State")
        if not isinstance(self.engine_digest, str) or not DIGEST_RE.fullmatch(
            self.engine_digest
        ):
            raise DeployError("Ungueltiger Engine-Digest im Deployment-State")
        if self.engine_image != f"{self.engine_repository}@{self.engine_digest}":
            raise DeployError("Engine-Image im Deployment-State ist inkonsistent")
        for name, value in (
            ("container_manifest_sha256", self.container_manifest_sha256),
            ("release_manifest_sha256", self.release_manifest_sha256),
            ("compose_sha256", self.compose_sha256),
            ("env_sha256", self.env_sha256),
        ):
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise DeployError(f"{name} im Deployment-State ist ungueltig")
        if (
            not isinstance(self.compose_artifact, str)
            or PurePosixPath(self.compose_artifact).name != self.compose_artifact
            or "/" in self.compose_artifact
            or "\\" in self.compose_artifact
        ):
            raise DeployError("compose_artifact im Deployment-State ist ungueltig")
        if (
            not isinstance(self.release_path, str)
            or not Path(self.release_path).is_absolute()
        ):
            raise DeployError("release_path im Deployment-State muss absolut sein")
        if (
            isinstance(self.env_generation, bool)
            or not isinstance(self.env_generation, int)
            or self.env_generation < 1
        ):
            raise DeployError("env_generation im Deployment-State ist ungueltig")
        expected_env = (
            PurePosixPath("generations")
            / f"{self.env_generation:020d}-{self.release_id}.env"
        ).as_posix()
        if self.env_file != expected_env:
            raise DeployError("env_file im Deployment-State ist nicht kanonisch")
        expected_compose = (
            PurePosixPath("generations")
            / f"{self.env_generation:020d}-{self.release_id}.compose.yml"
        ).as_posix()
        if self.compose_file != expected_compose:
            raise DeployError("compose_file im Deployment-State ist nicht kanonisch")
        if not isinstance(self.activated_at, str) or not self.activated_at:
            raise DeployError("activated_at im Deployment-State fehlt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "git_sha": self.git_sha,
            "schema_version": self.schema_version,
            "api_version": self.api_version,
            "features": list(self.features),
            "engine_repository": self.engine_repository,
            "engine_digest": self.engine_digest,
            "engine_image": self.engine_image,
            "container_manifest_sha256": self.container_manifest_sha256,
            "release_manifest_sha256": self.release_manifest_sha256,
            "compose_artifact": self.compose_artifact,
            "compose_sha256": self.compose_sha256,
            "compose_file": self.compose_file,
            "release_path": self.release_path,
            "env_generation": self.env_generation,
            "env_file": self.env_file,
            "env_sha256": self.env_sha256,
            "activated_at": self.activated_at,
        }


@dataclass(frozen=True, slots=True)
class DeploymentState:
    generation: int = 0
    current: DeploymentRecord | None = None
    previous: DeploymentRecord | None = None
    updated_at: str | None = None
    format_version: int = STATE_FORMAT_VERSION

    @classmethod
    def from_dict(cls, value: Any) -> "DeploymentState":
        data = _require_keys(
            value,
            {"format_version", "generation", "current", "previous", "updated_at"},
            "Deployment-State",
        )
        state = cls(
            format_version=data["format_version"],
            generation=data["generation"],
            current=(
                DeploymentRecord.from_dict(data["current"])
                if data["current"] is not None
                else None
            ),
            previous=(
                DeploymentRecord.from_dict(data["previous"])
                if data["previous"] is not None
                else None
            ),
            updated_at=data["updated_at"],
        )
        state.validate()
        return state

    def validate(self) -> None:
        if self.format_version != STATE_FORMAT_VERSION:
            raise DeployError("Nicht unterstuetzte Deployment-State-Version")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise DeployError("Ungueltige Deployment-State-Generation")
        if self.updated_at is not None and not isinstance(self.updated_at, str):
            raise DeployError("updated_at im Deployment-State ist ungueltig")
        for record in (self.current, self.previous):
            if record is not None and record.env_generation > self.generation:
                raise DeployError("Env-Generation liegt vor der State-Generation")
        if (
            self.current is not None
            and self.previous is not None
            and self.current.release_id.casefold()
            == self.previous.release_id.casefold()
        ):
            raise DeployError(
                "current und previous duerfen nicht dieselbe Release-ID haben"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "generation": self.generation,
            "current": self.current.to_dict() if self.current else None,
            "previous": self.previous.to_dict() if self.previous else None,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class DeploymentJournal:
    operation: str
    phase: str
    old_state: DeploymentState
    switched_state: DeploymentState
    recovery_state: DeploymentState | None = None
    updated_at: str | None = None
    format_version: int = JOURNAL_FORMAT_VERSION

    @classmethod
    def from_dict(cls, value: Any) -> "DeploymentJournal":
        data = _require_keys(
            value,
            {
                "format_version",
                "operation",
                "phase",
                "old_state",
                "switched_state",
                "recovery_state",
                "updated_at",
            },
            "Deployment-Journal",
        )
        journal = cls(
            format_version=data["format_version"],
            operation=data["operation"],
            phase=data["phase"],
            old_state=DeploymentState.from_dict(data["old_state"]),
            switched_state=DeploymentState.from_dict(data["switched_state"]),
            recovery_state=(
                DeploymentState.from_dict(data["recovery_state"])
                if data["recovery_state"] is not None
                else None
            ),
            updated_at=data["updated_at"],
        )
        journal.validate()
        return journal

    def validate(self) -> None:
        if self.format_version != JOURNAL_FORMAT_VERSION:
            raise DeployError("Nicht unterstuetzte Deployment-Journal-Version")
        if self.operation not in {"deploy", "rollback"}:
            raise DeployError("Ungueltige Operation im Deployment-Journal")
        if self.phase not in {"prepared", "switched", "compensating"}:
            raise DeployError("Ungueltige Phase im Deployment-Journal")
        if not isinstance(self.updated_at, str) or not self.updated_at:
            raise DeployError("updated_at im Deployment-Journal fehlt")
        if self.switched_state.current is None:
            raise DeployError("Deployment-Journal hat keinen Candidate")
        if self.switched_state.generation != self.old_state.generation + 1:
            raise DeployError("Deployment-Journal hat keine fortlaufende Generation")
        if self.phase == "compensating":
            if self.recovery_state is None:
                raise DeployError("Kompensations-Journal hat keinen Recovery-State")
            if self.recovery_state.generation != self.switched_state.generation + 1:
                raise DeployError("Recovery-State hat keine fortlaufende Generation")
            if (
                self.recovery_state.current != self.old_state.current
                or self.recovery_state.previous != self.old_state.previous
            ):
                raise DeployError("Recovery-State widerspricht dem Ausgangszustand")
        elif self.recovery_state is not None:
            raise DeployError("Recovery-State ist nur waehrend Kompensation erlaubt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "operation": self.operation,
            "phase": self.phase,
            "old_state": self.old_state.to_dict(),
            "switched_state": self.switched_state.to_dict(),
            "recovery_state": (
                self.recovery_state.to_dict()
                if self.recovery_state is not None
                else None
            ),
            "updated_at": self.updated_at,
        }


def _dotenv_value(value: str) -> str:
    if not value or any(ord(character) < 32 for character in value):
        raise DeployError("Nicht darstellbarer Wert fuer generierte Compose-Env")
    if re.fullmatch(r"[A-Za-z0-9_./:@\\-]+", value):
        return value
    if "'" in value:
        raise DeployError(
            "Release-Pfad mit Apostroph kann nicht sicher dargestellt werden"
        )
    return f"'{value}'"


def _env_payload(record: DeploymentRecord) -> bytes:
    values = {
        "RISK_LAYER_ENGINE_REPOSITORY": record.engine_repository,
        "RISK_LAYER_ENGINE_DIGEST": record.engine_digest,
        "RISK_LAYER_RELEASE_ID": record.release_id,
        "RISK_LAYER_RELEASE_GIT_SHA": record.git_sha,
        "RISK_LAYER_RELEASE_MANIFEST_SHA256": record.release_manifest_sha256,
        "RISK_LAYER_FESTWISSEN_DIR": record.release_path,
    }
    return (
        "\n".join(f"{key}={_dotenv_value(values[key])}" for key in sorted(values))
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes) -> None:
    _check_directory_chain(path.parent, "Ziel-Elternpfad")
    if _is_link_like(path):
        raise DeployError(f"Zieldatei darf kein Symlink/Reparse-Point sein: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _check_directory_chain(path.parent, "Ziel-Elternpfad")
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _write_immutable(
    path: Path, payload: bytes, *, windows_private_acl: bool = False
) -> None:
    _check_directory_chain(path.parent, "Env-Generationen")
    if _is_link_like(path) or path.exists():
        raise DeployError(f"Env-Generation existiert bereits: {path}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise DeployError(
            f"Env-Generation kann nicht exklusiv angelegt werden: {path}"
        ) from exc
    try:
        if windows_private_acl:
            _harden_windows_private_file_acl(path, "Private Datei")
            opened = os.fstat(descriptor)
            named = os.lstat(path)
            if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                raise DeployError("Private Datei wurde beim DACL-Haerten ausgetauscht")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
        if windows_private_acl:
            written = _secure_file_bytes(
                path,
                maximum=max(len(payload), 1),
                description="Private Datei",
                private=True,
                windows_private_acl=True,
            )
            if not hmac.compare_digest(written, payload):
                raise DeployError("Private Datei wurde nach dem Schreiben ausgetauscht")
        _fsync_directory(path.parent)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _parse_secret_env(path: Path) -> Mapping[str, str]:
    absolute = Path(os.path.abspath(path))
    payload = _secure_file_bytes(
        absolute,
        maximum=MAX_SECRET_ENV_BYTES,
        description="Secret-Env",
        private=True,
        windows_private_acl=True,
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeployError("Secret-Env ist kein gueltiges UTF-8") from exc
    values: dict[str, str] = {}
    folded_keys: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise DeployError(f"Ungueltige Zeile {line_number} im Secret-Env")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        folded_key = key.casefold()
        if not ENV_KEY_RE.fullmatch(key) or folded_key in folded_keys:
            raise DeployError(
                f"Ungueltiger/doppelter Schluessel im Secret-Env: {key!r}"
            )
        if key.casefold().startswith("compose_"):
            raise DeployError(
                f"Secret-Env darf keine Compose-Steuervariable setzen: {key}"
            )
        if not key.startswith("RISK_LAYER_"):
            raise DeployError(
                f"Secret-Env darf nur RISK_LAYER_*-Werte enthalten: {key}"
            )
        if key.casefold() in {managed.casefold() for managed in MANAGED_ENV_KEYS}:
            raise DeployError(
                f"Secret-Env darf verwalteten Wert nicht ueberschreiben: {key}"
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = value.replace(r"\"", '"').replace(r"\\", "\\")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise DeployError(f"Mehrzeiliger Wert im Secret-Env abgelehnt: {key}")
        values[key] = value
        folded_keys.add(folded_key)
    token = values.get("RISK_LAYER_TOKEN", "")
    if len(token.encode("utf-8")) < 32:
        raise DeployError(
            "Secret-Env muss RISK_LAYER_TOKEN mit mindestens 32 Bytes enthalten"
        )
    # MappingProxyType trennt den einmal sicher gelesenen Snapshot vom spaeter
    # erneut oeffnbaren Dateipfad. Docker erhaelt genau diese Werte per Prozess-
    # Environment; die Originaldatei wird nach dieser Stelle nie mehr gelesen.
    return MappingProxyType(values)


def _secret_env_payload(values: Mapping[str, str]) -> bytes:
    lines: list[str] = []
    for key in sorted(values):
        value = values[key]
        if not value:
            encoded = "''"
        else:
            encoded = _dotenv_value(value)
        lines.append(f"{key}={encoded}")
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class SecretSnapshot:
    path: Path
    directory: Path
    values: Mapping[str, str]

    def close(self) -> None:
        if _is_link_like(self.path) or _is_link_like(self.directory):
            raise DeployError("Secret-Snapshot wurde durch einen Link ersetzt")
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise DeployError(
                "Privater Secret-Snapshot konnte nicht entfernt werden"
            ) from exc
        try:
            self.directory.rmdir()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DeployError(
                "Privates Secret-Snapshot-Verzeichnis ist nicht leer"
            ) from exc
        _fsync_directory(self.directory.parent)


def _safe_subprocess_env() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("RISK_LAYER_")
        and not key.upper().startswith("COMPOSE_")
    }
    return environment


def _compose_base(
    docker_bin: str,
    compose_file: Path,
    generated_env: Path,
    secret_env: Path,
) -> list[str]:
    if not docker_bin or "\x00" in docker_bin:
        raise DeployError("Ungueltiges Docker-Programm")
    return [
        docker_bin,
        "compose",
        "--project-name",
        COMPOSE_PROJECT,
        "--project-directory",
        str(compose_file.parent),
        "--env-file",
        str(generated_env),
        "--env-file",
        str(secret_env),
        "-f",
        str(compose_file),
    ]


def _run_command(
    command: Sequence[str], *, environment: Mapping[str, str], timeout: float
) -> None:
    try:
        subprocess.run(
            list(command),
            check=True,
            shell=False,
            env=dict(environment),
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeployError(f"Externer Befehl fehlgeschlagen: {command[0]}") from exc


def _hook_command(raw: str) -> list[str]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, DeployError) as exc:
        raise DeployError(
            "--verify-hook-json muss ein gueltiges JSON-Array sein"
        ) from exc
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 64
        or any(
            not isinstance(item, str) or not item or "\x00" in item for item in value
        )
    ):
        raise DeployError("--verify-hook-json muss 1-64 nichtleere Strings enthalten")
    return value


def _validate_status_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/status"
    ):
        raise DeployError(
            "Status-URL muss eine saubere http(s)://.../v1/status-URL sein"
        )
    if parsed.scheme == "http":
        host = parsed.hostname
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.casefold() == "localhost"
        if not loopback:
            raise DeployError(
                "Unverschluesseltes Status-HTTP ist nur auf Loopback erlaubt"
            )
    return url


def _verify_status(
    url: str,
    record: DeploymentRecord,
    *,
    token: str,
    timeout: float,
) -> None:
    validated = _validate_status_url(url)
    status_request = request.Request(
        validated,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with _STATUS_OPENER.open(status_request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > MAX_STATUS_BYTES:
                raise DeployError("Status-Antwort ueberschreitet das Groessenlimit")
            payload = response.read(MAX_STATUS_BYTES + 1)
    except DeployError:
        raise
    except (OSError, ValueError) as exc:
        raise DeployError("/v1/status konnte nicht sicher gelesen werden") from exc
    if len(payload) > MAX_STATUS_BYTES:
        raise DeployError("Status-Antwort ueberschreitet das Groessenlimit")
    document = _decode_json(payload, "Status-Antwort")
    try:
        release = document["engine_status"]["release"]
    except (KeyError, TypeError) as exc:
        raise DeployError("Status-Antwort enthaelt keine Release-Identitaet") from exc
    expected = {
        "release_id": record.release_id,
        "git_sha": record.git_sha,
        "manifest_sha256": record.release_manifest_sha256,
    }
    if not isinstance(release, dict) or any(
        not hmac.compare_digest(str(release.get(key, "")), value)
        for key, value in expected.items()
    ):
        raise DeployError("/v1/status meldet nicht die erwartete Release-Identitaet")


def _canonical_lock_store() -> Path:
    """Ein pro lokalem Dienstkonto stabiler Lock-Scope fuer das feste Projekt."""
    geteuid = getattr(os, "geteuid", None)
    owner = str(geteuid()) if callable(geteuid) else "user"
    root = Path(tempfile.gettempdir()) / f"risk-layer-container-deploy-{owner}"
    return root / COMPOSE_PROJECT


def _bounded_timeout(
    value: float,
    name: str,
    *,
    maximum: float,
    allow_zero: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeployError(f"{name} muss eine endliche Zahl sein")
    normalized = float(value)
    minimum_valid = normalized >= 0 if allow_zero else normalized > 0
    if not math.isfinite(normalized) or not minimum_valid or normalized > maximum:
        lower = "mindestens 0" if allow_zero else "groesser 0"
        raise DeployError(
            f"{name} muss endlich, {lower} und hoechstens {maximum:g} sein"
        )
    return normalized


class ContainerDeployer:
    def __init__(
        self,
        state_dir: Path,
        *,
        compose_file: Path | None = None,
        secret_env: Path | None = None,
        docker_bin: str = "docker",
        wait_timeout: int = 180,
        command_timeout: float = 360.0,
        verify_timeout: float = 15.0,
        lock_timeout: float = 30.0,
        status_url: str = "http://127.0.0.1:8000/v1/status",
        verify_hook: Sequence[str] | None = None,
        backup_hook: Sequence[str] | None = None,
    ) -> None:
        if (
            isinstance(wait_timeout, bool)
            or not isinstance(wait_timeout, int)
            or wait_timeout < 1
            or wait_timeout > 3600
        ):
            raise DeployError("wait_timeout muss zwischen 1 und 3600 liegen")
        command_timeout = _bounded_timeout(
            command_timeout,
            "command_timeout",
            maximum=MAX_COMMAND_TIMEOUT,
        )
        verify_timeout = _bounded_timeout(
            verify_timeout,
            "verify_timeout",
            maximum=MAX_VERIFY_TIMEOUT,
        )
        lock_timeout = _bounded_timeout(
            lock_timeout,
            "lock_timeout",
            maximum=MAX_LOCK_TIMEOUT,
            allow_zero=True,
        )
        self.state_dir = Path(os.path.abspath(state_dir))
        self.generations_dir = self.state_dir / "generations"
        self.state_path = self.state_dir / "state.json"
        self.journal_path = self.state_dir / JOURNAL_FILENAME
        self.lock_store = _canonical_lock_store()
        self.compose_file = (
            Path(os.path.abspath(compose_file)) if compose_file is not None else None
        )
        self.secret_env = (
            Path(os.path.abspath(secret_env)) if secret_env is not None else None
        )
        self.docker_bin = docker_bin
        self.wait_timeout = wait_timeout
        self.command_timeout = command_timeout
        self.verify_timeout = verify_timeout
        self.lock_timeout = lock_timeout
        self.status_url = status_url
        self.verify_hook = list(verify_hook) if verify_hook is not None else None
        self.backup_hook = list(backup_hook) if backup_hook is not None else None

    def _prepare_state_dir(self) -> None:
        _ensure_directory(self.state_dir, "Deployment-State-Verzeichnis")
        _ensure_directory(self.generations_dir, "Env-Generationen-Verzeichnis")
        _ensure_directory(
            self.lock_store.parent, "Globales Deployment-Lock-Verzeichnis"
        )

    def _lock(self):
        return festwissen_sperre(self.lock_store, timeout=self.lock_timeout)

    def _read_state_payload(self) -> bytes | None:
        return _secure_optional_file_bytes(
            self.state_path,
            maximum=MAX_JSON_BYTES,
            description="Deployment-State",
            private=True,
        )

    @staticmethod
    def _state_from_payload(payload: bytes | None) -> DeploymentState:
        if payload is None:
            return DeploymentState()
        return DeploymentState.from_dict(_decode_json(payload, "Deployment-State"))

    def _read_state(self) -> DeploymentState:
        state = self._state_from_payload(self._read_state_payload())
        for record in (state.current, state.previous):
            if record is not None:
                self._verify_generation(record)
        return state

    def _write_state(self, state: DeploymentState) -> None:
        state.validate()
        payload = (
            json.dumps(
                state.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        _write_atomic(self.state_path, payload)

    def _read_journal_payload(self) -> bytes | None:
        return _secure_optional_file_bytes(
            self.journal_path,
            maximum=MAX_JSON_BYTES,
            description="Deployment-Journal",
            private=True,
        )

    @staticmethod
    def _journal_from_payload(payload: bytes | None) -> DeploymentJournal | None:
        if payload is None:
            return None
        return DeploymentJournal.from_dict(_decode_json(payload, "Deployment-Journal"))

    def _read_journal(self) -> DeploymentJournal | None:
        return self._journal_from_payload(self._read_journal_payload())

    def _write_journal(self, journal: DeploymentJournal) -> None:
        journal.validate()
        payload = (
            json.dumps(
                journal.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        _write_atomic(self.journal_path, payload)

    def _clear_journal(self) -> None:
        _check_directory_chain(self.journal_path.parent, "Journal-Elternpfad")
        if _is_link_like(self.journal_path):
            raise DeployError("Deployment-Journal darf kein Symlink/Reparse-Point sein")
        try:
            self.journal_path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DeployError(
                "Deployment-Journal konnte nicht entfernt werden"
            ) from exc
        _fsync_directory(self.journal_path.parent)

    def _verify_journal_material(self, journal: DeploymentJournal) -> None:
        seen: set[tuple[str, str, int]] = set()
        for state in (
            journal.old_state,
            journal.switched_state,
            journal.recovery_state,
        ):
            if state is None:
                continue
            for record in (state.current, state.previous):
                if record is None:
                    continue
                key = (
                    record.release_id,
                    record.container_manifest_sha256,
                    record.env_generation,
                )
                if key in seen:
                    continue
                seen.add(key)
                self._verify_generation(record)
                self._verify_release(record)

    def _discard_uncommitted_generation(self, journal: DeploymentJournal) -> None:
        if journal.operation != "deploy":
            return
        candidate = journal.switched_state.current
        if candidate is None:
            raise DeployError("Deployment-Journal hat keinen Candidate")
        referenced = {
            (record.env_file, record.compose_file)
            for record in (journal.old_state.current, journal.old_state.previous)
            if record is not None
        }
        if (candidate.env_file, candidate.compose_file) in referenced:
            raise DeployError(
                "Uncommitted Generation ist bereits im Ausgangs-State referenziert"
            )
        for path in (self._env_path(candidate), self._compose_path(candidate)):
            if _is_link_like(path):
                raise DeployError(
                    "Uncommitted Generation wurde durch einen Link ersetzt"
                )
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise DeployError(
                    "Uncommitted Generation konnte nicht entfernt werden"
                ) from exc
        _fsync_directory(self.generations_dir)

    def _env_path(self, record: DeploymentRecord) -> Path:
        path = self.state_dir.joinpath(*PurePosixPath(record.env_file).parts)
        try:
            path.relative_to(self.generations_dir)
        except ValueError as exc:
            raise DeployError(
                "Env-Generation liegt ausserhalb des State-Verzeichnisses"
            ) from exc
        return path

    def _compose_path(self, record: DeploymentRecord) -> Path:
        path = self.state_dir.joinpath(*PurePosixPath(record.compose_file).parts)
        try:
            path.relative_to(self.generations_dir)
        except ValueError as exc:
            raise DeployError(
                "Compose-Generation liegt ausserhalb des State-Verzeichnisses"
            ) from exc
        return path

    def _verify_generation(self, record: DeploymentRecord) -> tuple[Path, Path]:
        env_path = self._env_path(record)
        payload = _secure_file_bytes(
            env_path,
            maximum=64 * 1024,
            description="Env-Generation",
            private=True,
        )
        expected_payload = _env_payload(record)
        if not hmac.compare_digest(_sha256_bytes(payload), record.env_sha256):
            raise DeployError(
                "Env-Generation stimmt nicht mit ihrem State-SHA ueberein"
            )
        if not hmac.compare_digest(payload, expected_payload):
            raise DeployError("Env-Generation enthaelt abweichende oder geheime Werte")
        compose_path = self._compose_path(record)
        if not hmac.compare_digest(
            _sha256_file(compose_path, "Compose-Generation", private=True),
            record.compose_sha256,
        ):
            raise DeployError(
                "Compose-Generation stimmt nicht mit dem Release-Manifest ueberein"
            )
        return env_path, compose_path

    def _candidate_compose_payload(self, record: DeploymentRecord) -> bytes:
        if self.compose_file is None:
            raise DeployError(
                "Manifestgebundene Compose-Datei ist fuer Deploy erforderlich"
            )
        if self.compose_file.name != record.compose_artifact:
            raise DeployError(
                "Compose-Dateiname stimmt nicht mit dem Release-Manifest ueberein"
            )
        payload = _secure_file_bytes(
            self.compose_file,
            maximum=MAX_JSON_BYTES * 64,
            description="Compose-Datei",
        )
        if not hmac.compare_digest(_sha256_bytes(payload), record.compose_sha256):
            raise DeployError(
                "Compose-Datei stimmt nicht mit dem Release-Manifest ueberein"
            )
        return payload

    def _secret_values(self) -> Mapping[str, str]:
        if self.secret_env is None:
            raise DeployError("Separates Secret-Env ist erforderlich")
        try:
            self.secret_env.relative_to(self.state_dir)
        except ValueError:
            pass
        else:
            raise DeployError(
                "Secret-Env muss getrennt von State und Env-Generationen liegen"
            )
        return _parse_secret_env(self.secret_env)

    def _cleanup_stale_secret_snapshots(self) -> None:
        _check_directory_chain(self.lock_store.parent, "Secret-Snapshot-Verzeichnis")
        try:
            entries = list(os.scandir(self.lock_store.parent))
        except OSError as exc:
            raise DeployError("Secret-Snapshot-Verzeichnis ist unlesbar") from exc
        for entry in entries:
            if not entry.name.startswith(".secrets-"):
                continue
            directory = Path(entry.path)
            if _is_link_like(directory) or not entry.is_dir(follow_symlinks=False):
                raise DeployError("Unsicherer verwaister Secret-Snapshot")
            children = list(os.scandir(directory))
            if not children:
                try:
                    directory.rmdir()
                except OSError as exc:
                    raise DeployError(
                        "Leerer verwaister Secret-Snapshot konnte nicht entfernt werden"
                    ) from exc
                continue
            if len(children) != 1 or children[0].name != "secrets.env":
                raise DeployError("Verwaister Secret-Snapshot hat unbekannten Inhalt")
            snapshot = Path(children[0].path)
            _secure_file_bytes(
                snapshot,
                maximum=MAX_SECRET_ENV_BYTES,
                description="Verwaister Secret-Snapshot",
                private=True,
                windows_private_acl=True,
            )
            try:
                snapshot.unlink()
                directory.rmdir()
            except OSError as exc:
                raise DeployError(
                    "Verwaister Secret-Snapshot konnte nicht entfernt werden"
                ) from exc
        _fsync_directory(self.lock_store.parent)

    def _secret_snapshot(self) -> SecretSnapshot:
        self._cleanup_stale_secret_snapshots()
        values = self._secret_values()
        directory = Path(
            tempfile.mkdtemp(dir=self.lock_store.parent, prefix=".secrets-")
        )
        try:
            _check_directory_chain(directory, "Privater Secret-Snapshot")
            if os.name == "nt":
                _harden_windows_private_directory_acl(
                    directory, "Privates Secret-Snapshot-Verzeichnis"
                )
            else:
                os.chmod(directory, 0o700)
            path = directory / "secrets.env"
            _write_immutable(
                path,
                _secret_env_payload(values),
                windows_private_acl=True,
            )
            return SecretSnapshot(path=path, directory=directory, values=values)
        except BaseException:
            try:
                snapshot = directory / "secrets.env"
                snapshot.unlink(missing_ok=True)
                directory.rmdir()
            except OSError:
                pass
            raise

    @contextmanager
    def _secret_snapshot_context(self) -> Iterator[SecretSnapshot]:
        snapshot = self._secret_snapshot()
        try:
            yield snapshot
        finally:
            snapshot.close()

    def _verify_release(self, record: DeploymentRecord) -> ReleaseManifest:
        try:
            manifest = verify_bundle(record.release_path)
        except (IntegrityError, ManifestError, OSError) as exc:
            raise DeployError(
                "Installierter Release-Pfad ist nicht vollstaendig verifiziert"
            ) from exc
        if (
            manifest.release_id != record.release_id
            or manifest.git_sha != record.git_sha
            or manifest.schema_version != record.schema_version
            or manifest.api_version != record.api_version
            or manifest.features != record.features
            or not hmac.compare_digest(
                manifest.fingerprint(), record.release_manifest_sha256
            )
        ):
            raise DeployError(
                "Installierter Release widerspricht der gebundenen Identitaet"
            )
        return manifest

    @staticmethod
    def _api_family(value: str) -> str:
        match = re.match(r"v?(\d+)(?:\.|\Z)", value, flags=re.IGNORECASE)
        return match.group(1) if match is not None else value.casefold()

    def _verify_transition(
        self,
        current: DeploymentRecord | None,
        target: DeploymentRecord,
        *,
        allow_feature_reduction: bool,
    ) -> None:
        if current is None:
            return
        if current.schema_version != target.schema_version:
            raise DeployError(
                "Release-Wechsel ohne Migration abgelehnt: schema_version "
                f"{current.schema_version!r} -> {target.schema_version!r}"
            )
        if self._api_family(current.api_version) != self._api_family(
            target.api_version
        ):
            raise DeployError(
                "Release-Wechsel mit inkompatibler API-Familie abgelehnt: "
                f"{current.api_version!r} -> {target.api_version!r}"
            )
        if not allow_feature_reduction and not set(current.features).issubset(
            target.features
        ):
            removed = sorted(
                set(current.features) - set(target.features), key=str.casefold
            )
            raise DeployError(
                f"Release-Wechsel entfernt deklarierte Features: {removed}"
            )

    def _verify_state_releases(self, state: DeploymentState) -> None:
        for record in (state.current, state.previous):
            if record is not None:
                self._verify_release(record)

    def _commands(
        self, record: DeploymentRecord, secrets: SecretSnapshot
    ) -> list[list[str]]:
        base = _compose_base(
            self.docker_bin,
            self._compose_path(record),
            self._env_path(record),
            secrets.path,
        )
        return [
            [*base, "pull", "engine"],
            [
                *base,
                "up",
                "-d",
                "--wait",
                "--wait-timeout",
                str(self.wait_timeout),
                "engine",
            ],
        ]

    def _activate(
        self,
        record: DeploymentRecord,
        secrets: SecretSnapshot,
    ) -> None:
        docker_environment = _safe_subprocess_env()
        self._verify_generation(record)
        for command in self._commands(record, secrets):
            _run_command(
                command, environment=docker_environment, timeout=self.command_timeout
            )
        if self.verify_hook is not None:
            hook_environment = _safe_subprocess_env()
            hook_environment.update(
                {
                    "RISK_LAYER_EXPECTED_RELEASE_ID": record.release_id,
                    "RISK_LAYER_EXPECTED_GIT_SHA": record.git_sha,
                    "RISK_LAYER_EXPECTED_MANIFEST_SHA256": record.release_manifest_sha256,
                    "RISK_LAYER_EXPECTED_CONTAINER_MANIFEST_SHA256": (
                        record.container_manifest_sha256
                    ),
                    "RISK_LAYER_EXPECTED_ENGINE_IMAGE": record.engine_image,
                    "RISK_LAYER_EXPECTED_RELEASE_PATH": record.release_path,
                }
            )
            _run_command(
                self.verify_hook,
                environment=hook_environment,
                timeout=self.verify_timeout,
            )
        else:
            _verify_status(
                self.status_url,
                record,
                token=secrets.values["RISK_LAYER_TOKEN"],
                timeout=self.verify_timeout,
            )

    def _backup(self, old_state: DeploymentState, candidate: DeploymentRecord) -> None:
        if self.backup_hook is None:
            raise DeployError(
                "Produktives Deploy erfordert einen Festwissen-Pflichtbackup-Hook"
            )
        environment = _safe_subprocess_env()
        environment.update(
            {
                "RISK_LAYER_BACKUP_CURRENT_RELEASE_ID": (
                    old_state.current.release_id
                    if old_state.current is not None
                    else ""
                ),
                "RISK_LAYER_BACKUP_CURRENT_GIT_SHA": (
                    old_state.current.git_sha if old_state.current is not None else ""
                ),
                "RISK_LAYER_BACKUP_CURRENT_MANIFEST_SHA256": (
                    old_state.current.release_manifest_sha256
                    if old_state.current is not None
                    else ""
                ),
                "RISK_LAYER_BACKUP_CANDIDATE_RELEASE_ID": candidate.release_id,
                "RISK_LAYER_BACKUP_CANDIDATE_GIT_SHA": candidate.git_sha,
                "RISK_LAYER_BACKUP_CANDIDATE_MANIFEST_SHA256": (
                    candidate.release_manifest_sha256
                ),
                "RISK_LAYER_BACKUP_CANDIDATE_CONTAINER_MANIFEST_SHA256": (
                    candidate.container_manifest_sha256
                ),
                "RISK_LAYER_BACKUP_CANDIDATE_RELEASE_PATH": candidate.release_path,
            }
        )
        _run_command(
            self.backup_hook,
            environment=environment,
            timeout=self.command_timeout,
        )

    def _deactivate_initial(
        self, record: DeploymentRecord, secrets: SecretSnapshot
    ) -> None:
        base = _compose_base(
            self.docker_bin,
            self._compose_path(record),
            self._env_path(record),
            secrets.path,
        )
        _run_command(
            [*base, "down", "--remove-orphans"],
            environment=_safe_subprocess_env(),
            timeout=self.command_timeout,
        )

    def _compensate(
        self,
        journal: DeploymentJournal,
        failed_record: DeploymentRecord,
        secrets: SecretSnapshot,
    ) -> None:
        old_state = journal.old_state
        switched = journal.switched_state
        compensated = replace(
            old_state,
            generation=switched.generation + 1,
            updated_at=_utc_now(),
        )
        recovering = replace(
            journal,
            phase="compensating",
            recovery_state=compensated,
            updated_at=_utc_now(),
        )
        self._write_journal(recovering)
        self._write_state(compensated)
        if old_state.current is None:
            self._deactivate_initial(failed_record, secrets)
        else:
            self._verify_release(old_state.current)
            self._activate(old_state.current, secrets)
        self._clear_journal()

    def _reconcile(self, secrets: SecretSnapshot) -> DeploymentState | None:
        journal = self._read_journal()
        if journal is None:
            return None
        self._verify_journal_material(journal)
        state = self._read_state()
        self._verify_state_releases(state)
        failed_record = journal.switched_state.current
        if failed_record is None:  # durch Journal.validate bereits ausgeschlossen
            raise DeployError("Deployment-Journal hat keinen Candidate")

        if journal.phase == "compensating":
            recovery = journal.recovery_state
            if recovery is None:
                raise DeployError("Kompensations-Journal hat keinen Recovery-State")
            if state not in {journal.switched_state, recovery}:
                raise DeployError("Deployment-State widerspricht dem Recovery-Journal")
            if state != recovery:
                self._write_state(recovery)
            if journal.old_state.current is None:
                self._deactivate_initial(failed_record, secrets)
            else:
                self._activate(journal.old_state.current, secrets)
            self._clear_journal()
            return recovery

        if state == journal.old_state:
            # In phase=prepared wurde der externe Stack noch nicht beruehrt.
            # Bei phase=switched ist ein manuell/teilweise kompensierter Zustand
            # moeglich; dann stellen wir die alte Runtime ausdruecklich sicher.
            if journal.phase == "prepared":
                self._discard_uncommitted_generation(journal)
            else:
                if journal.old_state.current is None:
                    self._deactivate_initial(failed_record, secrets)
                else:
                    self._activate(journal.old_state.current, secrets)
            self._clear_journal()
            return state

        if state != journal.switched_state:
            raise DeployError("Deployment-State widerspricht dem offenen Journal")
        try:
            self._activate(failed_record, secrets)
        except BaseException as activation_error:
            try:
                self._compensate(journal, failed_record, secrets)
            except BaseException as rollback_error:
                raise DeployError(
                    "Journal-Reconciliation und automatisches Rollback sind fehlgeschlagen",
                    rolled_back=False,
                ) from rollback_error
            raise DeployError(
                "Unvollstaendige Aktivierung wurde bei Reconciliation zurueckgerollt",
                rolled_back=True,
            ) from activation_error
        self._clear_journal()
        return journal.switched_state

    def _switch_and_activate(
        self,
        *,
        operation: str,
        old_state: DeploymentState,
        switched: DeploymentState,
        candidate: DeploymentRecord,
        secrets: SecretSnapshot,
        failure_message: str,
        rollback_failure_message: str,
    ) -> None:
        journal = DeploymentJournal(
            operation=operation,
            phase="prepared",
            old_state=old_state,
            switched_state=switched,
            updated_at=_utc_now(),
        )
        self._write_journal(journal)
        try:
            self._write_state(switched)
            journal = replace(journal, phase="switched", updated_at=_utc_now())
            self._write_journal(journal)
            self._activate(candidate, secrets)
        except BaseException as activation_error:
            try:
                # Auch ein Fehler nach dem atomaren State-Replace wird anhand
                # des persistenten Journals deterministisch eingeordnet.
                current_journal = self._read_journal() or journal
                current_state = self._read_state()
                if current_state == old_state and current_journal.phase == "prepared":
                    self._discard_uncommitted_generation(current_journal)
                    self._clear_journal()
                    raise activation_error
                self._compensate(current_journal, candidate, secrets)
            except BaseException as rollback_error:
                if rollback_error is activation_error:
                    raise DeployError(
                        "State-Switch ist vor der externen Aktivierung fehlgeschlagen",
                        rolled_back=True,
                    ) from activation_error
                raise DeployError(
                    rollback_failure_message,
                    rolled_back=False,
                ) from rollback_error
            raise DeployError(failure_message, rolled_back=True) from activation_error
        self._clear_journal()

    def _candidate(
        self,
        identity: ManifestIdentity,
        release_path: Path,
        env_generation: int,
    ) -> DeploymentRecord:
        absolute_release = Path(os.path.abspath(release_path))
        try:
            manifest = verify_bundle(absolute_release)
        except (IntegrityError, ManifestError, OSError) as exc:
            raise DeployError(
                "Installierter Release-Pfad ist nicht vollstaendig verifiziert"
            ) from exc
        if (
            manifest.release_id != identity.release_id
            or manifest.git_sha != identity.git_sha
            or not hmac.compare_digest(
                manifest.fingerprint(), identity.release_manifest_sha256
            )
        ):
            raise DeployError(
                "Installierter Release widerspricht der gebundenen Identitaet"
            )
        record = DeploymentRecord(
            release_id=identity.release_id,
            git_sha=identity.git_sha,
            schema_version=manifest.schema_version,
            api_version=manifest.api_version,
            features=manifest.features,
            engine_repository=identity.engine_repository,
            engine_digest=identity.engine_digest,
            engine_image=identity.engine_image,
            container_manifest_sha256=identity.container_manifest_sha256,
            release_manifest_sha256=identity.release_manifest_sha256,
            compose_artifact=identity.compose_artifact,
            compose_sha256=identity.compose_sha256,
            compose_file=(
                PurePosixPath("generations")
                / f"{env_generation:020d}-{identity.release_id}.compose.yml"
            ).as_posix(),
            release_path=str(absolute_release),
            env_generation=env_generation,
            env_file=(
                PurePosixPath("generations")
                / f"{env_generation:020d}-{identity.release_id}.env"
            ).as_posix(),
            env_sha256="0" * 64,
            activated_at=_utc_now(),
        )
        record = replace(record, env_sha256=_sha256_bytes(_env_payload(record)))
        record.validate()
        return record

    def deploy(
        self,
        manifest: Path,
        expected_manifest_sha256: str,
        release_path: Path,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        identity = _load_container_manifest(manifest, expected_manifest_sha256)
        self._prepare_state_dir()
        try:
            with self._lock(), self._secret_snapshot_context() as secrets:
                if self._read_journal() is not None:
                    if dry_run:
                        raise DeployError(
                            "Offenes Deployment-Journal muss vor einem Dry-run reconciled werden"
                        )
                    self._reconcile(secrets)
                old_state = self._read_state()
                self._verify_state_releases(old_state)
                candidate = self._candidate(
                    identity, release_path, old_state.generation + 1
                )
                compose_payload = self._candidate_compose_payload(candidate)
                if self.backup_hook is None:
                    raise DeployError(
                        "Produktives Deploy erfordert einen Festwissen-Pflichtbackup-Hook"
                    )
                if old_state.current is not None and (
                    old_state.current.release_id.casefold()
                    == candidate.release_id.casefold()
                ):
                    if old_state.current != candidate:
                        comparable_old = replace(
                            old_state.current,
                            env_generation=candidate.env_generation,
                            env_file=candidate.env_file,
                            env_sha256=candidate.env_sha256,
                            compose_file=candidate.compose_file,
                            activated_at=candidate.activated_at,
                        )
                        if comparable_old != candidate:
                            raise DeployError(
                                "Release-ID ist bereits mit anderer Identitaet installiert"
                            )
                    if dry_run:
                        return self._dry_run(candidate, secrets, "deploy-idempotent")
                    self._activate(old_state.current, secrets)
                    return {"ok": True, "changed": False, "state": old_state.to_dict()}
                if old_state.previous is not None and (
                    old_state.previous.release_id.casefold()
                    == candidate.release_id.casefold()
                    and old_state.previous.container_manifest_sha256
                    != candidate.container_manifest_sha256
                ):
                    raise DeployError(
                        "Previous-Release-ID ist mit anderer Identitaet gebunden"
                    )
                self._verify_transition(
                    old_state.current,
                    candidate,
                    allow_feature_reduction=False,
                )
                if dry_run:
                    return self._dry_run(candidate, secrets, "deploy")

                self._backup(old_state, candidate)
                payload = _env_payload(candidate)
                _write_immutable(self._env_path(candidate), payload)
                try:
                    _write_immutable(self._compose_path(candidate), compose_payload)
                except BaseException:
                    try:
                        self._env_path(candidate).unlink()
                    except OSError:
                        pass
                    raise
                switched = DeploymentState(
                    generation=old_state.generation + 1,
                    current=candidate,
                    previous=old_state.current,
                    updated_at=_utc_now(),
                )
                self._switch_and_activate(
                    operation="deploy",
                    old_state=old_state,
                    switched=switched,
                    candidate=candidate,
                    secrets=secrets,
                    failure_message=(
                        "Aktivierung fehlgeschlagen und wurde vollstaendig zurueckgerollt"
                    ),
                    rollback_failure_message=(
                        "Aktivierung fehlgeschlagen; automatisches Rollback ist ebenfalls fehlgeschlagen"
                    ),
                )
                return {"ok": True, "changed": True, "state": switched.to_dict()}
        except DateisperrenFehler as exc:
            raise DeployError("Deployment-Lock konnte nicht erworben werden") from exc

    def rollback(self, *, dry_run: bool = False) -> dict[str, Any]:
        self._prepare_state_dir()
        try:
            with self._lock(), self._secret_snapshot_context() as secrets:
                if self._read_journal() is not None:
                    if dry_run:
                        raise DeployError(
                            "Offenes Deployment-Journal muss vor einem Dry-run reconciled werden"
                        )
                    self._reconcile(secrets)
                old_state = self._read_state()
                if old_state.current is None or old_state.previous is None:
                    raise DeployError(
                        "Kein current/previous-Paar fuer Rollback vorhanden"
                    )
                target = old_state.previous
                self._verify_release(target)
                self._verify_generation(target)
                self._verify_transition(
                    old_state.current,
                    target,
                    allow_feature_reduction=True,
                )
                if self.backup_hook is None:
                    raise DeployError(
                        "Produktives Rollback erfordert einen Festwissen-Pflichtbackup-Hook"
                    )
                if dry_run:
                    return self._dry_run(target, secrets, "rollback")
                self._backup(old_state, target)
                switched = DeploymentState(
                    generation=old_state.generation + 1,
                    current=target,
                    previous=old_state.current,
                    updated_at=_utc_now(),
                )
                self._switch_and_activate(
                    operation="rollback",
                    old_state=old_state,
                    switched=switched,
                    candidate=target,
                    secrets=secrets,
                    failure_message=(
                        "Rollback-Aktivierung fehlgeschlagen; Ausgangszustand wurde reaktiviert"
                    ),
                    rollback_failure_message=(
                        "Rollback-Aktivierung fehlgeschlagen; Kompensation ist ebenfalls fehlgeschlagen"
                    ),
                )
                return {"ok": True, "changed": True, "state": switched.to_dict()}
        except DateisperrenFehler as exc:
            raise DeployError("Deployment-Lock konnte nicht erworben werden") from exc

    def _dry_run(
        self,
        record: DeploymentRecord,
        secrets: SecretSnapshot,
        operation: str,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "dry_run": True,
            "operation": operation,
            "candidate": record.to_dict(),
            "commands": self._commands(record, secrets),
            "backup_hook": {
                "configured": self.backup_hook is not None,
                "executable": self.backup_hook[0] if self.backup_hook else None,
                "argument_count": len(self.backup_hook) if self.backup_hook else 0,
            },
            "verify": (
                {
                    "external_hook": True,
                    "executable": self.verify_hook[0],
                    "argument_count": len(self.verify_hook),
                }
                if self.verify_hook is not None
                else {"status_url": _validate_status_url(self.status_url)}
            ),
        }

    @staticmethod
    def _status_records(
        state: DeploymentState, journal: DeploymentJournal | None
    ) -> tuple[DeploymentRecord, ...]:
        states: list[DeploymentState] = [state]
        if journal is not None:
            states.extend((journal.old_state, journal.switched_state))
            if journal.recovery_state is not None:
                states.append(journal.recovery_state)
        records: list[DeploymentRecord] = []
        seen: set[DeploymentRecord] = set()
        for candidate_state in states:
            for record in (candidate_state.current, candidate_state.previous):
                if record is not None and record not in seen:
                    seen.add(record)
                    records.append(record)
        return tuple(records)

    def _confirm_status_control(
        self, state_payload: bytes | None, journal_payload: bytes | None
    ) -> None:
        confirmed_state = self._read_state_payload()
        confirmed_journal = self._read_journal_payload()
        if confirmed_state != state_payload or confirmed_journal != journal_payload:
            raise DeployError(
                "Deployment-State/Journal wurde waehrend des read-only Status geaendert"
            )

    def status(self) -> dict[str, Any]:
        _check_directory_chain(self.state_dir, "Deployment-State-Verzeichnis")
        state_payload = self._read_state_payload()
        journal_payload = self._read_journal_payload()
        state = self._state_from_payload(state_payload)
        journal = self._journal_from_payload(journal_payload)
        records = self._status_records(state, journal)
        for record in records:
            self._verify_generation(record)
            self._verify_release(record)
        if journal is not None:
            permitted = [journal.old_state, journal.switched_state]
            if journal.recovery_state is not None:
                permitted.append(journal.recovery_state)
            if state not in permitted:
                raise DeployError("Deployment-State widerspricht dem offenen Journal")

        # Ohne einen schreibenden Lock bleibt Status strikt read-only. Zwei
        # Kontrollpunkte machen konkurrierende atomare State-/Journal-Wechsel
        # sichtbar; die zweite Materialrunde bindet auch Generationen an den
        # bestaetigten Control-Snapshot. Ein beobachtetes Race schlaegt bewusst
        # kontrolliert fehl, statt einen gemischten Zustand zu melden.
        self._confirm_status_control(state_payload, journal_payload)
        for record in records:
            self._verify_generation(record)
        self._confirm_status_control(state_payload, journal_payload)
        return {
            "ok": journal is None,
            "state": state.to_dict(),
            "env_verified": True,
            "recovery_required": journal is not None,
            "journal_phase": journal.phase if journal is not None else None,
        }


def _add_runtime_arguments(
    parser: argparse.ArgumentParser, *, compose_required: bool
) -> None:
    parser.add_argument(
        "--compose-file",
        type=Path,
        required=compose_required,
        help="manifestgebundene Quelldatei (nur fuer deploy erforderlich)",
    )
    parser.add_argument("--secrets-env", type=Path, required=True)
    parser.add_argument(
        "--backup-hook-json",
        required=True,
        help="Pflichtbackup-Hook als JSON-Array; wird vor dem Switch ohne Shell ausgefuehrt",
    )
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--wait-timeout", type=int, default=180)
    parser.add_argument("--command-timeout", type=float, default=360.0)
    parser.add_argument("--verify-timeout", type=float, default=15.0)
    parser.add_argument("--status-url", default="http://127.0.0.1:8000/v1/status")
    parser.add_argument(
        "--verify-hook-json",
        help="externer Verify-Hook als JSON-Array; wird ohne Shell ausgefuehrt",
    )
    parser.add_argument("--dry-run", action="store_true")


def _deployer(args: argparse.Namespace, *, runtime: bool) -> ContainerDeployer:
    hook = (
        _hook_command(args.verify_hook_json)
        if runtime and args.verify_hook_json
        else None
    )
    backup_hook = (
        _hook_command(args.backup_hook_json)
        if runtime and getattr(args, "backup_hook_json", None)
        else None
    )
    return ContainerDeployer(
        args.state_dir,
        compose_file=getattr(args, "compose_file", None) if runtime else None,
        secret_env=args.secrets_env if runtime else None,
        docker_bin=args.docker_bin if runtime else "docker",
        wait_timeout=args.wait_timeout if runtime else 180,
        command_timeout=args.command_timeout if runtime else 360.0,
        verify_timeout=args.verify_timeout if runtime else 15.0,
        lock_timeout=args.lock_timeout,
        status_url=(args.status_url if runtime else "http://127.0.0.1:8000/v1/status"),
        verify_hook=hook,
        backup_hook=backup_hook,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    deploy = sub.add_parser("deploy", help="verifiziertes Container-Release aktivieren")
    deploy.add_argument("manifest", type=Path)
    deploy.add_argument("--manifest-sha256", required=True)
    deploy.add_argument("--release-path", type=Path, required=True)
    deploy.add_argument("--state-dir", type=Path, required=True)
    deploy.add_argument("--lock-timeout", type=float, default=30.0)
    _add_runtime_arguments(deploy, compose_required=True)

    rollback = sub.add_parser(
        "rollback", help="current und previous transaktional tauschen"
    )
    rollback.add_argument("--state-dir", type=Path, required=True)
    rollback.add_argument("--lock-timeout", type=float, default=30.0)
    _add_runtime_arguments(rollback, compose_required=False)

    status = sub.add_parser("status", help="State und Env-Integritaet anzeigen")
    status.add_argument("--state-dir", type=Path, required=True)
    status.add_argument("--lock-timeout", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "deploy":
            result = _deployer(args, runtime=True).deploy(
                args.manifest,
                args.manifest_sha256,
                args.release_path,
                dry_run=args.dry_run,
            )
        elif args.command == "rollback":
            result = _deployer(args, runtime=True).rollback(dry_run=args.dry_run)
        else:
            result = _deployer(args, runtime=False).status()
    except DeployError as exc:
        suffix = (
            f" (rolled_back={str(exc.rolled_back).lower()})"
            if exc.rolled_back is not None
            else ""
        )
        print(f"Container-Deployment abgelehnt: {exc}{suffix}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
