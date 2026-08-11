"""Betriebs-CLI fuer Health, Backup und transaktionale Releases.

``risk-layer`` bleibt die fachliche Engine-CLI. ``risk-layer-ops`` ist die
technische Delegate-/Expert-CLI hinter der profilbasierten ``signal``-
Fassade: Sie schaltet Release-Generationen atomar und kann einen stabilen
Dienst auf ``current`` starten, ohne Code aus einem Git-Checkout zu importieren.
"""

from __future__ import annotations

import argparse
import hmac
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .release_lifecycle import (
    ActivationError,
    ReleaseError,
    ReleaseManager,
    ReleaseRecord,
    ReleaseState,
    verify_bundle,
)
from .release_manifest import ArtifactDigest, ManifestError, ReleaseManifest
from .sicherung import (
    SicherungsFehler,
    bereinige_sicherungen,
    pruefe_backup_ziel,
    pruefe_sicherung,
    sichere_festwissen,
    stelle_wieder_her_mit_sicherung,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TOKEN_BYTES = 4096
_RUNTIME_MARKER = ".risk-layer-runtime.json"
_RUNTIME_MARKER_FORMAT = 1
_MAX_RUNTIME_MARKER_BYTES = 16 * 1024
_MAX_RUNTIME_LOCK_BYTES = 2 * 1024 * 1024
MAX_OPERATION_TIMEOUT_SECONDS = 3600.0
_PINNED_REQUIREMENT_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9,._-]+\])?"
    r"==[A-Za-z0-9][A-Za-z0-9.!+_-]*\Z"
)
_HASH_OPTION_RE = re.compile(r"--hash=sha256:[0-9a-f]{64}\Z")


class OpsFehler(RuntimeError):
    """Sicher behandelbarer Bedienungs- oder Betriebsfehler."""


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
        raise OpsFehler(
            f"{name} muss endlich und > 0 bis {MAX_OPERATION_TIMEOUT_SECONDS:g}s sein"
        )
    return normalized


def _nonnegative_timeout(value: Any, name: str) -> float:
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        normalized = math.nan
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(normalized)
        or normalized < 0
        or normalized > MAX_OPERATION_TIMEOUT_SECONDS
    ):
        raise OpsFehler(
            f"{name} muss endlich und >= 0 bis {MAX_OPERATION_TIMEOUT_SECONDS:g}s sein"
        )
    return normalized


def _timeout_argument(raw: str) -> float:
    try:
        value = float(raw)
        return _positive_timeout(value, "Timeout")
    except (ValueError, OpsFehler) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _lock_timeout_argument(raw: str) -> float:
    try:
        value = float(raw)
        return _nonnegative_timeout(value, "Lock-Timeout")
    except (ValueError, OpsFehler) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class _ReleaseRuntime:
    """Eine vollstaendig gebaute und an ein Release-Manifest gebundene Venv."""

    path: Path
    python: Path
    created: bool = False


@dataclass(frozen=True, slots=True)
class _RuntimeQuellen:
    lock: ArtifactDigest
    wheels: tuple[ArtifactDigest, ...]
    app_wheel: ArtifactDigest


def _runtime_platform_tag() -> str:
    """ABI-sicherer, zwischen Builder und Host reproduzierbarer Wheelhouse-Tag."""
    plattform = sysconfig.get_platform().replace("-", "_").replace(".", "_").lower()
    if not re.fullmatch(r"[a-z0-9_]{1,96}", plattform):
        raise OpsFehler(f"Nicht unterstuetzte Host-Plattformkennung: {plattform!r}")
    return f"{plattform}-cp{sys.version_info.major}{sys.version_info.minor}"


def _runtime_name(release_id: str, manifest_sha256: str) -> str:
    if not _SHA256_RE.fullmatch(manifest_sha256):
        raise OpsFehler("Runtime-Manifestbindung ist kein lowercase SHA-256")
    # release_id wurde bereits vom ReleaseManifest streng als ein Pfadsegment
    # validiert. Der Hash-Praefix verhindert Vermischung bei erneut verwendeter ID.
    return f"{release_id}-{manifest_sha256[:16]}"


def _runtime_python_relativ() -> Path:
    return Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")


def _ist_link_oder_reparse(pfad: Path) -> bool:
    try:
        info = os.lstat(pfad)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or _ist_reparse(info)


def _pruefe_verzeichniskette(pfad: Path, beschreibung: str) -> None:
    absolut = Path(os.path.abspath(pfad))
    for komponent in reversed((absolut, *absolut.parents)):
        try:
            info = os.lstat(komponent)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise OpsFehler(
                f"{beschreibung} kann nicht sicher geprueft werden: {komponent}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or _ist_reparse(info):
            raise OpsFehler(
                f"Symlink-/Reparse-Komponente in {beschreibung}: {komponent}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise OpsFehler(
                f"Nicht-Verzeichnis-Komponente in {beschreibung}: {komponent}"
            )


def _runtime_root(home: str | Path, *, erstellen: bool) -> Path:
    wurzel = Path(home).absolute() / "runtimes"
    _pruefe_verzeichniskette(wurzel, "Runtime-Verzeichnis")
    if not wurzel.exists():
        if not erstellen:
            return wurzel
        try:
            wurzel.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        except OSError as exc:
            raise OpsFehler(f"Runtime-Verzeichnis kann nicht erstellt werden: {wurzel}") from exc
    _pruefe_verzeichniskette(wurzel, "Runtime-Verzeichnis")
    if _ist_link_oder_reparse(wurzel) or not wurzel.is_dir():
        raise OpsFehler(f"Unsicheres Runtime-Verzeichnis: {wurzel}")
    if os.name != "nt":
        os.chmod(wurzel, 0o700)
    return wurzel


def _sichere_datei_lesen(pfad: Path, limit: int, beschreibung: str) -> bytes:
    try:
        vorher = os.lstat(pfad)
    except OSError as exc:
        raise OpsFehler(f"{beschreibung} nicht lesbar: {pfad}") from exc
    if (
        not stat.S_ISREG(vorher.st_mode)
        or stat.S_ISLNK(vorher.st_mode)
        or _ist_reparse(vorher)
    ):
        raise OpsFehler(f"{beschreibung} ist keine sichere regulaere Datei: {pfad}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(pfad, flags)
    except OSError as exc:
        raise OpsFehler(f"{beschreibung} kann nicht sicher geoeffnet werden: {pfad}") from exc
    try:
        info = os.fstat(descriptor)
        nachher = os.lstat(pfad)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_ISLNK(nachher.st_mode)
            or _ist_reparse(nachher)
            or (info.st_dev, info.st_ino) != (nachher.st_dev, nachher.st_ino)
            or info.st_size > limit
        ):
            raise OpsFehler(f"{beschreibung} wurde ausgetauscht oder ist unsicher: {pfad}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            daten = handle.read(limit + 1)
        if len(daten) > limit:
            raise OpsFehler(f"{beschreibung} ueberschreitet {limit} Bytes")
        return daten
    finally:
        os.close(descriptor)


def _json_ohne_duplikate(paare: list[tuple[str, Any]]) -> dict[str, Any]:
    ergebnis: dict[str, Any] = {}
    for schluessel, wert in paare:
        if schluessel in ergebnis:
            raise OpsFehler(f"Doppelter Schluessel im Runtime-Marker: {schluessel}")
        ergebnis[schluessel] = wert
    return ergebnis


def _runtime_marker_payload(record: ReleaseRecord, python_sha256: str, python_size: int) -> dict[str, Any]:
    return {
        "format_version": _RUNTIME_MARKER_FORMAT,
        "release_id": record.manifest.release_id,
        "manifest_sha256": record.manifest.fingerprint(),
        "platform_tag": _runtime_platform_tag(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_path": _runtime_python_relativ().as_posix(),
        "python_sha256": python_sha256,
        "python_size": python_size,
    }


def _datei_hash_und_groesse(pfad: Path, beschreibung: str) -> tuple[str, int]:
    daten = _sichere_datei_lesen(pfad, 1024 * 1024 * 1024, beschreibung)
    return hashlib.sha256(daten).hexdigest(), len(daten)


def _runtime_pruefen(home: str | Path, record: ReleaseRecord) -> _ReleaseRuntime:
    fingerprint = record.manifest.fingerprint()
    wurzel = _runtime_root(home, erstellen=False)
    ziel = wurzel / _runtime_name(record.manifest.release_id, fingerprint)
    if not wurzel.exists() or not ziel.exists():
        raise OpsFehler(
            f"Release-Runtime fehlt fuer {record.manifest.release_id}: {ziel}"
        )
    _pruefe_verzeichniskette(ziel, "Release-Runtime")
    if _ist_link_oder_reparse(ziel) or not ziel.is_dir():
        raise OpsFehler(f"Release-Runtime ist unsicher: {ziel}")
    marker_pfad = ziel / _RUNTIME_MARKER
    roh = _sichere_datei_lesen(
        marker_pfad, _MAX_RUNTIME_MARKER_BYTES, "Runtime-Marker"
    )
    try:
        marker = json.loads(
            roh.decode("utf-8"),
            object_pairs_hook=_json_ohne_duplikate,
            parse_constant=lambda wert: (_ for _ in ()).throw(
                OpsFehler(f"Nichtstandard-JSON im Runtime-Marker: {wert}")
            ),
        )
    except OpsFehler:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OpsFehler(f"Ungueltiger Runtime-Marker: {marker_pfad}") from exc
    erwartete_felder = {
        "format_version", "release_id", "manifest_sha256", "platform_tag",
        "python_version", "python_path", "python_sha256", "python_size",
    }
    if not isinstance(marker, dict) or set(marker) != erwartete_felder:
        raise OpsFehler(f"Runtime-Marker hat unerwartete Felder: {marker_pfad}")
    python_hash = marker.get("python_sha256")
    python_size = marker.get("python_size")
    if (
        not isinstance(python_hash, str)
        or not _SHA256_RE.fullmatch(python_hash)
        or isinstance(python_size, bool)
        or not isinstance(python_size, int)
        or python_size <= 0
    ):
        raise OpsFehler(f"Runtime-Marker bindet keinen gueltigen Interpreter: {marker_pfad}")
    erwartet = _runtime_marker_payload(record, python_hash, python_size)
    if marker != erwartet:
        raise OpsFehler(
            "Release-Runtime passt nicht zu Manifest, Host-Plattform oder Python-Minor"
        )
    python = ziel / _runtime_python_relativ()
    ist_hash, ist_groesse = _datei_hash_und_groesse(python, "Runtime-Interpreter")
    if not hmac.compare_digest(ist_hash, python_hash) or ist_groesse != python_size:
        raise OpsFehler(f"Runtime-Interpreter wurde veraendert: {python}")
    return _ReleaseRuntime(path=ziel, python=python)


def _runtime_entfernen(
    home: str | Path, release_id: str, manifest_sha256: str
) -> bool:
    """Entfernt nur den exakt markergebundenen Runtime-Baum eines Releases."""
    wurzel = _runtime_root(home, erstellen=False)
    if not wurzel.exists():
        return False
    ziel = wurzel / _runtime_name(release_id, manifest_sha256)
    if not ziel.exists() and not _ist_link_oder_reparse(ziel):
        return False
    _pruefe_verzeichniskette(ziel, "zu entfernende Release-Runtime")
    if _ist_link_oder_reparse(ziel) or not ziel.is_dir():
        raise OpsFehler(f"Unsichere Release-Runtime wird nicht entfernt: {ziel}")
    marker = _sichere_datei_lesen(
        ziel / _RUNTIME_MARKER, _MAX_RUNTIME_MARKER_BYTES, "Runtime-Marker"
    )
    try:
        dokument = json.loads(marker.decode("utf-8"), object_pairs_hook=_json_ohne_duplikate)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, OpsFehler) as exc:
        raise OpsFehler(f"Release-Runtime ohne gueltigen Marker wird nicht entfernt: {ziel}") from exc
    if (
        not isinstance(dokument, dict)
        or dokument.get("release_id") != release_id
        or dokument.get("manifest_sha256") != manifest_sha256
    ):
        raise OpsFehler(f"Fremde oder ungebundene Runtime wird nicht entfernt: {ziel}")
    shutil.rmtree(ziel)
    return True


class _KeineRedirects(HTTPRedirectHandler):
    """Health-Checks folgen nie Redirects und geben so kein Bearer-Token weiter."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_HTTP_OPENER = build_opener(_KeineRedirects())


def _utc_jetzt() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _json_ausgabe(objekt: Any) -> None:
    print(json.dumps(objekt, ensure_ascii=False, sort_keys=True, indent=2))


def _ist_reparse(info: os.stat_result) -> bool:
    attribut = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attribut and getattr(info, "st_file_attributes", 0) & attribut)


def _token_lesen(token_datei: str | None) -> str | None:
    """Liest ein Bearer-Token begrenzt und ohne Link-Folgen.

    Ohne ``--token-file`` gilt die Prozessumgebung. Ein fehlendes Token bleibt
    fuer lokale Entwicklungsserver erlaubt; ein vorhandener ungueltiger Wert
    wird nie still ignoriert.
    """
    if token_datei is None:
        token = os.environ.get("RISK_LAYER_TOKEN")
        if token is None:
            return None
    else:
        pfad = Path(token_datei)
        try:
            vorher = os.lstat(pfad)
        except OSError as exc:
            raise OpsFehler(f"Token-Datei nicht lesbar: {pfad}: {exc}") from exc
        if stat.S_ISLNK(vorher.st_mode) or _ist_reparse(vorher):
            raise OpsFehler(f"Token-Datei darf kein Link/Reparse-Point sein: {pfad}")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(pfad, flags)
        except OSError as exc:
            raise OpsFehler(f"Token-Datei kann nicht sicher geoeffnet werden: {pfad}") from exc
        try:
            info = os.fstat(descriptor)
            nachher = os.lstat(pfad)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(nachher.st_mode)
                or _ist_reparse(nachher)
                or (info.st_dev, info.st_ino) != (nachher.st_dev, nachher.st_ino)
            ):
                raise OpsFehler(f"Token-Datei wurde ausgetauscht oder ist unregelmaessig: {pfad}")
            if info.st_size > _MAX_TOKEN_BYTES:
                raise OpsFehler(f"Token-Datei ist groesser als {_MAX_TOKEN_BYTES} Bytes")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                roh = handle.read(_MAX_TOKEN_BYTES + 1)
        finally:
            os.close(descriptor)
        try:
            token = roh.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise OpsFehler("Token-Datei ist nicht gueltiges UTF-8") from exc

    if len(token.encode("utf-8")) < 32 or any(zeichen.isspace() for zeichen in token):
        raise OpsFehler("Bearer-Token muss mindestens 32 UTF-8-Bytes und ohne Whitespace sein")
    return token


def _basis_url(url: str, *, unsicheres_http: bool = False) -> str:
    wert = url.rstrip("/")
    parsed = urlparse(wert)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise OpsFehler(f"Ungueltige Health-URL: {url!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise OpsFehler("Health-URL darf keine Credentials, Query oder Fragmente enthalten")
    host = parsed.hostname.casefold()
    loopback = host == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if parsed.scheme != "https" and not loopback and not unsicheres_http:
        raise OpsFehler(
            "Bearer-Token wird nur ueber HTTPS oder Loopback gesendet; "
            "bewusste Ausnahme: --allow-insecure-http"
        )
    return wert


def _hole_json(basis: str, pfad: str, token: str | None, timeout: float) -> dict[str, Any]:
    timeout = _positive_timeout(timeout, "HTTP-Timeout")
    headers = {"Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"{basis}{pfad}", headers=headers, method="GET")
    try:
        with _HTTP_OPENER.open(request, timeout=timeout) as antwort:
            if antwort.status != 200:
                raise OpsFehler(f"{pfad} antwortete mit HTTP {antwort.status}")
            roh = antwort.read(2 * 1024 * 1024 + 1)
    except HTTPError as exc:
        raise OpsFehler(f"{pfad} antwortete mit HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OpsFehler(f"{pfad} nicht erreichbar: {exc}") from exc
    if len(roh) > 2 * 1024 * 1024:
        raise OpsFehler(f"{pfad}-Antwort ueberschreitet 2 MiB")
    try:
        dokument = json.loads(roh.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OpsFehler(f"{pfad} lieferte kein gueltiges UTF-8-JSON") from exc
    if not isinstance(dokument, dict):
        raise OpsFehler(f"{pfad}-Antwort muss ein JSON-Objekt sein")
    return dokument


def _doctor(
    *,
    url: str,
    token_datei: str | None,
    timeout: float,
    erwartet_release: str | None = None,
    erwartet_git_sha: str | None = None,
    erwartet_manifest_sha256: str | None = None,
    erwartet_embedding_fingerprint: str | None = None,
    schichten: Sequence[str] = (),
    unsicheres_http: bool = False,
) -> dict[str, Any]:
    timeout = _positive_timeout(timeout, "Health-Timeout")
    if (
        erwartet_embedding_fingerprint is not None
        and not _SHA256_RE.fullmatch(erwartet_embedding_fingerprint)
    ):
        raise OpsFehler(
            "--expected-embedding-fingerprint muss 64 lowercase Hex-Zeichen enthalten"
        )
    basis = _basis_url(url, unsicheres_http=unsicheres_http)
    token = _token_lesen(token_datei)
    deadline = time.monotonic() + timeout
    letzter_fehler: OpsFehler | None = None
    while True:
        rest = deadline - time.monotonic()
        if rest <= 0:
            detail = f": {letzter_fehler}" if letzter_fehler else ""
            raise OpsFehler(f"Engine wurde innerhalb von {timeout:.1f}s nicht bereit{detail}")
        try:
            einzel_timeout = min(5.0, max(0.1, rest))
            health = _hole_json(basis, "/v1/health", token, einzel_timeout)
            status = _hole_json(basis, "/v1/status", token, einzel_timeout)
            if health.get("ok") is not True or status.get("ok") is not True:
                raise OpsFehler("Health/Status meldet ok nicht als true")
            engine_status = status.get("engine_status")
            if not isinstance(engine_status, dict):
                raise OpsFehler("Status enthaelt keinen engine_status-Block")
            release = engine_status.get("release")
            if any((erwartet_release, erwartet_git_sha, erwartet_manifest_sha256)):
                if not isinstance(release, dict):
                    raise OpsFehler("Status enthaelt keine Release-Identitaet")
                erwartete_identitaet = {
                    "release_id": erwartet_release,
                    "git_sha": erwartet_git_sha,
                    "manifest_sha256": erwartet_manifest_sha256,
                }
                for feld, erwartet in erwartete_identitaet.items():
                    if erwartet is not None and release.get(feld) != erwartet:
                        raise OpsFehler(
                            f"Falsche aktive Release-Identitaet fuer {feld}: "
                            f"erwartet {erwartet!r}, ist {release.get(feld)!r}"
                        )
            layer_map = engine_status.get("schichten")
            if not isinstance(layer_map, dict):
                raise OpsFehler("Status enthaelt keine Schichtinformationen")
            for name in schichten:
                eintrag = layer_map.get(name)
                if not isinstance(eintrag, dict):
                    raise OpsFehler(f"Erforderliche Schicht fehlt: {name}")
                aktiv = eintrag.get(
                    "aktiv", eintrag.get("verfuegbar", eintrag.get("konfiguriert"))
                )
                if aktiv is not True:
                    raise OpsFehler(f"Erforderliche Schicht ist nicht aktiv: {name}")
            embedding: dict[str, Any] | None = None
            if erwartet_embedding_fingerprint is not None:
                embedding = _hole_json(
                    basis, "/v1/embedding/status", token, einzel_timeout
                )
                if embedding.get("ok") is not True:
                    raise OpsFehler("Embedding-Status meldet ok nicht als true")
                index = embedding.get("index")
                if not isinstance(index, dict):
                    raise OpsFehler("Embedding-Status enthaelt keinen index-Block")
                if index.get("current") is not True:
                    raise OpsFehler("Aktiver Embedding-Index ist nicht aktuell")
                fingerprint = index.get("input_fingerprint")
                if fingerprint != erwartet_embedding_fingerprint:
                    raise OpsFehler(
                        "Falscher aktiver Embedding-Fingerprint: "
                        f"erwartet {erwartet_embedding_fingerprint!r}, "
                        f"ist {fingerprint!r}"
                    )
            result: dict[str, Any] = {
                "ok": True,
                "health": health,
                "status": status,
            }
            if embedding is not None:
                result["embedding"] = embedding
            return result
        except OpsFehler as exc:
            letzter_fehler = exc
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def _cmd_doctor(args: argparse.Namespace) -> int:
    ergebnis = _doctor(
        url=args.url,
        token_datei=args.token_file,
        timeout=args.timeout,
        erwartet_release=args.expected_release,
        erwartet_git_sha=args.expected_git_sha,
        erwartet_manifest_sha256=args.expected_manifest_sha256,
        erwartet_embedding_fingerprint=args.expected_embedding_fingerprint,
        schichten=args.require_layer,
        unsicheres_http=args.allow_insecure_http,
    )
    if args.json:
        _json_ausgabe(ergebnis)
    else:
        release = ergebnis["status"]["engine_status"].get("release", {})
        print(
            "bereit: "
            f"release={release.get('release_id') or 'development'} "
            f"api={ergebnis['health'].get('engineVersion', '?')}"
        )
    return 0


def _cmd_backup_create(args: argparse.Namespace) -> int:
    retention_values = _retention_values(args)
    zeit = args.zeitstempel or _utc_jetzt()
    release_id, git_sha, manifest_sha256 = _backup_identity(args)
    pfad = sichere_festwissen(
        args.berater,
        args.ziel,
        zeit,
        release_id=release_id,
        git_commit=git_sha,
        release_manifest_sha256=manifest_sha256,
        backup_target_id=args.backup_target_id,
    )
    info = pruefe_sicherung(pfad)
    retention = None
    if retention_values is not None:
        keep_last, keep_days = retention_values
        retention = bereinige_sicherungen(
            args.ziel,
            backup_target_id=args.backup_target_id,
            keep_last=keep_last,
            keep_days=keep_days,
            apply=True,
        )
    if args.json:
        _json_ausgabe({"path": str(pfad), "backup": info, "retention": retention})
    else:
        print(pfad)
    return 0


def _backup_identity(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    return (
        args.release_id
        or os.environ.get("RISK_LAYER_BACKUP_CURRENT_RELEASE_ID")
        or os.environ.get("RISK_LAYER_RELEASE_ID"),
        args.git_sha
        or os.environ.get("RISK_LAYER_BACKUP_CURRENT_GIT_SHA")
        or os.environ.get("RISK_LAYER_RELEASE_GIT_SHA"),
        args.release_manifest_sha256
        or os.environ.get("RISK_LAYER_BACKUP_CURRENT_MANIFEST_SHA256")
        or os.environ.get("RISK_LAYER_RELEASE_MANIFEST_SHA256"),
    )


def _retention_values(args: argparse.Namespace) -> tuple[int, int] | None:
    keep_last = args.retention_keep_last
    keep_days = args.retention_keep_days
    if (keep_last is None) != (keep_days is None):
        raise OpsFehler(
            "--retention-keep-last und --retention-keep-days muessen gemeinsam gesetzt sein"
        )
    if keep_last is None:
        return None
    if (
        type(keep_last) is not int
        or type(keep_days) is not int
        or keep_last < 1
        or keep_days < 1
    ):
        raise OpsFehler("Retention-Werte muessen positive Ganzzahlen sein")
    return keep_last, keep_days


def _cmd_backup_verify(args: argparse.Namespace) -> int:
    info = pruefe_sicherung(args.archiv)
    _json_ausgabe(info)
    if info.get("legacy") is True:
        print(
            "WARNUNG: manifestloses Legacy-Archiv; Hashes sind nicht im Archiv gebunden.",
            file=sys.stderr,
        )
    return 0


def _cmd_backup_prune(args: argparse.Namespace) -> int:
    ergebnis = bereinige_sicherungen(
        args.ziel,
        backup_target_id=args.backup_target_id,
        keep_last=args.keep_last,
        keep_days=args.keep_days,
        apply=args.apply,
    )
    _json_ausgabe(ergebnis)
    return 0


def _cmd_backup_restore(args: argparse.Namespace) -> int:
    if args.expected_sha256 is not None:
        if not _SHA256_RE.fullmatch(args.expected_sha256):
            raise OpsFehler("--expected-sha256 muss 64 lowercase Hex-Zeichen enthalten")
    ziel = Path(args.berater)
    vorher_ziel = Path(args.backup_ziel) if args.backup_ziel else Path(args.archiv).parent
    release_id, git_sha, manifest_sha256 = _backup_identity(args)
    # Leerheitspruefung, Pflicht-Vorher-Sicherung und Promotion muessen unter
    # demselben Festwissen-Lock geschehen. Eine Vorab-Verzweigung hier wuerde
    # einen parallelen Write zwischen Pruefung und Restore ungesichert verlieren.
    dateien, vorher, info = stelle_wieder_her_mit_sicherung(
        args.archiv,
        ziel,
        vorher_ziel,
        f"{_utc_jetzt()}-pre-restore",
        force=args.force,
        release_id=release_id,
        git_commit=git_sha,
        release_manifest_sha256=manifest_sha256,
        expected_sha256=args.expected_sha256,
        backup_target_id=args.backup_target_id,
    )
    ergebnis = {
        "ok": True,
        "target": str(Path(args.berater)),
        "files": dateien,
        "source": info,
        "pre_restore_backup": str(vorher) if vorher else None,
    }
    if args.json:
        _json_ausgabe(ergebnis)
    else:
        print(
            f"wiederhergestellt: {args.berater} "
            f"({', '.join(dateien) if dateien else 'explizit leerer Snapshot'})"
        )
    return 0


def _pruefe_manifest_vertrauen(
    manifest: ReleaseManifest, erwartet: str | None, trust_local: bool
) -> str:
    fingerprint = manifest.fingerprint()
    if erwartet is not None:
        if not _SHA256_RE.fullmatch(erwartet):
            raise OpsFehler("--manifest-sha256 muss 64 lowercase Hex-Zeichen enthalten")
        if not hmac.compare_digest(fingerprint, erwartet):
            raise OpsFehler(
                f"Manifest-Fingerprint stimmt nicht: erwartet {erwartet}, ist {fingerprint}"
            )
    elif not trust_local:
        raise OpsFehler(
            "Release-Vertrauensanker fehlt: --manifest-sha256 aus einem "
            "getrennt verifizierten Kanal oder --trust-local-manifest angeben"
        )
    return fingerprint


def _manifest_info(manifest: ReleaseManifest) -> dict[str, Any]:
    return {
        "release_id": manifest.release_id,
        "git_sha": manifest.git_sha,
        "schema_version": manifest.schema_version,
        "api_version": manifest.api_version,
        "features": list(manifest.features),
        "artifacts": len(manifest.artifacts),
        "manifest_sha256": manifest.fingerprint(),
    }


def _cmd_release_verify(args: argparse.Namespace) -> int:
    manifest = verify_bundle(args.bundle)
    _pruefe_manifest_vertrauen(manifest, args.manifest_sha256, args.trust_local_manifest)
    _json_ausgabe({"ok": True, **_manifest_info(manifest)})
    return 0


def _parse_command(raw: str | None, name: str) -> tuple[str, ...] | None:
    if raw is None:
        return None
    try:
        wert = json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise OpsFehler(f"{name} muss ein JSON-Array von Argumenten sein") from exc
    if (
        not isinstance(wert, list)
        or not wert
        or not all(isinstance(teil, str) and teil for teil in wert)
    ):
        raise OpsFehler(f"{name} muss ein nichtleeres JSON-String-Array sein")
    return tuple(wert)


def _release_env(
    record: ReleaseRecord | None,
    phase: str,
    runtime: _ReleaseRuntime | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["RISK_LAYER_RELEASE_PHASE"] = phase
    if record is None:
        for name in (
            "RISK_LAYER_RELEASE_DIR",
            "RISK_LAYER_RELEASE_ID",
            "RISK_LAYER_RELEASE_GIT_SHA",
            "RISK_LAYER_RELEASE_MANIFEST_SHA256",
            "RISK_LAYER_RUNTIME_DIR",
            "RISK_LAYER_RUNTIME_PYTHON",
            "RISK_LAYER_EMBEDDING_MODEL",
            "RISK_LAYER_EMBEDDING_OFFLINE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ):
            env[name] = ""
        return env
    env.update(
        {
            "RISK_LAYER_RELEASE_DIR": str(record.path),
            "RISK_LAYER_RELEASE_ID": record.manifest.release_id,
            "RISK_LAYER_RELEASE_GIT_SHA": record.manifest.git_sha,
            "RISK_LAYER_RELEASE_MANIFEST_SHA256": record.manifest.fingerprint(),
        }
    )
    embedding_env = (
        "RISK_LAYER_EMBEDDING_MODEL",
        "RISK_LAYER_EMBEDDING_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    if "embedding" in record.manifest.features:
        env.update({
            "RISK_LAYER_EMBEDDING_MODEL": str(record.path / "models/bge-m3"),
            "RISK_LAYER_EMBEDDING_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
    else:
        for name in embedding_env:
            env.pop(name, None)
    if runtime is not None:
        env.update(
            {
                "RISK_LAYER_RUNTIME_DIR": str(runtime.path),
                "RISK_LAYER_RUNTIME_PYTHON": str(runtime.python),
            }
        )
    return env


def _kommando(
    command: tuple[str, ...],
    record: ReleaseRecord | None,
    phase: str,
    timeout: float,
    runtime: _ReleaseRuntime | None = None,
) -> None:
    timeout = _positive_timeout(timeout, f"{phase}-Timeout")
    try:
        result = subprocess.run(
            command,
            env=_release_env(record, phase, runtime),
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpsFehler(f"{phase}-Kommando konnte nicht ausgefuehrt werden: {exc}") from exc
    if result.returncode != 0:
        raise OpsFehler(f"{phase}-Kommando endete mit Exit-Code {result.returncode}")


def _runtime_quellen(record: ReleaseRecord) -> _RuntimeQuellen:
    lock_pfad = f"runtime/locks/{_runtime_platform_tag()}.txt"
    lock = next(
        (artefakt for artefakt in record.manifest.artifacts if artefakt.path == lock_pfad),
        None,
    )
    if lock is None:
        raise OpsFehler(f"Release manifestiert keinen Host-Runtime-Lock: {lock_pfad}")
    if lock.size <= 0 or lock.size > _MAX_RUNTIME_LOCK_BYTES:
        raise OpsFehler(
            f"Runtime-Lock muss 1-{_MAX_RUNTIME_LOCK_BYTES} Bytes gross sein"
        )

    wheelhouse_relativ = PurePosixPath(
        "runtime", "wheelhouse", _runtime_platform_tag()
    )
    passende: list[ArtifactDigest] = []
    for artefakt in record.manifest.artifacts:
        pfad = PurePosixPath(artefakt.path)
        if len(pfad.parts) >= len(wheelhouse_relativ.parts) and (
            pfad.parts[: len(wheelhouse_relativ.parts)] == wheelhouse_relativ.parts
        ):
            if pfad.parent != wheelhouse_relativ or pfad.suffix.lower() != ".whl":
                raise OpsFehler(
                    "Host-Wheelhouse darf nur direkte .whl-Dateien enthalten: "
                    f"{artefakt.path}"
                )
            passende.append(artefakt)
    if not passende:
        raise OpsFehler(
            "Release enthaelt kein Wheel fuer den Host: "
            f"runtime/wheelhouse/{_runtime_platform_tag()}/*.whl"
        )

    app = [
        artefakt
        for artefakt in record.manifest.artifacts
        if PurePosixPath(artefakt.path).parent == PurePosixPath("app")
        and PurePosixPath(artefakt.path).suffix.lower() == ".whl"
    ]
    if len(app) != 1:
        raise OpsFehler(f"Release braucht genau ein app/*.whl, gefunden: {len(app)}")
    return _RuntimeQuellen(lock=lock, wheels=tuple(passende), app_wheel=app[0])


def _artefakt_sicher_kopieren(
    record: ReleaseRecord, artefakt: ArtifactDigest, ziel: Path
) -> None:
    quelle = record.path.joinpath(*PurePosixPath(artefakt.path).parts)
    try:
        vorher = os.lstat(quelle)
    except OSError as exc:
        raise OpsFehler(f"Runtime-Artefakt fehlt: {artefakt.path}") from exc
    if (
        not stat.S_ISREG(vorher.st_mode)
        or stat.S_ISLNK(vorher.st_mode)
        or _ist_reparse(vorher)
        or vorher.st_nlink != 1
    ):
        raise OpsFehler(f"Unsicheres Runtime-Artefakt: {artefakt.path}")
    lese_flags = (
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    schreib_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    ziel.parent.mkdir(parents=True, exist_ok=True)
    try:
        quelle_fd = os.open(quelle, lese_flags)
    except OSError as exc:
        raise OpsFehler(
            f"Runtime-Artefakt kann nicht sicher geoeffnet werden: {artefakt.path}"
        ) from exc
    try:
        offen = os.fstat(quelle_fd)
        benannt = os.lstat(quelle)
        if (
            not stat.S_ISREG(offen.st_mode)
            or offen.st_nlink != 1
            or (offen.st_dev, offen.st_ino) != (benannt.st_dev, benannt.st_ino)
            or offen.st_size != artefakt.size
        ):
            raise OpsFehler(f"Runtime-Artefakt wurde ausgetauscht: {artefakt.path}")
        try:
            ziel_fd = os.open(ziel, schreib_flags, 0o600)
        except OSError as exc:
            raise OpsFehler(f"Private Runtime-Kopie kann nicht angelegt werden: {ziel}") from exc
        digest = hashlib.sha256()
        groesse = 0
        try:
            while True:
                block = os.read(quelle_fd, 1024 * 1024)
                if not block:
                    break
                groesse += len(block)
                if groesse > artefakt.size:
                    raise OpsFehler(f"Runtime-Artefakt wuchs beim Lesen: {artefakt.path}")
                digest.update(block)
                rest = block
                while rest:
                    geschrieben = os.write(ziel_fd, rest)
                    if geschrieben <= 0:
                        raise OpsFehler(f"Runtime-Kopie unvollstaendig: {ziel}")
                    rest = rest[geschrieben:]
            os.fsync(ziel_fd)
        finally:
            os.close(ziel_fd)
        if groesse != artefakt.size or not hmac.compare_digest(
            digest.hexdigest(), artefakt.sha256
        ):
            raise OpsFehler(f"Runtime-Artefakt stimmt nicht zum Manifest: {artefakt.path}")
    except BaseException:
        ziel.unlink(missing_ok=True)
        raise
    finally:
        os.close(quelle_fd)


def _runtime_lock_pruefen(pfad: Path) -> None:
    roh = _sichere_datei_lesen(pfad, _MAX_RUNTIME_LOCK_BYTES, "Runtime-Lock")
    try:
        text = roh.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OpsFehler("Runtime-Lock ist nicht gueltiges UTF-8") from exc
    if "\x00" in text:
        raise OpsFehler("Runtime-Lock enthaelt NUL-Bytes")
    anforderungen = 0
    hashes = 0
    for nummer, zeile in enumerate(text.splitlines(), start=1):
        inhalt = zeile.split("#", 1)[0].strip()
        if not inhalt:
            continue
        if inhalt.endswith("\\"):
            inhalt = inhalt[:-1].rstrip()
        if not inhalt:
            continue
        if inhalt.startswith("--hash="):
            if not _HASH_OPTION_RE.fullmatch(inhalt):
                raise OpsFehler(f"Ungueltige Hash-Option im Runtime-Lock, Zeile {nummer}")
            hashes += 1
            continue
        if (
            inhalt.startswith("-")
            or "://" in inhalt
            or " @ " in inhalt
            or "file:" in inhalt.casefold()
            or "${" in inhalt
            or " --" in inhalt
        ):
            raise OpsFehler(
                f"Runtime-Lock darf keine externen Quellen/Optionen enthalten, Zeile {nummer}"
            )
        anforderung = inhalt.split(";", 1)[0].strip()
        if not _PINNED_REQUIREMENT_RE.fullmatch(anforderung):
            raise OpsFehler(
                f"Runtime-Lock muss exakt mit == pinnen, Zeile {nummer}: {anforderung!r}"
            )
        anforderungen += 1
    if anforderungen == 0 or hashes == 0:
        raise OpsFehler("Runtime-Lock muss mindestens einen Hash-gepinnten Bedarf enthalten")


def _runtime_umgebung(record: ReleaseRecord, phase: str) -> dict[str, str]:
    env = _release_env(record, phase)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def _runtime_subprozess(
    kommando: Sequence[str],
    *,
    record: ReleaseRecord,
    cwd: Path,
    timeout: float,
    phase: str,
) -> None:
    timeout = _positive_timeout(timeout, f"{phase}-Timeout")
    try:
        ergebnis = subprocess.run(
            list(kommando),
            cwd=cwd,
            env=_runtime_umgebung(record, phase),
            capture_output=True,
            check=False,
            timeout=timeout,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpsFehler(f"{phase} konnte nicht ausgefuehrt werden: {exc}") from exc
    if ergebnis.returncode != 0:
        detail = (ergebnis.stderr or ergebnis.stdout).strip()[-2000:]
        raise OpsFehler(f"{phase} endete mit {ergebnis.returncode}: {detail}")


def _runtime_marker_schreiben(
    staging: Path, record: ReleaseRecord, python: Path
) -> None:
    python_hash, python_size = _datei_hash_und_groesse(
        python, "neu gebauter Runtime-Interpreter"
    )
    payload = (
        json.dumps(
            _runtime_marker_payload(record, python_hash, python_size),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    marker = staging / _RUNTIME_MARKER
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(marker, flags, 0o600)
    except OSError as exc:
        raise OpsFehler(f"Runtime-Marker kann nicht angelegt werden: {marker}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _runtime_sicherstellen(
    home: str | Path, record: ReleaseRecord, timeout: float
) -> _ReleaseRuntime:
    timeout = _positive_timeout(timeout, "Runtime-Bau-Timeout")
    wurzel = _runtime_root(home, erstellen=True)
    fingerprint = record.manifest.fingerprint()
    ziel = wurzel / _runtime_name(record.manifest.release_id, fingerprint)
    if ziel.exists() or _ist_link_oder_reparse(ziel):
        return _runtime_pruefen(home, record)

    quellen = _runtime_quellen(record)
    staging = Path(tempfile.mkdtemp(dir=wurzel, prefix=f".staging-{fingerprint[:12]}-"))
    if os.name != "nt":
        os.chmod(staging, 0o700)
    try:
        _runtime_subprozess(
            [sys.executable, "-I", "-m", "venv", "--copies", str(staging)],
            record=record,
            cwd=wurzel,
            timeout=timeout,
            phase="Runtime-Venv-Bau",
        )
        python = staging / _runtime_python_relativ()
        if not python.is_file() or _ist_link_oder_reparse(python):
            raise OpsFehler(f"Venv erzeugte keinen regulaeren kopierten Interpreter: {python}")

        inputs = staging / ".install-inputs"
        wheelhouse = inputs / "wheelhouse"
        wheelhouse.mkdir(parents=True)
        lock = inputs / "requirements-runtime-lock.txt"
        _artefakt_sicher_kopieren(record, quellen.lock, lock)
        _runtime_lock_pruefen(lock)
        for artefakt in quellen.wheels:
            _artefakt_sicher_kopieren(
                record, artefakt, wheelhouse / PurePosixPath(artefakt.path).name
            )
        app = inputs / "app" / PurePosixPath(quellen.app_wheel.path).name
        _artefakt_sicher_kopieren(record, quellen.app_wheel, app)

        _runtime_subprozess(
            [
                str(python), "-I", "-m", "pip", "--isolated", "install",
                "--require-hashes", "--no-index", "--no-cache-dir",
                "--only-binary=:all:", "--find-links", str(wheelhouse),
                "-r", str(lock),
            ],
            record=record,
            cwd=staging,
            timeout=timeout,
            phase="Runtime-Hashlock-Installation",
        )
        _runtime_subprozess(
            [
                str(python), "-I", "-m", "pip", "--isolated", "install",
                "--no-index", "--no-cache-dir", "--no-deps",
                "--only-binary=:all:", str(app),
            ],
            record=record,
            cwd=staging,
            timeout=timeout,
            phase="Runtime-Anwendungsinstallation",
        )
        code = (
            "import risk_layer, risk_layer.web; "
            "assert getattr(risk_layer, '__version__', None); "
            "print(risk_layer.__version__)"
        )
        if "embedding" in record.manifest.features:
            code = (
                "import os, sentence_transformers; "
                "from risk_layer.embedding_model import validate_local_bge_m3; "
                "validate_local_bge_m3(os.environ['RISK_LAYER_EMBEDDING_MODEL']); "
            ) + code
        _runtime_subprozess(
            [str(python), "-I", "-c", code],
            record=record,
            cwd=staging,
            timeout=timeout,
            phase="Candidate-Import",
        )
        shutil.rmtree(inputs)
        _runtime_marker_schreiben(staging, record, python)
        try:
            os.rename(staging, ziel)
        except FileExistsError:
            # Ein fremder Prozess darf nie zwei verschiedene Runtime-Baeume
            # unter demselben Ziel vermischen. Nur ein bereits gueltiger Baum
            # kann nach dem Entfernen unseres privaten Stagings wiederverwendet werden.
            shutil.rmtree(staging, ignore_errors=True)
            return _runtime_pruefen(home, record)
        if os.name != "nt":
            descriptor = os.open(wurzel, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return _ReleaseRuntime(path=ziel, python=ziel / _runtime_python_relativ(), created=True)
    except BaseException:
        if staging.exists() and not _ist_link_oder_reparse(staging):
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _candidate_pruefung(
    home: str | Path,
    command: tuple[str, ...] | None,
    hook_timeout: float,
    runtime_bereit: Callable[[_ReleaseRuntime], None] | None = None,
) -> Callable[[ReleaseRecord], bool]:
    hook_timeout = _positive_timeout(hook_timeout, "Candidate-Hook-Timeout")

    def pruefen(record: ReleaseRecord) -> bool:
        runtime = _runtime_sicherstellen(home, record, hook_timeout)
        if runtime_bereit is not None:
            runtime_bereit(runtime)
        if "festwissen" in record.manifest.features:
            for relativ in (
                "catalog/begriffe.yaml",
                "corpus/graph.sqlite",
                "corpus/trigger.sqlite",
                "corpus/norm_manifest.json",
            ):
                if not (record.path / relativ).is_file():
                    raise OpsFehler(f"Festwissen-Pflichtartefakt fehlt: {relativ}")
        if command is not None:
            _kommando(command, record, "candidate", hook_timeout, runtime)
        return True

    return pruefen


def _aktivierung(
    *,
    home: str | Path,
    command: tuple[str, ...] | None,
    hook_timeout: float,
    health_url: str | None,
    token_file: str | None,
    health_timeout: float,
    require_layers: Sequence[str],
    allow_insecure_http: bool,
) -> Callable[[ReleaseRecord | None], bool]:
    hook_timeout = _positive_timeout(hook_timeout, "Aktivierungs-Hook-Timeout")
    health_timeout = _positive_timeout(health_timeout, "Health-Timeout")

    def aktivieren(record: ReleaseRecord | None) -> bool:
        runtime = _runtime_pruefen(home, record) if record is not None else None
        if command is not None:
            _kommando(
                command,
                record,
                "activate" if record else "deactivate",
                hook_timeout,
                runtime,
            )
        if record is not None and health_url is not None:
            _doctor(
                url=health_url,
                token_datei=token_file,
                timeout=health_timeout,
                erwartet_release=record.manifest.release_id,
                erwartet_git_sha=record.manifest.git_sha,
                erwartet_manifest_sha256=record.manifest.fingerprint(),
                schichten=require_layers,
                unsicheres_http=allow_insecure_http,
            )
        return True

    return aktivieren


def _state_info(manager: ReleaseManager, state: ReleaseState) -> dict[str, Any]:
    def record_info(
        release_id: str | None, expected_manifest_sha256: str | None
    ) -> dict[str, Any] | None:
        if release_id is None:
            return None
        record = manager.verify(
            release_id, expected_manifest_sha256=expected_manifest_sha256
        )
        return {"path": str(record.path), **_manifest_info(record.manifest)}

    return {
        "format_version": state.format_version,
        "generation": state.generation,
        "updated_at": state.updated_at,
        "current": record_info(state.current, state.current_manifest_sha256),
        "previous": record_info(state.previous, state.previous_manifest_sha256),
    }


def _cmd_release_install(args: argparse.Namespace) -> int:
    retention_values = _retention_values(args)
    manifest = verify_bundle(args.bundle)
    fingerprint = _pruefe_manifest_vertrauen(
        manifest, args.manifest_sha256, args.trust_local_manifest
    )
    backup_ziel = Path(args.backup_ziel)
    # Der Expert-Einstieg darf vor einer gebundenen Off-host-Zielpruefung weder
    # Release-Home noch Berater-Verzeichnis oder Staging-Generationen anlegen.
    pruefe_backup_ziel(backup_ziel, args.backup_target_id, writable=True)
    berater = Path(args.berater)
    berater.mkdir(parents=True, exist_ok=True)
    manager = ReleaseManager(args.home, lock_timeout=args.lock_timeout)
    candidate_command = _parse_command(args.candidate_command, "--candidate-command")
    activation_command = _parse_command(args.activation_command, "--activation-command")
    backup_pfad: Path | None = None
    retention: dict[str, Any] | None = None
    candidate_runtime: _ReleaseRuntime | None = None

    def backup(state: ReleaseState, candidate: ReleaseRecord) -> bool:
        nonlocal backup_pfad, retention
        current = (
            manager.verify(
                state.current,
                expected_manifest_sha256=state.current_manifest_sha256,
            )
            if state.current is not None
            else None
        )
        backup_pfad = sichere_festwissen(
            berater,
            backup_ziel,
            _utc_jetzt(),
            release_id=(
                current.manifest.release_id if current is not None else "pre-install"
            ),
            git_commit=(current.manifest.git_sha if current is not None else None),
            release_manifest_sha256=(
                current.manifest.fingerprint() if current is not None else None
            ),
            backup_target_id=args.backup_target_id,
        )
        pruefe_sicherung(backup_pfad)
        if retention_values is not None:
            keep_last, keep_days = retention_values
            retention = bereinige_sicherungen(
                backup_ziel,
                backup_target_id=args.backup_target_id,
                keep_last=keep_last,
                keep_days=keep_days,
                apply=True,
            )
        return True

    activation = _aktivierung(
        home=args.home,
        command=activation_command,
        hook_timeout=args.hook_timeout,
        health_url=args.health_url,
        token_file=args.token_file,
        health_timeout=args.health_timeout,
        require_layers=args.require_layer,
        allow_insecure_http=args.allow_insecure_http,
    )

    def runtime_merken(runtime: _ReleaseRuntime) -> None:
        nonlocal candidate_runtime
        candidate_runtime = runtime

    try:
        state = manager.apply(
            args.bundle,
            backup=backup,
            candidate_check=_candidate_pruefung(
                args.home,
                candidate_command,
                args.hook_timeout,
                runtime_merken,
            ),
            activation=activation,
            expected_manifest_sha256=fingerprint,
        )
    except BaseException as exc:
        # Nur eine in diesem Versuch neu gebaute und sicher nicht mehr aktive
        # Runtime wird bereinigt. Bei unvollstaendiger externer Kompensation
        # bleibt sie vorsichtshalber erhalten.
        darf_bereinigen = not isinstance(exc, ActivationError) or exc.rolled_back
        if candidate_runtime is not None and candidate_runtime.created and darf_bereinigen:
            try:
                zustand = manager.read_state()
                ist_aktiv = (
                    zustand.current == manifest.release_id
                    and zustand.current_manifest_sha256 == fingerprint
                )
                if not ist_aktiv:
                    _runtime_entfernen(args.home, manifest.release_id, fingerprint)
            except (OpsFehler, ReleaseError, OSError) as cleanup_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"Neue Candidate-Runtime konnte nicht bereinigt werden: {cleanup_error}")
        raise
    _json_ausgabe(
        {
            "ok": True,
            "trusted_manifest_sha256": fingerprint,
            "backup": str(backup_pfad) if backup_pfad else None,
            "retention": retention,
            "state": _state_info(manager, state),
        }
    )
    return 0


def _cmd_release_status(args: argparse.Namespace) -> int:
    manager = ReleaseManager(args.home, lock_timeout=args.lock_timeout)
    _json_ausgabe({"ok": True, "state": _state_info(manager, manager.read_state())})
    return 0


def _cmd_release_prune(args: argparse.Namespace) -> int:
    manager = ReleaseManager(args.home, lock_timeout=args.lock_timeout)

    def runtime_bereinigen(record: ReleaseRecord) -> None:
        _runtime_entfernen(
            args.home,
            record.manifest.release_id,
            record.manifest.fingerprint(),
        )

    ergebnis = manager.prune(
        keep=args.keep,
        apply=args.apply,
        after_delete=runtime_bereinigen if args.apply else None,
    )
    _json_ausgabe(ergebnis)
    return 0


def _cmd_release_rollback(args: argparse.Namespace) -> int:
    manager = ReleaseManager(args.home, lock_timeout=args.lock_timeout)
    vorher = manager.read_state()
    if vorher.previous is not None:
        rollback_record = manager.verify(
            vorher.previous,
            expected_manifest_sha256=vorher.previous_manifest_sha256,
        )
        _runtime_pruefen(args.home, rollback_record)
    activation_command = _parse_command(args.activation_command, "--activation-command")
    activation = _aktivierung(
        home=args.home,
        command=activation_command,
        hook_timeout=args.hook_timeout,
        health_url=args.health_url,
        token_file=args.token_file,
        health_timeout=args.health_timeout,
        require_layers=args.require_layer,
        allow_insecure_http=args.allow_insecure_http,
    )
    state = manager.rollback(activation=activation)
    _json_ausgabe({"ok": True, "state": _state_info(manager, state)})
    return 0


def _cmd_service_run(args: argparse.Namespace) -> int:
    manager = ReleaseManager(args.home, lock_timeout=args.lock_timeout)
    record = manager.current()
    if record is None:
        raise OpsFehler("Kein current-Release installiert")
    runtime = _runtime_pruefen(args.home, record)
    env = _release_env(record, "run", runtime)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    kommando = [
        str(runtime.python),
        "-P",
        "-m",
        "risk_layer.web",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--berater",
        str(Path(args.berater).absolute()),
        "--graph",
        "corpus/graph.sqlite",
        "--llm-url",
        args.llm_url,
    ]
    if args.llm_model:
        kommando.extend(("--llm-model", str(Path(args.llm_model).absolute())))
    if args.llm_bin:
        kommando.extend(("--llm-bin", str(Path(args.llm_bin).absolute())))
    if args.llm_autostart:
        kommando.append("--llm-autostart")
    os.chdir(record.path)
    os.execve(str(runtime.python), kommando, env)
    raise AssertionError("os.execve kehrte unerwartet zurueck")


def _trust_args(parser: argparse.ArgumentParser) -> None:
    gruppe = parser.add_mutually_exclusive_group(required=True)
    gruppe.add_argument(
        "--manifest-sha256",
        help="kanonischer Manifest-SHA-256 aus getrennt verifiziertem Kanal",
    )
    gruppe.add_argument(
        "--trust-local-manifest",
        action="store_true",
        help="lokales Manifest bewusst als Vertrauensanker akzeptieren",
    )


def _activation_args(parser: argparse.ArgumentParser) -> None:
    aktivierung = parser.add_mutually_exclusive_group(required=True)
    aktivierung.add_argument(
        "--activation-command",
        help='JSON-Array ohne Shell, z. B. ["systemctl","--user","restart","risk-layer"]',
    )
    aktivierung.add_argument(
        "--no-activate",
        action="store_true",
        help="nur current/previous schalten; Dienst bewusst extern aktivieren",
    )
    health = parser.add_mutually_exclusive_group(required=True)
    health.add_argument("--health-url", help="Basis-URL der nach Aktivierung zu pruefenden Engine")
    health.add_argument(
        "--no-health-check",
        action="store_true",
        help="Post-Health bewusst extern verantworten",
    )
    parser.add_argument("--token-file")
    parser.add_argument("--health-timeout", type=_timeout_argument, default=180.0)
    parser.add_argument("--hook-timeout", type=_timeout_argument, default=180.0)
    parser.add_argument("--require-layer", action="append", default=[])
    parser.add_argument("--allow-insecure-http", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk-layer-ops",
        description="Transaktionaler Risk-Layer-Betrieb: pruefen, sichern, aktualisieren, rollback.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Health, Status, Release und Soll-Schichten pruefen")
    doctor.add_argument("--url", default="http://127.0.0.1:8000")
    doctor.add_argument("--token-file")
    doctor.add_argument("--timeout", type=_timeout_argument, default=30.0)
    doctor.add_argument("--expected-release")
    doctor.add_argument("--expected-git-sha")
    doctor.add_argument("--expected-manifest-sha256")
    doctor.add_argument("--expected-embedding-fingerprint")
    doctor.add_argument("--require-layer", action="append", default=[])
    doctor.add_argument("--allow-insecure-http", action="store_true")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=_cmd_doctor)

    backup = sub.add_parser("backup", help="Festwissen sichern, pruefen, wiederherstellen")
    backup_sub = backup.add_subparsers(dest="backup_command", required=True)
    create = backup_sub.add_parser("create", help="manifestierten konsistenten Snapshot anlegen")
    create.add_argument("--berater", required=True)
    create.add_argument("--ziel", required=True, help="explizites getrenntes Backup-Ziel")
    create.add_argument("--backup-target-id", required=True)
    create.add_argument("--zeitstempel")
    create.add_argument("--release-id")
    create.add_argument("--git-sha")
    create.add_argument("--release-manifest-sha256")
    create.add_argument("--retention-keep-last", type=int)
    create.add_argument("--retention-keep-days", type=int)
    create.add_argument("--json", action="store_true")
    create.set_defaults(func=_cmd_backup_create)
    verify = backup_sub.add_parser("verify", help="Archiv ohne Mutation vollstaendig pruefen")
    verify.add_argument("archiv")
    verify.set_defaults(func=_cmd_backup_verify)
    prune = backup_sub.add_parser(
        "prune", help="Retention planen oder verifizierte alte Archive loeschen"
    )
    prune.add_argument("--ziel", required=True)
    prune.add_argument("--backup-target-id", required=True)
    prune.add_argument("--keep-last", type=int, default=30)
    prune.add_argument("--keep-days", type=int, default=90)
    prune.add_argument(
        "--apply", action="store_true", help="Plan tatsaechlich anwenden (sonst Dry-run)"
    )
    prune.set_defaults(func=_cmd_backup_prune)
    restore = backup_sub.add_parser("restore", help="validieren, stagen und atomar wiederherstellen")
    restore.add_argument("archiv")
    restore.add_argument("--expected-sha256")
    restore.add_argument("--berater", required=True)
    restore.add_argument("--backup-target-id", required=True)
    restore.add_argument("--release-id")
    restore.add_argument("--git-sha")
    restore.add_argument("--release-manifest-sha256")
    restore.add_argument("--force", action="store_true")
    restore.add_argument(
        "--backup-ziel",
        help="Ziel fuer Pflicht-Vorher-Sicherung (Default: Verzeichnis des Archivs)",
    )
    restore.add_argument("--json", action="store_true")
    restore.set_defaults(func=_cmd_backup_restore)

    release = sub.add_parser("release", help="unveraenderliche Releases verwalten")
    release_sub = release.add_subparsers(dest="release_command", required=True)
    release_verify = release_sub.add_parser("verify", help="lokales Bundle und Trust-Hash pruefen")
    release_verify.add_argument("bundle")
    _trust_args(release_verify)
    release_verify.set_defaults(func=_cmd_release_verify)

    install = release_sub.add_parser("install", help="Backup, Stage, Smoke, Switch und Rollback")
    install.add_argument("bundle")
    install.add_argument("--home", required=True)
    install.add_argument("--berater", required=True)
    install.add_argument("--backup-ziel", required=True)
    install.add_argument("--backup-target-id", required=True)
    install.add_argument("--retention-keep-last", type=int)
    install.add_argument("--retention-keep-days", type=int)
    install.add_argument("--candidate-command", help="optionaler Pre-Switch-Hook als JSON-Array")
    install.add_argument("--lock-timeout", type=_lock_timeout_argument, default=30.0)
    _trust_args(install)
    _activation_args(install)
    install.set_defaults(func=_cmd_release_install)

    status = release_sub.add_parser("status", help="current/previous verifizieren und anzeigen")
    status.add_argument("--home", required=True)
    status.add_argument("--lock-timeout", type=_lock_timeout_argument, default=30.0)
    status.set_defaults(func=_cmd_release_status)

    release_prune = release_sub.add_parser(
        "prune", help="alte inaktive Releases und deren Runtime bereinigen"
    )
    release_prune.add_argument("--home", required=True)
    release_prune.add_argument("--keep", type=int, default=3)
    release_prune.add_argument("--lock-timeout", type=_lock_timeout_argument, default=30.0)
    release_prune.add_argument(
        "--apply", action="store_true", help="Plan tatsaechlich anwenden (sonst Dry-run)"
    )
    release_prune.set_defaults(func=_cmd_release_prune)

    rollback = release_sub.add_parser("rollback", help="atomar zu previous zurueckschalten")
    rollback.add_argument("--home", required=True)
    rollback.add_argument("--lock-timeout", type=_lock_timeout_argument, default=30.0)
    _activation_args(rollback)
    rollback.set_defaults(func=_cmd_release_rollback)

    service = sub.add_parser("service", help="stabilen Dienst aus current starten")
    service_sub = service.add_subparsers(dest="service_command", required=True)
    run = service_sub.add_parser(
        "run", help="verifizierte per-Release-Runtime von current ausfuehren"
    )
    run.add_argument("--home", required=True)
    run.add_argument("--berater", required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--llm-url", default="http://127.0.0.1:8080")
    run.add_argument("--llm-model")
    run.add_argument("--llm-bin")
    run.add_argument("--llm-autostart", action="store_true")
    run.add_argument("--lock-timeout", type=_lock_timeout_argument, default=30.0)
    run.set_defaults(func=_cmd_service_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ActivationError as exc:
        status = "vollstaendig zurueckgerollt" if exc.rolled_back else "Rollback unvollstaendig"
        print(f"Aktivierungs-Fehler ({status}): {exc}", file=sys.stderr)
        return 3
    except (OpsFehler, SicherungsFehler, ManifestError, ReleaseError) as exc:
        print(f"Betriebs-Fehler: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
