"""Transaktionaler lokaler Release-/Update-/Rollback-Kern.

Eine Installation besteht aus unveraenderlichen Generationen unter
``releases/<release_id>`` und genau einer atomar ersetzten ``state.json``.
Damit sehen Leser auch bei Prozessabbruch entweder den alten oder den neuen
vollstaendigen Zustand, niemals halb aktualisierte current/previous-Zeiger.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .release_manifest import (
    MANIFEST_FILENAME,
    MAX_PATH_COMPONENTS,
    MAX_TREE_ENTRIES,
    ArtifactDigest,
    ManifestError,
    ReleaseManifest,
    _is_link_like,
    validate_release_id,
)


MAX_OPERATION_TIMEOUT_SECONDS = 3600.0
MAX_POLL_INTERVAL_SECONDS = 60.0


def _bounded_timeout(
    value: float,
    name: str,
    maximum: float,
    *,
    allow_zero: bool = False,
) -> float:
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        normalized = math.nan
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(normalized)
        or normalized < 0
        or (normalized == 0 and not allow_zero)
        or normalized > maximum
    ):
        operator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} muss endlich und {operator} bis {maximum:g}s sein")
    return normalized


STATE_FORMAT_VERSION = 2
STATE_FILENAME = "state.json"
LOCK_FILENAME = "update.lock"
MAX_STATE_BYTES = 64 * 1024


class ReleaseError(RuntimeError):
    """Basisklasse fuer Installations- und Lifecycle-Fehler."""


class IntegrityError(ReleaseError):
    """Dateibaum, Groesse oder Hash stimmt nicht mit dem Manifest ueberein."""


class ReleaseStateError(ReleaseError):
    """Der persistierte Release-Zustand ist ungueltig oder unvollstaendig."""


class SchemaMismatchError(ReleaseStateError):
    """Ein Update wuerde ohne implementierte Migration das Datenschema wechseln."""


class ReleaseLockError(ReleaseError):
    """Der exklusive Update-Lock konnte nicht erworben werden."""


class ReleaseCheckError(ReleaseError):
    """Backup-, Candidate- oder Aktivierungs-Callback ist fehlgeschlagen."""


class ActivationError(ReleaseCheckError):
    """Aktivierung schlug nach dem State-Switch fehl.

    ``rolled_back`` ist nur wahr, wenn sowohl der persistierte Zustand als auch
    die externe Aktivierung auf den vorherigen Record zurueckgefuehrt wurden.
    """

    def __init__(
        self,
        message: str,
        *,
        rolled_back: bool,
        activation_error: BaseException,
        recovery_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.rolled_back = rolled_back
        self.activation_error = activation_error
        self.recovery_error = recovery_error


@dataclass(frozen=True, slots=True)
class ReleaseRecord:
    """Verifizierte Release-Generation und ihr lokaler Pfad."""

    path: Path
    manifest: ReleaseManifest


@dataclass(frozen=True, slots=True)
class ReleaseState:
    """Kleine, atomar persistierte current/previous-Zustandsmaschine."""

    generation: int = 0
    current: str | None = None
    current_manifest_sha256: str | None = None
    previous: str | None = None
    previous_manifest_sha256: str | None = None
    updated_at: str | None = None
    format_version: int = STATE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.format_version, bool) or self.format_version != STATE_FORMAT_VERSION:
            raise ReleaseStateError(
                f"Nicht unterstuetzte State-Formatversion: {self.format_version!r}"
            )
        if (isinstance(self.generation, bool) or not isinstance(self.generation, int)
                or self.generation < 0):
            raise ReleaseStateError("State-generation muss eine nichtnegative Ganzzahl sein")
        if self.current is not None:
            try:
                validate_release_id(self.current)
            except ManifestError as exc:
                raise ReleaseStateError(f"Ungueltige current-ID: {self.current!r}") from exc
        if (self.current is None) != (self.current_manifest_sha256 is None):
            raise ReleaseStateError(
                "current und current_manifest_sha256 muessen gemeinsam gesetzt sein"
            )
        if self.current_manifest_sha256 is not None and not _is_sha256(
            self.current_manifest_sha256
        ):
            raise ReleaseStateError("current_manifest_sha256 ist kein lowercase SHA-256")
        if self.previous is not None:
            try:
                validate_release_id(self.previous)
            except ManifestError as exc:
                raise ReleaseStateError(f"Ungueltige previous-ID: {self.previous!r}") from exc
        if (self.previous is None) != (self.previous_manifest_sha256 is None):
            raise ReleaseStateError(
                "previous und previous_manifest_sha256 muessen gemeinsam gesetzt sein"
            )
        if self.previous_manifest_sha256 is not None and not _is_sha256(
            self.previous_manifest_sha256
        ):
            raise ReleaseStateError("previous_manifest_sha256 ist kein lowercase SHA-256")
        if (self.current is not None and self.previous is not None
                and self.current.casefold() == self.previous.casefold()):
            raise ReleaseStateError("current und previous duerfen nicht identisch sein")
        if self.updated_at is not None and (
                not isinstance(self.updated_at, str) or len(self.updated_at) > 64):
            raise ReleaseStateError("updated_at muss null oder ein kurzer Zeitstempel sein")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseState":
        if not isinstance(data, Mapping):
            raise ReleaseStateError("State muss ein JSON-Objekt sein")
        required = {
            "format_version", "generation", "current", "current_manifest_sha256",
            "previous", "previous_manifest_sha256", "updated_at"
        }
        if set(data) != required:
            raise ReleaseStateError("State enthaelt fehlende oder unbekannte Felder")
        return cls(
            format_version=data["format_version"],
            generation=data["generation"],
            current=data["current"],
            current_manifest_sha256=data["current_manifest_sha256"],
            previous=data["previous"],
            previous_manifest_sha256=data["previous_manifest_sha256"],
            updated_at=data["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "generation": self.generation,
            "current": self.current,
            "current_manifest_sha256": self.current_manifest_sha256,
            "previous": self.previous,
            "previous_manifest_sha256": self.previous_manifest_sha256,
            "updated_at": self.updated_at,
        }


ReleaseCheck = Callable[[ReleaseRecord], bool | None]
BackupHook = Callable[[ReleaseState, ReleaseRecord], bool | None]
ActivationHook = Callable[[ReleaseRecord | None], bool | None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _check_directory_chain(path: Path, description: str) -> None:
    """Blockt Links/Reparse-Points in jeder existierenden Pfadkomponente."""
    absolute = Path(os.path.abspath(path))
    for component in reversed((absolute, *absolute.parents)):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise IntegrityError(
                f"{description} kann nicht sicher geprueft werden: {component}"
            ) from exc
        attributes = getattr(info, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(info.st_mode) or (reparse and attributes & reparse):
            raise IntegrityError(
                f"Symlink-/Reparse-Komponente in {description}: {component}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise IntegrityError(
                f"Nicht-Verzeichnis-Komponente in {description}: {component}"
            )


def _ensure_real_directory(path: Path, description: str) -> None:
    # Vor und nach mkdir pruefen: mkdir(parents=True) darf weder durch einen
    # vorbereiteten Link umgelenkt werden noch eine im Race ausgetauschte
    # Komponente unbemerkt akzeptieren.
    _check_directory_chain(path, description)
    if path.exists() or _is_link_like(path):
        if _is_link_like(path):
            raise IntegrityError(f"{description} darf kein Symlink/Reparse-Point sein: {path}")
        if not path.is_dir():
            raise IntegrityError(f"{description} ist kein Verzeichnis: {path}")
        _check_directory_chain(path, description)
        return
    path.mkdir(parents=True, exist_ok=True)
    _check_directory_chain(path, description)
    if _is_link_like(path) or not path.is_dir():
        raise IntegrityError(f"{description} konnte nicht sicher erstellt werden: {path}")


def _inventory(root: Path) -> tuple[set[str], set[str]]:
    """Inventarisiert ohne Links zu folgen; Pfade werden als POSIX geliefert."""
    _ensure_real_directory(root, "Release-Baum")
    files: set[str] = set()
    directories: set[str] = set()

    pending: list[tuple[Path, PurePosixPath | None]] = [(root, None)]
    entries_seen = 0
    while pending:
        directory, relative = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise IntegrityError(f"Release-Baum kann nicht gelesen werden: {directory}") from exc
        for entry in entries:
            entries_seen += 1
            if entries_seen > MAX_TREE_ENTRIES:
                raise IntegrityError(
                    f"Release-Baum ueberschreitet {MAX_TREE_ENTRIES} Eintraege"
                )
            child = Path(entry.path)
            rel = (PurePosixPath(entry.name) if relative is None
                   else relative / entry.name)
            rel_text = rel.as_posix()
            if len(rel.parts) > MAX_PATH_COMPONENTS:
                raise IntegrityError(
                    f"Release-Baum ist tiefer als {MAX_PATH_COMPONENTS}: {rel_text}"
                )
            if _is_link_like(child):
                raise IntegrityError(
                    f"Symlink/Junction/Reparse-Point im Release-Baum abgewehrt: {rel_text}"
                )
            try:
                if entry.is_dir(follow_symlinks=False):
                    directories.add(rel_text)
                    pending.append((child, rel))
                elif entry.is_file(follow_symlinks=False):
                    files.add(rel_text)
                else:
                    raise IntegrityError(
                        f"Nur regulaere Dateien/Verzeichnisse erlaubt: {rel_text}"
                    )
            except OSError as exc:
                raise IntegrityError(f"Unsicherer Eintrag im Release-Baum: {rel_text}") from exc

    return files, directories


def _expected_inventory(manifest: ReleaseManifest) -> tuple[set[str], set[str]]:
    files = {MANIFEST_FILENAME}
    directories: set[str] = set()
    for artifact in manifest.artifacts:
        files.add(artifact.path)
        parent = PurePosixPath(artifact.path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return files, directories


def _open_regular_readonly(path: Path) -> int:
    if _is_link_like(path):
        raise IntegrityError(f"Symlink/Reparse-Point als Artefakt abgewehrt: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IntegrityError(f"Artefakt kann nicht sicher geoeffnet werden: {path}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise IntegrityError(f"Artefakt ist keine regulaere Datei: {path}")
    return descriptor


def _verify_artifact(root: Path, artifact: ArtifactDigest, *, check_mode: bool) -> None:
    path = root.joinpath(*PurePosixPath(artifact.path).parts)
    descriptor = _open_regular_readonly(path)
    digest = hashlib.sha256()
    total = 0
    try:
        initial = os.fstat(descriptor)
        if initial.st_size != artifact.size:
            raise IntegrityError(
                f"Falsche Groesse fuer {artifact.path!r}: "
                f"{initial.st_size} statt {artifact.size}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                digest.update(block)
        final = os.fstat(descriptor)
        if (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns) != (
                final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            raise IntegrityError(f"Artefakt wurde waehrend der Pruefung veraendert: {artifact.path}")
    finally:
        os.close(descriptor)
    if total != artifact.size:
        raise IntegrityError(f"Falsche gelesene Groesse fuer {artifact.path!r}")
    if not hmac.compare_digest(digest.hexdigest(), artifact.sha256):
        raise IntegrityError(f"SHA-256 stimmt nicht fuer {artifact.path!r}")
    if check_mode and os.name != "nt":
        executable = bool(path.stat().st_mode & 0o111)
        if executable != artifact.executable:
            raise IntegrityError(
                f"Executable-Modus stimmt nicht fuer {artifact.path!r}"
            )


def _verify_tree(root: Path, manifest: ReleaseManifest, *, check_mode: bool) -> None:
    actual_files, actual_directories = _inventory(root)
    expected_files, expected_directories = _expected_inventory(manifest)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise IntegrityError(
            f"Dateiinventar stimmt nicht (fehlend={missing}, unerwartet={extra})"
        )
    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        extra = sorted(actual_directories - expected_directories)
        raise IntegrityError(
            f"Verzeichnisinventar stimmt nicht (fehlend={missing}, unerwartet={extra})"
        )
    for artifact in manifest.artifacts:
        _verify_artifact(root, artifact, check_mode=check_mode)


def verify_bundle(bundle_dir: str | Path) -> ReleaseManifest:
    """Prueft ein lokales Bundle vollstaendig, ohne es zu installieren."""
    bundle = Path(bundle_dir).absolute()
    _ensure_real_directory(bundle, "Bundle-Verzeichnis")
    manifest = ReleaseManifest.load(bundle / MANIFEST_FILENAME)
    _verify_tree(bundle, manifest, check_mode=False)
    return manifest


def _copy_artifact(source_root: Path, target_root: Path, artifact: ArtifactDigest) -> None:
    relative = PurePosixPath(artifact.path)
    source = source_root.joinpath(*relative.parts)
    target = target_root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_fd = _open_regular_readonly(source)
    target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        target_fd = os.open(target, target_flags, 0o600)
    except BaseException:
        os.close(source_fd)
        raise
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(source_fd)
        with os.fdopen(source_fd, "rb", closefd=False) as source_handle:
            with os.fdopen(target_fd, "wb", closefd=False) as target_handle:
                while True:
                    block = source_handle.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    digest.update(block)
                    target_handle.write(block)
                target_handle.flush()
                os.fsync(target_handle.fileno())
        after = os.fstat(source_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise IntegrityError(f"Quelle wurde beim Kopieren veraendert: {artifact.path}")
        if total != artifact.size or before.st_size != artifact.size:
            raise IntegrityError(f"Falsche Groesse beim Kopieren: {artifact.path}")
        if not hmac.compare_digest(digest.hexdigest(), artifact.sha256):
            raise IntegrityError(f"SHA-256 stimmt beim Kopieren nicht: {artifact.path}")
        os.chmod(target, 0o755 if artifact.executable else 0o644)
    finally:
        os.close(source_fd)
        os.close(target_fd)


def _seal_release_tree(root: Path, manifest: ReleaseManifest) -> None:
    """Macht eine fertig geschriebene Generation owner-read-only/betretbar."""
    for artifact in manifest.artifacts:
        path = root.joinpath(*PurePosixPath(artifact.path).parts)
        os.chmod(path, 0o555 if artifact.executable else 0o444)
    os.chmod(root / MANIFEST_FILENAME, 0o444)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        os.chmod(directory, 0o555)
    os.chmod(root, 0o555)


def _json_pairs_unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseStateError(f"Doppelter JSON-Schluessel im State: {key!r}")
        result[key] = value
    return result


def _read_state_file(path: Path) -> ReleaseState:
    try:
        _check_directory_chain(path.parent, "State-Elternpfad")
    except IntegrityError as exc:
        raise ReleaseStateError(str(exc)) from exc
    if not path.exists() and not _is_link_like(path):
        return ReleaseState()
    if _is_link_like(path):
        raise ReleaseStateError(f"State darf kein Symlink/Reparse-Point sein: {path}")
    descriptor = _open_regular_readonly(path)
    try:
        info = os.fstat(descriptor)
        if info.st_size > MAX_STATE_BYTES:
            raise ReleaseStateError("State-Datei ueberschreitet das Groessenlimit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(MAX_STATE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_STATE_BYTES:
        raise ReleaseStateError("State-Datei ueberschreitet das Groessenlimit")
    try:
        text = raw.decode("utf-8")
        data = json.loads(
            text,
            object_pairs_hook=_json_pairs_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReleaseStateError(f"Ungueltiger JSON-Wert im State: {value}")),
        )
    except ReleaseStateError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReleaseStateError(f"State ist kein gueltiges UTF-8-JSON: {path}") from exc
    return ReleaseState.from_dict(data)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_state_file(path: Path, state: ReleaseState) -> None:
    try:
        _check_directory_chain(path.parent, "State-Elternpfad")
    except IntegrityError as exc:
        raise ReleaseStateError(str(exc)) from exc
    if _is_link_like(path):
        raise ReleaseStateError(f"State darf kein Symlink/Reparse-Point sein: {path}")
    payload = (json.dumps(
        state.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        try:
            _check_directory_chain(path.parent, "State-Elternpfad")
        except IntegrityError as exc:
            raise ReleaseStateError(str(exc)) from exc
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _lock_stat_is_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attributes & reparse)


def _validate_lock_identity(descriptor: int, path: Path) -> None:
    """Bindet das offene Handle fail-closed an genau den sicheren Lockpfad."""
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(path)
    except OSError as exc:
        raise ReleaseLockError(f"Lockdatei kann nicht sicher geprueft werden: {path}") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or _lock_stat_is_reparse(named)
        or not stat.S_ISREG(named.st_mode)
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise ReleaseLockError(
            f"Lockdatei wurde ausgetauscht oder ist nicht regulaer: {path}"
        )
    if opened.st_nlink != 1:
        raise ReleaseLockError(
            f"Lockdatei hat unerwartete Hardlinks ({opened.st_nlink}): {path}"
        )
    if os.name != "nt":
        geteuid = getattr(os, "geteuid")
        if opened.st_uid != geteuid():
            raise ReleaseLockError(f"Lockdatei gehoert einem anderen Benutzer: {path}")
        if stat.S_IMODE(opened.st_mode) & 0o077:
            raise ReleaseLockError(
                f"Lockdatei hat zu weite Rechte (erwartet 0600): {path}"
            )


def _precheck_named_lock(path: Path) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ReleaseLockError(f"Lockdatei vor Oeffnen nicht pruefbar: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or _lock_stat_is_reparse(info):
        raise ReleaseLockError(f"Lock-Datei darf kein Symlink/Reparse-Point sein: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseLockError(f"Lockpfad ist keine regulaere Datei: {path}")
    return info


def _open_lock_descriptor(path: Path) -> int:
    """Erzeugt per O_EXCL oder oeffnet nach lstat/fstat-Identitaetspruefung."""
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        _precheck_named_lock(path)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ReleaseLockError(f"Lockdatei kann nicht sicher geoeffnet werden: {path}") from exc
    except OSError as exc:
        # Windows meldet bei einem existierenden Verzeichnis teilweise EACCES
        # statt EEXIST. lstat sorgt trotzdem fuer eine genaue Fail-closed-Diagnose.
        try:
            _precheck_named_lock(path)
        except ReleaseLockError as check_error:
            raise check_error from exc
        raise ReleaseLockError(f"Lockdatei kann nicht sicher angelegt werden: {path}") from exc

    try:
        if created and os.name != "nt":
            getattr(os, "fchmod")(descriptor, 0o600)
        _validate_lock_identity(descriptor, path)
        try:
            _check_directory_chain(path.parent, "Update-Lock-Elternpfad")
        except IntegrityError as exc:
            raise ReleaseLockError(str(exc)) from exc
        _validate_lock_identity(descriptor, path)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


class UpdateLock:
    """Prozessuebergreifender Advisory-Lock (flock bzw. msvcrt.locking)."""

    def __init__(self, path: str | Path, *, timeout: float = 30.0,
                 poll_interval: float = 0.05) -> None:
        timeout = _bounded_timeout(
            timeout,
            "timeout",
            MAX_OPERATION_TIMEOUT_SECONDS,
            allow_zero=True,
        )
        poll_interval = _bounded_timeout(
            poll_interval, "poll_interval", MAX_POLL_INTERVAL_SECONDS
        )
        self.path = Path(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._descriptor: int | None = None

    def _try_lock(self, descriptor: int) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(  # type: ignore[attr-defined]
                    descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise ReleaseLockError(f"Update-Lock fehlgeschlagen: {self.path}") from exc

    def __enter__(self) -> "UpdateLock":
        try:
            _ensure_real_directory(self.path.parent, "Update-Lock-Elternpfad")
        except IntegrityError as exc:
            raise ReleaseLockError(str(exc)) from exc
        descriptor = _open_lock_descriptor(self.path)
        try:
            deadline = time.monotonic() + self.timeout
            while not self._try_lock(descriptor):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ReleaseLockError(
                        f"Update-Lock ist nach {self.timeout:.3f}s noch belegt: {self.path}"
                    )
                time.sleep(min(self.poll_interval, remaining))
            # Ein Austausch nach open, aber vor Lock-Erwerb darf nicht dazu
            # fuehren, dass wir eine unbenannte alte Datei sperren.
            try:
                _check_directory_chain(self.path.parent, "Update-Lock-Elternpfad")
            except IntegrityError as exc:
                raise ReleaseLockError(str(exc)) from exc
            _validate_lock_identity(descriptor, self.path)
            self._descriptor = descriptor
            return self
        except BaseException:
            os.close(descriptor)
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(  # type: ignore[attr-defined]
                    descriptor, fcntl.LOCK_UN  # type: ignore[attr-defined]
                )
        finally:
            os.close(descriptor)


class ReleaseManager:
    """Verwaltet lokale Release-Generationen und atomare Aktivierung."""

    def __init__(self, home: str | Path, *, lock_timeout: float = 30.0) -> None:
        self.home = Path(home).absolute()
        self.releases_dir = self.home / "releases"
        self.state_path = self.home / STATE_FILENAME
        self.lock_path = self.home / LOCK_FILENAME
        self.lock_timeout = _bounded_timeout(
            lock_timeout,
            "lock_timeout",
            MAX_OPERATION_TIMEOUT_SECONDS,
            allow_zero=True,
        )
        _ensure_real_directory(self.home, "Release-Home")
        _ensure_real_directory(self.releases_dir, "Release-Verzeichnis")

    def _lock(self) -> UpdateLock:
        return UpdateLock(self.lock_path, timeout=self.lock_timeout)

    def inspect_bundle(self, bundle_dir: str | Path) -> ReleaseManifest:
        return verify_bundle(bundle_dir)

    def read_state(self) -> ReleaseState:
        return _read_state_file(self.state_path)

    def _release_path(self, release_id: str) -> Path:
        try:
            valid = validate_release_id(release_id)
        except ManifestError as exc:
            raise ReleaseStateError(f"Ungueltige Release-ID: {release_id!r}") from exc
        return self.releases_dir / valid

    def verify(
        self, release_id: str, *, expected_manifest_sha256: str | None = None
    ) -> ReleaseRecord:
        path = self._release_path(release_id)
        if not path.exists() or not path.is_dir() or _is_link_like(path):
            raise IntegrityError(f"Release nicht gefunden oder unsicher: {release_id}")
        try:
            manifest = ReleaseManifest.load(path / MANIFEST_FILENAME)
        except ManifestError as exc:
            raise IntegrityError(f"Ungueltiges Manifest in Release {release_id}") from exc
        if manifest.release_id != release_id:
            raise IntegrityError(
                f"Manifest-ID {manifest.release_id!r} passt nicht zu Verzeichnis {release_id!r}"
            )
        _verify_tree(path, manifest, check_mode=True)
        if expected_manifest_sha256 is not None and not hmac.compare_digest(
            manifest.fingerprint(), expected_manifest_sha256
        ):
            raise IntegrityError(
                f"Manifest-Fingerprint von Release {release_id!r} stimmt nicht mit "
                "dem atomaren Vertrauens-State ueberein"
            )
        return ReleaseRecord(path=path, manifest=manifest)

    def current(self) -> ReleaseRecord | None:
        state = self.read_state()
        return (
            self.verify(
                state.current,
                expected_manifest_sha256=state.current_manifest_sha256,
            )
            if state.current is not None
            else None
        )

    def previous(self) -> ReleaseRecord | None:
        state = self.read_state()
        return (
            self.verify(
                state.previous,
                expected_manifest_sha256=state.previous_manifest_sha256,
            )
            if state.previous is not None
            else None
        )

    def prune(
        self,
        *,
        keep: int = 3,
        apply: bool = False,
        after_delete: Callable[[ReleaseRecord], None] | None = None,
    ) -> dict[str, Any]:
        """Plant/loescht alte verifizierte Generationen unter Update-Lock.

        ``current`` und ``previous`` sind unabhaengig von ``keep`` immer
        geschuetzt. Unbekannte, manipulierte oder linkartige Generationen
        brechen den Lauf vor der ersten Loeschung ab.
        """
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 2:
            raise ReleaseStateError("keep muss eine Ganzzahl >= 2 sein")
        with self._lock():
            state = self.read_state()
            generationen: list[tuple[ReleaseRecord, os.stat_result]] = []
            staging_reste: list[tuple[Path, os.stat_result]] = []
            with os.scandir(self.releases_dir) as entries:
                namen = sorted(entry.name for entry in entries)
            for name in namen:
                if name.startswith(".staging-"):
                    path = self.releases_dir / name
                    if _is_link_like(path) or not path.is_dir():
                        raise IntegrityError(f"Unsicherer Staging-Rest: {path}")
                    staging_reste.append((path, os.lstat(path)))
                    continue
                path = self.releases_dir / name
                if _is_link_like(path) or not path.is_dir():
                    raise IntegrityError(f"Unsicherer Eintrag im Release-Verzeichnis: {path}")
                record = self.verify(name)
                generationen.append((record, os.lstat(path)))

            generationen.sort(
                key=lambda eintrag: (
                    eintrag[1].st_mtime_ns,
                    eintrag[0].manifest.release_id,
                ),
                reverse=True,
            )
            geschuetzt = {release_id for release_id in (state.current, state.previous)
                           if release_id is not None}
            geschuetzt.update(
                record.manifest.release_id for record, _ in generationen[:keep]
            )
            kandidaten = [
                (record, info)
                for record, info in generationen
                if record.manifest.release_id not in geschuetzt
            ]
            geloescht: list[str] = []
            staging_geloescht: list[str] = []

            def beschreibbar_und_nochmal(
                funktion: Callable[..., Any], pfad: str, _exc: Any
            ) -> None:
                kandidat = Path(pfad)
                if kandidat.parent != self.releases_dir:
                    os.chmod(kandidat.parent, 0o700)
                os.chmod(kandidat, 0o700)
                funktion(pfad)

            if apply:
                for record, vorher in kandidaten:
                    # Direkt vor der rekursiven Entfernung erneut das exakte
                    # Manifest/Inventar und die Verzeichnisidentitaet binden.
                    self.verify(record.manifest.release_id)
                    aktuell = os.lstat(record.path)
                    if (aktuell.st_dev, aktuell.st_ino) != (
                        vorher.st_dev,
                        vorher.st_ino,
                    ):
                        raise IntegrityError(
                            f"Release wurde vor Prune ausgetauscht: {record.path}"
                        )

                    shutil.rmtree(record.path, onerror=beschreibbar_und_nochmal)
                    _fsync_directory(self.releases_dir)
                    geloescht.append(record.manifest.release_id)
                    if after_delete is not None:
                        after_delete(record)
                for path, vorher in staging_reste:
                    aktuell = os.lstat(path)
                    if (
                        (aktuell.st_dev, aktuell.st_ino)
                        != (vorher.st_dev, vorher.st_ino)
                        or _is_link_like(path)
                        or not path.is_dir()
                    ):
                        raise IntegrityError(f"Staging-Rest wurde ausgetauscht: {path}")
                    shutil.rmtree(path, onerror=beschreibbar_und_nochmal)
                    staging_geloescht.append(path.name)
                if geloescht or staging_geloescht:
                    _fsync_directory(self.releases_dir)
            return {
                "ok": True,
                "apply": apply,
                "keep": keep,
                "protected": sorted(geschuetzt),
                "candidates": [
                    record.manifest.release_id for record, _ in kandidaten
                ],
                "deleted": geloescht,
                "staging_candidates": [path.name for path, _ in staging_reste],
                "staging_deleted": staging_geloescht,
            }

    def stage(
        self,
        bundle_dir: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> ReleaseRecord:
        with self._lock():
            return self._stage_locked(
                bundle_dir, expected_manifest_sha256=expected_manifest_sha256
            )

    def _stage_locked(
        self,
        bundle_dir: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> ReleaseRecord:
        _ensure_real_directory(self.releases_dir, "Release-Verzeichnis")
        bundle = Path(bundle_dir).absolute()
        manifest = verify_bundle(bundle)
        if expected_manifest_sha256 is not None and not hmac.compare_digest(
            manifest.fingerprint(), expected_manifest_sha256
        ):
            raise IntegrityError(
                "Bundle wurde nach der Trust-Pruefung ausgetauscht oder veraendert"
            )
        final = self._release_path(manifest.release_id)
        with os.scandir(self.releases_dir) as entries:
            for entry in entries:
                if (entry.name.casefold() == manifest.release_id.casefold()
                        and entry.name != manifest.release_id):
                    raise IntegrityError(
                        "Release-ID kollidiert ohne Beachtung der Grossschreibung: "
                        f"{entry.name!r}, {manifest.release_id!r}"
                    )
        if final.exists() or _is_link_like(final):
            if _is_link_like(final) or not final.is_dir():
                raise IntegrityError(f"Release-Ziel ist unsicher: {final}")
            existing = self.verify(manifest.release_id)
            if existing.manifest.fingerprint() != manifest.fingerprint():
                raise IntegrityError(
                    f"Release-ID {manifest.release_id!r} existiert mit anderem Manifest"
                )
            return existing

        temporary = Path(tempfile.mkdtemp(
            dir=str(self.releases_dir), prefix=f".staging-{manifest.release_id}-"
        ))
        try:
            for artifact in manifest.artifacts:
                _copy_artifact(bundle, temporary, artifact)
            manifest.write(temporary / MANIFEST_FILENAME)
            os.chmod(temporary / MANIFEST_FILENAME, 0o644)
            _verify_tree(temporary, manifest, check_mode=True)
            _seal_release_tree(temporary, manifest)
            _verify_tree(temporary, manifest, check_mode=True)
            os.rename(temporary, final)
            _fsync_directory(self.releases_dir)
        except BaseException:
            if temporary.exists() and not _is_link_like(temporary):
                shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.verify(manifest.release_id)

    def apply(
        self,
        bundle_dir: str | Path,
        *,
        backup: BackupHook | None = None,
        candidate_check: ReleaseCheck | None = None,
        activation: ActivationHook | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> ReleaseState:
        """Stage und aktiviert ein lokales Bundle unter demselben Update-Lock."""
        with self._lock():
            record = self._stage_locked(
                bundle_dir,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            return self._promote_locked(
                record.manifest.release_id,
                backup=backup,
                candidate_check=candidate_check,
                activation=activation,
            )

    def promote(
        self,
        release_id: str,
        *,
        backup: BackupHook | None = None,
        candidate_check: ReleaseCheck | None = None,
        activation: ActivationHook | None = None,
    ) -> ReleaseState:
        with self._lock():
            return self._promote_locked(
                release_id,
                backup=backup,
                candidate_check=candidate_check,
                activation=activation,
            )

    def _promote_locked(
        self,
        release_id: str,
        *,
        backup: BackupHook | None,
        candidate_check: ReleaseCheck | None,
        activation: ActivationHook | None,
    ) -> ReleaseState:
        candidate = self.verify(release_id)
        state = self.read_state()
        if state.current == release_id:
            if not hmac.compare_digest(
                candidate.manifest.fingerprint(),
                state.current_manifest_sha256 or "",
            ):
                raise IntegrityError(
                    "Current-Release stimmt nicht mit dem gebundenen State-Manifest ueberein"
                )
            if candidate_check is not None:
                self._run_record_hook("Candidate-Pruefung", candidate_check, candidate)
            return state

        old_record = (
            self.verify(
                state.current,
                expected_manifest_sha256=state.current_manifest_sha256,
            )
            if state.current is not None
            else None
        )
        if (old_record is not None
                and old_record.manifest.schema_version != candidate.manifest.schema_version):
            raise SchemaMismatchError(
                "Update ohne Migration abgelehnt: schema_version "
                f"{old_record.manifest.schema_version!r} -> "
                f"{candidate.manifest.schema_version!r}"
            )
        if candidate_check is not None:
            self._run_record_hook("Candidate-Pruefung", candidate_check, candidate)
        if backup is not None:
            self._run_backup_hook(backup, state, candidate)
        switched = ReleaseState(
            generation=state.generation + 1,
            current=release_id,
            current_manifest_sha256=candidate.manifest.fingerprint(),
            previous=state.current,
            previous_manifest_sha256=state.current_manifest_sha256,
            updated_at=_utc_now(),
        )
        return self._activate_with_compensation(
            old_state=state,
            switched=switched,
            target=candidate,
            recovery=old_record,
            activation=activation,
        )

    def rollback(self, *, activation: ActivationHook | None = None) -> ReleaseState:
        with self._lock():
            state = self.read_state()
            if state.previous is None:
                raise ReleaseStateError("Kein previous-Release fuer Rollback vorhanden")
            target = self.verify(
                state.previous,
                expected_manifest_sha256=state.previous_manifest_sha256,
            )
            old_record = (
                self.verify(
                    state.current,
                    expected_manifest_sha256=state.current_manifest_sha256,
                )
                if state.current is not None
                else None
            )
            if (old_record is not None
                    and old_record.manifest.schema_version != target.manifest.schema_version):
                raise SchemaMismatchError(
                    "Rollback ohne Rueckmigration abgelehnt: schema_version "
                    f"{old_record.manifest.schema_version!r} -> "
                    f"{target.manifest.schema_version!r}"
                )
            switched = ReleaseState(
                generation=state.generation + 1,
                current=target.manifest.release_id,
                current_manifest_sha256=target.manifest.fingerprint(),
                previous=state.current,
                previous_manifest_sha256=state.current_manifest_sha256,
                updated_at=_utc_now(),
            )
            return self._activate_with_compensation(
                old_state=state,
                switched=switched,
                target=target,
                recovery=old_record,
                activation=activation,
            )

    def _activate_with_compensation(
        self,
        *,
        old_state: ReleaseState,
        switched: ReleaseState,
        target: ReleaseRecord,
        recovery: ReleaseRecord | None,
        activation: ActivationHook | None,
    ) -> ReleaseState:
        _write_state_file(self.state_path, switched)
        if activation is None:
            return switched
        try:
            self._run_activation_hook(activation, target)
            return switched
        except BaseException as activation_error:
            compensated = ReleaseState(
                generation=switched.generation + 1,
                current=old_state.current,
                current_manifest_sha256=old_state.current_manifest_sha256,
                previous=old_state.previous,
                previous_manifest_sha256=old_state.previous_manifest_sha256,
                updated_at=_utc_now(),
            )
            try:
                _write_state_file(self.state_path, compensated)
            except BaseException as state_error:
                raise ActivationError(
                    "Aktivierung fehlgeschlagen; State-Rollback ist ebenfalls fehlgeschlagen",
                    rolled_back=False,
                    activation_error=activation_error,
                    recovery_error=state_error,
                ) from activation_error
            try:
                self._run_activation_hook(activation, recovery)
            except BaseException as recovery_error:
                raise ActivationError(
                    "Aktivierung fehlgeschlagen; State wurde kompensiert, externe "
                    "Reaktivierung jedoch nicht",
                    rolled_back=False,
                    activation_error=activation_error,
                    recovery_error=recovery_error,
                ) from activation_error
            raise ActivationError(
                "Aktivierung fehlgeschlagen und wurde vollstaendig zurueckgerollt",
                rolled_back=True,
                activation_error=activation_error,
            ) from activation_error

    @staticmethod
    def _run_record_hook(name: str, hook: ReleaseCheck, record: ReleaseRecord) -> None:
        try:
            result = hook(record)
        except Exception as exc:
            raise ReleaseCheckError(f"{name} warf einen Fehler: {exc}") from exc
        if result is not None and result is not True:
            raise ReleaseCheckError(f"{name} muss True oder None liefern")

    @staticmethod
    def _run_backup_hook(
        hook: BackupHook, state: ReleaseState, candidate: ReleaseRecord
    ) -> None:
        try:
            result = hook(state, candidate)
        except Exception as exc:
            raise ReleaseCheckError(f"Pflicht-Backup warf einen Fehler: {exc}") from exc
        if result is not None and result is not True:
            raise ReleaseCheckError("Pflicht-Backup muss True oder None liefern")

    @staticmethod
    def _run_activation_hook(
        hook: ActivationHook, record: ReleaseRecord | None
    ) -> None:
        try:
            result = hook(record)
        except Exception as exc:
            raise ReleaseCheckError(f"Aktivierung warf einen Fehler: {exc}") from exc
        if result is not None and result is not True:
            raise ReleaseCheckError("Aktivierung muss True oder None liefern")
