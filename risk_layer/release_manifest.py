"""Striktes, portables Manifest fuer lokale Risk-Layer-Releases.

Das Manifest ist absichtlich klein und langweilig: Es beschreibt genau die
Dateien, die zu einer Release-Generation gehoeren. Netzwerktransport,
Signaturen und Registry-Aufloesung sind vorgelagerte Aufgaben. Dadurch kann
derselbe verifizierte Kern native Installationen unter Windows und Linux
bedienen.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


MANIFEST_FILENAME = "release.json"
MANIFEST_FORMAT_VERSION = 1
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ARTIFACTS = 10_000
MAX_FEATURES = 256
MAX_PATH_COMPONENTS = 64
MAX_TREE_ENTRIES = 100_000
# Gross genug fuer heutige lokale GGUF-Modelle, aber klein genug, dass die
# Header-/Manifest-Pruefung einen versehentlichen oder boesartigen Multi-TiB-
# Stage wirklich vor dem ersten Kopieren abweist.
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024 * 1024  # 32 GiB
MAX_RELEASE_BYTES = 128 * 1024 * 1024 * 1024  # 128 GiB

_RELEASE_ID_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?\Z"
)
_GIT_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._+-]{0,63})?\Z")
_FEATURE_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})?\Z")
_WINDOWS_FORBIDDEN = frozenset('<>:"\\|?*')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


class ManifestError(ValueError):
    """Das Release-Manifest ist unvollstaendig, mehrdeutig oder unsicher."""


def _is_link_like(path: Path) -> bool:
    """Erkennt Symlinks und unter Windows auch Junctions/Reparse-Points."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attributes & reparse)


def _secure_file_bytes(path: Path, limit: int) -> bytes:
    """Liest eine regulaere Datei begrenzt und ohne Symlink-Folgen."""
    if _is_link_like(path):
        raise ManifestError(f"Manifest darf kein Symlink/Reparse-Point sein: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise ManifestError(f"Release-Manifest nicht gefunden: {path}") from None
    except OSError as exc:
        raise ManifestError(f"Release-Manifest kann nicht geoeffnet werden: {path}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ManifestError(f"Release-Manifest ist keine regulaere Datei: {path}")
        if info.st_size > limit:
            raise ManifestError(
                f"Release-Manifest ist zu gross ({info.st_size} Bytes; Limit {limit})"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise ManifestError(f"Release-Manifest ueberschreitet das Limit {limit}")
        return data
    finally:
        os.close(descriptor)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"Doppelter JSON-Schluessel im Manifest: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ManifestError(f"Nicht-standardkonstanter JSON-Wert ist unzulaessig: {value}")


def _decode_json(data: bytes, source: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{source}: Manifest ist nicht gueltiges UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except ManifestError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ManifestError(f"{source}: ungueltiges JSON ({exc})") from exc


def _validate_windows_component(component: str, field: str) -> None:
    if component.endswith((".", " ")):
        raise ManifestError(f"{field} endet nicht portabel: {component!r}")
    if any(ord(character) < 32 or character in _WINDOWS_FORBIDDEN
           for character in component):
        raise ManifestError(f"{field} enthaelt unportable Zeichen: {component!r}")
    stem = component.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED:
        raise ManifestError(f"{field} verwendet reservierten Windows-Namen: {component!r}")


def validate_release_id(value: Any) -> str:
    """Validiert eine Release-ID als einzelnes portables Pfadsegment."""
    if not isinstance(value, str) or not _RELEASE_ID_RE.fullmatch(value):
        raise ManifestError(
            "release_id muss 1-128 ASCII-Zeichen lang sein und darf nur "
            "Buchstaben, Ziffern, Punkt, Unterstrich und Bindestrich enthalten"
        )
    _validate_windows_component(value, "release_id")
    return value


def validate_artifact_path(value: Any) -> str:
    """Validiert einen kanonischen, relativen POSIX-Pfad fuer alle Zielsysteme."""
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ManifestError("Artefaktpfad muss eine nicht-leere Zeichenkette sein")
    if "\\" in value:
        raise ManifestError(f"Artefaktpfad muss POSIX-Trenner verwenden: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ManifestError(f"Artefaktpfad ist nicht relativ und kanonisch: {value!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ManifestError(f"Artefaktpfad enthaelt Traversal: {value!r}")
    if len(path.parts) > MAX_PATH_COMPONENTS:
        raise ManifestError(
            f"Artefaktpfad ist tiefer als {MAX_PATH_COMPONENTS} Komponenten: {value!r}"
        )
    for part in path.parts:
        _validate_windows_component(part, "Artefaktpfad")
    if value.casefold() == MANIFEST_FILENAME.casefold():
        raise ManifestError(f"Artefaktpfad ist fuer das Manifest reserviert: {value!r}")
    return value


def _token(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ManifestError(f"{field} hat kein gueltiges portables Versionsformat")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    """Hash- und Groessenbindung eines einzelnen Release-Artefakts."""

    path: str
    sha256: str
    size: int
    executable: bool = False

    def __post_init__(self) -> None:
        validate_artifact_path(self.path)
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ManifestError(f"Ungueltiger SHA-256 fuer {self.path!r}")
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise ManifestError(f"Artefaktgroesse fuer {self.path!r} muss ganzzahlig sein")
        if self.size < 0 or self.size > MAX_ARTIFACT_BYTES:
            raise ManifestError(
                f"Artefaktgroesse fuer {self.path!r} liegt ausserhalb des Limits"
            )
        if not isinstance(self.executable, bool):
            raise ManifestError(f"executable fuer {self.path!r} muss boolesch sein")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size": self.size,
            "executable": self.executable,
        }


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    """Vollstaendige, normalisierte Release-Beschreibung (Formatversion 1)."""

    release_id: str
    git_sha: str
    artifacts: tuple[ArtifactDigest, ...]
    schema_version: str
    api_version: str
    features: tuple[str, ...] = ()
    format_version: int = MANIFEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        validate_release_id(self.release_id)
        if not isinstance(self.git_sha, str) or not _GIT_SHA_RE.fullmatch(self.git_sha):
            raise ManifestError("git_sha muss ein vollstaendiger kleingeschriebener Git-Hash sein")
        if isinstance(self.format_version, bool) or self.format_version != MANIFEST_FORMAT_VERSION:
            raise ManifestError(
                f"Nicht unterstuetzte Manifest-Formatversion: {self.format_version!r}"
            )
        _token(self.schema_version, "schema_version", _VERSION_RE)
        _token(self.api_version, "api_version", _VERSION_RE)
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise ManifestError("artifacts muss mindestens ein Artefakt enthalten")
        if len(self.artifacts) > MAX_ARTIFACTS:
            raise ManifestError(f"Zu viele Artefakte (Limit {MAX_ARTIFACTS})")
        if not isinstance(self.features, tuple):
            raise ManifestError("features muss als Tupel normalisiert sein")
        if len(self.features) > MAX_FEATURES:
            raise ManifestError(f"Zu viele Features (Limit {MAX_FEATURES})")

        paths: set[str] = set()
        folded_paths: list[tuple[tuple[str, ...], str]] = []
        parent_paths: set[tuple[str, ...]] = set()
        total = 0
        for artifact in self.artifacts:
            if not isinstance(artifact, ArtifactDigest):
                raise ManifestError("artifacts enthaelt keinen ArtifactDigest")
            folded = artifact.path.casefold()
            if folded in paths:
                raise ManifestError(
                    f"Artefaktpfad kollidiert ohne Beachtung der Grossschreibung: "
                    f"{artifact.path!r}"
                )
            paths.add(folded)
            folded_paths.append(
                (tuple(part.casefold() for part in PurePosixPath(artifact.path).parts),
                 artifact.path)
            )
            parts = PurePosixPath(artifact.path).parts
            for depth in range(1, len(parts)):
                parent_paths.add(tuple(part.casefold() for part in parts[:depth]))
            total += artifact.size
        if total > MAX_RELEASE_BYTES:
            raise ManifestError(f"Release-Gesamtgroesse ueberschreitet {MAX_RELEASE_BYTES}")
        if len(parent_paths) + len(self.artifacts) + 1 > MAX_TREE_ENTRIES:
            raise ManifestError(
                f"Release-Dateibaum ueberschreitet {MAX_TREE_ENTRIES} Eintraege"
            )

        folded_paths.sort()
        for (left_parts, left), (right_parts, right) in zip(
                folded_paths, folded_paths[1:]):
            if len(left_parts) < len(right_parts) and right_parts[:len(left_parts)] == left_parts:
                raise ManifestError(
                    f"Artefakt kann nicht zugleich Datei und Verzeichnis sein: "
                    f"{left!r}, {right!r}"
                )

        seen_features: set[str] = set()
        for feature in self.features:
            _token(feature, "Feature", _FEATURE_RE)
            folded = feature.casefold()
            if folded in seen_features:
                raise ManifestError(f"Doppeltes Feature: {feature!r}")
            seen_features.add(folded)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseManifest":
        if not isinstance(data, Mapping):
            raise ManifestError("Oberste Manifest-Ebene muss ein JSON-Objekt sein")
        required = {
            "format_version",
            "release_id",
            "git_sha",
            "artifacts",
            "schema_version",
            "api_version",
            "features",
        }
        actual = set(data)
        if actual != required:
            missing = sorted(required - actual)
            unknown = sorted(actual - required)
            details = []
            if missing:
                details.append(f"fehlend: {', '.join(missing)}")
            if unknown:
                details.append(f"unbekannt: {', '.join(unknown)}")
            raise ManifestError("Ungueltige Manifest-Felder (" + "; ".join(details) + ")")

        raw_artifacts = data["artifacts"]
        if not isinstance(raw_artifacts, Mapping):
            raise ManifestError("artifacts muss ein JSON-Objekt sein")
        artifacts: list[ArtifactDigest] = []
        for raw_path, raw_entry in raw_artifacts.items():
            path = validate_artifact_path(raw_path)
            if not isinstance(raw_entry, Mapping):
                raise ManifestError(f"Artefaktbeschreibung fuer {path!r} muss Objekt sein")
            entry_keys = set(raw_entry)
            if not {"sha256", "size"} <= entry_keys <= {
                    "sha256", "size", "executable"}:
                raise ManifestError(f"Ungueltige Felder fuer Artefakt {path!r}")
            artifacts.append(ArtifactDigest(
                path=path,
                sha256=raw_entry["sha256"],
                size=raw_entry["size"],
                executable=raw_entry.get("executable", False),
            ))

        raw_features = data["features"]
        if (not isinstance(raw_features, Sequence)
                or isinstance(raw_features, (str, bytes, bytearray))):
            raise ManifestError("features muss eine JSON-Liste sein")
        features = tuple(sorted(raw_features, key=lambda value: str(value).casefold()))
        artifacts.sort(key=lambda item: item.path.casefold())
        return cls(
            release_id=data["release_id"],
            git_sha=data["git_sha"],
            artifacts=tuple(artifacts),
            schema_version=data["schema_version"],
            api_version=data["api_version"],
            features=features,
            format_version=data["format_version"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReleaseManifest":
        source = Path(path)
        data = _decode_json(_secure_file_bytes(source, MAX_MANIFEST_BYTES), str(source))
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "release_id": self.release_id,
            "git_sha": self.git_sha,
            "schema_version": self.schema_version,
            "api_version": self.api_version,
            "features": list(self.features),
            "artifacts": {
                artifact.path: artifact.to_dict()
                for artifact in sorted(self.artifacts, key=lambda item: item.path.casefold())
            },
        }

    def to_json_bytes(self) -> bytes:
        return (json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n").encode("utf-8")

    def fingerprint(self) -> str:
        """SHA-256 der kanonischen Manifestdarstellung."""
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def write(self, path: str | Path) -> Path:
        """Schreibt das Manifest atomar in dasselbe Dateisystem."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(self.to_json_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return target
