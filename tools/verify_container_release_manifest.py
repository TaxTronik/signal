#!/usr/bin/env python3
"""Prüfe ein Container-Deployment-Manifest und alle lokalen Begleitartefakte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any

from risk_layer.release_manifest import (
    MANIFEST_FILENAME,
    MAX_ARTIFACTS,
    MAX_MANIFEST_BYTES,
    MAX_RELEASE_BYTES,
    ArtifactDigest,
    ReleaseManifest,
    validate_artifact_path,
)
from risk_layer.embedding_model import (
    BGE_M3_REPO_ID,
    BGE_M3_REQUIRED_FILES,
    BGE_M3_REVISION,
    MODEL_MANIFEST_NAME,
    MODEL_MANIFEST_SCHEMA,
)


SCHEMA = "taxtronik-release/v1"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RELEASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FESTWISSEN_MANIFEST = "festwissen-manifest.json"
BGE_MODEL_PREFIX = "models/bge-m3/"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_digest(stream: Any, expected_size: int, label: str) -> str:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(block)
        if size > expected_size:
            raise ValueError(f"Archivdatei größer als manifestiert: {label}")
        digest.update(block)
    if size != expected_size:
        raise ValueError(f"Archivdatei abgeschnitten: {label}")
    return digest.hexdigest()


def _object_unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"doppelter JSON-Schlüssel: {key}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data,
            object_pairs_hook=_object_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"unzulässige JSON-Konstante: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{label} ist kein gültiges JSON: {exc}") from exc


def _safe_tar_members(
    archive: tarfile.TarFile,
    *,
    manifest_name: str,
) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    if not members or len(members) > MAX_ARTIFACTS + 1:
        raise ValueError("unzulässige Anzahl Archivmitglieder")
    by_folded: dict[str, tarfile.TarInfo] = {}
    total = 0
    for member in members:
        name = member.name
        if name != manifest_name:
            validate_artifact_path(name)
        folded = name.casefold()
        if folded in by_folded:
            raise ValueError(f"doppelter/kollidierender Archivpfad: {name}")
        if not member.isfile():
            raise ValueError(f"nur reguläre Dateien im Archiv erlaubt: {name}")
        if member.size < 0:
            raise ValueError(f"negative Dateigröße im Archiv: {name}")
        total += member.size
        if total > MAX_RELEASE_BYTES + MAX_MANIFEST_BYTES:
            raise ValueError("Archiv überschreitet das Größenlimit")
        by_folded[folded] = member
    return by_folded


def _verify_embedding_model_manifest(
    data: bytes,
    artifact_by_path: dict[str, ArtifactDigest],
) -> None:
    document = _load_json_bytes(data, "BGE-M3-Modellmanifest")
    if not isinstance(document, dict) or set(document) != {
        "schema", "repo_id", "revision", "files"
    }:
        raise ValueError("BGE-M3-Modellmanifest hat ungültige Felder")
    if (
        document["schema"] != MODEL_MANIFEST_SCHEMA
        or document["repo_id"] != BGE_M3_REPO_ID
        or document["revision"] != BGE_M3_REVISION
    ):
        raise ValueError("BGE-M3-Modellmanifest bindet Schema/Repository/Revision falsch")
    files = document["files"]
    if not isinstance(files, list) or len(files) != len(BGE_M3_REQUIRED_FILES):
        raise ValueError("BGE-M3-Modellmanifest hat kein vollständiges Pflichtinventar")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise ValueError("ungültiger BGE-M3-Modell-Dateieintrag")
        relative = entry["path"]
        if relative not in BGE_M3_REQUIRED_FILES or relative in seen:
            raise ValueError(f"unerwartete/doppelte BGE-M3-Modell-Datei: {relative!r}")
        outer = artifact_by_path.get(f"{BGE_MODEL_PREFIX}{relative}")
        if (
            outer is None
            or outer.sha256 != entry["sha256"]
            or outer.size != entry["size"]
        ):
            raise ValueError(f"BGE-M3-Modellmanifest widerspricht Festwissen: {relative}")
        seen.add(relative)
    if seen != set(BGE_M3_REQUIRED_FILES):
        raise ValueError("BGE-M3-Modellmanifest fehlen Pflichtdateien")


def _artifact(base: Path, block: Any, expected_keys: set[str]) -> Path:
    if not isinstance(block, dict) or set(block) != expected_keys:
        raise ValueError(f"ungültiger Artefaktblock (erwartet {sorted(expected_keys)})")
    name = block.get("artifact")
    digest = block.get("sha256")
    if (
        not isinstance(name, str)
        or PurePosixPath(name).name != name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(f"ungültiger Artefaktname: {name!r}")
    if not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
        raise ValueError(f"ungültiger Artefakt-Hash: {name!r}")
    path = base / name
    if not path.is_file() or path.is_symlink() or _sha256(path) != digest:
        raise ValueError(f"Artefakt fehlt/abweichend: {name}")
    return path


def _verify_festwissen_archive(
    archive_path: Path,
    manifest_bytes: bytes,
    inner: dict[str, Any],
) -> tuple[ArtifactDigest, ...]:
    if set(inner) != {
        "schema", "release_id", "git_sha", "created_at", "embedding", "files"
    }:
        raise ValueError("Festwissen-Manifest hat fehlende/unbekannte Felder")
    files = inner.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Festwissen-Manifest enthält keine Dateien")
    artifacts: list[ArtifactDigest] = []
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise ValueError("ungültiger Datei-Eintrag im Festwissen-Manifest")
        artifact = ArtifactDigest(
            path=validate_artifact_path(entry["path"]),
            sha256=entry["sha256"],
            size=entry["size"],
        )
        folded = artifact.path.casefold()
        if folded in seen:
            raise ValueError(f"doppelter/kollidierender Festwissen-Pfad: {artifact.path}")
        seen.add(folded)
        artifacts.append(artifact)

    embedding_paths = [
        artifact.path for artifact in artifacts
        if artifact.path.startswith("corpus/embedding/")
    ]
    if bool(embedding_paths) != inner["embedding"]:
        raise ValueError("Festwissen embedding-Flag widerspricht seinem Dateiinventar")
    artifact_by_path = {artifact.path: artifact for artifact in artifacts}
    model_paths = {
        artifact.path for artifact in artifacts
        if artifact.path.startswith(BGE_MODEL_PREFIX)
    }
    expected_model_paths = {
        f"{BGE_MODEL_PREFIX}{relative}" for relative in BGE_M3_REQUIRED_FILES
    } | {f"{BGE_MODEL_PREFIX}{MODEL_MANIFEST_NAME}"}
    if inner["embedding"]:
        if model_paths != expected_model_paths:
            raise ValueError("Embedding-Release hat kein exaktes BGE-M3-Pflichtinventar")
    elif model_paths:
        raise ValueError("Release ohne Embedding enthält unerwartete BGE-M3-Dateien")

    try:
        archive_context = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"Festwissen-Archiv unlesbar: {exc}") from exc
    with archive_context as archive:
        members = _safe_tar_members(archive, manifest_name=FESTWISSEN_MANIFEST)
        expected = {FESTWISSEN_MANIFEST, *(item.path for item in artifacts)}
        if {member.name for member in members.values()} != expected:
            raise ValueError("Festwissen-Tar und Sidecar haben verschiedenes Inventar")
        embedded = members[FESTWISSEN_MANIFEST.casefold()]
        if embedded.size != len(manifest_bytes):
            raise ValueError("eingebettetes Festwissen-Manifest hat falsche Größe")
        stream = archive.extractfile(embedded)
        if stream is None or stream.read(len(manifest_bytes) + 1) != manifest_bytes:
            raise ValueError("eingebettetes Festwissen-Manifest widerspricht dem Sidecar")
        for artifact in artifacts:
            member = members[artifact.path.casefold()]
            if member.name != artifact.path or member.size != artifact.size:
                raise ValueError(f"Festwissen-Tar-Header weicht ab: {artifact.path}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Festwissen-Tar-Datei unlesbar: {artifact.path}")
            if artifact.path == f"{BGE_MODEL_PREFIX}{MODEL_MANIFEST_NAME}":
                model_manifest_bytes = stream.read(artifact.size + 1)
                if (
                    len(model_manifest_bytes) != artifact.size
                    or hashlib.sha256(model_manifest_bytes).hexdigest() != artifact.sha256
                ):
                    raise ValueError("BGE-M3-Modellmanifest weicht im Tar ab")
                _verify_embedding_model_manifest(model_manifest_bytes, artifact_by_path)
            elif _stream_digest(stream, artifact.size, artifact.path) != artifact.sha256:
                raise ValueError(f"Festwissen-Tar-Datei weicht ab: {artifact.path}")
    return tuple(artifacts)


def _verify_native_bundle(
    archive_path: Path,
    *,
    expected_fingerprint: str,
    release_id: str,
    git_sha: str,
    embedding: bool,
    festwissen_manifest_sha256: str,
) -> ReleaseManifest:
    try:
        archive_context = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"natives Bundle unlesbar: {exc}") from exc
    with archive_context as archive:
        members = _safe_tar_members(archive, manifest_name=MANIFEST_FILENAME)
        manifest_member = members.get(MANIFEST_FILENAME.casefold())
        if manifest_member is None or manifest_member.size > MAX_MANIFEST_BYTES:
            raise ValueError("release.json fehlt oder ist zu groß")
        stream = archive.extractfile(manifest_member)
        if stream is None:
            raise ValueError("release.json im nativen Bundle ist unlesbar")
        manifest_bytes = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(manifest_bytes) != manifest_member.size:
            raise ValueError("release.json im nativen Bundle ist abgeschnitten")
        raw = _load_json_bytes(manifest_bytes, "release.json")
        manifest = ReleaseManifest.from_dict(raw)
        if manifest.fingerprint() != expected_fingerprint:
            raise ValueError("kanonischer release.json-Fingerprint weicht ab")
        if manifest.release_id != release_id or manifest.git_sha != git_sha:
            raise ValueError("natives release.json widerspricht Release-ID/Git-SHA")
        if ("embedding" in manifest.features) != embedding:
            raise ValueError("natives release.json widerspricht dem Embedding-Modus")

        expected = {MANIFEST_FILENAME, *(item.path for item in manifest.artifacts)}
        if {member.name for member in members.values()} != expected:
            raise ValueError("natives Bundle und release.json haben verschiedenes Inventar")
        metadata_seen = False
        for artifact in manifest.artifacts:
            member = members[artifact.path.casefold()]
            if member.name != artifact.path or member.size != artifact.size:
                raise ValueError(f"nativer Tar-Header weicht ab: {artifact.path}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"native Bundle-Datei unlesbar: {artifact.path}")
            digest = _stream_digest(stream, artifact.size, artifact.path)
            if digest != artifact.sha256:
                raise ValueError(f"native Bundle-Datei weicht ab: {artifact.path}")
            if artifact.path == "metadata/festwissen-manifest.json":
                metadata_seen = True
                if digest != festwissen_manifest_sha256:
                    raise ValueError(
                        "natives und externes Festwissen-Manifest sind nicht identisch"
                    )
        if not metadata_seen:
            raise ValueError("Festwissen-Metadaten fehlen im nativen Bundle")
    return manifest


def verify(path: Path, artifact_dir: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"kein reguläres Deployment-Manifest: {path}")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"unzulässige JSON-Konstante: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Deployment-Manifest unlesbar: {exc}") from exc
    expected_top = {
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
    if not isinstance(document, dict) or set(document) not in (
        expected_top,
        expected_top | {"llm_model"},
    ):
        raise ValueError("Deployment-Manifest hat fehlende/unbekannte Top-Level-Felder")
    if document["schema"] != SCHEMA:
        raise ValueError("unbekanntes Deployment-Manifest-Schema")
    if not isinstance(document["release_id"], str) or not RELEASE_ID_RE.fullmatch(
        document["release_id"]
    ):
        raise ValueError("release_id fehlt/ist ungültig")
    if not isinstance(document["git_sha"], str) or not GIT_SHA_RE.fullmatch(document["git_sha"]):
        raise ValueError("git_sha ist ungültig")
    if not isinstance(document["created_at"], str) or not document["created_at"]:
        raise ValueError("created_at fehlt")

    engine = document["engine"]
    if not isinstance(engine, dict) or set(engine) != {"repository", "digest", "image"}:
        raise ValueError("engine-Block ist ungültig")
    if not isinstance(engine["repository"], str) or not engine["repository"]:
        raise ValueError("Engine-Repository fehlt")
    if not isinstance(engine["digest"], str) or not DIGEST_RE.fullmatch(engine["digest"]):
        raise ValueError("Engine-Digest ist ungültig")
    if engine["image"] != f"{engine['repository']}@{engine['digest']}":
        raise ValueError("Engine-Image stimmt nicht mit Repository/Digest überein")

    festwissen = document["festwissen"]
    festwissen_path = _artifact(
        artifact_dir,
        festwissen,
        {"artifact", "sha256", "manifest", "manifest_sha256", "embedding"},
    )
    if not isinstance(festwissen["embedding"], bool):
        raise ValueError("festwissen.embedding muss boolesch sein")
    manifest_name = festwissen["manifest"]
    if (
        not isinstance(manifest_name, str)
        or PurePosixPath(manifest_name).name != manifest_name
        or manifest_name in {".", ".."}
        or "\\" in manifest_name
    ):
        raise ValueError("ungültiger Festwissen-Manifestname")
    if not isinstance(festwissen["manifest_sha256"], str) or not SHA_RE.fullmatch(
        festwissen["manifest_sha256"]
    ):
        raise ValueError("ungültiger Festwissen-Manifest-Hash")
    inner_path = artifact_dir / manifest_name
    if (
        not inner_path.is_file()
        or inner_path.is_symlink()
        or _sha256(inner_path) != festwissen["manifest_sha256"]
    ):
        raise ValueError("Festwissen-Manifest fehlt/abweichend")
    manifest_bytes = inner_path.read_bytes()
    inner = _load_json_bytes(manifest_bytes, "Festwissen-Manifest")
    if not isinstance(inner, dict) or not isinstance(inner.get("embedding"), bool):
        raise ValueError("Festwissen-Manifest ist strukturell ungültig")
    if (
        inner.get("schema") != "taxtronik.festwissen-release/v1"
        or inner.get("release_id") != document["release_id"]
        or inner.get("git_sha") != document["git_sha"]
        or (inner.get("embedding") is True) != festwissen["embedding"]
    ):
        raise ValueError("inneres Festwissen-Manifest widerspricht dem Deployment")
    _verify_festwissen_archive(festwissen_path, manifest_bytes, inner)

    native_path = _artifact(
        artifact_dir,
        document["native_bundle"],
        {"artifact", "sha256", "manifest_sha256"},
    )
    native_manifest_hash = document["native_bundle"]["manifest_sha256"]
    if not isinstance(native_manifest_hash, str) or not SHA_RE.fullmatch(native_manifest_hash):
        raise ValueError("kanonischer nativer Manifest-Hash ist ungültig")
    _verify_native_bundle(
        native_path,
        expected_fingerprint=native_manifest_hash,
        release_id=document["release_id"],
        git_sha=document["git_sha"],
        embedding=festwissen["embedding"],
        festwissen_manifest_sha256=festwissen["manifest_sha256"],
    )
    _artifact(artifact_dir, document["compose"], {"artifact", "sha256"})
    bootstrap = document["bootstrap"]
    if not isinstance(bootstrap, dict) or set(bootstrap) != BOOTSTRAP_KEYS:
        raise ValueError("bootstrap-Block ist unvollständig/ungültig")
    seen_bootstrap_names: set[str] = set()
    for name in sorted(BOOTSTRAP_KEYS):
        path = _artifact(artifact_dir, bootstrap[name], {"artifact", "sha256"})
        if path.name.casefold() in seen_bootstrap_names:
            raise ValueError("Bootstrap-Artefaktnamen kollidieren")
        seen_bootstrap_names.add(path.name.casefold())
    if "llm_model" in document:
        llm = document["llm_model"]
        _artifact(
            artifact_dir,
            llm,
            {"artifact", "sha256", "revision", "provenance"},
        )
        for field in ("revision", "provenance"):
            value = llm[field]
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 2048
                or value.strip() != value
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"llm_model.{field} ist ungültig")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir or args.manifest.parent
    document = verify(args.manifest, artifact_dir)
    print(
        f"verifiziert: {document['release_id']} "
        f"{document['engine']['image']} manifest_sha256={_sha256(args.manifest)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"Deployment-Manifest abgelehnt: {exc}") from exc
