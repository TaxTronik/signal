#!/usr/bin/env python3
"""Baue ein natives Release-Bundle im kanonischen ``release.json``-Format."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from typing import Any

from risk_layer.release_manifest import (
    ArtifactDigest,
    MAX_ARTIFACTS,
    MAX_RELEASE_BYTES,
    ReleaseManifest,
    validate_artifact_path,
)
from risk_layer.api_v1 import V1_VERSION


FESTWISSEN_SCHEMA = "taxtronik.festwissen-release/v1"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HOST_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_]*-cp[0-9]{2,3}$")
DIRECT_REQUIREMENT_URL_RE = re.compile(r"\s@\s*(?:https?|file):", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_regular(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"keine reguläre Quelldatei: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _object_unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"doppelter JSON-Schlüssel: {key}")
        result[key] = value
    return result


def _read_festwissen_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"kein reguläres Festwissen-Manifest: {path}")
    data = path.read_bytes()
    try:
        document = json.loads(
            data,
            object_pairs_hook=_object_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"unzulässige JSON-Konstante: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Festwissen-Manifest unlesbar: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit("Festwissen-Manifest muss ein JSON-Objekt sein")
    return document, data


def _copy_festwissen_archive(
    archive_path: Path,
    manifest_bytes: bytes,
    entries: list[ArtifactDigest],
    stage: Path,
) -> list[ArtifactDigest]:
    """Kopiere ausschließlich das exakt manifestierte Festwissen aus dem Tar."""
    if not archive_path.is_file() or archive_path.is_symlink():
        raise SystemExit(f"kein reguläres Festwissen-Archiv: {archive_path}")
    try:
        archive_context = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise SystemExit(f"Festwissen-Archiv unlesbar: {exc}") from exc

    with archive_context as archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_ARTIFACTS + 1:
            raise SystemExit("unzulässige Anzahl Dateien im Festwissen-Archiv")
        by_name: dict[str, tarfile.TarInfo] = {}
        for member in members:
            name = member.name
            if name != "festwissen-manifest.json":
                validate_artifact_path(name)
            folded = name.casefold()
            if folded in by_name:
                raise SystemExit(f"doppelter/kollidierender Archivpfad: {name}")
            if not member.isfile():
                raise SystemExit(f"nur reguläre Dateien im Festwissen-Archiv erlaubt: {name}")
            by_name[folded] = member

        expected_names = {"festwissen-manifest.json", *(item.path for item in entries)}
        if {member.name for member in members} != expected_names:
            raise SystemExit("Festwissen-Archiv und Sidecar haben verschiedenes Inventar")
        embedded_member = by_name["festwissen-manifest.json"]
        if embedded_member.size != len(manifest_bytes):
            raise SystemExit("eingebettetes Festwissen-Manifest hat falsche Größe")
        embedded_stream = archive.extractfile(embedded_member)
        if embedded_stream is None or embedded_stream.read(len(manifest_bytes) + 1) != manifest_bytes:
            raise SystemExit("eingebettetes Festwissen-Manifest widerspricht dem Sidecar")

        copied: list[ArtifactDigest] = []
        total = 0
        for artifact in entries:
            member = by_name[artifact.path.casefold()]
            if member.name != artifact.path or member.size != artifact.size:
                raise SystemExit(f"Archiv-Header widerspricht Manifest: {artifact.path}")
            total += member.size
            if total > MAX_RELEASE_BYTES:
                raise SystemExit("Festwissen-Archiv überschreitet das Größenlimit")
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"Archivdatei unlesbar: {artifact.path}")
            relative = Path(*PurePosixPath(artifact.path).parts)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    size += len(block)
                    if size > artifact.size:
                        raise SystemExit(f"Archivdatei größer als manifestiert: {artifact.path}")
                    digest.update(block)
                    output.write(block)
            if size != artifact.size or digest.hexdigest() != artifact.sha256:
                raise SystemExit(f"Festwissen-Datei weicht vom Manifest ab: {artifact.path}")
            copied.append(artifact)
    return copied


def _copy_runtime(
    *,
    runtime_locks: list[tuple[str, Path]],
    wheelhouses: list[tuple[str, Path]],
    stage: Path,
) -> list[ArtifactDigest]:
    artifacts: list[ArtifactDigest] = []

    def mapping(values: list[tuple[str, Path]], label: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for host_tag, path in values:
            if not HOST_TAG_RE.fullmatch(host_tag):
                raise SystemExit(f"ungültiger Runtime-Hosttag: {host_tag!r}")
            if host_tag in result:
                raise SystemExit(f"doppeltes {label} für Hosttag: {host_tag}")
            result[host_tag] = path
        return result

    locks = mapping(runtime_locks, "Runtime-Lock")
    houses = mapping(wheelhouses, "Runtime-Wheelhouse")
    if not locks or set(locks) != set(houses):
        raise SystemExit("Runtime-Locks und Wheelhouses müssen dieselben Hosttags abdecken")

    for host_tag in sorted(locks):
        runtime_lock = locks[host_tag]
        if not runtime_lock.is_file() or runtime_lock.is_symlink():
            raise SystemExit(f"keine reguläre Runtime-Lockdatei: {runtime_lock}")
        try:
            lock_text = runtime_lock.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f"Runtime-Lockdatei unlesbar: {exc}") from exc
        clean_lines: list[str] = []
        for line in lock_text.splitlines():
            stripped = line.strip()
            if stripped == "--extra-index-url https://download.pytorch.org/whl/cpu":
                continue
            if stripped.startswith((
                "--index-url", "--extra-index-url", "--find-links", "--trusted-host"
            )):
                raise SystemExit(f"unerlaubte Quelle in Runtime-Lockdatei: {stripped}")
            if DIRECT_REQUIREMENT_URL_RE.search(stripped):
                raise SystemExit(f"direkte URL in Runtime-Lockdatei: {stripped}")
            clean_lines.append(line)
        clean_lock = ("\n".join(clean_lines).rstrip("\n") + "\n").encode("utf-8")
        lock_target = Path("runtime/locks") / f"{host_tag}.txt"
        (stage / lock_target).parent.mkdir(parents=True, exist_ok=True)
        (stage / lock_target).write_bytes(clean_lock)
        artifacts.append(ArtifactDigest(
            path=lock_target.as_posix(),
            sha256=_sha256(stage / lock_target),
            size=(stage / lock_target).stat().st_size,
        ))

        directory = houses[host_tag]
        if not HOST_TAG_RE.fullmatch(host_tag):
            raise SystemExit(f"ungültiger Runtime-Hosttag: {host_tag!r}")
        if not directory.is_dir() or directory.is_symlink():
            raise SystemExit(f"Wheelhouse fehlt/ist kein reguläres Verzeichnis: {directory}")
        entries = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
        if not entries:
            raise SystemExit(f"Wheelhouse ist leer: {directory}")
        for wheel in entries:
            if not wheel.is_file() or wheel.is_symlink() or wheel.suffix != ".whl":
                raise SystemExit(f"Wheelhouse enthält keine reguläre .whl-Datei: {wheel}")
            relative = Path("runtime/wheelhouse") / host_tag / wheel.name
            validate_artifact_path(relative.as_posix())
            _copy_regular(wheel, stage / relative)
            artifacts.append(ArtifactDigest(
                path=relative.as_posix(),
                sha256=_sha256(stage / relative),
                size=(stage / relative).stat().st_size,
            ))
    return artifacts


def _normalise(info: tarfile.TarInfo, epoch: int) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = epoch
    info.mode = 0o644
    info.pax_headers = {}
    return info


def _archive_bundle(bundle: Path, archive: Path, epoch: int, manifest: ReleaseManifest) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{archive.name}.", dir=archive.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    members = [Path("release.json"), *(Path(item.path) for item in manifest.artifacts)]
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for relative in members:
                        source = bundle / relative
                        info = tar.gettarinfo(str(source), arcname=relative.as_posix())
                        with source.open("rb") as stream:
                            tar.addfile(_normalise(info, epoch), stream)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = Path(f"{archive}.sha256")
    sidecar.write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="ascii")


def build(
    *,
    release_id: str,
    git_sha: str,
    wheel: Path,
    runtime_locks: list[tuple[str, Path]],
    runtime_wheelhouses: list[tuple[str, Path]],
    festwissen_archive: Path,
    festwissen_manifest_path: Path,
    output_dir: Path,
    archive: Path | None,
    epoch: int,
    schema_version: str,
    api_version: str,
) -> ReleaseManifest:
    if not GIT_SHA_RE.fullmatch(git_sha):
        raise SystemExit("--git-sha muss ein voller lowercase Git-SHA sein")
    if output_dir.exists():
        raise SystemExit(f"Zielverzeichnis existiert bereits: {output_dir}")
    if not wheel.name.endswith(".whl"):
        raise SystemExit("--wheel muss eine .whl-Datei sein")

    festwissen, festwissen_manifest_bytes = _read_festwissen_manifest(
        festwissen_manifest_path
    )
    if festwissen.get("schema") != FESTWISSEN_SCHEMA:
        raise SystemExit("unerwartetes Festwissen-Manifest-Schema")
    if festwissen.get("release_id") != release_id or festwissen.get("git_sha") != git_sha:
        raise SystemExit(
            "Festwissen-Manifest und Bundle haben verschiedene Release-ID/Git-SHA"
        )
    if set(festwissen) != {
        "schema", "release_id", "git_sha", "created_at", "embedding", "files"
    }:
        raise SystemExit("Festwissen-Manifest hat fehlende/unbekannte Felder")
    if not isinstance(festwissen.get("embedding"), bool):
        raise SystemExit("Festwissen-Manifest enthält keinen booleschen embedding-Wert")
    raw_files = festwissen.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise SystemExit("Festwissen-Manifest enthält keine Dateien")

    festwissen_artifacts: list[ArtifactDigest] = []
    seen: set[str] = set()
    for entry in raw_files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise SystemExit("ungültiger Datei-Eintrag im Festwissen-Manifest")
        posix = validate_artifact_path(entry["path"])
        artifact = ArtifactDigest(
            path=posix,
            sha256=entry["sha256"],
            size=entry["size"],
        )
        folded = posix.casefold()
        if folded in seen:
            raise SystemExit(f"doppelter/kollidierender Festwissen-Pfad: {posix}")
        seen.add(folded)
        festwissen_artifacts.append(artifact)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    artifacts: list[ArtifactDigest] = []
    try:
        wheel_target = Path("app") / wheel.name
        _copy_regular(wheel, stage / wheel_target)
        artifacts.append(ArtifactDigest(
            path=wheel_target.as_posix(),
            sha256=_sha256(stage / wheel_target),
            size=(stage / wheel_target).stat().st_size,
        ))

        artifacts.extend(_copy_runtime(
            runtime_locks=runtime_locks,
            wheelhouses=runtime_wheelhouses,
            stage=stage,
        ))
        artifacts.extend(_copy_festwissen_archive(
            festwissen_archive,
            festwissen_manifest_bytes,
            festwissen_artifacts,
            stage,
        ))

        metadata_target = Path("metadata/festwissen-manifest.json")
        _copy_regular(festwissen_manifest_path, stage / metadata_target)
        artifacts.append(ArtifactDigest(
            path=metadata_target.as_posix(),
            sha256=_sha256(stage / metadata_target),
            size=(stage / metadata_target).stat().st_size,
        ))

        features = ["festwissen"]
        if festwissen.get("embedding") is True:
            features.append("embedding")
        # Gleiche kanonische Reihenfolge in release.json und Tar: Der Verifier
        # kann große gzip-Bundles dadurch sequentiell statt mit teuren
        # Rückwärts-Seeks über mehrere GiB lesen.
        artifacts.sort(key=lambda item: item.path.casefold())
        manifest = ReleaseManifest(
            release_id=release_id,
            git_sha=git_sha,
            schema_version=schema_version,
            api_version=api_version,
            features=tuple(sorted(features, key=str.casefold)),
            artifacts=tuple(artifacts),
        )
        manifest.write(stage / "release.json")
        # Roundtrip durch denselben strikten Parser, den der Installer benutzt.
        manifest = ReleaseManifest.load(stage / "release.json")
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    if archive is not None:
        _archive_bundle(output_dir, archive, epoch, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--runtime-lock",
        action="append",
        required=True,
        metavar="HOSTTAG=DATEI",
        help="wiederholbar; wird bereinigt als runtime/locks/HOSTTAG.txt manifestiert",
    )
    parser.add_argument(
        "--runtime-wheelhouse",
        action="append",
        required=True,
        metavar="HOSTTAG=VERZEICHNIS",
        help="wiederholbar, z. B. linux_x86_64-cp312=dist/wheelhouse/linux_x86_64-cp312",
    )
    parser.add_argument("--festwissen-archive", type=Path, required=True)
    parser.add_argument("--festwissen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--schema-version", default="1")
    parser.add_argument("--api-version", default=V1_VERSION)
    args = parser.parse_args()
    def parse_mapping(values: list[str], option: str) -> list[tuple[str, Path]]:
        result: list[tuple[str, Path]] = []
        for value in values:
            host_tag, separator, path = value.partition("=")
            if not separator or not path:
                parser.error(f"{option} erwartet HOSTTAG=PFAD")
            result.append((host_tag, Path(path)))
        return result

    locks = parse_mapping(args.runtime_lock, "--runtime-lock")
    wheelhouses = parse_mapping(args.runtime_wheelhouse, "--runtime-wheelhouse")
    build(
        release_id=args.release_id,
        git_sha=args.git_sha,
        wheel=args.wheel,
        runtime_locks=locks,
        runtime_wheelhouses=wheelhouses,
        festwissen_archive=args.festwissen_archive,
        festwissen_manifest_path=args.festwissen_manifest,
        output_dir=args.output_dir,
        archive=args.archive,
        epoch=args.source_date_epoch,
        schema_version=args.schema_version,
        api_version=args.api_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
