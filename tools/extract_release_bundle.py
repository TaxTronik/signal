#!/usr/bin/env python3
"""Entpacke ein natives Transferarchiv fail-closed und verifiziere das Bundle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from risk_layer.release_lifecycle import verify_bundle
from risk_layer.release_manifest import (
    MANIFEST_FILENAME,
    MAX_ARTIFACTS,
    MAX_MANIFEST_BYTES,
    MAX_RELEASE_BYTES,
    ReleaseManifest,
    validate_artifact_path,
)


def _safe_member_name(name: str) -> str:
    if name == MANIFEST_FILENAME:
        return name
    return validate_artifact_path(name)


def extract(archive_path: Path, destination: Path) -> ReleaseManifest:
    """Extrahiert streaming-basiert und schaltet erst nach Vollpruefung um."""
    if destination.exists():
        raise SystemExit(f"Ziel existiert bereits: {destination}")
    if not archive_path.is_file() or archive_path.is_symlink():
        raise SystemExit(f"kein regulaeres Transferarchiv: {archive_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        # Streaming verhindert, dass ein Angreifer vor dem Mitgliederlimit eine
        # beliebig grosse Liste von Tar-Headern im Speicher materialisiert.
        # Das kanonische Format setzt release.json bewusst an die erste Stelle.
        with tarfile.open(archive_path, mode="r|gz") as archive:
            iterator = iter(archive)
            manifest_member = next(iterator, None)
            if (
                manifest_member is None
                or manifest_member.name != MANIFEST_FILENAME
                or not manifest_member.isfile()
                or manifest_member.size < 0
                or manifest_member.size > MAX_MANIFEST_BYTES
            ):
                raise SystemExit(
                    "release.json muss erstes regulaeres Archivmitglied sein"
                )
            manifest_stream = archive.extractfile(manifest_member)
            if manifest_stream is None:
                raise SystemExit("release.json kann nicht gelesen werden")
            manifest_bytes = manifest_stream.read(MAX_MANIFEST_BYTES + 1)
            if len(manifest_bytes) != manifest_member.size:
                raise SystemExit("release.json ist abgeschnitten")
            manifest_path = stage / MANIFEST_FILENAME
            with manifest_path.open("xb") as output:
                output.write(manifest_bytes)
            os.chmod(manifest_path, 0o644)
            manifest = ReleaseManifest.load(manifest_path)

            expected = {item.path: item for item in manifest.artifacts}
            seen: set[str] = set()
            folded_seen = {MANIFEST_FILENAME.casefold()}
            total = manifest_member.size
            count = 1
            for member in iterator:
                count += 1
                if count > MAX_ARTIFACTS + 1:
                    raise SystemExit("unzulassige Anzahl Archivmitglieder")
                name = _safe_member_name(member.name)
                folded = name.casefold()
                if folded in folded_seen:
                    raise SystemExit(f"doppelter/kollidierender Archivpfad: {name}")
                folded_seen.add(folded)
                if not member.isfile():
                    raise SystemExit(
                        f"nur regulaere Dateien im Transferarchiv erlaubt: {name}"
                    )
                if member.size < 0:
                    raise SystemExit(f"negative Dateigroesse im Archiv: {name}")
                total += member.size
                if total > MAX_RELEASE_BYTES + MAX_MANIFEST_BYTES:
                    raise SystemExit("Transferarchiv ueberschreitet das Groessenlimit")

                artifact = expected.get(name)
                if artifact is None:
                    raise SystemExit(f"unerwartetes Archivmitglied: {name}")
                if member.size != artifact.size:
                    raise SystemExit(
                        f"Headergroesse widerspricht Manifest: {artifact.path}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise SystemExit(f"Archivdatei unlesbar: {artifact.path}")
                relative = Path(*PurePosixPath(artifact.path).parts)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.chmod(target, 0o755 if artifact.executable else 0o644)
                seen.add(name)

            if seen != set(expected):
                missing = sorted(set(expected) - seen)
                raise SystemExit(f"Transferarchiv ist unvollstaendig: {missing}")

        verify_bundle(stage)
        os.replace(stage, destination)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = extract(args.archive, args.destination)
    print(
        f"verifiziert: {manifest.release_id} "
        f"({len(manifest.artifacts)} Artefakte, {manifest.fingerprint()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
