#!/usr/bin/env python3
"""Erzeuge ein reproduzierbares, selbstbeschreibendes Festwissen-Release.

Das Archiv ist mandantenfrei. Es enthält ausschließlich die gebauten Korpus-,
Graph-, Trigger-, Katalog- und optionalen Embedding-Dateien. Metadaten des
Dateisystems werden normalisiert; SOURCE_DATE_EPOCH steuert den Zeitstempel.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from datetime import datetime, timezone

from risk_layer.embedding_model import validate_local_bge_m3


SCHEMA = "taxtronik.festwissen-release/v1"
CORE_PATHS = (
    Path("corpus/gesetze.yaml"),
    Path("corpus/graph.sqlite"),
    Path("corpus/trigger.sqlite"),
    Path("corpus/norm_manifest.json"),
    Path("catalog/begriffe.yaml"),
)
RELEASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(source_root: Path, include_embedding: bool) -> list[tuple[Path, Path]]:
    """Liefere ``(Manifestpfad, Quelldatei)`` aus genau einer Build-Wurzel."""
    result: list[tuple[Path, Path]] = []
    for relative in CORE_PATHS:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"Pflichtdatei fehlt oder ist kein reguläres File: {path}")
        result.append((relative, path))

    if include_embedding:
        embedding_relative = Path("corpus/embedding")
        embedding = source_root / embedding_relative
        if not embedding.is_dir() or embedding.is_symlink():
            raise SystemExit("--embedding gesetzt, aber corpus/embedding fehlt")
        for path in embedding.rglob("*"):
            if path.is_symlink():
                raise SystemExit(f"Symlink im Release-Inhalt verweigert: {path}")
            if path.is_file():
                result.append((path.relative_to(source_root), path))
            elif not path.is_dir():
                raise SystemExit(f"Nicht-regulärer Release-Inhalt verweigert: {path}")
        if not any(relative.is_relative_to(embedding_relative) for relative, _ in result):
            raise SystemExit("corpus/embedding enthält keine Dateien")

        model_relative = Path("models/bge-m3")
        model = source_root / model_relative
        try:
            validate_local_bge_m3(model, verify_hashes=True)
        except ValueError as exc:
            raise SystemExit(f"lokales BGE-M3-Modell abgelehnt: {exc}") from exc
        for path in model.rglob("*"):
            if path.is_symlink():
                raise SystemExit(f"Symlink im Modell-Release verweigert: {path}")
            if path.is_file():
                result.append((path.relative_to(source_root), path))
            elif not path.is_dir():
                raise SystemExit(f"Nicht-regulärer Modellinhalt verweigert: {path}")

    return sorted(result, key=lambda item: item[0].as_posix().encode("utf-8"))


def _normalise(info: tarfile.TarInfo, epoch: int) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = epoch
    info.mode = 0o644
    info.pax_headers = {}
    return info


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def package(
    *,
    release_id: str,
    git_sha: str,
    epoch: int,
    output: Path,
    include_embedding: bool,
    source_root: Path = Path("."),
) -> None:
    if not RELEASE_ID_RE.fullmatch(release_id):
        raise SystemExit("ungültige Release-ID (erlaubt: A-Z, a-z, 0-9, Punkt, _ und -)")
    if git_sha != "unknown" and not GIT_SHA_RE.fullmatch(git_sha):
        raise SystemExit("--git-sha muss 'unknown' oder ein voller lowercase Git-SHA sein")
    if epoch < 0:
        raise SystemExit("--source-date-epoch muss >= 0 sein")

    if not source_root.is_dir() or source_root.is_symlink():
        raise SystemExit(f"--source-root ist kein reguläres Verzeichnis: {source_root}")
    files = _files(source_root, include_embedding)
    file_entries = [
        {
            "path": PurePosixPath(relative.as_posix()).as_posix(),
            "sha256": _sha256(source),
            "size": source.stat().st_size,
        }
        for relative, source in files
    ]
    manifest = {
        "schema": SCHEMA,
        "release_id": release_id,
        "git_sha": git_sha,
        "created_at": datetime.fromtimestamp(epoch, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "embedding": include_embedding,
        "files": file_entries,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            # filename="" verhindert, dass der temporäre Zufallsname im
            # Gzip-Header landet. mtime macht denselben Input byte-identisch.
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    manifest_info = tarfile.TarInfo("festwissen-manifest.json")
                    manifest_info.size = len(manifest_bytes)
                    archive.addfile(_normalise(manifest_info, epoch), io.BytesIO(manifest_bytes))

                    for (relative, source), expected in zip(files, file_entries, strict=True):
                        # Zwischen Hash-Pass und Archiv-Pass veränderte Quellen
                        # dürfen keinen inkonsistenten Snapshot ergeben.
                        if (source.stat().st_size != expected["size"]
                                or _sha256(source) != expected["sha256"]):
                            raise SystemExit(
                                f"Datei während des Paketierens verändert: {source}"
                            )
                        info = archive.gettarinfo(str(source), arcname=relative.as_posix())
                        with source.open("rb") as stream:
                            archive.addfile(_normalise(info, epoch), stream)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    _write_atomic(Path(f"{output}.manifest.json"), manifest_bytes)
    archive_hash = _sha256(output)
    _write_atomic(
        Path(f"{output}.sha256"),
        f"{archive_hash}  {output.name}\n".encode("ascii"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("."),
        help="frische Build-Wurzel mit corpus/ und catalog/ (Default: aktuelles Verzeichnis)",
    )
    args = parser.parse_args()
    package(
        release_id=args.release_id,
        git_sha=args.git_sha,
        epoch=args.source_date_epoch,
        output=args.output,
        include_embedding=args.embedding,
        source_root=args.source_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
