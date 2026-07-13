"""Strikter Offline-Vertrag für das produktive BGE-M3-Modellartefakt."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any


BGE_M3_REPO_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_MANIFEST_SCHEMA = "taxtronik.embedding-model/v1"
MODEL_MANIFEST_NAME = "model-manifest.json"

# Exaktes PyTorch/Sentence-Transformers-Inventar am gepinnten Commit. Bilder,
# Repository-Metadaten und insbesondere die alternative 2,27-GiB-ONNX-Kopie
# gehören nicht in das Release. colbert/sparse sind kleine, commitgebundene
# BGE-M3-Head-Dateien und bleiben für spätere Betriebsmodi mitgebunden.
BGE_M3_REQUIRED_FILES = (
    "1_Pooling/config.json",
    "colbert_linear.pt",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "sparse_linear.pt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class EmbeddingModelError(ValueError):
    """Das lokale Modell ist unvollständig, mehrdeutig oder nicht gebunden."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EmbeddingModelError(f"doppelter JSON-Schlüssel: {key}")
        result[key] = value
    return result


def _regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and not path.is_symlink()


def build_model_manifest(model_dir: Path) -> dict[str, Any]:
    """Erzeuge das kanonische Manifest für ein bereits isoliertes Inventar."""
    expected = set(BGE_M3_REQUIRED_FILES)
    actual = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and path.name != MODEL_MANIFEST_NAME
    }
    if actual != expected:
        raise EmbeddingModelError(
            "BGE-M3-Inventar weicht ab "
            f"(fehlend={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )
    entries = []
    for relative in BGE_M3_REQUIRED_FILES:
        source = model_dir / Path(*PurePosixPath(relative).parts)
        if not _regular_file(source):
            raise EmbeddingModelError(f"keine reguläre Modelldatei: {relative}")
        entries.append({
            "path": relative,
            "sha256": _sha256(source),
            "size": source.stat().st_size,
        })
    return {
        "schema": MODEL_MANIFEST_SCHEMA,
        "repo_id": BGE_M3_REPO_ID,
        "revision": BGE_M3_REVISION,
        "files": entries,
    }


def write_model_manifest(model_dir: Path) -> Path:
    document = build_model_manifest(model_dir)
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    target = model_dir / MODEL_MANIFEST_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MODEL_MANIFEST_NAME}.", dir=model_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_local_bge_m3(model_dir: str | Path, *, verify_hashes: bool = True) -> Path:
    """Prüfe Revision, Pflichtinventar, reguläre Dateien und optional alle Hashes."""
    root = Path(model_dir)
    if not root.is_dir() or root.is_symlink():
        raise EmbeddingModelError(f"lokales BGE-M3-Verzeichnis fehlt: {root}")
    manifest_path = root / MODEL_MANIFEST_NAME
    if not _regular_file(manifest_path):
        raise EmbeddingModelError(f"BGE-M3-Modellmanifest fehlt: {manifest_path}")
    try:
        document = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EmbeddingModelError(f"unzulässige JSON-Konstante: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingModelError(f"BGE-M3-Modellmanifest unlesbar: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema", "repo_id", "revision", "files"
    }:
        raise EmbeddingModelError("BGE-M3-Modellmanifest hat ungültige Felder")
    if document["schema"] != MODEL_MANIFEST_SCHEMA:
        raise EmbeddingModelError("unbekanntes BGE-M3-Modellmanifest-Schema")
    if document["repo_id"] != BGE_M3_REPO_ID:
        raise EmbeddingModelError("BGE-M3-Modellmanifest bindet das falsche Repository")
    if document["revision"] != BGE_M3_REVISION:
        raise EmbeddingModelError("BGE-M3-Modellmanifest bindet die falsche Revision")
    entries = document["files"]
    if not isinstance(entries, list):
        raise EmbeddingModelError("BGE-M3-files muss eine Liste sein")

    expected = set(BGE_M3_REQUIRED_FILES)
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise EmbeddingModelError("ungültiger BGE-M3-Dateieintrag")
        path = entry["path"]
        if path not in expected or path in seen:
            raise EmbeddingModelError(f"unerwartete/doppelte BGE-M3-Datei: {path!r}")
        if not isinstance(entry["sha256"], str) or not _SHA_RE.fullmatch(entry["sha256"]):
            raise EmbeddingModelError(f"ungültiger BGE-M3-Hash: {path}")
        if (
            isinstance(entry["size"], bool)
            or not isinstance(entry["size"], int)
            or entry["size"] < 0
        ):
            raise EmbeddingModelError(f"ungültige BGE-M3-Größe: {path}")
        source = root / Path(*PurePosixPath(path).parts)
        if not _regular_file(source) or source.stat().st_size != entry["size"]:
            raise EmbeddingModelError(f"BGE-M3-Datei fehlt/hat falsche Größe: {path}")
        if verify_hashes and _sha256(source) != entry["sha256"]:
            raise EmbeddingModelError(f"BGE-M3-Datei hat falschen Hash: {path}")
        seen.add(path)
    if seen != expected:
        raise EmbeddingModelError(f"BGE-M3-Pflichtdateien fehlen: {sorted(expected - seen)}")

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise EmbeddingModelError(f"Link/Sonderdatei im BGE-M3-Modell: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    allowed = expected | {MODEL_MANIFEST_NAME}
    if actual_files != allowed:
        raise EmbeddingModelError(
            "BGE-M3-Verzeichnis enthält fehlende/unerlaubte Dateien "
            f"(fehlend={sorted(allowed - actual_files)}, extra={sorted(actual_files - allowed)})"
        )
    return root.resolve()


@lru_cache(maxsize=8)
def validate_local_bge_m3_cached(model_dir: str) -> Path:
    """Einmalige Vollhash-Prüfung je unveränderlicher Release-Generation."""
    return validate_local_bge_m3(model_dir, verify_hashes=True)
