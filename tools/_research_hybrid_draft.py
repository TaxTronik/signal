"""Draft only: persistent CPU BGE plus BM25 research source retrieval.

Nothing imports this module from production. Its fixed RRF rule reproduces the
source-availability diagnostic; it makes no applicability or quality decision.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from risk_layer.embedding import BgeM3Encoder
from risk_layer.embedding_model import (
    BGE_M3_REQUIRED_FILES, BGE_M3_REVISION, MODEL_MANIFEST_NAME, validate_local_bge_m3,
)
from risk_layer.passage_selection import _lexical_score, _lexical_stats, _tokens
from risk_layer.passages import Passage, PassageEvidence


RRF_K = 60
# Original dense encoder/tokenizer files at BGE_M3_REVISION. A normal release
# additionally binds its sparse/ColBERT heads with the existing model manifest.
DENSE_FILE_HASHES = {
    "1_Pooling/config.json": "e54c164a07274f2eb45bb724f54a79d1efcc90c41573887cd9a29aeee0597352",
    "config.json": "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
    "config_sentence_transformers.json": "1eef72430e7194a1e59680e635aed81ffa083f05668dbc5bb1c56c04c0999c38",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "pytorch_model.bin": "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
    "sentence_bert_config.json": "eb9b44b13c0f52a3b3685c3b1cbdea1ba8b04bea123b98f61610048940776eb1",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "tokenizer.json": "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
    "tokenizer_config.json": "a62b2b6784f990259fddef5f16388693a8043be4f69179e6a5257eeb3f9abac4",
}


class ResearchHybridError(ValueError):
    """The source/model identity, CPU execution or query contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResearchHybridError(message)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _default_snapshot() -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub is None:
        home = os.environ.get("HF_HOME")
        hub = str(Path(home) / "hub" if home else Path.home() / ".cache/huggingface/hub")
    return Path(hub) / "models--BAAI--bge-m3/snapshots" / BGE_M3_REVISION


def validate_model_path(model_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Validate a release artifact or the exact already cached dense snapshot."""
    root = Path(model_path) if model_path is not None else _default_snapshot()
    _require(root.is_dir(), "The pinned local BGE model is missing; no download is attempted")
    release = (root / MODEL_MANIFEST_NAME).is_file()
    if release:
        root = validate_local_bge_m3(root, verify_hashes=True)
    else:
        _require(root.name == BGE_M3_REVISION and root.parent.name == "snapshots"
                 and root.parent.parent.name == "models--BAAI--bge-m3",
                 "An explicit model needs a validated release manifest or the pinned HF snapshot")
    allowed = set(BGE_M3_REQUIRED_FILES) | {"README.md", MODEL_MANIFEST_NAME}
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    _require(set(DENSE_FILE_HASHES) <= files.keys() and files.keys() <= allowed,
             "The local dense model inventory is incomplete or includes alternative model files")
    hashes = {relative: _digest(path) for relative, path in sorted(files.items())}
    _require(all(hashes.get(name) == expected for name, expected in DENSE_FILE_HASHES.items()),
             "Local dense model weights/configuration differ from the pinned original revision")
    identity = {"encoder": "bge-m3", "revision": BGE_M3_REVISION, "device": "cpu", "dtype": "float32", "cpu_threads": 2,
                "local_files_only": True, "files": hashes,
                "weights_sha256": hashes["pytorch_model.bin"], "release_manifest": release}
    identity["fingerprint"] = _fingerprint(identity)
    return root.resolve(), identity


def _check_model_files(path: Path, identity: dict[str, Any]) -> None:
    current = {p.relative_to(path).as_posix(): _digest(p) for p in path.rglob("*") if p.is_file()}
    _require(current == identity["files"], "Local model files changed after their identity was bound")


def rank_passages(evidence: PassageEvidence, query: str, vector: np.ndarray,
                  limit: int = 6) -> list[Passage]:
    """Norm RRF(k=60); original source of the better-ranked modality, lexical ties."""
    _validate_query(query, limit)
    vector = np.asarray(vector)
    _require(vector.shape == (evidence.dim,) and np.isfinite(vector).all(), "Invalid dense query vector")
    scores = evidence.vectors @ vector
    terms = sorted(set(_tokens(query)))
    lexical_stats = _lexical_stats(evidence)
    dense_best: dict[str, tuple[float, int, int, int]] = {}
    lexical_best: dict[str, tuple[float, int, int, int]] = {}
    for row, passage in enumerate(evidence.passages):
        nid = passage.norm_id
        dense = (-float(scores[row]), passage.start, passage.end, row)
        if nid not in dense_best or dense < dense_best[nid]:
            dense_best[nid] = dense
        score = _lexical_score(lexical_stats, row, terms)
        lexical = (-score, passage.start, passage.end, row)
        if score > 0 and (nid not in lexical_best or lexical < lexical_best[nid]):
            lexical_best[nid] = lexical
    dense_ids = sorted(dense_best, key=lambda nid: (dense_best[nid][0], nid))
    lexical_ids = sorted(lexical_best, key=lambda nid: (lexical_best[nid][0], nid))
    dense_ranks = {nid: rank for rank, nid in enumerate(dense_ids, 1)}
    lexical_ranks = {nid: rank for rank, nid in enumerate(lexical_ids, 1)}
    rrf = {nid: 1 / (RRF_K + dense_ranks[nid])
           + (1 / (RRF_K + lexical_ranks[nid]) if nid in lexical_ranks else 0)
           for nid in dense_ids}
    selected = []
    for nid in sorted(rrf, key=lambda nid: (-rrf[nid], nid))[:limit]:
        best = lexical_best[nid] if lexical_ranks.get(nid, float("inf")) <= dense_ranks[nid] else dense_best[nid]
        selected.append(evidence.passages[best[-1]])
    return selected


def _validate_query(query: str, limit: int) -> None:
    _require(isinstance(query, str) and bool(query.strip()), "Query must be a nonempty string")
    _require(type(limit) is int and 1 <= limit <= 6, "limit must be an integer from one to six")


def _load_cpu_encoder(path: Path) -> BgeM3Encoder:
    import torch

    # Reject GPU-capable runtimes themselves; no CUDA/HIP availability or device
    # API is called, even incidentally during SentenceTransformers initialization.
    _require(torch.version.cuda is None and torch.version.hip is None,
             "Research hybrid encoding requires the dedicated CPU-only Torch runtime")
    torch.set_num_threads(2)
    encoder = BgeM3Encoder(str(path), device="cpu", revision=BGE_M3_REVISION, local_files_only=True)
    model = encoder._laden()
    _require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()),
             "The encoder must keep all original parameters in FP32 on CPU")
    _require(encoder.dim == 1024, "BGE query dimension differs from the original index")
    return encoder


class HybridResearchRetriever:
    """One immutable evidence snapshot and one persistent, explicitly CPU encoder."""

    def __init__(self, evidence: PassageEvidence, model_path: Path | None = None):
        _require(isinstance(evidence, PassageEvidence), "Original validated passage evidence is required")
        metadata, vectors = evidence.storage()
        _require(metadata["encoder"] == "bge-m3" and metadata["encoder_revision"] == BGE_M3_REVISION
                 and metadata["dim"] == 1024, "Passage index uses a different encoder or revision")
        self._evidence = PassageEvidence(metadata, vectors)
        _lexical_stats(self._evidence)
        started = time.perf_counter()
        self._model_path, model_identity = validate_model_path(model_path)
        self._metrics: dict[str, Any] = {"model_validation_seconds": time.perf_counter() - started,
            "model_load_seconds": 0., "model_load_count": 0, "query_count": 0,
            "encode_count": 0, "encode_seconds": 0., "retrieval_seconds": 0., "failed_queries": 0, "queries": []}
        self._identity = {"format": "research_cpu_hybrid/1", "model": model_identity,
            "evidence_fingerprint": self._evidence.fingerprint,
            "source_fingerprint": self._evidence.source_fingerprint,
            "catalog_fingerprint": metadata["catalog_fingerprint"],
            "policy": {"rrf_k": RRF_K, "dense_norm_score": "max_original_passage", "bm25_norm_score": "max_original_passage",
                       "source_selection": "better_modality_rank_lexical_ties", "limit": [1, 6],
                       "anchors": False, "dense_threshold": None}}
        self._identity["fingerprint"] = _fingerprint(self._identity)
        self._encoder: BgeM3Encoder | None = None
        self._closed = False
        self._lock = threading.RLock()

    def identity_metadata(self) -> dict[str, Any]:
        return deepcopy(self._identity)

    @property
    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._metrics)

    def load(self) -> None:
        """Optional explicit warmup; its cost is never included in encode_seconds."""
        with self._lock:
            _require(not self._closed, "Hybrid retriever is closed")
            if self._encoder is not None:
                return
            started = time.perf_counter()
            try:
                _check_model_files(self._model_path, self._identity["model"])
                encoder = _load_cpu_encoder(self._model_path)
                _check_model_files(self._model_path, self._identity["model"])
                self._encoder = encoder
                self._metrics["model_load_count"] += 1
            finally:
                self._metrics["model_load_seconds"] += time.perf_counter() - started

    def search(self, query: str, limit: int = 6) -> list[Passage]:
        _validate_query(query, limit)
        with self._lock:
            _require(not self._closed, "Hybrid retriever is closed")
            self._metrics["query_count"] += 1
            before = {key: self._metrics[key] for key in ("model_load_seconds", "encode_seconds", "retrieval_seconds", "encode_count")}
            event = {"number": self._metrics["query_count"], "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                     "characters": len(query), "limit": limit, "status": "running"}
            try:
                self.load()
                assert self._encoder is not None
                model = self._encoder._laden()
                tokens = model.tokenizer(query, truncation=False, add_special_tokens=True)["input_ids"]
                _require(len(tokens) <= model.max_seq_length, "Query exceeds the encoder context; no silent truncation is allowed")
                started = time.perf_counter()
                try:
                    self._metrics["encode_count"] += 1
                    matrix = self._encoder.encode([query])
                finally:
                    self._metrics["encode_seconds"] += time.perf_counter() - started
                _require(matrix.shape == (1, 1024) and np.isfinite(matrix).all()
                         and abs(float(np.linalg.norm(matrix[0])) - 1) < 1e-4, "Encoder returned an invalid normalized query vector")
                started = time.perf_counter()
                try:
                    passages = rank_passages(self._evidence, query, matrix[0], limit)
                    event["status"] = "complete"
                    return passages
                finally:
                    self._metrics["retrieval_seconds"] += time.perf_counter() - started
            except Exception:
                self._metrics["failed_queries"] += 1
                event["status"] = "failed"
                raise
            finally:
                event.update({key: self._metrics[key] - before[key] for key in before})
                self._metrics["queries"].append(event)

    def close(self) -> None:
        with self._lock:
            self._encoder = None
            self._closed = True
