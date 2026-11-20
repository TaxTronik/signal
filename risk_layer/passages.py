"""Deterministic corpus passages and optional persistent retrieval evidence.

No model is loaded here. Offsets always address the original paragraph node's
``text`` field, including its original whitespace. The caller ranks passages
and catalog anchors for each query before assembling a norm package.
"""

from __future__ import annotations

import re
import hashlib
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np

from risk_layer.idschema import para_level
from risk_layer.muster import saetze_mit_offset
from risk_layer.zitate import knoten_zitat, parse_normzitat

if TYPE_CHECKING:
    from risk_layer.graph import NormGraph


_DEAD_WORDS = re.compile(
    r"\b(?:weggefallen|gegenstandslos|aufgehoben|frei|nicht\s+belegt)\b",
    re.IGNORECASE,
)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+|\n(?=[ \t]*\([0-9]+[a-z]?\))")
_SPACE = re.compile(r"\s+")
EVIDENCE_SCHEMA = "risk-layer/passage-evidence/v1"
# Bump when source selection, the shared sentence splitter, passage boundaries,
# embedding text, or anchor mapping changes: all affect persisted evidence.
EVIDENCE_BUILDER_VERSION = "risk-layer/passages/v1"


@dataclass(frozen=True)
class Passage:
    passage_id: str
    norm_id: str
    law: str
    para: str
    title: str
    start: int
    end: int
    text: str

    @property
    def embedding_text(self) -> str:
        """Existing source title plus passage; titleless norms stay titleless."""
        return "\n".join(part for part in (self.title.strip(), self.text) if part)


def _live_text(text: str) -> bool:
    # Also excludes a paragraph consisting solely of several retired numbered
    # subsections. A live subsection next to a retired one remains eligible.
    return any(c.isalpha() for c in _DEAD_WORDS.sub("", text))


def _last_boundary(boundaries: list[int], minimum: int, maximum: int) -> int | None:
    i = bisect_right(boundaries, maximum) - 1
    return boundaries[i] if i >= 0 and boundaries[i] >= minimum else None


def passage_spans(text: str, *, max_chars: int = 900,
                  overlap_chars: int = 120) -> list[tuple[int, int]]:
    """Cover every non-whitespace source character with bounded source slices.

    Prefer paragraph boundaries, then sentence boundaries, in the latter half
    of a window. Very long sentences fall back to word boundaries, and very
    long tokens to a character boundary. Overlap starts at a word boundary
    within the requested overlap where possible; it never prevents progress.
    """
    if max_chars < 2:
        raise ValueError("max_chars must be at least 2")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be >= 0 and < max_chars")
    paragraph_ends = [m.start() for m in _PARAGRAPH_BREAK.finditer(text)]
    sentence_ends: list[int] = []
    for sentence in saetze_mit_offset(text):
        end = sentence.end
        # The shared splitter excludes separators. Retain them in the source
        # passage instead of silently dropping punctuation from the evidence.
        while end < len(text) and text[end] in '.!?;:“”"\'»›)]':
            end += 1
        sentence_ends.append(end)
    spaces = list(_SPACE.finditer(text))
    word_ends = [m.start() for m in spaces]
    word_starts = [m.end() for m in spaces]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break
        limit = min(len(text), start + max_chars)
        end = limit
        if limit < len(text):
            minimum = start + max_chars // 2
            end = (_last_boundary(paragraph_ends, minimum, limit)
                   or _last_boundary(sentence_ends, minimum, limit)
                   or _last_boundary(word_ends, minimum, limit)
                   or limit)
        while end > start and text[end - 1].isspace():
            end -= 1
        spans.append((start, end))
        if limit == len(text):
            break
        next_start = max(start + 1, end - overlap_chars)
        i = bisect_right(word_starts, next_start - 1)
        if i < len(word_starts) and word_starts[i] < end:
            next_start = word_starts[i]
        start = next_start
    return spans


def build_passages(graph: NormGraph, *, max_chars: int = 900,
                   overlap_chars: int = 120) -> list[Passage]:
    """Build norm-ID/source-offset ordered passages from live paragraph nodes.

    Descendant nodes are excluded even if their optional depth metadata is
    absent: their text is already represented by the paragraph node. No title
    requirement, generated title, paraphrase, or legal interpretation is used.
    """
    rows = graph._query(
        "SELECT id, law, para, titel, text FROM knoten "
        "WHERE absatz IS NULL AND nummer IS NULL ORDER BY id"
    )
    out: list[Passage] = []
    for row in rows:
        norm_id = str(row["id"])
        if not norm_id.startswith("norm:") or norm_id != para_level(norm_id):
            continue
        text = str(row["text"] or "")
        if not _live_text(text):
            continue
        for start, end in passage_spans(
                text, max_chars=max_chars, overlap_chars=overlap_chars):
            excerpt = text[start:end]
            if not _live_text(excerpt):
                continue
            out.append(Passage(
                passage_id=f"{norm_id}@{start}:{end}", norm_id=norm_id,
                law=str(row["law"] or ""), para=str(row["para"] or ""),
                title=str(row["titel"] or ""), start=start, end=end, text=excerpt,
            ))
    return out


def catalog_anchors(katalog) -> dict[str, tuple[str, ...]]:
    """Existing catalog paraphrases grouped by their declared paragraph targets.

    Norm anchors and chain heads follow the current reranker mapping. Repeated
    citations, subsection targets, and identical paraphrases are deduplicated.
    Returned order is deterministic; no model-generated anchors are added.
    """
    by_norm: dict[str, dict[str, str]] = {}
    for entry in katalog:
        citations = list(entry.norm_anker) + [
            chain.primaer for chain in entry.normketten if chain.primaer]
        targets = {
            para_level(anchor.norm_id.to_id())
            for citation in citations for anchor in parse_normzitat(citation)
            if anchor.norm_id is not None
        }
        for target in sorted(targets):
            phrases = by_norm.setdefault(target, {})
            for phrase in entry.paraphrasen:
                phrase = str(phrase).strip()
                key = " ".join(phrase.casefold().split())
                if key:
                    phrases.setdefault(key, phrase)
    return {norm: tuple(phrases[key] for key in sorted(phrases))
            for norm, phrases in sorted(by_norm.items())}


def assemble_norm_package(norm_id: str, ranked_passages: Iterable[Passage], *,
                          ranked_anchors: Iterable[str] = (),
                          max_passages: int = 2, max_anchors: int = 2,
                          max_chars: int = 2200) -> str:
    """Assemble one query-specific document from already ranked evidence.

    ``ranked_passages`` may be a global ranking; foreign norms are ignored.
    ``ranked_anchors`` must contain only anchors belonging to this norm, ranked
    by the caller for the same query (``catalog_anchors`` supplies ownership).
    Duplicate passages/anchors do not consume the respective limits. The first
    passage has priority if the final character budget requires truncation.
    """
    if max_passages < 0 or max_anchors < 0 or max_chars < 1:
        raise ValueError("Evidence limits must be nonnegative; max_chars must be positive")
    norm_id = para_level(norm_id)
    chosen: list[Passage] = []
    seen: set[str] = set()
    for passage in ranked_passages:
        if len(chosen) >= max_passages:
            break
        key = " ".join(passage.text.casefold().split())
        if passage.norm_id == norm_id and key and key not in seen:
            chosen.append(passage)
            seen.add(key)
    parts = [norm_id]
    if chosen and chosen[0].title.strip():
        parts.append(chosen[0].title.strip())
    parts.extend(passage.text for passage in chosen)
    n_anchors = 0
    for anchor in ranked_anchors:
        if n_anchors >= max_anchors:
            break
        anchor = str(anchor).strip()
        key = " ".join(anchor.casefold().split())
        if key and key not in seen:
            parts.append(anchor)
            seen.add(key)
            n_anchors += 1
    return "\n\n".join(parts)[:max_chars].rstrip()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _sources(graph) -> list[dict[str, str]]:
    rows = graph._query(
        "SELECT id, law, para, titel, text FROM knoten "
        "WHERE absatz IS NULL AND nummer IS NULL ORDER BY id")
    return [{k: str(row[k] or "") for k in ("id", "law", "para", "titel", "text")}
            for row in rows
            if str(row["id"]).startswith("norm:") and row["id"] == para_level(row["id"])]


@dataclass(frozen=True)
class NormPackage:
    text: str
    evidence: tuple[Passage, ...]
    anchors: tuple[str, ...]
    truncated: bool
    source_fingerprint: str


def assemble_source_first_package(norm_id: str, ranked_passages: Iterable[Passage], *,
                                  ranked_anchors: Iterable[str] = (), max_chars: int = 2200,
                                  source_fingerprint: str = "") -> NormPackage:
    """Keep the first live source window and one query-ranked later window.

    Pass the complete passage ranking for this norm, not only its top results:
    the first window is determined from source offsets, independently of scores.
    Foreign norms and retired windows are ignored. Overlap with the first window
    is removed by slicing the later source window; its stored passage ID remains
    the parent identity while returned offsets describe the actual excerpt.

    The first window is a source-order heuristic, not a generated general rule.
    Index evidence and its fingerprints are unchanged; this is query-time only.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    norm_id = para_level(norm_id)
    ranked = [p for p in ranked_passages if p.norm_id == norm_id and _live_text(p.text)]
    if not ranked:
        return NormPackage(norm_id[:max_chars], (), (), len(norm_id) > max_chars, source_fingerprint)
    first = min(ranked, key=lambda p: (p.start, p.end, p.passage_id))
    selected = [first]
    seen = {" ".join(first.text.casefold().split())}
    for p in ranked:
        if " ".join(p.text.casefold().split()) in seen:
            continue
        start = max(p.start, first.end)
        if start >= p.end:
            continue
        excerpt = p.text[start - p.start:]
        start += len(excerpt) - len(excerpt.lstrip())
        excerpt = excerpt.strip()
        if not _live_text(excerpt) or " ".join(excerpt.casefold().split()) in seen:
            continue
        selected.append(Passage(p.passage_id, p.norm_id, p.law, p.para, p.title,
                                start, start + len(excerpt), excerpt))
        seen.add(" ".join(excerpt.casefold().split()))
        break
    anchors = []
    for anchor in ranked_anchors:
        anchor = str(anchor).strip()
        key = " ".join(anchor.casefold().split())
        if key and key not in seen:
            anchors.append(anchor)
            seen.add(key)
        if len(anchors) == 2:
            break
    return render_norm_package(norm_id, selected, anchors, max_chars, source_fingerprint)


def render_norm_package(norm_id: str, selected: list[Passage], anchors: list[str],
                        max_chars: int, source_fingerprint: str) -> NormPackage:
    """Track each part's actual position, even when two parts share wording."""
    if not selected:
        return NormPackage(norm_id[:max_chars], (), (), len(norm_id) > max_chars, source_fingerprint)
    first = selected[0]
    # An unusually long heading must not displace the mandatory source window.
    # Short explicit budgets may still truncate that window, which is reported.
    title = first.title.strip()
    title_budget = max(0, max_chars - len(norm_id) - len(first.text) - 4)
    shown_title = title[:title_budget].rstrip()
    parts: list[tuple[str, Passage | None, bool]] = [(norm_id, None, False)]
    if shown_title:
        parts.append((shown_title, None, False))
    parts.extend((p.text, p, False) for p in selected)
    parts.extend((anchor, None, True) for anchor in anchors)
    full = "\n\n".join(part[0] for part in parts)
    text = full[:max_chars].rstrip()
    evidence, present_anchors = [], []
    position = 0
    for content, passage, is_anchor in parts:
        available = max(0, min(len(content), len(text) - position))
        if passage is not None and available:
            evidence.append(Passage(passage.passage_id, passage.norm_id, passage.law,
                                    passage.para, passage.title, passage.start,
                                    passage.start + available, passage.text[:available]))
        elif is_anchor and available == len(content):
            present_anchors.append(content)
        position += len(content) + 2
    return NormPackage(text, tuple(evidence), tuple(present_anchors),
                       len(text) < len(full) or shown_title != title, source_fingerprint)


class PassageEvidence:
    """Immutable source/embedding snapshot attached to an EmbeddingIndex.

    The serialized sources retain the exact original text, so load can verify
    every offset and excerpt without consulting a potentially newer graph.
    Current-graph compatibility is checked separately by ``matches_graph``.
    """

    def __init__(self, metadata: dict[str, Any], vectors: np.ndarray):
        self._metadata = json.loads(json.dumps(metadata, ensure_ascii=False))
        self._validate(vectors)
        self.fingerprint: str = self._metadata["fingerprint"]
        self.source_fingerprint: str = self._metadata["source_fingerprint"]
        self.passages = tuple(Passage(**p) for p in self._metadata["passages"])
        self.anchors = tuple((a["norm_id"], a["text"]) for a in self._metadata["anchors"])
        self.dim: int = self._metadata["dim"]
        self.vectors = np.array(vectors[:len(self.passages)], dtype=np.float32, copy=True)
        self.anchor_vectors = np.array(vectors[len(self.passages):], dtype=np.float32, copy=True)
        self.vectors.flags.writeable = False
        self.anchor_vectors.flags.writeable = False
        self._by_norm: dict[str, list[int]] = {}
        self._anchors_by_norm: dict[str, list[int]] = {}
        for i, p in enumerate(self.passages):
            self._by_norm.setdefault(p.norm_id, []).append(i)
        for i, (nid, _) in enumerate(self.anchors):
            self._anchors_by_norm.setdefault(nid, []).append(i)

    @classmethod
    def build(cls, graph, encoder, katalog=None, *, catalog_fingerprint=None,
              cancelled: Callable[[], bool] | None = None) -> PassageEvidence:
        from .embedding import EmbeddingBuildCancelled

        def check() -> None:
            if cancelled is not None and cancelled():
                raise EmbeddingBuildCancelled("Passage evidence build was cancelled")

        check()
        sources = _sources(graph)
        passages = build_passages(graph)
        covered_norms = {p.norm_id for p in passages}
        anchors = [(nid, text) for nid, phrases in catalog_anchors(katalog or []).items()
                   if nid in covered_norms for text in phrases]
        texts = [p.embedding_text for p in passages] + [text for _, text in anchors]
        batches = []
        for start in range(0, len(texts), 32):
            check()
            batches.append(np.asarray(encoder.encode(texts[start:start + 32]), dtype=np.float32))
            check()
        vectors = np.vstack(batches) if batches else np.zeros((0, encoder.dim), dtype=np.float32)
        if sources != _sources(graph):
            raise ValueError("Graph changed during passage evidence build")
        metadata = {
            "schema": EVIDENCE_SCHEMA, "builder_version": EVIDENCE_BUILDER_VERSION,
            "config": {"max_chars": 900, "overlap_chars": 120},
            "encoder": encoder.name, "encoder_revision": getattr(encoder, "revision", None),
            "catalog_fingerprint": catalog_fingerprint, "dim": int(vectors.shape[1]),
            "sources": sources, "source_fingerprint": _digest(sources),
            "source_text_hashes": {s["id"]: hashlib.sha256(s["text"].encode("utf-8")).hexdigest()
                                   for s in sources},
            "passages": [asdict(p) for p in passages],
            "anchors": [{"norm_id": nid, "text": text} for nid, text in anchors],
            "passage_rows": len(passages), "anchor_rows": len(anchors),
            "vectors_sha256": hashlib.sha256(vectors.tobytes(order="C")).hexdigest(),
        }
        metadata["fingerprint"] = _digest(metadata)
        return cls(metadata, vectors)

    def _validate(self, vectors: np.ndarray) -> None:
        m = self._metadata
        try:
            if set(m) != {"schema", "builder_version", "config", "encoder", "encoder_revision",
                          "catalog_fingerprint", "dim", "sources", "source_fingerprint",
                          "source_text_hashes", "passages", "anchors", "passage_rows", "anchor_rows",
                          "vectors_sha256", "fingerprint"}:
                raise ValueError("invalid evidence metadata fields")
            if m["schema"] != EVIDENCE_SCHEMA or m["builder_version"] != EVIDENCE_BUILDER_VERSION:
                raise ValueError("unknown evidence schema/builder")
            if m["config"] != {"max_chars": 900, "overlap_chars": 120}:
                raise ValueError("unsupported passage configuration")
            if m["fingerprint"] != _digest({k: v for k, v in m.items() if k != "fingerprint"}):
                raise ValueError("metadata fingerprint mismatch")
            dim, npass, nanchor = m["dim"], m["passage_rows"], m["anchor_rows"]
            if any(type(n) is not int or n < 0 for n in (dim, npass, nanchor)):
                raise ValueError("invalid vector dimensions/counts")
            if vectors.dtype != np.dtype("float32") or vectors.shape != (npass + nanchor, dim):
                raise ValueError("vector dimension/count mismatch")
            if not np.isfinite(vectors).all():
                raise ValueError("nonfinite evidence vectors")
            if m["vectors_sha256"] != hashlib.sha256(vectors.tobytes(order="C")).hexdigest():
                raise ValueError("evidence vector checksum mismatch")
            if len(m["passages"]) != npass or len(m["anchors"]) != nanchor:
                raise ValueError("record count mismatch")
            if m["source_fingerprint"] != _digest(m["sources"]):
                raise ValueError("source fingerprint mismatch")
            sources = {s["id"]: s for s in m["sources"]}
            if len(sources) != len(m["sources"]):
                raise ValueError("duplicate source norms")
            for nid, s in sources.items():
                if (set(s) != {"id", "law", "para", "titel", "text"}
                        or not all(isinstance(v, str) for v in s.values())
                        or not nid.startswith("norm:") or para_level(nid) != nid):
                    raise ValueError("invalid source norm metadata")
            expected_hashes = {nid: hashlib.sha256(s["text"].encode("utf-8")).hexdigest()
                               for nid, s in sources.items()}
            if expected_hashes != m["source_text_hashes"]:
                raise ValueError("original source text hash mismatch")
            seen = set()
            for raw in m["passages"]:
                p = Passage(**raw)
                s = sources[p.norm_id]
                if (type(p.start) is not int or type(p.end) is not int
                        or not 0 <= p.start < p.end <= len(s["text"])
                        or p.end - p.start > m["config"]["max_chars"]
                        or p.text != s["text"][p.start:p.end]
                        or p.law != s["law"] or p.para != s["para"] or p.title != s["titel"]
                        or p.passage_id != f"{p.norm_id}@{p.start}:{p.end}"
                        or p.passage_id in seen or not _live_text(p.text)):
                    raise ValueError("invalid passage offsets/source/identity")
                seen.add(p.passage_id)
            for anchor in m["anchors"]:
                if (set(anchor) != {"norm_id", "text"} or anchor["norm_id"] not in sources
                        or not isinstance(anchor["text"], str) or not anchor["text"].strip()):
                    raise ValueError("invalid anchor ownership/text")
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError(f"Invalid passage evidence: {exc}") from exc

    def storage(self) -> tuple[dict[str, Any], np.ndarray]:
        return (json.loads(json.dumps(self._metadata, ensure_ascii=False)),
                np.vstack([self.vectors, self.anchor_vectors]))

    def matches_identity(self, encoder_name, encoder_revision, catalog_fingerprint) -> bool:
        return (self._metadata["encoder"] == encoder_name
                and self._metadata["encoder_revision"] == encoder_revision
                and self._metadata["catalog_fingerprint"] == catalog_fingerprint)

    def matches_graph(self, graph) -> bool:
        # The app explicitly freezes its graph and memoizes the base input
        # fingerprint at startup. Cache the full passage hash only for that
        # immutable lifecycle; ordinary mutable graph handles stay uncached.
        frozen = isinstance(getattr(graph, "_embedding_graph_fingerprint_memo", None), str)
        current = getattr(graph, "_passage_graph_fingerprint_memo", None) if frozen else None
        if not isinstance(current, str):
            current = _digest(_sources(graph))
            if frozen:
                setattr(graph, "_passage_graph_fingerprint_memo", current)
        return self.source_fingerprint == current

    def _query(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector)
        if vector.shape != (self.dim,) or not np.isfinite(vector).all():
            raise ValueError("Passage query vector dimension/values invalid")
        return vector

    def candidates(self, vector: np.ndarray, *, k: int = 20, min_sim: float = .35) -> list[dict]:
        scores = self.vectors @ self._query(vector)
        candidates: list[dict[str, Any]] = []
        for nid, rows in self._by_norm.items():
            i = max(rows, key=lambda row: (float(scores[row]), -row))
            score = float(scores[i])
            if score >= min_sim:
                p = self.passages[i]
                candidates.append({"norm_id": nid, "zitat": knoten_zitat(p.law, p.para),
                                   "titel": p.title, "sim": score})
        return sorted(candidates, key=lambda n: (-n["sim"], n["norm_id"]))[:max(0, k)]

    def norm_package(self, norm_id: str, vector: np.ndarray, *, strategy: str = "ranked") -> NormPackage:
        if strategy not in ("ranked", "source_first"):
            raise ValueError(f"Unknown norm package strategy: {strategy}")
        vector = self._query(vector)
        norm_id = para_level(norm_id)
        rows = self._by_norm.get(norm_id, [])
        arows = self._anchors_by_norm.get(norm_id, [])
        scores = self.vectors[rows] @ vector
        ascores = self.anchor_vectors[arows] @ vector
        if strategy == "source_first":
            ranked = [self.passages[rows[i]] for i in sorted(
                range(len(rows)), key=lambda i: (-float(scores[i]), rows[i]))]
            ranked_anchors = [self.anchors[arows[i]][1] for i in sorted(
                range(len(arows)), key=lambda i: (-float(ascores[i]), arows[i]))]
            return assemble_source_first_package(norm_id, ranked, ranked_anchors=ranked_anchors,
                                                 source_fingerprint=self.source_fingerprint)
        selected: list[Passage] = []
        seen = set()
        for i in sorted(range(len(rows)), key=lambda i: (-float(scores[i]), rows[i])):
            p = self.passages[rows[i]]
            key = " ".join(p.text.casefold().split())
            if key not in seen:
                seen.add(key)
                selected.append(p)
            if len(selected) == 2:
                break
        anchors = []
        for i in sorted(range(len(arows)), key=lambda i: (-float(ascores[i]), arows[i])):
            text = self.anchors[arows[i]][1]
            key = " ".join(text.casefold().split())
            if key not in seen:
                seen.add(key)
                anchors.append(text)
            if len(anchors) == 2:
                break
        return render_norm_package(norm_id, selected, anchors, 2200, self.source_fingerprint)
