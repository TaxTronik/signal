"""Bounded corpus-only views for optional passage-level cross-encoder ranking.

The stored index and default package strategy are untouched. Lexical statistics
belong to the immutable evidence object; selection never uses evaluation labels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Any, Iterable, Sequence
import unicodedata

import numpy as np

from .idschema import para_level
from .passages import NormPackage, Passage, PassageEvidence, _live_text, render_norm_package
from .triggerindex import _STOPP
from .zitate import knoten_zitat


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")


_STOPWORDS = frozenset(map(_fold, _STOPP))


def _tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z]+", _fold(text)) if len(word) >= 3 and word not in _STOPWORDS]


@dataclass(frozen=True)
class _LexicalStats:
    counts: tuple[Counter[str], ...]
    lengths: tuple[int, ...]
    average_length: float
    idf: dict[str, float]


def _lexical_stats(evidence: PassageEvidence) -> _LexicalStats:
    cached: _LexicalStats | None = getattr(evidence, "_selection_lexical_stats", None)
    if cached is not None:
        return cached
    counts = tuple(Counter(_tokens(p.text) + _tokens(p.title) * 2) for p in evidence.passages)
    lengths = tuple(sum(c.values()) for c in counts)
    df: Counter[str] = Counter()
    for count in counts:
        df.update(count.keys())
    n = len(counts)
    cached = _LexicalStats(counts, lengths, sum(lengths) / max(n, 1),
        {term: math.log1p((n - frequency + .5) / (frequency + .5)) for term, frequency in df.items()})
    # Build privately, then publish once; concurrent first readers may compute
    # identical statistics, but cannot observe a partly filled cache.
    setattr(evidence, "_selection_lexical_stats", cached)
    return cached


def _lexical_score(stats: _LexicalStats, row: int, terms: Sequence[str]) -> float:
    normalizer = 1.2 * (.25 + .75 * stats.lengths[row] / max(stats.average_length, 1.))
    return sum(stats.idf.get(term, 0.) * frequency * 2.2 / (frequency + normalizer)
               for term in terms if (frequency := stats.counts[row].get(term, 0)))


def lexical_candidates(evidence: PassageEvidence, query: str, k: int = 20, *,
                       laws: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """BM25 maximum over original views, with explicit scope before the norm cap.

    The same ephemeral corpus statistics serve the within-norm view selector.
    Positive lexical scores propose candidates only; ``sim`` stays ``None``
    because BM25 is not a cosine similarity or a calibrated relevance score.
    """
    if type(k) is not int or k < 1 or not isinstance(query, str):
        raise ValueError("lexical retrieval requires a string query and positive integer k")
    if not isinstance(laws, tuple) or any(not isinstance(law, str) for law in laws):
        raise ValueError("laws must be a tuple of law IDs")
    terms = sorted(set(_tokens(query)))
    if not terms:
        return []
    stats, scope = _lexical_stats(evidence), set(laws)
    best: dict[str, tuple[float, Passage]] = {}
    for row, p in enumerate(evidence.passages):
        if scope and p.law not in scope:
            continue
        score = _lexical_score(stats, row, terms)
        if score > 0 and (p.norm_id not in best or score > best[p.norm_id][0]):
            best[p.norm_id] = (score, p)
    ranked = sorted(best, key=lambda nid: (-best[nid][0], nid))[:k]
    return [{"norm_id": nid, "zitat": knoten_zitat(best[nid][1].law, best[nid][1].para),
             "titel": best[nid][1].title, "sim": None} for nid in ranked]


def select_passage_candidates(evidence: PassageEvidence, norm_id: str, vector: np.ndarray,
                              query: str, limit: int = 8) -> list[Passage]:
    """First source view, then alternating semantic/BM25 views; at most eight.

    Scores use existing title+passage embeddings and source-only lexical statistics
    (BM25 k1=1.2, b=.75, title counted twice). Whole-source offsets are unchanged.
    A missing norm returns an empty list. No scoring model or corpus write occurs.
    """
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("passage limit must be an integer from 1 to 8")
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    rows = evidence._by_norm.get(para_level(norm_id), [])
    if not rows:
        return []
    values = evidence.vectors[rows] @ evidence._query(vector)
    live = [i for i in range(len(rows)) if _live_text(evidence.passages[rows[i]].text)]
    if not live:
        return []
    stats, terms = _lexical_stats(evidence), sorted(set(_tokens(query)))

    def source_order(i: int) -> tuple[int, int, str]:
        p = evidence.passages[rows[i]]
        return p.start, p.end, p.passage_id

    semantic = sorted(live, key=lambda i: (-float(values[i]), source_order(i)))
    lexical = sorted(live, key=lambda i: (-_lexical_score(stats, rows[i], terms), source_order(i)))
    selected: list[Passage] = []
    seen: set[str] = set()

    def append(i: int) -> None:
        p = evidence.passages[rows[i]]
        key = " ".join(p.text.casefold().split())
        if key not in seen and len(selected) < limit:
            selected.append(p)
            seen.add(key)

    append(min(live, key=source_order))
    for dense, sparse in zip(semantic, lexical, strict=True):
        append(dense)
        append(sparse)
        if len(selected) == limit:
            break
    return selected


def _novel_fragment(passage: Passage, previous: Passage) -> Passage | None:
    intervals = [(passage.start, passage.end)]
    if max(passage.start, previous.start) < min(passage.end, previous.end):
        intervals = [(passage.start, min(passage.end, previous.start)),
                     (max(passage.start, previous.end), passage.end)]
    # One view may surround the first. Keep its larger remaining interval;
    # never concatenate disjoint source intervals into fabricated provenance.
    for start, end in sorted(intervals, key=lambda span: (-(span[1] - span[0]), span[0])):
        if start >= end:
            continue
        raw = passage.text[start - passage.start:end - passage.start]
        start += len(raw) - len(raw.lstrip())
        text = raw.strip()
        if _live_text(text):
            return Passage(passage.passage_id, passage.norm_id, passage.law, passage.para,
                           passage.title, start, start + len(text), text)
    return None


def package_ranked_passages(passages: Sequence[Passage], scores: Sequence[float], anchors: Iterable[str],
                            source_fingerprint: str, *, max_chars: int = 2200) -> NormPackage:
    """Highest raw BGE views, at most two non-overlapping exact source slices.

    Callers score each ``norm_id + '\n\n' + passage.embedding_text`` independently.
    The first input need not win. The norm ID remains the first package line for
    score-cache tracing. Anchors must already belong to this norm. Empty inputs,
    mixed norms, inconsistent overlapping source or malformed scores fail loudly.
    """
    if not passages or len(passages) != len(scores):
        raise ValueError("passages and scores must have equal nonzero length")
    if (any(isinstance(s, bool) or not isinstance(s, Real) or not math.isfinite(float(s)) for s in scores)
            or not isinstance(source_fingerprint, str) or not source_fingerprint):
        raise ValueError("finite real scores and original source fingerprint are required")
    if type(max_chars) is not int or not 1 <= max_chars <= 2200:
        raise ValueError("package character limit must be an integer from 1 to 2200")
    first = passages[0]
    identity = first.norm_id, first.law, first.para, first.title
    for p in passages:
        if (type(p.start) is not int or type(p.end) is not int
                or not isinstance(p.text, str)
                or (p.norm_id, p.law, p.para, p.title) != identity
                or p.start < 0 or p.end <= p.start or p.end - p.start != len(p.text)):
            raise ValueError("passages must preserve one norm identity and exact source lengths")
        parent = (re.fullmatch(r"(.+)@([0-9]+):([0-9]+)", p.passage_id)
                  if isinstance(p.passage_id, str) else None)
        if (parent is None or parent[1] != p.norm_id
                or not int(parent[2]) <= p.start < p.end <= int(parent[3])):
            raise ValueError("passage ID must identify the same norm and enclose the source span")
        for other in passages:
            start, end = max(p.start, other.start), min(p.end, other.end)
            if start < end and p.text[start - p.start:end - p.start] != other.text[start - other.start:end - other.start]:
                raise ValueError("overlapping passages disagree about original source text")
    order = sorted(range(len(passages)), key=lambda i: (-float(scores[i]), passages[i].start,
                                                        passages[i].end, passages[i].passage_id))
    selected: list[Passage] = []
    seen = set()
    for i in order:
        p = passages[i]
        if not _live_text(p.text):
            continue
        if selected:
            novel = _novel_fragment(p, selected[0])
            if novel is None:
                continue
            p = novel
        key = " ".join(p.text.casefold().split())
        if key not in seen:
            selected.append(p)
            seen.add(key)
        if len(selected) == 2:
            break
    if not selected:
        raise ValueError("package requires a live original source passage")
    retained_anchors = []
    for anchor in anchors:
        if not isinstance(anchor, str):
            raise ValueError("catalog anchors must be strings")
        anchor = anchor.strip()
        key = " ".join(anchor.casefold().split())
        if key and key not in seen:
            retained_anchors.append(anchor)
            seen.add(key)
        if len(retained_anchors) == 2:
            break
    package = render_norm_package(first.norm_id, selected, retained_anchors, max_chars, source_fingerprint)
    if not package.evidence:
        raise ValueError("character budget does not fit the norm header and source evidence")
    return package
