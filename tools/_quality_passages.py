"""Deterministic corpus passages for local retrieval experiments.

No model is loaded here. Offsets always address the original paragraph node's
``text`` field, including its original whitespace. The caller ranks passages
and catalog anchors for each query before assembling a norm package.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from risk_layer.idschema import para_level
from risk_layer.muster import saetze_mit_offset
from risk_layer.zitate import parse_normzitat

if TYPE_CHECKING:
    from risk_layer.graph import NormGraph


_DEAD_WORDS = re.compile(
    r"\b(?:weggefallen|gegenstandslos|aufgehoben|frei|nicht\s+belegt)\b",
    re.IGNORECASE,
)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+|\n(?=[ \t]*\([0-9]+[a-z]?\))")
_SPACE = re.compile(r"\s+")


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
