"""One-hop citation candidates from the original, source-backed norm graph.

Edges discover candidates only. They do not establish legal relevance, and no
evaluation labels, inferred law scope, additional graph hop or model is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .graph import NormGraph
from .idschema import para_level
from .passages import Passage, PassageEvidence, _live_text
from .zitate import knoten_zitat


@dataclass(frozen=True, order=True)
class _OriginalEdge:
    direction: str
    src: str
    dst: str
    exact_target_present: bool


@dataclass(frozen=True)
class _Neighbor:
    source: Passage
    edge: _OriginalEdge


@dataclass(frozen=True)
class _Snapshot:
    graph_marker: str | None
    evidence: PassageEvidence
    source_fingerprint: str
    adjacency: Mapping[str, tuple[_Neighbor, ...]]


def _snapshot(graph: NormGraph, evidence: PassageEvidence) -> _Snapshot:
    marker = getattr(graph, "_embedding_graph_fingerprint_memo", None)
    frozen_marker = marker if isinstance(marker, str) else None
    cached = getattr(graph, "_citation_candidates_snapshot", None)
    if (frozen_marker is not None and isinstance(cached, _Snapshot)
            and cached.graph_marker == frozen_marker and cached.evidence is evidence
            and cached.source_fingerprint == evidence.source_fingerprint):
        return cached
    live: dict[str, Passage] = {}
    for p in evidence.passages:
        if _live_text(p.text) and (p.norm_id not in live or
                (p.start, p.end, p.passage_id) < (live[p.norm_id].start, live[p.norm_id].end, live[p.norm_id].passage_id)):
            live[p.norm_id] = p
    neighbors: dict[str, dict[str, _Neighbor]] = {}
    rows = graph._query("SELECT src,dst,vorhanden FROM kanten WHERE typ='verweist_auf' ORDER BY src,dst")
    for row in rows:
        raw_src, raw_dst = str(row["src"]), str(row["dst"])
        src, dst = para_level(raw_src), para_level(raw_dst)
        if src not in live or dst not in live or src == dst:
            continue
        # Exact subnodes can be absent while their source-backed parent norm
        # exists. Paragraph normalization is the existing retrieval identity.
        for start, end, direction in ((src, dst, "outgoing"), (dst, src, "incoming")):
            edge = _OriginalEdge(direction, raw_src, raw_dst, bool(row["vorhanden"]))
            adjacent = neighbors.setdefault(start, {})
            if end not in adjacent or edge < adjacent[end].edge:
                adjacent[end] = _Neighbor(live[end], edge)
    result = _Snapshot(frozen_marker, evidence, evidence.source_fingerprint,
                       MappingProxyType({seed: tuple(adjacent[nid] for nid in sorted(adjacent))
                                         for seed, adjacent in neighbors.items()}))
    # Only the application's explicitly frozen graph lifecycle permits reuse.
    # Mutable graph handles get a fresh request-local snapshot on every call.
    if frozen_marker is not None:
        setattr(graph, "_citation_candidates_snapshot", result)
    return result


def citation_candidates(graph: NormGraph, evidence: PassageEvidence, seeds: list[str],
                        existing: set[str], k: int = 50, laws: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Return up to 50 new neighbors ordered by first seed rank and norm ID.

    ``seeds`` must already follow the caller's BGE ordering; only its first ten
    entries are used. Both citation directions count as one hop. Scope and the
    existing candidate set are applied before the cap. Callers validate graph /
    evidence generation identity in the regular semantic-resource check.
    Original edge provenance remains in the internal snapshot, not user output.
    """
    if type(k) is not int or not 1 <= k <= 50:
        raise ValueError("citation candidate cap must be an integer from 1 to 50")
    if not isinstance(seeds, list) or any(not isinstance(nid, str) for nid in seeds):
        raise ValueError("citation seeds must be a list of norm IDs")
    if not isinstance(existing, (set, frozenset)) or any(not isinstance(nid, str) for nid in existing):
        raise ValueError("existing candidates must be a set of norm IDs")
    if not isinstance(laws, tuple) or any(not isinstance(law, str) for law in laws):
        raise ValueError("laws must be a tuple of law IDs")
    if not seeds:
        return []
    snapshot = _snapshot(graph, evidence)
    scope, blocked = set(laws), {para_level(nid) for nid in existing}
    result = []
    for seed in seeds[:10]:
        for neighbor in snapshot.adjacency.get(para_level(seed), ()):
            p = neighbor.source
            if p.norm_id in blocked or scope and p.law not in scope:
                continue
            result.append({"norm_id": p.norm_id, "zitat": knoten_zitat(p.law, p.para),
                           "titel": p.title, "sim": None})
            blocked.add(p.norm_id)
            if len(result) == k:
                return result
    return result
