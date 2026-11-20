"""Bounded local research tools; no generator, encoder inference or GPU calls.

An immutable corpus snapshot can serve many independent sessions. Evidence IDs
are accepted only after that session returned their original text to its caller.
Source presence never establishes legal applicability.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from .embedding import EmbeddingIndex
from .graph import NormGraph
from .passage_selection import _lexical_score, _lexical_stats, _tokens
from .passages import Passage, _live_text
from .semantic_context import build_context_queries
from .zitate import knoten_zitat


class ResearchSourceError(ValueError):
    """Invalid tool input or a conclusion without its original source evidence."""


class ResearchBudgetExceeded(ResearchSourceError):
    """A fixed total or per-tool research allowance has been exhausted."""


_LIMITS = {"search_sources": 3, "read_source": 5, "follow_citations": 2}
_NORM_ID = re.compile(r"norm:[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9][A-Za-z0-9.-]*(?::(?:abs|nr|s|lit|hs|alt)[A-Za-z0-9.-]+)*")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResearchSourceError(message)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _integer(value: Any, minimum: int, maximum: int, name: str) -> int:
    _require(type(value) is int and minimum <= value <= maximum, f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _norm_id(value: Any) -> str:
    _require(isinstance(value, str) and len(value) <= 200 and _NORM_ID.fullmatch(value) is not None,
             "norm_id must be a canonical norm identifier")
    return value


@dataclass(frozen=True)
class _Source:
    id: str
    title: str
    text: str
    law: str
    para: str
    text_hash: str


class ResearchCorpus:
    """Load and validate one graph/evidence generation, with only CPU work."""

    def __init__(self, graph_path: Path, index_dir: Path, *, retrieval: str = "bm25",
                 encoder_model_path: Path | None = None, catalog_path: Path | None = None):
        graph_path, index_dir = Path(graph_path), Path(index_dir)
        _require(retrieval in ("bm25", "hybrid"), "Unknown research retrieval mode")
        _require(retrieval == "hybrid" or encoder_model_path is None, "BM25 does not use an encoder model")
        _require(graph_path.is_file(), "Research graph does not exist")
        self._closed = False
        self._retrieval = retrieval
        self._hybrid: Any = None
        self._catalog: Any = None
        self._prepared = False
        self._graph = NormGraph(graph_path)
        try:
            self._graph.con.execute("PRAGMA query_only = ON")
            self._graph.con.execute("BEGIN")
            index = EmbeddingIndex.load(index_dir)
            _require(index is not None and index.passage_evidence is not None,
                     "Research requires an index with original passage evidence")
            assert index is not None and index.passage_evidence is not None
            evidence = index.passage_evidence
            graph_fingerprint = self._graph.fingerprint()
            _require(index.graph_fingerprint == graph_fingerprint and evidence.matches_graph(self._graph),
                     "Research graph and passage index belong to different generations")
            _require(evidence.matches_identity(index.encoder_name, index.encoder_revision, index.catalog_fingerprint),
                     "Research evidence and embedding index identities differ")
            if catalog_path is not None:
                from .research_catalog import ResearchCatalog
                self._catalog = ResearchCatalog(catalog_path, expected_fingerprint=index.catalog_fingerprint,
                                                evidence=evidence if retrieval == "hybrid" else None)
            rows = self._graph._query("SELECT id,titel,text,law,para FROM knoten ORDER BY id")
            sources = {r["id"]: _Source(r["id"], r["titel"] or "", r["text"] or "", r["law"] or "",
                                        r["para"] or "", _sha(r["text"] or "")) for r in rows}
            edges = [dict(r) for r in self._graph._query(
                "SELECT src,dst,typ,konfidenz,vorhanden FROM kanten WHERE typ='verweist_auf' ORDER BY src,dst")]
            # Include edge identity: the legacy graph fingerprint covers node
            # text, whereas a research session can also inspect citation edges.
            self._identity = {"graph_fingerprint": graph_fingerprint, "index_fingerprint": evidence.fingerprint,
                              "source_fingerprint": evidence.source_fingerprint,
                              "citation_fingerprint": _sha(_json(edges))}
            self._sources = MappingProxyType(sources)
            outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            incoming: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for edge in edges:
                frozen = MappingProxyType(edge)
                outgoing[edge["src"]].append(frozen)
                incoming[edge["dst"]].append(frozen)
            self._outgoing = MappingProxyType({nid: tuple(es) for nid, es in outgoing.items()})
            self._incoming = MappingProxyType({nid: tuple(es) for nid, es in incoming.items()})
            self._passages = evidence.passages
            self._lexical = _lexical_stats(evidence)
            _require(self._graph.fingerprint() == graph_fingerprint and evidence.matches_graph(self._graph),
                     "Research graph changed while loading the snapshot")
            self._graph.con.execute("COMMIT")
            if retrieval == "hybrid":
                from .research_hybrid import HybridResearchRetriever
                self._hybrid = HybridResearchRetriever(evidence, encoder_model_path)
                self._identity["retrieval_fingerprint"] = self._hybrid.identity_metadata()["fingerprint"]
        except Exception:
            self._graph.close()
            self._closed = True
            raise

    def close(self) -> None:
        if not self._closed:
            if self._hybrid is not None:
                self._hybrid.close()
            self._graph.close()
            self._closed = True

    def prepare(self) -> None:
        """Optional explicit model load and fixed CPU warmup before timed cases."""
        self._ensure_open()
        if not self._prepared and self._hybrid is not None:
            self._hybrid.load()
            self._hybrid.search("Technischer Probelauf der lokalen Quellensuche.", limit=1)
        self._prepared = True

    def retrieval_metadata(self) -> dict[str, Any]:
        if self._hybrid is None:
            return {"mode": "bm25", "policy": {"norm_score": "max_original_passage", "anchors": False}}
        identity = self._hybrid.identity_metadata()
        return {"mode": "hybrid", "model_path": str(self._hybrid.model_path),
                "model_identity": identity["model"],
                **{key: deepcopy(identity[key]) for key in ("format", "query_vector") if key in identity},
                **{key: identity[key] for key in ("fingerprint", "policy", "evidence_fingerprint", "source_fingerprint")}}

    @property
    def retrieval_metrics(self) -> dict[str, Any]:
        return self._hybrid.metrics if self._hybrid is not None else {}

    def _ensure_open(self) -> None:
        _require(not self._closed, "Research corpus is closed")

    def identity_metadata(self) -> dict[str, str]:
        return dict(self._identity)

    def catalog_metadata(self) -> dict[str, Any] | None:
        return self._catalog.metadata() if self._catalog is not None else None

    def new_session(self, text: str) -> ResearchSession:
        self._ensure_open()
        _require(isinstance(text, str), "Research document must be a string")
        return ResearchSession(self, text)

    def _resolve(self, norm_id: str) -> _Source | None:
        parts = norm_id.split(":")
        while len(parts) >= 3:
            source = self._sources.get(":".join(parts))
            if source is not None:
                return source
            parts.pop()
        return None

    def _search(self, query: str, limit: int):
        if self._hybrid is not None:
            return self._hybrid.search(query, limit)
        terms = sorted(set(_tokens(query)))
        if not terms:
            return []
        best: dict[str, tuple[float, int, int, Passage]] = {}
        for row, passage in enumerate(self._passages):
            score = _lexical_score(self._lexical, row, terms)
            previous = best.get(passage.norm_id)
            if score > 0 and (previous is None or (-score, passage.start, passage.end) < previous[:3]):
                best[passage.norm_id] = (-score, passage.start, passage.end, passage)
        return [best[nid][3] for nid in sorted(best, key=lambda nid: (best[nid][0], nid))[:limit]]


class ResearchSession:
    """One original document, bounded actions and a private source ledger."""

    def __init__(self, corpus: ResearchCorpus, text: str):
        self._corpus, self._text = corpus, text
        self._document_hash = _sha(text)
        self._focuses: list[dict[str, Any]] = [
            {"id": f"f{i}", "start": c.focus_start, "end": c.focus_end, "text": c.focus_text}
            for i, c in enumerate(build_context_queries(text, mode="sentence"))]
        self._used = 0
        self._counts = dict.fromkeys(_LIMITS, 0)
        self._evidence: dict[str, dict[str, Any]] = {}
        self._finished = False
        self._catalog_semantic = corpus._catalog is not None and corpus._catalog.semantic_enabled
        self._catalog_hints = (corpus._catalog.hints(text)
                               if corpus._catalog is not None and not self._catalog_semantic else None)
        self._catalog_query_vector: list[float] | None = None
        self._initial_passages: tuple[Passage, ...] | None = None

    @property
    def catalog_hints(self) -> dict[str, Any] | None:
        _require(not self._catalog_semantic or self._catalog_hints is not None,
                 "Semantic catalog hints require the original initial search first")
        return deepcopy(self._catalog_hints)

    @property
    def catalog_query_vector(self) -> list[float] | None:
        return deepcopy(self._catalog_query_vector)

    @property
    def focuses(self) -> list[dict[str, Any]]:
        return deepcopy(self._focuses)

    @property
    def budget(self) -> dict[str, Any]:
        return {"used_calls": self._used, "max_calls": 8, "per_tool": dict(self._counts), "max_per_tool": dict(_LIMITS)}

    def _active(self) -> None:
        self._corpus._ensure_open()
        _require(not self._finished, "Research session already finished")

    def _source(self, source: _Source, start: int, end: int, requested: str) -> dict[str, Any]:
        identity = self._corpus.identity_metadata()
        evidence_id = "ev:" + _sha(_json([identity, self._document_hash, source.id, source.text_hash, start, end]))
        record = {"evidence_id": evidence_id, "norm_id": source.id, "requested_id": requested,
                  "fallback": requested != source.id, "titel": source.title,
                  "text": source.text[start:end], "source_start": start, "source_end": end,
                  "source_length": len(source.text), "text_sha256": source.text_hash,
                  "source_fingerprint": identity["source_fingerprint"],
                  "graph_fingerprint": identity["graph_fingerprint"],
                  "next_cursor": end if end < len(source.text) else None,
                  "complete": start == 0 and end == len(source.text)}
        # Requested aliases do not change source identity. Keep the original
        # registration intact when the same physical page is encountered again.
        self._evidence.setdefault(evidence_id, deepcopy(record))
        return record

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        return [self._source(self._corpus._sources[p.norm_id], p.start, p.end, p.norm_id)
                for p in self._corpus._search(query, limit)]

    def initial_sources(self, limit: int = 6) -> list[dict[str, Any]]:
        """Deterministic BM25 start excerpts; only returned pages enter the ledger."""
        self._active()
        limit = _integer(limit, 1, 6, "limit")
        if self._catalog_semantic:
            if self._initial_passages is None:
                # Rank the bounded initial pool once. Smaller views register
                # only their returned prefix, and never require another encode.
                passages, vector = self._corpus._hybrid.search_with_vector(self._text, limit=6)
                hints = self._corpus._catalog.for_text(self._text, query_vector=vector)
                self._catalog_query_vector = vector.tolist()
                self._catalog_hints = hints
                self._initial_passages = tuple(passages)
            return [self._source(self._corpus._sources[p.norm_id], p.start, p.end, p.norm_id)
                    for p in self._initial_passages[:limit]]
        return self._search(self._text, limit)

    def execute(self, name: str, arguments: dict) -> dict[str, Any]:
        self._active()
        if self._used >= 8:
            raise ResearchBudgetExceeded("Research limit of eight tool calls reached")
        known = isinstance(name, str) and name in _LIMITS
        if known and self._counts[name] >= _LIMITS[name]:
            raise ResearchBudgetExceeded(f"Research limit for {name} reached")
        self._used += 1
        if known:
            self._counts[name] += 1
        _require(known, "Unknown research tool")
        _require(isinstance(arguments, dict), "Tool arguments must be an object")
        if name == "search_sources":
            _require(set(arguments) <= {"query", "limit"} and "query" in arguments, "Invalid search_sources fields")
            query = arguments["query"]
            _require(isinstance(query, str) and bool(query.strip()) and len(query) <= 1000,
                     "Search query must contain 1 to 1000 characters")
            limit = _integer(arguments.get("limit", 6), 1, 6, "limit")
            return {"results": self._search(query, limit), "budget": self.budget}
        if name == "read_source":
            _require(set(arguments) <= {"norm_id", "cursor"} and "norm_id" in arguments, "Invalid read_source fields")
            requested = _norm_id(arguments["norm_id"])
            cursor = arguments.get("cursor", 0)
            _require(type(cursor) is int and cursor >= 0, "cursor must be a nonnegative integer")
            source = self._corpus._resolve(requested)
            if source is None or not _live_text(source.text):
                return {"source": None, "requested_id": requested,
                        "status": "missing" if source is None else "not_live", "budget": self.budget}
            _require(cursor < len(source.text), "cursor is outside the original source")
            return {"source": self._source(source, cursor, min(len(source.text), cursor + 4000), requested),
                    "status": "found", "budget": self.budget}
        _require(set(arguments) <= {"norm_id", "direction", "limit"} and "norm_id" in arguments,
                 "Invalid follow_citations fields")
        requested = _norm_id(arguments["norm_id"])
        direction = arguments.get("direction", "out")
        _require(direction in ("out", "in"), "Citation direction must be out or in")
        limit = _integer(arguments.get("limit", 8), 1, 8, "limit")
        source = self._corpus._resolve(requested)
        if source is None:
            return {"citations": [], "requested_id": requested, "resolved_id": None,
                    "status": "missing", "budget": self.budget}
        edges = (self._corpus._outgoing if direction == "out" else self._corpus._incoming).get(source.id, ())
        results = []
        for edge in edges[:limit]:
            target_id = edge["dst"] if direction == "out" else edge["src"]
            target = self._corpus._sources.get(target_id)
            results.append({"src": edge["src"], "dst": edge["dst"], "type": edge["typ"],
                            "confidence": edge["konfidenz"], "direction": direction, "norm_id": target_id,
                            "exact_target_present": target is not None, "declared_present": bool(edge["vorhanden"]),
                            "titel": target.title if target is not None else None})
        return {"citations": results, "requested_id": requested, "resolved_id": source.id,
                "fallback": requested != source.id, "total_citations": len(edges), "has_more": len(edges) > limit,
                "citation_fingerprint": self._corpus.identity_metadata()["citation_fingerprint"], "budget": self.budget}

    def validate_finish(self, arguments: dict) -> list[dict[str, Any]]:
        """Validate the whole proposed conclusion atomically, without truncation."""
        self._active()
        _require(isinstance(arguments, dict) and set(arguments) == {"findings", "unresolved"}, "Invalid finish fields")
        findings, unresolved = arguments["findings"], arguments["unresolved"]
        _require(isinstance(findings, list) and isinstance(unresolved, list) and len(unresolved) <= 8,
                 "finish requires findings and at most eight unresolved entries")
        _require(all(isinstance(v, str) and 1 <= len(v) <= 400 and bool(v.strip()) for v in unresolved),
                 "Unresolved entries must contain 1 to 400 characters")
        focuses = {f["id"]: f for f in self._focuses}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen = set()
        for item in findings:
            text_fields = {"focus_id", "norm_id", "evidence_id", "source_quote", "norm_quote", "reason"}
            _require(isinstance(item, dict) and text_fields <= set(item)
                     and set(item) <= text_fields | {"norm_quote_start", "source_quote_start"},
                     "Invalid finding fields")
            _require(all(isinstance(item[key], str) for key in text_fields), "Finding fields must be strings")
            focus_id, norm_id, evidence_id = item["focus_id"], item["norm_id"], item["evidence_id"]
            _require(focus_id in focuses, "Finding uses an unknown focus")
            _require((focus_id, norm_id) not in seen, "Duplicate norm finding for one focus")
            seen.add((focus_id, norm_id))
            _require(len(grouped[focus_id]) < 5, "At most five norms are allowed per focus")
            _require(evidence_id in self._evidence, "Finding cites evidence never supplied to this session")
            record = self._evidence[evidence_id]
            _require(norm_id == record["norm_id"], "Finding norm differs from its resolved source evidence")
            for key, minimum, maximum in (("norm_quote", 8, 300), ("source_quote", 3, 300), ("reason", 1, 400)):
                _require(minimum <= len(item[key]) <= maximum and bool(item[key].strip()),
                         f"{key} must contain {minimum} to {maximum} characters")
            norm_quote, source_quote = item["norm_quote"], item["source_quote"]
            _require(norm_quote in record["text"], "Norm quote is absent from the supplied original page")
            focus = focuses[focus_id]
            if "source_quote_start" in item:
                source_start = item["source_quote_start"]
                _require(type(source_start) is int and 0 <= source_start
                         and source_start + len(source_quote) <= len(self._text)
                         and self._text[source_start:source_start + len(source_quote)] == source_quote,
                         "Selected source quote offset does not match the original case")
            elif source_quote in focus["text"]:
                source_start = focus["start"] + focus["text"].index(source_quote)
            else:
                source_start = self._text.find(source_quote)
                _require(source_start >= 0 and self._text.find(source_quote, source_start + 1) < 0,
                         "Source quote is absent or ambiguous outside the chosen focus")
            source = self._corpus._sources[norm_id]
            start, end = record["source_start"], record["source_end"]
            _require(source.text[start:end] == record["text"] and source.text_hash == record["text_sha256"],
                     "Original evidence identity changed")
            norm_start = start + record["text"].index(norm_quote)
            if "norm_quote_start" in item:
                norm_start = item["norm_quote_start"]
                _require(type(norm_start) is int and start <= norm_start
                         and norm_start + len(norm_quote) <= end
                         and source.text[norm_start:norm_start + len(norm_quote)] == norm_quote,
                         "Selected norm quote offset does not match the supplied original page")
            grouped[focus_id].append({"id": norm_id, "zitat": knoten_zitat(source.law, source.para), "titel": source.title,
                "sim": None, "rerank": None, "gestuetzt": True, "im_korpus": True, "ketten": [],
                "evidenz": {"fingerprint": record["source_fingerprint"], "gekuerzt": not record["complete"],
                    "passagen": [{"id": f"{norm_id}@{start}:{end}", "norm_id": norm_id,
                                  "quell_start": start, "quell_end": end, "text": record["text"]}], "anker": []},
                "research": {"evidence_id": evidence_id, "reason": item["reason"], "source_verified": True,
                    "legal_validity_verified": False, "text_sha256": record["text_sha256"],
                    "source_quote": {"start": source_start, "end": source_start + len(source_quote), "text": source_quote},
                    "norm_quote": {"start": norm_start, "end": norm_start + len(norm_quote), "text": norm_quote}}})
        result = [{"start": f["start"], "end": f["end"], "satz": f["text"], "normen": grouped[f["id"]]}
                  for f in self._focuses if grouped.get(f["id"])]
        self._finished = True
        return result
