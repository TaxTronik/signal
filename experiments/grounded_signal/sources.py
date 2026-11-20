"""Whole local rule units for the isolated grounded-Signal experiment.

Completeness concerns the stored original parent norm, never legal applicability,
currency, or completeness of a legal dependency graph. No model is downloaded;
the optional existing BGE research retriever explicitly executes on the CPU.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from typing import Any, Iterable


FORMAT = "grounded_signal_sources/1"
_ID = re.compile(r"norm:[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9][A-Za-z0-9.-]*(?::(?:abs|nr|s|lit|hs|alt)[A-Za-z0-9.-]+)*")
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


class SourceError(ValueError):
    """Invalid source identity, integrity, or explicit packaging request."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise SourceError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _norm_id(value: str) -> str:
    _require(isinstance(value, str) and len(value) <= 200 and _ID.fullmatch(value) is not None,
             "Expected a canonical norm identifier")
    return ":".join(value.split(":")[:3])


def package_fingerprint(payload: dict[str, Any]) -> str:
    """One shared content-hash rule for renderers, runners, and evaluators."""
    return _hash(canonical_json({k: v for k, v in payload.items() if k not in {"package_id", "fingerprint"}}))


@dataclass(frozen=True)
class SourceReference:
    source_node_id: str
    source_norm_id: str
    target_id: str
    resolved_norm_id: str | None
    relation: str
    confidence: str
    exact_target_available: bool
    origin: str = "stored_graph_edge"


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    norm_id: str
    original_node_id: str
    start: int
    end: int
    text: str
    title: str
    law: str
    kind: str
    whole_rule_id: str
    norm_text_sha256: str
    valid_from: str | None
    valid_until: str | None
    source_type: str


@dataclass(frozen=True)
class SourceQuote:
    unit_id: str
    norm_id: str
    start: int
    end: int
    text: str
    whole_unit: bool


@dataclass(frozen=True)
class SearchHit:
    norm_id: str
    rank: int
    score: float | None
    matched_unit_ids: tuple[str, ...]
    method: str


@dataclass(frozen=True)
class EvidencePackage:
    package_id: str
    fingerprint: str
    status: str
    candidate_norm_ids: tuple[str, ...]
    norm_ids: tuple[str, ...]
    units: tuple[EvidenceUnit, ...]
    references: tuple[SourceReference, ...]
    required_groups: tuple[tuple[str, tuple[str, ...]], ...]
    limitations: tuple[str, ...]
    missing_norm_ids: tuple[str, ...]
    overflow_norm_ids: tuple[str, ...]
    complete_local_norms: bool
    reference_closure_complete: bool
    reference_depth: int
    source_chars: int
    required_source_chars: int
    max_chars: int
    corpus_fingerprint: str


def _package_payload(package: EvidencePackage) -> dict[str, Any]:
    return {
        "format": "grounded_signal_evidence_package/1",
        "status": package.status,
        "candidate_norm_ids": list(package.candidate_norm_ids),
        "norm_ids": list(package.norm_ids),
        "unit_ids": [u.unit_id for u in package.units],
        "units": [asdict(u) for u in package.units],
        "references": [asdict(r) for r in package.references],
        "required_groups": [{"norm_id": n, "unit_ids": list(ids), "policy": "all_stored_parent_norm_text"}
                            for n, ids in package.required_groups],
        "limitations": list(package.limitations),
        "missing_norm_ids": list(package.missing_norm_ids),
        "overflow_norm_ids": list(package.overflow_norm_ids),
        "complete_local_norms": package.complete_local_norms,
        "reference_closure_complete": package.reference_closure_complete,
        "reference_closure_scope": "stored_graph_edges_only",
        "legal_completeness": "unassessed",
        "reference_depth": package.reference_depth,
        "source_chars": package.source_chars,
        "required_source_chars": package.required_source_chars,
        "max_chars": package.max_chars,
        "budget_unit": "original_text_characters_not_model_tokens",
        "corpus_fingerprint": package.corpus_fingerprint,
    }


def render_package(package: EvidencePackage) -> dict[str, Any]:
    """Return a fresh JSON payload; overflow/missing packages contain no units."""
    payload = _package_payload(package)
    expected = package_fingerprint(payload)
    _require(package.fingerprint == expected and package.package_id == "pkg:" + expected,
             "Evidence package content/identity differs")
    return {"package_id": package.package_id, "fingerprint": expected, **payload}


class SourceCorpus:
    """Immutable local original norms, required whole-norm groups, and search."""

    def __init__(self, graph_path: str | Path, index_dir: str | Path | None = None, *,
                 retrieval: str = "bm25", encoder_model_path: str | Path | None = None):
        _require(retrieval in {"bm25", "hybrid"}, "Retrieval must be bm25 or hybrid")
        _require(retrieval == "hybrid" or encoder_model_path is None, "BM25 does not use a model")
        self.graph_path = Path(graph_path).resolve()
        _require(self.graph_path.is_file(), "Original graph is missing")
        self._file_sha256 = _file_hash(self.graph_path)
        connection = sqlite3.connect(self.graph_path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN")
            rows = [dict(r) for r in connection.execute("SELECT * FROM knoten ORDER BY id")]
            edges = [dict(r) for r in connection.execute(
                "SELECT src,dst,typ,konfidenz,vorhanden FROM kanten WHERE typ='verweist_auf' ORDER BY src,dst,typ")]
        finally:
            connection.close()
        _require(_file_hash(self.graph_path) == self._file_sha256, "Graph changed while loading")
        self._nodes_json = canonical_json(rows)
        self._edges_json = canonical_json(edges)
        self._fingerprint = _hash(canonical_json({"format": FORMAT, "nodes": rows, "edges": edges}))
        self._nodes = MappingProxyType({r["id"]: MappingProxyType(r) for r in rows})
        _require(len(self._nodes) == len(rows), "Duplicate original node IDs")
        parents = {r["id"]: r for r in rows if _norm_id(r["id"]) == r["id"]}
        self._norms = MappingProxyType(parents)
        self._paragraphs: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row["id"].count(":") == 3 and row["id"].split(":")[3].startswith("abs"):
                self._paragraphs.setdefault(_norm_id(row["id"]), []).append(row)
        self._units: dict[str, tuple[EvidenceUnit, ...]] = {}
        self._structure: dict[str, tuple[str, ...]] = {}
        self._references: dict[str, tuple[SourceReference, ...]] = {}
        self._unit_by_id: dict[str, EvidenceUnit] = {}
        self._retrieval = retrieval
        self._legacy: Any = None
        self._closed = False
        self._lexical: tuple[Any, ...] | None = None
        for nid, parent in self._norms.items():
            units, warnings = self._make_units(parent)
            self._units[nid], self._structure[nid] = units, warnings
            self._unit_by_id.update((unit.unit_id, unit) for unit in units)
        outgoing: dict[str, list[SourceReference]] = {}
        for edge in edges:
            source, target = edge["src"], edge["dst"]
            source_norm = _norm_id(source)
            # Graph edges sometimes describe open/range IDs not resolvable here.
            try:
                target_norm = _norm_id(target)
            except SourceError:
                target_norm = ""
            exact = target in self._nodes and bool(self._nodes[target].get("text"))
            resolved = target_norm if target_norm in self._norms and bool(self._units[target_norm]) else None
            _require(bool(edge["vorhanden"]) == (target in self._nodes), "Stored edge availability differs from nodes")
            outgoing.setdefault(source_norm, []).append(SourceReference(
                source, source_norm, target, resolved, edge["typ"], edge["konfidenz"], exact))
        self._references = {n: tuple(rs) for n, rs in outgoing.items()}
        self.index_dir = Path(index_dir).resolve() if index_dir is not None else None
        if retrieval == "hybrid":
            _require(self.index_dir is not None, "Hybrid retrieval requires its existing original-passage index")
            assert self.index_dir is not None
            from risk_layer.research_sources import ResearchCorpus
            self._legacy = ResearchCorpus(self.graph_path, self.index_dir, retrieval="hybrid",
                encoder_model_path=Path(encoder_model_path) if encoder_model_path is not None else None)
            _require(_file_hash(self.graph_path) == self._file_sha256, "Graph changed before optional retrieval binding")

    def _make_units(self, parent: dict[str, Any]) -> tuple[tuple[EvidenceUnit, ...], tuple[str, ...]]:
        nid, text = parent["id"], parent.get("text") or ""
        _require(isinstance(text, str), "Original norm text is not a string")
        if not text.strip():
            return (), ("stored_parent_text_empty",)
        children = self._paragraphs.get(nid, [])
        intervals: list[tuple[int, int, str]] = []
        warnings: list[str] = []
        for child in children:
            child_text = child.get("text") or ""
            start = text.find(child_text) if child_text else -1
            if start < 0 or text.find(child_text, start + 1) >= 0:
                warnings.append("paragraph_embedding_missing_or_ambiguous")
                break
            intervals.append((start, start + len(child_text), child["id"]))
        intervals.sort()
        if intervals and not warnings:
            valid = intervals[0][0] == 0 and intervals[-1][1] == len(text)
            valid = valid and all(a[1] <= b[0] and not text[a[1]:b[0]].strip()
                                  for a, b in zip(intervals, intervals[1:]))
            if not valid:
                warnings.append("paragraph_structure_does_not_cover_whole_parent")
        elif not children:
            warnings.append("paragraph_structure_unavailable_whole_parent_used")
        if warnings or not intervals:
            intervals = [(0, len(text), nid)]
        else:
            # Include intervening original whitespace, never reconstruct strings.
            intervals = [(a, intervals[i + 1][0] if i + 1 < len(intervals) else b, child)
                         for i, (a, b, child) in enumerate(intervals)]
        norm_hash = _hash(text)
        whole_rule = "rule:" + _hash(canonical_json([self._fingerprint, nid, norm_hash]))
        units = []
        for start, end, child_id in intervals:
            kind = "paragraph" if child_id != nid else "whole_norm"
            identity = _hash(canonical_json([self._fingerprint, nid, child_id, start, end, norm_hash, kind]))
            units.append(EvidenceUnit("unit:" + identity, nid, child_id, start, end, text[start:end],
                parent.get("titel") or "", parent.get("law") or "", kind, whole_rule, norm_hash,
                parent.get("gueltig_ab"), parent.get("gueltig_bis"), parent.get("quellentyp") or ""))
        _require("".join(u.text for u in units) == text, "Whole-parent coverage differs")
        return tuple(units), tuple(sorted(set(warnings)))

    def _ensure_open(self) -> None:
        _require(not self._closed, "Source corpus is closed")

    def units(self, norm_id: str) -> tuple[EvidenceUnit, ...]:
        self._ensure_open()
        nid = _norm_id(norm_id)
        _require(nid in self._norms and bool(self._units[nid]), "A nonempty original parent norm is unavailable")
        return self._units[nid]

    def quote(self, unit_id: str, start: int = 0, end: int | None = None) -> SourceQuote:
        """Offsets are relative to a supplied unit; the default returns it whole."""
        self._ensure_open()
        _require(isinstance(unit_id, str) and unit_id in self._unit_by_id, "Unknown evidence unit")
        unit = self._unit_by_id[unit_id]
        stop = len(unit.text) if end is None else end
        _require(type(start) is int and type(stop) is int and 0 <= start < stop <= len(unit.text), "Invalid quote offsets")
        return SourceQuote(unit_id, unit.norm_id, unit.start + start, unit.start + stop,
                           unit.text[start:stop], start == 0 and stop == len(unit.text))

    def package(self, norm_ids: Iterable[str], *, max_chars: int = 32000,
                reference_depth: int = 0) -> EvidencePackage:
        self._ensure_open()
        _require(not isinstance(norm_ids, (str, bytes)), "Pass an explicit collection of norm IDs")
        requested = tuple(norm_ids)
        _require(1 <= len(requested) <= 50, "A package requires one to fifty explicit norm IDs")
        _require(type(max_chars) is int and max_chars > 0, "max_chars must be a positive integer")
        _require(type(reference_depth) is int and 0 <= reference_depth <= 3, "reference_depth must be from zero to three")
        candidates = tuple(sorted({_norm_id(nid) for nid in requested}))
        included = set(candidates)
        frontier = set(candidates)
        for _ in range(reference_depth):
            following = {r.resolved_norm_id for nid in frontier for r in self._references.get(nid, ())
                         if r.resolved_norm_id is not None} - included
            included.update(following)
            frontier = following
        ordered = tuple(sorted(included))
        missing = tuple(nid for nid in ordered if nid not in self._units or not self._units[nid])
        references = tuple(r for nid in ordered for r in self._references.get(nid, ()))
        open_refs = [r for r in references if not r.exact_target_available or r.resolved_norm_id not in included]
        all_units = tuple(u for nid in ordered for u in self._units.get(nid, ()))
        required_chars = sum(len(u.text) for u in all_units)
        overflow = required_chars > max_chars
        status = "missing" if missing else "overflow" if overflow else "ready"
        limitations = ["legal_dependency_completeness_unassessed", "validity_dates_are_stored_metadata_not_current_law_verification"]
        limitations.extend(f"{nid}:{warning}" for nid in ordered for warning in self._structure.get(nid, ()))
        if any(nid != _norm_id(nid) for nid in requested):
            limitations.append("requested_subunits_expanded_to_whole_parent_norms")
        if open_refs:
            limitations.append("stored_references_remain_unresolved_or_outside_package")
        if missing:
            limitations.append("requested_original_parent_missing_or_empty")
        if overflow:
            limitations.append("whole_package_exceeds_source_character_budget_no_partial_payload")
        delivered = all_units if status == "ready" else ()
        groups = tuple((nid, tuple(u.unit_id for u in self._units.get(nid, ()))) for nid in ordered)
        package = EvidencePackage(package_id="", fingerprint="", status=status,
            candidate_norm_ids=candidates, norm_ids=ordered, units=delivered,
            references=references, required_groups=groups, limitations=tuple(sorted(set(limitations))),
            missing_norm_ids=missing, overflow_norm_ids=ordered if overflow else (),
            complete_local_norms=status == "ready", reference_closure_complete=not open_refs and status == "ready",
            reference_depth=reference_depth, source_chars=sum(len(u.text) for u in delivered),
            required_source_chars=required_chars, max_chars=max_chars, corpus_fingerprint=self._fingerprint)
        fingerprint = package_fingerprint(_package_payload(package))
        return replace(package, package_id="pkg:" + fingerprint, fingerprint=fingerprint)

    def prepare_lookup(self) -> dict[str, Any]:
        """Build lexical statistics; explicitly warm a requested existing CPU BGE."""
        self._ensure_open()
        if self._legacy is not None:
            self._legacy.prepare()
        if self._lexical is None:
            units = tuple(self._unit_by_id.values())
            counts = tuple(Counter(_WORDS.findall((u.title + " " + u.text).casefold())) for u in units)
            frequency: Counter[str] = Counter()
            for count in counts:
                frequency.update(count.keys())
            average = sum(sum(c.values()) for c in counts) / max(1, len(counts))
            self._lexical = units, counts, frequency, average
        return self.retrieval_metadata()

    def search(self, query: str, limit: int = 6) -> tuple[SearchHit, ...]:
        self._ensure_open()
        _require(isinstance(query, str) and bool(query.strip()), "A nonempty search query is required")
        _require(type(limit) is int and 1 <= limit <= 6, "Search limit must be from one to six")
        if self._legacy is not None:
            passages = self._legacy._search(query, limit)
            hits: list[SearchHit] = []
            seen = set()
            for passage in passages:
                nid = _norm_id(passage.norm_id)
                if nid in seen or nid not in self._units or not self._units[nid]:
                    continue
                seen.add(nid)
                matches = tuple(u.unit_id for u in self._units[nid] if u.start < passage.end and passage.start < u.end)
                hits.append(SearchHit(nid, len(hits) + 1, None, matches, "existing_cpu_bge_passage_bm25_rrf60"))
            return tuple(hits)
        self.prepare_lookup()
        assert self._lexical is not None
        units, counts, frequency, average = self._lexical
        terms = set(_WORDS.findall(query.casefold()))
        best: dict[str, tuple[float, EvidenceUnit]] = {}
        for unit, count in zip(units, counts):
            length = sum(count.values())
            score = sum(math.log(1 + (len(units) - frequency[t] + .5) / (frequency[t] + .5))
                        * count[t] * 2.2 / (count[t] + 1.2 * (.25 + .75 * length / max(average, 1)))
                        for t in terms if count[t])
            if score > 0 and (unit.norm_id not in best or score > best[unit.norm_id][0]):
                best[unit.norm_id] = score, unit
        selected = sorted(best, key=lambda nid: (-best[nid][0], nid))[:limit]
        return tuple(SearchHit(nid, rank, best[nid][0], (best[nid][1].unit_id,), "bm25_whole_units_max_norm")
                     for rank, nid in enumerate(selected, 1))

    def retrieval_metadata(self) -> dict[str, Any]:
        self._ensure_open()
        if self._legacy is not None:
            return self._legacy.retrieval_metadata()
        return {"mode": "bm25", "policy": {"unit": "whole_original_paragraph_or_parent", "norm_score": "max_unit",
                "k1": 1.2, "b": .75, "title_included": True, "anchors": False, "models_loaded": False}}

    @property
    def retrieval_metrics(self) -> dict[str, Any]:
        self._ensure_open()
        return self._legacy.retrieval_metrics if self._legacy is not None else {}

    def metadata(self) -> dict[str, Any]:
        self._ensure_open()
        return {"format": FORMAT, "fingerprint": self._fingerprint, "graph_path": str(self.graph_path),
                "graph_file_sha256": self._file_sha256, "nodes_sha256": _hash(self._nodes_json),
                "edges_sha256": _hash(self._edges_json), "norms": len(self._norms),
                "units": len(self._unit_by_id), "index_dir": str(self.index_dir) if self.index_dir else None,
                "retrieval": self.retrieval_metadata()}

    def export_snapshot(self, directory: str | Path) -> dict[str, Any]:
        """Freeze original nodes/edges/units and retrieval metadata/journal as JSON.

        The runner must separately bind/copy an optional original dense index and
        model files named in metadata; this export never duplicates model weights.
        """
        self._ensure_open()
        out = Path(directory)
        _require(not out.exists(), "Snapshot directory must be fresh")
        out.mkdir(parents=True)
        artifacts = {
            "nodes.json": self._nodes_json,
            "edges.json": self._edges_json,
            "units.json": canonical_json([asdict(u) for u in self._unit_by_id.values()]),
            "retrieval-metrics.json": canonical_json(self.retrieval_metrics),
        }
        for name, content in artifacts.items():
            (out / name).write_bytes(content.encode("utf-8"))
        manifest = {"format": "grounded_signal_source_snapshot/1", "metadata": self.metadata(),
                    "artifacts": {name: _hash(content) for name, content in artifacts.items()},
                    "contains_model_or_index_vectors": False}
        (out / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
        return manifest

    def close(self) -> None:
        if not self._closed and self._legacy is not None:
            self._legacy.close()
        self._closed = True

    def __enter__(self) -> SourceCorpus:
        self._ensure_open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
