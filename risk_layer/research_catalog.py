"""Bounded catalog search hints; never original-source evidence or legal findings."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import lade_katalog
from .embedding import katalog_fingerprint_berechnen
from .ingest import ingest
from .matcher import Schicht1Matcher
from .muster import MusterMatcher
from .passages import PassageEvidence, catalog_anchors
from .idschema import para_level
from .zitate import parse_normzitat


PURPOSE = (
    "Ungeprüfte Suchhinweise aus dem Katalog, keine Belege. Genannte Normen müssen erst "
    "über die Quellenwerkzeuge gelesen werden; Fundstellen und Normketten sind keine "
    "gelesenen Graphverweise."
)
_POLICY: dict[str, Any] = {
    "matching": "literal_and_pattern_without_lemma",
    "order": "first_original_position_then_concept_id",
    "max_concepts": 4,
    "max_matches": 4,
    "max_norm_anchors": 4,
    "max_definition_references": 4,
    "max_norm_chains": 2,
    "max_chain_members": 8,
    "max_match_chars": 400,
    "max_citation_chars": 400,
    "max_term_chars": 160,
    "max_concept_id_chars": 160,
    "max_type_chars": 80,
    "max_relation_chars": 80,
    "max_note_chars": 400,
    "max_concept_json_chars": 1600,
    "size_reduction_order": ["norm_chains", "definition_references", "norm_anchors", "matches"],
    "overflow": "omit_whole_items_and_count",
    "source_evidence": False,
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _bounded(value: str, maximum: int, *, allow_empty: bool = False) -> bool:
    return isinstance(value, str) and len(value) <= maximum and (allow_empty or bool(value.strip()))


class ResearchCatalog:
    """One loaded YAML snapshot, tied to the index's existing catalog identity."""

    def __init__(self, path: Path, *, expected_fingerprint: str | None,
                 evidence: PassageEvidence | None = None):
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        catalog = lade_katalog(path)
        if path.read_bytes() != raw:
            raise ValueError("Research catalog changed while loading its snapshot")
        semantic_fingerprint = katalog_fingerprint_berechnen(catalog)
        if semantic_fingerprint != expected_fingerprint:
            raise ValueError("Research catalog and passage index belong to different catalog generations")
        # No caller-owned mutable catalog entries or matcher state are exposed.
        self._catalog = catalog
        self._literal = Schicht1Matcher(catalog)
        self._pattern = MusterMatcher(catalog)
        self._metadata: dict[str, Any] = {
            "format": "research_catalog/1", "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw), "version": str(catalog.version),
            "semantic_fingerprint": semantic_fingerprint, "policy": deepcopy(_POLICY),
        }
        self._semantic_vectors: np.ndarray | None = None
        self._semantic_anchors: tuple[tuple[str, str], ...] = ()
        self._anchor_owners: tuple[tuple[str, ...], ...] = ()
        if evidence is not None:
            self._prepare_semantic(evidence, semantic_fingerprint)
        self._metadata["fingerprint"] = _digest(self._metadata)
        # Paths describe the frozen input location, not its portable identity.
        self._metadata["path"] = str(path)

    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    @property
    def semantic_enabled(self) -> bool:
        return self._semantic_vectors is not None

    def _prepare_semantic(self, evidence: PassageEvidence, fingerprint: str) -> None:
        if not isinstance(evidence, PassageEvidence):
            raise ValueError("Semantic catalog hints require validated passage evidence")
        metadata, vectors = evidence.storage()
        frozen = PassageEvidence(metadata, vectors)
        if metadata["catalog_fingerprint"] != fingerprint:
            raise ValueError("Catalog semantic anchor generation differs")
        covered = {passage.norm_id for passage in frozen.passages}
        expected = tuple((norm_id, phrase) for norm_id, phrases in catalog_anchors(self._catalog).items()
                         if norm_id in covered for phrase in phrases)
        if frozen.anchors != expected:
            raise ValueError("Stored norm/paraphrase anchors differ from the full YAML catalog")
        owners: dict[tuple[str, str], set[str]] = defaultdict(set)
        for entry in self._catalog:
            for norm_id, phrases in catalog_anchors([entry]).items():
                for phrase in phrases:
                    owners[norm_id, phrase].add(entry.id)
        if any(not owners[pair] for pair in frozen.anchors):
            raise ValueError("Stored paraphrase has no exact YAML concept owner")
        matrix = np.array(frozen.anchor_vectors, dtype="<f4", order="C", copy=True)
        if len(matrix) and not np.allclose(np.linalg.norm(matrix, axis=1), 1., atol=1e-4, rtol=0.):
            raise ValueError("Stored semantic anchor vectors are not normalized")
        matrix.flags.writeable = False
        self._semantic_vectors = matrix
        self._semantic_anchors = frozen.anchors
        self._anchor_owners = tuple(tuple(sorted(owners[pair])) for pair in frozen.anchors)
        self._metadata["format"] = "research_catalog/2"
        self._metadata["semantic"] = {
            "evidence_fingerprint": frozen.fingerprint, "anchor_rows": len(matrix),
            "anchor_records_sha256": _digest(frozen.anchors),
            "anchor_vectors_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
            "encoder": metadata["encoder"], "encoder_revision": metadata["encoder_revision"],
            "query_vector": {"dtype": "float32", "dimension": frozen.dim, "byte_order": "little", "order": "C"},
        }
        self._metadata["policy"].update({
            "matching": "literal_pattern_then_stored_semantic_anchors",
            "semantic_order": "descending_score_then_norm_id_then_concept_id_then_anchor_row",
            "semantic_diversity": "unrepresented_anchor_norm_first_then_remaining_score_order",
            "semantic_score": "normalized_vector_dot_product_not_probability",
            "semantic_min_score": None, "max_semantic_matches": 1,
            "literal_first": True,
        })

    def for_text(self, text: str, query_vector=None) -> dict[str, Any]:
        return self.hints(text, query_vector=query_vector)

    def _query(self, vector) -> np.ndarray:
        if isinstance(vector, list):
            if not all(type(value) is float for value in vector):
                raise ValueError("Catalog query list must contain exact Python floats")
            array = np.asarray(vector, dtype="<f4")
            if array.tolist() != vector:
                raise ValueError("Catalog query list is not an exact float32 roundtrip")
        elif isinstance(vector, np.ndarray) and vector.dtype == np.dtype("<f4"):
            array = np.ascontiguousarray(vector)
        else:
            raise ValueError("Catalog query must be an original float32 vector")
        dimension = self._metadata["semantic"]["query_vector"]["dimension"]
        if (array.shape != (dimension,) or not np.isfinite(array).all()
                or abs(float(np.linalg.norm(array)) - 1.) >= 1e-4):
            raise ValueError("Catalog query dimension, finite values or normalization invalid")
        return array

    @staticmethod
    def _references(entry) -> tuple[dict[str, list], dict[str, int]]:
        anchors = [value for value in entry.norm_anker
                   if _bounded(value, _POLICY["max_citation_chars"])][:_POLICY["max_norm_anchors"]]
        definitions = [source.as_dict() for source in entry.definitionsquelle
                       if _bounded(source.typ, _POLICY["max_type_chars"])
                       and _bounded(source.fundstelle, _POLICY["max_citation_chars"])]
        definitions = definitions[:_POLICY["max_definition_references"]]
        chains = []
        for chain in entry.normketten:
            # Never shorten a chain into a different claim about its members.
            if (len(chain.glieder) <= _POLICY["max_chain_members"]
                    and all(_bounded(member, _POLICY["max_citation_chars"]) for member in chain.glieder)
                    and _bounded(chain.verknuepfung, _POLICY["max_relation_chars"])
                    and _bounded(chain.hinweis, _POLICY["max_note_chars"], allow_empty=True)):
                chains.append({"glieder": list(chain.glieder), "verknuepfung": chain.verknuepfung,
                               "hinweis": chain.hinweis})
        chains = chains[:_POLICY["max_norm_chains"]]
        references = {"norm_anchors": anchors, "definition_references": definitions, "norm_chains": chains}
        omitted = {"norm_anchors": len(entry.norm_anker) - len(anchors),
                   "definition_references": len(entry.definitionsquelle) - len(definitions),
                   "norm_chains": len(entry.normketten) - len(chains)}
        return references, omitted

    def _literal_hints(self, text: str, *, ids_out: set[str] | None = None) -> dict[str, Any]:
        if not isinstance(text, str):
            raise ValueError("Research catalog matching requires original text")
        original = ingest(text)
        # Combine identical spans; distinct occurrences remain independently bound.
        matches: dict[str, dict[tuple[int, int], set[str]]] = defaultdict(lambda: defaultdict(set))
        for hit in self._literal.match(original):
            matches[hit.begriff_id][hit.start, hit.end].add("literal")
        for pattern_hit in self._pattern.match(original.normalized_text):
            if (type(pattern_hit.start) is not int or type(pattern_hit.end) is not int
                    or not 0 <= pattern_hit.start < pattern_hit.end <= len(original.offset_map)):
                raise ValueError("Research catalog pattern has invalid normalized offsets")
            start = original.offset_map[pattern_hit.start]
            end = original.offset_map[pattern_hit.end - 1] + 1
            if not 0 <= start < end <= len(text):
                raise ValueError("Research catalog pattern has invalid original offsets")
            matches[pattern_hit.begriff_id][start, end].add("pattern")
        ordered_ids = sorted(matches, key=lambda concept_id: (min(matches[concept_id])[0], concept_id))
        if ids_out is not None:
            ids_out.update(matches)
        concepts: list[dict[str, Any]] = []
        for concept_id in ordered_ids:
            if len(concepts) == _POLICY["max_concepts"]:
                break
            entry = self._catalog.get(concept_id)
            if (not _bounded(concept_id, _POLICY["max_concept_id_chars"])
                    or not _bounded(entry.begriff_kanonisch, _POLICY["max_term_chars"])):
                continue
            occurrences = [{"start": start, "end": end, "text": text[start:end], "methods": sorted(methods)}
                           for (start, end), methods in sorted(matches[concept_id].items())
                           if _bounded(text[start:end], _POLICY["max_match_chars"])]
            occurrences = occurrences[:_POLICY["max_matches"]]
            if not occurrences:
                continue
            references, omitted = self._references(entry)
            concept: dict[str, Any] = {"hint_id": f"k{len(concepts) + 1}", "status": "unverified_hint",
                       "concept_id": concept_id, "term": entry.begriff_kanonisch,
                       "matches": occurrences, **references,
                       "omitted": {"matches": len(matches[concept_id]) - len(occurrences), **omitted}}
            # Keep complete data items, including complete original quotes and
            # chains. A large catalog card must not consume the whole dialog.
            for field in _POLICY["size_reduction_order"]:
                minimum = 1 if field == "matches" else 0
                while len(_json(concept)) > _POLICY["max_concept_json_chars"] and len(concept[field]) > minimum:
                    concept[field].pop()
                    concept["omitted"][field] += 1
            if len(_json(concept)) <= _POLICY["max_concept_json_chars"]:
                concepts.append(concept)
        return {"format": "research_catalog_hints/1", "status": "unverified_hint", "purpose": PURPOSE,
                "catalog_fingerprint": self._metadata["fingerprint"], "concepts": concepts,
                "total_concepts": len(matches), "omitted_concepts": len(matches) - len(concepts)}

    def hints(self, text: str, query_vector=None) -> dict[str, Any]:
        if not self.semantic_enabled:
            if query_vector is not None:
                raise ValueError("Literal catalog hints do not consume query vectors")
            return self._literal_hints(text)
        vector = self._query(query_vector)
        literal_ids: set[str] = set()
        result = self._literal_hints(text, ids_out=literal_ids)
        assert self._semantic_vectors is not None
        scores = self._semantic_vectors @ vector
        ranked = sorted(((-float(scores[row]), norm_id, concept_id, row)
                         for row, (norm_id, _) in enumerate(self._semantic_anchors)
                         for concept_id in self._anchor_owners[row]))
        concepts = result["concepts"]
        selected = {concept["concept_id"] for concept in concepts}
        # Count the literal union before caps, including unrenderable concepts.
        all_ids = literal_ids | {concept_id for _, _, concept_id, _ in ranked}
        used_norms = {para_level(anchor.norm_id.to_id()) for concept in concepts
                      for citation in concept["norm_anchors"] for anchor in parse_normzitat(citation)
                      if anchor.norm_id is not None}
        literal_count = len(concepts)
        for diverse in (True, False):
            for negative_score, norm_id, concept_id, row in ranked:
                if len(concepts) == _POLICY["max_concepts"]:
                    break
                if concept_id in selected or (diverse and norm_id in used_norms):
                    continue
                entry = self._catalog.get(concept_id)
                anchor_text = self._semantic_anchors[row][1]
                if (not _bounded(concept_id, _POLICY["max_concept_id_chars"])
                        or not _bounded(entry.begriff_kanonisch, _POLICY["max_term_chars"])
                        or not _bounded(anchor_text, _POLICY["max_citation_chars"])):
                    continue
                references, omitted = self._references(entry)
                concept: dict[str, Any] = {"hint_id": f"k{len(concepts) + 1}", "status": "unverified_hint",
                           "concept_id": concept_id, "term": entry.begriff_kanonisch,
                           "matching": "semantic_anchor", "matches": [],
                           "semantic_matches": [{"norm_id": norm_id, "anchor_text": anchor_text,
                                                 "score": -negative_score}],
                           **references, "omitted": {"matches": 0, **omitted}}
                for field in _POLICY["size_reduction_order"]:
                    while len(_json(concept)) > _POLICY["max_concept_json_chars"] and concept[field]:
                        concept[field].pop()
                        concept["omitted"][field] += 1
                if len(_json(concept)) <= _POLICY["max_concept_json_chars"]:
                    concepts.append(concept)
                    selected.add(concept_id)
                    used_norms.add(norm_id)
        result.update({"format": "research_catalog_hints/2", "total_concepts": len(all_ids),
                       "omitted_concepts": len(all_ids) - len(concepts),
                       "literal_concepts": literal_count, "semantic_concepts": len(concepts) - literal_count})
        return result
