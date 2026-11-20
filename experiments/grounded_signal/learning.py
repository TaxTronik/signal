"""Small, deterministic diagnostic learner; no model runtime or automatic gold labels.

The shared norm corpus supplies lexical features. Only explicitly designated
development families supply contrast labels. Scores are retrieval similarities,
not probabilities, legal conclusions, or a decision to suppress a candidate.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence


FORMAT = "grounded_signal_learning/1"
ADAPTATION_WEIGHT = 0.25
LABEL_PROVENANCE = "assistant_reviewed_synthetic_diagnostic_not_legal_gold"


class LearningError(ValueError):
    """The fixed train-only protocol cannot be respected."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearningError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _features(text: str) -> Counter[str]:
    # Unicode word trigrams; IDs and family membership never become features.
    tokens = re.findall(r"[^\W_]+", _text_key(text), flags=re.UNICODE)
    return Counter(gram for word in tokens for padded in ["^" + word + "$"]
                   for gram in (padded[i:i + 3] for i in range(max(1, len(padded) - 2))))


def _vector(text: str, idf: Mapping[str, float]) -> dict[str, float]:
    values = {term: (1.0 + math.log(count)) * idf[term]
              for term, count in _features(text).items() if term in idf}
    length = math.sqrt(sum(value * value for value in values.values()))
    return {term: value / length for term, value in values.items()} if length else {}


def _dot(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def _unique_strings(value: Any, name: str) -> list[str]:
    _require(isinstance(value, (list, tuple)) and all(isinstance(x, str) and x for x in value), name)
    _require(len(value) == len(set(value)), name + " contains duplicates")
    return list(value)


def fit_examples(
    cases: Sequence[Mapping[str, Any]], *, development_families: Sequence[str],
    evaluation_families: Sequence[str], norm_documents: Mapping[str, str],
) -> dict[str, Any]:
    """Build a JSON artifact with a fixed lexical model and explicit contrasts.

    ``diagnostic_labels`` uses ``target_norm_ids`` for positive contrasts and
    optional ``no_hint_norm_ids`` for explicit negative contrasts. Missing
    positive targets NEVER create negatives. Evaluation labels are not read.
    All texts may be examined solely for the train/evaluation duplicate guard.
    """
    development = set(_unique_strings(development_families, "development families"))
    evaluation = set(_unique_strings(evaluation_families, "evaluation families"))
    _require(bool(development) and bool(evaluation) and not development & evaluation,
             "Development and evaluation families must be nonempty and disjoint")
    _require(bool(norm_documents) and all(isinstance(k, str) and k and isinstance(v, str) and v
                                         for k, v in norm_documents.items()), "Invalid norm corpus")
    seen_ids: set[str] = set()
    family_seen: set[str] = set()
    texts: dict[str, set[str]] = {"development": set(), "evaluation": set()}
    train: list[Mapping[str, Any]] = []
    for case in cases:
        cid, family, text = case.get("case_id"), case.get("family_id"), case.get("text")
        _require(isinstance(cid, str) and bool(cid) and cid not in seen_ids, "Invalid/duplicate case ID")
        _require(isinstance(family, str) and family in development | evaluation, "Undeclared family")
        _require(isinstance(text, str) and bool(text.strip()), "Empty case text")
        assert isinstance(cid, str) and isinstance(family, str) and isinstance(text, str)
        side = "development" if family in development else "evaluation"
        _require(case.get("split") == side, "Case split conflicts with declared family split")
        seen_ids.add(cid)
        family_seen.add(family)
        texts[side].add(_text_key(text))
        if side == "development":
            train.append(case)
    _require(family_seen == development | evaluation, "Declared family has no cases")
    _require(not texts["development"] & texts["evaluation"], "Train/evaluation text duplicate")

    # Vocabulary and IDF derive only from the fixed shared source corpus.
    documents = {norm: norm_documents[norm] for norm in sorted(norm_documents)}
    frequencies: Counter[str] = Counter()
    for text in documents.values():
        frequencies.update(_features(text).keys())
    idf = {term: 1.0 + math.log((len(documents) + 1) / (count + 1))
           for term, count in sorted(frequencies.items())}
    contrasts: list[dict[str, Any]] = []
    deduplicated: dict[tuple[str, str], str] = {}
    for case in train:
        labels = case.get("diagnostic_labels")
        _require(isinstance(labels, dict), "Development diagnostic labels missing")
        assert isinstance(labels, dict)
        _require(labels.get("annotation_status") == "assistant_reviewed_synthetic",
                 "Development label provenance must be explicit")
        targets = _unique_strings(labels.get("target_norm_ids"), "target norm IDs")
        negatives = _unique_strings(labels.get("no_hint_norm_ids", []), "explicit no-hint norm IDs")
        prepared = set(_unique_strings(case.get("prepared_norm_ids"), "prepared norm IDs"))
        _require(not set(targets) & set(negatives), "Conflicting positive/negative contrast")
        _require(set(targets + negatives) <= prepared <= documents.keys(), "Unbound training norm")
        for action, norms in (("hint", targets), ("no_hint", negatives)):
            for norm in norms:
                key = (_text_key(case["text"]), norm)
                _require(key not in deduplicated or deduplicated[key] == action,
                         "Duplicate text/norm has conflicting diagnostic labels")
                if key in deduplicated:
                    continue
                deduplicated[key] = action
                contrasts.append({"case_id": case["case_id"], "family_id": case["family_id"],
                                  "text": case["text"], "norm_id": norm, "label": action,
                                  "label_provenance": LABEL_PROVENANCE,
                                  "vector": _vector(case["text"], idf)})
    _require(bool(contrasts), "No explicit development contrast labels")
    result: dict[str, Any] = {
        "format": FORMAT, "label_provenance": LABEL_PROVENANCE,
        "development_families": sorted(development), "evaluation_families": sorted(evaluation),
        "settings": {"features": "unicode_word_trigrams", "weighting": "log_tf_corpus_idf_l2",
                     "adaptation_weight": ADAPTATION_WEIGHT, "threshold": None,
                     "selection": "fixed_before_evaluation_no_search"},
        "corpus_sha256": _sha(documents), "norm_documents": documents, "idf": idf,
        "norm_vectors": {norm: _vector(text, idf) for norm, text in documents.items()},
        "contrasts": contrasts,
        "notes": ["Synthetic diagnostic supervision; no expert gold or blind evaluation.",
                  "No evaluation labels were read or used for fitting.",
                  "Unlisted positive-case norms are unlabeled, never inferred negatives.",
                  "Few-shot examples require a separately declared runtime arm."]}
    result["fingerprint"] = _sha(result)
    return result


def _validate_prototype(prototype: Mapping[str, Any]) -> None:
    _require(prototype.get("format") == FORMAT, "Unknown prototype format")
    payload = {k: v for k, v in prototype.items() if k != "fingerprint"}
    _require(prototype.get("fingerprint") == _sha(payload), "Prototype fingerprint changed")
    _require(prototype["settings"]["adaptation_weight"] == ADAPTATION_WEIGHT,
             "Adaptation weight differs from fixed protocol")


def retrieve(query: str, prototype: Mapping[str, Any], *, k: int,
             adapted: bool = False) -> list[dict[str, Any]]:
    """Rank the same complete norm universe with or without train contrasts."""
    _validate_prototype(prototype)
    _require(isinstance(query, str) and bool(query.strip()), "Invalid retrieval query")
    _require(type(k) is int and k > 0 and type(adapted) is bool, "Invalid retrieval settings")
    vector = _vector(query, prototype["idf"])
    result = []
    for norm, source_vector in prototype["norm_vectors"].items():
        base = _dot(vector, source_vector)
        positive = negative = 0.0
        if adapted:
            for contrast in prototype["contrasts"]:
                if contrast["norm_id"] != norm:
                    continue
                similarity = _dot(vector, contrast["vector"])
                if contrast["label"] == "hint":
                    positive = max(positive, similarity)
                else:
                    negative = max(negative, similarity)
        result.append({"norm_id": norm, "score": base + ADAPTATION_WEIGHT * (positive - negative),
                       "source_similarity": base, "positive_example_similarity": positive,
                       "negative_example_similarity": negative})
    return sorted(result, key=lambda row: (-row["score"], row["norm_id"]))[:k]


def select_examples(query: str, prototype: Mapping[str, Any], *, k: int = 2) -> dict[str, Any]:
    """Return train-only contrasts as an explicitly optional private artifact."""
    _validate_prototype(prototype)
    _require(isinstance(query, str) and bool(query.strip()) and type(k) is int and k > 0,
             "Invalid few-shot request")
    vector = _vector(query, prototype["idf"])
    ordered = sorted(prototype["contrasts"], key=lambda row:
                     (-_dot(vector, row["vector"]), row["case_id"], row["norm_id"], row["label"]))
    return {"format": "grounded_signal_examples/1", "prototype_fingerprint": prototype["fingerprint"],
            "label_provenance": LABEL_PROVENANCE, "requires_separate_runtime_arm": True,
            "examples": [{k: v for k, v in row.items() if k != "vector"} for row in ordered[:k]]}


def compare_retrieval(cases: Sequence[Mapping[str, Any]], prototype: Mapping[str, Any], *, k: int = 3) -> dict[str, Any]:
    """Evaluate a fixed top-k on held-out families, without changing the learner.

    All offered cases must belong to the evaluation partition. The artifact is
    query-free; its identifiers/rankings are private audit data. This is a
    retrieval diagnostic, not a no-hint classifier or an end-to-end model test.
    """
    _validate_prototype(prototype)
    _require(type(k) is int and k > 0, "Invalid fixed retrieval limit")
    allowed = set(prototype["evaluation_families"])
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for case in cases:
        _require(case.get("family_id") in allowed and case.get("split") == "evaluation",
                 "Retrieval comparison only accepts held-out families")
        cid = case.get("case_id")
        _require(isinstance(cid, str) and bool(cid) and cid not in seen, "Duplicate/invalid comparison case")
        assert isinstance(cid, str)
        seen.add(cid)
        # Generate both rankings before reading diagnostic evaluation labels.
        baseline = retrieve(case["text"], prototype, k=k)
        adapted = retrieve(case["text"], prototype, k=k, adapted=True)
        labels = case.get("diagnostic_labels")
        _require(isinstance(labels, dict), "Evaluation diagnostic labels missing")
        assert isinstance(labels, dict)
        targets = _unique_strings(labels.get("target_norm_ids"), "evaluation targets")
        negatives = _unique_strings(labels.get("no_hint_norm_ids", []), "evaluation negative annotations")
        _require(not set(targets) & set(negatives), "Conflicting evaluation annotations")
        rows.append({"case_id": cid, "family_id": case["family_id"], "target_norm_ids": targets,
                     "no_hint_norm_ids": negatives, "baseline": baseline, "adapted": adapted})
    _require(bool(rows), "Empty evaluation comparison")
    metrics = {}
    for arm in ("baseline", "adapted"):
        hits = target_hits = explicit_negatives = 0
        for row in rows:
            norms = {hit["norm_id"] for hit in row[arm]}
            matching = norms & set(row["target_norm_ids"])
            hits += bool(matching)
            target_hits += len(matching)
            explicit_negatives += len(norms & set(row["no_hint_norm_ids"]))
        metrics[arm] = {"planned_cases": len(rows), "positive_cases": sum(bool(r["target_norm_ids"]) for r in rows),
                        "positive_cases_with_retrieved_target": hits,
                        "target_norms": sum(len(r["target_norm_ids"]) for r in rows), "target_norms_retrieved": target_hits,
                        "explicit_no_hint_norms_retrieved": explicit_negatives,
                        "interpretation": "Retrieval exposure only; not a model hint or a correct abstention."}
    families = {r["family_id"] for r in rows}
    return {"format": "grounded_signal_retrieval_comparison/1", "prototype_fingerprint": prototype["fingerprint"],
            "k": k, "evaluation_cases_sha256": _sha(list(cases)), "metrics": metrics, "per_case": rows,
            "evaluation_scope": "all_declared_families" if families == allowed else "diagnostic_subset",
            "label_provenance": LABEL_PROVENANCE, "threshold_search": False,
            "notes": ["Family-disjoint known synthetic diagnostics, not independent legal validation.",
                      "No labels, thresholds or weights were updated from evaluation outputs."]}
