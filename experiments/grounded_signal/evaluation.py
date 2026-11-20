"""Denominator-preserving evaluation of synthetic grounded-signal diagnostics.

Mechanical attribution, diagnostic target matching, and semantic correctness
are distinct. This module never treats a model paraphrase or relation as gold.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


KINDS = {"actor", "role", "event", "negation", "time", "context"}
POLARITIES = {"affirmed", "denied", "uncertain"}
RELATIONS = {"definition", "duty", "procedure", "exception", "scope", "none"}
STATUSES = {"hint", "no_hint", "insufficient_context"}


class EvaluationError(ValueError):
    """A planned denominator or frozen identity cannot be established."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) and bool(x) for x in value)


def _ids(value: Any) -> bool:
    return _strings(value) and len(value) == len(set(value))


def validate_answer(answer: Any, *, text: str, candidate_norm_ids: Sequence[str],
                    source_units: Mapping[str, Mapping[str, Any]],
                    available_unit_ids: Sequence[str]) -> dict[str, Any]:
    """Validate attribution only, returning errors rather than a false abstention.

    A valid statement is merely a model paraphrase with a real original span.
    It is not thereby entailed by that span. The same limitation applies to
    supported-fact links, evidence selections, relations, and open conditions.
    """
    errors: list[str] = []
    valid_facts: set[str] = set()
    counts: Counter[str] = Counter()
    available = set(available_unit_ids)
    if not isinstance(answer, dict) or set(answer) != {"facts", "hints", "unresolved"}:
        return {"valid": False, "errors": ["answer_schema"], "counts": {}}
    if not isinstance(answer["facts"], list) or not isinstance(answer["hints"], list):
        return {"valid": False, "errors": ["answer_lists"], "counts": {}}
    if (not _strings(answer["unresolved"]) or len(answer["unresolved"]) > 4 or
            any(not item.strip() or len(item) > 150 for item in answer["unresolved"])):
        errors.append("unresolved_schema")
    if len(answer["facts"]) > 8:
        errors.append("fact_count")
    seen_facts: set[str] = set()
    for fact in answer["facts"]:
        counts["facts_submitted"] += 1
        if not isinstance(fact, dict) or set(fact) != {
            "id", "start", "end", "text", "kind", "polarity", "statement"
        }:
            errors.append("fact_schema")
            continue
        fid = fact["id"]
        if not isinstance(fid, str) or fid not in {f"f{i}" for i in range(1, 9)} or fid in seen_facts:
            errors.append("fact_id")
            continue
        seen_facts.add(fid)
        start, end = fact["start"], fact["end"]
        valid_span = (type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                      and isinstance(fact["text"], str) and fact["text"] == text[start:end])
        if not valid_span:
            errors.append("fact_original_span")
        else:
            counts["facts_with_exact_original_span"] += 1
        if (not isinstance(fact["kind"], str) or fact["kind"] not in KINDS
                or not isinstance(fact["polarity"], str) or fact["polarity"] not in POLARITIES
                or not isinstance(fact["statement"], str) or not fact["statement"].strip()
                or len(fact["statement"]) > 180):
            errors.append("fact_interpretation_schema")
        elif valid_span:
            valid_facts.add(fid)
            counts["fact_paraphrases_semantically_unchecked"] += 1
    submitted_norms: list[str] = []
    for hint in answer["hints"]:
        counts["norm_decisions_submitted"] += 1
        if not isinstance(hint, dict) or set(hint) != {
            "norm_id", "status", "evidence_unit_ids", "supported_fact_ids", "open_conditions", "relation"
        }:
            errors.append("hint_schema")
            continue
        norm = hint["norm_id"]
        if not isinstance(norm, str):
            errors.append("norm_id")
            continue
        submitted_norms.append(norm)
        status, relation = hint["status"], hint["relation"]
        if not isinstance(status, str) or status not in STATUSES:
            errors.append("hint_status")
        if not isinstance(relation, str) or relation not in RELATIONS:
            errors.append("hint_relation")
        elif (status == "hint" and relation == "none") or (status != "hint" and relation != "none"):
            errors.append("relation_status")
        evidence, facts = hint["evidence_unit_ids"], hint["supported_fact_ids"]
        if (not _ids(evidence) or len(evidence) > 6 or not _ids(facts) or len(facts) > 8
                or not _strings(hint["open_conditions"]) or len(hint["open_conditions"]) > 4
                or any(not item.strip() or len(item) > 160 for item in hint["open_conditions"])):
            errors.append("hint_bindings_schema")
            continue
        if status == "hint" and (not evidence or not facts):
            errors.append("hint_missing_binding")
        for uid in evidence:
            unit = source_units.get(uid)
            if uid not in available or unit is None:
                errors.append("evidence_unavailable")
            elif unit.get("norm_id") != norm:
                errors.append("evidence_wrong_norm")
            else:
                counts["same_norm_evidence_links"] += 1
        for fid in facts:
            if fid not in valid_facts:
                errors.append("supported_fact_unavailable")
            else:
                counts["exact_span_fact_links"] += 1
        counts["open_conditions_semantically_unchecked"] += len(hint["open_conditions"])
        if status == "hint":
            counts["hint_relations_semantically_unchecked"] += 1
    if len(submitted_norms) != len(set(submitted_norms)):
        errors.append("duplicate_norm_decision")
    if set(submitted_norms) != set(candidate_norm_ids):
        errors.append("candidate_coverage")
    return {"valid": not errors, "errors": errors, "counts": dict(counts),
            "semantic_truth_verified": False}


def _validate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    family_split: dict[str, str] = {}
    for case in cases:
        cid, family, split = case.get("case_id"), case.get("family_id"), case.get("split")
        _require(isinstance(cid, str) and bool(cid) and cid not in result, "Invalid/duplicate planned case")
        _require(isinstance(family, str) and bool(family), "Invalid family")
        _require(split in ("development", "evaluation"), "Unknown family split")
        assert isinstance(cid, str) and isinstance(family, str) and isinstance(split, str)
        _require(family_split.setdefault(family, split) == split, "Family crosses train/evaluation split")
        _require(isinstance(case.get("text"), str) and bool(case["text"].strip()), "Empty case text")
        prepared = case.get("prepared_norm_ids")
        _require(_ids(prepared) and bool(prepared), "Prepared candidate IDs must be unique and nonempty")
        assert isinstance(prepared, list)
        labels = case.get("diagnostic_labels")
        _require(isinstance(labels, dict) and labels.get("annotation_status") == "assistant_reviewed_synthetic",
                 "Diagnostic annotation provenance missing")
        assert isinstance(labels, dict)
        _require(type(labels.get("hint_expected")) is bool, "Diagnostic hint expectation missing")
        sets = []
        for key in ("target_norm_ids", "allowed_hint_norm_ids", "no_hint_norm_ids", "insufficient_context_norm_ids"):
            value = labels.get(key, []) if key == "insufficient_context_norm_ids" else labels.get(key)
            _require(_ids(value), "Invalid diagnostic norm IDs: " + key)
            assert isinstance(value, list)
            sets.append(set(value))
        _require(not any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)), "Conflicting norm annotations")
        _require(set.union(*sets) <= set(prepared), "Diagnostic norms outside prepared source plan")
        _require(labels["hint_expected"] == bool(sets[0]), "Positive diagnostic cases need explicit targets")
        result[cid] = case
    _require(bool(result), "No planned cases")
    return result


def _package(package_id: str, packages: Mapping[str, Mapping[str, Any]],
             source_units: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    from experiments.grounded_signal.sources import package_fingerprint

    _require(package_id in packages, "Unbound source package")
    package = packages[package_id]
    _require(package.get("package_id") == package_id and
             package.get("fingerprint") == package_fingerprint(dict(package)), "Source package fingerprint differs")
    _require(package_id == "pkg:" + package["fingerprint"], "Source package ID differs")
    _require(package.get("status") in ("ready", "missing", "overflow"), "Unknown source package status")
    _require(type(package.get("complete_local_norms")) is bool and
             package["complete_local_norms"] == (package["status"] == "ready"), "Package completeness status differs")
    for key in ("candidate_norm_ids", "norm_ids", "unit_ids"):
        _require(_ids(package.get(key)), "Invalid package IDs: " + key)
    units = package.get("units")
    _require(isinstance(units, list), "Package units missing")
    assert isinstance(units, list)
    _require(all(isinstance(unit, dict) for unit in units), "Invalid package unit")
    _require([unit.get("unit_id") for unit in units] == package["unit_ids"], "Package unit inventory differs")
    _require(set(package["candidate_norm_ids"]) <= set(package["norm_ids"]) or package["status"] == "missing",
             "Candidate source missing without missing status")
    for unit in units:
        uid = unit["unit_id"]
        _require(uid in source_units and _sha(unit) == _sha(source_units[uid]), "Unit differs from frozen table")
        start, end, text = unit.get("start"), unit.get("end"), unit.get("text")
        _require(type(start) is int and type(end) is int and 0 <= start < end and isinstance(text, str)
                 and end - start == len(text) and unit.get("norm_id") in package["norm_ids"], "Invalid source unit span")
    if package["status"] == "ready":
        groups = package.get("required_groups")
        _require(isinstance(groups, list) and all(isinstance(g, dict) for g in groups), "Missing whole-norm groups")
        assert isinstance(groups, list)
        _require([g.get("norm_id") for g in groups] == package["norm_ids"], "Whole-norm group inventory differs")
        for group in groups:
            parent_units = [unit for unit in units if unit["norm_id"] == group["norm_id"]]
            _require(group.get("policy") == "all_stored_parent_norm_text" and bool(parent_units) and
                     group.get("unit_ids") == [u["unit_id"] for u in parent_units], "Incomplete whole-norm group")
            _require(parent_units[0]["start"] == 0 and all(a["end"] == b["start"] for a, b in
                                                          zip(parent_units, parent_units[1:])), "Whole-norm span gap or overlap")
            original_hash = hashlib.sha256("".join(u["text"] for u in parent_units).encode()).hexdigest()
            _require(all(u.get("norm_text_sha256") == original_hash for u in parent_units),
                     "Whole-norm original text fingerprint differs")
    else:
        _require(not units, "Incomplete source package contains partial payload")
    return package


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [r for r in rows if r["hint_expected"]]
    negatives = [r for r in rows if not r["hint_expected"] and not r["expected_context_norm_ids"]]
    targets = sum(len(r["targets"]) for r in positives)
    matched = sum(len(set(r["hint_norm_ids"]) & set(r["targets"])) for r in positives)
    unknown_targets = sum(len(set(r["unknown_norm_ids"]) & set(r["targets"])) for r in positives)
    def hit(row: Mapping[str, Any]) -> bool:
        return bool(set(row["hint_norm_ids"]) & set(row["targets"]))
    positive_hit = sum(hit(r) for r in positives)
    negative_output = sum(bool(r["hint_norm_ids"]) for r in negatives)
    unavailable_negatives = sum(r["execution"] not in ("complete",) for r in negatives)
    return {
        "planned_cases": len(rows), "positive_cases": len(positives), "negative_cases": len(negatives),
        "execution": dict(Counter(r["execution"] for r in rows)),
        "source_package_status": dict(Counter(r["source_package_status"] for r in rows)),
        "positive_cases_with_any_hint": sum(bool(r["hint_norm_ids"]) for r in positives),
        "positive_cases_with_known_target_hint": positive_hit,
        "known_target_case_recall": positive_hit / len(positives) if positives else None,
        "known_target_case_hit_upper_bound": positive_hit + sum(
            not hit(r) and bool(set(r["unknown_norm_ids"]) & set(r["targets"])) for r in positives),
        "known_target_norms": targets, "known_target_norms_emitted": matched,
        "known_target_norms_unknown": unknown_targets,
        "known_target_norm_recall": matched / targets if targets else None,
        "known_target_norm_hit_upper_bound": matched + unknown_targets,
        "negative_cases_with_hint": negative_output,
        "negative_hint_norm_pairs": sum(len(r["hint_norm_ids"]) for r in negatives),
        "negative_cases_technically_unavailable": unavailable_negatives,
        "negative_cases_with_uncertain_norms": sum(bool(r["unknown_norm_ids"]) for r in negatives),
        "negative_case_output_upper_bound": negative_output + sum(
            not r["hint_norm_ids"] and bool(r["unknown_norm_ids"]) for r in negatives),
        "negative_norm_output_upper_bound": sum(len(r["hint_norm_ids"]) + len(r["unknown_norm_ids"]) for r in negatives),
        "confirmed_no_hint_negative_cases": sum(r["execution"] == "complete" and
                                                not r["hint_norm_ids"] and not r["unknown_norm_ids"] for r in negatives),
        "context_annotated_cases": sum(bool(r["expected_context_norm_ids"]) for r in rows),
        "context_annotated_norms": sum(len(r["expected_context_norm_ids"]) for r in rows),
        "context_annotated_norms_answered_insufficient": sum(
            len(set(r["expected_context_norm_ids"]) & set(r["unknown_norm_ids"]))
            for r in rows if r["execution"] == "complete"),
        "diagnostic_negative_norms_emitted": sum(len(set(r["hint_norm_ids"]) & set(r["no_hint_norm_ids"])) for r in rows),
        "unadjudicated_hint_norms": sum(len(set(r["hint_norm_ids"]) - set(r["targets"]) -
                                              set(r["no_hint_norm_ids"]) - set(r["allowed_hint_norm_ids"]) -
                                              set(r["expected_context_norm_ids"])) for r in rows),
        "output_precision": None, "semantic_truth_verified": False,
        "retrieval_missing_known_targets": sum(len(r["retrieval_missing_targets"]) for r in positives),
    }


def evaluate(planned_cases: Sequence[Mapping[str, Any]], arm_results: Mapping[str, Sequence[Mapping[str, Any]]],
             *, protocol: Mapping[str, Any], source_units: Mapping[str, Mapping[str, Any]],
             source_packages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Audit every planned case×arm; emit a query-free aggregate plus private rows.

    Required protocol: ``arms``, ``arm_policies[arm].source_mode`` and
    ``case_packages[arm][case_id]`` freeze every offered package before inference.
    ``family_prepared`` sources must be identical across a family and models.
    ``automatic`` arms may retrieve different norms; targets stay in denominators.
    """
    cases = _validate_cases(planned_cases)
    arms, policies = protocol.get("arms"), protocol.get("arm_policies")
    _require(_ids(arms) and bool(arms), "Declare every planned arm before evaluation")
    assert isinstance(arms, list)
    _require(isinstance(policies, dict) and set(policies) == set(arms), "Declare every arm source policy")
    assert isinstance(policies, dict)
    for policy in policies.values():
        _require(isinstance(policy, dict) and policy.get("source_mode") in ("family_prepared", "automatic"),
                 "Unknown arm source policy")
    _require(set(arm_results) <= set(arms), "Unplanned result arm")
    case_packages = protocol.get("case_packages")
    _require(isinstance(case_packages, dict) and set(case_packages) == set(arms), "Invalid frozen package plan")
    assert isinstance(case_packages, dict)
    for mapping in case_packages.values():
        _require(isinstance(mapping, dict) and set(mapping) == set(cases) and
                 all(isinstance(pid, str) for pid in mapping.values()), "Source plan is incomplete")
    checked_packages = {pid: _package(pid, source_packages, source_units)
                        for pid in {pid for mapping in case_packages.values() for pid in mapping.values()}}
    family_packages: dict[str, str] = {}
    for arm in arms:
        if policies[arm]["source_mode"] != "family_prepared":
            continue
        for case in cases.values():
            pid = case_packages[arm][case["case_id"]]
            _require(family_packages.setdefault(case["family_id"], pid) == pid,
                     "Prepared sources differ across family contrasts or model arms")
            _require(set(checked_packages[pid]["candidate_norm_ids"]) == set(case["prepared_norm_ids"]),
                     "Family contrast has different prepared norm candidates")
    per_case: list[dict[str, Any]] = []
    by_arm: dict[str, Any] = {}
    for arm in arms:
        observed: dict[str, Mapping[str, Any]] = {}
        for result in arm_results.get(arm, []):
            cid = result.get("case_id")
            _require(isinstance(cid, str) and cid in cases and cid not in observed, "Unknown/duplicate result case")
            assert isinstance(cid, str)
            observed[cid] = result
        rows: list[dict[str, Any]] = []
        mechanical: Counter[str] = Counter()
        schema_errors: Counter[str] = Counter()
        elapsed: list[float] = []
        for cid, case in cases.items():
            labels = case["diagnostic_labels"]
            pid = case_packages[arm][cid]
            package = checked_packages[pid]
            candidates = package["candidate_norm_ids"]
            observed_result = observed.get(cid)
            execution, raw_status, audit = "not_run", "not_run", None
            hint_norms: list[str] = []
            # The frozen candidate pool limits even a failed/unrun arm. Missing
            # retrieval targets stay misses; transport cannot make them available.
            unknown = set(candidates)
            if observed_result is not None:
                result = observed_result
                _require(result.get("source_package_id") == pid, "Result changed its planned package")
                _require(_ids(result.get("candidate_norm_ids")) and set(result["candidate_norm_ids"]) == set(candidates),
                         "Result changed its offered candidates")
                status_value = result.get("status")
                _require(isinstance(status_value, str) and bool(status_value), "Missing execution status")
                assert isinstance(status_value, str)
                raw_status = status_value
                execution = "not_run" if raw_status == "not_run" else "technical_error"
                timing = result.get("timing")
                value = timing.get("elapsed_seconds") if isinstance(timing, dict) else None
                if value is not None:
                    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                             "Invalid recorded timing")
                    elapsed.append(float(value))
                if raw_status == "complete":
                    _require(package["status"] == "ready", "Model completed with incomplete/overflow source package")
                    audit = validate_answer(result.get("answer"), text=case["text"], candidate_norm_ids=candidates,
                                            source_units=source_units, available_unit_ids=package["unit_ids"])
                    mechanical.update(audit["counts"])
                    schema_errors.update(audit["errors"])
                    execution = "complete" if audit["valid"] else "invalid_answer"
                    if audit["valid"]:
                        hint_norms = [h["norm_id"] for h in result["answer"]["hints"] if h["status"] == "hint"]
                        unknown = {h["norm_id"] for h in result["answer"]["hints"] if h["status"] == "insufficient_context"}
                else:
                    _require(result.get("answer") is None, "Technical failure contains an unaccepted answer")
            row = {"case_id": cid, "family_id": case["family_id"], "split": case["split"],
                   "execution": execution, "raw_status": raw_status, "source_package_id": pid,
                   "source_package_status": package["status"],
                   "hint_expected": labels["hint_expected"], "targets": labels["target_norm_ids"],
                   "allowed_hint_norm_ids": labels["allowed_hint_norm_ids"],
                   "no_hint_norm_ids": labels["no_hint_norm_ids"], "hint_norm_ids": hint_norms,
                   "expected_context_norm_ids": labels.get("insufficient_context_norm_ids", []),
                   "unknown_norm_ids": sorted(unknown), "mechanical_audit": audit,
                   "retrieval_missing_targets": sorted(set(labels["target_norm_ids"]) - set(candidates)),
                   "qualitative_review_required": bool(execution == "complete")}
            rows.append(row)
        by_arm[arm] = {"all": _metrics(rows), **{split: _metrics([r for r in rows if r["split"] == split])
                       for split in ("development", "evaluation")},
                       "mechanical_counts": dict(mechanical), "mechanical_error_counts": dict(schema_errors),
                       "timing": {"recorded_attempts": len(elapsed), "total_seconds": sum(elapsed),
                                  "scope": "Recorded attempts including failures; missing rows receive no invented duration."}}
        per_case.extend({"arm": arm, **row} for row in rows)
    aggregate = {"format": "grounded_signal_evaluation/2", "annotation_status": "assistant_reviewed_synthetic",
                 "protocol_sha256": _sha(protocol), "planned_cases_sha256": _sha(list(planned_cases)),
                 "results_sha256": _sha(arm_results), "source_packages_sha256": _sha(source_packages),
                 "source_units_sha256": _sha(source_units), "planned_cases": len(cases),
                 "planned_case_arm_rows": len(cases) * len(arms), "arms": by_arm,
                 "notes": ["Known synthetic diagnostics, assistant-reviewed, not legal gold or a blind test.",
                           "Every planned row remains in the denominator; unknown and technical failure are not correct negatives.",
                           "Source/fact-span identity does not validate paraphrases, relevance, conditions, or legal applicability.",
                           "Diagnostic target lists and allowed extras do not establish proposal precision.",
                           "Automatic source arms keep target denominators and report retrieval misses separately."]}
    return {"aggregate": aggregate, "per_case": deepcopy(per_case)}
