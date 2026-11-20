from copy import deepcopy
import hashlib
import json

import pytest

from experiments.grounded_signal.evaluation import EvaluationError, evaluate, validate_answer
from experiments.grounded_signal.sources import package_fingerprint


def package(norms=("n1", "n2"), suffix=""):
    units = [{"unit_id": norm + suffix, "norm_id": norm, "start": 0, "end": 12,
              "text": "Ein Regeltext", "title": "Text"} for norm in norms]
    for unit in units:
        unit["end"] = len(unit["text"])
        unit["norm_text_sha256"] = hashlib.sha256(unit["text"].encode()).hexdigest()
    p = {"status": "ready", "complete_local_norms": True, "candidate_norm_ids": list(norms),
         "norm_ids": list(norms), "unit_ids": [u["unit_id"] for u in units], "units": units,
         "required_groups": [{"norm_id": norm, "unit_ids": [norm + suffix],
                              "policy": "all_stored_parent_norm_text"} for norm in norms]}
    p["fingerprint"] = package_fingerprint(p)
    p["package_id"] = "pkg:" + p["fingerprint"]
    return p


def data():
    p = package()
    cases = [{"case_id": cid, "family_id": "f", "split": "evaluation", "text": text,
              "prepared_norm_ids": ["n1", "n2"], "diagnostic_labels": {
                  "annotation_status": "assistant_reviewed_synthetic", "hint_expected": positive,
                  "target_norm_ids": ["n1", "n2"] if positive else [], "allowed_hint_norm_ids": [],
                  "no_hint_norm_ids": [] if positive else ["n1", "n2"]}}
             for cid, text, positive in [("p", "Ein Eintrag ist vorhanden.", True),
                                         ("n", "Keine Angabe ist vorhanden.", False)]]
    protocol = {"arms": ["a"], "arm_policies": {"a": {"source_mode": "family_prepared"}},
                "case_packages": {"a": {"p": p["package_id"], "n": p["package_id"]}}}
    return cases, protocol, {p["package_id"]: p}, {u["unit_id"]: u for u in p["units"]}


def answer(text, n1="hint", n2="no_hint"):
    return {"facts": [{"id": "f1", "start": 0, "end": len(text), "text": text,
                       "kind": "event", "polarity": "affirmed", "statement": "Ungeprüfte Paraphrase"}],
            "hints": [{"norm_id": norm, "status": status,
                       "evidence_unit_ids": [norm] if status == "hint" else [],
                       "supported_fact_ids": ["f1"] if status == "hint" else [],
                       "open_conditions": [], "relation": "duty" if status == "hint" else "none"}
                      for norm, status in [("n1", n1), ("n2", n2)]], "unresolved": []}


def row(case, p, status="complete", **kwargs):
    return {"case_id": case["case_id"], "status": status,
            "answer": answer(case["text"], **kwargs) if status == "complete" else None,
            "error": None if status == "complete" else "transport failure",
            "timing": {"elapsed_seconds": 2.0}, "source_package_id": p["package_id"],
            "candidate_norm_ids": p["candidate_norm_ids"]}


def run(rows=None, dataset=None):
    cases, protocol, packages, units = data() if dataset is None else dataset
    return evaluate(cases, {} if rows is None else {"a": rows}, protocol=protocol,
                    source_units=units, source_packages=packages)


def test_missing_rows_are_in_denominator_and_no_negative_success():
    r = run()["aggregate"]["arms"]["a"]["all"]
    assert r["planned_cases"] == 2 and r["execution"] == {"not_run": 2}
    assert r["known_target_norms"] == 2 and r["known_target_norm_hit_upper_bound"] == 2
    assert r["confirmed_no_hint_negative_cases"] == 0
    assert r["negative_case_output_upper_bound"] == 1
    assert r["negative_norm_output_upper_bound"] == 2


def test_target_cases_and_individual_norms_are_distinct():
    cases, _, packages, _ = data()
    p = next(iter(packages.values()))
    result = run([row(cases[0], p), row(cases[1], p, n1="no_hint")])
    r = result["aggregate"]["arms"]["a"]["all"]
    assert r["positive_cases_with_known_target_hint"] == 1
    assert r["known_target_norms_emitted"] == 1 and r["known_target_norm_recall"] == .5
    assert r["confirmed_no_hint_negative_cases"] == 1
    assert r["output_precision"] is None
    assert "Ungeprüfte Paraphrase" not in json.dumps(result["aggregate"], ensure_ascii=False)
    assert "Ein Eintrag" not in json.dumps(result["aggregate"])


def test_technical_error_and_insufficient_context_are_not_no_hint():
    cases, _, packages, _ = data()
    p = next(iter(packages.values()))
    r = run([row(cases[0], p, status="timeout"),
             row(cases[1], p, n1="insufficient_context")])["aggregate"]["arms"]["a"]["all"]
    assert r["known_target_norms_unknown"] == 2
    assert r["negative_cases_technically_unavailable"] == 0
    assert r["negative_cases_with_uncertain_norms"] == 1
    assert r["confirmed_no_hint_negative_cases"] == 0


def test_context_annotation_stays_separate_from_negatives_and_unjudged_hints():
    dataset = data()
    labels = dataset[0][1]["diagnostic_labels"]
    labels["no_hint_norm_ids"] = ["n2"]
    labels["insufficient_context_norm_ids"] = ["n1"]
    p = next(iter(dataset[2].values()))
    result = run([row(dataset[0][1], p, n1="hint")], dataset)["aggregate"]
    assert result["format"] == "grounded_signal_evaluation/2"
    r = result["arms"]["a"]["all"]
    assert r["context_annotated_norms"] == 1
    assert r["negative_cases"] == r["negative_cases_with_hint"] == r["negative_hint_norm_pairs"] == 0
    assert r["unadjudicated_hint_norms"] == 0
    assert r["confirmed_no_hint_negative_cases"] == 0


@pytest.mark.parametrize("mutation,error", [
    (lambda a: a["facts"][0].update(start=True), "fact_original_span"),
    (lambda a: a["facts"][0].update(text="erfunden"), "fact_original_span"),
    (lambda a: a["facts"][0].update(statement="x" * 181), "fact_interpretation_schema"),
    (lambda a: a["hints"][0].update(supported_fact_ids=["missing"]), "supported_fact_unavailable"),
    (lambda a: a["hints"][0].update(evidence_unit_ids=["n2"]), "evidence_wrong_norm"),
    (lambda a: a["hints"][0].update(evidence_unit_ids=["other"]), "evidence_unavailable"),
    (lambda a: a["hints"][0].update(relation="none"), "relation_status"),
    (lambda a: a["hints"][1].update(relation="scope"), "relation_status"),
    (lambda a: a["hints"].pop(), "candidate_coverage"),
    (lambda a: a["hints"].append(deepcopy(a["hints"][0])), "duplicate_norm_decision"),
    (lambda a: a["hints"][0].update(norm_id="new"), "candidate_coverage"),
    (lambda a: a.update(unresolved=["x"] * 5), "unresolved_schema"),
    (lambda a: a["hints"][0].update(open_conditions=[" "]), "hint_bindings_schema"),
])
def test_malformed_bindings_become_invalid_answers_not_success(mutation, error):
    cases, _, packages, _ = data()
    p = next(iter(packages.values()))
    r = row(cases[0], p)
    mutation(r["answer"])
    out = run([r])
    audit = out["per_case"][0]
    assert audit["execution"] == "invalid_answer"
    assert error in audit["mechanical_audit"]["errors"]
    assert out["aggregate"]["arms"]["a"]["all"]["positive_cases_with_known_target_hint"] == 0


def test_mechanical_validity_does_not_approve_semantic_paraphrase():
    cases, _, _, units = data()
    a = answer(cases[0]["text"])
    a["facts"][0]["statement"] = "Eine nicht genannte Rolle wird erfunden."
    result = validate_answer(a, text=cases[0]["text"], candidate_norm_ids=["n1", "n2"],
                             source_units=units, available_unit_ids=list(units))
    assert result["valid"] is True
    assert result["semantic_truth_verified"] is False
    assert result["counts"]["fact_paraphrases_semantically_unchecked"] == 1


def test_crlf_nbsp_and_repeated_text_use_exact_original_offsets():
    text = "Eintrag.\r\nEintrag.\u00a0"
    a = answer(text)
    a["facts"][0].update(start=10, end=18, text="Eintrag.")
    units = data()[3]
    result = validate_answer(a, text=text, candidate_norm_ids=["n1", "n2"],
                             source_units=units, available_unit_ids=list(units))
    assert result["valid"]
    a["facts"][0]["start"] = 9
    assert not validate_answer(a, text=text, candidate_norm_ids=["n1", "n2"],
                               source_units=units, available_unit_ids=list(units))["valid"]


def test_duplicate_unplanned_results_and_timing_nonfinite_rejected():
    cases, _, packages, _ = data()
    p = next(iter(packages.values()))
    r = row(cases[0], p)
    with pytest.raises(EvaluationError, match="duplicate"):
        run([r, r])
    r["case_id"] = "other"
    with pytest.raises(EvaluationError, match="Unknown"):
        run([r])
    r["case_id"] = "p"
    r["timing"]["elapsed_seconds"] = float("nan")
    with pytest.raises(EvaluationError, match="timing"):
        run([r])


def test_result_cannot_choose_its_own_package_or_candidate_set():
    cases, _, packages, _ = data()
    r = row(cases[0], next(iter(packages.values())))
    r["candidate_norm_ids"] = ["n1"]
    with pytest.raises(EvaluationError, match="candidates"):
        run([r])


def test_family_prepared_packages_must_match_even_when_norm_ids_match():
    dataset = data()
    alternative = package(suffix="different")
    dataset[2][alternative["package_id"]] = alternative
    dataset[3].update({u["unit_id"]: u for u in alternative["units"]})
    dataset[1]["case_packages"]["a"]["n"] = alternative["package_id"]
    with pytest.raises(EvaluationError, match="Prepared sources differ"):
        run(dataset=dataset)


def test_automatic_retrieval_misses_do_not_remove_target_denominator():
    dataset = data()
    alternative = package(("n2",))
    dataset[2][alternative["package_id"]] = alternative
    dataset[1]["arm_policies"]["a"]["source_mode"] = "automatic"
    dataset[1]["case_packages"]["a"]["p"] = alternative["package_id"]
    r = row(dataset[0][0], alternative)
    r["answer"]["hints"] = [r["answer"]["hints"][1]]
    out = run([r], dataset)["aggregate"]["arms"]["a"]["all"]
    assert out["known_target_norms"] == 2
    assert out["retrieval_missing_known_targets"] == 1
    assert out["known_target_norms_unknown"] == 0


@pytest.mark.parametrize("status", ["timeout", "not_run"])
def test_retrieval_misses_remain_misses_when_model_did_not_complete(status):
    dataset = data()
    alternative = package(("n2",))
    dataset[2][alternative["package_id"]] = alternative
    dataset[1]["arm_policies"]["a"]["source_mode"] = "automatic"
    dataset[1]["case_packages"]["a"]["p"] = alternative["package_id"]
    rows = [] if status == "not_run" else [row(dataset[0][0], alternative, status=status)]
    out = run(rows, dataset)["aggregate"]["arms"]["a"]["all"]
    assert out["known_target_norms"] == 2
    assert out["retrieval_missing_known_targets"] == 1
    assert out["known_target_norms_unknown"] == 1
    assert out["known_target_norm_hit_upper_bound"] == 1


def test_source_snapshot_tamper_fails_closed():
    dataset = data()
    dataset[3]["n1"] = {**dataset[3]["n1"], "text": "changed text"}
    with pytest.raises(EvaluationError, match="frozen table"):
        run(dataset=dataset)


def test_all_planned_arms_count_even_if_one_is_completely_absent():
    dataset = data()
    dataset[1]["arms"].append("b")
    dataset[1]["arm_policies"]["b"] = {"source_mode": "family_prepared"}
    dataset[1]["case_packages"]["b"] = deepcopy(dataset[1]["case_packages"]["a"])
    result = run(dataset=dataset)["aggregate"]
    assert result["planned_case_arm_rows"] == 4
    assert result["arms"]["b"]["all"]["execution"] == {"not_run": 2}
