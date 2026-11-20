from copy import deepcopy
import json

import pytest

from experiments.grounded_signal.learning import (
    LearningError, compare_retrieval, fit_examples, retrieve, select_examples,
)


def cases():
    result = [
        {"case_id": "d1", "family_id": "dev", "split": "development",
         "text": "Ein roter Eintrag bleibt sichtbar.", "prepared_norm_ids": ["n1", "n2"],
         "diagnostic_labels": {"target_norm_ids": ["n1"], "no_hint_norm_ids": []}},
        {"case_id": "d2", "family_id": "dev", "split": "development",
         "text": "Die blaue Nummer enthält keinen Eintrag.", "prepared_norm_ids": ["n1", "n2"],
         "diagnostic_labels": {"target_norm_ids": ["n2"], "no_hint_norm_ids": ["n1"]}},
        {"case_id": "e1", "family_id": "eval", "split": "evaluation",
         "text": "Der andere rote Eintrag ist lesbar.", "prepared_norm_ids": ["n1", "n2"],
         "diagnostic_labels": {"target_norm_ids": ["n1"], "no_hint_norm_ids": []}},
    ]
    for case in result:
        case["diagnostic_labels"]["annotation_status"] = "assistant_reviewed_synthetic"
    return result


def fit(rows=None):
    return fit_examples(cases() if rows is None else rows, development_families=["dev"],
                        evaluation_families=["eval"], norm_documents={
                            "n1": "Der rote Eintrag ist eine sichtbare Angabe.",
                            "n2": "Die blaue Nummer bezeichnet den Vorgang."})


def test_train_only_reproducible_artifact_and_no_inferred_negative():
    a, b = fit(), fit()
    assert a == b
    assert {r["case_id"] for r in a["contrasts"]} == {"d1", "d2"}
    assert len(a["contrasts"]) == 3
    assert not any(r["case_id"] == "d1" and r["norm_id"] == "n2" for r in a["contrasts"])
    assert '"e1"' not in json.dumps(a)
    assert a["settings"]["threshold"] is None


def test_evaluation_labels_are_never_read_or_copied():
    rows = cases()
    rows[-1]["diagnostic_labels"] = object()
    assert fit(rows) == fit()


@pytest.mark.parametrize("change", [
    lambda r: r[-1].update(family_id="dev"),
    lambda r: r[-1].update(split="development"),
    lambda r: r[-1].update(family_id="undeclared"),
    lambda r: r[-1].update(case_id="d1"),
    lambda r: r[-1].update(text="  EIN\u00a0ROTER EINTRAG BLEIBT SICHTBAR. "),
])
def test_split_identity_and_normalized_text_leak_guards(change):
    rows = cases()
    change(rows)
    with pytest.raises(LearningError):
        fit(rows)


def test_overlapping_declared_families_rejected():
    with pytest.raises(LearningError, match="disjoint"):
        fit_examples(cases(), development_families=["dev"], evaluation_families=["dev"],
                     norm_documents={"n1": "Text"})


@pytest.mark.parametrize("labels", [
    {"target_norm_ids": ["n1"], "no_hint_norm_ids": ["n1"]},
    {"target_norm_ids": ["elsewhere"], "no_hint_norm_ids": []},
    {"target_norm_ids": ["n1", "n1"], "no_hint_norm_ids": []},
])
def test_bad_explicit_contrasts_rejected(labels):
    rows = cases()
    rows[0]["diagnostic_labels"] = {**labels, "annotation_status": "assistant_reviewed_synthetic"}
    with pytest.raises(LearningError):
        fit(rows)


def test_train_duplicate_dedup_and_conflict():
    rows = cases()
    extra = deepcopy(rows[0])
    extra["case_id"] = "d3"
    rows.append(extra)
    assert len(fit(rows)["contrasts"]) == 3
    extra["diagnostic_labels"] = {"target_norm_ids": [], "no_hint_norm_ids": ["n1"],
                                  "annotation_status": "assistant_reviewed_synthetic"}
    with pytest.raises(LearningError, match="conflicting"):
        fit(rows)


def test_training_requires_explicit_annotation_provenance():
    rows = cases()
    rows[0]["diagnostic_labels"].pop("annotation_status")
    with pytest.raises(LearningError, match="provenance"):
        fit(rows)


def test_adaptation_changes_similarity_without_changing_candidate_universe():
    p = fit()
    baseline = retrieve(cases()[0]["text"], p, k=10)
    adapted = retrieve(cases()[0]["text"], p, k=10, adapted=True)
    b, a = ({r["norm_id"]: r for r in rows} for rows in (baseline, adapted))
    assert set(b) == set(a) == {"n1", "n2"}
    assert a["n1"]["score"] > b["n1"]["score"]
    assert a["n1"]["source_similarity"] == b["n1"]["score"]
    negative = retrieve(cases()[1]["text"], p, k=10, adapted=True)
    n1 = next(r for r in negative if r["norm_id"] == "n1")
    assert n1["negative_example_similarity"] > n1["positive_example_similarity"]
    assert n1["score"] < n1["source_similarity"]


def test_examples_are_explicit_optional_train_artifacts():
    p = fit()
    result = select_examples("Roter Eintrag", p, k=2)
    assert result["requires_separate_runtime_arm"] is True
    assert len(result["examples"]) == 2
    assert {r["family_id"] for r in result["examples"]} == {"dev"}
    assert "not_legal_gold" in result["label_provenance"]
    assert all("vector" not in r for r in result["examples"])


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_bad_output_limits(k):
    with pytest.raises(LearningError):
        retrieve("Eintrag", fit(), k=k)


def test_artifact_tampering_is_detected():
    p = fit()
    p["contrasts"][0]["label"] = "no_hint"
    with pytest.raises(LearningError, match="fingerprint"):
        retrieve("Eintrag", p, k=1)


def test_fixed_retrieval_comparison_uses_all_supplied_evaluation_cases():
    prototype = fit()
    before = deepcopy(prototype)
    comparison = compare_retrieval(cases()[2:], prototype, k=1)
    assert prototype == before
    assert comparison["metrics"]["baseline"]["planned_cases"] == 1
    assert comparison["metrics"]["adapted"]["target_norms"] == 1
    assert comparison["threshold_search"] is False
    assert cases()[2]["text"] not in json.dumps(comparison, ensure_ascii=False)
    with pytest.raises(LearningError, match="held-out"):
        compare_retrieval(cases(), prototype, k=1)


def test_missing_retrieval_target_stays_in_denominator():
    evaluation = cases()[2:]
    evaluation[0]["diagnostic_labels"]["target_norm_ids"] = ["unavailable"]
    result = compare_retrieval(evaluation, fit(), k=2)
    for arm in ("baseline", "adapted"):
        assert result["metrics"][arm]["target_norms"] == 1
        assert result["metrics"][arm]["target_norms_retrieved"] == 0
