"""Model-free contract checks for the standalone real-GPU current validator."""

from copy import deepcopy

import pytest

from tools._gpu_eval_metrics import evaluate_variant
from tools.quality_precision_live import comparison, helper, validate_reference, verify_hashes


def reference():
    report = {"format": "quality_precision_current/1", "status": "pass", "torch_imported": False,
              "mismatches": [], "hard_mark_mismatches": [], "cache_misses": [], "errors": [],
              "profiles": {"praezision": {"package": "arbitrary_final_strategy"}, "dokument": {}},
              "reference_mapping": {"praezision": "new_ranked_policy", "dokument": "new_context_policy"},
              "source_hashes": {"risk_layer/pipeline.py": "a" * 64}, "source_run": "source-run",
              "source_report_sha256": "b" * 64, "analysis_report_sha256": "c" * 64, "rows": []}
    for index in range(162):
        row = {"case_id": str(index), "dataset": "main" if index < 132 else "additional",
               "query": "Ein ausdrücklicher Sachverhalt.", "targets": ["norm:AO:1"], "negative": False}
        for profile in report["profiles"]:
            row[profile] = {"suggestions": [], "cards": [], "trigger_marks": [], "comparison": {"all_equal": True}}
        report["rows"].append(row)
    return report


def test_complete_current_schema_accepts_its_actual_profile_config_without_hardcoded_strategy():
    validate_reference(reference())


@pytest.mark.parametrize(("field", "value"), [
    ("format", "quality_production_check/1"), ("status", "mismatch"),
    ("torch_imported", True), ("torch_imported", None), ("mismatches", [{}]),
    ("hard_mark_mismatches", [{}]), ("errors", [{}]), ("cache_misses", [{}]),
    ("reference_mapping", {}), ("profiles", ["praezision", "dokument"]),
    ("source_report_sha256", "unknown"), ("source_hashes", {}),
])
def test_incompatible_or_unsuccessful_reference_is_not_relabelled_as_current(field, value):
    report = reference()
    report[field] = value
    with pytest.raises(ValueError):
        validate_reference(report)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "profile_error", "failed_comparison", "changed_dataset"])
def test_incomplete_case_matrix_or_missing_output_fails_before_gpu(damage):
    report = reference()
    if damage == "missing":
        report["rows"].pop()
    elif damage == "duplicate":
        report["rows"][1] = report["rows"][0]
    elif damage == "profile_error":
        report["rows"][0]["praezision"] = {"error": "missing cache"}
    elif damage == "failed_comparison":
        report["rows"][0]["dokument"]["comparison"]["all_equal"] = False
    else:
        report["rows"][0]["dataset"] = "additional"
    with pytest.raises(ValueError):
        validate_reference(report)


def output():
    return {"suggestions": [{"start": 0, "end": 10, "satz": "Sachverhalt", "profil": "praezision",
                             "kontext": {"start": 0, "end": 10, "text": "Sachverhalt"},
                             "normen": [{"id": "norm:AO:1", "sim": .8, "rerank": .9,
                                         "evidenz": {"fingerprint": "a", "passagen": [{"text": "Quelle"}]}},
                                        {"id": "norm:AO:2", "sim": .7, "rerank": .8,
                                         "evidenz": {"fingerprint": "b", "passagen": [{"text": "Quelle2"}]}}]}],
            "cards": [{"id": "hard", "via": "regex", "span": [0, 5]}], "trigger_marks": []}


def test_identical_output_matches_all_independent_checks():
    report = output()
    checks = comparison(report, deepcopy(report), ["norm:AO:1"], evaluate_variant)
    assert all(checks[name] for name in ("all_equal", "norm_order_equal", "scores_equal", "evidence_equal",
                                        "context_equal", "hard_marks_equal", "metrics_equal"))
    assert checks["score_display_differences"] == []


def test_pure_order_jitter_is_visible_without_claiming_changed_source():
    expected = output()
    actual = deepcopy(expected)
    actual["suggestions"][0]["normen"].reverse()
    checks = comparison(actual, expected, ["norm:AO:1"], evaluate_variant)
    assert not checks["all_equal"] and not checks["norm_order_equal"] and not checks["metrics_equal"]
    assert checks["scores_equal"] and checks["evidence_equal"] and checks["context_equal"]


def test_display_score_difference_is_not_hidden_when_case_hit_is_unchanged():
    expected = output()
    actual = deepcopy(expected)
    actual["suggestions"][0]["normen"][0]["rerank"] = .901
    checks = comparison(actual, expected, ["norm:AO:1"], evaluate_variant)
    assert not checks["all_equal"] and not checks["scores_equal"]
    assert checks["metrics_equal"] and checks["norm_order_equal"]
    assert checks["score_display_differences"][0]["norm_id"] == "norm:AO:1"


@pytest.mark.parametrize(("damage", "failed"), [
    ("evidence", "evidence_equal"), ("context", "context_equal"), ("hardmark", "hard_marks_equal"),
])
def test_same_scores_and_norms_do_not_mask_source_context_or_hardmark_drift(damage, failed):
    expected = output()
    actual = deepcopy(expected)
    if damage == "evidence":
        actual["suggestions"][0]["normen"][0]["evidenz"]["passagen"][0]["text"] = "Falsche Quelle"
    elif damage == "context":
        actual["suggestions"][0]["kontext"]["text"] = "Falscher Kontext"
    else:
        actual["cards"] = []
    checks = comparison(actual, expected, ["norm:AO:1"], evaluate_variant)
    assert not checks["all_equal"] and not checks[failed]
    assert checks["scores_equal"] and checks["norm_order_equal"] and checks["metrics_equal"]


def test_changed_norm_selection_cannot_match_even_with_same_hit_at_five():
    expected = output()
    actual = deepcopy(expected)
    actual["suggestions"][0]["normen"][1]["id"] = "norm:AO:3"
    checks = comparison(actual, expected, ["norm:AO:1"], evaluate_variant)
    assert checks["metrics_equal"]
    assert not checks["norm_order_equal"] and not checks["scores_equal"] and not checks["evidence_equal"]


def test_conflicting_reference_hashes_are_not_silently_overwritten(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("original", encoding="utf-8")
    bound = {str(source.resolve()): "a" * 64}
    with pytest.raises(ValueError, match="Conflicting"):
        verify_hashes({str(source): helper.digest(source)}, bound)


def test_referenced_input_mutation_is_detected(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("original", encoding="utf-8")
    claims = {str(source): helper.digest(source)}
    bound = {}
    verify_hashes(claims, bound)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_hashes(claims, bound)
