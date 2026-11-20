"""Model-free provenance, nested cutoff and reporting regression tests."""
from copy import deepcopy
import hashlib

import pytest

from tools._quality_recall_classifiers import (
    _bind_inputs,
    _checked_dataset_metrics,
    _datasets,
    _nested_thresholds,
    _purges,
    _read,
    _score,
)
from tools.quality_recall_report import render


@pytest.fixture
def nested():
    ids = [f"TEST:{i}" for i in range(6)]
    cases = {key: {"negative": i % 2 == 1} for i, key in enumerate(ids)}
    folds = [{"held_out_source": "inner_a", "train": [2, 3], "test": [0, 1], "train_local": [2, 3],
              "test_local": [0, 1], "anchor_train": [0], "binary_fit_valid": True},
             {"held_out_source": "inner_b", "train": [0, 1], "test": [2, 3], "train_local": [0, 1],
              "test_local": [2, 3], "anchor_train": [0], "binary_fit_valid": True}]
    protocol = {"case_order": ids, "folds": [{"held_out_source": "outer", "train": [0, 1, 2, 3],
                                            "test": [4, 5], "inner_folds": folds}]}
    fitted = {"held_out_source": "outer", "threshold": .4, "model_fits": 2,
              "fallback": None, "predictions": [.4, .1, .7, .2]}
    rows = [{} for _ in range(4)] + [{"threshold": .4, "value_need": .39, "predicted_need": False},
                                   {"threshold": .4, "value_need": .8, "predicted_need": True}]
    report = {"folds": [fitted], "rows": rows, "model_fits": {"inner": 2, "outer": 1}}
    return protocol, report, cases


def test_nested_cutoff_uses_only_inner_original_positive_values(nested):
    protocol, report, cases = nested
    assert _nested_thresholds(protocol, report, cases) == 0
    changed = deepcopy(cases)
    changed["TEST:4"]["negative"] = not changed["TEST:4"]["negative"]
    changed["TEST:5"]["negative"] = not changed["TEST:5"]["negative"]
    assert _nested_thresholds(protocol, report, changed) == 0


def test_even_more_permissive_unprescribed_threshold_is_rejected(nested):
    protocol, report, cases = nested
    report["folds"][0]["threshold"] = .39
    with pytest.raises(ValueError, match="cutoff"):
        _nested_thresholds(protocol, report, cases)


def test_catalog_anchor_cannot_be_appended_as_threshold_example(nested):
    protocol, report, cases = nested
    report["folds"][0]["predictions"].append(.2)
    with pytest.raises(ValueError, match="inner OOF"):
        _nested_thresholds(protocol, report, cases)


def test_missing_inner_case_cannot_shrink_calibration_denominator(nested):
    protocol, report, cases = nested
    protocol["folds"][0]["inner_folds"][0]["test"].pop()
    with pytest.raises(ValueError, match="denominator"):
        _nested_thresholds(protocol, report, cases)


def test_inner_fit_cannot_contain_outer_test_row(nested):
    protocol, report, cases = nested
    protocol["folds"][0]["inner_folds"][0]["train"].append(4)
    with pytest.raises(ValueError, match="leakage"):
        _nested_thresholds(protocol, report, cases)


def test_degenerate_inner_training_class_requires_zero_threshold(nested):
    protocol, report, cases = nested
    inner = protocol["folds"][0]["inner_folds"][0]
    inner.update(train=[2], train_local=[2], binary_fit_valid=False)
    fitted = report["folds"][0]
    fitted.update(threshold=0., model_fits=0, fallback="inner_training_fold_missing_class", predictions=[])
    report["model_fits"]["inner"] = 0
    for row in report["rows"][4:]:
        row.update(threshold=0., predicted_need=True)
    assert _nested_thresholds(protocol, report, cases) == 1
    fitted["threshold"] = .1
    with pytest.raises(ValueError, match="fallback"):
        _nested_thresholds(protocol, report, cases)


def test_exact_frozen_metric_source_survives_later_tool_edits(tmp_path):
    path = tmp_path / "current.py"
    source = b"original metric source"
    path.write_bytes(b"new tool implementation")
    (tmp_path / "metrics.py").write_bytes(source)
    protocol = {"input_hashes": {str(path): hashlib.sha256(source).hexdigest()}}
    assert _bind_inputs(protocol, tmp_path)["current_tools_replaced_by_exact_frozen_source"] == 1
    (tmp_path / "metrics.py").write_bytes(b"also changed")
    with pytest.raises(ValueError, match="without exact source snapshot"):
        _bind_inputs(protocol, tmp_path)


def test_declared_near_original_and_anchor_edges_are_enforced():
    ids = [f"TEST:{i}" for i in range(6)]
    protocol = {"case_order": ids, "folds": [{"train": [1, 2, 3], "test": [4, 5], "anchor_train": [1]}]}
    original = {"purge_edges": [{"a": ids[0], "b": ids[4]}]}
    anchor_protocol = {"anchor_purge_edges": [{"anchor_index": 0, "case_id": ids[5]}]}
    anchors = [{"text": "first extra"}, {"text": "second extra"}]
    rows = [{"case_id": cid, "held_out_source": "outer" if i >= 4 else "train"} for i, cid in enumerate(ids)]
    cases = {cid: {"query": f"query number {i}"} for i, cid in enumerate(ids)}
    _purges(protocol, original, anchor_protocol, anchors, rows, cases)
    protocol["folds"][0]["train"].insert(0, 0)
    with pytest.raises(ValueError, match="source/near"):
        _purges(protocol, original, anchor_protocol, anchors, rows, cases)
    protocol["folds"][0]["train"].pop(0)
    protocol["folds"][0]["anchor_train"].insert(0, 0)
    with pytest.raises(ValueError, match="anchor purge"):
        _purges(protocol, original, anchor_protocol, anchors, rows, cases)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -.01, 1.01, "0.5"])
def test_invalid_scores_cannot_become_valid_decisions(value):
    assert not _score(value)


@pytest.mark.parametrize("text", ['{"status":1,"status":2}', '{"value":NaN}', '{"value":Infinity}'])
def test_ambiguous_classifier_json_is_rejected(tmp_path, text):
    path = tmp_path / "report.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        _read(path)


def test_render_distinguishes_fixed_anchor_losses_from_nested_no_effect():
    def values(hits, fp, pairs, positives, negatives):
        return {"hit5_cases": hits, "negative_with_suggestion": fp, "negative_unique_case_norm_pairs": pairs,
                "positive_cases": positives, "negative_cases": negatives}
    before = {"main": values(68, 11, 29, 102, 30), "additional": values(13, 8, 11, 15, 15)}
    after = {"main": values(68, 4, 9, 102, 30), "additional": values(11, 1, 1, 15, 15)}
    report = {"variants": {}, "anchor_query_classifier": {
        "anchors": {"original_rows": 73, "unique_training_texts": 48},
        "datasets": {ds: {"before": before[ds], "after": after[ds]} for ds in before}},
        "nested_query_classifier": {"fallback_outer_folds": 2, "variants": {"kaskade": {
            "datasets": {ds: {"before": before[ds], "after": before[ds]} for ds in before}}}}}
    text = render(report)
    for phrase in ("13→11/15", "11→4/30", "29→9", "13→13/15", "keine gemeinsame Auswahl", "Alle Fälle bleiben im Nenner"):
        assert phrase in text


@pytest.fixture
def historical_gate_metrics():
    rows = [{"case_id": f"{dataset}:{negative}", "dataset": dataset, "negative": negative,
             "targets": [] if negative else ["norm:AO:42", "norm:AO:39"],
             "test": {"suggestions": [{"normen": [{"id": "norm:AO:42"}]}]}}
            for dataset in ("main", "additional") for negative in (False, True)]
    actual = _datasets(rows, "test", {row["case_id"]: not row["negative"] for row in rows}, include_all=True)
    legacy = deepcopy(actual)
    for sides in legacy.values():
        for metrics in sides.values():
            metrics.pop("historical_norm_targets")
            metrics.pop("matched_historical_norm_targets")
    return actual, legacy


def test_historical_instruction_and_nested_gates_keep_all_old_metrics_strict(historical_gate_metrics):
    actual, legacy = historical_gate_metrics
    original = deepcopy(legacy)
    _checked_dataset_metrics(actual, legacy, "gate mismatch")
    _checked_dataset_metrics(actual, actual, "gate mismatch")
    assert legacy == original
    assert actual["main"]["after"]["historical_norm_targets"] == 2
    assert actual["main"]["after"]["matched_historical_norm_targets"] == 1
    assert actual["all"]["after"]["historical_norm_targets"] == 4
    assert actual["all"]["after"]["matched_historical_norm_targets"] == 2


@pytest.mark.parametrize("tamper", ["old_value", "missing_old", "unknown_metric", "partial_new", "wrong_new",
                                   "missing_dataset", "unknown_dataset", "missing_side", "unknown_side", "wrong_combined"])
def test_historical_gate_compatibility_cannot_hide_changed_or_missing_metrics(historical_gate_metrics, tamper):
    actual, legacy = historical_gate_metrics
    metrics = legacy["main"]["after"]
    if tamper == "old_value":
        metrics["hit5_cases"] += 1
    elif tamper == "missing_old":
        metrics.pop("negative_cases")
    elif tamper == "unknown_metric":
        metrics["unverified"] = 0
    elif tamper == "partial_new":
        metrics["historical_norm_targets"] = 2
    elif tamper == "wrong_new":
        metrics.update(historical_norm_targets=2, matched_historical_norm_targets=2)
    elif tamper == "missing_dataset":
        legacy.pop("additional")
    elif tamper == "unknown_dataset":
        legacy["unverified"] = deepcopy(legacy["main"])
    elif tamper == "missing_side":
        legacy["main"].pop("before")
    elif tamper == "wrong_combined":
        legacy["all"]["before"]["negative_cases"] += 1
    else:
        legacy["main"]["unverified"] = deepcopy(metrics)
    with pytest.raises(ValueError, match="gate mismatch"):
        _checked_dataset_metrics(actual, legacy, "gate mismatch")
