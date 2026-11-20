"""Model-free accounting/audit tests with synthetic TEST identifiers, not Gold."""

from types import SimpleNamespace
import json

import pytest

from tools.quality_recall_report import (
    bind_report, classifier_aggregate, key_digest, paired_changes, public_payload_check, read,
    live_suggestion_difference, render, score_cache, score_workload, source_evidence_audit, variant_metrics,
)


def row(case_id, *, negative=False, norms=(), targets=("norm:TEST:1",)):
    return {"case_id": case_id, "negative": negative, "targets": [] if negative else list(targets),
            "v": {"suggestions": [{"normen": [{"id": nid} for nid in ids]} for ids in norms]}}


def test_negative_pairs_deduplicate_within_case_across_focus_but_not_across_cases():
    rows = [row("one", negative=True, norms=[("norm:TEST:1", "norm:TEST:2"), ("norm:TEST:1",)]),
            row("two", negative=True, norms=[("norm:TEST:1",)]),
            row("three", negative=True)]
    result = variant_metrics(rows, "v")
    assert result["negative_cases"] == 3
    assert result["negative_with_suggestion"] == 2
    assert result["negative_entries"] == 4
    assert result["negative_unique_case_norm_pairs"] == 3


def test_unlisted_positive_norms_remain_unadjudicated_and_prefixes_have_segment_boundaries():
    rows = [row("one", norms=[("norm:TEST:10", "norm:TEST:1:abs2"), ("norm:TEST:1",)])]
    result = variant_metrics(rows, "v")
    assert result["hit1_cases"] == result["hit5_cases"] == 1
    assert result["historical_target_matching_entries"] == 2
    assert result["unadjudicated_non_target_entries"] == 1
    assert result["suggestion_precision"] is None and result["full_legal_recall"] is None
    assert result["mean_target_coverage"] == result["mrr_best_sentence"] == 1.


def test_historical_target_counts_deduplicate_targets_and_foci_within_each_positive_case():
    rows = [row("one", targets=("norm:TEST:1", "norm:TEST:1", "norm:TEST:2", "norm:TEST:3"),
                norms=[("norm:TEST:1:abs2", "norm:TEST:99", "norm:TEST:10"), ("norm:TEST:1", "norm:TEST:2")]),
            row("two", norms=[("norm:TEST:1",)]),
            row("negative", negative=True, norms=[("norm:TEST:1",)])]
    result = variant_metrics(rows, "v")
    assert result["historical_norm_targets"] == 4
    assert result["matched_historical_norm_targets"] == 3
    assert result["positive_cases"] == result["hit5_cases"] == 2
    assert result["historical_target_matching_entries"] == 4
    assert result["unadjudicated_non_target_entries"] == 2
    assert result["mean_target_coverage"] == pytest.approx((2 / 3 + 1) / 2)
    assert result["negative_entries"] == result["negative_with_suggestion"] == 1


def test_more_false_norms_in_an_already_affected_case_do_not_disappear_in_case_metric():
    before = {"one": row("one", negative=True, norms=[("norm:TEST:1",)])}
    after = {"one": row("one", negative=True, norms=[("norm:TEST:1", "norm:TEST:2"), ("norm:TEST:2",)])}
    change = paired_changes(before, after, "v", "v")
    assert change["negative_case_losses"] == change["negative_case_wins"] == []
    assert change["negative_norm_pairs_added"] == [["one", "norm:TEST:2"]]
    assert change["negative_norm_pairs_removed"] == []


def test_equal_negative_case_totals_preserve_paired_churn():
    before = {"one": row("one", negative=True, norms=[("norm:TEST:1",)]),
              "two": row("two", negative=True)}
    after = {"one": row("one", negative=True),
             "two": row("two", negative=True, norms=[("norm:TEST:1",)])}
    change = paired_changes(before, after, "v", "v")
    assert change["negative_case_wins"] == ["one"]
    assert change["negative_case_losses"] == ["two"]


def test_best_sentence_topfive_differs_from_unbounded_target_presence():
    result = variant_metrics([row("one", norms=[tuple(f"norm:TEST:{i}" for i in range(2, 7)) + ("norm:TEST:1",)])], "v")
    assert result["hit5_cases"] == 0 and result["mean_target_coverage"] == 1.
    assert result["mrr_best_sentence"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("private_key", ["query", "document", "case_id", "targets", "passagen"])
def test_public_projection_refuses_nested_case_payload(private_key):
    with pytest.raises(ValueError, match="payload"):
        public_payload_check({"safe_counts": [1, {private_key: "private"}]})
    public_payload_check({"negative_cases": 2, "models": {"revision": "known"}, "paired_counts": {"wins": 1}})


@pytest.mark.parametrize("report", [{"status": "running"}, {"status": "complete", "diagnostic_subset": True},
                                     {"status": "failed", "error": "backend"}])
def test_incomplete_runs_are_rejected_before_files_or_metrics(tmp_path, report):
    with pytest.raises(ValueError, match="completed full"):
        bind_report(tmp_path / "missing.json", report)


def test_exact_cache_keys_separate_instruction_and_token_budget(tmp_path):
    path = tmp_path / "scores.jsonl"
    entries = [[None, 1024, "query", "document", .2], ["instruction", 2048, "query", "document", .8]]
    path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    values = score_cache(path)
    assert values[(None, 1024, "query", "document")] == .2
    assert values[("instruction", 2048, "query", "document")] == .8
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n" + json.dumps([None, 1024, "query", "document", .3]))
    with pytest.raises(ValueError, match="Conflicting"):
        score_cache(path)


def test_source_audit_checks_original_and_parent_offsets_after_overlap_trimming():
    source = "abcdefghijklmnopqrstuvwxyz"
    parent = SimpleNamespace(passage_id="parent", norm_id="norm:TEST:1", start=3, end=20, text=source[3:20])
    index = SimpleNamespace(passage_evidence=SimpleNamespace(passages=[parent], source_fingerprint="source-generation"))
    graph = SimpleNamespace(node=lambda nid: {"text": source})
    passage = {"id": "parent", "norm_id": "norm:TEST:1", "quell_start": 8, "quell_end": 15, "text": source[8:15]}
    suggestions = [{"normen": [{"id": "norm:TEST:1", "evidenz": {"fingerprint": "source-generation", "passagen": [passage]}}]}]
    assert source_evidence_audit(suggestions, index, graph) == 1
    passage["quell_start"], passage["quell_end"], passage["text"] = 1, 3, source[1:3]
    with pytest.raises(ValueError, match="parent slice"):
        source_evidence_audit(suggestions, index, graph)


def test_strict_json_rejects_duplicate_report_keys(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"status":"running","status":"complete"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read(path)


@pytest.mark.parametrize("report", [
    {"format": "query_classifier_purged_logo/1", "status": "not_evaluable", "model_saved": False},
    {"format": "query_classifier_purged_logo/1", "status": "complete", "model_saved": True},
])
def test_classifier_aggregate_rejects_incomplete_or_saved_model_before_metrics(tmp_path, report):
    path = tmp_path / "classifier.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="undeployed CPU"):
        classifier_aggregate(path, tmp_path / "baseline.json", {})


def test_classifier_render_preserves_both_hit_losses_and_false_positive_reductions():
    def metric(hits, positive, false, negative):
        return {"hit5_cases": hits, "positive_cases": positive,
                "negative_with_suggestion": false, "negative_cases": negative}

    value = {"variants": {}, "query_need_classifier": {"datasets": {
        "main": {"before": metric(68, 102, 11, 30), "after": metric(66, 102, 2, 30)},
        "additional": {"before": metric(13, 15, 8, 15), "after": metric(8, 15, 0, 15)},
    }}}
    text = render(value)
    for phrase in ("68→66/102", "11→2/30", "13→8/15", "8→0/15", "Laufzeit verworfen"):
        assert phrase in text
    public_payload_check(value)


def test_render_shows_micro_target_counts_for_baseline_research_and_live_separately_from_case_hits():
    def metrics(hits, targets, matched):
        return {"positive_cases": 102, "negative_cases": 30, "hit5_cases": hits, "negative_with_suggestion": 11,
                "negative_entries": 29, "negative_unique_case_norm_pairs": 29,
                "historical_norm_targets": targets, "matched_historical_norm_targets": matched}
    variants = {name: {"datasets": {"main": metrics(hits, 176, matched), "additional": metrics(13, 27, 18)}}
                for name, hits, matched in (("baseline/kaskade", 68, 84), ("research/package20", 80, 99),
                                            ("live/treffer", 80, 99))}
    text = render({"variants": variants})
    for name in variants:
        assert text.count(f"| {name} |") == 2
    for value in ("84/176", "99/176", "18/27", "68/102", "80/102", "pro positivem Fall einmal"):
        assert value in text


def test_live_metadata_details_separate_catalog_chains_from_scores_and_other_source_changes():
    from copy import deepcopy
    expected = [{"start": 0, "end": 10, "normen": [{"id": "norm:TEST:1", "ketten": [], "sim": .8,
                 "rerank": .9, "evidenz": {"fingerprint": "source"}}]}]
    actual = deepcopy(expected)
    assert live_suggestion_difference(actual, expected)["classification"] == "identical_suggestions"
    actual[0]["normen"][0]["ketten"] = [{"test_chain": "catalog"}]
    assert live_suggestion_difference(actual, expected) == {"classification": "chain_metadata_only", "changed_chain_entries": 1}
    actual[0]["normen"][0]["rerank"] = .899
    assert live_suggestion_difference(actual, expected)["classification"] == "chain_metadata_and_scores_only"
    actual[0]["normen"][0]["evidenz"]["fingerprint"] = "different source"
    assert live_suggestion_difference(actual, expected)["classification"] == "other_suggestion_difference"
    actual[0]["normen"][0]["id"] = "norm:TEST:2"
    assert live_suggestion_difference(actual, expected)["classification"] == "unaligned_suggestions"
    assert expected[0]["normen"][0]["ketten"] == []


def test_score_workload_distinguishes_unused_imports_from_fresh_runtime_inputs(tmp_path):
    first, unused, fresh = ((None, 1024, name, "document") for name in ("first", "unused", "fresh"))
    parent = tmp_path / "parent.jsonl"
    parent.write_text("\n".join(json.dumps([*key, score]) for key, score in ((first, .2), (unused, .3))), encoding="utf-8")
    counts = score_workload({first: .2, unused: .3, fresh: .4}, {key_digest(first), key_digest(fresh)}, parent)
    assert counts == {"pairs": 3, "inherited_pairs": 2, "fresh_pairs": 1, "used_pairs": 2,
                      "inherited_used_pairs": 1, "fresh_used_pairs": 1, "inherited_unused_pairs": 1}


def test_score_workload_rejects_changed_inherited_score_and_unexplained_fresh_input(tmp_path):
    key = ("instruction", 2048, "query", "document")
    parent = tmp_path / "parent.jsonl"
    parent.write_text(json.dumps([*key, .2]), encoding="utf-8")
    with pytest.raises(ValueError, match="score changed"):
        score_workload({key: .3}, {key_digest(key)}, parent)
    with pytest.raises(ValueError, match="not used"):
        score_workload({key: .2}, set())


def test_render_marks_partial_pilot_as_excluded_and_cache_work_as_shared():
    counts = {"inherited_pairs": 2, "fresh_pairs": 1, "used_pairs": 2,
              "inherited_used_pairs": 1, "inherited_unused_pairs": 1}
    report = {"variants": {}, "reports": {"corrected": {
        "inherited_pilot": {"completed_cases": 61, "planned_cases": 162},
        "audit": {"actual_cache_workload": {"bge": counts, "qwen": counts}},
    }}}
    text = render(report)
    assert "61/162" in text and "Pilot liefert keine vollständige Qualitätskennzahl" in text
    assert "keine frische Inferenz jedes einzelnen Profils" in text
    assert "8192-Token-Budget" in text and "keine tatsächliche BGE-Eingabegrenze" in text


def test_complete_live_mismatch_is_auditable_but_source_failure_is_rejected(tmp_path):
    import hashlib
    rows = []
    for dataset, positive, negative in (("main", 102, 30), ("additional", 15, 15)):
        for i in range(positive + negative):
            rows.append({"case_id": f"{dataset}:{i}", "dataset": dataset,
                         "negative": i >= positive, "targets": [] if i >= positive else ["norm:TEST:1"]})
    data = tmp_path / "rows.jsonl"
    data.write_text("", encoding="utf-8")
    report = {"format": "quality_recall_live/1", "status": "mismatch", "rows": rows,
              "source_hashes": {}, "input_hashes": {}, "frozen_inputs": {},
              "rows_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "source_audit": {"errors": []}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert str(data) in bind_report(path, report)
    report["source_audit"]["errors"] = ["invalid source"]
    with pytest.raises(ValueError, match="source evidence audit failed"):
        bind_report(path, report)


def test_render_prioritizes_actual_live_values_and_keeps_difference_counts_and_timing_separate():
    def metrics(hits):
        return {"hit5_cases": hits, "positive_cases": 102, "negative_with_suggestion": 3,
                "negative_cases": 30, "negative_entries": 7, "negative_unique_case_norm_pairs": 6}
    timing = {"n": 162, "mean_seconds": 1.23, "median_seconds": 1., "p95_seconds": 2.5}
    report = {"variants": {
        "live/public": {"datasets": {"main": metrics(80), "additional": metrics(12)}},
        "research/chosen": {"datasets": {"main": metrics(79), "additional": metrics(13)}},
    }, "live_result": {"cases": 162, "reference_differences": 2,
                       "difference_categories": {"score_only": 1, "selection": 1}, "timings": {"all": timing}}}
    text = render(report)
    assert text.index("| live/public") < text.index("| research/chosen")
    assert "80/102" in text and "79/102" in text and "2 abweichende Ausgaben" in text
    assert "score_only: 1, selection: 1" in text and "1.230" in text
    public_payload_check(report)


def test_rejected_mean_is_separate_from_main_variant_table_and_preserves_all_denominators():
    def value(hits, positive, false, negative, pairs):
        return {"hit5_cases": hits, "positive_cases": positive, "negative_with_suggestion": false,
                "negative_cases": negative, "negative_unique_case_norm_pairs": pairs}
    report = {"variants": {}, "instruction_mean": {"datasets": {
        "main": value(79, 102, 13, 30, 36), "additional": value(13, 15, 9, 15, 21)}}}
    text = render(report)
    for phrase in ("79/102", "13/30", "36 eindeutige", "13/15", "9/15", "21 eindeutige",
                   "nachträglicher Entwicklungsversuch", "keine neuen Modellaufrufe", "keine Aktivierung"):
        assert phrase in text
    public_payload_check(report)
