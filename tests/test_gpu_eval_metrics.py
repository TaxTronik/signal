"""Handpruefbare A/B-Metriken: mehrere Satzlisten, Gold-Abdeckung und Negativ-Gold."""

import pytest

from tools._gpu_eval_metrics import compare_rows, evaluate_variant, summarize


def _suggestions(*sentences):
    return [{"normen": [{"id": norm_id} for norm_id in sentence]} for sentence in sentences]


def _row(case_id, before, after, *, negative=False, group="positive", targets=None):
    return {
        "case_id": case_id, "query": "Ein Gold-Fall.", "negative": negative,
        "group": group, "targets": ["norm:AO:14"] if targets is None else targets,
        "before": {"suggestions": before}, "after": {"suggestions": after},
    }


def test_multiple_sentences_use_best_rank_and_union_of_targets():
    suggestions = _suggestions(
        ["norm:AO:140", "norm:AO:42", "norm:AO:69"],
        ["norm:AO:14:abs1", "norm:AO:42"],
    )
    result = evaluate_variant(suggestions, ["norm:AO:14", "norm:AO:42", "norm:AO:34"])
    assert result == {
        "best_rank": 1, "target_coverage": 2 / 3, "n_suggestions": 2, "n_norms": 5,
    }


@pytest.mark.parametrize(("candidate", "target", "hit"), [
    ("norm:AO:14", "norm:AO:14", True),
    ("norm:AO:14:abs1:nr2", "norm:AO:14", True),
    ("norm:AO:140", "norm:AO:14", False),
    ("norm:AO:140:abs1", "norm:AO:14", False),
    ("norm:AO:14", "norm:AO:14:abs1", False),
    ("norm:KStG:14", "norm:AO:14", False),
])
def test_target_matching_respects_segment_boundaries(candidate, target, hit):
    result = evaluate_variant(_suggestions([candidate]), [target])
    assert result["best_rank"] == (1 if hit else None)
    assert result["target_coverage"] == (1.0 if hit else 0.0)


def test_duplicate_targets_and_repeated_norms_do_not_inflate_coverage():
    result = evaluate_variant(
        _suggestions(["norm:AO:14"], ["norm:AO:14"]),
        ["norm:AO:14", "norm:AO:14", "norm:AO:42"],
    )
    assert result["target_coverage"] == 0.5
    assert result["n_norms"] == 2


def test_empty_suggestions_and_no_gold_targets_are_defined():
    assert evaluate_variant([], []) == {
        "best_rank": None, "target_coverage": 0.0, "n_suggestions": 0, "n_norms": 0,
    }
    result = evaluate_variant(_suggestions(["norm:AO:14"]), [])
    assert result["best_rank"] is None
    assert result["target_coverage"] == 0.0
    assert result["n_norms"] == 1


def test_query_metrics_are_not_weighted_by_number_of_sentences():
    rows = [
        _row("hit1", [], _suggestions(["norm:AO:140"], ["norm:AO:14"])),
        _row("hit3", [], _suggestions(["norm:AO:140", "norm:AO:42", "norm:AO:14"])),
        _row("miss", [], []),
    ]
    result = summarize(iter(rows), "after")["positive"]
    assert result["n"] == 3
    assert result["hit@1"] == pytest.approx(1 / 3)
    assert result["hit@5"] == pytest.approx(2 / 3)
    assert result["mrr_best_sentence"] == pytest.approx((1 + 1 / 3) / 3)
    assert result["mean_target_coverage"] == pytest.approx(2 / 3)
    assert result["mean_suggestions"] == 1.0
    assert result["mean_norms"] == pytest.approx(5 / 3)


def test_rank_six_does_not_count_as_hit_five():
    result = summarize([
        _row("rank6", [], _suggestions([
            "norm:AO:1", "norm:AO:2", "norm:AO:3", "norm:AO:4", "norm:AO:5", "norm:AO:14",
        ])),
    ], "after")["positive"]
    assert result["hit@5"] == 0.0
    assert result["mrr_best_sentence"] == pytest.approx(1 / 6)


def test_negative_metrics_count_original_queries_and_all_displayed_norms():
    rows = [
        _row("fp", [], _suggestions(["norm:AO:14"], ["norm:AO:42", "norm:AO:69"]),
             negative=True, group="tax_neutral", targets=[]),
        _row("clean", [], [], negative=True, group="everyday", targets=[]),
        _row("empty_placeholder", [], _suggestions([]),
             negative=True, group="everyday", targets=[]),
    ]
    result = summarize(rows, "after")
    assert result["negative"]["n"] == 3
    assert result["negative"]["false_suggestion_rate"] == pytest.approx(1 / 3)
    assert result["negative"]["mean_norms"] == 1.0
    assert result["groups"]["tax_neutral"]["negative"]["false_suggestion_rate"] == 1.0
    assert result["groups"]["everyday"]["negative"]["false_suggestion_rate"] == 0.0
    assert list(result["groups"]) == ["everyday", "tax_neutral"]


def test_empty_aggregates_are_unmeasured_and_mixed_groups_remain_separate():
    empty = summarize([], "before")
    assert empty["groups"] == {}
    for kind in ("positive", "negative"):
        assert empty[kind]["n"] == 0
        assert all(value is None for key, value in empty[kind].items() if key != "n")
    result = summarize([
        _row("positive", [], _suggestions(["norm:AO:14"]), group="mixed"),
        _row("negative", [], [], group="mixed", negative=True, targets=[]),
    ], "after")
    assert result["groups"]["mixed"]["positive"]["n"] == 1
    assert result["groups"]["mixed"]["negative"]["n"] == 1


def test_positive_case_without_targets_is_a_data_error():
    with pytest.raises(ValueError, match="empty_gold"):
        summarize([_row("empty_gold", [], [], targets=[])], "after")


def test_paired_wins_losses_use_hit_five_and_negative_abstention():
    hit = _suggestions(["norm:AO:14"])
    rank2 = _suggestions(["norm:AO:140", "norm:AO:14"])
    miss = _suggestions(["norm:AO:140"])
    rows = [
        _row("p_win", miss, hit),
        _row("p_loss", hit, []),
        _row("p_same_class", rank2, hit),
        _row("n_win", hit, [], negative=True, targets=[]),
        _row("n_loss", [], miss, negative=True, targets=[]),
        _row("n_same_class", hit, miss, negative=True, targets=[]),
    ]
    assert compare_rows(rows) == {
        "positive_hit5_wins": ["p_win"], "positive_hit5_losses": ["p_loss"],
        "negative_abstention_wins": ["n_win"], "negative_abstention_losses": ["n_loss"],
    }
    assert all(not cases for cases in compare_rows([]).values())
