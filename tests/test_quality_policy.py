"""Policy comparisons must preserve raw thresholds and sentence-level evidence."""

from copy import deepcopy

import pytest

from tools._quality_policy import POLICY_SPECS, apply_policy


def candidate(name, rerank, sim):
    return {"norm_id": name, "rerank": rerank, "sim": sim}


@pytest.mark.parametrize("policy, expected", [
    ("baseline_or", ["strong", "weak", "unsupported", "supported"]),
    ("rerank_only", ["strong"]),
    ("weak_rescue", ["strong", "weak"]),
    ("supported_rescue", ["strong", "supported"]),
])
def test_predeclared_tradeoffs(policy, expected):
    pool = [candidate("strong", 0.1, None), candidate("weak", 0.01, 0.55),
            candidate("unsupported", 0.005, 0.6), candidate("supported", 0.001, 0.6),
            candidate("trigger_only", 0.09, None)]
    assert [n["norm_id"] for n in apply_policy(
        pool, policy, trigger_ids={"supported", "trigger_only"})] == expected


def test_thresholds_use_unrounded_values_and_filter_before_topk():
    pool = [candidate("sim_below", 0.099999, 0.549999),
            candidate("floor_below", 0.009999, 0.9),
            candidate("survivor", 0.01, 0.55)]
    assert [n["norm_id"] for n in apply_policy(pool, "weak_rescue", topk=1)] == ["survivor"]


def test_order_is_stable_and_inputs_unchanged():
    pool = [candidate("z", 0.5, 0.6), candidate("a", 0.5, 0.6),
            candidate("b", 0.5, 0.7)]
    original = deepcopy(pool)
    result = apply_policy(pool, "baseline_or")
    assert [n["norm_id"] for n in result] == ["b", "a", "z"]
    result[0]["rerank"] = 0
    assert pool == original


@pytest.mark.parametrize("rerank", [None, float("nan"), float("inf"), True, "0.1"])
def test_missing_or_invalid_inference_is_not_silent_abstention(rerank):
    with pytest.raises(ValueError, match="finite raw score"):
        apply_policy([candidate("x", rerank, 0.7)], "baseline_or")


def test_support_requires_independent_trigger_provenance():
    with pytest.raises(ValueError, match="actual trigger_ids"):
        apply_policy([], "supported_rescue")


def test_duplicate_candidates_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        apply_policy([candidate("x", 0.5, 0.6)] * 2, "baseline_or")


def test_policy_set_is_compact_and_fixed():
    assert tuple(POLICY_SPECS) == (
        "baseline_or", "rerank_only", "weak_rescue", "supported_rescue")
