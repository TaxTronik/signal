"""CPU-only recovery guards for observed states, binary logits and full coverage."""

from copy import deepcopy

import numpy as np
import pytest

from tools.quality_qwen8b_recovery import check_arrays, guard_original


def observations():
    """Small exact dyadic projections include both threshold sides and a tie."""
    hidden = np.array([[1, 2, 0, 0], [3, 4, 0, 1], [-1, .5, 1, 0],
                       [0, 0, 0, 0], [2, 0, 1, 2]], dtype=np.float16)
    rows = np.array([[.25, .5, -.5, 0], [.5, .25, 0, .25]], dtype=np.float16)
    logits = hidden.astype(np.float32) @ rows.astype(np.float32).T
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    scores = (np.exp(shifted[:, 1]) / np.exp(shifted).sum(axis=1)).astype(np.float32)
    tokens = [[1] * n for n in (3, 5, 2, 4, 3)]
    batches = [{"start": 0, "count": 2, "padded_tokens": 5},
               {"start": 2, "count": 2, "padded_tokens": 4},
               {"start": 4, "count": 1, "padded_tokens": 3}]
    return hidden, logits, scores, rows, batches, tokens, []


def test_valid_original_projections_and_odd_final_batch_are_preserved():
    values = observations()
    before = [x.copy() if isinstance(x, np.ndarray) else deepcopy(x) for x in values]
    assert values[2][3] == .5
    audit = check_arrays(*values)
    assert (audit["pairs"], audit["batches"], audit["oom_retries"]) == (5, 3, 0)
    assert audit["reduced_raw_scores_preserved"] is True
    assert audit["hidden_projection_max_abs_delta"] == 0
    assert audit["logits_to_score_max_abs_delta"] <= 2e-7
    for actual, original in zip(values, before, strict=True):
        if isinstance(actual, np.ndarray):
            assert np.array_equal(actual, original)
        else:
            assert actual == original


def test_declared_single_oom_retry_preserves_every_original_pair():
    hidden, logits, scores, rows, batches, tokens, _ = observations()
    batches[1:] = [{"start": i, "count": 1, "padded_tokens": len(tokens[i])} for i in range(2, 5)]
    retries = [{"phase": "decoder", "start": 2, "attempted_batch_size": 2,
                "retry_batch_size": 1, "error": "synthetic OOM"}]
    check_arrays(hidden, logits, scores, rows, batches, tokens, retries)


@pytest.mark.parametrize("array_index", [0, 1, 2, 3])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_observations_are_errors_instead_of_missing_results(array_index, invalid):
    values = list(observations())
    values[array_index].flat[-1] = invalid
    with pytest.raises(ValueError):
        check_arrays(*values)


@pytest.mark.parametrize("array_index", [0, 1, 2, 3])
def test_observation_precision_cannot_change_silently(array_index):
    values = list(observations())
    values[array_index] = values[array_index].astype(np.float64)
    with pytest.raises(ValueError):
        check_arrays(*values)


@pytest.mark.parametrize("corruption", ["hidden_missing", "logit_missing", "score_missing", "head_row_missing"])
def test_incomplete_observation_arrays_cannot_drop_the_last_pair(corruption):
    values = list(observations())
    index = {"hidden_missing": 0, "logit_missing": 1, "score_missing": 2, "head_row_missing": 3}[corruption]
    values[index] = values[index][:-1]
    with pytest.raises(ValueError):
        check_arrays(*values)


@pytest.mark.parametrize("corruption", ["changed_hidden", "swapped_no_yes_rows", "changed_logits", "changed_score"])
def test_scores_remain_bound_to_observed_states_and_original_no_yes_rows(corruption):
    values = list(observations())
    if corruption == "changed_hidden":
        values[0][-1, 0] += 10
    elif corruption == "swapped_no_yes_rows":
        values[3] = values[3][::-1].copy()
    elif corruption == "changed_logits":
        values[1][-1, 0] += 1
    else:
        values[2][-1] += .01
    with pytest.raises(ValueError):
        check_arrays(*values)


def test_a_rounding_sized_score_change_cannot_flip_an_exact_half_tie():
    values = list(observations())
    values[2][3] = np.nextafter(np.float32(.5), np.float32(0))
    with pytest.raises(ValueError):
        check_arrays(*values)


@pytest.mark.parametrize("corruption", ["gap", "duplicate", "missing_final", "wrong_padding", "unexpected_single"])
def test_batch_boundaries_cannot_skip_duplicate_or_rebatch_observations(corruption):
    values = list(observations())
    batches = values[4]
    if corruption == "gap":
        batches[1]["start"] = 3
    elif corruption == "duplicate":
        batches[1]["start"] = 0
    elif corruption == "missing_final":
        batches.pop()
    elif corruption == "wrong_padding":
        batches[0]["padded_tokens"] = 3
    else:
        batches[0]["count"] = 1
    with pytest.raises(ValueError):
        check_arrays(*values)


@pytest.mark.parametrize("invalid", [-.1, 1.1])
def test_out_of_range_scores_are_not_clamped(invalid):
    values = list(observations())
    values[2][0] = invalid
    with pytest.raises(ValueError):
        check_arrays(*values)


@pytest.mark.parametrize("invalid", [[], [True], [-1], [1.0], [1] * 2049])
def test_original_token_inventory_requires_nonempty_bounded_integer_sequences(invalid):
    values = list(observations())
    values[5][0] = invalid
    with pytest.raises(ValueError):
        check_arrays(*values)


@pytest.mark.parametrize("start", [-1, 3, 4, 5])
def test_retry_must_correspond_to_an_actual_failed_two_pair_batch(start):
    values = list(observations())
    values[6] = [{"phase": "decoder", "start": start,
                  "attempted_batch_size": 2, "retry_batch_size": 1}]
    with pytest.raises(ValueError):
        check_arrays(*values)


def original_failure():
    """Synthetic metadata reflects the single authorized failure contract."""
    manifest = {
        "format": "quality_qwen8b_probe/1", "status": "running",
        "unique_pairs": 14961, "case_counts": {"main": 132, "additional": 30},
        "model": {"name": "Qwen/Qwen3-Reranker-8B", "revision": "77d193c791ed757ca307ee72715aa132723da912"},
        "architecture_audit": {"original_parameter_tensors": 399},
        "instruction": "Judge whether the legal provision (Document) matches the legal concept "
                       "or factual situation described in the Query.",
        "max_length": 2048, "threshold": .5, "topk": 5,
        "source_hashes": {"tools/quality_qwen8b_probe.py": "synthetic-original-runner-sha256"},
        "runtime": {"dtype": "torch.float16", "initial_batch_size": 2,
                    "declared_oom_retry_batch_size": 1, "truncation": False, "quantization": None},
    }
    report = deepcopy(manifest)
    report.update(status="failed", rows=[], progress={"phase": "reduced_decoder", "completed_pairs": 14961,
                                                      "planned_pairs": 14961},
                  error="ValueError('GPU decoder was not released before full original head validation')")
    report["runtime"].update(post_decoder_allocated_bytes=79691776, reduced_scoring_seconds=123.0,
                              decoder_parameter_devices=["cuda:0"], input_embedding_device="cpu")
    return report, manifest


def test_only_completed_decoder_failure_is_accepted_without_rewriting_original_metadata():
    report, manifest = original_failure()
    originals = deepcopy((report, manifest))
    guard_original(report, manifest)
    assert (report, manifest) == originals
    assert report["status"] == "failed" and "error" in report


@pytest.mark.parametrize("field,value", [
    ("format", "quality_qwen8b_recovery/1"), ("status", "complete"), ("status", "running"),
    ("error", "ValueError('CUDA out of memory')"), ("error", None),
    ("unique_pairs", 14960), ("rows", [{"case_id": "synthetic"}]),
    ("head_binding", {}), ("numerical_audit", {}),
])
def test_other_failure_stages_or_partial_decoder_results_are_not_recoverable(field, value):
    report, manifest = original_failure()
    report[field] = value
    with pytest.raises(ValueError):
        guard_original(report, manifest)


@pytest.mark.parametrize("field,value", [
    ("completed_pairs", 14960), ("planned_pairs", 14960), ("phase", "full_head_recovery"),
])
def test_original_progress_requires_all_14961_decoder_inputs(field, value):
    report, manifest = original_failure()
    report["progress"][field] = value
    with pytest.raises(ValueError):
        guard_original(report, manifest)


@pytest.mark.parametrize("field,value", [
    ("post_decoder_allocated_bytes", 0), ("post_decoder_allocated_bytes", 79691777),
    ("reduced_scoring_seconds", 0), ("dtype", "torch.float32"),
    ("decoder_parameter_devices", ["cpu"]), ("input_embedding_device", "cuda:0"),
])
def test_original_runtime_must_match_the_observed_gpu_failure(field, value):
    report, manifest = original_failure()
    report["runtime"][field] = value
    with pytest.raises(ValueError):
        guard_original(report, manifest)


@pytest.mark.parametrize("field,value", [
    ("model", {"name": "Qwen/Qwen3-Reranker-4B", "revision": "another"}),
    ("architecture_audit", {"original_parameter_tensors": 398}),
])
def test_changing_both_manifest_and_report_cannot_admit_another_model(field, value):
    report, manifest = original_failure()
    report[field] = manifest[field] = value
    with pytest.raises(ValueError):
        guard_original(report, manifest)


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("instruction", "A different model instruction."),
    ("max_length", 1024), ("threshold", .4), ("topk", 6),
    ("source_hashes", {"tools/quality_qwen8b_probe.py": "modified"}),
    ("runtime", {"initial_batch_size": 1}),
])
def test_original_prerun_manifest_stays_bound_to_the_failed_run(field, value):
    report, manifest = original_failure()
    manifest[field] = value
    with pytest.raises(ValueError):
        guard_original(report, manifest)
