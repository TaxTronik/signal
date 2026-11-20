"""Independent model-free Qwen8B accounting, input and publication contracts."""
from copy import deepcopy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from tools import _quality_recall_qwen8b as helper
from tools import quality_recall_report as audit
from tools._quality_recall_qwen8b import (
    VARIANTS, checked_array, checked_batches, checked_numerics, checked_outputs,
    checked_projections, replay, validate_inputs,
)


def candidate(nid, rank, *, sim=.4, bge=.8):
    document = nid + "\n\nOriginale synthetische Testquelle."
    return {"id": nid, "sim": sim, "bge": bge, "bge_rank": rank, "document": document,
            "document_sha256": hashlib.sha256(document.encode()).hexdigest(),
            "original_input_sha256": "original", "old_qwen": None, "old_qwen_in_kaskade20": rank <= 20}


def case(candidates, *, case_id="test", negative=False):
    query = "Der synthetische Sachverhalt steht vollständig hier."
    group = {"query": query, "start": 0, "end": len(query), "context_query": query,
             "context_start": 0, "context_end": len(query), "top20_count": min(20, len(candidates)),
             "top50_count": len(candidates), "candidates": candidates}
    return {"case_id": case_id, "negative": negative, "targets": [] if negative else ["norm:TEST:1"],
            "query": query, "dataset": "main", "group": "synthetic", "category": "test", "uncertain": False,
            "groups": [group]}


def score_map(row, values):
    group = row["groups"][0]
    return {(group["query"], c["document"]): value for c, value in zip(group["candidates"], values, strict=True)}


def full_inputs():
    cases = []
    for dataset, positives, negatives, double_targets in (("main", 102, 30, 74), ("additional", 15, 15, 12)):
        for i in range(positives + negatives):
            row = case([candidate("norm:TEST:1", 1)], case_id=f"{dataset}:{i}", negative=i >= positives)
            row["dataset"] = dataset
            if i < double_targets:
                row["targets"].append("norm:TEST:2")
            cases.append(row)
    return {"format": "quality_recall_inputs/1", "status": "complete", "torch_imported": False,
            "profile": "kaskade", "package": "ranked", "cases": cases,
            "counts": {"cases": 162, "groups": 162, "top20_candidates": 162, "top50_candidates": 162, "known_old_qwen": 0}}


def test_replay_applies_raw_half_without_similarity_rescue():
    row = case([candidate("norm:TEST:1", 1, sim=.99), candidate("norm:TEST:2", 2, sim=None)])
    result = replay([row], score_map(row, [.4999, .5]), 20)["test"]
    assert [n["id"] for n in result[0]["normen"]] == ["norm:TEST:2"]
    assert result[0]["normen"][0]["rerank"] == .5


def test_top20_and_top50_are_prefixes_before_final_raw_score_selection():
    row = case([candidate(f"norm:TEST:{i:02}", i) for i in range(1, 22)])
    scores = score_map(row, [.1] * 20 + [.99])
    assert replay([row], scores, 20)["test"] == []
    assert replay([row], scores, 50)["test"][0]["normen"] == [
        {"id": "norm:TEST:21", "sim": .4, "rerank": .99}]


def test_ties_use_similarity_then_identity_and_output_is_capped_at_five():
    ids = ["06", "05", "04", "03", "02", "01"]
    row = case([candidate(f"norm:TEST:{suffix}", i + 1, sim=None if suffix == "01" else .4)
                for i, suffix in enumerate(ids)])
    norms = replay([row], score_map(row, [.5] * 6), 50)["test"][0]["normen"]
    assert [n["id"] for n in norms] == [f"norm:TEST:{i:02}" for i in range(2, 7)]


def test_labels_never_change_replay_and_repeated_foci_are_preserved():
    row = case([candidate("norm:TEST:1", 1)])
    row["groups"].append(deepcopy(row["groups"][0]))
    scores = score_map(row, [.9])
    before = replay([row], scores, 20)
    row.update(negative=True, targets=["norm:TEST:999"])
    assert replay([row], scores, 20) == before
    assert len(before["test"]) == 2


@pytest.mark.parametrize("value", [True, "0.9", float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_raw_score_is_not_treated_as_abstention(value):
    row = case([candidate("norm:TEST:1", 1)])
    with pytest.raises(ValueError, match="raw score"):
        replay([row], score_map(row, [value]), 20)


def test_missing_exact_input_score_fails_loudly():
    row = case([candidate("norm:TEST:1", 1)])
    with pytest.raises(ValueError, match="Missing exact"):
        replay([row], {}, 20)


def test_input_validation_keeps_all_historical_case_and_target_denominators():
    inputs = full_inputs()
    assert len(validate_inputs(inputs)) == 162
    inputs["cases"][0]["targets"].pop()
    with pytest.raises(ValueError, match="norm-target denominator"):
        validate_inputs(inputs)


@pytest.mark.parametrize("change", ["case_missing", "partial", "bad_source", "bad_focus", "bool_rank", "duplicate_norm"])
def test_inputs_reject_incomplete_or_unbound_candidate_data(change):
    inputs = full_inputs()
    row = inputs["cases"][0]
    group = row["groups"][0]
    if change == "case_missing":
        inputs["cases"].pop()
    elif change == "partial":
        inputs["status"] = "running"
    elif change == "bad_source":
        group["candidates"][0]["document"] += " changed"
    elif change == "bad_focus":
        group["start"] = 1
    elif change == "bool_rank":
        group["candidates"][0]["bge_rank"] = True
    else:
        group["candidates"].append(deepcopy(group["candidates"][0]))
    with pytest.raises(ValueError):
        validate_inputs(inputs)


def complete_outputs():
    from tools._gpu_eval_metrics import summarize
    inputs = full_inputs()
    rows, baseline = [], []
    scores = {"reduced": np.array([.5], np.float32), "full_gpu": np.array([.4999], np.float32)}
    for case_row in inputs["cases"]:
        row = {key: value for key, value in case_row.items() if key != "groups"}
        baseline.append({**row, "kaskade": {"suggestions": []}})
        for name, config in VARIANTS.items():
            value = float(scores[config["head"]][0])
            row[name] = {"suggestions": replay([case_row], score_map(case_row, [value]), config["shortlist"])[row["case_id"]]}
        rows.append(row)
    report = {"rows": rows, "variants": VARIANTS,
              "summaries": {dataset: {name: summarize([r for r in rows if r["dataset"] == dataset], name)
                                      for name in VARIANTS} for dataset in audit.DATASETS},
              "output_differences": [{"case_id": r["case_id"], "dataset": r["dataset"], "shortlist": size,
                                      "norm_order_equal": False} for size in (20, 50) for r in rows]}
    group = inputs["cases"][0]["groups"][0]
    pairs = [{"query": group["query"], "document": group["candidates"][0]["document"]}]
    return report, inputs, pairs, scores, {"rows": baseline}


def test_complete_raw_replay_includes_both_heads_full_target_denominators_and_only_aggregate_payload():
    report, inputs, pairs, scores, baseline = complete_outputs()
    values, differences = checked_outputs(report, inputs, pairs, scores, baseline)
    assert set(values) == set(VARIANTS)
    for dataset, positives, negatives, targets in (("main", 102, 30, 176), ("additional", 15, 15, 27)):
        positive = values["qwen8b_reduced_top20"]["datasets"][dataset]
        assert positive["hit5_cases"] == positive["matched_historical_norm_targets"] == positives
        assert positive["negative_with_suggestion"] == positive["negative_unique_case_norm_pairs"] == negatives
        assert positive["historical_norm_targets"] == targets and positive["suggestion_precision"] is None
        assert values["qwen8b_full_gpu_top20"]["datasets"][dataset]["hit5_cases"] == 0
        assert differences[dataset]["50"]["different_norm_orders"] == positives + negatives
    projection = {"variants": values, "differences": differences}
    audit.public_payload_check(projection)
    serialized = json.dumps(projection)
    assert "norm:TEST" not in serialized and inputs["cases"][0]["query"] not in serialized
    assert inputs["cases"][0]["case_id"] not in serialized


@pytest.mark.parametrize("corruption", ["scores", "arm_missing", "pairs", "metrics", "differences", "labels", "focus"])
def test_independent_output_audit_rejects_tampering_and_partial_arms(corruption):
    report, inputs, pairs, scores, baseline = complete_outputs()
    if corruption == "scores":
        scores["reduced"][0] = .4999
    elif corruption == "arm_missing":
        report["variants"] = {k: v for k, v in VARIANTS.items() if not k.endswith("top50")}
    elif corruption == "pairs":
        pairs[0]["document"] += " altered"
    elif corruption == "metrics":
        report["summaries"]["main"]["qwen8b_reduced_top20"]["negative"]["n"] = 29
    elif corruption == "differences":
        report["output_differences"].pop()
    elif corruption == "labels":
        baseline["rows"][0]["targets"] = ["norm:TEST:99"]
    else:
        report["rows"][0]["qwen8b_reduced_top20"]["suggestions"][0]["start"] = 1
    with pytest.raises(ValueError):
        checked_outputs(report, inputs, pairs, scores, baseline)


@pytest.mark.parametrize("shape,dtype,invalid", [((2,), "float32", False), ((2, 2), "float16", False),
                                               ((2, 2), "float32", True)])
def test_array_audit_refuses_missing_rows_wrong_precision_and_nonfinite_data(tmp_path, shape, dtype, invalid):
    value = np.zeros(shape, dtype=dtype)
    if invalid:
        value.flat[0] = np.nan
    path = tmp_path / "array.npy"
    np.save(path, value, allow_pickle=False)
    with pytest.raises(ValueError):
        checked_array(path, (2, 2), "float32")


def numeric_data():
    logits = {"reduced": np.array([[0., 0.], [0., 1.]], np.float32),
              "full_gpu": np.array([[0., -.0004], [0., 1.]], np.float32)}
    scores = {name: (np.exp(value.astype(np.float64)[:, 1]) / np.exp(value.astype(np.float64)).sum(axis=1)).astype(np.float32)
              for name, value in logits.items()}
    delta = np.abs(scores["reduced"].astype(np.float64) - scores["full_gpu"].astype(np.float64))
    stored = {"pairs": 2, "different_logit_pairs": 1, "different_scores": 1,
              "max_abs_score_difference": float(delta.max()), "max_abs_logit_difference": float(np.float32(.0004)),
              "threshold_crossings": 1, "exact_scores": False}
    return logits, scores, stored


def test_binary_softmax_and_threshold_crossings_are_recomputed_from_both_raw_logit_arrays():
    logits, scores, stored = numeric_data()
    assert checked_numerics(logits, scores, stored) == stored
    scores["full_gpu"][0] = .5
    with pytest.raises(ValueError, match="binary logits"):
        checked_numerics(logits, scores, stored)


def test_small_numerical_change_cannot_hide_a_raw_threshold_crossing():
    logits = {name: np.array([[0., 0.]], np.float32) for name in ("reduced", "full_gpu")}
    scores = {name: np.array([.5], np.float32) for name in logits}
    scores["full_gpu"][0] = np.nextafter(np.float32(.5), np.float32(0))
    with pytest.raises(ValueError, match="threshold side"):
        checked_numerics(logits, scores, {})


@pytest.mark.parametrize("status", ["running", "preflight_pass", "failed"])
def test_aggregate_rejects_unfinished_or_cpu_only_runs_before_loading_model_assets(tmp_path, status):
    path, baseline = tmp_path / "report.json", tmp_path / "baseline.json"
    audit.save(path, {"format": "quality_qwen8b_probe/1", "status": status})
    audit.save(baseline, {})
    with pytest.raises(ValueError, match="completed original GPU"):
        helper.aggregate(path, baseline, tmp_path / "audit")


def test_public_render_keeps_all_four_fixed_arms_numerical_differences_and_research_scope():
    report, inputs, pairs, scores, baseline = complete_outputs()
    values, differences = checked_outputs(report, inputs, pairs, scores, baseline)
    text = audit.render({"variants": {}, "qwen8b": {"variants": values, "output_difference_counts": differences,
                        "numerical_audit": {"pairs": 1, "different_scores": 1, "threshold_crossings": 1}}})
    assert all(name in text for name in VARIANTS)
    for value in ("102/102", "0/102", "15/15", "0/15", "102/176", "15/27", "keine zweite Decoderinferenz",
                  "keine Produktionsfreigabe", "Unterschiedliche Rohscores: 1", "Normlisten über beide Shortlists: 324"):
        assert value in text
    assert "norm:TEST" not in text and inputs["cases"][0]["query"] not in text


def test_batches_preserve_same_original_pair_positions_padding_and_only_declared_retry():
    tokens = [[1] * n for n in (3, 5, 2, 4, 3)]
    batches = [{"start": 0, "count": 2, "padded_tokens": 5}, {"start": 2, "count": 2, "padded_tokens": 4},
               {"start": 4, "count": 1, "padded_tokens": 3}]
    checked_batches(tokens, batches, [])
    batches[1:] = [{"start": i, "count": 1, "padded_tokens": len(tokens[i])} for i in range(2, 5)]
    retry = {"phase": "decoder", "start": 2, "attempted_batch_size": 2, "retry_batch_size": 1}
    checked_batches(tokens, batches, [retry])
    with pytest.raises(ValueError, match="batch boundaries"):
        checked_batches(tokens, batches, [])
    batches[-1]["start"] = 3
    with pytest.raises(ValueError, match="batch boundaries"):
        checked_batches(tokens, batches, [retry])


def test_every_gpu_projection_is_checked_against_original_rows_and_actual_hidden_states():
    hidden = np.array([[1., 2.], [3., 4.], [5., 6.]], np.float16)
    rows = np.array([[.5, 1.], [1., .5]], np.float16)
    expected = hidden.astype(np.float32) @ rows.astype(np.float32).T
    logits = {name: expected.copy() for name in ("reduced", "full_gpu")}
    cpu = {"device": "cpu", "observed_pair_indices": [0, 1], "max_abs_fp16_difference": 0.,
           "original_head_dtype": "torch.float16", "reduced_head_dtype": "torch.float16",
           **{name: expected[:2].tolist() for name in
              ("fp32_row_reference_logits", "original_fp16_logits", "reduced_fp16_logits")}}
    assert checked_projections(hidden, rows, logits, cpu) == {"reduced": 0., "full_gpu": 0.}
    logits["full_gpu"][-1] += 1.
    with pytest.raises(ValueError, match="stored states"):
        checked_projections(hidden, rows, logits, cpu)
    logits["full_gpu"] = expected
    cpu["max_abs_fp16_difference"] = 1.
    with pytest.raises(ValueError, match="difference accounting"):
        checked_projections(hidden, rows, logits, cpu)


def test_cli_accepts_optional_8b_alone_with_baseline_and_binds_helper_dependency(tmp_path):
    from tools._gpu_eval_metrics import summarize
    _, _, _, _, baseline = complete_outputs()
    baseline.update(format="quality_precision_probe/1", status="complete", source_hashes={}, input_hashes={},
                    summaries={dataset: {"kaskade": summarize([r for r in baseline["rows"] if r["dataset"] == dataset], "kaskade")}
                               for dataset in audit.DATASETS})
    baseline_path = tmp_path / "baseline.json"
    audit.save(baseline_path, baseline)
    driver = tmp_path / "driver.py"
    driver.write_text("""import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools import quality_recall_report as report
from tools import _quality_recall_qwen8b as helper
def aggregate(path, baseline, directory):
    assert Path(path).name == '8b.json' and Path(baseline).name == 'baseline.json'
    assert directory.name == 'qwen8b-audit'
    return {'format': 'synthetic_public_aggregate/1', 'count': 648}
helper.aggregate = aggregate
raise SystemExit(report.main(sys.argv[2:]))
""", encoding="utf-8")
    result = subprocess.run([sys.executable, str(driver), str(helper.ROOT), "--baseline", str(baseline_path),
                             "--qwen8b", str(tmp_path / "8b.json"), "--out", str(tmp_path / "result")],
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    public = audit.read(tmp_path / "result/report.json")
    assert public["qwen8b"]["count"] == 648
    assert public["generator_dependencies"]["tools/_quality_recall_qwen8b.py"] == audit.digest(helper.__file__)
    assert set(public["variants"]) == {"baseline/kaskade"}
    audit.public_payload_check(public)


@pytest.mark.parametrize("option", ["--instruction-mean", "--qwen8b"])
def test_direct_script_cli_imports_optional_helpers_from_an_unrelated_working_directory(tmp_path, option):
    from tools._gpu_eval_metrics import summarize
    _, _, _, _, baseline = complete_outputs()
    baseline.update(format="quality_precision_probe/1", status="complete", source_hashes={}, input_hashes={},
                    summaries={dataset: {"kaskade": summarize([r for r in baseline["rows"] if r["dataset"] == dataset], "kaskade")}
                               for dataset in audit.DATASETS})
    baseline_path = tmp_path / "baseline.json"
    audit.save(baseline_path, baseline)
    missing = tmp_path / "missing-optional-report.json"
    args = [sys.executable, str(helper.ROOT / "tools/quality_recall_report.py"), "--baseline", str(baseline_path),
            "--out", str(tmp_path / "result"), option, str(missing)]
    # The 8B option permits baseline-only input and is visited after each other
    # selected helper, so the missing file proves that its lazy import succeeded.
    if option != "--qwen8b":
        args.extend(["--qwen8b", str(missing)])
    completed = subprocess.run(args, cwd=tmp_path, capture_output=True, check=False)
    error = completed.stderr.decode()
    assert completed.returncode == 2 and "missing-optional-report.json" in error
    assert "ModuleNotFoundError" not in error and "Traceback" not in error


def test_script_bootstrap_exposes_all_optional_helpers_without_cwd_or_pythonpath(tmp_path):
    modules = ["tools._quality_recall_mean", "tools._quality_recall_classifiers", "tools._quality_recall_qwen8b",
               "tools.quality_recall_live", "tools._gpu_eval_metrics"]
    driver = """import importlib, json, runpy, sys
runpy.run_path(sys.argv[1], run_name='reporter_bootstrap')
for name in sys.argv[2:]:
    importlib.import_module(name)
print(json.dumps({'imports': len(sys.argv[2:]), 'torch': 'torch' in sys.modules}))
"""
    completed = subprocess.run([sys.executable, "-c", driver, str(helper.ROOT / "tools/quality_recall_report.py"), *modules],
                               cwd=tmp_path, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr.decode()
    assert json.loads(completed.stdout) == {"imports": 5, "torch": False}


def test_model_manifest_cannot_omit_decoder_shards_while_retaining_the_full_output_head():
    index = {f"original_parameter_{i}": f"shard{i % 4}.safetensors" for i in range(398)}
    index["lm_head.weight"] = "head.safetensors"
    complete = dict.fromkeys(set(index.values()), {"sha256": "bound"})
    helper.checked_checkpoint_files(index, complete)
    with pytest.raises(ValueError, match="decoder or head"):
        helper.checked_checkpoint_files(index, {"head.safetensors": complete["head.safetensors"]})


@pytest.mark.parametrize("names", [
    {"tools/producer.py": "same", "tools\\producer.py": "same"},
    {"tools/Producer.py": "same", "tools/producer.py": "changed"},
])
def test_source_inventory_rejects_normalized_path_collisions(names):
    with pytest.raises(ValueError, match="collide"):
        helper.canonical_source_hashes(names)


@pytest.mark.parametrize("name", ["/tools/producer.py", "C:\\tools\\producer.py", "C:producer.py", "../producer.py",
                                  "tools\\..\\producer.py", "\\\\server\\producer.py", "tools//producer.py"])
def test_source_inventory_rejects_ambiguous_or_nonrelative_paths(name):
    with pytest.raises(ValueError, match="relative source"):
        helper.canonical_source_hashes({name: "bound"})


def test_source_inventory_accepts_mixed_windows_posix_separators_without_rewriting_input():
    original = {"tools\\quality_qwen8b_probe.py": "runner", "tools/_gpu_eval_metrics.py": "metrics",
                "risk_layer\\torch_compat.py": "shim"}
    copy = original.copy()
    assert helper.canonical_source_hashes(original) == {
        "tools/quality_qwen8b_probe.py": "runner", "tools/_gpu_eval_metrics.py": "metrics",
        "risk_layer/torch_compat.py": "shim"}
    assert original == copy


@pytest.fixture(params=["posix", "mixed_windows"])
def recovery_archive(tmp_path, monkeypatch, request):
    original, recovered = tmp_path / "original", tmp_path / "recovered"
    original.mkdir()
    recovered.mkdir()
    source = original / "source/tools/producer.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Original synthetic producer", encoding="utf-8")
    original_report = {key: {} for key in (
        "format", "inputs_sha256", "source_report_sha256", "bound_input_files", "model_manifest_sha256", "model_assets",
        "model", "instruction", "max_length", "threshold", "topk", "variants", "case_counts", "unique_pairs",
        "tokenizer", "architecture_audit", "cpu_head_audit")}
    original_report.update(status="failed", error=helper.RECOVERY_ERROR, rows=[],
        progress={"phase": "reduced_decoder", "completed_pairs": 14961, "planned_pairs": 14961},
        source_hashes={"tools\\producer.py" if request.param == "mixed_windows" else "tools/producer.py": audit.digest(source)},
        runtime={"packages": {"torch": "synthetic"}, "gpu": "synthetic", "hip": "fixed", "cuda": None,
                 "device_total_bytes": 16 * 1024**3, "post_decoder_allocated_bytes": 79691776, "dependency_sources": {},
                 "matmul_allow_tf32": False, "deterministic_algorithms": False})
    audit.save(original / "report.json", original_report)
    (original / "manifest.json").write_text("synthetic declaration", encoding="utf-8")
    original_hash = audit.digest(original / "report.json")
    monkeypatch.setattr(helper, "RECOVERY_ORIGINAL_SHA256", original_hash)
    monkeypatch.setattr(helper, "RECOVERY_ORIGINAL_MANIFEST_SHA256", audit.digest(original / "manifest.json"))
    for name in helper.INHERITED_ARTIFACTS:
        (original / name).write_text("original synthetic " + name, encoding="utf-8")
        if name in ("batches.json", "oom_retries.json"):
            audit.save(original / name, [])
        (recovered / name).write_bytes((original / name).read_bytes())
    for name in ("report.json", "manifest.json"):
        (recovered / ("original-" + name)).write_bytes((original / name).read_bytes())
    target_source = recovered / "source/tools/producer.py"
    target_source.parent.mkdir(parents=True)
    target_source.write_bytes(source.read_bytes())
    recovery_source = recovered / "recovery-source/tools/recover.py"
    recovery_source.parent.mkdir(parents=True)
    recovery_source.write_text("# Synthetic recovery", encoding="utf-8")
    report = deepcopy(original_report)
    report.update(status="complete", rows=["completed rows"], runtime={**original_report["runtime"], "full_head_validation_seconds": 1.})
    report.pop("error")
    report["artifact_hashes"] = {p.name: audit.digest(p) for p in recovered.iterdir() if p.is_file()}
    report["recovery"] = {"format": "quality_qwen8b_recovery/1", "source_directory": str(original),
        "original_report_sha256": original_hash, "original_manifest_sha256": audit.digest(original / "manifest.json"),
        "original_status": "failed", "original_error": helper.RECOVERY_ERROR, "original_post_decoder_allocated_bytes": 79691776,
        "fresh_process": True, "full_head_only": True, "decoder_rerun": False,
        "inherited_artifact_hashes": {name: report["artifact_hashes"][name] for name in helper.INHERITED_ARTIFACTS},
        "inherited_source_hashes": {"source/tools\\producer.py" if request.param == "mixed_windows" else "source/tools/producer.py": audit.digest(source)},
        "source_hashes": {"tools\\recover.py" if request.param == "mixed_windows" else "tools/recover.py": audit.digest(recovery_source)},
        "preflight": {"pairs": 14961, "batches": 0, "oom_retries": 0, "reduced_raw_scores_preserved": True,
            "original_bound_input_files": {}, "original_model_assets": 0, "exact_tokenized_pairs": 14961,
            "exact_original_head_rows": True, "torch_imported": False},
        "runtime": {**{key: original_report["runtime"][key] for key in (
            "packages", "gpu", "hip", "cuda", "device_total_bytes", "matmul_allow_tf32", "deterministic_algorithms")},
            "dtype": "torch.float16", "pid": 123, "initial_allocated_bytes": 0, "full_gpu_projected_pairs": 14961,
            "head_load_seconds": 1., "full_head_validation_seconds": 1., "peak_allocated_bytes": 1300000000,
            "peak_reserved_bytes": 1350000000}}
    audit.save(recovered / "recovery-manifest.json", {**report["recovery"], "runtime": {}})
    report["artifact_hashes"]["recovery-manifest.json"] = audit.digest(recovered / "recovery-manifest.json")
    def bind(path, expected):
        audit.require(audit.digest(path) == expected, "Synthetic bound file hash differs")
    return original, recovered, report, bind


def test_recovery_keeps_failed_original_and_proves_complete_inherited_coverage(recovery_archive):
    original, recovered, report, bind = recovery_archive
    result = helper.checked_recovery_origin(recovered, report, bind)
    assert result["original_status"] == "failed" and result["original_post_decoder_allocated_bytes"] == 79691776
    assert result["decoder_rerun"] is False and result["complete_inherited_decoder_pairs"] == 14961
    assert len(result["inherited_artifact_hashes"]) == 10
    assert audit.read(original / "report.json")["status"] == "failed"
    assert str(original) not in json.dumps(result)
    audit.public_payload_check(result)


@pytest.mark.parametrize("tamper", ["hidden", "score", "source", "missing_artifact", "runtime", "new_decoder", "other_gpu",
                                   "preexisting_gpu_tensor", "incomplete_head", "declaration"])
def test_recovery_cannot_hide_changed_scores_sources_failure_or_a_second_decoder(recovery_archive, tamper):
    original, recovered, report, bind = recovery_archive
    if tamper == "hidden":
        (recovered / "hidden_states.npy").write_text("changed states", encoding="utf-8")
    elif tamper == "score":
        (original / "reduced_scores.npy").write_text("changed scores", encoding="utf-8")
    elif tamper == "source":
        (original / "source/tools/producer.py").write_text("changed source", encoding="utf-8")
    elif tamper == "missing_artifact":
        report["recovery"]["inherited_artifact_hashes"].pop("batches.json")
    elif tamper == "runtime":
        report["runtime"]["post_decoder_allocated_bytes"] = 0
    elif tamper == "new_decoder":
        report["recovery"]["decoder_rerun"] = True
    elif tamper == "other_gpu":
        report["recovery"]["runtime"]["gpu"] = "different device"
    elif tamper == "preexisting_gpu_tensor":
        report["recovery"]["runtime"]["initial_allocated_bytes"] = 1024
    elif tamper == "incomplete_head":
        report["recovery"]["runtime"]["full_gpu_projected_pairs"] -= 1
    else:
        report["recovery"]["preflight"]["exact_tokenized_pairs"] -= 1
    with pytest.raises(ValueError):
        helper.checked_recovery_origin(recovered, report, bind)


def test_recovery_render_discloses_original_failure_and_separate_full_head_process():
    report, inputs, pairs, scores, baseline = complete_outputs()
    values, differences = checked_outputs(report, inputs, pairs, scores, baseline)
    public = {"variants": values, "output_difference_counts": differences,
              "numerical_audit": {"pairs": 14961, "different_scores": 1, "threshold_crossings": 1},
              "recovery": {"complete_inherited_decoder_pairs": 14961, "original_post_decoder_allocated_bytes": 79691776,
                           "original_release_guard_bytes": 67108864}}
    text = audit.render({"variants": {}, "qwen8b": public})
    for phrase in ("fehlgeschlagenen Ursprungslauf wiederhergestellt", "14961 Decoderpaaren", "79,691,776",
                   "67,108,864", "Der Decoder wurde nicht erneut ausgeführt", "keinen erfolgreichen ursprünglichen Gesamtlauf"):
        assert phrase in text
    assert all(name in text for name in helper.VARIANTS)


def test_new_release_guard_proves_object_lifecycle_without_rewriting_legacy_allocation_rule():
    runtime = {"post_decoder_allocated_bytes": 79691776}
    with pytest.raises(ValueError, match="Legacy"):
        helper.checked_decoder_release(runtime)
    runtime["decoder_lifecycle"] = {"guard": "original_decoder_object_weakrefs/1", "module_released": True,
        "parameter_objects_checked": 398, "buffer_objects_checked": 2, "live_parameter_objects": 0, "live_buffer_objects": 0}
    assert helper.checked_decoder_release(runtime)["module_released"] is True
    runtime["decoder_lifecycle"]["live_parameter_objects"] = 1
    with pytest.raises(ValueError, match="completely released"):
        helper.checked_decoder_release(runtime)
