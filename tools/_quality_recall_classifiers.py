"""Aggregate audits of fixed anchor-OOF and nested source-OOF diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from tools import quality_recall_report as audit


def _read(path, expected=None):
    data = Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    audit.require(expected is None or sha == expected, "Classifier artifact hash differs")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            audit.require(key not in value, "Duplicate classifier JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError(f"Nonfinite classifier JSON value: {value}")
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid), sha


def _bind_inputs(protocol, directory):
    """Producer/metric snapshots remain valid when the current tools advance."""
    archived = {audit.digest(directory / name) for name in ("source.py", "metrics.py") if (directory / name).exists()}
    replaced = 0
    for filename, expected in protocol["input_hashes"].items():
        path = Path(filename)
        if path.exists() and audit.digest(path) == expected:
            continue
        audit.require(expected in archived, "Classifier bound data input changed without exact source snapshot")
        replaced += 1
    for name, expected in protocol.get("artifact_hashes", {}).items():
        audit.check_hash(directory / name, expected)
    return {"bound_input_files": len(protocol["input_hashes"]), "current_tools_replaced_by_exact_frozen_source": replaced}


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _load(path, baseline_path, base_cases, expected_format):
    path = Path(path).resolve()
    report, sha = _read(path)
    audit.require(report.get("format") == expected_format and report.get("status") == "complete"
                  and report.get("model_saved") is False and report.get("runtime_activated") is False
                  and report.get("torch_imported") is False, "Expected complete undeployed CPU classifier report")
    protocol, _ = _read(path.parent / "protocol.json", report["protocol_sha256"])
    audit.require(protocol["status"] == "frozen_before_fit" and protocol["source_sha256"] == report["source_sha256"],
                  "Classifier source/protocol mismatch")
    audit.check_hash(path.parent / "source.py", report["source_sha256"])
    checks = _bind_inputs(protocol, path.parent)
    audit.require(protocol["input_hashes"].get(str(Path(baseline_path).resolve())) == audit.digest(baseline_path),
                  "Classifier uses another baseline")
    ids, rows = protocol["case_order"], report["rows"]
    audit.require(len(ids) == len(set(ids)) == len(rows) == len(base_cases) == 162
                  and set(ids) == set(base_cases) and [row["case_id"] for row in rows] == ids,
                  "Classifier case coverage differs")
    for row in rows:
        audit.require(all(row[k] == base_cases[row["case_id"]][k] for k in ("negative", "dataset"))
                      and type(row["predicted_need"]) is bool, "Classifier labels/decision type differ")
    folds = protocol["folds"]
    audit.require(len(folds) == 11 and sorted(i for f in folds for i in f["test"]) == list(range(162)),
                  "Classifier outer coverage differs")
    for fold in folds:
        expected_test = [i for i, row in enumerate(rows) if row["held_out_source"] == fold["held_out_source"]]
        audit.require(fold["test"] == expected_test and not set(fold["train"]) & set(expected_test)
                      and fold["binary_fit_valid"], "Classifier source split differs")
    return path, report, protocol, {"report_sha256": sha, "protocol_sha256": report["protocol_sha256"],
        "source_sha256": report["source_sha256"], "audit": checks, "source_folds": 11, "cases": 162,
        "model_saved": False, "runtime_activated": False, "decision": "rejected_for_runtime"}


def _datasets(rows, variant, decisions, *, include_all=False):
    gated = [{**row, "gated": {"suggestions": row[variant]["suggestions"] if decisions[row["case_id"]] else []}} for row in rows]
    return {ds: {"before": audit.variant_metrics([r for r in gated if ds == "all" or r["dataset"] == ds], variant),
                 "after": audit.variant_metrics([r for r in gated if ds == "all" or r["dataset"] == ds], "gated")}
            for ds in (("all", *audit.DATASETS) if include_all else audit.DATASETS)}


def _checked_dataset_metrics(actual, stored, message):
    """Old archives omit only the two later additive norm-target counters."""
    added = {"historical_norm_targets", "matched_historical_norm_targets"}
    audit.require(set(stored) == set(actual), message)
    for dataset, values in actual.items():
        audit.require(set(stored[dataset]) == set(values), message)
        for side, metrics in values.items():
            saved = stored[dataset][side]
            audit.require(set(saved) in (set(metrics), set(metrics) - added)
                          and saved == {key: metrics[key] for key in saved}, message)


def _purges(protocol, original, anchor_protocol, anchors, rows, base_cases):
    ids = protocol["case_order"]
    groups = {row["case_id"]: row["held_out_source"] for row in rows}
    edges = {frozenset((ids.index(e["a"]), ids.index(e["b"]))) for e in original["purge_edges"]}
    anchor_edges = {(e["anchor_index"], ids.index(e["case_id"])) for e in anchor_protocol["anchor_purge_edges"]}
    def normalized(text):
        return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
    def check(train, test, pool, anchor_train, anchor_test):
        expected = [i for i in pool if i not in test and not any(frozenset((i, j)) in edges for j in test)]
        audit.require(train == expected, "Declared source/near purge differs")
        texts = {normalized(base_cases[ids[i]]["query"]) for i in train}
        kept = [a for a in range(len(anchors)) if not any((a, i) in anchor_edges for i in anchor_test)
                and normalized(anchors[a]["text"]) not in texts]
        audit.require(anchor_train == kept, "Declared anchor purge/dedup differs")
    for fold in protocol["folds"]:
        check(fold["train"], fold["test"], list(range(len(ids))), fold["anchor_train"], fold["test"])
        for inner in fold.get("inner_folds", []):
            expected_test = [i for i in fold["train"] if groups[ids[i]] == inner["held_out_source"]]
            audit.require(inner["test"] == expected_test, "Inner original source holdout differs")
            check(inner["train"], inner["test"], fold["train"], inner["anchor_train"], fold["test"] + inner["test"])


def aggregate_anchors(path, baseline_path, base_cases):
    path, report, protocol, result = _load(path, baseline_path, base_cases, "query_classifier_anchor_logo/1")
    model = protocol["model"]
    audit.require(model["C"] == 1 and model["class_weight"] == "balanced" and model["threshold"] == .5
                  and report["model_fits"] == 11 and not protocol["label_audit"]["contradictions"], "Anchor estimator/supervision differs")
    original_paths = [name for name, value in protocol["input_hashes"].items() if value == protocol["previous_protocol_sha256"]]
    audit.require(len(original_paths) == 1, "Missing original purge protocol")
    original, _ = _read(original_paths[0], protocol["previous_protocol_sha256"])
    anchors, _ = _read(path.parent / "anchors.json", protocol["artifact_hashes"]["anchors.json"])
    _purges(protocol, original, protocol, anchors, report["rows"], base_cases)
    for row in report["rows"]:
        audit.require(_score(row["probability_need"]) and row["predicted_need"] == (row["probability_need"] >= .5),
                      "Anchor fixed decision differs")
    anchor_edges = {(edge["anchor_index"], protocol["case_order"].index(edge["case_id"])) for edge in protocol["anchor_purge_edges"]}
    for fold in protocol["folds"]:
        audit.require(not set(fold["anchor_train"]) & set(fold["anchor_purged"] + fold["anchor_train_duplicates_omitted"])
                      and not any((a, i) in anchor_edges for a in fold["anchor_train"] for i in fold["test"]),
                      "Anchor/test purge leakage")
    decisions = {row["case_id"]: row["predicted_need"] for row in report["rows"]}
    datasets = _datasets(list(base_cases.values()), "kaskade", decisions)
    for ds, values in datasets.items():
        for key, stored_key in (("before", "kaskade"), ("after", "anchor_augmented_oof_gate")):
            stored = report["summaries"][ds][stored_key]
            actual = values[key]
            audit.require(math.isclose(actual["hit5_cases"] / actual["positive_cases"], stored["positive"]["hit@5"])
                          and math.isclose(actual["negative_with_suggestion"] / actual["negative_cases"],
                                           stored["negative"]["false_suggestion_rate"]), "Anchor metric mismatch")
    result.update(format="query_classifier_anchor_aggregate/1", model=model, datasets=datasets,
        anchors={"original_rows": protocol["original_anchor_rows"], "unique_training_texts": protocol["deduplicated_anchor_texts"],
                 "retained_per_fold": [len(f["anchor_train"]) for f in protocol["folds"]]},
        notes=["Existing curated catalog descriptions supply weak positive matching supervision for fitting only, not new evaluation cases.",
               "The fixed 0.5 gate preserves main hits but removes two additional positive cases; it remains undeployed.",
               "All 162 cases are known development data; whole-source and declared similarity purges precede every fit."])
    instruction_path = path.parent / "instruction-gates/report.json"
    if instruction_path.exists():
        instruction, instruction_hash = _read(instruction_path)
        instruction_protocol, _ = _read(instruction_path.parent / "protocol.json", instruction["protocol_sha256"])
        audit.require(instruction["status"] == "complete" and instruction["model_fits"] == 0
                      and instruction["gate_report_sha256"] == result["report_sha256"], "Instruction gate identity differs")
        references = {}
        for filename, expected in instruction_protocol["input_hashes"].items():
            candidate = Path(filename)
            if candidate.name != "report.json" or "quality-instruction-" not in candidate.parent.name:
                continue
            reference, _ = _read(candidate, expected)
            audit.same_cases(base_cases, audit.cases(reference["rows"]))
            for name in reference["instructions"]:
                references[name] = reference
        audit.require(set(references) == set(instruction["variants"]), "Instruction gate variants differ")
        result["instruction_report_sha256"] = instruction_hash
        result["instruction_variants"] = {}
        for name, value in instruction["variants"].items():
            current = _datasets(references[name]["rows"], name, decisions, include_all=True)
            _checked_dataset_metrics(current, value["datasets"], "Instruction gate metrics differ")
            result["instruction_variants"][name] = {"datasets": {ds: current[ds] for ds in audit.DATASETS}}
    audit.public_payload_check(result)
    return result


def _nested_thresholds(protocol, report, base_cases):
    ids = protocol["case_order"]
    audit.require(len(report["folds"]) == len(protocol["folds"]), "Missing nested fold results")
    inner_fits, fallbacks = 0, 0
    for fold, fitted in zip(protocol["folds"], report["folds"], strict=True):
        train, test = fold["train"], fold["test"]
        audit.require(fitted["held_out_source"] == fold["held_out_source"], "Nested fold result order differs")
        inner = fold["inner_folds"]
        audit.require(sorted(i for f in inner for i in f["test"]) == sorted(train), "Nested calibration denominator differs")
        for f in inner:
            audit.require(not set(f["train"]) & set(test + f["test"])
                          and set(f["train"] + f["test"]) <= set(train)
                          and f["train_local"] == [train.index(i) for i in f["train"]]
                          and f["test_local"] == [train.index(i) for i in f["test"]], "Nested index split leakage")
            positive = sum(not base_cases[ids[i]]["negative"] for i in f["train"]) + len(f["anchor_train"])
            negative = sum(base_cases[ids[i]]["negative"] for i in f["train"])
            audit.require(f["binary_fit_valid"] == bool(positive and negative), "Nested class coverage differs")
        fallback = any(not f["binary_fit_valid"] for f in inner)
        if fallback:
            audit.require(fitted["threshold"] == 0 and fitted["model_fits"] == 0
                          and fitted["fallback"] == "inner_training_fold_missing_class", "Nested conservative fallback differs")
            fallbacks += 1
        else:
            values = fitted["predictions"]
            audit.require(len(values) == len(train) and all(_score(v) for v in values), "Missing inner OOF values")
            positives = [v for i, v in zip(train, values, strict=True) if not base_cases[ids[i]]["negative"]]
            audit.require(positives and fitted["threshold"] == min(.5, min(positives))
                          and fitted["model_fits"] == len(inner) and fitted["fallback"] is None,
                          "Nested cutoff differs from original inner positives")
            inner_fits += len(inner)
        for i in test:
            row = report["rows"][i]
            audit.require(_score(row["value_need"]) and row["threshold"] == fitted["threshold"]
                          and row["predicted_need"] == (row["value_need"] >= row["threshold"]), "Nested outer decision differs")
    audit.require(report["model_fits"] == {"inner": inner_fits, "outer": len(protocol["folds"])}, "Nested fit accounting differs")
    return fallbacks


def aggregate_nested(path, baseline_path, base_cases):
    path, report, protocol, result = _load(path, baseline_path, base_cases, "query_classifier_nested_logo/1")
    model = protocol["base_estimator"]
    audit.require(model["C"] == 1 and model["class_weight"] == "balanced", "Nested estimator differs")
    original, original_hash = _read(path.parent / "inputs/original-purge-protocol.json")
    anchor_protocol, anchor_hash = _read(path.parent / "inputs/fixed-anchor-protocol.json")
    anchors, anchors_hash = _read(path.parent / "inputs/anchors.json")
    audit.require(all(value in protocol["input_hashes"].values() for value in (original_hash, anchor_hash, anchors_hash)),
                  "Unbound nested purge/anchor data")
    _purges(protocol, original, anchor_protocol, anchors, report["rows"], base_cases)
    fallbacks = _nested_thresholds(protocol, report, base_cases)
    reference_path = path.parent / "inputs/instruction-reference.json"
    reference, reference_hash = _read(reference_path)
    audit.require(reference_hash in protocol["input_hashes"].values(), "Unbound instruction reference")
    audit.same_cases(base_cases, audit.cases(reference["rows"]))
    decisions = {row["case_id"]: row["predicted_need"] for row in report["rows"]}
    variants = {}
    for rows, variant in ((list(base_cases.values()), "kaskade"), (reference["rows"], "english_match")):
        datasets = _datasets(rows, variant, decisions, include_all=True)
        _checked_dataset_metrics(datasets, report["variants"][variant]["datasets"], "Nested gate metric mismatch")
        variants[variant] = {"datasets": {ds: datasets[ds] for ds in audit.DATASETS}}
    result.update(format="query_classifier_nested_aggregate/1", model=model, model_fits=report["model_fits"],
        fallback_outer_folds=fallbacks, variants=variants,
        rule="Within each outer training split, use the largest cutoff at most 0.5 retaining all inner source-OOF original positive cases. Catalog anchors enter fitting only. A degenerate inner training class forces outer cutoff zero.",
        notes=["Thresholds are derived separately inside outer training folds, never from global OOF scores or outer test labels.",
               "Two negative source collections cause two conservative fallback folds; all cases are retained and no false suggestions are removed.",
               "The 162 cases remain previously inspected development data; no model file or runtime activation."])
    audit.public_payload_check(result)
    return result
