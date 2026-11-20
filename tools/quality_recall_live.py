"""Verify one current public recall profile with actual uncached GPU Pipeline.run.

Only a completed 162-case structural experiment can supply the comparison. The
profile is selected by the operator, never by this script. No encoder/score cache,
candidate replacement, runtime monkeypatch, index activation or training occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import quality_recall_report as audit  # noqa: E402
from tools import quality_precision_live as precision  # noqa: E402

helper = precision.helper
CASE_FIELDS = audit.CASE_FIELDS


def validate_reference(report, variant):
    audit.require(report.get("format") == "quality_passage_rerank_probe/1"
                  and report.get("status") == "complete" and not report.get("diagnostic_subset")
                  and not report.get("error"), "Requires a completed full structural experiment")
    audit.cases(report["rows"])
    audit.require(variant in report["variants"], "Selected structural variant is absent")
    audit.require(report["variants"][variant].get("cascade") is True,
                  "This live validator requires the selected three-model cascade profile")
    for row in report["rows"]:
        audit.require(isinstance(row.get(variant, {}).get("suggestions"), list)
                      and not row[variant].get("error"), "Incomplete selected variant output")
    audit.validate_summaries(report, [variant])
    audit.require(set(report["models"]) == {"encoder", "bge", "qwen"}, "Missing pinned model identities")


def renamed_reference(suggestions, variant, profile):
    result = deepcopy(suggestions)
    for sentence in result:
        audit.require(sentence.get("profil") == variant, "Unexpected reference output profile")
        sentence["profil"] = profile
    return result


def validate_public_profile(profile, config, allowed, current):
    audit.require(profile in allowed and profile != "standard", "A current public semantic profile is required")
    audit.require(current.get(profile) == config, "Public profile configuration differs from selected experiment")


def compare(actual, expected, targets, evaluate):
    value = precision.comparison(actual, expected, targets, evaluate)
    def selection(suggestions):
        return {(s["start"], s["end"], n["id"]) for s in suggestions for n in s["normen"]}
    value["selection_equal"] = selection(actual["suggestions"]) == selection(expected["suggestions"])
    categories = []
    if not value["selection_equal"]:
        categories.append("selection")
    elif not value["norm_order_equal"]:
        categories.append("order")
    for field, label in (("context_equal", "context"), ("hard_marks_equal", "hard_marks"),
                         ("evidence_equal", "evidence")):
        if not value[field]:
            categories.append(label)
    if not value["scores_equal"] and not categories:
        # Other norm metadata could differ in addition to displayed scores.
        def without_scores(suggestions):
            projected = deepcopy(suggestions)
            for sentence in projected:
                for norm in sentence["normen"]:
                    norm.pop("sim", None)
                    norm.pop("rerank", None)
            return projected
        categories.append("score_only" if without_scores(actual["suggestions"]) == without_scores(expected["suggestions"])
                          else "scores_and_metadata")
    if not value["all_equal"] and not categories:
        categories.append("metadata_other")
    value["categories"] = categories
    return value


def freeze_sources(out):
    files = sorted((ROOT / "risk_layer").rglob("*.py")) + [ROOT / "catalog/begriffe.yaml",
        ROOT / "tools/_gpu_eval_metrics.py", ROOT / "tools/quality_live_check.py",
        ROOT / "tools/quality_precision_live.py", ROOT / "tools/quality_recall_report.py", Path(__file__)]
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative.as_posix()] = helper.digest(destination)
        audit.check_hash(source, hashes[relative.as_posix()])
    return hashes


def prepare(args, out, result):
    reference_path = args.reference.resolve()
    reference = helper.freeze_json(reference_path, out, "structural-reference.json", result)
    validate_reference(reference, args.variant)
    result["input_hashes"].update(audit.bind_report(reference_path, reference))
    result["reference_report_sha256"] = helper.digest(reference_path)
    original_path = reference_path.parent / "reference.json"
    audit.require(helper.digest(original_path) in reference["input_hashes"].values(), "Unbound original case/hard-mark reference")
    original = helper.freeze_json(original_path, out, "hard-mark-reference.json", result)
    audit.require(original.get("format") == "quality_production_check/1"
                  and original.get("status") in ("pass", "mismatch")
                  and not original.get("error") and not original.get("cache_misses"), "Invalid original hard-mark reference")
    originals = audit.cases(original["rows"])
    audit.same_cases(audit.cases(reference["rows"]), originals)
    rows = []
    for row in reference["rows"]:
        hard = originals[row["case_id"]]["standard"]
        audit.require(all(isinstance(hard.get(k), list) for k in ("cards", "trigger_marks")), "Missing original hard marks")
        rows.append({**{key: row[key] for key in CASE_FIELDS}, "expected": {
            "suggestions": renamed_reference(row[args.variant]["suggestions"], args.variant, args.profile),
            "cards": hard["cards"], "trigger_marks": hard["trigger_marks"]}})

    def input_path(name):
        paths = [Path(p) for p in reference["input_hashes"] if Path(p).name == name]
        audit.require(len(paths) == 1, f"Expected exactly one bound {name}")
        return paths[0]
    paths = {"index": input_path("meta.json").parent, "graph": input_path("graph.sqlite"),
             "trigger": input_path("trigger.sqlite")}
    audit.require(input_path("vectors.npy").parent == paths["index"], "Mixed index generation")
    audit.require(result["source_hashes"]["catalog/begriffe.yaml"] == reference["source_hashes"]["catalog/begriffe.yaml"],
                  "Current catalog differs from selected structural reference")
    result["profile_config"] = deepcopy(reference["variants"][args.variant])
    result["reference_models"] = reference["models"]
    result["case_counts"] = dict(Counter(row["dataset"] for row in rows))
    result["source_differences_from_reference"] = [name for name, value in result["source_hashes"].items()
        if name.startswith("risk_layer/") and reference["source_hashes"].get(name) != value]
    return rows, paths


def json_value(value):
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Unexpected raw RunReport value: {type(value).__name__}")


def execute(args, rows, paths, out, result):
    if any(name == "risk_layer" or name.startswith("risk_layer.") for name in sys.modules):
        raise RuntimeError("Runtime imported before current source snapshot")
    sys.path[:0] = [str(out / "source"), str(ROOT)]
    from risk_layer.api_v1 import _SEMANTIK_PROFILE
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import BgeM3Encoder, EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.pipeline import AnalyseOptionen, Pipeline
    from risk_layer.reranker import CascadedQwenReranker
    from risk_layer.semantic_search import PROFILE_CONFIG
    from risk_layer.triggerindex import TriggerIndex
    import risk_layer.pipeline as pipeline_module
    import tools._gpu_eval_metrics as metrics
    for module in (pipeline_module, metrics):
        audit.require(Path(module.__file__).resolve().is_relative_to(out / "source"), "Wrong frozen runtime import")
    validate_public_profile(args.profile, result["profile_config"], _SEMANTIK_PROFILE, PROFILE_CONFIG)
    import torch
    audit.require(torch.cuda.is_available(), "Real local GPU required")
    catalog = lade_katalog(out / "source/catalog/begriffe.yaml")
    index = EmbeddingIndex.load(paths["index"])
    graph, trigger = NormGraph(paths["graph"]), TriggerIndex(paths["trigger"])
    try:
        audit.require(index is not None and index.passage_evidence is not None
                      and pruefe_aktualitaet(index, graph, catalog), "Incoherent index/catalog/graph")
        encoder = BgeM3Encoder(device="cuda", local_files_only=True)
        cascade = CascadedQwenReranker(device="cuda")
        result["runtime"] = {"python": sys.version, "torch": torch.__version__, "hip": torch.version.hip,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "transformers", "sentence-transformers")}}
        result["models"] = {}
        for label, model in (("encoder", encoder), ("bge", cascade.vorauswahl), ("qwen", cascade)):
            identity = result["reference_models"][label]
            audit.require(model.name == identity["name"] and model.revision == identity["revision"], "Live model differs from reference pin")
            print(f"Loading {label}: {model.name} ({model.revision})", flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loaded = model._laden()
            torch.cuda.synchronize()
            info = helper.model_info(loaded.model if label == "bge" else loaded, model, time.perf_counter() - start)
            audit.require(info["dtypes"] == (["torch.float32"] if label == "encoder" else ["torch.float16"]), "Unexpected GPU model dtype")
            result["models"][label] = info
        audit.require(encoder.name == index.encoder_name and encoder.revision == index.encoder_revision, "Encoder/index identity differs")
        pipeline = Pipeline(catalog)
        def run(row):
            return pipeline.run(row["query"], optionen=AnalyseOptionen(erzeugt_am="quality-recall-live",
                detektor=True, streit=True, semantik_profil=args.profile), graph=graph, trigger=trigger,
                embedding=index, embedding_encoder=encoder, reranker=cascade)
        torch.cuda.synchronize()
        start = time.perf_counter()
        run(rows[0])
        torch.cuda.synchronize()
        result["warmup"] = {"case_id": rows[0]["case_id"], "seconds": time.perf_counter() - start,
                            "excluded_from_case_timings": True}
        torch.cuda.reset_peak_memory_stats()
        for number, reference in enumerate(rows, 1):
            print(f"Live {number}/{len(rows)} {reference['case_id']} {args.profile}", flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            raw = run(reference)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            actual = {"suggestions": raw.semantik_vorschlaege, **helper.compact_marks(raw), "elapsed_seconds": elapsed}
            actual["metrics"] = metrics.evaluate_variant(actual["suggestions"], reference["targets"])
            actual["comparison"] = compare(actual, reference["expected"], reference["targets"], metrics.evaluate_variant)
            try:
                spans = audit.source_evidence_audit(actual["suggestions"], index, graph)
                actual["source_audit"] = {"status": "pass", "exact_passage_spans": spans}
                result["source_audit"]["exact_passage_spans"] += spans
            except (ValueError, KeyError, TypeError) as exc:
                actual["source_audit"] = {"status": "failed", "error": str(exc)}
                result["source_audit"]["errors"].append({"case_id": reference["case_id"], "error": str(exc)})
            row = {key: reference[key] for key in CASE_FIELDS}
            row[args.profile] = actual
            row["raw_pipeline_report"] = json.loads(json.dumps(asdict(raw), default=json_value, ensure_ascii=False))
            result["rows"].append(row)
            if number == 1:
                from risk_layer.report import report_html, report_text
                # Render the actual measured result after timing and preserve
                # its identity; this preview is never an additional test case.
                previews = {"preview.html": report_html(raw, row["query"]), "preview.txt": report_text(raw)}
                result["preview"] = {"case_id": row["case_id"], "profile": args.profile, "files": {}}
                for filename, rendered in previews.items():
                    (out / filename).write_text(rendered, encoding="utf-8")
                    result["preview"]["files"][filename] = helper.digest(out / filename)
            if not actual["comparison"]["all_equal"]:
                result["mismatches"].append({"case_id": row["case_id"], "dataset": row["dataset"],
                    "categories": actual["comparison"]["categories"], "comparison": actual["comparison"]})
            with (out / "rows.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            helper.save(out / "report.json", result)
        result["summaries"] = {dataset: {args.profile: metrics.summarize([
            row for row in result["rows"] if row["dataset"] == dataset], args.profile)} for dataset in ("main", "additional")}
        result["reference_summaries"] = {dataset: {args.variant: metrics.summarize([
            {**row, args.variant: row["expected"]} for row in rows if row["dataset"] == dataset], args.variant)}
            for dataset in ("main", "additional")}
        result["timings"] = {dataset: helper.timing_summary([row[args.profile]["elapsed_seconds"]
            for row in result["rows"] if dataset == "all" or row["dataset"] == dataset]) for dataset in ("all", "main", "additional")}
        result["difference_categories"] = dict(Counter(category for difference in result["mismatches"] for category in difference["categories"]))
        result["runtime"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["runtime"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    finally:
        graph.close()
        trigger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Choose a fresh output directory")
    out = args.out.resolve()
    out.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus", RISK_LAYER_FUSION="an", RISK_LAYER_EMB_DEVICE="cuda",
        RISK_LAYER_RERANK_SCHWELLE="0.10", RISK_LAYER_SIM_RETTUNG="0.55", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = {"format": "quality_recall_live/1", "status": "running", "selected_variant": args.variant,
        "public_profile": args.profile, "rows": [], "mismatches": [], "source_hashes": {},
        "input_hashes": {}, "frozen_inputs": {}, "source_audit": {"exact_passage_spans": 0, "errors": []},
        "notes": ["Real current public Pipeline.run with three pinned local GPU models; no encoder/score cache or monkeypatch.",
                  "Only the profile name in reference suggestions is renamed; all other output differences remain visible.",
                  "Original hard markings are checked against the R2-bound production reference.",
                  "Model load and first-case warmup are separate; only this run supplies synchronized live timings.",
                  "All 162 historical cases are previously inspected development data, not independent blind gold.",
                  "Metrics are recomputed from actual outputs even when comparison fails; unknown norm pairs are not negative labels."]}
    try:
        result["source_hashes"] = freeze_sources(out)
        rows, paths = prepare(args, out, result)
        helper.save(out / "manifest.json", {k: v for k, v in result.items() if k not in ("rows", "mismatches")})
        execute(args, rows, paths, out, result)
        for path, expected in result["input_hashes"].items():
            audit.check_hash(path, expected)
        for relative, expected in result["source_hashes"].items():
            audit.check_hash(out / "source" / relative, expected)
        for relative, expected in result["frozen_inputs"].items():
            audit.check_hash(out / relative, expected)
        result["status"] = ("failed" if result["source_audit"]["errors"] else "mismatch" if result["mismatches"] else "pass")
        code = 2 if result["status"] == "failed" else 1 if result["status"] == "mismatch" else 0
    except Exception as exc:
        result["status"], result["error"] = "failed", {"type": type(exc).__name__, "message": str(exc)}
        code = 2
    result["torch_imported"] = "torch" in sys.modules
    if (out / "rows.jsonl").exists():
        result["rows_sha256"] = helper.digest(out / "rows.jsonl")
    helper.save(out / "report.json", result)
    print(f"{result['status']}: {len(result['rows'])}/162 cases, {len(result['mismatches'])} differences; {out}", flush=True)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
