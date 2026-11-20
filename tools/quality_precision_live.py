"""Validate current public precision profiles through real GPU Pipeline.run.

The reference is explicitly quality_precision_current/1, bound to its complete
R2 experiment and CPU analysis. No score cache, model replacement, candidate
patch or index activation is used. GPU batch differences remain visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("praezision", "dokument")
CASE_FIELDS = ("case_id", "group", "query", "targets", "negative", "category", "uncertain", "dataset")
_spec = importlib.util.spec_from_file_location("precision_live_helpers", ROOT / "tools/quality_live_check.py")
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


def validate_reference(report):
    if (report.get("format") != "quality_precision_current/1" or report.get("status") != "pass"
            or report.get("torch_imported") is not False or report.get("fatal_error")
            or any(report.get(key) for key in ("mismatches", "hard_mark_mismatches", "cache_misses", "errors"))):
        raise ValueError("Requires a successful model-free quality_precision_current/1 reference")
    if (not isinstance(report.get("profiles"), dict)
            or any(profile not in report["profiles"] for profile in PROFILES)
            or not isinstance(report.get("reference_mapping"), dict)
            or any(not isinstance(report["reference_mapping"].get(profile), str) for profile in PROFILES)):
        raise ValueError("Current reference does not describe the required public profiles")
    if (not isinstance(report.get("source_hashes"), dict) or not report["source_hashes"]
            or not isinstance(report.get("source_run"), str)
            or any(not isinstance(report.get(key), str) or len(report[key]) != 64
                   or any(char not in "0123456789abcdef" for char in report[key])
                   for key in ("source_report_sha256", "analysis_report_sha256"))
            or any(report.get(key) != [] for key in ("mismatches", "hard_mark_mismatches", "cache_misses", "errors"))):
        raise ValueError("Current reference lacks complete successful provenance")
    rows = report.get("rows", [])
    if (Counter(row["dataset"] for row in rows) != Counter({"main": 132, "additional": 30})
            or len({row["case_id"] for row in rows}) != 162):
        raise ValueError("Current reference needs all 162 unique historical cases")
    for row in rows:
        if (not isinstance(row.get("query"), str) or type(row.get("negative")) is not bool
                or not isinstance(row.get("targets"), list) or (not row["negative"] and not row["targets"])):
            raise ValueError("Invalid current-reference case")
        for profile in PROFILES:
            value = row.get(profile, {})
            if ("error" in value or not all(isinstance(value.get(key), list) for key in (
                    "suggestions", "cards", "trigger_marks"))
                    or value.get("comparison", {}).get("all_equal") is not True):
                raise ValueError("Incomplete or mismatching current-reference output")


def comparison(actual, expected, targets, evaluate):
    def projection(suggestions, field):
        return {(s["start"], s["end"], n["id"]): (
            (n.get("sim"), n.get("rerank")) if field == "scores" else n.get("evidenz"))
            for s in suggestions for n in s["normen"]}
    actual_s, expected_s = actual["suggestions"], expected["suggestions"]
    actual_scores, expected_scores = (projection(s, "scores") for s in (actual_s, expected_s))
    score_differences = [{"start": key[0], "end": key[1], "norm_id": key[2],
                          "actual": list(actual_scores[key]), "expected": list(expected_scores[key])}
                         for key in sorted(actual_scores.keys() & expected_scores.keys())
                         if actual_scores[key] != expected_scores[key]]
    def contexts(suggestions):
        return [{"start": s["start"], "end": s["end"], "satz": s["satz"], "kontext": s.get("kontext")}
                for s in suggestions]
    actual_metrics, expected_metrics = (evaluate(s, targets) for s in (actual_s, expected_s))
    marks_equal = all(actual[key] == expected[key] for key in ("cards", "trigger_marks"))
    return {
        "all_equal": actual_s == expected_s and marks_equal,
        "norm_order_equal": helper.norm_order(actual_s) == helper.norm_order(expected_s),
        "scores_equal": actual_scores == expected_scores,
        "evidence_equal": projection(actual_s, "evidence") == projection(expected_s, "evidence"),
        "context_equal": contexts(actual_s) == contexts(expected_s),
        "hard_marks_equal": marks_equal, "metrics_equal": actual_metrics == expected_metrics,
        "actual_metrics": actual_metrics, "expected_metrics": expected_metrics,
        "score_display_differences": score_differences,
    }


def freeze_sources(out):
    paths = sorted((ROOT / "risk_layer").rglob("*.py")) + [
        ROOT / "catalog/begriffe.yaml", ROOT / "tools/_gpu_eval_metrics.py",
        ROOT / "tools/quality_live_check.py", Path(__file__)]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = helper.digest(target)
        if helper.digest(path) != hashes[relative.as_posix()]:
            raise ValueError("Source changed while freezing live validation")
    return hashes


def verify_hashes(claims, bound_hashes):
    for path, expected in claims.items():
        resolved = str(Path(path).resolve())
        if resolved in bound_hashes and bound_hashes[resolved] != expected:
            raise ValueError(f"Conflicting bound input hashes: {resolved}")
        if helper.digest(path) != expected:
            raise ValueError(f"Bound R2/analysis input changed: {path}")
        bound_hashes[resolved] = expected


def prepare(args, out, result):
    current = helper.freeze_json(args.reference, out, "current-reference.json", result)
    validate_reference(current)
    result["reference_report_sha256"] = result["input_hashes"][str(args.reference.resolve())]
    run = Path(current["source_run"])
    r2 = helper.freeze_json(run / "report.json", out, "r2-reference.json", result)
    analysis = helper.freeze_json(args.analysis, out, "analysis-reference.json", result)
    result["source_report_sha256"] = result["input_hashes"][str((run / "report.json").resolve())]
    result["analysis_report_sha256"] = result["input_hashes"][str(args.analysis.resolve())]
    if (current["source_report_sha256"] != result["source_report_sha256"]
            or current["analysis_report_sha256"] != result["analysis_report_sha256"]
            or r2.get("format") != "quality_precision_probe/1" or r2.get("status") != "complete"
            or r2.get("diagnostic_subset") or analysis.get("status") != "pass"
            or analysis.get("torch_imported") is not False
            or analysis["source_report_sha256"] != result["source_report_sha256"]):
        raise ValueError("Current reference is not bound to the selected complete R2/CPU analysis")
    for name, expected in current["source_hashes"].items():
        path = args.reference.parent / "source" / name
        if helper.digest(path) != expected:
            raise ValueError(f"Current-reference source archive changed: {path}")
        result["input_hashes"][str(path.resolve())] = expected
        if name.startswith(("risk_layer/", "catalog/")) or name == "tools/_gpu_eval_metrics.py":
            if result["source_hashes"].get(name) != expected:
                raise ValueError(f"Current runtime source differs from CPU reference: {name}")
    verify_hashes(analysis["verified_hashes"], result["input_hashes"])
    verify_hashes(r2["input_hashes"], result["input_hashes"])
    by_id = {row["case_id"]: row for row in r2["rows"]}
    if len(by_id) != 162 or len(r2["rows"]) != 162:
        raise ValueError("R2 source report is incomplete")
    for row in current["rows"]:
        if any(row.get(field) != by_id.get(row["case_id"], {}).get(field) for field in CASE_FIELDS):
            raise ValueError("Case text, labels or dataset differs between current and R2 reference")
    def input_path(name):
        paths = [Path(path) for path in r2["input_hashes"] if Path(path).name == name]
        if len(paths) != 1:
            raise ValueError(f"R2 report must bind exactly one {name}")
        return paths[0]
    paths = {"index": input_path("meta.json").parent, "graph": input_path("graph.sqlite"),
             "trigger": input_path("trigger.sqlite")}
    if input_path("vectors.npy").parent != paths["index"]:
        raise ValueError("Index metadata and vectors do not share a generation")
    rows = current["rows"] if args.case_limit is None else current["rows"][:args.case_limit]
    result["case_counts"] = dict(Counter(row["dataset"] for row in current["rows"]))
    result["selected_case_counts"] = dict(Counter(row["dataset"] for row in rows))
    result["diagnostic_subset"] = len(rows) != 162
    result["omitted_case_ids"] = [row["case_id"] for row in current["rows"][len(rows):]]
    result["profile_config"] = {profile: current["profiles"][profile] for profile in PROFILES}
    return rows, paths, r2["models"]


def execute(rows, paths, identities, out, result):
    if any(name == "risk_layer" or name.startswith("risk_layer.") for name in sys.modules):
        raise RuntimeError("Production package was imported before its snapshot")
    sys.path[:0] = [str(out / "source"), str(ROOT)]
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
        if not Path(module.__file__).resolve().is_relative_to(out / "source"):
            raise RuntimeError(f"Module was not imported from frozen sources: {module.__file__}")
    if any(PROFILE_CONFIG[p] != result["profile_config"][p] for p in PROFILES):
        raise ValueError("Public profile configurations differ from CPU reference")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A real local GPU is required")
    catalog = lade_katalog(out / "source/catalog/begriffe.yaml")
    index = EmbeddingIndex.load(paths["index"])
    graph, trigger = NormGraph(paths["graph"]), TriggerIndex(paths["trigger"])
    try:
        if index is None or index.passage_evidence is None or not pruefe_aktualitaet(index, graph, catalog):
            raise ValueError("Incompatible passage index, catalog or graph")
        encoder = BgeM3Encoder(device="cuda", local_files_only=True)
        cascade = CascadedQwenReranker(device="cuda")
        result["runtime"] = {
            "python": sys.version, "torch": torch.__version__, "hip": torch.version.hip,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "packages": {name: importlib.metadata.version(name) for name in (
                "numpy", "transformers", "sentence-transformers")},
            "bge_batch_size": 8, "cascade_qwen_batch_size": 2,
        }
        result["models"] = {}
        for label, model in (("encoder", encoder), ("bge", cascade.vorauswahl), ("qwen", cascade)):
            if model.name != identities[label]["name"] or model.revision != identities[label]["revision"]:
                raise ValueError(f"Live {label} model differs from pinned R2 identity")
            print(f"Loading {label}: {model.name} ({model.revision})", flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loaded = model._laden()
            torch.cuda.synchronize()
            info = helper.model_info(loaded.model if label == "bge" else loaded, model,
                                     time.perf_counter() - start)
            if info["dtypes"] != (["torch.float32"] if label == "encoder" else ["torch.float16"]):
                raise ValueError(f"Unexpected live {label} dtype: {info['dtypes']}")
            result["models"][label] = info
        if encoder.name != index.encoder_name or encoder.revision != index.encoder_revision:
            raise ValueError("Loaded encoder differs from saved index")
        pipe = Pipeline(catalog)
        def run(case, profile):
            return pipe.run(case["query"], optionen=AnalyseOptionen(
                erzeugt_am="quality-precision-live", detektor=True, streit=True, semantik_profil=profile),
                graph=graph, trigger=trigger, embedding=index, embedding_encoder=encoder, reranker=cascade)
        result["warmup"] = []
        for profile in PROFILES:
            torch.cuda.synchronize()
            start = time.perf_counter()
            run(rows[0], profile)
            torch.cuda.synchronize()
            result["warmup"].append({"profile": profile, "case_id": rows[0]["case_id"],
                                     "seconds": time.perf_counter() - start})
        torch.cuda.reset_peak_memory_stats()
        for number, reference in enumerate(rows, 1):
            row = {field: reference[field] for field in CASE_FIELDS}
            for profile in PROFILES:
                print(f"Live {number}/{len(rows)} {row['dataset']} {row['case_id']} {profile}", flush=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                report = run(row, profile)
                torch.cuda.synchronize()
                actual = {"suggestions": report.semantik_vorschlaege, **helper.compact_marks(report),
                          "elapsed_seconds": time.perf_counter() - start}
                actual["metrics"] = metrics.evaluate_variant(actual["suggestions"], row["targets"])
                actual["comparison"] = comparison(actual, reference[profile], row["targets"], metrics.evaluate_variant)
                row[profile] = actual
                if not actual["comparison"]["all_equal"]:
                    result["mismatches"].append({"case_id": row["case_id"], "dataset": row["dataset"],
                        "profile": profile, "comparison": actual["comparison"],
                        "actual": helper.norm_order(actual["suggestions"]),
                        "expected": helper.norm_order(reference[profile]["suggestions"])})
            result["rows"].append(row)
            with (out / "rows.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            helper.save(out / "report.json", result)
        result["summaries"] = {dataset: {profile: metrics.summarize([
            row for row in result["rows"] if row["dataset"] == dataset], profile) for profile in PROFILES}
            for dataset in ("main", "additional")}
        result["reference_summaries"] = {dataset: {profile: metrics.summarize([
            row for row in rows if row["dataset"] == dataset], profile) for profile in PROFILES}
            for dataset in ("main", "additional")}
        result["timings"] = {dataset: {profile: helper.timing_summary([
            row[profile]["elapsed_seconds"] for row in result["rows"] if dataset == "all" or row["dataset"] == dataset])
            for profile in PROFILES} for dataset in ("all", "main", "additional")}
        result["runtime"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["runtime"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    finally:
        graph.close()
        trigger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "out/quality-precision-current/report.json")
    parser.add_argument("--analysis", type=Path, default=ROOT / "out/quality-precision-analysis/r2/report.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-limit", type=int, help="Diagnostic first N cases only")
    args = parser.parse_args(argv)
    if args.out.exists() or (args.case_limit is not None and args.case_limit < 1):
        parser.error("Choose a fresh output directory and positive case-limit")
    out = args.out.resolve()
    out.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus", RISK_LAYER_FUSION="an",
                      RISK_LAYER_RERANK_SCHWELLE="0.10", RISK_LAYER_SIM_RETTUNG="0.55",
                      RISK_LAYER_EMB_DEVICE="cuda", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = {"format": "quality_precision_live/1", "status": "running", "profiles": list(PROFILES),
              "rows": [], "mismatches": [], "input_hashes": {}, "frozen_inputs": {}, "notes": [
                  "Real frozen Pipeline.run with pinned local GPU encoder and both rerankers, without score caches or monkeypatches.",
                  "Current public profiles are compared directly to their exact CPU replay, bound to R2 and its analysis.",
                  "Load and one warmup per profile are excluded from synchronized case timings.",
                  "Differences in norm order, displayed scores, source evidence, context and hard marks remain visible.",
                  "Historical development cases are not fresh blind gold; Hit@5 is not complete suggestion precision."]}
    try:
        result["source_hashes"] = freeze_sources(out)
        rows, paths, identities = prepare(args, out, result)
        helper.save(out / "manifest.json", {key: value for key, value in result.items() if key not in ("rows", "mismatches")})
        execute(rows, paths, identities, out, result)
        for path, expected in result["input_hashes"].items():
            if helper.digest(path) != expected:
                raise ValueError(f"Input changed during live validation: {path}")
        for name, expected in result["source_hashes"].items():
            if helper.digest(out / "source" / name) != expected:
                raise ValueError(f"Frozen source changed during live validation: {name}")
        result["status"] = ("mismatch" if result["mismatches"] else
                            "diagnostic_pass" if result["diagnostic_subset"] else "pass")
        code = 1 if result["mismatches"] else 0
    except Exception as exc:
        result["status"], result["error"] = "failed", {"type": type(exc).__name__, "message": str(exc)}
        code = 2
    result["torch_imported"] = "torch" in sys.modules
    if (out / "rows.jsonl").exists():
        result["rows_sha256"] = helper.digest(out / "rows.jsonl")
    helper.save(out / "report.json", result)
    print(f"{result['status']}: {len(result['rows'])} cases, {len(result['mismatches'])} differences; {out / 'report.json'}", flush=True)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
