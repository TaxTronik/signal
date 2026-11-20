"""Run the actual production Pipeline with local GPU models on 162 existing cases.

The complete CPU production replay supplies cases, labels, and exact comparison
rows. Additional CPU reports can supply cascade references. No retrieval or
scoring method is replaced, no model is downloaded, and no index is activated.
Missing references and changed predictions are reported, never silently passed.
Use --case-limit only for diagnostics; it is not a full quality evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASE_FIELDS = ("case_id", "group", "query", "targets", "negative", "category", "uncertain", "dataset")
PROFILES = ("normpaket", "kontext", "kaskade")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot_sources(out):
    paths = sorted((ROOT / "risk_layer").rglob("*.py")) + [
        ROOT / "catalog/begriffe.yaml", ROOT / "tools/_gpu_eval_metrics.py", Path(__file__)]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[relative.as_posix()] = digest(destination)
        if digest(path) != hashes[relative.as_posix()]:
            raise ValueError(f"Source changed while freezing: {path}")
    return hashes


def freeze_json(path, out, name, result):
    path = path.resolve()
    destination = out / "inputs" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    frozen_hash = digest(destination)
    if digest(path) != frozen_hash:
        raise ValueError(f"Input changed while freezing: {path}")
    result["input_hashes"][str(path)] = frozen_hash
    result["frozen_inputs"][destination.relative_to(out).as_posix()] = frozen_hash
    return json.loads(destination.read_text(encoding="utf-8"))


def source_hashes(report):
    return {name.replace("\\", "/"): value for name, value in report["source_hashes"].items()}


def validate_cpu_report(report, directory, result):
    if (report.get("format") != "quality_production_check/1"
            or report.get("status") not in ("pass", "mismatch")
            or report.get("error") or report.get("cache_misses") or report.get("torch_imported")):
        raise ValueError(f"Not a completed model-free production reference: {directory}")
    # A difference from the earlier research runner is not a failed CPU run.
    # Its ACTUAL rows remain valid references for this live production run.
    for name, expected in source_hashes(report).items():
        if digest(directory / "source" / name) != expected:
            raise ValueError(f"CPU reference source archive changed: {directory / name}")
    for name, expected in report["index"]["hashes"].items():
        path = directory / "index" / name
        if digest(path) != expected:
            raise ValueError(f"CPU reference index changed: {path}")
        result["input_hashes"][str(path.resolve())] = expected
    seen = set()
    for row in report["rows"]:
        if row["case_id"] in seen:
            raise ValueError("Duplicate case ID in CPU reference")
        seen.add(row["case_id"])
        for profile in report["profiles"]:
            actual = row.get(profile, {})
            if "error" in actual or not all(key in actual for key in ("suggestions", "cards", "trigger_marks")):
                raise ValueError(f"Incomplete CPU reference: {row['case_id']}/{profile}")


def prepare_inputs(args, out, result):
    production = args.production.resolve()
    primary = freeze_json(production / "report.json", out, "production-report.json", result)
    validate_cpu_report(primary, production, result)
    cases = [{key: row.get(key) for key in CASE_FIELDS} for row in primary["rows"]]
    counts = dict(Counter(case["dataset"] for case in cases))
    if counts != {"main": 132, "additional": 30}:
        raise ValueError(f"Expected complete 132 main + 30 historical cases, got {counts}")
    by_id = {case["case_id"]: case for case in cases}
    baseline_path = args.baseline.resolve() / "report.json"
    baseline = freeze_json(baseline_path, out, "baseline-report.json", result)
    baseline_hash = result["input_hashes"][str(baseline_path)]
    baseline_artifacts = {name.replace("\\", "/"): value for name, value in baseline["source_hashes"].items()}
    for name in ("artifacts/graph.sqlite", "artifacts/trigger.sqlite"):
        path = args.baseline.resolve() / name
        expected = baseline_artifacts[name]
        if digest(path) != expected:
            raise ValueError(f"Archived graph/trigger changed: {path}")
        result["input_hashes"][str(path)] = expected
    reports = [(production / "report.json", primary)]
    for i, path in enumerate(args.reference):
        path = path.resolve()
        extra = freeze_json(path, out, f"reference-{i}.json", result)
        validate_cpu_report(extra, path.parent, result)
        reports.append((path, extra))
    references = {}
    for path, reference in reports:
        if baseline_hash not in reference["input_reports"].values():
            raise ValueError(f"CPU reference is not bound to the selected original baseline: {path}")
        if reference["index"]["input_fingerprint"] != primary["index"]["input_fingerprint"]:
            raise ValueError(f"CPU reference uses a different passage index: {path}")
        result["reference_reports"].append({
            "path": str(path), "status": reference["status"],
            "research_mismatches": reference.get("mismatches", []),
            "hard_mark_mismatches": reference.get("hard_mark_mismatches", []),
            "source_differences": [name for name, value in source_hashes(reference).items()
                                   if name.startswith(("risk_layer/", "catalog/"))
                                   and result["source_hashes"].get(name) != value]})
        for row in reference["rows"]:
            case = {key: row.get(key) for key in CASE_FIELDS}
            if by_id.get(case["case_id"]) != case:
                raise ValueError(f"Reference case text, labels, or dataset changed: {case['case_id']}")
            for profile in reference["profiles"]:
                key = (case["case_id"], profile)
                entry = (row[profile], str(path))
                if key in references and references[key][0] != entry[0]:
                    raise ValueError(f"Conflicting CPU reference rows: {key}")
                references[key] = entry
    selected = cases if args.case_limit is None else cases[:args.case_limit]
    result["case_counts"] = counts
    result["selected_case_counts"] = dict(Counter(case["dataset"] for case in selected))
    result["omitted_case_ids"] = [case["case_id"] for case in cases[len(selected):]]
    result["diagnostic_subset"] = len(selected) != len(cases)
    result["index"] = primary["index"]
    save(out / "manifest.json", {key: result[key] for key in (
        "source_hashes", "input_hashes", "frozen_inputs", "profiles", "case_counts",
        "selected_case_counts", "omitted_case_ids", "diagnostic_subset", "reference_reports", "index")})
    return selected, references


def norm_order(suggestions):
    return [{"start": item["start"], "end": item["end"], "satz": item["satz"],
             "norm_ids": [norm["id"] for norm in item["normen"]]} for item in suggestions]


def compact_marks(report):
    return {
        "cards": [{"id": card.eintrag.id, "via": card.via,
                   "span": [card.treffer.start, card.treffer.end]} for card in report.karten],
        "trigger_marks": [{"span": [risk.start, risk.end], "title": risk.titel,
                           "norm_ids": [norm["id"] for norm in risk.norm_vorschlag]}
                          for risk in report.risiken if risk.quelle == "trigger"],
    }


def comparison(actual, expected, targets, evaluate):
    order_equal = norm_order(actual["suggestions"]) == norm_order(expected["suggestions"])
    actual_metrics, expected_metrics = (evaluate(value["suggestions"], targets) for value in (actual, expected))
    score_differences = []
    if order_equal:
        for sentence, (actual_s, expected_s) in enumerate(zip(actual["suggestions"], expected["suggestions"], strict=True)):
            for rank, (an, en) in enumerate(zip(actual_s["normen"], expected_s["normen"], strict=True), 1):
                for score in ("sim", "rerank"):
                    if an.get(score) != en.get(score):
                        score_differences.append({"sentence": sentence, "rank": rank, "score": score,
                                                  "actual_display": an.get(score), "expected_display": en.get(score)})
    return {"order_equal": order_equal, "metrics_equal": actual_metrics == expected_metrics,
            "actual_metrics": actual_metrics, "expected_metrics": expected_metrics,
            "hard_marks_equal": all(actual[key] == expected[key] for key in ("cards", "trigger_marks")),
            "score_display_differences": score_differences}


def timing_summary(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "total_seconds": sum(values), "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values), "min_seconds": ordered[0],
            "max_seconds": ordered[-1], "p95_seconds": ordered[max(0, (95 * len(values) + 99) // 100 - 1)]}


def model_info(model, wrapper, seconds):
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if not devices or any(not device.startswith("cuda:") for device in devices):
        raise ValueError(f"Model is not fully on GPU: {wrapper.name}: {devices}")
    return {"name": wrapper.name, "revision": wrapper.revision, "devices": devices,
            "dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "load_seconds": seconds}


def execute(args, out, result, cases, references):
    # All sources, reference reports, and data hashes are frozen BEFORE imports.
    if any(name == "risk_layer" or name.startswith("risk_layer.") for name in sys.modules):
        raise RuntimeError("Production package was imported before its source snapshot")
    sys.path[:0] = [str(out / "source"), str(ROOT)]
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import BgeM3Encoder, EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.pipeline import AnalyseOptionen, Pipeline
    from risk_layer.reranker import CrossEncoderReranker, reranker_oeffnen
    from risk_layer.triggerindex import TriggerIndex
    from tools._gpu_eval_metrics import evaluate_variant, summarize
    import risk_layer.pipeline as pipeline_module
    import tools._gpu_eval_metrics as metrics_module

    for module in (pipeline_module, metrics_module):
        if not Path(module.__file__).resolve().is_relative_to(out / "source"):
            raise RuntimeError(f"Module not imported from frozen sources: {module.__file__}")
    index = EmbeddingIndex.load(args.production.resolve() / "index")
    if index is None or index.passage_evidence is None:
        raise ValueError("A real saved passage index is required")
    catalog = lade_katalog(out / "source/catalog/begriffe.yaml")
    graph = NormGraph(args.baseline.resolve() / "artifacts/graph.sqlite")
    trigger = TriggerIndex(args.baseline.resolve() / "artifacts/trigger.sqlite")
    try:
        if not pruefe_aktualitaet(index, graph, catalog):
            raise ValueError("Passage index is incompatible with the frozen production source/catalog/graph")
        start = time.perf_counter()
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("This is a GPU-only live check; CUDA/ROCm GPU is unavailable")
        torch.cuda.synchronize()
        result["runtime"] = {
            "python": sys.version, "torch": torch.__version__, "hip": torch.version.hip,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "runtime_init_seconds": time.perf_counter() - start,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "transformers", "sentence-transformers")},
            "bge_batch_size": 8, "cascade_qwen_batch_size": 2,
        }
        encoder = BgeM3Encoder(device="cuda", local_files_only=True)
        bge = CrossEncoderReranker(device="cuda", batch_size=8)
        cascade = reranker_oeffnen("bge-qwen3-reranker-4b", device="cuda") if "kaskade" in args.profiles else None
        if cascade is not None:
            # Factory-created models are lazy. Reuse the single real BGE model,
            # so no second BGE instance is loaded beside encoder and Qwen4B.
            cascade.vorauswahl = bge
        if encoder.name != index.encoder_name or encoder.revision != index.encoder_revision:
            raise ValueError("GPU encoder revision differs from the saved passage index")
        result["models"] = {}
        for label, wrapper in (("encoder", encoder), ("bge", bge), ("qwen", cascade)):
            if wrapper is None:
                continue
            print(f"Loading {label}: {wrapper.name} ({wrapper.revision})", flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            model = wrapper._laden()
            torch.cuda.synchronize()
            # SentenceTransformer and Qwen expose parameters directly;
            # sentence-transformers' CrossEncoder exposes its torch model.
            result["models"][label] = model_info(model.model if label == "bge" else model,
                                                 wrapper, time.perf_counter() - start)
        pipe = Pipeline(catalog)

        def run_case(case, profile):
            return pipe.run(case["query"],
                            optionen=AnalyseOptionen(erzeugt_am="quality-live-check", detektor=True,
                                                     streit=True, semantik_profil=profile),
                            graph=graph, trigger=trigger, embedding=index, embedding_encoder=encoder,
                            reranker=cascade if profile == "kaskade" else bge)

        result["warmup"] = []
        for profile in args.profiles:
            torch.cuda.synchronize()
            start = time.perf_counter()
            run_case(cases[0], profile)
            torch.cuda.synchronize()
            result["warmup"].append({"profile": profile, "case_id": cases[0]["case_id"],
                                     "seconds": time.perf_counter() - start})
        torch.cuda.reset_peak_memory_stats()
        for i, case in enumerate(cases, 1):
            row = dict(case)
            result["rows"].append(row)
            for profile in args.profiles:
                print(f"Live {i}/{len(cases)} {case['dataset']} {case['case_id']} {profile}", flush=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                report = run_case(case, profile)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                actual = {"suggestions": report.semantik_vorschlaege, **compact_marks(report),
                          "elapsed_seconds": elapsed}
                actual["metrics"] = evaluate_variant(actual["suggestions"], case["targets"])
                row[profile] = actual
                key = (case["case_id"], profile)
                if key in references:
                    expected, reference_path = references[key]
                    actual["reference"] = reference_path
                    actual["comparison"] = comparison(actual, expected, case["targets"], evaluate_variant)
                    checks = actual["comparison"]
                    if not all(checks[name] for name in ("order_equal", "metrics_equal", "hard_marks_equal")):
                        result["mismatches"].append({"case_id": case["case_id"], "dataset": case["dataset"],
                                                     "profile": profile, "comparison": checks,
                                                     "expected": norm_order(expected["suggestions"]),
                                                     "actual": norm_order(actual["suggestions"])})
                else:
                    actual["comparison"] = None
                    result["uncompared"].append({"case_id": case["case_id"], "profile": profile})
                standard = references.get((case["case_id"], "standard"))
                if standard is not None:
                    equal = all(actual[name] == standard[0][name] for name in ("cards", "trigger_marks"))
                    actual["hard_marks_equal_cpu_standard"] = equal
                    if not equal:
                        result["hard_mark_mismatches"].append({"case_id": case["case_id"], "profile": profile})
            with (out / "rows.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                save(out / "report.json", result)
        result["summaries"] = {
            dataset: {profile: summarize([row for row in result["rows"] if row["dataset"] == dataset], profile)
                      for profile in args.profiles} for dataset in ("main", "additional")}
        result["timings"] = {
            dataset: {profile: timing_summary([row[profile]["elapsed_seconds"] for row in result["rows"]
                                               if dataset == "all" or row["dataset"] == dataset])
                      for profile in args.profiles} for dataset in ("all", "main", "additional")}
        result["runtime"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["runtime"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    finally:
        graph.close()
        trigger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, default=ROOT / "out/quality-production-r2",
                        help="Complete 162-case CPU production replay with its saved index")
    parser.add_argument("--baseline", type=Path, default=ROOT / "out/gpu-ab-20260907-final")
    parser.add_argument("--reference", type=Path, action="append", default=[],
                        help="Additional completed CPU production report.json, e.g. cascade; repeatable")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["kontext", "kaskade"])
    parser.add_argument("--case-limit", type=int, help="Diagnostic first N cases only; not a complete evaluation")
    parser.add_argument("--out", type=Path, required=True, help="Fresh evidence directory; never overwrite/resume")
    args = parser.parse_args(argv)
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")
    if len(args.profiles) != len(set(args.profiles)):
        parser.error("--profiles must not contain duplicates")
    out = args.out.resolve()
    if out.exists():
        parser.error("Output directory already exists; choose a fresh --out directory")
    out.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus", RISK_LAYER_FUSION="an",
                      RISK_LAYER_RERANK_SCHWELLE="0.10", RISK_LAYER_SIM_RETTUNG="0.55",
                      RISK_LAYER_EMB_DEVICE="cuda", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = {"format": "quality_live_check/1", "status": "running", "profiles": args.profiles,
              "input_hashes": {}, "frozen_inputs": {}, "reference_reports": [], "rows": [],
              "mismatches": [], "hard_mark_mismatches": [], "uncompared": [], "notes": [
                  "Actual frozen production Pipeline.run, real local GPU query encoder and rerankers; no method replacement.",
                  "The saved passage index is loaded through production code and is not rebuilt or activated.",
                  "Model load and one warmup case per profile are excluded from synchronized per-case GPU timings.",
                  "BGE batch 8 and factory cascade Qwen batch 2; one shared BGE preselector.",
                  "Main development cases and 30 historical additional cases remain separate; neither is fresh blind gold.",
                  "CPU research mismatches are retained as provenance; actual CPU rows are the live comparison target.",
                  "GPU batch effects may change scores or ranks; differences are measured, not normalized away.",
                  "Returned scores are production display scores, not unrounded model logits."]}
    try:
        result["source_hashes"] = snapshot_sources(out)
        cases, references = prepare_inputs(args, out, result)
        execute(args, out, result, cases, references)
        for path, expected in result["input_hashes"].items():
            if digest(path) != expected:
                raise ValueError(f"Input changed during live evaluation: {path}")
        for name, expected in result["source_hashes"].items():
            if digest(out / "source" / name) != expected:
                raise ValueError(f"Frozen source changed during live evaluation: {name}")
        result["status"] = ("mismatch" if result["mismatches"] or result["hard_mark_mismatches"] else
                            "uncompared" if result["uncompared"] else
                            "diagnostic_pass" if result["diagnostic_subset"] else "pass")
        code = 0 if result["status"] in ("pass", "diagnostic_pass") else 1
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        code = 2
    result["torch_imported"] = "torch" in sys.modules
    if (out / "rows.jsonl").exists():
        result["rows_sha256"] = digest(out / "rows.jsonl")
    save(out / "report.json", result)
    print(f"{result['status']}: {len(result['rows'])} cases, {len(result['mismatches'])} mismatches, "
          f"{len(result['uncompared'])} missing references; {out / 'report.json'}", flush=True)
    if "error" in result:
        print(result["error"], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
