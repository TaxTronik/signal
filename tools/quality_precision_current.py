"""Verify current public Pipeline profiles against this run's exact GPU caches.

No models or GPU. This snapshots current production source, replays all 162
historical cases through Pipeline.run, and compares norm order, displayed raw
score rounding, source evidence, and hard markings with bound references.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"kaskade": "kaskade", "praezision": "kaskade_bge_weak", "dokument": "document50_bge_weak"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_marks(report):
    return {"cards": [{"id": card.eintrag.id, "via": card.via,
                       "span": [card.treffer.start, card.treffer.end]} for card in report.karten],
            "trigger_marks": [{"span": [risk.start, risk.end], "title": risk.titel,
                               "norm_ids": [n["id"] for n in risk.norm_vorschlag]}
                              for risk in report.risiken if risk.quelle == "trigger"]}


def canonical(suggestions):
    """Normalize only known reference field names and API score rounding."""
    result = []
    for suggestion in suggestions:
        norms = []
        for norm in suggestion["normen"]:
            evidence = norm.get("evidenz", norm.get("evidence"))
            if evidence is None:
                raise ValueError("Reference lacks source evidence")
            spans = evidence.get("passagen", evidence.get("passages"))
            norms.append({"id": norm["id"],
                          "sim": round(norm["sim"], 3) if norm["sim"] is not None else None,
                          "rerank": round(norm["rerank"], 3),
                          "source": {"fingerprint": evidence["fingerprint"],
                              "truncated": evidence.get("gekuerzt", evidence.get("truncated")),
                              "anchors": evidence.get("anker", evidence.get("anchors")),
                              "passages": [{"id": p["id"], "norm_id": p.get("norm_id", norm["id"]),
                                  "start": p.get("quell_start", p.get("source_start")),
                                  "end": p.get("quell_end", p.get("source_end")), "text": p["text"]}
                                  for p in spans]}})
        result.append({"start": suggestion["start"], "end": suggestion["end"],
                       "text": suggestion["satz"], "norms": norms,
                       "context": suggestion.get("kontext", suggestion.get("context"))})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "out/quality-precision-20260907-r2")
    parser.add_argument("--analysis", type=Path, required=True,
                        help="Successful report produced by tools/quality_precision_analysis.py")
    parser.add_argument("--out", type=Path, required=True, help="Fresh current-replay output directory")
    args = parser.parse_args(argv)
    directory, output = args.run.resolve(), args.out.resolve()
    if output.exists():
        parser.error("Choose a fresh output directory to preserve previous validation")
    if any(name == "risk_layer" or name.startswith("risk_layer.") for name in sys.modules):
        parser.error("Run as a separate process so the current snapshot is imported first")
    helper_path = Path(__file__).with_name("quality_precision_analysis.py")
    helper = load_module(helper_path, "precision_replay_helpers")
    report, analysis = helper.read(directory / "report.json"), helper.read(args.analysis)
    if (report["status"] != "complete" or report["diagnostic_subset"] or analysis["status"] != "pass"
            or analysis["source_report_sha256"] != helper.digest(directory / "report.json")):
        parser.error("Requires a complete R2 report and its successful exact CPU analysis")
    if analysis["analysis_source_sha256"] != helper.digest(helper_path):
        raise ValueError("Replay helper differs from the successfully evaluated analysis source")
    if (Counter(row["dataset"] for row in report["rows"]) != Counter({"main": 132, "additional": 30})
            or len({row["case_id"] for row in report["rows"]}) != 162):
        raise ValueError("Missing or duplicate R2 cases")
    for path, value in analysis["verified_hashes"].items():
        if helper.digest(path) != value:
            raise ValueError(f"Bound input/source/cache changed: {path}")
    output.mkdir(parents=True)
    source_hashes = {}
    source_paths = sorted((ROOT / "risk_layer").rglob("*.py")) + [ROOT / "catalog/begriffe.yaml",
                     ROOT / "tools/_gpu_eval_metrics.py", Path(__file__), helper_path]
    for path in source_paths:
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[relative.as_posix()] = helper.digest(target)
        if helper.digest(path) != helper.digest(target):
            raise ValueError("Source changed while snapshotting current production")
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus",
                      RISK_LAYER_FUSION="an", RISK_LAYER_RERANK_SCHWELLE="0.10", RISK_LAYER_SIM_RETTUNG="0.55")
    sys.path.insert(0, str(output / "source"))
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.pipeline import AnalyseOptionen, Pipeline
    from risk_layer.semantic_search import PROFILE_CONFIG
    from risk_layer.triggerindex import TriggerIndex
    import risk_layer.pipeline as pipeline_module
    if not Path(pipeline_module.__file__).resolve().is_relative_to(output / "source"):
        raise ValueError("Current Pipeline was imported before its source snapshot")
    metrics = load_module(output / "source/tools/_gpu_eval_metrics.py", "current_metrics")
    trace = helper.Trace()
    encoder = helper.Encoder(directory, report["models"]["encoder"], trace)

    class Scores(helper.Ranker):
        def __init__(self, path, identity):
            super().__init__(path, identity, trace, "bge" if path.name.startswith("bge") else "qwen", {})
            self.used = set()

        def _score(self, pairs, instruction=None, max_length=1024):
            if instruction is not None:
                raise ValueError("A public profile unexpectedly requested a binary model check")
            keys = [helper.key_for(None, max_length, query, document) for query, document in pairs]
            self.used.update(keys)
            self.trace.record_work(self.stage, keys, sum(len(q) + len(d) for q, d in pairs))
            return [self.scores[key] for key in keys]

    ranker = Scores(directory / "qwen-scores.jsonl", report["models"]["qwen"])
    ranker.vorauswahl = Scores(directory / "bge-scores.jsonl", report["models"]["bge"])
    paths = list(map(Path, report["input_hashes"]))
    index = EmbeddingIndex.load(next(path.parent for path in paths if path.name == "meta.json"))
    graph = NormGraph(next(path for path in paths if path.name == "graph.sqlite"))
    trigger = TriggerIndex(next(path for path in paths if path.name == "trigger.sqlite"))
    catalog = lade_katalog(output / "source/catalog/begriffe.yaml")
    pipeline = Pipeline(catalog)
    reference = helper.read(directory / "reference.json")
    standard_rows = {row["case_id"]: row for row in reference["rows"]}
    analyzed_rows = {row["case_id"]: row for row in analysis["rows"]}
    result = {"format": "quality_precision_current/1", "status": "running", "source_hashes": source_hashes,
              "source_run": str(directory), "source_report_sha256": helper.digest(directory / "report.json"),
              "analysis_report_sha256": helper.digest(args.analysis), "reference_mapping": EXPECTED,
              "profiles": {name: PROFILE_CONFIG[name] for name in EXPECTED}, "rows": [],
              "mismatches": [], "hard_mark_mismatches": [], "cache_misses": [], "errors": [],
              "notes": ["Actual current public Pipeline.run from the captured source; no profile-config patching.",
                        "R2 raw query vectors and exact query/document model scores only; no GPU or weights.",
                        "Norm order, 3-decimal API scores, context spans, and source evidence must match.",
                        "Kaskade compares with the live R2 variant; praezision/dokument with the exact post-hoc ranked/context agreement rules.",
                        "Sentence-context offsets for offline agreement come from the same case's unfiltered native base output.",
                        "Hard cards and trigger marks compare with the bound original production standard reference.",
                        "Model workload counts requested pairs and unique exact inputs per profile/dataset, not GPU latency.",
                        "R2 document reference repeats some focus-specific relevance calls; unique-input savings separate this from deduplication.",
                        "Context relevance does not establish support for every individual focus sentence.",
                        "Historical development data; exact implementation agreement is not new blind quality evidence."]}
    fields = ("case_id", "dataset", "group", "query", "targets", "negative", "category", "uncertain")
    try:
        if index is None or not pruefe_aktualitaet(index, graph, catalog):
            raise ValueError("Current catalog/graph/index are incompatible")
        if any(PROFILE_CONFIG[name].get("checks") for name in EXPECTED):
            raise ValueError("Public profiles unexpectedly enable research binary checks")
        for i, row in enumerate(report["rows"], 1):
            baseline, analyzed = standard_rows[row["case_id"]], analyzed_rows[row["case_id"]]
            if any(row[key] != baseline.get(key) or row[key] != analyzed.get(key) for key in fields):
                raise ValueError("Case content/labels differ between bound references")
            current = {key: row[key] for key in fields}
            for profile, variant in EXPECTED.items():
                trace.variant = (row["dataset"], profile)
                expected = analyzed[variant]["suggestions"]
                offline = analysis["offline_variants"].get(variant)
                if offline is not None and not offline["context_only"]:
                    contexts = {(s["start"], s["end"], s["satz"]): s["kontext"]
                                for s in row[offline["base"]]["suggestions"]}
                    expected = [{**s, "context": contexts[(s["start"], s["end"], s["satz"])]}
                                for s in expected]
                try:
                    actual = pipeline.run(row["query"], optionen=AnalyseOptionen(
                        erzeugt_am="quality-precision-current", detektor=True, streit=True, semantik_profil=profile),
                        graph=graph, trigger=trigger, embedding=index, embedding_encoder=encoder, reranker=ranker)
                    marks = compact_marks(actual)
                    normalized_actual, normalized_expected = canonical(actual.semantik_vorschlaege), canonical(expected)
                    order_equal = helper.norm_order(actual.semantik_vorschlaege) == helper.norm_order(expected)
                    all_equal = normalized_actual == normalized_expected
                    scores_equal, evidence_equal = False, False
                    if order_equal:
                        scores_equal = all((a["sim"], a["rerank"]) == (b["sim"], b["rerank"])
                            for sa, sb in zip(normalized_actual, normalized_expected, strict=True)
                            for a, b in zip(sa["norms"], sb["norms"], strict=True))
                        evidence_equal = all(a["source"] == b["source"]
                            for sa, sb in zip(normalized_actual, normalized_expected, strict=True)
                            for a, b in zip(sa["norms"], sb["norms"], strict=True))
                    current[profile] = {"suggestions": actual.semantik_vorschlaege, **marks,
                        "comparison": {"all_equal": all_equal, "norm_order_equal": order_equal,
                                       "scores_equal": scores_equal, "evidence_equal": evidence_equal}}
                    if not all_equal:
                        result["mismatches"].append({"case_id": row["case_id"], "profile": profile,
                                                    **current[profile]["comparison"], "expected": normalized_expected,
                                                    "actual": normalized_actual})
                    if any(marks[key] != baseline["standard"][key] for key in marks):
                        result["hard_mark_mismatches"].append({"case_id": row["case_id"], "profile": profile,
                            "actual": marks, "expected": {key: baseline["standard"][key] for key in marks}})
                except KeyError as exc:
                    current[profile] = {"error": "Exact cache/input key missing", "detail": str(exc)}
                    result["cache_misses"].append({"case_id": row["case_id"], "profile": profile, "detail": str(exc)})
                except Exception as exc:
                    current[profile] = {"error": type(exc).__name__, "detail": str(exc)}
                    result["errors"].append({"case_id": row["case_id"], "profile": profile, "detail": repr(exc)})
            result["rows"].append(current)
            if i % 10 == 0:
                print(f"Current Pipeline {i}/162: mismatches={len(result['mismatches'])}, "
                      f"hard-mark mismatches={len(result['hard_mark_mismatches'])}, cache misses={len(result['cache_misses'])}", flush=True)
        result["summaries"] = {dataset: {profile: metrics.summarize([
            row for row in result["rows"] if row["dataset"] == dataset], profile) for profile in EXPECTED
            if all("error" not in row[profile] for row in result["rows"] if row["dataset"] == dataset)}
            for dataset in ("main", "additional")}
        result["used_model_inputs"] = {"bge": len(ranker.vorauswahl.used), "qwen": len(ranker.used)}
        workload, savings = {}, {}
        for (dataset, profile), stage_work in trace.work.items():
            workload.setdefault(dataset, {})[profile] = {
                stage: {**{key: value for key, value in values.items() if key != "unique"},
                        "unique_inputs_if_run_alone": len(values["unique"])}
                for stage, values in stage_work.items()}
            reference_work = analysis["workload"][dataset][EXPECTED[profile]]
            comparison = {}
            for stage in ("bge", "qwen"):
                current_counts = workload[dataset][profile].get(stage, {"pairs": 0, "unique_inputs_if_run_alone": 0})
                reference_counts = reference_work.get(stage, {"pairs": 0, "unique_inputs_if_run_alone": 0})
                comparison[stage] = {
                    "reference_pairs": reference_counts["pairs"], "current_pairs": current_counts["pairs"],
                    "pairs_saved": reference_counts["pairs"] - current_counts["pairs"],
                    "reference_unique_inputs": reference_counts["unique_inputs_if_run_alone"],
                    "current_unique_inputs": current_counts["unique_inputs_if_run_alone"],
                    "unique_inputs_saved": reference_counts["unique_inputs_if_run_alone"] - current_counts["unique_inputs_if_run_alone"],
                }
            savings.setdefault(dataset, {})[profile] = {"reference_variant": EXPECTED[profile], **comparison}
        result["workload"], result["workload_savings_vs_r2"] = workload, savings
        result["torch_imported"] = "torch" in sys.modules
        if result["torch_imported"]:
            raise RuntimeError("Unexpected model runtime imported during CPU-only Pipeline replay")
        result["status"] = "pass" if not any(result[key] for key in (
            "mismatches", "hard_mark_mismatches", "cache_misses", "errors")) else "mismatch"
    except Exception as exc:
        result["status"] = "failed"
        result["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        graph.close()
        trigger.close()
        (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{result['status']}: {len(result['rows'])} cases x {len(EXPECTED)} public profiles; {output / 'report.json'}", flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
