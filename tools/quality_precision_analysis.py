"""Exact CPU-only replay and stage analysis of a completed precision GPU run.

Run with .venv/Scripts/python.exe tools/quality_precision_analysis.py
    --run out/quality-precision-20260907-r2 --out out/precision-rerun/analysis
No weights, tokenizer, GPU, or historical model output are used. The replay
imports the evaluated run's verified source snapshot, not today's production.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

AGREEMENT_POLICIES = {
    "floor": {"bge_min": .01},
    "weak": {"bge_min": .1, "sim_rescue": .55, "rescue_bge_min": .01},
    "strict": {"bge_min": .1},
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key_for(instruction, budget, query, document):
    return hashlib.sha256(json.dumps([instruction, budget, query, document],
                                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def norm_id(document):
    value = document.split("\n", 1)[0]
    if not value.startswith("norm:"):
        raise ValueError("Expected the original norm package, with a norm ID header")
    return value


def matches(norm, target):
    return norm == target or norm.startswith(target + ":")


def norm_order(suggestions):
    return [(s["start"], s["end"], [n["id"] for n in s["normen"]]) for s in suggestions]


class Trace:
    def __init__(self):
        self.groups, self.work = [], {}
        self.variant = None

    @property
    def group(self):
        if not self.groups:
            raise ValueError("A model call occurred without preceding retrieval")
        return self.groups[-1]

    def start(self, candidates):
        self.groups.append({"retrieved": {n["norm_id"]: n.get("sim") for n in candidates},
                            "packages": {}, "provenance": {}, "bge": {}, "qwen": {},
                            "support": {}, "contradiction": {}})

    def record_work(self, stage, keys, chars=0):
        work = self.work.setdefault(self.variant, {}).setdefault(stage, {"calls": 0, "pairs": 0,
                                                                       "input_characters": 0, "unique": set()})
        work["calls"] += 1
        work["pairs"] += len(keys)
        work["input_characters"] += chars
        work["unique"].update(keys)


class Encoder:
    def __init__(self, directory, identity, trace):
        import numpy as np
        texts, vectors = read(directory / "queries.texts.json"), np.load(directory / "queries.npy", allow_pickle=False)
        if (len(set(texts)) != len(texts) or vectors.ndim != 2 or len(texts) != len(vectors)
                or vectors.dtype != np.float32 or not np.isfinite(vectors).all()):
            raise ValueError("Invalid exact query vector archive")
        self.name, self.revision, self.dim = identity["name"], identity["revision"], vectors.shape[1]
        self.vectors, self.trace = dict(zip(texts, vectors, strict=True)), trace

    def encode(self, texts):
        import numpy as np
        self.trace.record_work("encoder", texts, sum(map(len, texts)))
        return np.stack([self.vectors[text] for text in texts])


class Ranker:
    def __init__(self, path, identity, trace, stage, instructions):
        self.name, self.revision = identity["name"], identity["revision"]
        self.trace, self.stage, self.instructions = trace, stage, instructions
        self.scores = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                instruction, budget, query, document, score = json.loads(line)
                if (type(score) not in (int, float) or not math.isfinite(score)
                        or instruction is not None and not 0 <= score <= 1):
                    raise ValueError("Invalid raw cached score")
                key = key_for(instruction, budget, query, document)
                if key in self.scores and self.scores[key] != score:
                    raise ValueError("Conflicting scores for the same exact model input")
                self.scores[key] = score

    def _score(self, pairs, instruction=None, max_length=1024):
        stage = self.stage if instruction is None else self.instructions[instruction]
        keys = [key_for(instruction, max_length, query, document) for query, document in pairs]
        values = [self.scores[key] for key in keys]
        if instruction is None and pairs:
            if len({query for query, _ in pairs}) != 1:
                raise ValueError("Expected one original ranking query per scoring call")
            self.trace.group["ranking_query"] = pairs[0][0]
        self.trace.record_work(stage, keys, sum(len(q) + len(d) + len(instruction or "") for q, d in pairs))
        for (_, document), value in zip(pairs, values, strict=True):
            self.trace.group[stage][norm_id(document)] = value
        return values

    def score(self, pairs):
        return self._score(pairs)

    def score_with_instruction(self, pairs, instruction, *, max_length=2048):
        return self._score(pairs, instruction, max_length)


class Index:
    def __init__(self, wrapped, trace):
        self.wrapped, self.trace = wrapped, trace

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def hybrid_topk(self, *args, **kwargs):
        result = self.wrapped.hybrid_topk(*args, **kwargs)
        self.trace.start(result)
        return result

    def _topk(self, *args, **kwargs):
        result = self.wrapped._topk(*args, **kwargs)
        self.trace.start(result)
        return result

    def norm_package(self, nid, *args, **kwargs):
        result = self.wrapped.norm_package(nid, *args, **kwargs)
        self.trace.group["packages"][nid] = bool(result and result.evidence)
        if result:
            self.trace.group["provenance"][nid] = {
                "document_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
                "fingerprint": result.source_fingerprint, "truncated": result.truncated,
                "passages": [{"id": p.passage_id, "source_start": p.start, "source_end": p.end,
                              "text": p.text} for p in result.evidence], "anchors": list(result.anchors)}
        return result


class Trigger:
    def __init__(self, wrapped, trace):
        self.wrapped, self.trace = wrapped, trace

    def kandidaten(self, *args, **kwargs):
        result = self.wrapped.kandidaten(*args, **kwargs)
        for n in result:
            self.trace.group["retrieved"].setdefault(n["norm_id"], n.get("sim"))
        return result


def agrees(bge, sim, policy):
    if policy is None:
        return True
    spec = AGREEMENT_POLICIES[policy]
    return bge >= spec["bge_min"] or ("sim_rescue" in spec and sim is not None
        and sim >= spec["sim_rescue"] and bge >= spec["rescue_bge_min"])


def agreement_suggestions(groups, contexts, policy, *, focus_specific=False, context_only=False):
    if focus_specific:
        if len(groups) != len(contexts):
            raise ValueError("Focus-specific groups differ from the document-profile replay")
        mapped = list(zip(contexts, groups, strict=True))
        if any(group.get("ranking_query", context.query_text) != context.query_text for context, group in mapped):
            raise ValueError("The recorded ranking query differs from the original document context")
    else:
        unique_queries = list(dict.fromkeys(context.query_text for context in contexts))
        if len(groups) != len(unique_queries):
            raise ValueError("Sentence context groups differ from the real source-profile replay")
        by_query = dict(zip(unique_queries, groups, strict=True))
        mapped = [(context, by_query[context.query_text]) for context in contexts]
    suggestions = []
    for context, group in mapped:
        candidates = [nid for nid, score in group["qwen"].items() if score >= .5 and
                      agrees(group["bge"][nid], group["retrieved"].get(nid), policy)]
        candidates.sort(key=lambda nid: (-group["qwen"][nid],
            -(group["retrieved"].get(nid) if group["retrieved"].get(nid) is not None else float("-inf")), nid))
        if candidates:
            suggestions.append({"start": context.focus_start, "end": context.focus_end, "satz": context.focus_text,
                "normen": [{"id": nid, "rerank": group["qwen"][nid], "bge": group["bge"][nid],
                            "sim": group["retrieved"].get(nid), "evidence": group["provenance"][nid]}
                           for nid in candidates[:5]]})
            if context_only:
                suggestions[-1]["support_scope"] = "context_relevance_only"
                suggestions[-1]["context"] = {"start": context.context_start, "end": context.context_end,
                                               "text": context.query_text}
    return suggestions


def stages(groups, targets, actual, config, classify):
    pools = {name: set() for name in ("retrieved", "packaged", "source", "bge_shortlist",
                                     "qwen_gate", "agreement", "checks", "top5")}
    checks = Counter()
    details = {target: [] for target in targets}
    for group in groups:
        pools["retrieved"].update(group["retrieved"])
        pools["packaged"].update(group["packages"])
        pools["source"].update(nid for nid, present in group["packages"].items() if present)
        pools["bge_shortlist"].update(group["qwen"])
        ranks = {nid: i for i, nid in enumerate(sorted(group["bge"],
                 key=lambda nid: (-group["bge"][nid], nid)), 1)}
        for nid, bge in group["bge"].items():
            for target in targets:
                if matches(nid, target):
                    details[target].append({"norm_id": nid, "bge": bge, "bge_rank": ranks[nid],
                                            "qwen": group["qwen"].get(nid), "sim": group["retrieved"].get(nid),
                                            "support": group["support"].get(nid),
                                            "contradiction": group["contradiction"].get(nid)})
        for nid, qwen in group["qwen"].items():
            bge, sim = group["bge"][nid], group["retrieved"].get(nid)
            passed_qwen = qwen >= config["rerank_threshold"]
            agreement = agrees(bge, sim, config.get("agreement_policy") or ("weak" if config.get("agreement") else None))
            status = (classify(group["support"][nid], group["contradiction"][nid])
                      if nid in group["support"] else None)
            if status and config.get("checks"):
                checks[status] += 1
                if group["support"][nid] >= .5 and group["contradiction"][nid] >= .5:
                    checks["both_yes"] += 1
            if passed_qwen:
                pools["qwen_gate"].add(nid)
                if agreement:
                    pools["agreement"].add(nid)
                    if not config.get("checks") or status == "modell_gestuetzt":
                        pools["checks"].add(nid)
    pools["top5"] = {n["id"] for suggestion in actual for n in suggestion["normen"]}
    failures = {}
    names = ("retrieval", "fusion_pool_cap", "missing_source", "bge_shortlist", "qwen_threshold",
             "agreement", "factual_check", "top5")
    for target in targets:
        failures[target] = "hit"
        for (stage, pool), failure in zip(pools.items(), names, strict=True):
            if not any(matches(nid, target) for nid in pool):
                failures[target] = "no_eligible_context" if not groups else failure
                break
    return {"target_outcomes": failures, "target_scores": details, "check_status_counts": dict(checks),
            "stage_norm_counts": {stage: len(pool) for stage, pool in pools.items()}, "query_groups": len(groups)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "out/quality-precision-20260907-r2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    directory, output = args.run.resolve(), args.out.resolve()
    if any(name == "risk_layer" or name.startswith("risk_layer.") for name in sys.modules):
        parser.error("Run as a separate process so the verified snapshot is imported first")
    if output.exists():
        parser.error("Choose a fresh analysis output directory")
    report, manifest = read(directory / "report.json"), read(directory / "manifest.json")
    if report["status"] != "complete" or report["diagnostic_subset"] and not args.allow_diagnostic:
        parser.error("Requires a complete full run; diagnostic subsets need explicit --allow-diagnostic")
    if (len({row["case_id"] for row in report["rows"]}) != len(report["rows"])
            or not report["diagnostic_subset"] and Counter(row["dataset"] for row in report["rows"])
            != Counter({"main": 132, "additional": 30})):
        raise ValueError("Duplicate or incomplete evaluated case set")
    for field in ("source_hashes", "input_hashes", "models", "variants"):
        if report[field] != manifest[field]:
            raise ValueError(f"Report/manifest identity differs: {field}")
    hashes = {str(directory / "source" / path): value for path, value in report["source_hashes"].items()}
    hashes.update(report["input_hashes"])
    hashes.update({str(directory / name): value for name, value in report["query_artifacts"].items()})
    hashes.update({str(directory / f"{name}-scores.jsonl"): value["sha256"] for name, value in report["scoring"].items()})
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Source/input/cache checksum mismatch: {path}")
    paths = list(map(Path, report["input_hashes"]))
    source_report = next(path for path in paths if path.name == "report.json")
    if digest(directory / "reference.json") != digest(source_report):
        raise ValueError("Copied reference differs from the declared baseline")
    reference = read(directory / "reference.json")
    if reference["status"] != "pass":
        raise ValueError("Production reference did not pass its own exact replay")
    index_dir = next(path.parent for path in paths if path.name == "meta.json")
    if any(digest(index_dir / name) != value for name, value in reference["index"]["hashes"].items()):
        raise ValueError("Run index differs from the production reference generation")
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus")
    sys.path.insert(0, str(directory / "source"))
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.semantic_search import passage_suggestions, PROFILE_CONFIG
    from risk_layer.semantic_checks import SUPPORT_INSTRUCTION, CONTRADICTION_INSTRUCTION, classify
    from risk_layer.semantic_context import build_context_queries
    from risk_layer.triggerindex import TriggerIndex
    import risk_layer.semantic_search as semantic_module
    if not Path(semantic_module.__file__).resolve().is_relative_to(directory / "source"):
        raise ValueError("Replay imported production code outside the verified run snapshot")
    spec = importlib.util.spec_from_file_location("frozen_metrics", directory / "source/tools/_gpu_eval_metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    PROFILE_CONFIG.update(report["variants"])
    trace = Trace()
    encoder = Encoder(directory, report["models"]["encoder"], trace)
    ranker = Ranker(directory / "qwen-scores.jsonl", report["models"]["qwen"], trace, "qwen",
                    {SUPPORT_INSTRUCTION: "support", CONTRADICTION_INSTRUCTION: "contradiction"})
    ranker.vorauswahl = Ranker(directory / "bge-scores.jsonl", report["models"]["bge"], trace, "bge", {})
    original = EmbeddingIndex.load(index_dir)
    index = Index(original, trace)
    graph = NormGraph(next(path for path in paths if path.name == "graph.sqlite"))
    original_trigger = TriggerIndex(next(path for path in paths if path.name == "trigger.sqlite"))
    trigger = Trigger(original_trigger, trace)
    catalog = lade_katalog(directory / "source/catalog/begriffe.yaml")
    rows, mismatches = [], []
    base_rows = {row["case_id"]: row for row in reference["rows"]}
    fields = ("case_id", "dataset", "group", "query", "targets", "negative", "category", "uncertain")
    offline_variants = {f"{base}_bge_{policy}": {"base": base, "agreement_policy": policy, "qwen_min": .5,
                                                "checks": False, "context_only": base == "document50",
                                                **spec} for base in ("kaskade", "source20", "source50", "document50")
                        for policy, spec in AGREEMENT_POLICIES.items()
                        if base != "kaskade" or policy == "weak"}
    all_variants = [*report["variants"], *offline_variants]
    try:
        if original is None or not pruefe_aktualitaet(original, graph, catalog):
            raise ValueError("Index/source/catalog identity failed")
        for i, row in enumerate(report["rows"], 1):
            baseline = base_rows[row["case_id"]]
            if any(row[field] != baseline.get(field) for field in fields):
                raise ValueError("Case content or labels differ from the reference")
            result = {field: row[field] for field in fields}
            result["standard"] = baseline["standard"]
            result["stages"] = {}
            for variant, config in report["variants"].items():
                if config.get("sim_rescue") != 0 or not config.get("cascade"):
                    raise ValueError("Stage classifier only supports the declared no-rescue cascade variants")
                trace.groups, trace.variant = [], (row["dataset"], variant)
                actual = passage_suggestions(row["query"], profile=variant, laws=(), index=index,
                                             encoder=encoder, reranker=ranker, trigger=trigger,
                                             graph=graph, chains={}, catalog=catalog)
                if actual != row[variant]["suggestions"]:
                    mismatches.append({"case_id": row["case_id"], "variant": variant})
                result[variant] = {"suggestions": row[variant]["suggestions"]}
                result["stages"][variant] = stages(trace.groups, row["targets"], actual, config, classify)
                if variant in ("kaskade", "source20", "source50", "document50"):
                    context_only = variant == "document50"
                    contexts = build_context_queries(row["query"], mode=config["context"],
                                                     neighbor_radius=config.get("neighbor_radius", 1))
                    # Keep the original document run's eligibility: it skipped
                    # truncated foci before retrieving or scoring any pairs.
                    # Re-enabling those would require new inference.
                    contexts = [c for c in contexts if not context_only or not c.focus_truncated]
                    options = {"focus_specific": context_only, "context_only": context_only}
                    reconstructed = agreement_suggestions(trace.groups, contexts, None, **options)
                    if not context_only and norm_order(reconstructed) != norm_order(actual):
                        raise ValueError("Raw source-profile reconstruction differs before offline policy filtering")
                    for policy in (("weak",) if variant == "kaskade" else AGREEMENT_POLICIES):
                        name = f"{variant}_bge_{policy}"
                        suggestions = agreement_suggestions(trace.groups, contexts, policy, **options)
                        result[name] = {"suggestions": suggestions}
                        result["stages"][name] = stages(trace.groups, row["targets"], suggestions,
                            {**config, "agreement_policy": policy, "checks": False}, classify)
            if norm_order(result["source50_bge_weak"]["suggestions"]) != norm_order(result["agreement50"]["suggestions"]):
                raise ValueError("Offline source50 weak agreement differs from the live production agreement variant")
            rows.append(result)
            if i % 10 == 0:
                print(f"CPU replay {i}/{len(report['rows'])}; mismatches={len(mismatches)}", flush=True)
    finally:
        graph.close()
        original_trigger.close()
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected model runtime imported")
    datasets = {}
    for dataset in ("main", "additional"):
        selected = [row for row in rows if row["dataset"] == dataset]
        if not selected:
            continue
        variants = ["standard", *all_variants]
        comparisons = [(base, variant) for base in ("standard", "kaskade") for variant in all_variants if base != variant]
        comparisons += [("source20", "source50"), ("source20", "checked20"), ("source50", "checked50"),
                        ("source50", "agreement50"), ("checked50", "document50")]
        comparisons += [(spec["base"], name) for name, spec in offline_variants.items()]
        datasets[dataset] = {
            "summaries": {variant: metrics.summarize(selected, variant) for variant in variants},
            "changes": {f"{base}->{variant}": metrics.compare_rows([
                {**row, "before": row[base], "after": row[variant]} for row in selected])
                for base, variant in comparisons},
            "target_outcomes": {variant: dict(Counter(outcome for row in selected
                for outcome in row["stages"][variant]["target_outcomes"].values())) for variant in all_variants},
            "check_status_counts": {variant: dict(sum((Counter(row["stages"][variant]["check_status_counts"])
                for row in selected), Counter())) for variant in all_variants}}
        for variant, summary in datasets[dataset]["summaries"].items():
            if variant in report["variants"] and summary != report["summaries"][dataset][variant]:
                raise ValueError("Reported metrics differ from recomputation")
            pos, neg = summary["positive"], summary["negative"]
            print(f"{dataset} {variant}: hits={pos['hit@5'] * pos['n']:.0f}/{pos['n']} "
                  f"FP={(neg['false_suggestion_rate'] or 0) * neg['n']:.0f}/{neg['n']} "
                  f"target_coverage={pos['mean_target_coverage']:.3f} "
                  f"positive_mean_norms={pos['mean_norms']:.2f}", flush=True)
    work = {}
    for (dataset, variant), stage_work in trace.work.items():
        work.setdefault(dataset, {})[variant] = {
            stage: {**{key: value for key, value in values.items() if key != "unique"},
                    "unique_inputs_if_run_alone": len(values["unique"])}
            for stage, values in stage_work.items()}
    for dataset, variant_work in work.items():
        for name, spec in offline_variants.items():
            variant_work[name] = {stage: values for stage, values in variant_work[spec["base"]].items()
                                  if stage in ("encoder", "bge", "qwen")}
    frontier = []
    if set(datasets) == {"main", "additional"}:
        objectives = {variant: tuple(value for dataset in ("main", "additional")
            for value in (datasets[dataset]["summaries"][variant]["positive"]["hit@5"],
                          -datasets[dataset]["summaries"][variant]["negative"]["false_suggestion_rate"]))
            for variant in ["standard", *all_variants]}
        frontier = [name for name, point in objectives.items() if not any(
            all(a >= b for a, b in zip(other, point, strict=True)) and any(a > b for a, b in zip(other, point, strict=True))
            for other_name, other in objectives.items() if other_name != name)]
    result = {"format": "quality_precision_analysis/1", "status": "pass" if not mismatches else "mismatch",
              "source_run": str(directory), "source_report_sha256": digest(directory / "report.json"),
              "analysis_source_sha256": digest(Path(__file__)), "diagnostic_subset": report["diagnostic_subset"],
              "verified_hashes": hashes, "torch_imported": False, "mismatches": mismatches,
              "offline_variants": offline_variants, "observed_joint_hit_fp_pareto_frontier": frontier,
              "datasets": datasets, "workload": work, "rows": rows,
              "notes": ["Exact frozen production replay using this run's own query vectors and model scores.",
                        "Development evidence only; Gold target lists need not include all acceptable suggestions.",
                        "Stage failures count individual labeled targets, not positive cases; no labels affect replay.",
                        "Suggestion load and mean target coverage use the original sentence-based metrics.",
                        "Logical pairs and unique inputs show work demand, not token count or production latency.",
                        "Nine requested post-hoc source20/source50/document50 agreement rules, plus the fixed ranked-package kaskade weak-agreement rule; no arbitrary threshold search or Gold filtering.",
                        "The observed Pareto frontier uses Hit@5 and negative false-suggestion rate on both datasets, not blind validation.",
                        "Offline agreement variants use their base's recorded relevance pairs, with no binary checks or further inference.",
                        "Document agreement suggests context-relevant norms; a focus span does not establish that individual sentence's support.",
                        "Document agreement preserves the original run's skipped truncated foci; no missing pairs are invented.",
                        "Document workload counts retain original focus-specific calls; unique inputs describe potential within-variant caching.",
                        "Observed per-variant wall times were intentionally excluded due shared-cache order effects."]}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(bool(mismatches))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
