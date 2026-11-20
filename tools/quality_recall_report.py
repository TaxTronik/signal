"""Audit completed recall experiments and publish aggregate-only comparisons.

No decoder, model service or GPU is loaded. Frozen passage runs are replayed
in isolated CPU processes before metrics are accepted. Historical target lists
are incomplete: a non-target suggestion is unadjudicated, never implicitly wrong.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CASE_FIELDS = ("case_id", "dataset", "group", "query", "targets", "negative", "category", "uncertain")
DATASETS = {"main": (102, 30), "additional": (15, 15)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_hash(path, expected):
    require(digest(path) == expected, f"Source/input checksum differs: {path}")


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cases(rows):
    keyed = {row["case_id"]: row for row in rows}
    require(len(keyed) == len(rows) == 162, "Expected 162 distinct historical cases")
    actual = Counter((row["dataset"], row["negative"]) for row in rows)
    require(actual == Counter({(name, negative): counts[int(negative)]
                               for name, counts in DATASETS.items() for negative in (False, True)}),
            "Case denominators differ from the declared historical sets")
    require(all(type(row["negative"]) is bool and (row["negative"] or row["targets"]) for row in rows),
            "Invalid case labels")
    return keyed


def same_cases(left, right):
    require(left.keys() == right.keys() and all(
        all(left[key].get(field) == right[key].get(field) for field in CASE_FIELDS) for key in left),
        "Case content, labels or source groups differ between reports")


def variant_metrics(rows, variant):
    """Count query failures and distinct (case,norm) burden separately."""
    result = Counter()
    reciprocal, coverage = 0., 0.
    for row in rows:
        suggestions = row[variant]["suggestions"]
        ids = [norm["id"] for sentence in suggestions for norm in sentence["normen"]]
        if row["negative"]:
            result["negative_cases"] += 1
            result["negative_with_suggestion"] += bool(ids)
            result["negative_entries"] += len(ids)
            result["negative_unique_case_norm_pairs"] += len(set(ids))
            continue
        result["positive_cases"] += 1
        result["positive_entries"] += len(ids)
        result["positive_unique_case_norm_pairs"] += len(set(ids))
        targets = set(row["targets"])
        matched, ranks = set(), []
        for sentence in suggestions:
            for rank, norm in enumerate(sentence["normen"], 1):
                hits = {t for t in targets if norm["id"] == t or norm["id"].startswith(t + ":")}
                if hits:
                    matched.update(hits)
                    ranks.append(rank)
                    result["historical_target_matching_entries"] += 1
                else:
                    result["unadjudicated_non_target_entries"] += 1
        best = min(ranks) if ranks else None
        result["historical_norm_targets"] += len(targets)
        result["matched_historical_norm_targets"] += len(matched)
        result["hit1_cases"] += best == 1
        result["hit5_cases"] += best is not None and best <= 5
        reciprocal += 1 / best if best else 0.
        coverage += len(matched) / len(targets)
    names = ("positive_cases", "negative_cases", "hit1_cases", "hit5_cases", "negative_with_suggestion",
             "positive_entries", "positive_unique_case_norm_pairs", "negative_entries",
             "negative_unique_case_norm_pairs", "historical_target_matching_entries", "unadjudicated_non_target_entries",
             "historical_norm_targets", "matched_historical_norm_targets")
    n = result["positive_cases"]
    return {**{name: result[name] for name in names}, "mrr_best_sentence": reciprocal / n if n else None,
            "mean_target_coverage": coverage / n if n else None, "suggestion_precision": None,
            "full_legal_recall": None}


def paired_changes(before, after, before_variant, after_variant):
    changes = {key: [] for key in ("positive_hit_wins", "positive_hit_losses", "negative_case_wins",
                                   "negative_case_losses", "negative_norm_pairs_added", "negative_norm_pairs_removed")}
    def hit(row, variant):
        return any(norm["id"] == target or norm["id"].startswith(target + ":")
                   for sentence in row[variant]["suggestions"] for norm in sentence["normen"][:5]
                   for target in row["targets"])
    for key, row in before.items():
        other = after[key]
        if row["negative"]:
            old = {n["id"] for s in row[before_variant]["suggestions"] for n in s["normen"]}
            new = {n["id"] for s in other[after_variant]["suggestions"] for n in s["normen"]}
            if bool(old) != bool(new):
                changes["negative_case_wins" if not new else "negative_case_losses"].append(key)
            changes["negative_norm_pairs_added"].extend([key, nid] for nid in sorted(new - old))
            changes["negative_norm_pairs_removed"].extend([key, nid] for nid in sorted(old - new))
        elif hit(row, before_variant) != hit(other, after_variant):
            changes["positive_hit_wins" if hit(other, after_variant) else "positive_hit_losses"].append(key)
    return changes


def bind_report(path, report):
    live = report.get("format") == "quality_recall_live/1"
    require((report.get("status") in ("complete", "pass") or live and report.get("status") == "mismatch") and not report.get("diagnostic_subset")
            and not report.get("error"), "Only completed full reports can be evaluated")
    cases(report["rows"])
    hashes = {str(path): digest(path)}
    for name, expected in report["source_hashes"].items():
        hashes[str(path.parent / "source" / name)] = expected
    hashes.update(report.get("input_hashes", {}))
    for name, expected in report.get("query_artifacts", {}).items():
        hashes[str(path.parent / name)] = expected
    for name, value in report.get("scoring", {}).items():
        hashes[str(path.parent / f"{name}-scores.jsonl")] = value["sha256"]
    if "scores_sha256" in report:
        hashes[str(path.parent / "scores.jsonl")] = report["scores_sha256"]
    if "inputs_sha256" in report:
        hashes[str(path.parent / "inputs.json")] = report["inputs_sha256"]
    if live:
        require(report.get("source_audit", {}).get("errors") == [], "Live source evidence audit failed")
        hashes[str(path.parent / "rows.jsonl")] = report["rows_sha256"]
        for name, expected in report["frozen_inputs"].items():
            hashes[str(path.parent / name)] = expected
    hashes.update(bind_inherited_pilot(report))
    for filename, expected in hashes.items():
        check_hash(filename, expected)
    return hashes


def bind_inherited_pilot(report):
    """A stopped pilot may supply raw inputs, never full-dataset quality rows."""
    inherited = report.get("inherited_pilot")
    if inherited is None:
        return {}
    directory = Path(inherited["path"])
    cancellation_path = directory / "cancellation.json"
    require(report["input_hashes"].get(str(cancellation_path)) == digest(cancellation_path), "Unbound pilot cancellation")
    cancellation = read(cancellation_path)
    require(cancellation["format"] == "quality_pilot_cancellation/1"
            and cancellation["status"] == "interrupted_diagnostic" and cancellation["gpu_process_stopped"],
            "Pilot must be explicitly stopped and excluded from full quality evaluation")
    require(cancellation["artifact_hashes"] == inherited["artifact_hashes"]
            and cancellation["cache_pairs"] == inherited["pairs"]
            and cancellation["query_count"] == inherited["queries"], "Pilot provenance counts differ")
    hashes = {str(cancellation_path): digest(cancellation_path)}
    for name, expected in cancellation["artifact_hashes"].items():
        filename = str(directory / name)
        require(report["input_hashes"].get(filename) == expected, "Unbound inherited cache file")
        check_hash(filename, expected)
        hashes[filename] = expected
    parent = read(directory / "report.json")
    require(len(parent["rows"]) == cancellation["completed_cases"] < cancellation["planned_cases"],
            "Pilot completion count differs")
    require(parent["models"] == report["models"]
            and all(report["runtime"].get(k) == v for k, v in parent["runtime"].items()),
            "Inherited encoder/reranker revision or runtime differs")
    for name in ("risk_layer/reranker.py", "risk_layer/embedding.py", "risk_layer/embedding_model.py", "risk_layer/torch_compat.py"):
        require(parent["source_hashes"][name] == report["source_hashes"][name], "Inherited model implementation differs")
    for name, expected in parent["source_hashes"].items():
        hashes[str(directory / "source" / name)] = expected
    for filename, expected in parent["input_hashes"].items():
        require(report["input_hashes"].get(filename) == expected, "Inherited corpus/reference differs")
        hashes[filename] = expected
    for filename, expected in hashes.items():
        check_hash(filename, expected)
    return hashes


def key_digest(key):
    return hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def score_workload(cache, used, inherited_cache_path=None):
    """Count actual replay use, distinguishing imported but unused pilot keys."""
    inherited = set()
    if inherited_cache_path is not None:
        with Path(inherited_cache_path).open(encoding="utf-8") as stream:
            for line in stream:
                *values, score = json.loads(line)
                key = tuple(values)
                require(key in cache and cache[key] == score, "Inherited exact input score changed")
                inherited.add(key_digest(key))
    cache_keys = {key_digest(key) for key in cache}
    require(used <= cache_keys, "Replay used absent score inputs")
    fresh = cache_keys - inherited
    require(fresh <= used, "Fresh scores include inputs not used by the completed runtime")
    return {"pairs": len(cache_keys), "inherited_pairs": len(inherited), "fresh_pairs": len(fresh),
            "used_pairs": len(used), "inherited_used_pairs": len(used & inherited),
            "fresh_used_pairs": len(used & fresh), "inherited_unused_pairs": len(inherited - used)}


def validate_summaries(report, names):
    for dataset in DATASETS:
        selected = [row for row in report["rows"] if row["dataset"] == dataset]
        for name in names:
            actual = variant_metrics(selected, name)
            expected = report["summaries"][dataset][name]
            positive, negative = expected["positive"], expected["negative"]
            require(positive["n"] == actual["positive_cases"] and negative["n"] == actual["negative_cases"],
                    "Stored metric denominators differ")
            values = [(positive["hit@1"] * positive["n"], actual["hit1_cases"]),
                      (positive["hit@5"] * positive["n"], actual["hit5_cases"]),
                      (positive["mean_norms"] * positive["n"], actual["positive_entries"]),
                      (negative["mean_norms"] * negative["n"], actual["negative_entries"]),
                      (negative["false_suggestion_rate"] * negative["n"], actual["negative_with_suggestion"]),
                      (positive["mrr_best_sentence"], actual["mrr_best_sentence"]),
                      (positive["mean_target_coverage"], actual["mean_target_coverage"])]
            require(all(math.isclose(a, b, abs_tol=1e-10) for a, b in values), "Stored metrics differ from actual output rows")


def public_payload_check(value):
    if isinstance(value, dict):
        require(not {"rows", "query", "document", "case_id", "case_ids", "targets", "suggestions",
                     "passagen", "source_quote", "norm_quote", "raw_response"}.intersection(value),
                "Individual case payload escaped aggregate projection")
        for child in value.values():
            public_payload_check(child)
    elif isinstance(value, list):
        for child in value:
            public_payload_check(child)


def score_cache(path):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            instruction, budget, query, document, score = json.loads(line)
            require(type(score) in (int, float) and math.isfinite(score), "Invalid raw model score")
            require(instruction is None or 0 <= score <= 1, "Invalid instruction score")
            key = (instruction, budget, query, document)
            require(key not in result or result[key] == score, "Conflicting exact score inputs")
            result[key] = score
    return result


def source_evidence_audit(suggestions, index, graph):
    parents = {p.passage_id: p for p in index.passage_evidence.passages}
    count = 0
    for sentence in suggestions:
        for norm in sentence["normen"]:
            evidence = norm["evidenz"]
            require(evidence["fingerprint"] == index.passage_evidence.source_fingerprint,
                    "Output evidence generation differs")
            require(bool(evidence["passagen"]), "Output has no source evidence")
            for passage in evidence["passagen"]:
                parent = parents[passage["id"]]
                start, end = passage["quell_start"], passage["quell_end"]
                source = graph.node(norm["id"])["text"]
                require(parent.norm_id == passage["norm_id"] == norm["id"]
                        and parent.start <= start < end <= parent.end
                        and source[start:end] == passage["text"]
                        and parent.text[start - parent.start:end - parent.start] == passage["text"],
                        "Output passage is not an exact original source/parent slice")
                count += 1
    return count


def audit_live(path, report, hashes):
    """Validate actual GPU outputs and sources; do not replay models on CPU."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus")
    sys.path.insert(0, str(path.parent / "source"))
    from tools import quality_recall_live as live
    from tools import _gpu_eval_metrics as metrics
    from risk_layer.api_v1 import _SEMANTIK_PROFILE
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.idschema import para_level
    from risk_layer.semantic_search import PROFILE_CONFIG
    from risk_layer.zitate import parse_normzitat
    import risk_layer.semantic_search as runtime
    for module in (live, metrics, runtime):
        require(Path(module.__file__).resolve().is_relative_to(path.parent / "source"), "Live audit imported unfrozen implementation")
    reference = read(path.parent / "inputs/structural-reference.json")
    variant, profile = report["selected_variant"], report["public_profile"]
    live.validate_reference(reference, variant)
    require(digest(path.parent / "inputs/structural-reference.json") == report["reference_report_sha256"], "Live reference hash differs")
    require(report["profile_config"] == reference["variants"][variant], "Live/structural configuration differs")
    live.validate_public_profile(profile, report["profile_config"], _SEMANTIK_PROFILE, PROFILE_CONFIG)
    expected_cases = cases(reference["rows"])
    same_cases(cases(report["rows"]), expected_cases)
    hard = read(path.parent / "inputs/hard-mark-reference.json")
    require(digest(path.parent / "inputs/hard-mark-reference.json") in reference["input_hashes"].values(), "Unbound live hard-mark reference")
    hard_cases = cases(hard["rows"])
    same_cases(cases(report["rows"]), hard_cases)
    require(set(report["models"]) == {"encoder", "bge", "qwen"}, "Live model set differs")
    for name, model in report["models"].items():
        require(all(model[key] == reference["models"][name][key] for key in ("name", "revision")), "Live model pin differs")
        require(model["devices"] and all(device.startswith("cuda:") for device in model["devices"]), "Live model was not fully on GPU")
        require(model["dtypes"] == (["torch.float32"] if name == "encoder" else ["torch.float16"]), "Live model dtype differs")
    def resource(name):
        found = [Path(filename) for filename in reference["input_hashes"] if Path(filename).name == name]
        require(len(found) == 1, f"Ambiguous live resource: {name}")
        return found[0]
    index = EmbeddingIndex.load(resource("meta.json").parent)
    graph = NormGraph(resource("graph.sqlite"))
    catalog = lade_katalog(path.parent / "source/catalog/begriffe.yaml")
    catalog_chains = {}
    for entry in catalog:
        for chain in entry.normketten:
            for anchor in parse_normzitat(chain.primaer):
                if anchor.norm_id:
                    catalog_chains.setdefault(para_level(anchor.norm_id.to_id()), []).append(chain.as_dict())
                    break
    metadata_counts, changed_chain_entries, verified_chain_entries = Counter(), 0, 0
    snippets, differences = 0, []
    try:
        require(index is not None and pruefe_aktualitaet(index, graph, catalog), "Live index/catalog/graph no longer coherent")
        for row in report["rows"]:
            actual = row[profile]
            raw = row["raw_pipeline_report"]
            require(raw["semantik_vorschlaege"] == actual["suggestions"], "Projected output differs from raw Pipeline.run")
            marks = {"cards": [{"id": c["eintrag"]["id"], "via": c["via"], "span": [c["treffer"]["start"], c["treffer"]["end"]]}
                               for c in raw["karten"]],
                     "trigger_marks": [{"span": [r["start"], r["end"]], "title": r["titel"], "norm_ids": [n["id"] for n in r["norm_vorschlag"]]}
                                       for r in raw["risiken"] if r["quelle"] == "trigger"]}
            require(all(actual[key] == value for key, value in marks.items()), "Projected hard marks differ from raw Pipeline.run")
            expected = {"suggestions": live.renamed_reference(expected_cases[row["case_id"]][variant]["suggestions"], variant, profile),
                        **{key: hard_cases[row["case_id"]]["standard"][key] for key in ("cards", "trigger_marks")}}
            compared = live.compare(actual, expected, row["targets"], metrics.evaluate_variant)
            require(compared == actual["comparison"] and actual["metrics"] == compared["actual_metrics"], "Stored live comparison/metrics differ")
            metadata = live_suggestion_difference(actual["suggestions"], expected["suggestions"])
            metadata_counts[metadata["classification"]] += 1
            changed_chain_entries += metadata["changed_chain_entries"]
            for sentence in actual["suggestions"]:
                for norm in sentence["normen"]:
                    require(norm.get("ketten") == catalog_chains.get(norm["id"], []), "Live chain metadata differs from frozen catalog")
                    verified_chain_entries += bool(norm["ketten"])
            exact = source_evidence_audit(actual["suggestions"], index, graph)
            require(actual["source_audit"] == {"status": "pass", "exact_passage_spans": exact}, "Stored live source audit differs")
            snippets += exact
            elapsed = actual["elapsed_seconds"]
            require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, "Invalid actual live timing")
            if not compared["all_equal"]:
                differences.append({"case_id": row["case_id"], "dataset": row["dataset"],
                                    "categories": compared["categories"], "comparison": compared})
        require(differences == report["mismatches"], "Live difference accounting differs")
        require(report["status"] == ("mismatch" if differences else "pass"), "Live status hides differences")
        require(report["source_audit"] == {"exact_passage_spans": snippets, "errors": []}, "Live source denominator differs")
        validate_summaries(report, [profile])
        require(report["reference_summaries"] == {dataset: {variant: metrics.summarize([
            row for row in reference["rows"] if row["dataset"] == dataset], variant)} for dataset in DATASETS}, "Live reference summaries differ")
        timings = {dataset: live.helper.timing_summary([row[profile]["elapsed_seconds"] for row in report["rows"]
            if dataset == "all" or row["dataset"] == dataset]) for dataset in ("all", *DATASETS)}
        require(timings == report["timings"], "Stored live timings differ from actual cases")
        require(report["warmup"]["excluded_from_case_timings"] is True
                and report["warmup"]["case_id"] == report["rows"][0]["case_id"], "Warmup is not separately identified")
        require(report["difference_categories"] == dict(Counter(c for d in differences for c in d["categories"])), "Live category totals differ")
    finally:
        graph.close()
    require("torch" not in sys.modules, "Live output audit imported a model runtime")
    for filename, expected in hashes.items():
        check_hash(filename, expected)
    return {"status": "pass", "report_sha256": digest(path), "torch_imported": False,
            "actual_profile_outputs": len(report["rows"]), "exact_source_passage_spans": snippets,
            "reference_differences": len(differences), "bound_files": len(hashes),
            "metadata_difference_counts": {"case_counts": dict(metadata_counts),
                "changed_chain_entries_in_aligned_outputs": changed_chain_entries,
                "catalog_chain_norm_entries_verified": verified_chain_entries,
                "reference_chain_metadata_empty": all(not norm.get("ketten") for row in reference["rows"]
                    for sentence in row[variant]["suggestions"] for norm in sentence["normen"])},
            "note": "Stored actual GPU output, comparisons and source intervals audited; models were not replayed on CPU."}


def live_suggestion_difference(actual, expected):
    """Supplement the frozen broad categories; never rewrite their raw values."""
    def identity(suggestions):
        return [(s["start"], s["end"], [n["id"] for n in s["normen"]]) for s in suggestions]
    if identity(actual) != identity(expected):
        return {"classification": "unaligned_suggestions", "changed_chain_entries": 0}
    missing = object()
    chain_changes = sum(a.get("ketten", missing) != b.get("ketten", missing)
                        for left, right in zip(actual, expected, strict=True)
                        for a, b in zip(left["normen"], right["normen"], strict=True))
    def stripped(suggestions, fields):
        return [{**s, "normen": [{key: value for key, value in n.items() if key not in fields} for n in s["normen"]]}
                for s in suggestions]
    if actual == expected:
        classification = "identical_suggestions"
    elif chain_changes and stripped(actual, {"ketten"}) == stripped(expected, {"ketten"}):
        classification = "chain_metadata_only"
    elif stripped(actual, {"sim", "rerank"}) == stripped(expected, {"sim", "rerank"}):
        classification = "score_only"
    elif chain_changes and stripped(actual, {"ketten", "sim", "rerank"}) == stripped(expected, {"ketten", "sim", "rerank"}):
        classification = "chain_metadata_and_scores_only"
    else:
        classification = "other_suggestion_difference"
    return {"classification": classification, "changed_chain_entries": chain_changes}


def audit_one(path):
    """Called in a fresh interpreter so only that report's runtime is imported."""
    report = read(path)
    hashes = bind_report(path, report)
    kind = report["format"]
    if kind == "quality_recall_live/1":
        return audit_live(path, report, hashes)
    require(kind in ("quality_instruction_probe/1", "quality_passage_rerank_probe/1"), "Unknown new run schema")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENBLAS_NUM_THREADS="1",
                      OMP_NUM_THREADS="1", RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus")
    if kind == "quality_instruction_probe/1":
        inputs = read(path.parent / "inputs.json")
        require(inputs["format"] == "quality_recall_inputs/1" and inputs["status"] == "complete", "Invalid frozen inputs")
        hashes.update(inputs["source_hashes"])
        hashes.update(inputs["input_hashes"])
        runtime_root = Path(inputs["source_run"])
        input_paths = list(map(Path, inputs["input_hashes"]))
        same_cases(cases(report["rows"]), cases(inputs["cases"]))
        require(report["model"]["revision"] == inputs["models"]["qwen"]["revision"], "Qwen revision differs")
    else:
        runtime_root = path.parent
        input_paths = list(map(Path, report["input_hashes"]))
        require(digest(path.parent / "reference.json") in report["input_hashes"].values(), "Unbound case reference")
        same_cases(cases(report["rows"]), cases(read(path.parent / "reference.json")["rows"]))
    for filename, expected in hashes.items():
        check_hash(filename, expected)
    sys.path.insert(0, str(runtime_root / "source"))
    import numpy as np
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.triggerindex import TriggerIndex
    from risk_layer.semantic_search import passage_suggestions, PROFILE_CONFIG
    import risk_layer.semantic_search as runtime
    require(Path(runtime.__file__).resolve().is_relative_to(runtime_root / "source"), "Unexpected runtime source imported")
    index = EmbeddingIndex.load(next(p.parent for p in input_paths if p.name == "meta.json"))
    graph = NormGraph(next(p for p in input_paths if p.name == "graph.sqlite"))
    catalog = lade_katalog(runtime_root / "source/catalog/begriffe.yaml")
    require(index is not None and pruefe_aktualitaet(index, graph, catalog), "Incoherent source index")
    snippets, output_count, package_count = 0, 0, 0
    workload = None
    try:
        if kind == "quality_instruction_probe/1":
            require(report["threshold"] == .5 and report["topk"] == 5 and report["shortlist"] in (20, 50),
                    "Undeclared instruction output rule")
            cached = score_cache(path.parent / "scores.jsonl")
            texts = read(runtime_root / "queries.texts.json")
            vectors = np.load(runtime_root / "queries.npy", allow_pickle=False)
            by_text = dict(zip(texts, vectors, strict=True))
            indexed_cases = cases(inputs["cases"])
            for row in report["rows"]:
                case = indexed_cases[row["case_id"]]
                for name, instruction in report["instructions"].items():
                    actual = []
                    for group in case["groups"]:
                        require(case["query"][group["start"]:group["end"]] == group["query"], "Invalid original focus")
                        candidates = group["candidates"][:report["shortlist"]]
                        eligible = []
                        for candidate in candidates:
                            package = index.norm_package(candidate["id"], by_text[group["context_query"]])
                            require(package is not None and package.text == candidate["document"], "Scored package differs from original source construction")
                            package_count += 1
                            value = cached[(instruction, report["max_length"], group["query"], candidate["document"])]
                            if value >= .5:
                                eligible.append({"id": candidate["id"], "sim": candidate["sim"], "rerank": value})
                        eligible.sort(key=lambda n: (-n["rerank"], -(n["sim"] if n["sim"] is not None else -math.inf), n["id"]))
                        if eligible:
                            actual.append({"start": group["start"], "end": group["end"], "normen": eligible[:5]})
                    require(actual == row[name]["suggestions"], "Instruction output differs from exact score/input replay")
                    output_count += 1
        else:
            class Encoder:
                def __init__(self):
                    self.name, self.revision = (report["models"]["encoder"][k] for k in ("name", "revision"))
                    vectors = np.load(path.parent / "queries.npy", allow_pickle=False)
                    self.dim = vectors.shape[1]
                    self.cache = dict(zip(read(path.parent / "queries.texts.json"), vectors, strict=True))
                    self.used = set()
                def encode(self, texts):
                    self.used.update(texts)
                    return np.stack([self.cache[text] for text in texts])
            class Ranker:
                def __init__(self, name):
                    self.name, self.revision = (report["models"][name][k] for k in ("name", "revision"))
                    self.cache = score_cache(path.parent / f"{name}-scores.jsonl")
                    self.used = set()
                def score(self, pairs):
                    keys = [(None, 1024, query, document) for query, document in pairs]
                    self.used.update(map(key_digest, keys))
                    return [self.cache[key] for key in keys]
                def score_with_instruction(self, pairs, instruction, *, max_length):
                    keys = [(instruction, max_length, query, document) for query, document in pairs]
                    self.used.update(map(key_digest, keys))
                    return [self.cache[key] for key in keys]
            encoder, ranker = Encoder(), Ranker("qwen")
            ranker.vorauswahl = Ranker("bge")
            trigger = TriggerIndex(next(p for p in input_paths if p.name == "trigger.sqlite"))
            PROFILE_CONFIG.update(report["variants"])
            try:
                for row in report["rows"]:
                    for name in report["variants"]:
                        actual = passage_suggestions(row["query"], profile=name, laws=(), index=index,
                            encoder=encoder, reranker=ranker, trigger=trigger, graph=graph, chains={}, catalog=catalog)
                        require(actual == row[name]["suggestions"], "Frozen passage runtime differs from archived output")
                        snippets += source_evidence_audit(actual, index, graph)
                        output_count += 1
                parent_dir = Path(report["inherited_pilot"]["path"]) if report.get("inherited_pilot") else None
                workload = {}
                for name, model in (("bge", ranker.vorauswahl), ("qwen", ranker)):
                    counts = score_workload(model.cache, model.used, parent_dir / f"{name}-scores.jsonl" if parent_dir else None)
                    expected = report["scoring"][name]
                    require(all(expected.get(key) == counts[key] for key in ("pairs", "inherited_pairs", "fresh_pairs", "used_pairs", "inherited_used_pairs") if key in expected),
                            "Stored fresh/inherited/used score accounting differs from actual runtime")
                    if parent_dir:
                        require(counts["inherited_pairs"] == report["inherited_pilot"]["pairs"][name], "Pilot raw cache count differs")
                    workload[name] = counts
                parent_texts = set(read(parent_dir / "queries.texts.json")) if parent_dir else set()
                if parent_dir:
                    parent_vectors = np.load(parent_dir / "queries.npy", allow_pickle=False)
                    for text, vector in zip(read(parent_dir / "queries.texts.json"), parent_vectors, strict=True):
                        require(text in encoder.cache and np.array_equal(vector, encoder.cache[text]), "Inherited query vector changed")
                new_texts = set(encoder.cache) - parent_texts
                require(new_texts <= encoder.used, "Unused newly encoded query")
                workload["encoder"] = {"queries": len(encoder.cache), "inherited_queries": len(parent_texts),
                    "fresh_queries": len(new_texts), "used_queries": len(encoder.used),
                    "inherited_used_queries": len(encoder.used & parent_texts)}
            finally:
                trigger.close()
    finally:
        graph.close()
    require("torch" not in sys.modules, "Audit unexpectedly imported a model runtime")
    for filename, expected in hashes.items():
        check_hash(filename, expected)
    return {"status": "pass", "report_sha256": digest(path), "torch_imported": False,
            "exact_profile_outputs": output_count, "exact_source_passage_spans": snippets,
            "exact_source_packages": package_count, "bound_files": len(hashes), "mismatches": 0,
            "actual_cache_workload": workload}


def classifier_aggregate(path, baseline_path, base_cases):
    """Project the rejected fixed OOF gate to public counts, retaining no cases."""
    report = read(path)
    require(report.get("format") == "query_classifier_purged_logo/1"
            and report.get("status") == "complete" and not report.get("torch_imported")
            and report.get("model_saved") is False, "Expected complete undeployed CPU classifier diagnostic")
    protocol_path = Path(report["protocol_path"])
    check_hash(protocol_path, report["protocol_sha256"])
    protocol = read(protocol_path)
    require(protocol["status"] == "frozen_before_fit" and protocol["source_sha256"] == report["source_sha256"],
            "Classifier protocol/source mismatch")
    check_hash(ROOT / "out/evaluate_query_classifier.py", report["source_sha256"])
    hashes = protocol["input_hashes"]
    require(hashes.get(str(baseline_path.resolve())) == digest(baseline_path), "Classifier baseline differs")
    for filename, expected in hashes.items():
        check_hash(filename, expected)
    model = protocol["model"]
    require(model["C"] == 1.0 and model["class_weight"] == "balanced" and model["threshold"] == 0.5,
            "Classifier fixed model changed")
    predictions = report["rows"]
    ids = protocol["case_order"]
    require(len(ids) == len(set(ids)) == len(predictions) == len(base_cases)
            and set(ids) == set(base_cases) and [r["case_id"] for r in predictions] == ids,
            "Classifier OOF case coverage differs")
    folds = protocol["folds"]
    require(sorted(i for f in folds for i in f["test"]) == list(range(len(ids)))
            and report["model_fits"] == len(folds), "Classifier missing OOF fold")
    edge_ids = {frozenset((e["a"], e["b"])) for e in protocol["purge_edges"]}
    for fold in folds:
        train, test = set(fold["train"]), set(fold["test"])
        require(fold["binary_fit_valid"] and not train & test
                and len({base_cases[ids[i]]["negative"] for i in train}) == 2,
                "Classifier invalid training fold")
        expected_test = {i for i, row in enumerate(predictions) if row["held_out_source"] == fold["held_out_source"]}
        require(test == expected_test and all(frozenset((ids[i], ids[j])) not in edge_ids for i in train for j in test),
                "Classifier source split or declared purge leakage")
    gated = []
    for prediction in predictions:
        original = base_cases[prediction["case_id"]]
        score = prediction["probability_need"]
        require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1
                and prediction["predicted_need"] == (score >= .5)
                and all(prediction[k] == original[k] for k in ("dataset", "negative")),
                "Classifier prediction/label contract differs")
        gated.append({**original, "need_gate": {"suggestions": original["kaskade"]["suggestions"] if score >= .5 else []}})
    datasets = {}
    for dataset in DATASETS:
        rows = [r for r in gated if r["dataset"] == dataset]
        before, after = (variant_metrics(rows, variant) for variant in ("kaskade", "need_gate"))
        for variant, actual in (("kaskade", before), ("kaskade_with_oof_need_gate", after)):
            expected = report["summaries"][dataset][variant]
            require(math.isclose(actual["hit5_cases"] / actual["positive_cases"], expected["positive"]["hit@5"])
                    and math.isclose(actual["negative_with_suggestion"] / actual["negative_cases"],
                                     expected["negative"]["false_suggestion_rate"]), "Classifier gated counts differ")
        datasets[dataset] = {"before": before, "after": after}
    return {"decision": "rejected_for_runtime", "report_sha256": digest(path),
            "protocol_sha256": report["protocol_sha256"], "source_sha256": report["source_sha256"],
            "model": model, "source_folds": len(folds), "cases": len(ids),
            "training_exclusions_across_folds": sum(len(f["purged"]) for f in folds),
            "model_saved": False, "datasets": datasets,
            "limitations": ["Already inspected development data; no independent blind validation.",
                            "Only two source collections contain negative cases; source style and query length are confounded.",
                            "Whole-source OOF, with query-only near variants removed from training before each fixed fit.",
                            "The gate removes genuine positive short queries and is not activated."]}


def render(report):
    lines = ["# Recall- und Fehlvorschlagsvergleich", "",
        "Alle 162 Fälle sind bereits bekannte Entwicklungsdaten. Die historischen Ziellisten sind "
        "unvollständig; daraus wird keine Präzision einzelner Normvorschläge berechnet.", "",
        "Negativfälle mit Ausgabe, die Zahl aller negativen Einträge und eindeutige (Fall, Norm)-Paare "
        "werden getrennt gezählt. Mehr Vorschläge in einem bereits betroffenen Fall erhöhen die Fallquote nicht.", "",
        "| Lauf / Variante | Haupt-Hit@5 | Haupt-Negativfälle | Haupt-negative Einträge / eindeutige Paare | Zusatz-Hit@5 | Zusatz-Negativfälle | Zusatz-negative Einträge / eindeutige Paare |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["variants"].items():
        main, extra = value["datasets"]["main"], value["datasets"]["additional"]
        lines.append(f"| {name} | {main['hit5_cases']}/{main['positive_cases']} "
            f"| {main['negative_with_suggestion']}/{main['negative_cases']} "
            f"| {main['negative_entries']} / {main['negative_unique_case_norm_pairs']} "
            f"| {extra['hit5_cases']}/{extra['positive_cases']} "
            f"| {extra['negative_with_suggestion']}/{extra['negative_cases']} "
            f"| {extra['negative_entries']} / {extra['negative_unique_case_norm_pairs']} |")
    coverage_values = [(name, value) for name, value in report["variants"].items()
                       if all({"historical_norm_targets", "matched_historical_norm_targets"} <= set(value["datasets"][dataset])
                              for dataset in DATASETS)]
    if coverage_values:
        lines += ["", "Zielnorm-Abdeckung: Jede historische Zielnorm zählt pro positivem Fall einmal, "
            "auch bei mehreren Fokusstellen. Dieselbe Zielnorm in einem weiteren Fall zählt dort erneut. "
            "Unbewertete zusätzliche Vorschläge erhöhen diese Abdeckung nicht. Der Nenner enthält "
            "ausschließlich die unvollständigen historischen Ziellisten.", "",
            "| Lauf / Variante | Haupt-Zielnormen erreicht / gesamt | Zusatz-Zielnormen erreicht / gesamt |",
            "|---|---:|---:|"]
        for name, value in coverage_values:
            main, extra = value["datasets"]["main"], value["datasets"]["additional"]
            lines.append(f"| {name} | {main['matched_historical_norm_targets']}/{main['historical_norm_targets']} "
                         f"| {extra['matched_historical_norm_targets']}/{extra['historical_norm_targets']} |")
    lines += ["", "Alle deklarierten Varianten der angegebenen vollständigen Läufe sind enthalten. "
        "Schwellen wurden in dieser Auswertung nicht ausgewählt. Die deutsche Kontrolle verwendet "
        "das verlängerte Eingabebudget des Instruktionstests; sie ist vom ursprünglichen Kaskadenpfad getrennt.", "",
        "Die Passagenläufe werden vor Veröffentlichung mit ihrem eingefrorenen Runtimecode und genau "
        "ihren gespeicherten Modellinputs auf CPU wiederholt. Ausgaben müssen vollständig übereinstimmen; "
        "alle ausgegebenen Quellenintervalle müssen zum Originalgraphen und ihrem Elternfenster passen. "
        "Instruktionsausgaben werden aus ihrem Scorecache rekonstruiert; die Normpakete werden erneut "
        "aus dem gebundenen Originalindex aufgebaut. Diese Kontrollen sind Implementierungsnachweise, keine neuen Fachlabels.", ""]
    if report.get("live_result"):
        live = report["live_result"]
        lines += ["Die oberste Tabellenzeile zeigt die aktuelle öffentliche Pipeline im neuen GPU-Lauf. "
            f"Alle {live['cases']} Fälle wurden mit echten Encoder- und Rerankeraufrufen geprüft. "
            f"Der Vergleich zur gewählten Strukturreferenz enthält {live['reference_differences']} "
            "abweichende Ausgaben; die Tabelle zählt die tatsächlich ausgegebenen Vorschläge.", ""]
        if live["difference_categories"]:
            lines += ["Abweichungskategorien (können sich überlappen): " + ", ".join(
                f"{name}: {count}" for name, count in live["difference_categories"].items()) + ".", ""]
        if live.get("metadata_difference_counts"):
            detail = live["metadata_difference_counts"]
            counts = detail["case_counts"]
            explanation = ("Die Strukturreferenz übergab eine leere Kettenliste; die tatsächliche Pipeline ergänzt die "
                "im eingefrorenen Katalog hinterlegten Normketten. " if detail.get("reference_chain_metadata_empty") else "")
            lines += ["Zusätzliche Aufschlüsselung der Vorschlagsmetadaten: "
                f"{counts.get('chain_metadata_only', 0)} Fälle unterscheiden sich ausschließlich durch Katalog-Normketten, "
                f"{counts.get('chain_metadata_and_scores_only', 0)} durch diese Normketten und Scores. "
                + explanation + "Die Zuordnung der Live-Normketten wurde gegen den eingefrorenen Katalog geprüft. "
                f"Sonstige Unterschiede der Vorschläge: {counts.get('other_suggestion_difference', 0)} Fälle; "
                f"abweichende Fokus-/Normanordnung: {counts.get('unaligned_suggestions', 0)} Fälle. "
                "Die ursprünglichen Abweichungskategorien bleiben unverändert erhalten.", ""]
        lines += ["Nur die folgenden Zeiten stammen aus diesem neuen Live-Lauf. Modellladen und "
            "ein separater Aufwärmlauf sind ausgeschlossen; jede Fallzeit ist mit der GPU synchronisiert. "
            "NER und Lemma sind ausgeschaltet; dies ist keine Messung der vollständigen Serverlatenz.", "",
            "| Live-Datensatz | Fälle | Mittel (s) | Median (s) | P95 (s) |",
            "|---|---:|---:|---:|---:|"]
        for dataset, timing in live["timings"].items():
            lines.append(f"| {dataset} | {timing['n']} | {timing['mean_seconds']:.3f} "
                         f"| {timing['median_seconds']:.3f} | {timing['p95_seconds']:.3f} |")
        lines += [""]
    for label, run in report.get("reports", {}).items():
        pilot = run.get("inherited_pilot")
        if pilot:
            lines += [f"{label} übernimmt exakt gebundene Modellinputs aus einem nach "
                f"{pilot['completed_cases']}/{pilot['planned_cases']} Fällen abgebrochenen Pilotlauf. "
                "Dieser Pilot liefert keine vollständige Qualitätskennzahl. Der korrigierte Lauf "
                "verwendet seine Scores nur bei identischer Modellrevision, Modellimplementierung, "
                "Runtime und identischem Query-/Dokument-/Instruktions-/Budgetschlüssel.", ""]
        workload = (run.get("audit") or {}).get("actual_cache_workload")
        if workload:
            lines += [f"Cache-Arbeit in {label} (eindeutige Modellinputs):", "",
                "| Modell | Übernommen | Neu berechnet | Tatsächlich verwendet | Davon übernommen | Übernommen, unbenutzt |",
                "|---|---:|---:|---:|---:|---:|"]
            for name in ("bge", "qwen"):
                counts = workload[name]
                lines.append(f"| {name} | {counts['inherited_pairs']} | {counts['fresh_pairs']} "
                    f"| {counts['used_pairs']} | {counts['inherited_used_pairs']} | {counts['inherited_unused_pairs']} |")
            lines += ["", "Diese Zählung beschreibt gemeinsam gecachte Experimente, keine frische "
                "Inferenz jedes einzelnen Profils und keine vergleichbare Laufzeitmessung. Das BGE-Modell "
                "verwendet sein 8192-Token-Budget; die Zahl 1024 im historischen Cache-Schlüssel benennt "
                "keine tatsächliche BGE-Eingabegrenze. Qwen-Instruktionstests verwenden 2048 Token.", ""]
    if "query_need_classifier" in report:
        classifier = report["query_need_classifier"]
        lines += ["Ein zusätzliches festes logistisches Bedarfsmodell (BGE-Anfragevektoren, C=1, "
            "ausgeglichene Klassengewichte, Schwelle 0,5) wurde über elf vollständige Quellgruppen geprüft. "
            "Nahe Varianten wurden vor jedem Fit ausschließlich aus dem Training entfernt. Auch dieser "
            "Versuch verwendet bekannte Entwicklungsdaten mit nur zwei Negativquellen.", ""]
        parts = []
        for dataset, label in (("main", "Hauptdaten"), ("additional", "Zusatzdaten")):
            before, after = (classifier["datasets"][dataset][key] for key in ("before", "after"))
            parts.append(f"{label}: Treffer {before['hit5_cases']}→{after['hit5_cases']}/{before['positive_cases']}, "
                         f"Negativfälle mit Ausgabe {before['negative_with_suggestion']}→{after['negative_with_suggestion']}/{before['negative_cases']}")
        lines += ["; ".join(parts) + ". Das Modell wurde für die Laufzeit verworfen: "
                  "Es unterdrückt auch echte kurze Problembeschreibungen. Es wurde keine Modelldatei "
                  "gespeichert oder aktiviert.", ""]
    if report.get("instruction_mean"):
        mean = report["instruction_mean"]
        parts = []
        for dataset, label in (("main", "Hauptdaten"), ("additional", "Zusatzdaten")):
            value = mean["datasets"][dataset]
            parts.append(f"{label}: {value['hit5_cases']}/{value['positive_cases']} Treffer, "
                         f"{value['negative_with_suggestion']}/{value['negative_cases']} Negativfälle mit Ausgabe, "
                         f"{value['negative_unique_case_norm_pairs']} eindeutige negative Normpaare")
        lines += ["Ein separater nachträglicher Entwicklungsversuch mittelt die drei englischen "
            "Instruktionsscores arithmetisch; Top20, Schwelle 0,5 und Top5 bleiben fest. "
            "Es wurden keine Gewichte oder Schwellen ausgewählt und keine neuen Modellaufrufe durchgeführt. "
            + "; ".join(parts) + ". Ein Einsatz bräuchte alle drei Instruktionsauswertungen. "
            "Der Fehlerkompromiss rechtfertigt keine Aktivierung. "
            "Die drei Ansichten desselben Modells sind keine unabhängige Validierung.", ""]
    if report.get("qwen8b"):
        experiment = report["qwen8b"]
        if experiment.get("recovery"):
            recovery = experiment["recovery"]
            lines += ["Der 8B-Vergleich wurde aus einem fehlgeschlagenen Ursprungslauf wiederhergestellt. "
                f"Nach allen {recovery['complete_inherited_decoder_pairs']} Decoderpaaren meldete der Prozess "
                f"noch {recovery['original_post_decoder_allocated_bytes']:,} belegte GPU-Bytes; der feste "
                f"Freigabecheck erlaubte höchstens {recovery['original_release_guard_bytes']:,}. "
                "Das unveränderte Originalarchiv bleibt als fehlgeschlagen dokumentiert. "
                "Ein neuer Prozess prüfte ausschließlich den vollständigen Original-Ausgabekopf auf allen "
                "gespeicherten Hidden States und denselben Batchgrenzen. Der Decoder wurde nicht erneut ausgeführt. "
                "Die Wiederaufnahme ist getrennt gehasht und stellt keinen erfolgreichen ursprünglichen Gesamtlauf dar.", ""]
        lines += ["Separater Qwen3-Reranker-8B-Versuch: Alle vier festen Kombinationen aus "
            "BGE-Top20/Top50 und reduziertem/vollständigem Original-Ausgabekopf werden gezeigt. "
            "Die unveränderten Normpakete und Fälle stammen aus der ursprünglichen Kaskade. "
            "Schwelle 0,5 und maximal fünf Vorschläge pro Fokus bleiben fest.", "",
            "| 8B-Variante | Haupt-Treffer | Haupt-Negativfälle | Haupt-negative Normpaare | Haupt-Zielnormen | Zusatz-Treffer | Zusatz-Negativfälle | Zusatz-negative Normpaare | Zusatz-Zielnormen |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, value in experiment["variants"].items():
            main, extra = value["datasets"]["main"], value["datasets"]["additional"]
            lines.append(f"| {name} | {main['hit5_cases']}/{main['positive_cases']} "
                f"| {main['negative_with_suggestion']}/{main['negative_cases']} | {main['negative_unique_case_norm_pairs']} "
                f"| {main['matched_historical_norm_targets']}/{main['historical_norm_targets']} "
                f"| {extra['hit5_cases']}/{extra['positive_cases']} "
                f"| {extra['negative_with_suggestion']}/{extra['negative_cases']} | {extra['negative_unique_case_norm_pairs']} "
                f"| {extra['matched_historical_norm_targets']}/{extra['historical_norm_targets']} |")
        numerical = experiment["numerical_audit"]
        changed_orders = sum(counts["different_norm_orders"] for dataset in experiment["output_difference_counts"].values()
                             for counts in dataset.values())
        lines += ["", f"Beide Köpfe projizieren dieselben {numerical['pairs']} tatsächlich beobachteten "
            "Hidden States mit denselben Batchgrenzen. Der vollständige Kopf berechnet das gesamte Originalvokabular "
            "auf der GPU. Dies ist keine zweite Decoderinferenz und kein unabhängiges Modellvotum. "
            f"Unterschiedliche Rohscores: {numerical['different_scores']}; Übertritte der 0,5-Schwelle: "
            f"{numerical['threshold_crossings']}; abweichende Normlisten über beide Shortlists: {changed_orders}. "
            "Kleine FP16-GEMM-Abweichungen bleiben sichtbar. Die Messung beschreibt einen eigenständigen "
            "8B-Versuch; sie belegt weder Produktionslaufzeit noch gleichzeitigen Speicherbedarf dreier Modelle. "
            "Alle Fälle waren zuvor bekannt; daraus folgt keine Produktionsfreigabe.", ""]
    if report.get("anchor_query_classifier") or report.get("nested_query_classifier"):
        lines += ["Weitere CPU-Filterversuche verwenden dieselben bekannten Prüffälle und denselben "
            "festen logistischen Schätzer. Die folgenden Zahlen zeigen jeweils die Ausgabe vor und nach "
            "dem Filter; vorhandene Katalogparaphrasen sind zusätzliche schwache Trainingsbeispiele, keine neuen Prüffälle.", "",
            "| Filter / Ranking | Haupt-Treffer | Haupt-Negativfälle | Haupt-negative Normpaare | Zusatz-Treffer | Zusatz-Negativfälle | Zusatz-negative Normpaare |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        comparisons = []
        anchor = report.get("anchor_query_classifier")
        if anchor:
            comparisons.append(("Anker-OOF / kaskade", anchor["datasets"]))
            comparisons.extend(("Anker-OOF / " + name, value["datasets"])
                               for name, value in anchor.get("instruction_variants", {}).items())
        nested = report.get("nested_query_classifier")
        if nested:
            comparisons.extend(("Verschachtelt / " + name, value["datasets"]) for name, value in nested["variants"].items())
        for name, datasets in comparisons:
            values = []
            for dataset in ("main", "additional"):
                before, after = (datasets[dataset][key] for key in ("before", "after"))
                values += [f"{before['hit5_cases']}→{after['hit5_cases']}/{before['positive_cases']}",
                           f"{before['negative_with_suggestion']}→{after['negative_with_suggestion']}/{before['negative_cases']}",
                           f"{before['negative_unique_case_norm_pairs']}→{after['negative_unique_case_norm_pairs']}"]
            lines.append("| " + " | ".join([name, *values]) + " |")
        lines += [""]
        if anchor:
            lines += [f"Der Ankervergleich dedupliziert {anchor['anchors']['original_rows']} vorhandene Katalogeinträge "
                f"zu {anchor['anchors']['unique_training_texts']} Trainingstexten. Ganze Originalquellen und nahe Varianten "
                "bleiben pro äußerem Fold ausgeschlossen; die Schwelle bleibt 0,5. Zwei echte Zusatztreffer gehen "
                "weiterhin verloren. Der Filter wurde nicht aktiviert.", ""]
        if nested:
            lines += ["Die verschachtelte Variante bestimmt die größte Grenze bis 0,5, die alle positiven "
                "Originalfälle der inneren Quellen-OOF erhält, ausschließlich innerhalb des jeweiligen äußeren Trainingssatzes. "
                "Kataloganker dienen nur dem Fit. Es gibt keine gemeinsame Auswahl anhand globaler OOF-Werte oder äußerer Testlabels. "
                f"In {nested['fallback_outer_folds']} äußeren Folds fehlt einer inneren Trainingsgruppe eine Klasse; "
                "die vorab festgelegte Rückfallgrenze 0 unterdrückt dort nichts. Alle Fälle bleiben im Nenner. "
                "Hier bleiben dadurch alle Ausgaben erhalten, einschließlich der Fehlalarme. Auch dieser Filter bleibt deaktiviert; "
                "keiner der Versuche speichert ein Modell.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--instruction", type=Path, action="append", default=[])
    parser.add_argument("--passages", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--benchmark-prefix", type=Path)
    parser.add_argument("--query-classifier", type=Path, help="Optional rejected fixed OOF classifier diagnostic")
    parser.add_argument("--live", type=Path, help="Actual current public GPU Pipeline.run comparison")
    parser.add_argument("--instruction-mean", type=Path, help="Optional rejected fixed post-hoc instruction-score mean")
    parser.add_argument("--qwen8b", type=Path, help="Complete standalone original 8B reduced/full GPU head experiment")
    parser.add_argument("--anchor-query-classifier", type=Path, help="Rejected fixed anchor-augmented source-OOF diagnostic")
    parser.add_argument("--nested-query-classifier", type=Path, help="Rejected nested source-OOF cutoff diagnostic")
    parser.add_argument("--audit-one", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--audit-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.audit_one:
            save(args.audit_output, audit_one(args.audit_one.resolve()))
            return 0
        require(args.baseline and args.out and (args.instruction or args.passages or args.qwen8b), "Baseline, new completed runs and fresh out are required")
        paths = [args.baseline.resolve(), *(p.resolve() for p in args.instruction), *(p.resolve() for p in args.passages)]
        require(len(paths) == len(set(paths)) and not args.out.exists(), "Reports must be distinct; choose fresh outputs")
        reports = [read(path) for path in paths]
        bound_hashes = {}
        for path, report in zip(paths, reports, strict=True):
            bound_hashes.update(bind_report(path, report))
        baseline = reports[0]
        require(baseline["format"] == "quality_precision_probe/1", "Expected completed original R2 baseline")
        base_cases = cases(baseline["rows"])
        for report in reports[1:]:
            same_cases(base_cases, cases(report["rows"]))
        live = None
        if args.live:
            live_path = args.live.resolve()
            require(live_path not in paths, "Live report must be distinct from research inputs")
            live = read(live_path)
            require(live.get("format") == "quality_recall_live/1", "Expected current public recall live report")
            bound_hashes.update(bind_report(live_path, live))
            same_cases(base_cases, cases(live["rows"]))
            selected = [(path, report) for path, report in zip(paths, reports, strict=True)
                        if digest(path) == live["reference_report_sha256"]]
            require(len(selected) == 1 and selected[0][1]["format"] == "quality_passage_rerank_probe/1",
                    "Live reference must be one of the supplied complete structural experiments")
            require(live["profile_config"] == selected[0][1]["variants"][live["selected_variant"]], "Live profile configuration differs")
        args.out.mkdir(parents=True)
        public = {"format": "quality_recall_comparison/1", "status": "complete",
            "generator_sha256": digest(__file__), "generator_dependencies": {},
            "datasets": {name: {"positive": p, "negative": n} for name, (p, n) in DATASETS.items()},
            "notes": ["All 162 cases were previously examined; this is development exploration.",
                      "Unlisted norms in positive cases are unadjudicated, not negative labels.",
                      "Suggestion precision and full legal recall remain unavailable without adjudication.",
                      "Negative unique pairs count each (case ID, norm ID) once regardless of focus multiplicity."],
            "reports": {}, "variants": {}}
        private = {"format": "quality_recall_paired_details/1", "changes": {}}
        for number, (path, report) in enumerate(zip(paths, reports, strict=True)):
            label = "baseline" if number == 0 else f"run{number}"
            audit = None
            if number:
                audit_path = args.out / f"{label}-audit.json"
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--audit-one", str(path),
                    "--audit-output", str(audit_path)], capture_output=True, check=False)
                (args.out / f"{label}-audit.stdout.txt").write_bytes(completed.stdout)
                (args.out / f"{label}-audit.stderr.txt").write_bytes(completed.stderr)
                require(completed.returncode == 0, f"CPU audit failed for {label}; inspect private audit logs")
                audit = read(audit_path)
                require(audit["status"] == "pass" and audit["report_sha256"] == digest(path), "Audit binding differs")
            names = ["kaskade"] if number == 0 else list(report.get("instructions", report.get("variants", {})))
            require(bool(names), "No declared variants")
            validate_summaries(report, names)
            public["reports"][label] = {"sha256": digest(path), "format": report["format"], "audit": audit,
                "configuration": {key: report[key] for key in ("instructions", "variants", "models", "model", "shortlist", "threshold", "topk", "max_length") if key in report}}
            if report.get("inherited_pilot"):
                cancellation_path = Path(report["inherited_pilot"]["path"]) / "cancellation.json"
                cancellation = read(cancellation_path)
                public["reports"][label]["inherited_pilot"] = {
                    "cancellation_sha256": digest(cancellation_path), "status": cancellation["status"],
                    "completed_cases": cancellation["completed_cases"], "planned_cases": cancellation["planned_cases"],
                    "excluded_from_full_quality_metrics": True, "cache_pairs": cancellation["cache_pairs"],
                    "query_vectors": cancellation["query_count"]}
            current_cases = cases(report["rows"])
            for name in names:
                key = f"{label}/{name}"
                public["variants"][key] = {"datasets": {dataset: variant_metrics(
                    [row for row in report["rows"] if row["dataset"] == dataset], name) for dataset in DATASETS},
                    "paired_change_counts": {}}
                for dataset in DATASETS:
                    before = {k: v for k, v in base_cases.items() if v["dataset"] == dataset}
                    after = {k: v for k, v in current_cases.items() if v["dataset"] == dataset}
                    change = paired_changes(before, after, "kaskade", name)
                    private["changes"][f"{key}/{dataset}"] = change
                    public["variants"][key]["paired_change_counts"][dataset] = {k: len(v) for k, v in change.items()}
        if args.query_classifier:
            public["query_need_classifier"] = classifier_aggregate(args.query_classifier.resolve(), paths[0], base_cases)
        if args.instruction_mean:
            from tools._quality_recall_mean import aggregate
            mean_path = args.instruction_mean.resolve()
            same_cases(base_cases, cases(read(mean_path)["rows"]))
            public["instruction_mean"] = aggregate(mean_path)
            public["generator_dependencies"]["tools/_quality_recall_mean.py"] = digest(ROOT / "tools/_quality_recall_mean.py")
        if args.qwen8b:
            from tools._quality_recall_qwen8b import aggregate
            public["qwen8b"] = aggregate(args.qwen8b, paths[0], args.out / "qwen8b-audit")
            public["generator_dependencies"]["tools/_quality_recall_qwen8b.py"] = digest(ROOT / "tools/_quality_recall_qwen8b.py")
        if args.anchor_query_classifier or args.nested_query_classifier:
            from tools._quality_recall_classifiers import aggregate_anchors, aggregate_nested
            for option, key, function in ((args.anchor_query_classifier, "anchor_query_classifier", aggregate_anchors),
                                          (args.nested_query_classifier, "nested_query_classifier", aggregate_nested)):
                if option:
                    public[key] = function(option.resolve(), paths[0], base_cases)
            public["generator_dependencies"]["tools/_quality_recall_classifiers.py"] = digest(ROOT / "tools/_quality_recall_classifiers.py")
        if live is not None:
            audit_path = args.out / "live-audit.json"
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--audit-one", str(live_path),
                "--audit-output", str(audit_path)], capture_output=True, check=False)
            (args.out / "live-audit.stdout.txt").write_bytes(completed.stdout)
            (args.out / "live-audit.stderr.txt").write_bytes(completed.stderr)
            require(completed.returncode == 0, "Live output audit failed; inspect private audit logs")
            live_audit = read(audit_path)
            require(live_audit["status"] == "pass" and live_audit["report_sha256"] == digest(live_path), "Live audit binding differs")
            profile = live["public_profile"]
            key = f"live/{profile}"
            values = {"datasets": {dataset: variant_metrics([row for row in live["rows"] if row["dataset"] == dataset], profile)
                                    for dataset in DATASETS}, "paired_change_counts": {}}
            current_cases = cases(live["rows"])
            for dataset in DATASETS:
                change = paired_changes({k: v for k, v in base_cases.items() if v["dataset"] == dataset},
                    {k: v for k, v in current_cases.items() if v["dataset"] == dataset}, "kaskade", profile)
                private["changes"][f"{key}/{dataset}"] = change
                values["paired_change_counts"][dataset] = {name: len(items) for name, items in change.items()}
            public["variants"] = {key: values, **public["variants"]}
            public["live_result"] = {"report_sha256": digest(live_path), "reference_report_sha256": live["reference_report_sha256"],
                "selected_variant": live["selected_variant"], "public_profile": profile, "profile_config": live["profile_config"],
                "source_hashes": live["source_hashes"], "status": live["status"], "cases": len(live["rows"]),
                "reference_differences": len(live["mismatches"]), "difference_categories": live["difference_categories"],
                "metadata_difference_counts": live_audit["metadata_difference_counts"],
                "models": live["models"], "runtime": {name: live["runtime"][name] for name in (
                    "torch", "hip", "cuda", "device", "device_total_bytes", "packages")},
                "warmup_seconds": live["warmup"]["seconds"], "timings": live["timings"], "audit": live_audit,
                "notes": ["All table metrics come from actual current Pipeline.run, including when research outputs differ.",
                          "Only these synchronized case timings describe this fresh live run; model loading and warmup are excluded.",
                          "NER and lemma are disabled; these timings do not measure full server latency."]}
        for filename, expected in bound_hashes.items():
            check_hash(filename, expected)
        require("torch" not in sys.modules, "Aggregator unexpectedly imported a model runtime")
        public_payload_check(public)
        save(args.out / "paired-details.json", private)
        save(args.out / "report.json", public)
        if args.benchmark_prefix:
            require(not args.benchmark_prefix.with_suffix(".json").exists()
                    and not args.benchmark_prefix.with_suffix(".md").exists(), "Existing benchmarks must remain unchanged")
            args.benchmark_prefix.parent.mkdir(parents=True, exist_ok=True)
            save(args.benchmark_prefix.with_suffix(".json"), public)
            args.benchmark_prefix.with_suffix(".md").write_text(render(public), encoding="utf-8")
        print(json.dumps({"status": "complete", "variants": len(public["variants"]), "out": str(args.out)}))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
