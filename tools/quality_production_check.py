"""Replay the actual production Pipeline from exact cached GPU model outputs.

Build, save, and reload an ordinary passage index with current production code.
No model is instantiated, no GPU is used, and no index generation is activated.
Every original model input must match an archived input exactly. In particular,
duplicate passage/anchor texts retain their recorded row-specific GPU values;
small batch-dependent differences are never averaged or silently overwritten.

The 132 development cases and 30 historical additional cases remain separate.
This checks implementation equivalence, not fresh blind quality or latency.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CASE_FIELDS = ("case_id", "group", "query", "targets", "negative", "category", "uncertain")
REFERENCE_VARIANTS = {
    "standard": "baseline/baseline_or",
    "normpaket": "norm_package/weak_rescue",
    "kontext": "context_package/weak_rescue",
    "kaskade": "passage_package_200/qwen_half",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CacheMiss(RuntimeError):
    def __init__(self, cache, items):
        self.detail = {"cache": cache, "missing_inputs": items}
        super().__init__(f"{cache}: {len(items)} exact input(s) missing")


def load_vectors(directory, stem, *, expected_hashes=None, text_name=None):
    import numpy as np

    vector_path = directory / f"{stem}.npy"
    text_path = directory / (text_name or f"{stem}.texts.json")
    if expected_hashes is not None:
        for path in (vector_path, text_path):
            if expected_hashes.get(path.name) != digest(path):
                raise ValueError(f"Recorded vector input checksum mismatch: {path}")
    texts, vectors = read(text_path), np.load(vector_path, allow_pickle=False)
    if (not isinstance(texts, list) or not all(isinstance(text, str) for text in texts)
            or vectors.dtype != np.float32 or vectors.ndim != 2
            or len(texts) != len(vectors) or not np.isfinite(vectors).all()):
        raise ValueError(f"Invalid vector cache: {vector_path}")
    return texts, vectors


class QueryEncoder:
    def __init__(self, texts, vectors, *, name, revision, label):
        if len(texts) != len(set(texts)):
            raise ValueError(f"Query cache must have unique exact inputs: {label}")
        self.name, self.revision, self.dim = name, revision, int(vectors.shape[1])
        self.label = label
        self.vectors = dict(zip(texts, vectors, strict=True))
        self.calls = 0

    def encode(self, texts):
        import numpy as np

        missing = [text for text in texts if text not in self.vectors]
        if missing:
            raise CacheMiss(self.label, list(dict.fromkeys(missing)))
        self.calls += 1
        return (np.stack([self.vectors[text] for text in texts]) if texts
                else np.empty((0, self.dim), dtype=np.float32))


class RecordingEncoder:
    """Discover current base-builder inputs; validate row identity before reuse."""

    def __init__(self, base):
        self.name, self.revision, self.dim = base.encoder_name, base.encoder_revision, base.dim
        self._vectors, self.texts = base.vectors, None

    def encode(self, texts):
        if self.texts is not None or len(texts) != len(self._vectors):
            raise ValueError("Base builder changed its encoding call/count; cannot map old vectors")
        self.texts = list(texts)
        return self._vectors.copy()


class BuildEncoder:
    """Replay only the preverified exact build sequence, retaining duplicate rows."""

    def __init__(self, texts, vectors, *, name, revision):
        self.name, self.revision, self.dim = name, revision, int(vectors.shape[1])
        self.texts, self.vectors, self.position = texts, vectors, 0

    def encode(self, texts):
        start, end = self.position, self.position + len(texts)
        expected = self.texts[start:end]
        if list(texts) != expected:
            raise CacheMiss("verified build sequence", [
                {"row": start + i, "actual": actual,
                 "expected": expected[i] if i < len(expected) else None}
                for i, actual in enumerate(texts)
                if i >= len(expected) or actual != expected[i]])
        self.position = end
        return self.vectors[start:end].copy()


class CachedReranker:
    def __init__(self, paths, *, name, revision):
        self.name, self.revision = name, revision
        self.scores, self.used, self.overrides = {}, set(), []
        for path in paths:
            records = read(path) if path.suffix == ".json" else (
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
            for record in records:
                query, doc, value = ((record["query"], record["document"], record["score"])
                                     if isinstance(record, dict) else record)
                if not isinstance(query, str) or not isinstance(doc, str) or not math.isfinite(value):
                    raise ValueError(f"Invalid raw reranker record: {path}")
                key = (query, doc)
                if key in self.scores and self.scores[key] != value:
                    self.overrides.append({"path": str(path), "query": query,
                                           "document": doc, "old": self.scores[key], "new": value})
                self.scores[key] = value

    def score(self, pairs):
        pairs = [tuple(pair) for pair in pairs]
        missing = [pair for pair in pairs if pair not in self.scores]
        if missing:
            raise CacheMiss(self.name, list(dict.fromkeys(missing)))
        self.used.update(pairs)
        return [self.scores[pair] for pair in pairs]


def norm_order(suggestions):
    return [{"start": s["start"], "end": s["end"], "satz": s["satz"],
             "norm_ids": [n["id"] for n in s["normen"]]} for s in suggestions]


def compact_marks(report):
    return {
        "cards": [{"id": card.eintrag.id, "via": card.via,
                   "span": [card.treffer.start, card.treffer.end]} for card in report.karten],
        "trigger_marks": [{"span": [risk.start, risk.end], "title": risk.titel,
                           "norm_ids": [n["id"] for n in risk.norm_vorschlag]}
                          for risk in report.risiken if risk.quelle == "trigger"],
    }


def compare_case(actual, expected, targets):
    from tools._gpu_eval_metrics import evaluate_variant

    expected_suggestions, actual_suggestions = expected["suggestions"], actual["suggestions"]
    expected_metrics = evaluate_variant(expected_suggestions, targets)
    actual_metrics = evaluate_variant(actual_suggestions, targets)
    order_equal = norm_order(expected_suggestions) == norm_order(actual_suggestions)
    # Production displays 3 decimals. This records display discrepancies only;
    # selection/order and every case metric are independently compared exactly.
    score_differences = []
    if order_equal:
        for si, (es, ac) in enumerate(zip(expected_suggestions, actual_suggestions, strict=True)):
            for rank, (en, an) in enumerate(zip(es["normen"], ac["normen"], strict=True), 1):
                for score in ("sim", "rerank"):
                    ev, av = en.get(score), an.get(score)
                    if (None if ev is None else round(ev, 3)) != av:
                        score_differences.append({"sentence": si, "rank": rank, "score": score,
                                                  "expected_raw": ev, "actual_display": av})
    return {"order_equal": order_equal, "metrics_equal": expected_metrics == actual_metrics,
            "expected_metrics": expected_metrics, "actual_metrics": actual_metrics,
            "score_display_differences": score_differences}


def snapshot_sources(out):
    paths = sorted((ROOT / "risk_layer").glob("*.py")) + [
        ROOT / "catalog/begriffe.yaml", ROOT / "tools/_gpu_eval_metrics.py", Path(__file__)]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = digest(destination)
    return hashes


def archived_build_inputs(baseline, graph, base):
    """Reconstruct the archived builder's exact inputs, including duplicate rows."""
    source = baseline / "source-before"
    alias = "_quality_production_original"
    spec = importlib.util.spec_from_file_location(
        alias, source / "risk_layer/__init__.py",
        submodule_search_locations=[str(source / "risk_layer")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    original_embedding = importlib.import_module(alias + ".embedding")
    original_catalog = importlib.import_module(alias + ".catalog").lade_katalog(source / "catalog/begriffe.yaml")
    recorder = RecordingEncoder(base)
    original = original_embedding.EmbeddingIndex.build(graph, recorder, katalog=original_catalog)
    fields = ("ids", "zitate", "titel", "graph_fingerprint", "embedding_graph_fingerprint",
              "encoder_name", "encoder_revision", "dim", "catalog_version", "catalog_fingerprint",
              "builder_version", "input_fingerprint")
    if any(getattr(original, field) != getattr(base, field) for field in fields):
        raise ValueError("Archived builder does not reproduce the archived index identity")
    return recorder.texts


def execute(args, out, result):
    import numpy as np

    # Import the archived current production package, preserving its normal
    # package name and relative imports. No internal retrieval methods are patched.
    sys.path[:0] = [str(out / "source"), str(ROOT)]
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import EmbeddingIndex
    from risk_layer.graph import NormGraph
    from risk_layer.passages import build_passages, catalog_anchors
    from risk_layer.pipeline import AnalyseOptionen, Pipeline
    from risk_layer.triggerindex import TriggerIndex
    from tools._gpu_eval_metrics import summarize
    import risk_layer.pipeline as pipeline_module

    if not Path(pipeline_module.__file__).resolve().is_relative_to(out / "source"):
        raise RuntimeError("Production package was imported before its source snapshot")
    baseline, experiment, additional = args.baseline.resolve(), args.experiments.resolve(), args.additional.resolve()
    base_report, main_report, extra_report = (read(path / "report.json")
                                             for path in (baseline, experiment, additional))
    for name, expected in base_report["source_hashes"].items():
        if name.startswith(("artifacts", "source-before")) or name in (
                "query-texts.json", "query-vectors.npy", "reranker-scores.json"):
            if digest(baseline / name) != expected:
                raise ValueError(f"Archived baseline changed: {name}")
    for directory, reference in ((experiment, main_report), (additional, extra_report)):
        if reference["manifest"]["baseline_report_sha256"] != digest(baseline / "report.json"):
            raise ValueError("Experiment refers to a different original Pipeline baseline")
        for name, expected in reference["manifest"]["input_hashes"].items():
            if digest(baseline / name) != expected:
                raise ValueError(f"Experiment baseline input differs: {name}")
        for name in ("queries.npy", "queries.texts.json", "passages.npy", "passages.texts.json",
                     "anchors.npy", "anchors.texts.json", "passages.json", "anchors.json", "new-scores.jsonl"):
            if digest(directory / name) != reference["artifact_hashes"][name]:
                raise ValueError(f"Experiment artifact changed: {directory / name}")
    for name in ("passages.npy", "passages.texts.json", "passages.json",
                 "anchors.npy", "anchors.texts.json", "anchors.json"):
        if digest(experiment / name) != digest(additional / name):
            raise ValueError(f"Main/additional experiments did not share identical index evidence: {name}")
    result["input_reports"] = {str(path / "report.json"): digest(path / "report.json")
                               for path in (baseline, experiment, additional)}
    graph = NormGraph(baseline / "artifacts/graph.sqlite")
    trigger = TriggerIndex(baseline / "artifacts/trigger.sqlite")
    try:
        base = EmbeddingIndex.load(baseline / "artifacts/index")
        if base is None:
            raise ValueError("Archived base index missing")
        catalog = lade_katalog(out / "source/catalog/begriffe.yaml")
        recorder = RecordingEncoder(base)
        recorded = EmbeddingIndex.build(graph, recorder, katalog=catalog, passagen=False)
        original_texts = archived_build_inputs(baseline, graph, base)
        checks = {field: getattr(recorded, field) == getattr(base, field)
                  for field in ("ids", "zitate", "titel", "graph_fingerprint", "embedding_graph_fingerprint",
                                "encoder_name", "encoder_revision", "dim")}
        checks["exact_archived_builder_inputs"] = recorder.texts == original_texts
        result["base_identity_checks"] = checks
        if not all(checks.values()) or recorder.texts is None:
            raise ValueError("Current base builder does not reproduce archived vector row identity")
        save(out / "base-build-inputs.json", [
            {"row": i, "norm_id": nid, "text": text}
            for i, (nid, text) in enumerate(zip(base.ids, recorder.texts, strict=True))])
        passages = build_passages(graph)
        anchors = [[nid, text] for nid, texts in catalog_anchors(catalog).items()
                   if nid in {p.norm_id for p in passages} for text in texts]
        if [asdict(p) for p in passages] != read(experiment / "passages.json"):
            raise ValueError("Current passage identities/texts differ from the experiment")
        if anchors != read(experiment / "anchors.json"):
            raise ValueError("Current anchor ownership/order differs from the experiment")
        passage_texts, passage_vectors = load_vectors(experiment, "passages")
        anchor_texts, anchor_vectors = load_vectors(experiment, "anchors")
        if passage_texts != [p.embedding_text for p in passages] or anchor_texts != [a[1] for a in anchors]:
            raise ValueError("Passage/anchor vector input texts differ from current build inputs")
        # A single build encoder verifies the concatenated input sequence across
        # the builder's real batch boundaries; equal strings keep distinct rows.
        texts = recorder.texts + passage_texts + anchor_texts
        vectors = np.vstack([base.vectors, passage_vectors, anchor_vectors])
        build_encoder = BuildEncoder(texts, vectors, name=base.encoder_name, revision=base.encoder_revision)
        index = EmbeddingIndex.build(graph, build_encoder, katalog=catalog, passagen=True)
        if build_encoder.position != len(texts):
            raise ValueError("Current passage builder did not consume the complete verified input sequence")
        index.save(out / "index")
        loaded = EmbeddingIndex.load(out / "index")
        if (loaded is None or loaded.passage_evidence is None
                or loaded.input_fingerprint != index.input_fingerprint
                or not np.array_equal(loaded.vectors, base.vectors)
                or not np.array_equal(loaded.passage_evidence.vectors, passage_vectors)
                or not np.array_equal(loaded.passage_evidence.anchor_vectors, anchor_vectors)):
            raise ValueError("Passage index round-trip changed identity or cached vectors")
        result["index"] = {"path": str(out / "index"), "base_rows": len(loaded.ids),
                           "passage_rows": len(passages), "anchor_rows": len(anchors),
                           "input_fingerprint": loaded.input_fingerprint,
                           "passage_fingerprint": loaded.passage_evidence.fingerprint,
                           "graph_fingerprint": loaded.graph_fingerprint,
                           "catalog_fingerprint": loaded.catalog_fingerprint,
                           "round_trip_exact": True,
                           "hashes": {name: digest(out / "index" / name)
                                      for name in ("meta.json", "vectors.npy")}}
        print(f"Real index built/saved/reloaded: {len(loaded.ids)} + {len(passages)} + {len(anchors)} rows", flush=True)
        encoder_args = {"name": base.encoder_name, "revision": base.encoder_revision}
        legacy_encoder = QueryEncoder(*load_vectors(baseline, "query-vectors", text_name="query-texts.json"),
                                      **encoder_args, label="archived original baseline queries")
        main_encoder = QueryEncoder(*load_vectors(experiment, "queries"), **encoder_args, label="main experiment queries")
        extra_encoder = QueryEncoder(*load_vectors(additional, "queries"), **encoder_args, label="additional queries")
        for reference in (main_report, extra_report):
            if reference["runtime"]["encoder_revision"] != base.encoder_revision:
                raise ValueError("Experiment encoder revision differs from archived index")
        ranker_args = {"name": "bge-reranker-v2-m3", "revision": main_report["runtime"]["reranker_revision"]}
        if any(reference["runtime"]["reranker_revision"] != ranker_args["revision"]
               for reference in (extra_report, base_report)):
            raise ValueError("Additional reranker revision differs")
        legacy_ranker = CachedReranker([baseline / "reranker-scores.json"], **ranker_args)
        main_ranker = CachedReranker([baseline / "reranker-scores.json", experiment / "new-scores.jsonl"], **ranker_args)
        extra_ranker = CachedReranker([baseline / "reranker-scores.json", additional / "new-scores.jsonl"], **ranker_args)
        base_rows = {row["case_id"]: row for row in base_report["rows"]}
        qwen_cases = {}
        for directory in args.qwen_dir:
            directory = directory.resolve()
            qwen = read(directory / "report.json")
            matched_sources = [path for path in (experiment, additional)
                               if qwen["manifest"]["inputs"]["report.json"] == digest(path / "report.json")]
            if len(matched_sources) != 1:
                raise ValueError("Qwen report is not bound to one selected BGE experiment")
            for name, expected_hash in qwen["manifest"]["inputs"].items():
                if digest(matched_sources[0] / name) != expected_hash:
                    raise ValueError(f"Qwen/BGE candidate-source mismatch: {name}")
            if qwen["manifest"]["bge_shortlist"] != 20 or "passage_package_200" not in qwen["manifest"]["configs"]:
                raise ValueError("Cascade reference must use passage_package_200 with BGE top20")
            if qwen["score_sha256"] != digest(directory / "scores.jsonl"):
                raise ValueError("Qwen score cache differs from completed report")
            ranker = CachedReranker([directory / "scores.jsonl"], name="bge-qwen3-reranker-4b",
                                    revision=qwen["manifest"]["model_revision"])
            result["input_reports"][str(directory / "report.json")] = digest(directory / "report.json")
            for row in qwen["rows"]:
                if row["case_id"] in qwen_cases:
                    raise ValueError("Overlapping Qwen case references")
                qwen_cases[row["case_id"]] = (row, ranker)
        profiles = (["kaskade"] if args.qwen_only else
                    ["standard", "normpaket", "kontext"] + (["kaskade"] if args.qwen_dir else []))
        result["qwen_only"] = args.qwen_only
        result["not_evaluated_cases"] = []
        pipe = Pipeline(catalog)
        all_ids = set()
        for dataset, reference, query_encoder, bge_ranker in (
            ("main", main_report, main_encoder, main_ranker),
            ("additional", extra_report, extra_encoder, extra_ranker),
        ):
            for i, reference_row in enumerate(reference["rows"]):
                case = {key: reference_row.get(key) for key in CASE_FIELDS}
                if args.qwen_only and case["case_id"] not in qwen_cases:
                    result["not_evaluated_cases"].append({"case_id": case["case_id"], "dataset": dataset,
                                                         "reason": "No completed Qwen reference provided"})
                    continue
                if case["case_id"] in all_ids:
                    raise ValueError("Duplicate case ID across validation datasets")
                all_ids.add(case["case_id"])
                row = {**case, "dataset": dataset}
                result["rows"].append(row)
                for profile in profiles:
                    encoder, ranker = query_encoder, bge_ranker
                    expected = reference_row[REFERENCE_VARIANTS[profile]] if profile != "kaskade" else None
                    if dataset == "main" and profile == "standard":
                        original = base_rows[case["case_id"]]
                        if any(case[key] != original.get(key) for key in CASE_FIELDS):
                            raise ValueError("Main case content/labels changed from original baseline")
                        expected, encoder, ranker = original["after"], legacy_encoder, legacy_ranker
                    if profile == "kaskade":
                        if case["case_id"] not in qwen_cases:
                            raise CacheMiss("Qwen case reference", [case["case_id"]])
                        qrow, ranker = qwen_cases[case["case_id"]]
                        if any(case[key] != qrow.get(key) for key in CASE_FIELDS):
                            raise ValueError("Qwen case content/labels differ")
                        expected = qrow[REFERENCE_VARIANTS[profile]]
                        ranker.vorauswahl = bge_ranker
                    try:
                        report = pipe.run(case["query"],
                                          optionen=AnalyseOptionen(erzeugt_am="quality-production-check",
                                                                  detektor=True, streit=True, semantik_profil=profile),
                                          graph=graph, trigger=trigger, embedding=loaded,
                                          embedding_encoder=encoder, reranker=ranker)
                        actual = {"suggestions": report.semantik_vorschlaege, **compact_marks(report)}
                        actual["comparison"] = compare_case(actual, expected, case["targets"])
                        actual["reference"] = ("original_pipeline/after" if dataset == "main" and profile == "standard"
                                               else REFERENCE_VARIANTS[profile])
                        row[profile] = actual
                        comparison = actual["comparison"]
                        if not comparison["order_equal"] or not comparison["metrics_equal"]:
                            result["mismatches"].append({"case_id": case["case_id"], "dataset": dataset,
                                                         "profile": profile, "comparison": comparison,
                                                         "expected": norm_order(expected["suggestions"]),
                                                         "actual": norm_order(actual["suggestions"])})
                        if profile != "standard" and "error" not in row.get("standard", {"error": True}):
                            if any(actual[key] != row["standard"][key] for key in ("cards", "trigger_marks")):
                                result["hard_mark_mismatches"].append({"case_id": case["case_id"], "profile": profile})
                        elif args.qwen_only:
                            if dataset == "main":
                                hard_reference = base_rows[case["case_id"]]["after"]
                                actual["hard_mark_reference"] = "original_pipeline/after"
                            else:
                                standard_report = pipe.run(
                                    case["query"], optionen=AnalyseOptionen(erzeugt_am="quality-production-check",
                                                                          detektor=True, streit=True),
                                    graph=graph, trigger=trigger, embedding=loaded,
                                    embedding_encoder=query_encoder, reranker=bge_ranker)
                                hard_reference = compact_marks(standard_report)
                                actual["hard_mark_reference"] = "same_source_standard"
                            if any(actual[key] != hard_reference[key] for key in ("cards", "trigger_marks")):
                                result["hard_mark_mismatches"].append({"case_id": case["case_id"], "profile": profile})
                    except CacheMiss as exc:
                        row[profile] = {"error": str(exc)}
                        result["cache_misses"].append({"case_id": case["case_id"], "profile": profile, **exc.detail})
                if i % 10 == 0:
                    print(f"{dataset}: {i + 1}/{len(reference['rows'])} cases; {len(result['mismatches'])} mismatches; "
                          f"{len(result['cache_misses'])} cache misses", flush=True)
        result["profiles"] = profiles
        if args.qwen_only and all_ids != set(qwen_cases):
            raise ValueError("Qwen reference contains cases outside the selected BGE validation datasets")
        result["summaries"] = {
            dataset: {profile: summarize([row for row in result["rows"] if row["dataset"] == dataset], profile)
                      for profile in profiles if all("error" not in row[profile] for row in result["rows"] if row["dataset"] == dataset)}
            for dataset in ("main", "additional")}
        result["score_cache_overrides"] = {"main": main_ranker.overrides, "additional": extra_ranker.overrides}
        result["used_rerank_pairs"] = {"legacy": len(legacy_ranker.used), "main": len(main_ranker.used),
                                        "additional": len(extra_ranker.used)}
    finally:
        graph.close()
        trigger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "out/gpu-ab-20260907-final")
    parser.add_argument("--experiments", type=Path, default=ROOT / "out/quality-experiments-20260907-r2")
    parser.add_argument("--additional", type=Path, default=ROOT / "out/quality-additional-validation/bge")
    parser.add_argument("--out", type=Path, default=ROOT / "out/quality-production")
    parser.add_argument("--qwen-dir", type=Path, action="append", default=[],
                        help="Completed cascade probe directory; repeat to cover both datasets")
    parser.add_argument("--qwen-only", action="store_true",
                        help="Replay only kaskade and only cases present in completed Qwen references; report omitted cases")
    args = parser.parse_args(argv)
    if args.qwen_only and not args.qwen_dir:
        parser.error("--qwen-only requires at least one completed --qwen-dir")
    for directory in args.qwen_dir:
        if not (directory / "report.json").is_file():
            parser.error(f"Qwen probe is not complete: {directory / 'report.json'}")
    out = args.out.resolve()
    if out.exists():
        parser.error("Output directory already exists; use a new directory to preserve previous evidence")
    out.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", RISK_LAYER_NER="aus",
                      RISK_LAYER_LEMMA="aus", RISK_LAYER_FUSION="an", RISK_LAYER_RERANK_SCHWELLE="0.10",
                      RISK_LAYER_SIM_RETTUNG="0.55", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = {"format": "quality_production_check/1", "source_hashes": snapshot_sources(out),
              "numpy_version": importlib.metadata.version("numpy"), "rows": [], "mismatches": [],
              "hard_mark_mismatches": [], "cache_misses": [], "notes": [
                  "Actual production Pipeline; actual build(passagen=True), save, and load.",
                  "Only exact recorded GPU outputs are replayed; no models loaded and no GPU used.",
                  "Original standard uses original baseline query vectors; profiles use their experiment query vectors.",
                  "Build row-specific passage and anchor vectors preserve batch-dependent differences for duplicate texts.",
                  "Production score display rounding is reported separately from exact norm order and case metrics.",
                  "Kaskade uses the predeclared Qwen >=0.5 policy without similarity rescue, after BGE top20 from pool200.",
                  "Main development data and historical additional validation are separate; neither is fresh blind gold.",
                  "No activation, no performance conclusion from cached execution."]}
    try:
        execute(args, out, result)
        result["status"] = "pass" if not any(result[key] for key in
                                              ("mismatches", "hard_mark_mismatches", "cache_misses")) else "mismatch"
        code = 0 if result["status"] == "pass" else 1
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, CacheMiss):
            result["cache_misses"].append(exc.detail)
        code = 2
    result["torch_imported"] = "torch" in sys.modules
    if result["torch_imported"]:
        result["status"], code = "failed", 2
        result["model_runtime_error"] = "Torch was unexpectedly imported during CPU-only replay"
    save(out / "report.json", result)
    print(f"{result['status']}: {len(result['rows'])} cases, {len(result['mismatches'])} order/metric mismatches, "
          f"{len(result['cache_misses'])} cache misses; {out / 'report.json'}", flush=True)
    if "error" in result:
        print(result["error"], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
