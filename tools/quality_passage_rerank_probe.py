"""Compare coherent facts and neural source-passage selection on local cases.

All variants call the frozen production passage_suggestions implementation.
Exact scores are keyed by model/instruction/budget/input identity. An explicitly
declared audited pilot can supply identical inputs; inherited and fresh work are
reported separately. Gold targets never enter runtime decisions.
The 162 cases are previously seen development data, not a blind accuracy test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class CachedScores:
    """A cache belongs to exactly one live model and one immutable run."""

    def __init__(self, model, path):
        self.model, self.path = model, path
        self.cache = {}
        self.seconds = 0.
        self.inherited_pairs = 0
        self.used = set()
        self.name, self.revision = model.name, model.revision

    def _score(self, pairs, instruction=None, max_length=1024):
        keys = [(instruction, max_length, q, d) for q, d in pairs]
        self.used.update(keys)
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        batch_size = 8 if instruction is None and self.name.startswith("bge-reranker") else 2
        with self.path.open("a", encoding="utf-8") as stream:
            for offset in range(0, len(missing), batch_size):
                batch = missing[offset:offset + batch_size]
                inputs = [(key[2], key[3]) for key in batch]
                start = time.perf_counter()
                scores = (self.model.score(inputs) if instruction is None else
                          self.model.score_with_instruction(inputs, instruction, max_length=max_length))
                self.seconds += time.perf_counter() - start
                if len(scores) != len(batch) or any(
                        isinstance(s, bool) or not isinstance(s, Real) or not math.isfinite(s)
                        or (instruction is not None and not 0 <= s <= 1) for s in scores):
                    raise ValueError("Incomplete/nonfinite live model output")
                for key, value in zip(batch, scores, strict=True):
                    self.cache[key] = float(value)
                    stream.write(json.dumps([*key, float(value)], ensure_ascii=False) + "\n")
                stream.flush()
        return [self.cache[key] for key in keys]

    def score(self, pairs):
        return self._score(pairs)

    def score_with_instruction(self, pairs, instruction, *, max_length=2048):
        return self._score(pairs, instruction, max_length)


class CachedEncoder:
    def __init__(self, model, directory):
        self.model, self.cache, self.directory = model, {}, directory
        self.name, self.revision = model.name, model.revision

    def encode(self, texts):
        import numpy as np
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            for text, vector in zip(missing, self.model.encode(missing), strict=True):
                self.cache[text] = vector
            save(self.directory / "queries.texts.json", list(self.cache))
            np.save(self.directory / "queries.npy", np.stack(list(self.cache.values())))
        return np.stack([self.cache[text] for text in texts])


def inherit_pilot(directory, result, encoder, qwen, out):
    """Reuse only complete, hash-bound exact scores, never missing-score zeros."""
    import numpy as np
    directory = directory.resolve()
    cancellation = read(directory / "cancellation.json")
    parent = read(directory / "report.json")
    if (cancellation.get("format") != "quality_pilot_cancellation/1"
            or cancellation.get("status") != "interrupted_diagnostic"
            or not cancellation.get("gpu_process_stopped")):
        raise ValueError("Parent must be an explicitly stopped, audited diagnostic pilot")
    for name, expected in cancellation["artifact_hashes"].items():
        path = directory / name
        if digest(path) != expected:
            raise ValueError("Pilot cache/report changed after cancellation audit")
        result["input_hashes"][str(path)] = expected
    result["input_hashes"][str(directory / "cancellation.json")] = digest(directory / "cancellation.json")
    for name, expected in parent["source_hashes"].items():
        if digest(directory / "source" / name) != expected:
            raise ValueError("Pilot source snapshot changed")
    for name in ("risk_layer/reranker.py", "risk_layer/embedding.py", "risk_layer/torch_compat.py"):
        if result["source_hashes"][name] != parent["source_hashes"][name]:
            raise ValueError("Pilot uses another model implementation")
    if result["models"] != parent["models"] or result["runtime"] != parent["runtime"]:
        raise ValueError("Pilot model revision, device or package identity differs")
    for path, expected in parent["input_hashes"].items():
        if result["input_hashes"].get(path) != expected or digest(path) != expected:
            raise ValueError("Pilot corpus/reference identity differs")
    for stage, model in (("bge", qwen.vorauswahl), ("qwen", qwen)):
        path = directory / f"{stage}-scores.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (len(row) != 5 or not isinstance(row[4], Real) or isinstance(row[4], bool)
                    or not math.isfinite(row[4]) or not isinstance(row[2], str) or not isinstance(row[3], str)
                    or type(row[1]) is not int or row[1] < 1
                    or row[0] is not None and not isinstance(row[0], str)):
                raise ValueError("Invalid complete pilot score record")
            key = tuple(row[:4])
            if key in model.cache and model.cache[key] != row[4]:
                raise ValueError("Conflicting exact pilot scores")
            model.cache[key] = float(row[4])
        if len(model.cache) != cancellation["cache_pairs"][stage]:
            raise ValueError("Pilot score count differs from audit")
        shutil.copy2(path, model.path)
        model.inherited_pairs = len(model.cache)
    texts = read(directory / "queries.texts.json")
    vectors = np.load(directory / "queries.npy", allow_pickle=False)
    if (len(set(texts)) != len(texts) or vectors.shape != (len(texts), encoder.model.dim)
            or vectors.dtype != np.float32 or not np.isfinite(vectors).all()):
        raise ValueError("Invalid exact pilot query vectors")
    encoder.cache = dict(zip(texts, vectors, strict=True))
    for name in ("queries.texts.json", "queries.npy"):
        shutil.copy2(directory / name, out / name)
    result["inherited_pilot"] = {"path": str(directory), "artifact_hashes": cancellation["artifact_hashes"],
                                 "pairs": cancellation["cache_pairs"], "queries": len(texts),
                                 "note": "Same pinned live model inputs; no labels or fabricated scores."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, default=ROOT / "out/quality-production-final")
    parser.add_argument("--baseline", type=Path, default=ROOT / "out/gpu-ab-20260907-final")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-limit", type=int, help="Diagnostic subset, never a full quality result")
    parser.add_argument("--instruction", choices=["german_full", "english_match", "english_issue"],
                        default="english_match")
    parser.add_argument("--parent-pilot", type=Path)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        parser.error("Choose a fresh output directory")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("case-limit must be positive")
    out.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = {"format": "quality_passage_rerank_probe/1", "status": "running", "rows": [],
              "source_hashes": {}, "input_hashes": {}, "diagnostic_subset": bool(args.case_limit),
              "notes": ["Historical development cases; neither dataset is blind.",
                        "Hit@5 is any labeled target per positive case, not full suggestion precision.",
                        "All variants retain the fixed .5 Qwen threshold and top5 output budget.",
                        "Passage scores select source evidence and candidates, not legal confidence.",
                        "Timing includes run-local caching and is not production latency."]}
    for path in sorted((ROOT / "risk_layer").rglob("*.py")) + [
            ROOT / "catalog/begriffe.yaml", ROOT / "tools/_gpu_eval_metrics.py",
            ROOT / "tools/quality_instruction_probe.py", Path(__file__)]:
        rel = path.relative_to(ROOT)
        target = out / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        result["source_hashes"][rel.as_posix()] = digest(target)
        if digest(path) != digest(target):
            raise ValueError("Source changed during snapshot")
    for path in [args.production / "report.json", args.production / "index/meta.json",
                 args.production / "index/vectors.npy", args.baseline / "artifacts/graph.sqlite",
                 args.baseline / "artifacts/trigger.sqlite"]:
        result["input_hashes"][str(path.resolve())] = digest(path)
    reference = read(args.production / "report.json")
    shutil.copy2(args.production / "report.json", out / "reference.json")
    fields = ("case_id", "group", "query", "targets", "negative", "category", "uncertain", "dataset")
    cases = [{key: row.get(key) for key in fields} for row in reference["rows"]]
    if dict(Counter(case["dataset"] for case in cases)) != {"main": 132, "additional": 30}:
        raise ValueError("Expected all 162 historical reference cases")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    cases = cases[:args.case_limit] if args.case_limit else cases
    sys.path[:0] = [str(out / "source"), str(ROOT)]
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import BgeM3Encoder, EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.reranker import CascadedQwenReranker
    from risk_layer.semantic_search import passage_suggestions, PROFILE_CONFIG
    from risk_layer.triggerindex import TriggerIndex
    from tools._gpu_eval_metrics import summarize
    from tools.quality_instruction_probe import INSTRUCTIONS
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Real local GPU required")
    catalog = lade_katalog(out / "source/catalog/begriffe.yaml")
    index = EmbeddingIndex.load(args.production / "index")
    graph = NormGraph(args.baseline / "artifacts/graph.sqlite")
    trigger = TriggerIndex(args.baseline / "artifacts/trigger.sqlite")
    result["runtime"] = {"gpu": torch.cuda.get_device_name(0), "hip": torch.version.hip,
                         "packages": {name: importlib.metadata.version(name) for name in
                                      ("torch", "transformers", "sentence-transformers", "numpy")}}
    try:
        if index is None or not pruefe_aktualitaet(index, graph, catalog):
            raise ValueError("Incompatible source index/catalog/graph")
        encoder_model = BgeM3Encoder(device="cuda", local_files_only=True)
        qwen_model = CascadedQwenReranker(device="cuda")
        encoder = CachedEncoder(encoder_model, out)
        qwen = CachedScores(qwen_model, out / "qwen-scores.jsonl")
        qwen.vorauswahl = CachedScores(qwen_model.vorauswahl, out / "bge-scores.jsonl")
        result["models"] = {name: {"name": model.name, "revision": model.revision}
                            for name, model in [("encoder", encoder), ("bge", qwen.vorauswahl), ("qwen", qwen)]}
        # Hold instruction and token budget fixed across structural variants.
        # This is the selected instruction control, not the unchanged public
        # kaskade profile. Selection uses only earlier development experiments.
        base = {**PROFILE_CONFIG["kaskade"], "instruction": INSTRUCTIONS[args.instruction]}
        selected = {**base, "passage_rerank": True, "passage_shortlist": "package"}
        variants = {
            "instruction_control": base,
            "coherent50": {**base, "context": "coherent", "shortlist": 50},
            "package20": {**selected, "shortlist": 20},
            "package50": {**selected, "shortlist": 50},
            "coherent_package50": {**selected, "context": "coherent",
                                   "shortlist": 50},
            "coherent_package_lexical50": {**selected, "context": "coherent",
                                           "shortlist": 50, "lexical_pool": 20},
            "coherent_package_graph50": {**selected, "context": "coherent", "shortlist": 50,
                                         "lexical_pool": 20, "citation_pool": 50},
        }
        PROFILE_CONFIG.update(variants)
        result["variants"] = variants
        result["instruction_name"] = args.instruction
        if args.parent_pilot is not None:
            inherit_pilot(args.parent_pilot, result, encoder, qwen, out)
        save(out / "manifest.json", {k: v for k, v in result.items() if k != "rows"})
        print("Loading three co-resident GPU models", flush=True)
        for model in (encoder_model, qwen_model.vorauswahl, qwen_model):
            model._laden()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for i, case in enumerate(cases, 1):
            row = dict(case)
            for name in variants:
                start = time.perf_counter()
                suggestions = passage_suggestions(
                    case["query"], profile=name, laws=(), index=index, encoder=encoder,
                    reranker=qwen, trigger=trigger, graph=graph, chains={}, catalog=catalog)
                row[name] = {"suggestions": suggestions, "elapsed_seconds": time.perf_counter() - start}
                print(f"{i}/{len(cases)} {case['case_id']} {name}: "
                      f"{sum(len(s['normen']) for s in suggestions)} suggestions; "
                      f"BGE={len(qwen.vorauswahl.cache)} Qwen={len(qwen.cache)}", flush=True)
            result["rows"].append(row)
            with (out / "rows.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            save(out / "report.json", result)
        result["summaries"] = {dataset: {
            name: summarize([row for row in result["rows"] if row["dataset"] == dataset], name)
            for name in variants} for dataset in ("main", "additional")}
        result["total_seconds"] = time.perf_counter() - started
        result["scoring"] = {name: {"pairs": len(model.cache), "inherited_pairs": model.inherited_pairs,
                                    "fresh_pairs": len(model.cache) - model.inherited_pairs, "seconds": model.seconds,
                                    "used_pairs": len(model.used),
                                    "inherited_used_pairs": len(model.used) - (len(model.cache) - model.inherited_pairs),
                                    "sha256": digest(model.path)}
                             for name, model in [("bge", qwen.vorauswahl), ("qwen", qwen)]}
        result["query_artifacts"] = {name: digest(out / name)
                                     for name in ("queries.texts.json", "queries.npy")}
        result["runtime"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["runtime"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if any(digest(path) != expected for path, expected in result["input_hashes"].items()):
            raise ValueError("Input changed during execution")
        result["status"] = "complete"
    except Exception as exc:
        result["status"], result["error"] = "failed", repr(exc)
        raise
    finally:
        save(out / "report.json", result)
        graph.close()
        trigger.close()
    for dataset, values in result["summaries"].items():
        for name, summary in values.items():
            p, n = summary["positive"], summary["negative"]
            print(f"{dataset} {name}: hits={(p['hit@5'] or 0) * p['n']:.0f}/{p['n']} "
                  f"FP={(n['false_suggestion_rate'] or 0) * n['n']:.0f}/{n['n']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
