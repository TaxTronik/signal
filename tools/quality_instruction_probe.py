"""Fresh GPU instruction experiment on source-bound frozen candidate inputs.

Candidate selection contains no labels. Historical labels are read only by the
final evaluator. These are development experiments, not blind accuracy claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = {
    "german_full": (
        "Beurteile, ob die Rechtsnorm (Document) den im Sachverhalt (Query) "
        "beschriebenen Rechtsbegriff/Sachverhalt trifft."
    ),
    "english_match": (
        "Judge whether the legal provision (Document) matches the legal concept "
        "or factual situation described in the Query."
    ),
    "english_retrieval": (
        "Given a factual situation, retrieve legal provisions that directly "
        "govern or define the legal relationships, transactions, or obligations "
        "described. A provision is relevant when its actual rule helps assess "
        "the stated facts; a shared topic or isolated word is insufficient."
    ),
    "english_issue": (
        "Given a tax or legal fact pattern, retrieve the legal rule needed for "
        "a substantive assessment of those facts. Match concrete taxable "
        "transactions, legal classifications, entitlements, obligations or "
        "disputed treatment. Routine administrative status updates, greetings, "
        "scheduling, and statements that merely confirm compliant completion "
        "do not require a match. Mere vocabulary overlap is insufficient."
    ),
}


def read(path):
    return Path(path).read_text(encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shortlist", type=int, choices=[20, 50], default=20)
    parser.add_argument("--instructions", nargs="+", choices=list(INSTRUCTIONS),
                        default=["english_match", "english_retrieval"])
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        parser.error("Use a fresh immutable output directory")
    input_bytes = args.inputs.read_bytes()
    inputs = json.loads(input_bytes)
    instructions = {name: INSTRUCTIONS[name] for name in dict.fromkeys(args.instructions)}
    cases = inputs["cases"]
    if (inputs.get("format") != "quality_recall_inputs/1" or inputs.get("status") != "complete"
            or len(cases) != 162 or len({c["case_id"] for c in cases}) != 162
            or Counter(c["dataset"] for c in cases) != Counter({"main": 132, "additional": 30})):
        raise ValueError("Expected all 162 historical cases")
    for case in cases:
        for group in case["groups"]:
            candidates = group["candidates"]
            if ([c["id"] for c in candidates] != [c["id"] for c in sorted(
                    candidates, key=lambda c: (-c["bge"], c["id"]))]
                    or len({c["id"] for c in candidates}) != len(candidates)
                    or len(candidates) != group["top50_count"]
                    or min(len(candidates), 20) != group["top20_count"]):
                raise ValueError("Candidate selection differs from frozen BGE shortlist")
            for rank, candidate in enumerate(candidates, 1):
                if (candidate["bge_rank"] != rank
                        or hashlib.sha256(candidate["document"].encode()).hexdigest() != candidate["document_sha256"]
                        or not candidate["document"].startswith(candidate["id"] + "\n")):
                    raise ValueError("Candidate document/rank identity is invalid")
    out.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    source_hashes = {}
    for path in [*sorted((ROOT / "risk_layer").glob("*.py")),
                 ROOT / "tools/_gpu_eval_metrics.py", Path(__file__).resolve()]:
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(path.relative_to(ROOT))] = digest(target)
        if digest(path) != digest(target):
            raise ValueError("Source changed while snapshotting")
    (out / "inputs.json").write_bytes(input_bytes)
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    sys.path[:0] = [str(out / "source"), str(ROOT)]
    from risk_layer.reranker import Qwen3Reranker
    from tools._gpu_eval_metrics import summarize
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Real local GPU required")
    model = Qwen3Reranker(device="cuda")
    if inputs["models"]["qwen"]["revision"] != model.revision:
        raise ValueError("Frozen candidate scores belong to another Qwen revision")
    manifest = {"format": "quality_instruction_probe/1", "status": "running",
                "inputs_sha256": input_hash, "source_hashes": source_hashes,
                "instructions": instructions, "shortlist": args.shortlist,
                "threshold": .5, "topk": 5, "max_length": 2048,
                "model": {"name": model.name, "revision": model.revision},
                "runtime": {"gpu": torch.cuda.get_device_name(0), "hip": torch.version.hip,
                            "packages": {n: importlib.metadata.version(n) for n in
                                         ("torch", "transformers", "numpy")}},
                "notes": ["All cases are previously used development data.",
                          "Only final metric calculation uses historical target labels.",
                          "No extra score thresholds were searched.",
                          "Timing excludes retrieval/BGE and uses exact run-local caching."],
                "rows": []}
    save(out / "manifest.json", {k: v for k, v in manifest.items() if k != "rows"})
    cache = {}
    started = time.perf_counter()
    try:
        model._laden()
        manifest["runtime"]["dtype"] = str(next(model._model.parameters()).dtype)
        with (out / "scores.jsonl").open("a", encoding="utf-8") as log:
            for position, case in enumerate(cases, 1):
                row = {k: v for k, v in case.items() if k != "groups"}
                for name, instruction in instructions.items():
                    suggestions = []
                    for group in case["groups"]:
                        candidates = group["candidates"][:args.shortlist]
                        pairs = [(group["query"], c["document"]) for c in candidates]
                        keys = [(instruction, 2048, q, d) for q, d in pairs]
                        missing = list(dict.fromkeys(k for k in keys if k not in cache))
                        for offset in range(0, len(missing), 2):
                            batch = missing[offset:offset + 2]
                            scores = model.score_with_instruction(
                                [(k[2], k[3]) for k in batch], instruction, max_length=2048)
                            if len(scores) != len(batch) or any(
                                    not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
                                raise ValueError("Invalid GPU scores")
                            for key, value in zip(batch, scores, strict=True):
                                cache[key] = value
                                log.write(json.dumps([*key, value], ensure_ascii=False) + "\n")
                            log.flush()
                        eligible = [{"id": c["id"], "sim": c["sim"], "rerank": cache[key]}
                                    for c, key in zip(candidates, keys, strict=True) if cache[key] >= .5]
                        eligible.sort(key=lambda c: (-c["rerank"],
                                                    -(c["sim"] if c["sim"] is not None else -math.inf), c["id"]))
                        if eligible:
                            suggestions.append({"start": group["start"], "end": group["end"],
                                                "normen": eligible[:5]})
                    row[name] = {"suggestions": suggestions}
                manifest["rows"].append(row)
                save(out / "report.json", manifest)
                print(f"{position}/162 {case['case_id']}; unique GPU pairs={len(cache)}", flush=True)
        manifest["summaries"] = {dataset: {
            name: summarize([r for r in manifest["rows"] if r["dataset"] == dataset], name)
            for name in instructions} for dataset in ("main", "additional")}
        manifest["seconds"] = time.perf_counter() - started
        manifest["gpu_pairs"] = len(cache)
        manifest["scores_sha256"] = digest(out / "scores.jsonl")
        if digest(args.inputs) != input_hash:
            raise ValueError("Input changed during execution")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", repr(exc)
        raise
    finally:
        save(out / "report.json", manifest)
    for dataset, variants in manifest["summaries"].items():
        for name, metrics in variants.items():
            p, n = metrics["positive"], metrics["negative"]
            print(f"{dataset} {name}: hits={p['hit@5'] * p['n']:.0f}/{p['n']} "
                  f"FP={n['false_suggestion_rate'] * n['n']:.0f}/{n['n']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
