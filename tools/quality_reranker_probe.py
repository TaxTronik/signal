"""Rescore frozen candidate plans with the cached local Qwen3 4B reranker.

Retrieval, query windows, and document packages are unchanged. This is an
exploratory model comparison on development data, not calibrated confidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bge-shortlist", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 16:
        parser.error("batch-size must be between 1 and 16")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from risk_layer.reranker import Qwen3Reranker
    from tools._gpu_eval_metrics import summarize, compare_rows
    from tools._quality_policy import apply_policy, POLICY_SPECS
    if not torch.cuda.is_available():
        raise RuntimeError("Real GPU required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.experiment_dir.resolve()
    plans = read(source / "plans.json")
    bge = read(source / "report.json")
    if any(name not in plans for name in args.configs):
        parser.error("Unknown configuration")
    if args.bge_shortlist < 0:
        parser.error("bge-shortlist must be nonnegative")
    if args.bge_shortlist:
        # First stage has already scored all candidates. This shortlist uses
        # raw BGE scores only, never labels. It bounds the larger model's cost.
        baseline = ROOT / "out/gpu-ab-20260907-final/reranker-scores.json"
        bge_scores = {(x["query"], x["document"]): x["score"] for x in read(baseline)}
        for line in (source / "new-scores.jsonl").read_text(encoding="utf-8").splitlines():
            q, d, value = json.loads(line)
            bge_scores[q, d] = value
        for name in args.configs:
            for case in plans[name]:
                for sentence in case["sentences"]:
                    selected = sorted(range(len(sentence["pairs"])),
                                      key=lambda i: (-bge_scores[tuple(sentence["pairs"][i])],
                                                     sentence["candidates"][i]["norm_id"]))[:args.bge_shortlist]
                    sentence["candidates"] = [sentence["candidates"][i] for i in selected]
                    sentence["pairs"] = [sentence["pairs"][i] for i in selected]
    model = Qwen3Reranker(variante="4B", device="cuda")
    manifest = dict(format="quality_reranker_probe/1", configs=args.configs,
                    bge_shortlist=args.bge_shortlist,
                    empty_cache_every=args.empty_cache_every,
                    model_revision=model.revision, batch_size=args.batch_size,
                    packages={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "numpy")},
                    inputs={name: digest(source / name) for name in ("report.json", "plans.json", "new-scores.jsonl")},
                    source_hashes={name: digest(ROOT / name) for name in
                                   ("risk_layer/reranker.py", "tools/quality_reranker_probe.py",
                                    "tools/_quality_policy.py", "tools/_gpu_eval_metrics.py")},
                    policies={**POLICY_SPECS, "qwen_half": {"rerank_threshold": .5}},
                    notes=["Same preselected candidates and exact query/document strings as BGE.",
                           "Scores are model-specific, not calibrated probabilities of legal correctness."])
    if (out / "manifest.json").exists() and read(out / "manifest.json") != manifest:
        raise ValueError("Changed inputs or runtime; choose a fresh output directory")
    save(out / "manifest.json", manifest)
    pairs = sorted({tuple(pair) for name in args.configs for case in plans[name]
                    for sentence in case["sentences"] for pair in sentence["pairs"]},
                   key=lambda pair: (len(pair[0]) + len(pair[1]), pair))
    scores = {}
    score_path = out / "scores.jsonl"
    if score_path.exists():
        for line in score_path.read_text(encoding="utf-8").splitlines():
            q, d, score = json.loads(line)
            scores[q, d] = score
    missing = [pair for pair in pairs if pair not in scores]
    print(f"Qwen3 4B: {len(missing)} new pairs / {len(pairs)} total", flush=True)
    started = time.perf_counter()
    with score_path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start:start + args.batch_size]
            values = model.score(batch)
            for pair, score in zip(batch, values, strict=True):
                scores[pair] = score
                stream.write(json.dumps([*pair, score], ensure_ascii=False) + "\n")
            stream.flush()
            if args.empty_cache_every and (start // args.batch_size + 1) % args.empty_cache_every == 0:
                torch.cuda.empty_cache()
            if start % (args.batch_size * 25) == 0:
                print(f"Qwen: {start}/{len(missing)} ({time.perf_counter()-started:.1f}s) "
                      f"allocated={torch.cuda.memory_allocated()/2**30:.2f}GiB "
                      f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB", flush=True)
    rows = [{k: v for k, v in row.items() if "/" not in k} for row in bge["rows"]]
    summaries, changes = {}, {}
    for name in args.configs:
        for policy in [*POLICY_SPECS, "qwen_half"]:
            variant = name + "/" + policy
            for row, plan in zip(rows, plans[name], strict=True):
                suggestions = []
                for sentence in plan["sentences"]:
                    candidates = [{**n, "rerank": scores[tuple(pair)]} for n, pair in
                                  zip(sentence["candidates"], sentence["pairs"], strict=True)]
                    if policy == "qwen_half":
                        kept = sorted((n for n in candidates if n["rerank"] >= .5),
                                      key=lambda n: (-n["rerank"], -(n["sim"] or 0), n["norm_id"]))[:5]
                    else:
                        kept = apply_policy(candidates, policy, trigger_ids=set(sentence["trigger_ids"]))
                    c = sentence["context"]
                    if kept:
                        suggestions.append(dict(start=c["focus_start"], end=c["focus_end"], satz=c["focus_text"],
                                                normen=[dict(id=n["norm_id"], sim=n["sim"], rerank=n["rerank"]) for n in kept]))
                row[variant] = {"suggestions": suggestions}
            summaries[variant] = summarize(rows, variant)
            paired = [{**row, "before": old["baseline/baseline_or"], "after": row[variant]}
                      for row, old in zip(rows, bge["rows"], strict=True)]
            changes[variant] = compare_rows(paired)
            p, n = summaries[variant]["positive"], summaries[variant]["negative"]
            print(f"{variant:40} hits={p['hit@5']*p['n']:.0f}/{p['n']} MRR={p['mrr_best_sentence']:.3f} FP={n['false_suggestion_rate']*n['n']:.0f}/{n['n']}", flush=True)
    save(out / "report.json", dict(manifest=manifest, summaries=summaries, changes=changes, rows=rows,
                                   total_seconds=time.perf_counter()-started, score_sha256=digest(score_path),
                                   gpu=torch.cuda.get_device_name(0), pairs=len(pairs)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
