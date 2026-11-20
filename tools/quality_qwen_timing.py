"""Small real-GPU batching probe on frozen Qwen query/document pairs."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")


def main():
    import torch
    from risk_layer.reranker import Qwen3Reranker
    root = ROOT / "out/quality-experiments-20260907-r2"
    plans = json.loads((root / "plans.json").read_text(encoding="utf-8"))
    pairs = sorted({tuple(pair) for case in plans["passage_package_200"]
                    for sentence in case["sentences"] for pair in sentence["pairs"]},
                   key=lambda pair: (len(pair[0]) + len(pair[1]), pair))
    # Fixed medium/long source lengths, chosen without reference to scores/Gold.
    selected = [pair for pair in pairs if 1800 <= sum(map(len, pair)) <= 1900][:8]
    model = Qwen3Reranker(device="cuda")
    model._laden()
    print({"device": str(next(model._model.parameters()).device),
           "dtype": str(next(model._model.parameters()).dtype),
           "attention": model._model.config._attn_implementation}, flush=True)
    rows = []
    for batch_size in (1, 8, 2, 4, 1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = []
        for start in range(0, len(selected), batch_size):
            scores.extend(model.score(selected[start:start + batch_size]))
        torch.cuda.synchronize()
        row = dict(batch_size=batch_size, seconds=time.perf_counter()-started, scores=scores,
                   peak_allocated=torch.cuda.max_memory_allocated())
        rows.append(row)
        print(row, flush=True)
    path = ROOT / "out/quality-qwen-timing.json"
    path.write_text(json.dumps(dict(pairs=selected, rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
