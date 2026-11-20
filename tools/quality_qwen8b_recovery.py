"""Fresh-process full-head recovery of one completed, hash-bound 8B decoder run.

The original failed archive is never changed. No decoder is loaded or run.
Recorded reduced logits/scores remain byte-identical; all original hidden states
are projected through the original full FP16 head with the original batches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FAILED_SHA = "02ad6f4e6c61ce4206a107b4969b2f324bceb134c82d755ff8b36d90b6a98706"
FAILED_ERROR = "ValueError('GPU decoder was not released before full original head validation')"
INHERITED = ("inputs.json", "pairs.json", "tokens.json", "model-download.json", "batches.json",
             "oom_retries.json", "hidden_states.npy", "reduced_head_rows.npy", "reduced_logits.npy", "reduced_scores.npy")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_bytes())


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def guard_original(report, manifest):
    """Only the identified post-decoder guard failure is recoverable here."""
    require(report.get("format") == "quality_qwen8b_probe/1" and report.get("status") == "failed"
            and report.get("error") == FAILED_ERROR, "Not the specifically authorized completed-decoder failure")
    require(report.get("unique_pairs") == 14961 and report.get("progress") == {
        "phase": "reduced_decoder", "completed_pairs": 14961, "planned_pairs": 14961},
        "Original decoder did not finish every fixed input")
    require(report.get("rows") == [] and report.get("head_binding") is None
            and report.get("numerical_audit") is None, "Unexpected original output/head stage")
    runtime = report["runtime"]
    require(runtime.get("post_decoder_allocated_bytes") == 79691776
            and runtime.get("reduced_scoring_seconds", 0) > 0
            and runtime.get("dtype") == "torch.float16"
            and runtime.get("decoder_parameter_devices") == ["cuda:0"]
            and runtime.get("input_embedding_device") == "cpu", "Original runtime identity/failure differs")
    require(report.get("architecture_audit", {}).get("original_parameter_tensors") == 399
            and report.get("model") == {"name": "Qwen/Qwen3-Reranker-8B",
                                       "revision": "77d193c791ed757ca307ee72715aa132723da912"},
            "Original model architecture or revision differs")
    require(manifest.get("format") == report["format"] and manifest.get("status") == "running",
            "Original pre-run manifest is invalid")
    for name, expected in manifest.items():
        if name == "status":
            continue
        if name == "runtime":
            require(all(runtime.get(key) == value for key, value in expected.items()),
                    "Original runtime changed relative to its pre-run manifest")
        else:
            require(report.get(name) == expected, f"Original manifest field differs: {name}")


def check_arrays(hidden, logits, scores, rows, batches, tokens, retries):
    """CPU-only integrity controls; preserve recorded scores, never recalibrate."""
    import numpy as np
    require(hidden.ndim == 2 and rows.shape == (2, hidden.shape[1]) and hidden.shape[0] > 0,
            "Hidden-state/head-row shape differs")
    count = len(hidden)
    require(hidden.dtype == rows.dtype == np.float16 and logits.dtype == scores.dtype == np.float32
            and logits.shape == (count, 2) and scores.shape == (count,), "Recorded array dtype/shape differs")
    require(all(np.isfinite(array).all() for array in (hidden, logits, scores, rows))
            and ((scores >= 0) & (scores <= 1)).all(), "Invalid/nonfinite recorded tensors")
    require(len(tokens) == count and all(isinstance(t, list) and 0 < len(t) <= 2048
            and all(type(x) is int and x >= 0 for x in t) for t in tokens), "Token inventory is incomplete")
    position, initial_size, downgraded = 0, 2, False
    retry_positions = {}
    for retry in retries:
        require(retry.get("phase") == "decoder" and retry.get("attempted_batch_size") == 2
                and retry.get("retry_batch_size") == 1 and type(retry.get("start")) is int,
                "Undeclared original retry")
        require(retry["start"] not in retry_positions, "Duplicate original retry")
        retry_positions[retry["start"]] = retry
    require(len(retry_positions) <= 1, "Only one declared B2-to-B1 downgrade is permitted")
    for batch in batches:
        require(type(batch.get("start")) is int and type(batch.get("count")) is int
                and batch["start"] == position and batch["count"] in (1, 2), "Incomplete/overlapping batch boundaries")
        if position in retry_positions:
            require(not downgraded and position + 1 < count, "Invalid B2 retry location")
            initial_size, downgraded = 1, True
        size = batch["count"]
        require(size == min(initial_size, count - position)
                and type(batch.get("padded_tokens")) is int
                and batch["padded_tokens"] == max(map(len, tokens[position:position + size])),
                "Batch size or original padded length differs")
        position += size
    require(position == count and (not retries or downgraded), "Not all decoder outputs have recorded batches")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    recomputed = exp[:, 1] / exp.sum(axis=1)
    max_score_delta = float(np.abs(recomputed - scores).max())
    require(max_score_delta <= 2e-7, "Reduced raw logits/scores are inconsistent")
    difference = logits[:, 1] - logits[:, 0]
    stable_side = (difference == 0) | (np.abs(difference) > 2e-6)
    require(((scores[stable_side] >= .5) == (difference[stable_side] >= 0)).all(),
            "Recorded score changes the stable no/yes threshold side")
    max_projection_delta, maximum_bound = 0., 0.
    for start in range(0, count, 256):
        states = hidden[start:start + 256].astype(np.float32)
        reference = states @ rows.astype(np.float32).T
        bound = .002 * (np.abs(states) @ np.abs(rows.astype(np.float32)).T) + .002
        delta = np.abs(reference - logits[start:start + 256])
        require((delta <= bound).all(), "Stored hidden states do not reproduce their reduced head logits")
        max_projection_delta = max(max_projection_delta, float(delta.max()))
        maximum_bound = max(maximum_bound, float(bound.max()))
    return {"pairs": count, "batches": len(batches), "oom_retries": len(retries),
            "reduced_raw_scores_preserved": True, "logits_to_score_max_abs_delta": max_score_delta,
            "logits_to_score_tolerance": 2e-7, "hidden_projection_max_abs_delta": max_projection_delta,
            "hidden_projection_max_bound": maximum_bound,
            "hidden_projection_tolerance": "0.002 * sum(abs(hidden * original_fp16_row)) + 0.002 per logit",
            "cpu_projection_scope": "integrity check only; no replacement of measured reduced logits or scores"}


def original_head_rows(snapshot, token_ids):
    """Read only two original BF16 rows directly; conversion is explicit FP16."""
    import numpy as np
    index = read(snapshot / "model.safetensors.index.json")["weight_map"]
    shard = snapshot / index["lm_head.weight"]
    with shard.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        metadata = json.loads(stream.read(header_size))["lm_head.weight"]
        require(metadata["dtype"] == "BF16" and metadata["shape"] == [151669, 4096],
                "Original full head architecture differs")
        data_start = 8 + header_size + metadata["data_offsets"][0]
        values = []
        for token_id in token_ids:
            stream.seek(data_start + token_id * 4096 * 2)
            raw = np.frombuffer(stream.read(4096 * 2), dtype="<u2")
            require(len(raw) == 4096, "Original head row is truncated")
            values.append((raw.astype(np.uint32) << 16).view(np.float32).astype(np.float16))
    return np.stack(values), index["lm_head.weight"]


def token_replay(snapshot, pairs, tokens, base):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False).ids
    prefix, suffix = encode(base.PREFIX), encode(base.SUFFIX)
    token_ids = [tokenizer.token_to_id(word) for word in ("no", "yes")]
    require(token_ids == [2152, 9693], "Original no/yes token IDs differ")
    require(len(tokens) == len(pairs), "Token input count differs")
    for pair, recorded in zip(pairs, tokens, strict=True):
        expected = prefix + encode(f"<Instruct>: {base.INSTRUCTION}\n<Query>: {pair['query']}\n<Document>: {pair['document']}") + suffix
        require(recorded == expected, "Original recorded tokenized input differs")
    return token_ids


def prepare(source, out):
    """Freeze and validate everything before importing Torch or touching GPU."""
    import numpy as np
    report_bytes, manifest_bytes = (source / "report.json").read_bytes(), (source / "manifest.json").read_bytes()
    require(hashlib.sha256(report_bytes).hexdigest() == FAILED_SHA, "Unexpected original failed report")
    original, manifest = json.loads(report_bytes), json.loads(manifest_bytes)
    guard_original(original, manifest)
    inherited = {name: digest(source / name) for name in INHERITED}
    source_hashes = {}
    for name, expected in original["source_hashes"].items():
        path = source / "source" / name
        require(digest(path) == expected, "Original runner/evaluator source changed")
        source_hashes[(Path("source") / name).as_posix()] = expected
    expected_dependencies = set()
    for module_name, metadata in original["runtime"]["dependency_sources"].items():
        require(digest(metadata["original_path"]) == metadata["sha256"], "Installed original model implementation changed")
        relative = f"dependency-source/{module_name}.py"
        require(digest(source / relative) == metadata["sha256"], "Original dependency snapshot changed")
        source_hashes[relative] = metadata["sha256"]
        expected_dependencies.add((source / relative).name)
    require({p.name for p in (source / "dependency-source").glob("*.py")} == expected_dependencies,
            "Unexpected original dependency source inventory")
    require(len(source_hashes) == 11, "Original three runner/evaluator and eight dependency sources are incomplete")
    require(not out.exists(), "Use a fresh immutable recovery directory")
    out.mkdir(parents=True)
    for name in INHERITED:
        shutil.copy2(source / name, out / name)
        require(digest(out / name) == inherited[name], "Inherited decoder artifact changed while copying")
    (out / "original-report.json").write_bytes(report_bytes)
    (out / "original-manifest.json").write_bytes(manifest_bytes)
    for relative in source_hashes:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    own = out / "recovery-source/tools/quality_qwen8b_recovery.py"
    own.parent.mkdir(parents=True)
    shutil.copy2(__file__, own)
    base = module_from_path("frozen_original_qwen8b_runner", out / "source/tools/quality_qwen8b_probe.py")
    require(original["inputs_sha256"] == base.INPUT_SHA256 == inherited["inputs.json"], "Original source-bound inputs differ")
    inputs = read(out / "inputs.json")
    pairs, pair_index = base.frozen_pairs(inputs)
    require(pairs == read(out / "pairs.json"), "Recorded unique pair order differs from original Top50 inputs")
    bound_count = base.check_bound_inputs(inputs)
    model_bytes = (out / "model-download.json").read_bytes()
    require(hashlib.sha256(model_bytes).hexdigest() == original["model_manifest_sha256"], "Original model manifest changed")
    assets, _config, _index = base.validate_model_assets(out / "model-download.json", model_bytes)
    require(assets["files"] == original["model_assets"], "Original model assets differ")
    snapshot = Path(assets["snapshot"])
    tokens = read(out / "tokens.json")
    token_ids = token_replay(snapshot, pairs, tokens, base)
    source_rows, head_shard = original_head_rows(snapshot, token_ids)
    rows = np.load(out / "reduced_head_rows.npy", allow_pickle=False)
    require(np.array_equal(source_rows, rows), "Recorded reduced head is not exactly original BF16-to-FP16 rows")
    hidden = np.load(out / "hidden_states.npy", mmap_mode="r", allow_pickle=False)
    logits = np.load(out / "reduced_logits.npy", allow_pickle=False)
    scores = np.load(out / "reduced_scores.npy", allow_pickle=False)
    require(hidden.shape == (14961, 4096), "Original hidden-state matrix is incomplete")
    batches, retries = read(out / "batches.json"), read(out / "oom_retries.json")
    preflight = check_arrays(hidden, logits, scores, rows, batches, tokens, retries)
    preflight.update(original_bound_input_files=bound_count, original_model_assets=len(assets["files"]),
                     exact_tokenized_pairs=len(pairs), exact_original_head_rows=True, torch_imported=False)
    require("torch" not in sys.modules, "Preflight unexpectedly imported Torch")
    report = copy.deepcopy(original)
    report.update(status="running", rows=[], progress={"phase": "full_head_recovery", "completed_pairs": 0, "planned_pairs": 14961})
    report.pop("error", None)
    report["recovery"] = {
        "format": "quality_qwen8b_recovery/1", "source_directory": str(source),
        "original_report_sha256": FAILED_SHA, "original_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "original_status": "failed", "original_error": FAILED_ERROR,
        "original_post_decoder_allocated_bytes": original["runtime"]["post_decoder_allocated_bytes"],
        "fresh_process": True, "full_head_only": True, "decoder_rerun": False,
        "inherited_artifact_hashes": inherited, "inherited_source_hashes": source_hashes,
        "source_hashes": {"tools/quality_qwen8b_recovery.py": digest(own)}, "preflight": preflight,
        "runtime": {},
        "cause": "Original decoder tensors occupied about 12.94 GiB; the post-release guard found 76 MiB. The terminated process cannot retrospectively distinguish retained runtime allocations from tensor references. A fresh head-only process avoids that ambiguity without changing the old guard or any decoder result.",
    }
    save(out / "recovery-manifest.json", copy.deepcopy(report["recovery"]))
    return report, original, base, inputs, pairs, pair_index, assets, head_shard, token_ids, hidden, logits, scores, batches


def full_head_gpu(report, out, base, assets, head_shard, token_ids, hidden, batches):
    import numpy as np
    import torch
    from safetensors import safe_open
    require({n: importlib.metadata.version(n) for n in base.GPU_PACKAGES} == base.GPU_PACKAGES,
            "Fresh GPU package versions differ from the original audited runtime")
    require(torch.cuda.is_available(), "Real GPU required; CPU fallback is forbidden")
    require(torch.cuda.get_device_name(0) == report["runtime"]["gpu"], "Fresh GPU differs from original decoder GPU")
    torch.backends.cuda.matmul.allow_tf32 = report["runtime"]["matmul_allow_tf32"]
    torch.use_deterministic_algorithms(report["runtime"]["deterministic_algorithms"])
    runtime = report["recovery"]["runtime"]
    runtime.update(pid=os.getpid(), python=sys.version, gpu=torch.cuda.get_device_name(0),
                   packages={n: importlib.metadata.version(n) for n in base.GPU_PACKAGES},
                   hip=torch.version.hip, cuda=torch.version.cuda, dtype="torch.float16",
                   device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                   initial_allocated_bytes=torch.cuda.memory_allocated(),
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    require(runtime["initial_allocated_bytes"] == 0, "Fresh head-only process already owns CUDA tensors")
    start = time.perf_counter()
    with safe_open(str(Path(assets["snapshot"]) / head_shard), framework="pt", device="cpu") as stream:
        full_cpu = stream.get_tensor("lm_head.weight").to(dtype=torch.float16)
    require(full_cpu.shape == (151669, 4096) and torch.isfinite(full_cpu).all(), "Invalid original full FP16 head")
    source_rows = torch.from_numpy(np.load(out / "reduced_head_rows.npy", allow_pickle=False))
    require(torch.equal(full_cpu[token_ids], source_rows), "Full GPU source head rows differ from recorded original rows")
    full_gpu = full_cpu.to("cuda")
    torch.cuda.synchronize()
    runtime["head_load_seconds"] = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    count = len(hidden)
    full_logits, full_scores = np.zeros((count, 2), dtype=np.float32), np.zeros(count, dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for number, batch in enumerate(batches, 1):
            offset, size = batch["start"], batch["count"]
            states = torch.from_numpy(hidden[offset:offset + size].copy()).unsqueeze(1).to("cuda")
            vocabulary_logits = torch.nn.functional.linear(states, full_gpu)
            selected = vocabulary_logits[:, -1, token_ids].float()
            full_logits[offset:offset + size] = selected.cpu().numpy()
            full_scores[offset:offset + size] = base.probability(selected).cpu().numpy()
            if number % 250 == 0 or number == len(batches):
                report["progress"]["completed_pairs"] = offset + size
                save(out / "report.json", report)
                print(f"original full head {offset + size}/{count}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    runtime.update(full_head_validation_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), full_gpu_projected_pairs=count)
    report["runtime"]["full_head_validation_seconds"] = elapsed
    np.save(out / "full_gpu_logits.npy", full_logits, allow_pickle=False)
    np.save(out / "full_gpu_scores.npy", full_scores, allow_pickle=False)
    report["head_binding"] = {
        "hidden_states_sha256": digest(out / "hidden_states.npy"), "pairs_sha256": digest(out / "pairs.json"),
        "batches_sha256": digest(out / "batches.json"), "observed_pairs": count,
        "full_gpu_projected_pairs": count, "same_batch_boundaries": True,
        "original_head_shape": [151669, 4096], "reduced_head_shape": [2, 4096],
        "original_head_fp16_sha256": hashlib.sha256(full_cpu.numpy().tobytes()).hexdigest(),
        "reduced_head_fp16_sha256": hashlib.sha256(source_rows.numpy().tobytes()).hexdigest(),
        "bias": None, "dtype": "torch.float16", "full_vocabulary_projection": True,
    }
    return full_logits, full_scores


def finish_integrity(report, source, out, base, inputs):
    require(digest(source / "report.json") == FAILED_SHA, "Original failed report changed")
    require(digest(source / "manifest.json") == report["recovery"]["original_manifest_sha256"], "Original manifest changed")
    for name, expected in report["recovery"]["inherited_artifact_hashes"].items():
        require(digest(source / name) == expected == digest(out / name), "Inherited decoder result was changed")
    for relative, expected in report["recovery"]["inherited_source_hashes"].items():
        require(digest(source / relative) == expected == digest(out / relative), "Original implementation source changed")
    for metadata in report["runtime"]["dependency_sources"].values():
        require(digest(metadata["original_path"]) == metadata["sha256"], "Installed model implementation changed")
    require(digest(__file__) == report["recovery"]["source_hashes"]["tools/quality_qwen8b_recovery.py"],
            "Recovery implementation changed during execution")
    base.check_bound_inputs(inputs)
    base.validate_model_assets(out / "model-download.json", (out / "model-download.json").read_bytes())
    report["artifact_hashes"] = {p.name: digest(p) for p in sorted(out.iterdir())
                                  if p.is_file() and p.name != "report.json"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "out/quality-qwen8b-20260907-r1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    source, out = args.source.resolve(), args.out.resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    state = prepare(source, out)
    report, original, base, inputs, pairs, pair_index, assets, shard, token_ids, hidden, reduced_logits, reduced_scores, batches = state
    try:
        if args.preflight_only:
            report["status"] = "recovery_preflight_pass"
            report["gpu_used"] = False
        else:
            full_logits, full_scores = full_head_gpu(report, out, base, assets, shard, token_ids, hidden, batches)
            report["numerical_audit"] = base.numerical_comparison(reduced_logits, full_logits, reduced_scores, full_scores)
            metrics = module_from_path("frozen_qwen8b_recovery_metrics", out / "source/tools/_gpu_eval_metrics.py")
            arrays = {"reduced": reduced_scores, "full_gpu": full_scores}
            rows = [{k: v for k, v in case.items() if k != "groups"} for case in inputs["cases"]]
            for name, (head, size) in base.VARIANTS.items():
                values = base.build_outputs(inputs["cases"], pair_index, arrays[head], size)
                for row, value in zip(rows, values, strict=True):
                    row[name] = {"suggestions": value["suggestions"]}
            report["rows"] = rows
            report["summaries"] = {dataset: {name: metrics.summarize([r for r in rows if r["dataset"] == dataset], name)
                                              for name in base.VARIANTS} for dataset in ("main", "additional")}
            differences = []
            for size in (20, 50):
                for row in rows:
                    reduced = row[f"qwen8b_reduced_top{size}"]["suggestions"]
                    full = row[f"qwen8b_full_gpu_top{size}"]["suggestions"]
                    def order(values):
                        return [(s["start"], s["end"], [n["id"] for n in s["normen"]]) for s in values]
                    if reduced != full:
                        differences.append({"case_id": row["case_id"], "dataset": row["dataset"],
                                            "shortlist": size, "norm_order_equal": order(reduced) == order(full)})
            report["output_differences"] = differences
            report["status"] = "complete" if report["numerical_audit"]["exact_scores"] else "mismatch"
            report["gpu_used"] = True
        require({k: v for k, v in report["runtime"].items() if k != "full_head_validation_seconds"} == original["runtime"],
                "Original decoder runtime metadata was changed")
        finish_integrity(report, source, out, base, inputs)
    except Exception as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        save(out / "report.json", report)
    print(json.dumps({"status": report["status"], "recovery": True,
                      "numerical_audit": report.get("numerical_audit")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] in ("complete", "recovery_preflight_pass") else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
