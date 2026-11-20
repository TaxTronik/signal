"""Frozen Top20/Top50 Qwen3-8B experiment with an audited two-row output head.

No production model is changed. Only final metric construction sees historical
labels. The original decoder is streamed without quantization; its embedding
stays on CPU. A second GPU phase projects every recorded final hidden state
through the original full FP16 head. Numerical differences retain BOTH arms.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-Reranker-8B"
REVISION = "77d193c791ed757ca307ee72715aa132723da912"
INPUT_SHA256 = "145e1edfe2ab2cda1f54ed20889be2395ae1bf32d2c97c5f4d632e177edd9e30"
GPU_PACKAGES = {"torch": "2.9.1+rocm7.2.1", "transformers": "5.15.1",
                "numpy": "2.2.6", "safetensors": "0.8.0"}
GPU_QWEN_SOURCE_SHA256 = "cbb7f2dc274c2f5592746c0dc6985ca50353efa07376f92cc922b77680a74f69"
INSTRUCTION = (
    "Judge whether the legal provision (Document) matches the legal concept "
    "or factual situation described in the Query."
)
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based '
    'on the Query and the Instruct provided. Note that the answer can only be '
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
VARIANTS = {
    "qwen8b_reduced_top20": ("reduced", 20),
    "qwen8b_reduced_top50": ("reduced", 50),
    "qwen8b_full_gpu_top20": ("full_gpu", 20),
    "qwen8b_full_gpu_top50": ("full_gpu", 50),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def validate_architecture(config):
    """Fail closed for the exact pinned, untied, dense original architecture."""
    expected = {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "tie_word_embeddings": False, "attention_bias": False,
        "hidden_size": 4096, "intermediate_size": 12288,
        "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151669,
        "rope_theta": 1000000, "rope_scaling": None,
        "use_sliding_window": False,
    }
    for key, value in expected.items():
        if key not in config or config[key] != value:
            raise ValueError(f"Unsupported pinned architecture: {key}")
    if config.get("quantization_config") is not None:
        raise ValueError("Quantized checkpoints are not permitted")


def checkpoint_shapes(config):
    """Original parameter inventory, also usable with small CPU test configs."""
    h, f, v = (config[k] for k in ("hidden_size", "intermediate_size", "vocab_size"))
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    shapes = {"model.embed_tokens.weight": (v, h), "model.norm.weight": (h,),
              "lm_head.weight": (v, h)}
    layer = {
        "self_attn.q_proj.weight": (q, h), "self_attn.k_proj.weight": (kv, h),
        "self_attn.v_proj.weight": (kv, h), "self_attn.o_proj.weight": (h, q),
        "self_attn.q_norm.weight": (config["head_dim"],),
        "self_attn.k_norm.weight": (config["head_dim"],),
        "mlp.gate_proj.weight": (f, h), "mlp.up_proj.weight": (f, h),
        "mlp.down_proj.weight": (h, f), "input_layernorm.weight": (h,),
        "post_attention_layernorm.weight": (h,),
    }
    for index in range(config["num_hidden_layers"]):
        shapes.update({f"model.layers.{index}.{name}": shape for name, shape in layer.items()})
    return shapes


def reduce_linear_head(original_linear, token_ids):
    """Copy exactly the original no/yes rows in their explicit order, plus bias."""
    import torch
    if (len(token_ids) != 2 or len(set(token_ids)) != 2
            or any(not isinstance(i, int) or i < 0 or i >= original_linear.out_features
                   for i in token_ids)):
        raise ValueError("Expected distinct valid original no/yes token IDs")
    result = torch.nn.Linear(original_linear.in_features, 2,
                             bias=original_linear.bias is not None,
                             device=original_linear.weight.device,
                             dtype=original_linear.weight.dtype)
    with torch.no_grad():
        result.weight.copy_(original_linear.weight[list(token_ids)])
        if result.bias is not None:
            result.bias.copy_(original_linear.bias[list(token_ids)])
    return result.requires_grad_(False).eval()


def materialize_rope(model, config, device):
    """Use original CPU FP32 initialization, then transfer without a dtype cast."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    with torch.device("cpu"):
        rope = Qwen3RotaryEmbedding(config)
    model.model.rotary_emb = rope.to(device=device)
    for name in ("inv_freq", "original_inv_freq"):
        value = getattr(model.model.rotary_emb, name)
        if value.is_meta or value.dtype != torch.float32:
            raise ValueError(f"Original FP32 rotary buffer not materialized: {name}")


def build_outputs(cases, pair_index, scores, shortlist):
    """Labels are copied for evaluation only; ranking uses frozen pairs and scores."""
    if shortlist not in (20, 50):
        raise ValueError("Only preregistered Top20 and Top50 arms are permitted")
    rows = []
    for case in cases:
        row = {k: v for k, v in case.items() if k != "groups"}
        suggestions = []
        for group in case["groups"]:
            eligible = []
            for candidate in group["candidates"][:shortlist]:
                score = float(scores[pair_index[(group["query"], candidate["document"])]])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid model score; no case may be skipped")
                if score >= .5:
                    eligible.append({"id": candidate["id"], "sim": candidate["sim"],
                                     "rerank": score})
            eligible.sort(key=lambda c: (-c["rerank"],
                                        -(c["sim"] if c["sim"] is not None else -math.inf),
                                        c["id"]))
            if eligible:
                suggestions.append({"start": group["start"], "end": group["end"],
                                    "normen": eligible[:5]})
        row["suggestions"] = suggestions
        rows.append(row)
    return rows


def frozen_pairs(inputs):
    """Produce model inputs only: no case IDs, labels, targets, or source routing."""
    cases = inputs.get("cases", [])
    if (inputs.get("format") != "quality_recall_inputs/1" or inputs.get("status") != "complete"
            or len(cases) != 162 or len({c["case_id"] for c in cases}) != 162
            or Counter(c["dataset"] for c in cases) != Counter(main=132, additional=30)):
        raise ValueError("Expected all 162 source-bound historical cases")
    pairs, index = [], {}
    for case in cases:
        for group in case["groups"]:
            if case["query"][group["start"]:group["end"]] != group["query"]:
                raise ValueError("Focus offsets do not reproduce the original query")
            candidates = group["candidates"]
            if (len(candidates) != group["top50_count"] or len(candidates) > 50
                    or min(len(candidates), 20) != group["top20_count"]
                    or len({c["id"] for c in candidates}) != len(candidates)
                    or candidates != sorted(candidates, key=lambda c: (-c["bge"], c["id"]))):
                raise ValueError("Frozen BGE-ranked Top50 candidate list is invalid")
            for rank, candidate in enumerate(candidates, 1):
                document = candidate["document"]
                if (candidate["bge_rank"] != rank or not math.isfinite(candidate["bge"])
                        or not document.startswith(candidate["id"] + "\n")
                        or hashlib.sha256(document.encode()).hexdigest() != candidate["document_sha256"]):
                    raise ValueError("Candidate package hash or original BGE rank differs")
                key = (group["query"], document)
                if key not in index:
                    index[key] = len(pairs)
                    pairs.append({"query": key[0], "document": key[1]})
    return pairs, index


def check_bound_inputs(inputs):
    count = 0
    for mapping in (inputs["source_hashes"], inputs["input_hashes"],
                    {str(Path(inputs["source_run"]) / "report.json"): inputs["source_report_sha256"]}):
        for path, expected in mapping.items():
            if digest(path) != expected:
                raise ValueError(f"Frozen source/input hash mismatch: {path}")
            count += 1
    return count


def validate_model_assets(manifest_path, expected_bytes=None):
    manifest_bytes = Path(manifest_path).read_bytes()
    if expected_bytes is not None and manifest_bytes != expected_bytes:
        raise ValueError("Model download manifest changed after its initial snapshot")
    manifest = json.loads(manifest_bytes)
    if (manifest.get("format") != "quality_model_download/1"
            or manifest.get("status") != "complete" or manifest.get("repo") != MODEL
            or manifest.get("revision") != REVISION):
        raise ValueError("Expected completed download manifest for pinned original Qwen3-8B")
    snapshot = Path(manifest["snapshot"]).resolve()
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
    if not required <= set(manifest["files"]):
        raise ValueError("Required model/tokenizer metadata is not hash-bound")
    for optional in ("generation_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"):
        if (snapshot / optional).exists() and optional not in manifest["files"]:
            raise ValueError("Present model/tokenizer metadata is not hash-bound")
    for name, entry in manifest["files"].items():
        path = (snapshot / name).resolve()
        # HF snapshot files may resolve to its blob store through links. The
        # relative manifest name itself must nevertheless stay within snapshot.
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid model asset relative path")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Pinned model asset has changed: {name}")
    config = json.loads((snapshot / "config.json").read_bytes())
    validate_architecture(config)
    index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
    expected = checkpoint_shapes(config)
    if set(index) != set(expected) or len(index) != 399:
        raise ValueError("Original checkpoint parameter inventory is not exactly 399")
    seen = {}
    for shard in sorted(set(index.values())):
        if shard not in manifest["files"]:
            raise ValueError("Checkpoint shard is absent from the pinned download manifest")
        with (snapshot / shard).open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if header_length > 16_000_000:
                raise ValueError("Unexpected safetensors header size")
            header = json.loads(stream.read(header_length))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if (name in seen or index.get(name) != shard or tensor["dtype"] != "BF16"
                    or tuple(tensor["shape"]) != expected.get(name)):
                raise ValueError(f"Original dense BF16 tensor identity is invalid: {name}")
            seen[name] = tensor["shape"]
    if set(seen) != set(expected):
        raise ValueError("Incomplete original checkpoint shard inventory")
    return manifest, config, index


def tokenize_pairs(tokenizer, pairs):
    if tokenizer.padding_side != "left" or tokenizer.pad_token_id is None:
        raise ValueError("Original tokenizer requires explicit left padding and a pad token")
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    token_ids = [tokenizer.convert_tokens_to_ids(word) for word in ("no", "yes")]
    if (len(set(token_ids)) != 2 or any(i is None or i == tokenizer.unk_token_id for i in token_ids)
            or any(tokenizer.encode(word, add_special_tokens=False) != [idx]
                   for word, idx in zip(("no", "yes"), token_ids, strict=True))):
        raise ValueError("Original no/yes tokenizer IDs are not distinct single tokens")
    encoded = []
    for pair in pairs:
        middle = tokenizer.encode(
            f"<Instruct>: {INSTRUCTION}\n<Query>: {pair['query']}\n<Document>: {pair['document']}",
            add_special_tokens=False, truncation=False)
        tokens = prefix + middle + suffix
        if len(tokens) > 2048:
            raise ValueError("Exact frozen input exceeds 2048 tokens; truncation is forbidden")
        encoded.append(tokens)
    return encoded, token_ids, {"prefix_ids": prefix, "suffix_ids": suffix,
                               "no_token_id": token_ids[0], "yes_token_id": token_ids[1],
                               "padding_side": tokenizer.padding_side, "pad_token_id": tokenizer.pad_token_id,
                               "max_observed_tokens": max(map(len, encoded))}


def load_streamed_model(snapshot, raw_config, index, token_ids, device):
    """Load original tensors once each, with no initialization after assignment."""
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config.from_dict(raw_config)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    shapes = {name: tuple(value.shape) for name, value in model.named_parameters()}
    if shapes != checkpoint_shapes(raw_config):
        raise ValueError("Installed original Transformers architecture differs from checkpoint")
    full_head = None
    for shard in sorted(set(index.values())):
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as stream:
            for name in sorted(k for k, file in index.items() if file == shard):
                tensor = stream.get_tensor(name).to(dtype=torch.float16)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite original weight after FP16 conversion: {name}")
                if name == "lm_head.weight":
                    full_head = tensor
                    continue
                target_device = "cpu" if name == "model.embed_tokens.weight" else device
                module_name, parameter_name = name.rsplit(".", 1)
                module = model.get_submodule(module_name)
                module._parameters[parameter_name] = torch.nn.Parameter(
                    tensor.to(device=target_device), requires_grad=False)
                del tensor
    if full_head is None:
        raise ValueError("Original untied full output head is missing")
    with torch.device("meta"):
        original = torch.nn.Linear(raw_config["hidden_size"], raw_config["vocab_size"], bias=False)
    original.weight = torch.nn.Parameter(full_head, requires_grad=False)
    reduced = reduce_linear_head(original, token_ids)
    if not torch.equal(reduced.weight, full_head[token_ids]):
        raise ValueError("Reduced head is not exactly original FP16 no/yes rows")
    model.lm_head = reduced.to(device=device)
    del original, reduced
    materialize_rope(model, config, device)
    for name, value in model.named_parameters():
        expected_device = "cpu" if name == "model.embed_tokens.weight" else str(device).split(":")[0]
        if value.is_meta or value.dtype != torch.float16 or value.device.type != expected_device:
            raise ValueError(f"Unexpected final parameter dtype/device: {name}")
    for name, value in model.named_buffers():
        if (value.is_meta or value.dtype != torch.float32
                or value.device.type != str(device).split(":")[0]
                or not name.startswith("model.rotary_emb.")):
            raise ValueError(f"Unexpected non-checkpoint model buffer: {name}")
    return model.eval().requires_grad_(False), full_head


def preflight_structure(raw_config):
    """Instantiate only meta parameters and the tiny original CPU rotary buffer."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config.from_dict(raw_config)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    shapes = {name: tuple(value.shape) for name, value in model.named_parameters()}
    if shapes != checkpoint_shapes(raw_config):
        raise ValueError("Installed original meta architecture differs from the exact checkpoint")
    materialize_rope(model, config, "cpu")
    buffers = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                      "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
               for name, value in model.named_buffers()}
    sizes = {name: math.prod(shape) * 2 for name, shape in shapes.items()}
    return {"original_parameter_tensors": len(shapes), "all_parameters_meta": all(p.is_meta for p in model.parameters()),
            "original_fp16_bytes": sum(sizes.values()),
            "decoder_fp16_bytes": sum(n for key, n in sizes.items() if key not in
                                      ("model.embed_tokens.weight", "lm_head.weight")),
            "cpu_embedding_fp16_bytes": sizes["model.embed_tokens.weight"],
            "original_head_fp16_bytes": sizes["lm_head.weight"],
            "reduced_head_fp16_bytes": raw_config["hidden_size"] * 2 * 2,
            "original_cpu_rotary_buffers": buffers}


def retry_batches(count, initial_batch_size, score_batch, oom_type, on_oom):
    """A single declared B2-to-B1 fallback; no skipped or duplicated pair outputs."""
    if initial_batch_size not in (1, 2):
        raise ValueError("Only declared B2/B1 batch sizes are permitted")
    position, batch_size = 0, initial_batch_size
    while position < count:
        size = min(batch_size, count - position)
        try:
            result = score_batch(position, size)
        except oom_type as exc:
            if batch_size == 1:
                raise
            message = str(exc)
            exc.__traceback__ = None  # release failed GPU frame before cache cleanup
            on_oom(position, size, message)
            batch_size = 1
            continue
        yield position, size, result
        position += size


def probability(logits):
    import torch
    if logits.shape[-1] != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite original no/yes logit pairs")
    return torch.softmax(logits.float(), dim=-1)[..., 1]


def numerical_comparison(reduced_logits, full_logits, reduced_scores, full_scores):
    import numpy as np
    delta = np.abs(reduced_scores.astype("float64") - full_scores.astype("float64"))
    return {
        "pairs": len(delta), "different_logit_pairs": int(np.any(reduced_logits != full_logits, axis=1).sum()),
        "different_scores": int((delta != 0).sum()), "max_abs_score_difference": float(delta.max(initial=0)),
        "max_abs_logit_difference": float(np.abs(reduced_logits - full_logits).max(initial=0)),
        "threshold_crossings": int(((reduced_scores >= .5) != (full_scores >= .5)).sum()),
        "exact_scores": bool(np.array_equal(reduced_scores, full_scores)),
    }


def cpu_head_audit(full_head, hidden, token_ids):
    """Fixed first two observed states, CPU arithmetic only; never a GPU proxy."""
    import torch
    states = torch.from_numpy(hidden[:min(2, len(hidden))].copy()).unsqueeze(1)
    original = torch.nn.functional.linear(states, full_head)[..., token_ids].float()
    reduced = torch.nn.functional.linear(states, full_head[token_ids]).float()
    float32_rows = torch.nn.functional.linear(states.float(), full_head[token_ids].float())
    return {"device": "cpu", "observed_pair_indices": list(range(len(states))),
            "original_head_dtype": "torch.float16", "reduced_head_dtype": "torch.float16",
            "original_fp16_logits": original[:, 0].tolist(),
            "reduced_fp16_logits": reduced[:, 0].tolist(),
            "fp32_row_reference_logits": float32_rows[:, 0].tolist(),
            "max_abs_fp16_difference": float((original - reduced).abs().max()),
            "scope": "CPU projection of actual observed states; separate from full GPU audit"}


def snapshot_source(out):
    paths = [Path(__file__).resolve(), ROOT / "tools/_gpu_eval_metrics.py", ROOT / "risk_layer/torch_compat.py"]
    hashes = {}
    for path in paths:
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path.relative_to(ROOT))] = digest(target)
        if digest(path) != digest(target):
            raise ValueError("Runner source changed while freezing")
    return hashes


def prepare_torch_runtime(shim_path):
    """Same narrow Windows ROCm compatibility as the existing 4B loader."""
    import torch.distributed as distributed
    missing = not hasattr(distributed, "is_initialized")
    spec = importlib.util.spec_from_file_location("_frozen_qwen8b_torch_compat", shim_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_rocm_dist()
    return {"missing_is_initialized_patched_false": missing,
            "scope": "single-process distributed availability, no tensor/model override"}


def watch_decoder_lifecycle(model):
    """Observe actual decoder parameter objects without keeping them alive."""
    return {
        "module": weakref.ref(model),
        "parameters": {name: weakref.ref(value) for name, value in model.named_parameters()
                       if name != "model.embed_tokens.weight"},
        "buffers": {name: weakref.ref(value) for name, value in model.named_buffers()},
    }


def confirm_decoder_released(witness):
    """Fail on retained decoder objects; allocator bytes are separate evidence."""
    parameters = [name for name, reference in witness["parameters"].items() if reference() is not None]
    buffers = [name for name, reference in witness["buffers"].items() if reference() is not None]
    if witness["module"]() is not None or parameters or buffers:
        raise ValueError(f"Original decoder objects remain alive: parameters={parameters}, buffers={buffers}")
    return {"guard": "original_decoder_object_weakrefs/1", "module_released": True,
            "parameter_objects_checked": len(witness["parameters"]),
            "buffer_objects_checked": len(witness["buffers"]),
            "live_parameter_objects": 0, "live_buffer_objects": 0,
            "scope": "Original module/parameter/buffer lifetime; residual CUDA allocator bytes are recorded separately and are not classified as model weights or workspace."}


def run_gpu(report, out, snapshot, config, index, tokenizer, encoded, token_ids):
    import numpy as np
    import torch
    if (report["runtime"]["packages"] != GPU_PACKAGES
            or report["runtime"]["dependency_sources"]["transformers_qwen3"]["sha256"] != GPU_QWEN_SOURCE_SHA256):
        raise ValueError("GPU implementation differs from the audited pinned local runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("Real GPU is required; CPU fallback is forbidden")
    count = len(encoded)
    report["runtime"].update(gpu=torch.cuda.get_device_name(0), hip=torch.version.hip,
                             cuda=torch.version.cuda, device_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    report["runtime"]["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    report["runtime"]["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model, full_head = load_streamed_model(snapshot, config, index, token_ids, "cuda")
    lifecycle = watch_decoder_lifecycle(model)
    np.save(out / "reduced_head_rows.npy", full_head[token_ids].numpy(), allow_pickle=False)
    torch.cuda.synchronize()
    report["runtime"]["load_seconds"] = time.perf_counter() - load_start
    report["runtime"]["decoder_parameter_devices"] = sorted({str(p.device) for name, p in model.named_parameters()
                                                              if name != "model.embed_tokens.weight"})
    report["runtime"]["input_embedding_device"] = str(model.model.embed_tokens.weight.device)
    report["runtime"]["rotary_buffers"] = {n: {"device": str(v.device), "dtype": str(v.dtype)}
                                            for n, v in model.named_buffers()}
    hidden = np.lib.format.open_memmap(out / "hidden_states.npy", mode="w+", dtype="float16",
                                      shape=(count, config["hidden_size"]))
    reduced_logits = np.zeros((count, 2), dtype="float32")
    reduced_scores = np.zeros(count, dtype="float32")
    batches, retries = [], []
    captured = []

    def capture_head(_module, args):
        value = args[0]
        if (value.ndim != 3 or value.shape[1:] != (1, config["hidden_size"])
                or value.dtype != torch.float16 or value.device.type != "cuda"):
            raise ValueError("Original CausalLM did not project exactly its final hidden state")
        captured.append(value.detach().to(device="cpu", copy=True).squeeze(1).contiguous())

    hook = model.lm_head.register_forward_pre_hook(capture_head)

    def score_batch(start, size):
        captured.clear()
        padded = tokenizer.pad({"input_ids": encoded[start:start + size]}, padding=True, return_tensors="pt")
        with torch.inference_mode():
            embedded = model.model.embed_tokens(padded["input_ids"]).to(device="cuda")
            result = model(inputs_embeds=embedded, attention_mask=padded["attention_mask"].to("cuda"),
                           position_ids=None, past_key_values=None, use_cache=False, logits_to_keep=1)
            logits = result.logits[:, -1, :].float()
            scores = probability(logits)
        if len(captured) != 1 or captured[0].shape != (size, config["hidden_size"]):
            raise ValueError("Missing or ambiguous original final hidden states")
        if not torch.isfinite(captured[0]).all():
            raise ValueError("Non-finite original final hidden states")
        return logits.cpu().numpy(), scores.cpu().numpy(), captured.pop().numpy(), padded["input_ids"].shape[1]

    def on_oom(start, size, message):
        retries.append({"phase": "decoder", "start": start, "attempted_batch_size": size,
                        "retry_batch_size": 1, "error": message})
        captured.clear()
        gc.collect()
        torch.cuda.empty_cache()

    report["runtime"]["warmup"] = {"pairs": 1, "excluded_from_scoring_seconds": True}
    warm_start = time.perf_counter()
    score_batch(0, 1)
    torch.cuda.synchronize()
    report["runtime"]["warmup"]["seconds"] = time.perf_counter() - warm_start
    captured.clear()
    start_time = time.perf_counter()
    for start, size, result in retry_batches(count, 2, score_batch, torch.OutOfMemoryError, on_oom):
        logits, scores, states, width = result
        reduced_logits[start:start + size] = logits
        reduced_scores[start:start + size] = scores
        hidden[start:start + size] = states
        batches.append({"start": start, "count": size, "padded_tokens": width})
        if len(batches) % 25 == 0 or start + size == count:
            hidden.flush()
            report["progress"] = {"phase": "reduced_decoder", "completed_pairs": start + size,
                                  "planned_pairs": count}
            save(out / "report.json", report)
            print(f"reduced {start + size}/{count} pairs", flush=True)
    torch.cuda.synchronize()
    report["runtime"]["reduced_scoring_seconds"] = time.perf_counter() - start_time
    report["runtime"]["decoder_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["runtime"]["decoder_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    hook.remove()
    captured.clear()
    hidden.flush()
    np.save(out / "reduced_logits.npy", reduced_logits, allow_pickle=False)
    np.save(out / "reduced_scores.npy", reduced_scores, allow_pickle=False)
    save(out / "batches.json", batches)
    save(out / "oom_retries.json", retries)
    hidden_hash = digest(out / "hidden_states.npy")
    report["cpu_head_audit"] = cpu_head_audit(full_head, hidden, token_ids)
    # No stale model closures or head hooks may retain GPU decoder allocations.
    del score_batch, hook
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    report["runtime"]["post_decoder_allocated_bytes"] = torch.cuda.memory_allocated()
    report["runtime"]["decoder_lifecycle"] = confirm_decoder_released(lifecycle)
    full_gpu = full_head.to(device="cuda")
    full_logits = np.zeros((count, 2), dtype="float32")
    full_scores = np.zeros(count, dtype="float32")
    full_start = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            start, size = batch["start"], batch["count"]
            states_gpu = torch.from_numpy(hidden[start:start + size].copy()).unsqueeze(1).to("cuda")
            # Deliberately the entire original vocabulary and the SAME batch
            # shape. Selecting rows beforehand would make this audit circular.
            vocabulary_logits = torch.nn.functional.linear(states_gpu, full_gpu)
            selected = vocabulary_logits[:, -1, token_ids].float()
            full_logits[start:start + size] = selected.cpu().numpy()
            full_scores[start:start + size] = probability(selected).cpu().numpy()
    torch.cuda.synchronize()
    report["runtime"]["full_head_validation_seconds"] = time.perf_counter() - full_start
    np.save(out / "full_gpu_logits.npy", full_logits, allow_pickle=False)
    np.save(out / "full_gpu_scores.npy", full_scores, allow_pickle=False)
    if digest(out / "hidden_states.npy") != hidden_hash:
        raise ValueError("Observed hidden states changed between the two head projections")
    report["head_binding"] = {
        "hidden_states_sha256": hidden_hash, "pairs_sha256": digest(out / "pairs.json"),
        "batches_sha256": digest(out / "batches.json"), "observed_pairs": count,
        "full_gpu_projected_pairs": count, "same_batch_boundaries": True,
        "original_head_shape": list(full_head.shape), "reduced_head_shape": [2, config["hidden_size"]],
        "original_head_fp16_sha256": hashlib.sha256(full_head.numpy().tobytes()).hexdigest(),
        "reduced_head_fp16_sha256": hashlib.sha256(full_head[token_ids].numpy().tobytes()).hexdigest(),
        "bias": None, "dtype": "torch.float16", "full_vocabulary_projection": True,
    }
    report["numerical_audit"] = numerical_comparison(reduced_logits, full_logits, reduced_scores, full_scores)
    return {"reduced": reduced_scores, "full_gpu": full_scores}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "out/quality-recall-inputs/inputs.json")
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "out/quality-qwen8b-download.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true", help="CPU metadata/tokenizer validation; no GPU access")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        parser.error("Use a fresh immutable output directory")
    input_bytes = args.inputs.read_bytes()
    if hashlib.sha256(input_bytes).hexdigest() != INPUT_SHA256:
        raise ValueError("This preregistered experiment requires the exact frozen 162-case input bytes")
    inputs = json.loads(input_bytes)
    pairs, pair_index = frozen_pairs(inputs)
    bound_count = check_bound_inputs(inputs)
    model_manifest_bytes = args.model_manifest.read_bytes()
    assets, config, index = validate_model_assets(args.model_manifest, model_manifest_bytes)
    out.mkdir(parents=True)
    (out / "inputs.json").write_bytes(input_bytes)
    (out / "model-download.json").write_bytes(model_manifest_bytes)
    save(out / "pairs.json", pairs)
    source_hashes = snapshot_source(out)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    distributed_compat = prepare_torch_runtime(out / "source/risk_layer/torch_compat.py")
    structure = preflight_structure(config)
    from transformers import AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as qwen_module
    tokenizer = AutoTokenizer.from_pretrained(assets["snapshot"], local_files_only=True,
                                             padding_side="left", trust_remote_code=False)
    encoded, token_ids, tokenizer_info = tokenize_pairs(tokenizer, pairs)
    save(out / "tokens.json", encoded)
    dependency_sources = {}
    dependencies = {"transformers_qwen3": qwen_module}
    for name in ("transformers.models.qwen3.configuration_qwen3", "transformers.modeling_utils",
                 "transformers.masking_utils", "transformers.modeling_rope_utils",
                 "transformers.integrations.sdpa_attention", "torch.nn.modules.linear", "torch.nn.functional"):
        dependencies[name] = importlib.import_module(name)
    for module_name, module in dependencies.items():
        path = Path(inspect.getfile(module))
        target = out / "dependency-source" / (module_name + ".py")
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        dependency_sources[module_name] = {"original_path": str(path), "sha256": digest(target)}
    report = {
        "format": "quality_qwen8b_probe/1", "status": "running",
        "inputs_sha256": INPUT_SHA256, "source_hashes": source_hashes,
        "source_report_sha256": inputs["source_report_sha256"],
        "bound_input_files": bound_count, "model_manifest_sha256": hashlib.sha256(model_manifest_bytes).hexdigest(),
        "model_assets": assets["files"], "model": {"name": MODEL, "revision": REVISION},
        "instruction": INSTRUCTION, "max_length": 2048, "threshold": .5, "topk": 5,
        "variants": {name: {"head": head, "shortlist": size} for name, (head, size) in VARIANTS.items()},
        "case_counts": dict(Counter(c["dataset"] for c in inputs["cases"])),
        "unique_pairs": len(pairs), "tokenizer": tokenizer_info,
        "architecture_audit": structure,
        "runtime": {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
                                                       ("torch", "transformers", "numpy", "safetensors")},
                    "dependency_sources": dependency_sources, "dtype": "torch.float16",
                    "distributed_compatibility": distributed_compat,
                    "attention_implementation": "sdpa", "use_cache": False,
                    "input_embedding": "original FP16 CPU lookup, copied to GPU inputs_embeds",
                    "initial_batch_size": 2, "declared_oom_retry_batch_size": 1,
                    "truncation": False, "quantization": None},
        "notes": ["All 162 cases are previously examined development or historical cases; no blind test.",
                  "Historical labels are used only after all model scoring, not in model inputs or candidate selection.",
                  "Both complete head outputs are retained; numerical mismatches never remove cases.",
                  "FP16 is the declared inference dtype; the original checkpoint stores BF16 weights.",
                  "The full-head audit projects the same observed hidden states, not a second decoder inference.",
                  "Timing measures this standalone 8B experiment, not the production pipeline or three-model co-residence."],
        "rows": [],
    }
    save(out / "manifest.json", {k: v for k, v in report.items() if k != "rows"})
    try:
        if args.preflight_only:
            report["status"] = "preflight_pass"
            report["gpu_used"] = False
        else:
            scores = run_gpu(report, out, Path(assets["snapshot"]), config, index, tokenizer, encoded, token_ids)
            sys.path.insert(0, str(out / "source"))
            from tools._gpu_eval_metrics import summarize
            outputs = {name: build_outputs(inputs["cases"], pair_index, scores[head], size)
                       for name, (head, size) in VARIANTS.items()}
            rows = [{k: v for k, v in case.items() if k != "groups"} for case in inputs["cases"]]
            for name, values in outputs.items():
                for row, value in zip(rows, values, strict=True):
                    row[name] = {"suggestions": value["suggestions"]}
            report["rows"] = rows
            report["summaries"] = {dataset: {name: summarize([r for r in rows if r["dataset"] == dataset], name)
                                              for name in VARIANTS} for dataset in ("main", "additional")}
            differences = []
            for size in (20, 50):
                for row in rows:
                    a = row[f"qwen8b_reduced_top{size}"]["suggestions"]
                    b = row[f"qwen8b_full_gpu_top{size}"]["suggestions"]
                    if a != b:
                        def order(suggestions):
                            return [(s["start"], s["end"], [n["id"] for n in s["normen"]])
                                    for s in suggestions]
                        differences.append({"case_id": row["case_id"], "dataset": row["dataset"],
                                            "shortlist": size, "norm_order_equal": order(a) == order(b)})
            report["output_differences"] = differences
            report["status"] = "complete" if report["numerical_audit"]["exact_scores"] else "mismatch"
            report["gpu_used"] = True
        if digest(args.inputs) != INPUT_SHA256 or digest(args.model_manifest) != report["model_manifest_sha256"]:
            raise ValueError("Bound inputs/model manifest changed during the run")
        check_bound_inputs(inputs)
        validate_model_assets(args.model_manifest, model_manifest_bytes)
        for relative, expected in source_hashes.items():
            if digest(ROOT / relative) != expected:
                raise ValueError("Runner or evaluator source changed during the run")
        for metadata in dependency_sources.values():
            if digest(metadata["original_path"]) != metadata["sha256"]:
                raise ValueError("Installed model implementation changed during the run")
        report["artifact_hashes"] = {path.name: digest(path) for path in sorted(out.iterdir())
                                      if path.is_file() and path.name not in ("report.json", "manifest.json")}
    except Exception as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        save(out / "report.json", report)
    print(json.dumps({"status": report["status"], "unique_pairs": len(pairs),
                      "numerical_audit": report.get("numerical_audit")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] in ("complete", "preflight_pass") else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
