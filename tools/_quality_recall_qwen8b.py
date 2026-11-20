"""Independent CPU audit and aggregate projection for the fixed Qwen3 8B probe.

Candidate construction is replayed from original source and score caches in a
fresh process. Original head rows and stored projections are checked with NumPy;
no Torch, decoder, GPU or model service is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

from tools import quality_recall_report as audit

MODEL = "Qwen/Qwen3-Reranker-8B"
REVISION = "77d193c791ed757ca307ee72715aa132723da912"
TARGET_TOTALS = {"main": 176, "additional": 27}
INPUT_SHA256 = "145e1edfe2ab2cda1f54ed20889be2395ae1bf32d2c97c5f4d632e177edd9e30"
VARIANTS = {f"qwen8b_{head}_top{size}": {"head": head, "shortlist": size}
            for head in ("reduced", "full_gpu") for size in (20, 50)}
INSTRUCTION = ("Judge whether the legal provision (Document) matches the legal concept "
               "or factual situation described in the Query.")
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based '
          'on the Query and the Instruct provided. Note that the answer can only be '
          '"yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ORIGINAL_SHA256 = "02ad6f4e6c61ce4206a107b4969b2f324bceb134c82d755ff8b36d90b6a98706"
RECOVERY_ORIGINAL_MANIFEST_SHA256 = "79e3c7419d573f6a72266d5cc5d441508762638b67580ea3b4e1880c2c165ee2"
RECOVERY_ERROR = "ValueError('GPU decoder was not released before full original head validation')"
INHERITED_ARTIFACTS = {"inputs.json", "pairs.json", "tokens.json", "model-download.json", "batches.json", "oom_retries.json",
                       "hidden_states.npy", "reduced_head_rows.npy", "reduced_logits.npy", "reduced_scores.npy"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_inputs(inputs):
    audit.require(inputs.get("format") == "quality_recall_inputs/1" and inputs.get("status") == "complete"
                  and inputs.get("torch_imported") is False and inputs.get("profile") == "kaskade"
                  and inputs.get("package") == "ranked", "Expected original source-bound ranked BGE inputs")
    cases = audit.cases(inputs["cases"])
    for dataset, count in TARGET_TOTALS.items():
        actual = sum(len(set(c["targets"])) for c in cases.values() if c["dataset"] == dataset and not c["negative"])
        audit.require(actual == count, "Historical norm-target denominator differs")
    counts = {"cases": len(cases), "groups": 0, "top20_candidates": 0, "top50_candidates": 0,
              "known_old_qwen": 0}
    for case in inputs["cases"]:
        for group in case["groups"]:
            for start, end, text in ((group["start"], group["end"], group["query"]),
                                     (group["context_start"], group["context_end"], group["context_query"])):
                audit.require(type(start) is int and type(end) is int and 0 <= start < end <= len(case["query"])
                              and case["query"][start:end] == text, "Invalid source-bound query/focus offset")
            candidates = group["candidates"]
            audit.require(len(candidates) <= 50 and len({c["id"] for c in candidates}) == len(candidates),
                          "Invalid or duplicate BGE shortlist candidates")
            audit.require(all(_number(c["bge"]) and (c["sim"] is None or _number(c["sim"])) for c in candidates),
                          "Invalid original BGE/retrieval score")
            audit.require(candidates == sorted(candidates, key=lambda c: (-c["bge"], c["id"]))
                          and type(group["top20_count"]) is int and type(group["top50_count"]) is int
                          and group["top20_count"] == min(20, len(candidates))
                          and group["top50_count"] == len(candidates), "BGE shortlist order/count differs")
            for position, candidate in enumerate(candidates, 1):
                document = candidate["document"]
                audit.require(type(candidate["bge_rank"]) is int and candidate["bge_rank"] == position
                              and isinstance(document, str) and document.startswith(candidate["id"] + "\n")
                              and hashlib.sha256(document.encode()).hexdigest() == candidate["document_sha256"],
                              "Original candidate source/rank differs")
                counts["known_old_qwen"] += candidate["old_qwen"] is not None
            counts["groups"] += 1
            counts["top20_candidates"] += group["top20_count"]
            counts["top50_candidates"] += group["top50_count"]
    audit.require(inputs["counts"] == counts, "Frozen input accounting differs")
    return cases


def replay(cases, scores, shortlist):
    """Independently apply the fixed raw half threshold and final per-focus Top5."""
    audit.require(type(shortlist) is int and shortlist in (20, 50), "Undeclared 8B shortlist")
    outputs = {}
    for case in cases:
        suggestions = []
        for group in case["groups"]:
            eligible = []
            for candidate in group["candidates"][:shortlist]:
                key = group["query"], candidate["document"]
                audit.require(key in scores, "Missing exact Qwen8B input score")
                value = scores[key]
                audit.require(_number(value) and 0 <= value <= 1, "Invalid Qwen8B raw score")
                if value >= .5:
                    eligible.append({"id": candidate["id"], "sim": candidate["sim"], "rerank": value})
            eligible.sort(key=lambda n: (-n["rerank"], -(n["sim"] if n["sim"] is not None else -math.inf), n["id"]))
            if eligible:
                suggestions.append({"start": group["start"], "end": group["end"], "normen": eligible[:5]})
        audit.require(case["case_id"] not in outputs, "Duplicate Qwen8B output case")
        outputs[case["case_id"]] = suggestions
    return outputs


def audit_original_inputs(path):
    """Fresh-process replay of the original retrieval pool, packages and BGE cap."""
    path = Path(path).resolve()
    inputs = audit.read(path)
    input_cases = validate_inputs(inputs)
    hashes = {str(path): audit.digest(path), **inputs["source_hashes"], **inputs["input_hashes"]}
    for filename, expected in hashes.items():
        audit.check_hash(filename, expected)
    run = Path(inputs["source_run"])
    report = audit.read(run / "report.json")
    audit.require(report["status"] == "complete" and audit.digest(run / "report.json") == inputs["source_report_sha256"]
                  and report["models"] == inputs["models"], "Original R2 source/model identity differs")
    audit.same_cases(input_cases, audit.cases(report["rows"]))
    helper_paths = [Path(name) for name in inputs["source_hashes"] if Path(name).name == "analyze.py"]
    audit.require(len(helper_paths) == 1, "Expected one bound original CPU replay helper")
    spec = importlib.util.spec_from_file_location("qwen8b_original_replay", helper_paths[0])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus")
    sys.path.insert(0, str(run / "source"))
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.semantic_context import build_context_queries
    from risk_layer.semantic_search import PROFILE_CONFIG, passage_suggestions
    from risk_layer.triggerindex import TriggerIndex
    import risk_layer.semantic_search as runtime
    audit.require(Path(runtime.__file__).resolve().is_relative_to(run / "source"), "CPU input audit imported unfrozen runtime")
    PROFILE_CONFIG.update(report["variants"])
    audit.require(PROFILE_CONFIG["kaskade"].get("package", "ranked") == "ranked"
                  and PROFILE_CONFIG["kaskade"].get("shortlist", 20) == 20, "Original ranking configuration differs")
    trace = helper.Trace()
    encoder = helper.Encoder(run, report["models"]["encoder"], trace)

    class RecordingBGE(helper.Ranker):
        def score(self, pairs):
            values = super().score(pairs)
            if pairs:
                group = self.trace.group
                group["documents"] = {helper.norm_id(document): document for _, document in pairs}
                audit.require(len(group["documents"]) == len(pairs), "Duplicate source norm in BGE pool")
                group["query"] = pairs[0][0]
            return values

    ranker = helper.Ranker(run / "qwen-scores.jsonl", report["models"]["qwen"], trace, "qwen", {})
    ranker.vorauswahl = RecordingBGE(run / "bge-scores.jsonl", report["models"]["bge"], trace, "bge", {})
    paths = list(map(Path, report["input_hashes"]))
    original = EmbeddingIndex.load(next(p.parent for p in paths if p.name == "meta.json"))
    graph = NormGraph(next(p for p in paths if p.name == "graph.sqlite"))
    original_trigger = TriggerIndex(next(p for p in paths if p.name == "trigger.sqlite"))
    catalog = lade_katalog(run / "source/catalog/begriffe.yaml")
    packages = 0
    try:
        audit.require(original is not None and pruefe_aktualitaet(original, graph, catalog), "Original source generation differs")
        index, trigger = helper.Index(original, trace), helper.Trigger(original_trigger, trace)
        for row in report["rows"]:
            trace.groups, trace.variant = [], (row["dataset"], "kaskade")
            actual = passage_suggestions(row["query"], profile="kaskade", laws=(), index=index, encoder=encoder,
                                        reranker=ranker, trigger=trigger, graph=graph, chains={}, catalog=catalog)
            audit.require(actual == row["kaskade"]["suggestions"], "Original R2 output replay differs")
            contexts = build_context_queries(row["query"], mode="sentence")
            queries = list(dict.fromkeys(c.query_text for c in contexts))
            audit.require(len(queries) == len(trace.groups), "Original retrieval group count differs")
            groups = dict(zip(queries, trace.groups, strict=True))
            expected_groups = []
            for context in contexts:
                group = groups[context.query_text]
                query = group.get("query", context.focus_text)
                audit.require(query == context.focus_text, "Original BGE query is not the focus")
                ordered = sorted(group["bge"], key=lambda nid: (-group["bge"][nid], nid))[:50]
                audit.require(set(ordered[:20]) == set(group["qwen"]), "Original BGE Top20 differs")
                candidates = []
                for rank, nid in enumerate(ordered, 1):
                    document = group["documents"][nid]
                    key = helper.key_for(None, 1024, query, document)
                    candidates.append({"id": nid, "sim": group["retrieved"].get(nid), "bge": group["bge"][nid],
                        "bge_rank": rank, "document": document, "document_sha256": hashlib.sha256(document.encode()).hexdigest(),
                        "original_input_sha256": key, "old_qwen": ranker.scores.get(key),
                        "old_qwen_in_kaskade20": nid in group["qwen"]})
                packages += len(candidates)
                expected_groups.append({"start": context.focus_start, "end": context.focus_end, "query": query,
                    "context_start": context.context_start, "context_end": context.context_end,
                    "context_query": context.query_text, "top20_count": min(20, len(candidates)),
                    "top50_count": len(candidates), "candidates": candidates})
            audit.require(expected_groups == input_cases[row["case_id"]]["groups"], "Frozen BGE20/50 inputs differ from original retrieval")
    finally:
        graph.close()
        original_trigger.close()
    audit.require("torch" not in sys.modules, "Input replay unexpectedly imported Torch")
    for filename, expected in hashes.items():
        audit.check_hash(filename, expected)
    return {"status": "pass", "inputs_sha256": audit.digest(path), "cases": len(input_cases),
            "exact_source_packages": packages, "bound_files": len(hashes), "torch_imported": False, "mismatches": 0}


def checked_array(path, shape, dtype):
    import numpy as np
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    audit.require(value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all(),
                  "Incomplete, nonfinite or wrong-dtype model array")
    return value


def checked_numerics(logits, scores, stored):
    """Float64 softmax reference tolerates only FP32 arithmetic rounding."""
    import numpy as np
    for head in ("reduced", "full_gpu"):
        shifted = logits[head].astype(np.float64) - logits[head].max(axis=1, keepdims=True)
        probability = np.exp(shifted[:, 1]) / np.exp(shifted).sum(axis=1)
        audit.require(np.all((scores[head] >= 0) & (scores[head] <= 1))
                      and np.allclose(probability, scores[head], rtol=0, atol=2e-7),
                      "Stored raw scores differ from original binary logits")
        difference = logits[head][:, 1].astype(np.float64) - logits[head][:, 0]
        unambiguous = (difference == 0) | (np.abs(difference) > 2e-6)
        audit.require(np.array_equal((scores[head] >= .5)[unambiguous], (difference >= 0)[unambiguous]),
                      "Softmax threshold side differs from original binary logits")
    delta = np.abs(scores["reduced"].astype(np.float64) - scores["full_gpu"].astype(np.float64))
    actual = {"pairs": len(delta),
              "different_logit_pairs": int(np.any(logits["reduced"] != logits["full_gpu"], axis=1).sum()),
              "different_scores": int(np.count_nonzero(delta)),
              "max_abs_score_difference": float(delta.max(initial=0)),
              "max_abs_logit_difference": float(np.abs(logits["reduced"] - logits["full_gpu"]).max(initial=0)),
              "threshold_crossings": int(np.count_nonzero((scores["reduced"] >= .5) != (scores["full_gpu"] >= .5))),
              "exact_scores": bool(np.array_equal(scores["reduced"], scores["full_gpu"]))}
    audit.require(actual == stored, "Stored numerical comparison differs from actual complete arrays")
    return actual


def checked_outputs(report, inputs, pairs, scores, baseline):
    """Reconstruct all four arms before computing any historical-label metric."""
    expected_pairs = list(dict.fromkeys((group["query"], c["document"])
                          for case in inputs["cases"] for group in case["groups"] for c in group["candidates"]))
    audit.require(pairs == [{"query": q, "document": d} for q, d in expected_pairs],
                  "Stored pair order is not the original stable Top50 traversal")
    rows = audit.cases(report["rows"])
    audit.same_cases(validate_inputs(inputs), rows)
    audit.same_cases(audit.cases(baseline["rows"]), rows)
    audit.require(report["variants"] == VARIANTS, "All four fixed head/shortlist arms are required")
    for name, config in VARIANTS.items():
        values = scores[config["head"]]
        audit.require(len(values) == len(expected_pairs), "Incomplete model score coverage")
        output = replay(inputs["cases"], dict(zip(expected_pairs, map(float, values), strict=True)), config["shortlist"])
        audit.require(all(rows[key][name] == {"suggestions": value} for key, value in output.items()),
                      "8B output differs from exact raw-score replay")
    differences = []
    def order(values):
        return [(s["start"], s["end"], [n["id"] for n in s["normen"]]) for s in values]
    for size in (20, 50):
        for row in report["rows"]:
            left, right = (row[f"qwen8b_{head}_top{size}"]["suggestions"] for head in ("reduced", "full_gpu"))
            if left != right:
                differences.append({"case_id": row["case_id"], "dataset": row["dataset"],
                                    "shortlist": size, "norm_order_equal": order(left) == order(right)})
    audit.require(differences == report["output_differences"], "Head output difference accounting differs")
    audit.validate_summaries(report, VARIANTS)
    results = {}
    for name in VARIANTS:
        datasets, changes = {}, {}
        for dataset in audit.DATASETS:
            selected = [row for row in report["rows"] if row["dataset"] == dataset]
            metrics = audit.variant_metrics(selected, name)
            matched = sum(sum(any(n["id"] == t or n["id"].startswith(t + ":")
                                 for s in row[name]["suggestions"] for n in s["normen"])
                              for t in set(row["targets"])) for row in selected if not row["negative"])
            datasets[dataset] = {**metrics, "historical_norm_targets": TARGET_TOTALS[dataset],
                                 "matched_historical_norm_targets": matched}
            before = {r["case_id"]: r for r in baseline["rows"] if r["dataset"] == dataset}
            changes[dataset] = {k: len(v) for k, v in audit.paired_changes(
                before, {r["case_id"]: r for r in selected}, "kaskade", name).items()}
        results[name] = {"datasets": datasets, "paired_change_counts_vs_original_kaskade": changes}
    counts = {dataset: {str(size): {"different_outputs": sum(d["dataset"] == dataset and d["shortlist"] == size
                                                           for d in differences),
                                   "different_norm_orders": sum(d["dataset"] == dataset and d["shortlist"] == size
                                                                and not d["norm_order_equal"] for d in differences)}
                        for size in (20, 50)} for dataset in audit.DATASETS}
    return results, counts


def checked_token_inputs(snapshot, metadata, pairs, tokens):
    """Use the original tokenizer JSON directly, without importing Transformers."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids
    no, yes = [tokenizer.token_to_id(word) for word in ("no", "yes")]
    audit.require(type(no) is int and type(yes) is int and no != yes
                  and encode("no") == [no] and encode("yes") == [yes], "Original binary token identity differs")
    prefix, suffix = encode(PREFIX), encode(SUFFIX)
    audit.require(len(tokens) == len(pairs), "Incomplete exact token inputs")
    for pair, stored in zip(pairs, tokens, strict=True):
        actual = prefix + encode(f"<Instruct>: {INSTRUCTION}\n<Query>: {pair['query']}\n<Document>: {pair['document']}") + suffix
        audit.require(len(actual) <= 2048 and actual == stored, "Exact original model token input differs or was truncated")
    audit.require(metadata["prefix_ids"] == prefix and metadata["suffix_ids"] == suffix
                  and metadata["no_token_id"] == no and metadata["yes_token_id"] == yes
                  and metadata["padding_side"] == "left" and type(metadata["pad_token_id"]) is int
                  and metadata["max_observed_tokens"] == max(map(len, tokens)), "Stored tokenizer metadata differs")
    token_config = audit.read(snapshot / "tokenizer_config.json")
    pad = token_config["pad_token"]
    if isinstance(pad, dict):
        pad = pad["content"]
    audit.require(tokenizer.token_to_id(pad) == metadata["pad_token_id"], "Original padding token differs")
    return no, yes


def checked_checkpoint_files(weight_map, files):
    audit.require(len(weight_map) == 399 and set(weight_map.values()) <= set(files),
                  "Original decoder or head checkpoint shard is unbound")


def checked_model_and_tokens(directory, report, pairs, bind):
    """Original pinned file identities, exact tokenizer inputs and BF16 head rows."""
    import numpy as np
    model = audit.read(directory / "model-download.json")
    audit.require(model.get("format") == "quality_model_download/1" and model.get("status") == "complete"
                  and model.get("repo") == MODEL and model.get("revision") == REVISION
                  and model["files"] == report["model_assets"], "Original pinned model manifest differs")
    snapshot = Path(model["snapshot"])
    audit.require({"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
                  <= set(model["files"]), "Required original model/tokenizer inputs are unbound")
    bind(snapshot / "model.safetensors.index.json", "8a22e7ffd6001796256c0be7c73594835b6f907eec442fde691301ebf35087d9")
    weight_map = audit.read(snapshot / "model.safetensors.index.json")["weight_map"]
    checked_checkpoint_files(weight_map, model["files"])
    for name, data in model["files"].items():
        audit.require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid model asset path")
        path = snapshot / name
        audit.require(path.stat().st_size == data["bytes"], "Original model asset size differs")
        bind(path, data["sha256"])
    bind(snapshot / "config.json", "e87a4f50c475c7276fc642eabfc986a7c8da1c8a30a4ee34b4979048b330f2be")
    tokens = audit.read(directory / "tokens.json")
    no, yes = checked_token_inputs(snapshot, report["tokenizer"], pairs, tokens)
    original_rows = []
    with (snapshot / weight_map["lm_head.weight"]).open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        audit.require(header_size < 16_000_000, "Unexpected original weight header")
        head = json.loads(stream.read(header_size))["lm_head.weight"]
        audit.require(head["shape"] == [151669, 4096] and head["dtype"] == "BF16"
                      and head["data_offsets"][1] - head["data_offsets"][0] == 151669 * 4096 * 2,
                      "Original full head is not the pinned dense BF16 matrix")
        beginning = 8 + header_size + head["data_offsets"][0]
        def fp16(raw):
            return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4").astype("<f2")
        for row in (no, yes):
            stream.seek(beginning + row * 4096 * 2)
            original_rows.append(fp16(stream.read(4096 * 2)))
        stream.seek(beginning)
        remaining, head_hash = 151669 * 4096 * 2, hashlib.sha256()
        while remaining:
            raw = stream.read(min(remaining, 16 * 1024**2))
            audit.require(bool(raw), "Original full head is truncated")
            head_hash.update(fp16(raw).tobytes())
            remaining -= len(raw)
    source_rows = np.stack(original_rows)
    stored_rows = checked_array(directory / "reduced_head_rows.npy", (2, 4096), "float16")
    audit.require(np.array_equal(stored_rows, source_rows)
                  and hashlib.sha256(source_rows.tobytes()).hexdigest() == report["head_binding"]["reduced_head_fp16_sha256"]
                  and head_hash.hexdigest() == report["head_binding"]["original_head_fp16_sha256"],
                  "Projected head weights differ from original BF16-to-FP16 checkpoint")
    return tokens, source_rows


def checked_batches(tokens, batches, retries):
    count, position = len(tokens), 0
    audit.require(len(retries) <= 1, "Undeclared repeated OOM retry")
    retry_start = retries[0]["start"] if retries else count
    for retry in retries:
        audit.require(retry["phase"] == "decoder" and retry["attempted_batch_size"] == 2 and retry["retry_batch_size"] == 1
                      and type(retry_start) is int and 0 <= retry_start < count - 1
                      and retry_start in {b["start"] for b in batches}, "Undeclared OOM fallback")
    for batch in batches:
        expected_size = 1 if position >= retry_start else min(2, count - position)
        audit.require(type(batch["count"]) is int and batch["count"] == expected_size
                      and type(batch["start"]) is int and batch["start"] == position
                      and type(batch["padded_tokens"]) is int
                      and batch["padded_tokens"] == max(map(len, tokens[position:position + expected_size]), default=-1),
                      "Observed GPU batch boundaries or padding differ")
        position += expected_size
    audit.require(position == count, "Incomplete observed GPU batch coverage")


def checked_projections(hidden, source_rows, logits, cpu):
    """Check every observed state using its source rows, with explicit FP tolerance."""
    import numpy as np
    observed = list(range(min(2, len(hidden))))
    reference = hidden[observed].astype(np.float32) @ source_rows.astype(np.float32).T
    arrays = {key: np.asarray(cpu[key], dtype=np.float64) for key in
              ("fp32_row_reference_logits", "original_fp16_logits", "reduced_fp16_logits")}
    audit.require(cpu["device"] == "cpu" and cpu["observed_pair_indices"] == observed
                  and cpu["original_head_dtype"] == cpu["reduced_head_dtype"] == "torch.float16"
                  and all(value.shape == reference.shape and np.isfinite(value).all() for value in arrays.values())
                  and np.allclose(arrays["fp32_row_reference_logits"], reference, rtol=1e-5, atol=1e-4),
                  "Original CPU row projection reference differs")
    delta = float(np.abs(arrays["original_fp16_logits"] - arrays["reduced_fp16_logits"]).max())
    audit.require(_number(cpu["max_abs_fp16_difference"]) and cpu["max_abs_fp16_difference"] == delta,
                  "CPU head difference accounting differs")
    max_reference_delta = dict.fromkeys(logits, 0.)
    for start in range(0, len(hidden), 512):
        states = hidden[start:start + 512].astype(np.float32)
        reference = states @ source_rows.astype(np.float32).T
        absolute_sum = np.abs(states) @ np.abs(source_rows.astype(np.float32)).T
        bound = .002 * absolute_sum + .002
        for head, value in logits.items():
            difference = np.abs(value[start:start + 512].astype(np.float64) - reference)
            audit.require(np.all(difference <= bound), "GPU logits do not match the stored states and original head rows")
            max_reference_delta[head] = max(max_reference_delta[head], float(difference.max()))
        if start == 0:
            for key in ("original_fp16_logits", "reduced_fp16_logits"):
                audit.require(np.all(np.abs(arrays[key] - reference[:len(observed)]) <= bound[:len(observed)]),
                              "CPU FP16 projection differs from actual observed states")
    return max_reference_delta


def canonical_source_hashes(hashes):
    """Compare relative source inventories across Windows/POSIX producers."""
    result, seen = {}, set()
    for name, expected in hashes.items():
        audit.require(isinstance(name, str), "Invalid relative source path")
        relative = name.replace("\\", "/")
        audit.require(all(part not in ("", ".", "..") and ":" not in part
                          for part in relative.split("/")), "Invalid relative source path")
        audit.require(relative.casefold() not in seen, "Source paths collide after normalization")
        seen.add(relative.casefold())
        result[relative] = expected
    return result


def checked_recovery_origin(directory, report, bind):
    """Only the exact fully scored, stopped decoder archive may be recovered."""
    recovery = report["recovery"]
    audit.require(recovery.get("format") == "quality_qwen8b_recovery/1"
                  and recovery.get("original_report_sha256") == RECOVERY_ORIGINAL_SHA256
                  and recovery.get("original_manifest_sha256") == RECOVERY_ORIGINAL_MANIFEST_SHA256
                  and recovery.get("original_status") == "failed" and recovery.get("original_error") == RECOVERY_ERROR
                  and recovery.get("original_post_decoder_allocated_bytes") == 79_691_776
                  and recovery.get("fresh_process") is True and recovery.get("full_head_only") is True
                  and recovery.get("decoder_rerun") is False, "Recovery does not bind the declared stopped decoder run")
    original_directory = Path(recovery["source_directory"]).resolve()
    audit.require(original_directory != directory.resolve(), "Recovery must preserve the original archive")
    for name, expected in (("report.json", RECOVERY_ORIGINAL_SHA256),
                           ("manifest.json", recovery["original_manifest_sha256"])):
        bind(original_directory / name, expected)
        bind(directory / ("original-" + name), expected)
    original = audit.read(directory / "original-report.json")
    audit.require(original.get("status") == "failed" and original.get("error") == RECOVERY_ERROR and original.get("rows") == []
                  and original["progress"] == {"phase": "reduced_decoder", "completed_pairs": 14961, "planned_pairs": 14961},
                  "Original recovery source did not finish every declared decoder pair")
    fixed_fields = ("format", "inputs_sha256", "source_hashes", "source_report_sha256", "bound_input_files",
                    "model_manifest_sha256", "model_assets", "model", "instruction", "max_length", "threshold", "topk",
                    "variants", "case_counts", "unique_pairs", "tokenizer", "architecture_audit", "cpu_head_audit")
    audit.require(all(report[key] == original[key] for key in fixed_fields), "Recovery changed original model, inputs or decoder audit")
    audit.require(set(report["runtime"]) == set(original["runtime"]) | {"full_head_validation_seconds"}
                  and all(report["runtime"][key] == value for key, value in original["runtime"].items())
                  and report["runtime"]["post_decoder_allocated_bytes"] == 79_691_776,
                  "Recovery rewrote the original decoder runtime or failed release measurement")
    inherited = recovery["inherited_artifact_hashes"]
    audit.require(set(inherited) == INHERITED_ARTIFACTS, "Recovery must bind every inherited model artifact")
    for name, expected in inherited.items():
        audit.require(report["artifact_hashes"].get(name) == expected, "Recovery changed an inherited model artifact binding")
        bind(original_directory / name, expected)
        bind(directory / name, expected)
    inherited_sources = {"source/" + name: value for name, value in canonical_source_hashes(original["source_hashes"]).items()}
    inherited_sources.update({"dependency-source/" + name + ".py": data["sha256"]
                              for name, data in original["runtime"]["dependency_sources"].items()})
    audit.require(canonical_source_hashes(recovery["inherited_source_hashes"]) == inherited_sources,
                  "Recovery original source inventory differs")
    for relative, expected in inherited_sources.items():
        bind(original_directory / relative, expected)
        bind(directory / relative, expected)
    audit.require(bool(recovery["source_hashes"]), "Recovery implementation is not hash-bound")
    for relative, expected in canonical_source_hashes(recovery["source_hashes"]).items():
        bind(directory / "recovery-source" / relative, expected)
    audit.require({"original-report.json", "original-manifest.json", "recovery-manifest.json"}
                  <= set(report["artifact_hashes"]), "Recovery manifest or original failed reports are unbound")
    manifest = audit.read(directory / "recovery-manifest.json")
    audit.require(manifest == {**recovery, "runtime": {}}, "Recovery changed its declared origin, sources or CPU preflight")
    preflight = recovery["preflight"]
    audit.require(all(preflight.get(key) == value for key, value in {
        "pairs": 14961, "batches": len(audit.read(directory / "batches.json")),
        "oom_retries": len(audit.read(directory / "oom_retries.json")), "reduced_raw_scores_preserved": True,
        "original_bound_input_files": original["bound_input_files"], "original_model_assets": len(original["model_assets"]),
        "exact_tokenized_pairs": 14961, "exact_original_head_rows": True, "torch_imported": False}.items()),
        "Recovery CPU preflight did not cover every inherited original artifact")
    runtime = recovery["runtime"]
    audit.require(runtime["packages"] == original["runtime"]["packages"]
                  and all(runtime[key] == original["runtime"][key] for key in (
                      "gpu", "hip", "cuda", "device_total_bytes", "matmul_allow_tf32", "deterministic_algorithms"))
                  and runtime["dtype"] == "torch.float16" and type(runtime["pid"]) is int and runtime["pid"] > 0
                  and runtime["initial_allocated_bytes"] == 0 and runtime["full_gpu_projected_pairs"] == 14961
                  and runtime["full_head_validation_seconds"] == report["runtime"]["full_head_validation_seconds"],
                  "Recovery full-head GPU/runtime identity differs from the decoder run")
    for key in ("head_load_seconds", "full_head_validation_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
        audit.require(_number(runtime[key]) and runtime[key] >= 0, "Invalid recovery GPU runtime measurement")
    return {"format": recovery["format"], "original_report_sha256": RECOVERY_ORIGINAL_SHA256,
            "original_manifest_sha256": recovery["original_manifest_sha256"], "original_status": "failed",
            "original_error": RECOVERY_ERROR, "original_post_decoder_allocated_bytes": 79_691_776,
            "original_release_guard_bytes": 64 * 1024**2, "fresh_process": True, "full_head_only": True,
            "decoder_rerun": False, "complete_inherited_decoder_pairs": 14961,
            "inherited_artifact_hashes": inherited, "source_hashes": recovery["source_hashes"],
            "runtime": {key: runtime[key] for key in ("packages", "gpu", "hip", "cuda", "device_total_bytes", "dtype",
                "initial_allocated_bytes", "head_load_seconds", "full_head_validation_seconds", "peak_allocated_bytes",
                "peak_reserved_bytes", "full_gpu_projected_pairs")}}


def checked_decoder_release(runtime):
    """Old archives use their old guard; new runs prove tensor object release."""
    lifecycle = runtime.get("decoder_lifecycle")
    if lifecycle is None:
        audit.require(runtime["post_decoder_allocated_bytes"] <= 64 * 1024**2,
                      "Legacy decoder release allocation guard failed")
        return {"guard": "legacy_post_decoder_allocated_bytes", "limit_bytes": 64 * 1024**2}
    expected = {"guard": "original_decoder_object_weakrefs/1", "module_released": True,
                "parameter_objects_checked": 398, "buffer_objects_checked": 2,
                "live_parameter_objects": 0, "live_buffer_objects": 0}
    audit.require(all(lifecycle.get(key) == value for key, value in expected.items())
                  and lifecycle["module_released"] is True
                  and all(type(lifecycle[key]) is int for key in (
                      "parameter_objects_checked", "buffer_objects_checked", "live_parameter_objects", "live_buffer_objects")),
                  "Original decoder objects are not completely released")
    return expected


def aggregate(path, baseline_path, audit_dir):
    """Validate the complete standalone experiment; return an explicit whitelist."""
    path, baseline_path, audit_dir = Path(path).resolve(), Path(baseline_path).resolve(), Path(audit_dir).resolve()
    directory, hashes = path.parent, {}
    def bind(filename, expected=None):
        filename = Path(filename).resolve()
        if str(filename) in hashes:
            actual = hashes[str(filename)]
        else:
            actual = audit.digest(filename)
            hashes[str(filename)] = actual
        audit.require(expected is None or expected == actual, "8B bound file hash differs")
        return actual
    report_hash, baseline_hash = bind(path), bind(baseline_path)
    report, baseline = audit.read(path), audit.read(baseline_path)
    audit.require(report.get("format") == "quality_qwen8b_probe/1" and report.get("status") in ("complete", "mismatch")
                  and report.get("gpu_used") is True and not report.get("error") and not report.get("diagnostic_subset"),
                  "Requires all completed original GPU decoder and full-head outputs")
    audit.require(report["model"] == {"name": MODEL, "revision": REVISION}
                  and report["instruction"] == INSTRUCTION and report["max_length"] == 2048
                  and report["threshold"] == .5 and report["topk"] == 5 and report["variants"] == VARIANTS
                  and report["case_counts"] == {"main": 132, "additional": 30}, "Fixed model or experiment configuration differs")
    audit.require(report["inputs_sha256"] == INPUT_SHA256 and report["source_report_sha256"] == baseline_hash,
                  "8B experiment is not bound to the original inputs and supplied baseline")
    artifacts = report["artifact_hashes"]
    required = {"inputs.json", "pairs.json", "tokens.json", "model-download.json", "batches.json", "oom_retries.json",
                "hidden_states.npy", "reduced_head_rows.npy", "reduced_logits.npy", "full_gpu_logits.npy",
                "reduced_scores.npy", "full_gpu_scores.npy"}
    audit.require(required <= set(artifacts), "Incomplete 8B model artifacts")
    for name, expected in artifacts.items():
        audit.require(Path(name).name == name, "Invalid artifact name")
        bind(directory / name, expected)
    bind(directory / "inputs.json", INPUT_SHA256)
    bind(directory / "model-download.json", report["model_manifest_sha256"])
    inputs, pairs = audit.read(directory / "inputs.json"), audit.read(directory / "pairs.json")
    validate_inputs(inputs)
    audit.require(inputs["source_report_sha256"] == baseline_hash and report["bound_input_files"] ==
                  len(inputs["source_hashes"]) + len(inputs["input_hashes"]) + 1, "Original input baseline/count differs")
    required_sources = {"tools/quality_qwen8b_probe.py", "tools/_gpu_eval_metrics.py", "risk_layer/torch_compat.py"}
    sources = canonical_source_hashes(report["source_hashes"])
    audit.require(set(sources) == required_sources, "Runner/evaluator/compatibility source is incomplete")
    for name, expected in sources.items():
        bind(directory / "source" / name, expected)
    recovery = checked_recovery_origin(directory, report, bind) if "recovery" in report else None
    runtime = report["runtime"]
    release = checked_decoder_release(runtime) if recovery is None else None
    architecture = report["architecture_audit"]
    audit.require(all(architecture.get(k) == v for k, v in {
        "original_parameter_tensors": 399, "all_parameters_meta": True, "original_fp16_bytes": 16_377_096_192,
        "decoder_fp16_bytes": 13_892_151_296, "cpu_embedding_fp16_bytes": 1_242_472_448,
        "original_head_fp16_bytes": 1_242_472_448, "reduced_head_fp16_bytes": 16_384}.items()),
        "Original parameter architecture or declared precision differs")
    audit.require(runtime["packages"] == {"torch": "2.9.1+rocm7.2.1", "transformers": "5.15.1",
                                           "numpy": "2.2.6", "safetensors": "0.8.0"}
                  and runtime["dtype"] == "torch.float16" and runtime["attention_implementation"] == "sdpa"
                  and runtime["use_cache"] is False and runtime["truncation"] is False and runtime["quantization"] is None
                  and runtime["input_embedding_device"] == "cpu" and runtime["decoder_parameter_devices"] == ["cuda:0"]
                  and runtime["initial_batch_size"] == 2 and runtime["declared_oom_retry_batch_size"] == 1,
                  "Original standalone GPU runtime differs")
    expected_dependency = "cbb7f2dc274c2f5592746c0dc6985ca50353efa07376f92cc922b77680a74f69"
    audit.require(set(runtime["dependency_sources"]) == {
        "transformers_qwen3", "transformers.models.qwen3.configuration_qwen3", "transformers.modeling_utils",
        "transformers.masking_utils", "transformers.modeling_rope_utils", "transformers.integrations.sdpa_attention",
        "torch.nn.modules.linear", "torch.nn.functional"}, "Original model dependency source inventory is incomplete")
    audit.require(runtime["dependency_sources"]["transformers_qwen3"]["sha256"] == expected_dependency,
                  "Pinned original Transformers implementation differs")
    for name, metadata in runtime["dependency_sources"].items():
        audit.require(Path(name).name == name, "Invalid dependency source name")
        bind(directory / "dependency-source" / (name + ".py"), metadata["sha256"])
    audit.require(runtime["rotary_buffers"] and all(v["dtype"] == "torch.float32" and v["device"] == "cuda:0"
                                                   for v in runtime["rotary_buffers"].values()), "Original rotary buffers differ")
    for key in ("device_total_bytes", "load_seconds", "reduced_scoring_seconds", "full_head_validation_seconds",
                "decoder_peak_allocated_bytes", "decoder_peak_reserved_bytes", "post_decoder_allocated_bytes"):
        audit.require(_number(runtime[key]) and runtime[key] >= 0, "Invalid standalone runtime measurement")
    warmup = runtime["warmup"]
    audit.require(warmup["pairs"] == 1 and warmup["excluded_from_scoring_seconds"] is True
                  and _number(warmup["seconds"]) and warmup["seconds"] >= 0, "Warmup timing scope differs")
    count = len(pairs)
    audit.require(type(report["unique_pairs"]) is int and report["unique_pairs"] == count and count > 0,
                  "Incomplete unique model input denominator")
    logits = {name: checked_array(directory / f"{name}_logits.npy", (count, 2), "float32") for name in ("reduced", "full_gpu")}
    scores = {name: checked_array(directory / f"{name}_scores.npy", (count,), "float32") for name in logits}
    hidden = checked_array(directory / "hidden_states.npy", (count, 4096), "float16")
    numerical = checked_numerics(logits, scores, report["numerical_audit"])
    audit.require(report["status"] == ("complete" if numerical["exact_scores"] else "mismatch"), "Numerical mismatch status differs")
    binding = report["head_binding"]
    audit.require(all(binding.get(k) == v for k, v in {
        "hidden_states_sha256": artifacts["hidden_states.npy"], "pairs_sha256": artifacts["pairs.json"],
        "batches_sha256": artifacts["batches.json"], "observed_pairs": count, "full_gpu_projected_pairs": count,
        "same_batch_boundaries": True, "original_head_shape": [151669, 4096], "reduced_head_shape": [2, 4096],
        "bias": None, "dtype": "torch.float16", "full_vocabulary_projection": True}.items()),
        "Full and reduced head input/weight bindings differ")
    tokens, source_rows = checked_model_and_tokens(directory, report, pairs, bind)
    batches, retries = audit.read(directory / "batches.json"), audit.read(directory / "oom_retries.json")
    checked_batches(tokens, batches, retries)
    max_reference_delta = checked_projections(hidden, source_rows, logits, report["cpu_head_audit"])
    variants, differences = checked_outputs(report, inputs, pairs, scores, baseline)
    audit.require(not audit_dir.exists(), "Choose a fresh independent input audit directory")
    audit_dir.mkdir(parents=True)
    original_audit_path = audit_dir / "original-inputs.json"
    completed = subprocess.run([sys.executable, "-m", "tools._quality_recall_qwen8b", "--audit-inputs",
                                str(directory / "inputs.json"), "--out", str(original_audit_path)],
                               cwd=ROOT, capture_output=True, check=False)
    (audit_dir / "stdout.txt").write_bytes(completed.stdout)
    (audit_dir / "stderr.txt").write_bytes(completed.stderr)
    audit.require(completed.returncode == 0, "Original input reconstruction failed; inspect private audit logs")
    original_audit = audit.read(original_audit_path)
    audit.require(original_audit["status"] == "pass" and original_audit["inputs_sha256"] == INPUT_SHA256
                  and original_audit["cases"] == 162 and original_audit["mismatches"] == 0
                  and original_audit["torch_imported"] is False, "Original input audit is incomplete")
    # Small mutable reports and produced arrays are rechecked after the audit;
    # immutable original model shards were already streamed and hash-verified.
    for filename, expected in hashes.items():
        if (Path(filename).is_relative_to(directory) or Path(filename) == baseline_path
                or recovery is not None and Path(filename).is_relative_to(Path(report["recovery"]["source_directory"]).resolve())):
            audit.check_hash(filename, expected)
    public = {"format": "quality_qwen8b_aggregate/1", "status": ("recovered_" if recovery is not None else "") + report["status"],
              "decision": "research_only", "report_sha256": report_hash, "baseline_sha256": baseline_hash,
              "inputs_sha256": INPUT_SHA256, "model_manifest_sha256": report["model_manifest_sha256"],
              "source_hashes": report["source_hashes"], "model": {"name": MODEL, "revision": REVISION},
              "configuration": {"instruction": INSTRUCTION, "max_length": 2048, "threshold": .5, "topk": 5,
                                "variants": VARIANTS, "truncation": False, "quantization": None, "dtype": "float16"},
              "variants": variants, "numerical_audit": numerical, "output_difference_counts": differences,
              "workload": {"cases": 162, "unique_pairs": count, "batches": len(batches), "oom_retries": len(retries),
                           "input_groups": inputs["counts"]["groups"], "top20_candidate_entries": inputs["counts"]["top20_candidates"],
                           "top50_candidate_entries": inputs["counts"]["top50_candidates"]},
              "runtime": {key: runtime[key] for key in ("packages", "dtype", "attention_implementation", "gpu", "hip", "cuda",
                  "device_total_bytes", "load_seconds", "reduced_scoring_seconds", "full_head_validation_seconds",
                  "decoder_peak_allocated_bytes", "decoder_peak_reserved_bytes", "post_decoder_allocated_bytes")},
              "audit": {"exact_case_outputs": 162 * 4, "bound_files": len(hashes), "original_inputs": original_audit,
                        "original_input_audit_sha256": audit.digest(original_audit_path), "mismatches": 0,
                        "max_abs_gpu_vs_fp32_row_reference": max_reference_delta,
                        "fp16_projection_tolerance": "0.002 * sum(abs(hidden * original_row)) + 0.002 per logit",
                        "head_comparison_uses_same_hidden_states": True, "full_original_head_verified": True},
              "notes": ["All four fixed arms and all 162 historical cases are included, including worse results.",
                        "Historical norm-target denominators are 176 and 27; unlisted positive-case norms are unadjudicated.",
                        "The head comparison reuses identical observed hidden states and batch boundaries, not a second decoder inference.",
                        "FP16 GEMM shape can affect logits and raw scores; both heads are reported without a bit-equality promise.",
                        "Timing covers this standalone 8B experiment, not production pipeline or three-model co-residence.",
                        "Previously examined development data; no independent validation or production activation."]}
    if recovery is not None:
        public["recovery"] = recovery
    else:
        public["audit"]["decoder_release"] = release
    audit.public_payload_check(public)
    audit.save(audit_dir / "binding.json", {"report_sha256": report_hash, "bound_hashes": hashes})
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    audit.require(not args.out.exists(), "Choose a fresh CPU audit output")
    result = audit_original_inputs(args.audit_inputs)
    audit.save(args.out, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
