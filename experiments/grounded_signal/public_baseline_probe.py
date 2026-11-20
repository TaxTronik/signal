"""Bound GPU execution of unchanged public treffer on all sixteen new cases.

Preparation and imports are model-free. Only the explicit run command loads
models; the operator must first finish and unload the generative model runs.
"""
# ruff: noqa: E402 -- private script bootstrap
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_layer.norm_check import _strict_json

PREPARE_FORMAT = "grounded_signal_public_baseline_inputs/1"
RUN_FORMAT = "grounded_signal_public_baseline/1"
PROFILE = "treffer"
PINS = {
    "encoder": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", "bge-m3", "torch.float32"),
    "bge": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "bge-reranker-v2-m3", "torch.float16"),
    "qwen": ("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5", "bge-qwen3-reranker-4b", "torch.float16"),
}
PACKAGES = {"torch": "2.9.1+rocm7.2.1", "transformers": "5.15.1",
            "sentence-transformers": "5.6.0", "numpy": "2.2.6"}
NOTES = [
    "All sixteen known assistant-reviewed synthetic diagnostics; no independent legal gold.",
    "Unchanged public treffer Pipeline.run with real BGE encoder, BGE and Qwen4B rerankers; no cached scores or model monkeypatch.",
    "Semantic suggestions and hard markings remain separate; missing target labels do not turn other norms into negatives.",
    "No structured fact/condition claims are fabricated for this baseline; source identity does not validate legal relevance.",
    "CPU/GPU dtypes differ; this run uses the earlier measured GPU route. No generation server is called.",
]


class ProbeError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProbeError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return _strict_json(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def inventory(directory, exclude=()):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.relative_to(directory).as_posix() not in exclude}


def check_inventory(directory, expected, exclude=()):
    require(isinstance(expected, dict) and all(isinstance(k, str) and not Path(k).is_absolute()
            and ".." not in Path(k).parts and (directory / k).resolve().is_relative_to(directory.resolve())
            for k in expected), "Invalid inventory paths")
    require(inventory(directory, exclude) == expected, "Artifact inventory changed")


def snapshot(source, destination, bindings):
    source, destination = Path(source), Path(destination)
    digest = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    require(sha(destination) == digest == sha(source), "Source changed during snapshot")
    bindings[str(source.resolve())] = digest


def check_files(bindings):
    require(all(sha(name) == digest for name, digest in bindings.items()), "Bound original file changed")


MODEL_INVENTORY_FORMAT = "signal/hf-snapshot-inventory/1"


def _absolute(path):
    # Preserve links for inspection: resolve() here would conceal their presence.
    return Path(os.path.abspath(path))


def _plain_directory(path):
    for parent in (_absolute(path), *_absolute(path).parents):
        require(parent.is_dir() and not parent.is_symlink()
                and not getattr(parent, "is_junction", lambda: False)(),
                "Model inventory contains a directory link or missing directory")


def _model_relative(name):
    require(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name
            and not Path(name).is_absolute() and all(part not in ("", ".", "..") for part in name.split("/")),
            "Invalid model inventory relative path")


def model_file_targets(directory, repo, revision):
    """Bind ordinary snapshot files or direct links into that HF repo's blobs.

    Inspect every parent before resolution, forbid directory links/junctions and
    blob link chains, and preserve the exact source-link spelling and target.
    This exception never applies to general artifact or source inventories.
    """
    require(isinstance(repo, str) and re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo), "Invalid HF repository")
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), "Invalid pinned HF revision")
    directory = _absolute(directory)
    repo_dir = directory.parent.parent
    require(directory.name == revision and directory.parent.name == "snapshots"
            and repo_dir.name == "models--" + repo.replace("/", "--"), "Snapshot does not match its pinned HF repository")
    _plain_directory(directory)
    blobs = repo_dir / "blobs"
    result = {}
    def walk_error(error):
        raise ProbeError("Cannot enumerate the complete model inventory") from error
    for current, directories, filenames in os.walk(directory, followlinks=False, onerror=walk_error):
        for child in directories:
            _plain_directory(Path(current) / child)
        for filename in filenames:
            path = Path(current) / filename
            relative = path.relative_to(directory).as_posix()
            _model_relative(relative)
            require(not getattr(path, "is_junction", lambda: False)(), "Model file is a junction")
            link = os.readlink(path) if path.is_symlink() else None
            if link is not None:
                _plain_directory(blobs)
                target = _absolute(path.parent / link)
                require(target.parent == blobs and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", target.name),
                        "Model symlink is not a direct blob in its own HF repository")
                require(target.is_file() and not target.is_symlink()
                        and not getattr(target, "is_junction", lambda: False)(), "Missing blob or chained blob link")
                kind = "hf_blob_symlink"
            else:
                target, kind = path, "regular"
                require(path.is_file(), "Model entry is not a regular file")
            require(path.resolve(strict=True) == target and target.resolve(strict=True) == target,
                    "Model path resolution differs from inspected target")
            result[relative] = {"kind": kind, "link_target": link, "resolved_target": str(target),
                                "bytes": target.stat().st_size}
    return result


def model_inventory(directory, repo, revision):
    before = model_file_targets(directory, repo, revision)
    files = {name: sha(Path(directory) / name) for name in sorted(before)}
    require(canonical(before) == canonical(model_file_targets(directory, repo, revision)),
            "Model file/link target changed while hashing")
    return files, before


def check_model_inventory(model):
    require(model.get("inventory_format") == MODEL_INVENTORY_FORMAT, "Missing bound HF model inventory policy")
    files, targets = model.get("files"), model.get("file_targets")
    require(isinstance(files, dict) and isinstance(targets, dict) and set(files) == set(targets),
            "Model file/target inventory differs")
    for name in files:
        _model_relative(name)
    before = model_file_targets(model["snapshot"], model["repo"], model["revision"])
    require(canonical(before) == canonical(targets), "Model source link or resolved target changed")
    actual, after = model_inventory(model["snapshot"], model["repo"], model["revision"])
    require(canonical(actual) == canonical(files) and canonical(after) == canonical(targets),
            "Model file bytes or link inventory changed")


def validate_cases(cases, primary):
    require(isinstance(cases, list) and len(cases) == 16, "Exactly sixteen planned cases required")
    require([c["case_id"] for c in cases] == primary["case_ids"] and len({c["case_id"] for c in cases}) == 16,
            "Case plan differs")
    require(primary.get("model_interface", {}).get("format") == "fixed_norm_slots/2", "Requires the R2 interface plan")
    families: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        require(isinstance(case["text"], str) and case["text"].strip(), "Missing original text")
        require(case["split"] in ("development", "evaluation"), "Unexpected split")
        families.setdefault(case["family_id"], []).append(case)
        labels = case["diagnostic_labels"]
        require(type(labels["hint_expected"]) is bool and labels["annotation_status"] == "assistant_reviewed_synthetic",
                "Unknown diagnostic annotation")
        groups = [labels.get(k, []) for k in ("target_norm_ids", "allowed_hint_norm_ids", "no_hint_norm_ids", "insufficient_context_norm_ids")]
        require(all(isinstance(g, list) and len(g) == len(set(g)) and all(isinstance(n, str) for n in g) for g in groups)
                and len(set(n for g in groups for n in g)) == sum(map(len, groups)), "Conflicting norm annotations")
        require(bool(labels["target_norm_ids"]) == labels["hint_expected"], "Hint/target denominator differs")
    require(len(families) == 8 and all(len(v) == 2 and len({c["split"] for c in v}) == 1 for v in families.values()),
            "Family split differs")
    require(Counter(c["split"] for c in cases) == {"development": 8, "evaluation": 8}, "Split counts differ")


def validate_gpu_reference(reference, code, profile_config):
    require(reference.get("format") == "quality_recall_live/1" and reference.get("status") in ("pass", "mismatch")
            and len(reference.get("rows", [])) == 162 and not reference.get("error"), "Incomplete earlier real GPU run")
    require(reference.get("public_profile") == PROFILE and canonical(reference.get("profile_config")) == canonical(profile_config),
            "Earlier measured public profile differs")
    for name in ("risk_layer/reranker.py", "risk_layer/semantic_search.py", "risk_layer/embedding.py"):
        require(reference["source_hashes"].get(name) == code.get(name), "VRAM-relevant source differs from measured GPU route")
    for label, (_repo, revision, name, dtype) in PINS.items():
        model = reference["models"][label]
        require(model["revision"] == revision and model["name"] == name and model["devices"] == ["cuda:0"]
                and model["dtypes"] == [dtype], "Earlier model identity/dtype differs")
    rt = reference["runtime"]
    require(rt["torch"] == PACKAGES["torch"] and rt["device"] == "AMD Radeon RX 9070 XT"
            and rt["packages"] == {k: v for k, v in PACKAGES.items() if k != "torch"}, "Earlier GPU runtime differs")
    require(0 < rt["peak_allocated_bytes"] <= rt["peak_reserved_bytes"] < rt["device_total_bytes"], "No measured GPU memory headroom")
    return {k: rt[k] for k in ("device", "device_total_bytes", "peak_allocated_bytes", "peak_reserved_bytes")}


def freeze_models(cache, out, bindings):
    models = {}
    for label, (repo, revision, name, dtype) in PINS.items():
        directory = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        require(directory.is_dir(), "Pinned local model snapshot missing")
        files, targets = model_inventory(directory, repo, revision)
        required = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
        if label == "encoder":
            required |= {"pytorch_model.bin", "modules.json", "1_Pooling/config.json", "sentence_bert_config.json", "sentencepiece.bpe.model"}
        elif label == "bge":
            required |= {"model.safetensors", "sentencepiece.bpe.model"}
        else:
            required |= {"model.safetensors.index.json", "merges.txt", "vocab.json"}
            require("model.safetensors.index.json" in files, "Missing Qwen shard inventory")
            mapping = read(directory / "model.safetensors.index.json")["weight_map"]
            required |= set(mapping.values())
            require(len(mapping) == 398, "Qwen4B original tensor inventory differs")
        require(required <= set(files), "Incomplete pinned snapshot")
        for relative, digest in files.items():
            path = directory / relative
            bindings[str(path.resolve())] = digest
            if path.suffix not in (".bin", ".safetensors"):
                snapshot(path, out / "model-metadata" / label / relative, bindings)
        models[label] = {"repo": repo, "revision": revision, "name": name, "dtype": dtype,
                         "snapshot": str(directory.resolve()), "files": files,
                         "inventory_format": MODEL_INVENTORY_FORMAT, "file_targets": targets,
                         "bytes": sum(target["bytes"] for target in targets.values())}
        check_model_inventory(models[label])
    return models


def options():
    from risk_layer.pipeline import AnalyseOptionen
    return AnalyseOptionen(erzeugt_am="grounded-signal-public-baseline", detektor=True, streit=True,
                           semantik_profil=PROFILE, rechtsgebiete=())


def configure_environment(cache=None):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus", RISK_LAYER_FUSION="an",
                      RISK_LAYER_EMB_DEVICE="cuda", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      RISK_LAYER_RERANK_SCHWELLE="0.10", RISK_LAYER_SIM_RETTUNG="0.55")
    if cache is not None:
        os.environ["HF_HUB_CACHE"] = str(cache)


def foci_for(text):
    from risk_layer.semantic_context import build_context_queries
    return [asdict(f) for f in build_context_queries(text, mode="sentence")]


def prepare(args):
    from risk_layer.semantic_search import PROFILE_CONFIG
    inputs, out = args.inputs.resolve(), args.out.resolve()
    require(not out.exists(), "Use a fresh output directory")
    manifest_raw = (inputs / "input-manifest.json").read_bytes()
    manifest = _strict_json(manifest_raw)
    require(manifest.get("format") == "grounded_signal_inputs/1", "Unknown primary inputs")
    check_inventory(inputs, manifest["artifacts"], ("input-manifest.json",))
    primary = read(inputs / "protocol.json")
    cases = read(inputs / "cases.json")
    validate_cases(cases, primary)
    code = primary["source_hashes"]
    require(all(sha(ROOT / n) == d for n, d in code.items()), "Primary source freeze differs")
    out.mkdir(parents=True)
    bindings: dict[str, str] = {}
    for name in ("input-manifest.json", "protocol.json", "cases.json"):
        snapshot(inputs / name, out / "primary" / name, bindings)
    for name in code:
        snapshot(ROOT / name, out / "source" / name, bindings)
    snapshot(__file__, out / "probe-source.py", bindings)
    snapshot(args.gpu_reference.resolve(), out / "gpu-reference.json", bindings)
    reference = read(out / "gpu-reference.json")
    memory_basis = validate_gpu_reference(reference, code, PROFILE_CONFIG[PROFILE])
    snapshot(args.graph.resolve(), out / "graph.sqlite", bindings)
    snapshot(args.trigger.resolve(), out / "trigger.sqlite", bindings)
    snapshot(args.catalog.resolve(), out / "catalog.yaml", bindings)
    require(sha(out / "graph.sqlite") == primary["graph_sha256"], "Different source graph")
    index_files = inventory(args.index.resolve())
    expected_index = {Path(n).relative_to(args.index.resolve()).as_posix(): d
                      for n, d in primary["index_original_hashes"].items()}
    require(index_files == expected_index and set(index_files) >= {"meta.json", "vectors.npy"}, "Different production index")
    for name in index_files:
        snapshot(args.index / name, out / "index" / name, bindings)
    models = freeze_models(args.hf_cache.resolve(), out, bindings)
    foci = {c["case_id"]: foci_for(c["text"]) for c in cases}
    # Loading source/index metadata is model-free; incompatible sources fail before GPU allocation.
    graph, trigger, index, catalog = open_resources(out)
    try:
        source_identity = {"graph": graph.fingerprint(), "catalog": index.catalog_fingerprint,
                           "encoder": index.encoder_name, "revision": index.encoder_revision,
                           "passage_sources": index.passage_evidence.source_fingerprint}
    finally:
        graph.close()
        trigger.close()
    protocol = {"format": PREPARE_FORMAT, "profile": PROFILE, "profile_config": deepcopy(PROFILE_CONFIG[PROFILE]),
                "options": asdict(options()), "notes": NOTES, "primary_inputs": str(inputs),
                "primary_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "primary_artifacts": manifest["artifacts"], "source_hashes": code, "original_bindings": bindings,
                "case_ids": [c["case_id"] for c in cases], "foci": foci, "models": models,
                "hf_cache": str(args.hf_cache.resolve()), "runtime_packages": PACKAGES, "cpu_threads": 1,
                "source_identity": source_identity, "earlier_gpu_memory": memory_basis,
                "gpu_reference_sha256": sha(out / "gpu-reference.json"), "probe_source_sha256": sha(__file__),
                "capture": "Read-only Python call/return observation; exact Qwen2-pair inputs/token IDs and BGE pair inputs/scores. Included in timings.",
                "warmup": "One full first-case pass before measured rows; discarded output retained privately.",
                "scheduling": "Only after all generative/lexical model runs and unloading; no concurrent model workload."}
    save(out / "cases.json", cases)
    save(out / "protocol.json", protocol)
    check_files(bindings)
    for model in models.values():
        check_model_inventory(model)
    require(sha(inputs / "input-manifest.json") == protocol["primary_manifest_sha256"], "Primary manifest changed")
    check_inventory(inputs, manifest["artifacts"], ("input-manifest.json",))
    save(out / "prepared-manifest.json", {"format": PREPARE_FORMAT, "artifacts": inventory(out)})
    print(canonical({"status": "prepared", "cases": 16, "foci": sum(map(len, foci.values())), "model_calls": 0}), flush=True)


def open_resources(directory):
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import EmbeddingIndex, pruefe_aktualitaet
    from risk_layer.graph import NormGraph
    from risk_layer.triggerindex import TriggerIndex
    graph = NormGraph(directory / "graph.sqlite")
    trigger = None
    try:
        trigger = TriggerIndex(directory / "trigger.sqlite")
        index = EmbeddingIndex.load(directory / "index")
        catalog = lade_katalog(directory / "catalog.yaml")
        require(index is not None and index.passage_evidence is not None and pruefe_aktualitaet(index, graph, catalog),
                "Incoherent real pipeline resources")
        assert index is not None and index.passage_evidence is not None
        require(index.passage_evidence.matches_graph(graph) and index.passage_evidence.matches_identity(
            index.encoder_name, index.encoder_revision, index.catalog_fingerprint), "Passage generation differs")
        return graph, trigger, index, catalog
    except Exception:
        graph.close()
        if trigger is not None:
            trigger.close()
        raise


def source_audit(text, suggestions, index, graph):
    contexts = {(f["focus_start"], f["focus_end"]): f for f in foci_for(text)}
    parents = {p.passage_id: p for p in index.passage_evidence.passages}
    seen, count = set(), 0
    for sentence in suggestions:
        start, end = sentence.get("start"), sentence.get("end")
        require(type(start) is int and type(end) is int and (start, end) in contexts and (start, end) not in seen,
                "Invalid/duplicate original focus")
        seen.add((start, end))
        context = contexts[start, end]
        require(sentence["satz"] == text[start:end] == context["focus_text"] and sentence["profil"] == PROFILE,
                "Focus text/profile differs")
        require(canonical(sentence["kontext"]) == canonical({"start": context["context_start"],
                "end": context["context_end"], "text": context["query_text"]}), "Context differs")
        norms = sentence["normen"]
        require(0 < len(norms) <= 5 and len({n["id"] for n in norms}) == len(norms), "Invalid output cap/norm duplicate")
        for norm in norms:
            source = graph.node(norm["id"])
            require(source is not None and norm["titel"] == source["titel"], "Norm/source title differs")
            evidence = norm["evidenz"]
            require(evidence["fingerprint"] == index.passage_evidence.source_fingerprint and evidence["passagen"],
                    "Missing or different source evidence")
            require(all((norm["id"], anchor) in index.passage_evidence.anchors for anchor in evidence["anker"]),
                    "Catalog anchor belongs to another norm or source generation")
            for passage in evidence["passagen"]:
                parent = parents.get(passage["id"])
                a, b = passage["quell_start"], passage["quell_end"]
                require(parent is not None and type(a) is int and type(b) is int
                        and parent.norm_id == passage["norm_id"] == norm["id"]
                        and parent.start <= a < b <= parent.end and source["text"][a:b] == passage["text"]
                        and parent.text[a-parent.start:b-parent.start] == passage["text"], "Original passage span differs")
                count += 1
    return {"focuses": len(suggestions), "passages": count, "errors": [], "semantic_relevance": "unassessed"}


class CallCapture:
    """Observe original methods, without changing arguments, batching, outputs or models."""

    def __init__(self, targets):
        self.targets = targets
        self.events = []
        self.previous = None

    def __enter__(self):
        self.previous = sys.gettrace()
        require(self.previous is None, "An existing tracer could alter capture")
        sys.settrace(self.dispatch)
        return self

    def __exit__(self, *_):
        sys.settrace(self.previous)

    def dispatch(self, frame, event, arg):
        if event != "call" or frame.f_code not in self.targets:
            return None
        kind = self.targets[frame.f_code]
        local = frame.f_locals
        entry = {"kind": kind, "pairs": deepcopy(local["paare"]), "status": "entered"}
        if kind == "qwen":
            entry.update(instruction=local["instruction"], max_length=local["max_length"])
        self.events.append(entry)

        def observe(current, action, value):
            if action == "return":
                entry["status"] = "complete" if isinstance(value, list) else "error"
                if isinstance(value, list):
                    entry["scores"] = deepcopy(value)
                if kind == "qwen" and "ids" in current.f_locals:
                    entry["token_ids"] = deepcopy(current.f_locals["ids"])
                entry["pairs_sha256"] = value_sha(entry["pairs"])
            return observe
        return observe


def audit_capture(events, foci, config, *, complete):
    allowed = {f["focus_text"] for f in foci}
    pairs_by_query: Counter[str] = Counter()
    for event in events:
        pairs = event["pairs"]
        require(event["kind"] in ("bge", "qwen") and isinstance(pairs, list)
                and all(len(p) == 2 and p[0] in allowed and isinstance(p[1], str) for p in pairs), "Unexpected scored query")
        require(event["pairs_sha256"] == value_sha(pairs), "Scored pair hash differs")
        if complete:
            require(event["status"] == "complete", "Unfinished score call in a complete case")
        if event["status"] == "complete":
            require(len(event["scores"]) == len(pairs) and all(type(v) in (int, float) and math.isfinite(v)
                    for v in event["scores"]), "Invalid captured score count")
        if event["kind"] == "qwen":
            require(0 < len(pairs) <= 2 and event["instruction"] == config["instruction"] and event["max_length"] == 2048,
                    "Qwen batch/instruction/budget changed")
            if event["status"] == "complete":
                require(len(event.get("token_ids", [])) == len(pairs) and all(0 < len(ids) <= 2048
                        and all(type(i) is int and i >= 0 for i in ids) for ids in event["token_ids"]), "Missing/invalid actual Qwen tokens")
                require(all(0 <= score <= 1 for score in event["scores"]), "Invalid Qwen yes/no score")
            pairs_by_query.update(p[0] for p in pairs)
    require(all(count <= config["shortlist"] * len({f["query_text"] for f in foci if f["focus_text"] == query})
                for query, count in pairs_by_query.items()), "Qwen shortlist expanded")
    return {"qwen_batches": sum(e["kind"] == "qwen" for e in events),
            "qwen_pairs": sum(pairs_by_query.values()), "bge_calls": sum(e["kind"] == "bge" for e in events),
            "bge_pairs": sum(len(e["pairs"]) for e in events if e["kind"] == "bge")}


def hard_norm_ids(cards, risks):
    from risk_layer.idschema import para_level
    from risk_layer.zitate import parse_normzitat
    ids = {para_level(n["id"]) for r in risks for n in r["norm_vorschlag"] if isinstance(n.get("id"), str)}
    for card in cards:
        for citation in card["norm_anker"]:
            ids.update(para_level(a.norm_id.to_id()) for a in parse_normzitat(citation) if a.norm_id)
    return sorted(ids)


def hard_audit(text, cards, risks):
    for mark in [*cards, *risks]:
        start, end = mark["start"], mark["end"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(text)
                and text[start:end] == mark["matched_text"], "Hard marking is not an original text span")
    return {"cards": len(cards), "risk_marks": len(risks), "errors": [], "semantic_relevance": "unassessed"}


def output_from(raw):
    cards = [c.as_dict() for c in raw.karten]
    risks = [r.as_dict() for r in raw.risiken]
    return {"suggestions": deepcopy(raw.semantik_vorschlaege), "cards": cards, "risk_marks": risks,
            "hard_norm_ids": hard_norm_ids(cards, risks), "text_hash": raw.text_hash,
            "catalog_version": raw.katalog_version, "schicht2_info": raw.schicht2_info,
            "candidates": deepcopy(raw.kandidaten), "summary": raw.zusammenfassung()}


def summarize(cases, rows):
    observed = {r["case_id"]: r for r in rows}
    require(len(observed) == len(rows) and set(observed) <= {c["case_id"] for c in cases}, "Unknown/repeated case row")
    def group(selected, view):
        counts = Counter(planned_cases=len(selected))
        for case in selected:
            row = observed.get(case["case_id"], {"status": "not_run"})
            status = row["status"]
            require(status in ("complete", "technical_error", "invalid_output", "not_run"), "Unknown row status")
            counts[status] += 1
            labels = case["diagnostic_labels"]
            positive = labels["hint_expected"]
            negative_group = ("context_only" if labels.get("insufficient_context_norm_ids") else "clear_negative") if not positive else None
            targets, negative = set(labels["target_norm_ids"]), set(labels["no_hint_norm_ids"])
            counts["positive_cases" if positive else "non_hint_cases"] += 1
            if negative_group:
                counts[f"planned_{negative_group}_cases"] += 1
            counts["known_target_norm_pairs"] += len(targets)
            counts["explicit_no_hint_norm_pairs"] += len(negative)
            if status != "complete":
                counts["positive_cases_unknown" if positive else "non_hint_cases_unknown"] += 1
                counts["known_target_norm_pairs_unknown"] += len(targets)
                counts["explicit_no_hint_pairs_unknown"] += len(negative)
                if negative_group:
                    counts[f"{negative_group}_cases_unknown"] += 1
                continue
            output = row["output"]
            norms = {n["id"] for s in output["suggestions"] for n in s["normen"]}
            if view == "all_visible_norms":
                norms.update(output["hard_norm_ids"])
            counts["output_norm_pairs"] += len(norms)
            counts["known_target_norm_hits"] += len(targets & norms)
            counts["positive_case_hits"] += bool(positive and targets & norms)
            counts["non_hint_cases_with_output"] += bool(not positive and norms)
            counts["complete_non_hint_cases_without_output"] += bool(not positive and not norms)
            if negative_group:
                counts[f"{negative_group}_cases_with_output"] += bool(norms)
                counts[f"{negative_group}_output_norm_pairs"] += len(norms)
                counts[f"complete_{negative_group}_cases_without_output"] += not norms
            counts["explicit_no_hint_output_pairs"] += len(negative & norms)
            counts["explicit_no_hint_output_cases"] += bool(negative & norms)
            known = targets | negative | set(labels["allowed_hint_norm_ids"]) | set(labels.get("insufficient_context_norm_ids", []))
            counts["unadjudicated_output_pairs"] += len(norms - known)
        values: dict[str, Any] = {k: counts[k] for k in (
            "planned_cases", "complete", "technical_error", "invalid_output", "not_run", "positive_cases", "non_hint_cases",
            "positive_case_hits", "positive_cases_unknown", "known_target_norm_pairs", "known_target_norm_hits",
            "known_target_norm_pairs_unknown", "non_hint_cases_with_output", "complete_non_hint_cases_without_output",
            "non_hint_cases_unknown", "explicit_no_hint_norm_pairs", "explicit_no_hint_output_pairs",
            "explicit_no_hint_output_cases", "explicit_no_hint_pairs_unknown", "output_norm_pairs", "unadjudicated_output_pairs",
            "planned_clear_negative_cases", "clear_negative_cases_with_output", "clear_negative_output_norm_pairs",
            "clear_negative_cases_unknown", "complete_clear_negative_cases_without_output",
            "planned_context_only_cases", "context_only_cases_with_output", "context_only_output_norm_pairs",
            "context_only_cases_unknown", "complete_context_only_cases_without_output")}
        values.update(hint_recall_lower_bound=counts["positive_case_hits"] / counts["positive_cases"] if counts["positive_cases"] else None,
                      hint_recall_upper_bound=(counts["positive_case_hits"] + counts["positive_cases_unknown"]) / counts["positive_cases"] if counts["positive_cases"] else None,
                      precision=None, facts_and_conditions="not provided by this baseline")
        return values
    result = {view: {split: group([c for c in cases if split == "all" or c["split"] == split], view)
                    for split in ("all", "development", "evaluation")}
              for view in ("semantic_suggestions", "all_visible_norms")}
    result["timing"] = timing_summary(rows)
    return result


def timing_summary(rows):
    measured = [r["timing"]["elapsed_seconds"] for r in rows if "elapsed_seconds" in r.get("timing", {})]
    require(all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in measured), "Invalid elapsed time")
    times = sorted(measured)
    def quantile(q):
        if not times:
            return None
        position = (len(times) - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        return times[lower] + (times[upper] - times[lower]) * (position - lower)
    return {"recorded_attempts": len(times), "total_seconds": sum(times), "median_seconds": quantile(.5),
            "p95_seconds": quantile(.95), "max_seconds": max(times) if times else None,
            "synchronization_unavailable_attempts": sum(r.get("timing", {}).get("synchronized") is False for r in rows),
            "scope": "Synchronized Pipeline.run including read-only trace overhead and failed attempts; excludes model load, warmup, output audit, persistence and final hashing."}


def model_info(model, wrapper, label):
    devices = sorted({str(p.device) for p in model.parameters()})
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    require(all(str(b.device) == "cuda:0" for b in model.buffers()), "Model buffer is outside the declared GPU")
    require(devices == ["cuda:0"] and dtypes == [PINS[label][3]], "Model is not entirely on the declared GPU/dtype")
    require(wrapper.name == PINS[label][2] and wrapper.revision == PINS[label][1], "Loaded model revision/name differs")
    return {"name": wrapper.name, "revision": wrapper.revision, "devices": devices, "dtypes": dtypes}


def execute_cases(cases, out, report, run_one):
    for number, case in enumerate(cases, 1):
        print(f"treffer {number}/{len(cases)} {case['case_id']}", flush=True)
        row = run_one(case["text"])
        row.update(case_id=case["case_id"], family_id=case["family_id"], split=case["split"])
        save(out / f"case-{number:03d}.json", row)
        with (out / "rows.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(canonical(row) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        report["rows"].append(row)
        report["aggregate"] = summarize(cases, report["rows"])
        save(out / "report.json", report)


def run_gpu(cases, directory, out, report):
    # Import only in the explicit operator-run path, after all preflight bindings.
    import torch
    from risk_layer.embedding import BgeM3Encoder
    from risk_layer.pipeline import Pipeline
    from risk_layer.reranker import CascadedQwenReranker, CrossEncoderReranker, Qwen3Reranker
    torch.set_num_threads(1)
    require(torch.cuda.is_available(), "A real GPU is required")
    require(torch.cuda.get_device_name(0) == "AMD Radeon RX 9070 XT", "Different GPU from measured route")
    free, memory_api_total = torch.cuda.mem_get_info()
    total = torch.cuda.get_device_properties(0).total_memory
    basis = report["protocol"]["earlier_gpu_memory"]
    require(total == basis["device_total_bytes"] and free >= basis["peak_reserved_bytes"] + 1024**3,
            "Insufficient free VRAM; finish/unload other model runs first")
    graph, trigger, index, catalog = open_resources(directory)
    try:
        encoder, cascade = BgeM3Encoder(device="cuda", local_files_only=True), CascadedQwenReranker(device="cuda")
        report["runtime"] = {"python": sys.version, "packages": dict(PACKAGES), "device": torch.cuda.get_device_name(0),
                             "hip": torch.version.hip, "cuda": torch.version.cuda, "free_bytes_before_load": free,
                             "memory_api_total_bytes": memory_api_total, "device_total_bytes": total}
        report["models"] = {}
        torch.cuda.reset_peak_memory_stats()
        for label, wrapper in (("encoder", encoder), ("bge", cascade.vorauswahl), ("qwen", cascade)):
            start = time.perf_counter()
            model = wrapper._laden()
            torch.cuda.synchronize()
            report["models"][label] = {**model_info(model.model if label == "bge" else model, wrapper, label),
                                        "load_seconds": time.perf_counter() - start}
        require(encoder.name == index.encoder_name and encoder.revision == index.encoder_revision, "Encoder/index mismatch")
        pipeline = Pipeline(catalog)
        targets = {CrossEncoderReranker.score.__code__: "bge", Qwen3Reranker.score_with_instruction.__code__: "qwen"}

        def one(text):
            capture = CallCapture(targets)
            start = time.perf_counter()
            row: dict[str, Any] = {"status": "technical_error", "output": None, "errors": [], "trace": capture.events}
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                with capture:
                    raw = pipeline.run(text, optionen=options(), graph=graph, trigger=trigger,
                                       embedding=index, embedding_encoder=encoder, reranker=cascade)
                torch.cuda.synchronize()
                row["timing"] = {"elapsed_seconds": time.perf_counter() - start, "synchronized": True}
                row["output"] = output_from(raw)
                row["source_audit"] = source_audit(text, row["output"]["suggestions"], index, graph)
                row["hard_audit"] = hard_audit(text, row["output"]["cards"], row["output"]["risk_marks"])
                row["capture_audit"] = audit_capture(capture.events, foci_for(text), report["protocol"]["profile_config"], complete=True)
                row["status"] = "complete"
            except Exception as exc:
                row["status"] = "invalid_output" if row["output"] is not None else "technical_error"
                row["errors"].append({"type": type(exc).__name__, "message": str(exc)})
                if "timing" not in row:
                    synchronized = True
                    try:
                        torch.cuda.synchronize()
                    except Exception as sync_exc:
                        synchronized = False
                        row["errors"].append({"type": type(sync_exc).__name__, "message": str(sync_exc),
                                              "operation": "synchronize_after_error"})
                    row["timing"] = {"elapsed_seconds": time.perf_counter() - start, "synchronized": synchronized}
                # Invalid raw output stays visible but never enters successful metrics.
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            return row

        warmup = one(cases[0]["text"])
        save(out / "warmup.json", warmup)
        report["warmup"] = {"case_id": cases[0]["case_id"], "status": warmup["status"],
                             "timing": warmup["timing"], "excluded_from_case_timings": True}
        # No retries or altered batch limits. A failed warmup is a setup failure.
        require(warmup["status"] == "complete", "Full-pipeline warmup failed; no settings are changed")
        execute_cases(cases, out, report, one)
        report["runtime"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        if "runtime" in report:
            try:
                report["runtime"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            except Exception as exc:
                report["runtime"]["peak_observation_error"] = str(exc)
        graph.close()
        trigger.close()


def run(args):
    prepared, out = args.prepared.resolve(), args.out.resolve()
    require(not out.exists(), "Use a fresh run output directory")
    manifest = read(prepared / "prepared-manifest.json")
    require(manifest["format"] == PREPARE_FORMAT, "Unknown prepared baseline")
    check_inventory(prepared, manifest["artifacts"], ("prepared-manifest.json",))
    protocol = read(prepared / "protocol.json")
    configure_environment(protocol["hf_cache"])
    require(protocol["probe_source_sha256"] == sha(__file__), "Probe differs from prepared source")
    check_files(protocol["original_bindings"])
    for model in protocol["models"].values():
        check_model_inventory(model)
    original = Path(protocol["primary_inputs"])
    require(sha(original / "input-manifest.json") == protocol["primary_manifest_sha256"], "Primary manifest changed")
    check_inventory(original, protocol["primary_artifacts"], ("input-manifest.json",))
    from risk_layer.semantic_search import PROFILE_CONFIG
    require(canonical(PROFILE_CONFIG[PROFILE]) == canonical(protocol["profile_config"])
            and canonical(asdict(options())) == canonical(protocol["options"]), "Public configuration changed")
    require({name: importlib.metadata.version(name) for name in PACKAGES} == PACKAGES, "Use the frozen .venv-amd GPU runtime")
    cases = read(prepared / "cases.json")
    require(canonical(cases) == canonical(read(prepared / "primary/cases.json")), "Original case annotations changed")
    require([c["case_id"] for c in cases] == protocol["case_ids"], "Run denominator differs")
    for case in cases:
        require(canonical(foci_for(case["text"])) == canonical(protocol["foci"][case["case_id"]]), "Focus plan changed")
    out.mkdir(parents=True)
    report = {"format": RUN_FORMAT, "status": "running", "protocol": protocol, "protocol_sha256": value_sha(protocol),
              "prepared": str(prepared), "prepared_manifest_sha256": sha(prepared / "prepared-manifest.json"),
              "rows": [], "errors": [], "notes": NOTES, "aggregate": summarize(cases, [])}
    save(out / "report.json", report)
    for name in ("protocol.json", "cases.json", "prepared-manifest.json", "probe-source.py"):
        shutil.copyfile(prepared / name, out / name)
    try:
        run_gpu(cases, prepared, out, report)
        report["status"] = "complete" if len(report["rows"]) == 16 and all(r["status"] == "complete" for r in report["rows"]) else "incomplete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        report["aggregate"] = summarize(cases, report["rows"])
        try:
            check_files(protocol["original_bindings"])
            for model in protocol["models"].values():
                check_model_inventory(model)
            check_inventory(prepared, manifest["artifacts"], ("prepared-manifest.json",))
            require(sha(prepared / "prepared-manifest.json") == report["prepared_manifest_sha256"], "Prepared manifest changed")
            original = Path(protocol["primary_inputs"])
            require(sha(original / "input-manifest.json") == protocol["primary_manifest_sha256"], "Primary manifest changed")
            check_inventory(original, protocol["primary_artifacts"], ("input-manifest.json",))
            report["end_verification"] = {"status": "pass", "original_files": len(protocol["original_bindings"])}
        except Exception as exc:
            report["status"] = "failed"
            report["end_verification"] = {"status": "failed", "error": str(exc)}
        report["artifacts"] = inventory(out, ("report.json",))
        save(out / "report.json", report)
    print(canonical({"status": report["status"], "rows": len(report["rows"]), "out": str(out)}), flush=True)
    return 0 if report["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--trigger", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--catalog", type=Path, default=ROOT / "catalog/begriffe.yaml")
    p.add_argument("--gpu-reference", type=Path, default=ROOT / "out/quality-recall-live/report.json")
    p.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("run")
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    configure_environment()
    if args.command == "prepare":
        prepare(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
