"""Compare fixed static/research agents on already examined local cases.

No generative model or server is started here. Optional hybrid retrieval loads
one pinned local CPU encoder. Historical labels stay exclusively in this
evaluator. All outputs are private research artifacts, including traces.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
CASE_FIELDS = ("case_id", "dataset", "group", "query", "targets", "negative", "category", "uncertain")
DATASETS = {"main": (102, 30), "additional": (15, 15)}
ARMS = ("static", "research")
MODEL_FTYPE_MAPPING_VERSION = "llama.cpp/model_ftype/1"
MODEL_FTYPE_BY_BUILD = {"b9986-91c631b21": {"Q4_K_M": "Q4_K - Medium", "Q6_K": "Q6_K"}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_event(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_reference(report):
    require(report.get("format") == "quality_recall_live/1"
            and report.get("status") in ("pass", "mismatch")
            and report.get("public_profile") == "treffer"
            and not report.get("diagnostic_subset") and not report.get("error"),
            "Reference must be a complete actual treffer live report")
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) == 162, "Reference must include all 162 cases")
    ids, counts = set(), Counter()
    for row in rows:
        require(all(key in row for key in CASE_FIELDS), "Reference case metadata is incomplete")
        require(isinstance(row["case_id"], str) and row["case_id"] and row["case_id"] not in ids,
                "Missing or duplicate reference case ID")
        ids.add(row["case_id"])
        require(row["dataset"] in DATASETS and type(row["negative"]) is bool
                and isinstance(row["query"], str) and bool(row["query"].strip()), "Invalid reference case")
        require(isinstance(row["targets"], list)
                and all(isinstance(value, str) and value.startswith("norm:") for value in row["targets"])
                and len(row["targets"]) == len(set(row["targets"]))
                and bool(row["targets"]) != row["negative"], "Reference target labels differ")
        require(isinstance(row.get("treffer", {}).get("suggestions"), list)
                and not row["treffer"].get("error"), "Reference lacks actual treffer suggestions")
        counts[(row["dataset"], row["negative"])] += 1
    require(all(counts[(name, False)] == positive and counts[(name, True)] == negative
                for name, (positive, negative) in DATASETS.items()), "Reference dataset denominators differ")
    require(report.get("source_audit", {}).get("errors") == [], "Reference source audit is not clean")
    return rows


def select_cases(rows, case_ids=(), case_limit=None):
    require(case_limit is None or type(case_limit) is int and case_limit > 0, "case-limit must be positive")
    require(len(case_ids) == len(set(case_ids)), "Duplicate selected case IDs")
    require(set(case_ids) <= {row["case_id"] for row in rows}, "Unknown selected case ID")
    selected = [row for row in rows if not case_ids or row["case_id"] in case_ids]
    if case_limit is not None:
        selected = selected[:case_limit]
    require(bool(selected), "No cases selected")
    return selected


def parse_arms(value):
    arms = value.split(",")
    require(bool(arms) and len(arms) == len(set(arms)) and set(arms) <= set(ARMS),
            "arms must be static,research or one of these two names")
    return arms


def validate_suggestions(text, suggestions, focuses):
    """Enforce the fixed original-sentence output budget without changing output."""
    require(isinstance(suggestions, list), "Agent suggestions must be a list")
    seen = set()
    for sentence in suggestions:
        require(isinstance(sentence, dict), "Malformed agent suggestion")
        start, end = sentence.get("start"), sentence.get("end")
        require(type(start) is int and type(end) is int and (start, end) in focuses
                and sentence.get("satz") == text[start:end], "Suggestion focus is not an original sentence")
        require((start, end) not in seen, "Duplicate suggestion focus")
        seen.add((start, end))
        norms = sentence.get("normen")
        require(isinstance(norms, list) and 1 <= len(norms) <= 5, "Output must contain at most five norms per focus")
        norm_ids = [norm.get("id") if isinstance(norm, dict) else None for norm in norms]
        require(all(isinstance(value, str) and value.startswith("norm:") for value in norm_ids)
                and len(norm_ids) == len(set(norm_ids)), "Invalid or duplicate norm ID within one focus")


def counts(rows, arm, *, diagnostic_subset):
    """Use every selected case; unavailable negatives cannot become successes."""
    value = {key: 0 for key in ("cases", "complete_cases", "failed_cases", "pending_cases", "positive_cases",
        "negative_cases", "failed_positive_cases", "failed_negative_cases", "pending_positive_cases",
        "pending_negative_cases", "hit5_cases", "historical_norm_targets", "matched_historical_norm_targets",
        "negative_with_suggestion", "negative_entries", "negative_unique_case_norm_pairs", "unresolved_cases")}
    for row in rows:
        value["cases"] += 1
        negative = row["negative"]
        value["negative_cases" if negative else "positive_cases"] += 1
        result = {"status": "complete", **row["baseline"]} if arm == "baseline" else row["arms"].get(arm, {}).get("result")
        if result is None or result.get("status") != "complete":
            prefix = "pending" if result is None else "failed"
            value[prefix + "_cases"] += 1
            value[prefix + ("_negative_cases" if negative else "_positive_cases")] += 1
            suggestions = []
        else:
            value["complete_cases"] += 1
            suggestions = result["suggestions"]
            value["unresolved_cases"] += bool(result.get("unresolved"))
        entries = [norm["id"] for sentence in suggestions for norm in sentence["normen"]]
        if negative:
            value["negative_with_suggestion"] += bool(entries)
            value["negative_entries"] += len(entries)
            value["negative_unique_case_norm_pairs"] += len(set(entries))
        else:
            targets = set(row["targets"])
            matched = targets.intersection(entries)
            value["hit5_cases"] += bool(matched)
            value["historical_norm_targets"] += len(targets)
            value["matched_historical_norm_targets"] += len(matched)
    unavailable = value["failed_negative_cases"] + value["pending_negative_cases"]
    negative = value["negative_cases"]
    positive = value["positive_cases"]
    value["rates"] = None if diagnostic_subset else {
        "hit5_lower_bound": value["hit5_cases"] / positive if positive else None,
        "negative_output_lower_bound": value["negative_with_suggestion"] / negative if negative else None,
        "negative_output_upper_bound": (value["negative_with_suggestion"] + unavailable) / negative if negative else None}
    value["suggestion_precision"] = {"status": "unknown", "reason": "Historical positive-case target lists are incomplete."}
    return value


def update_summaries(report):
    report["summaries"] = {dataset: {arm: counts(
        [row for row in report["rows"] if dataset == "all" or row["dataset"] == dataset], arm,
        diagnostic_subset=report["diagnostic_subset"]) for arm in ("baseline", *report["arms"])}
        for dataset in ("all", *DATASETS)}


def execute_cases(report, out, corpus, agent_factory, focus_builder):
    """A fresh agent/session per case and arm; only text and corpus cross the boundary."""
    for number, row in enumerate(report["rows"], 1):
        focuses = {(focus.focus_start, focus.focus_end) for focus in focus_builder(row["query"])}
        for arm in report["arms"]:
            started = time.perf_counter()
            failure = None
            returned_result = None
            retrieval = None
            before_retrieval = None
            try:
                before_retrieval = retrieval_metrics(corpus)
                agent = agent_factory()
                result = agent.run(row["query"], corpus, allow_research=arm == "research")
                retrieval = retrieval_delta(before_retrieval, retrieval_metrics(corpus))
                require(isinstance(result, dict) and isinstance(result.get("status"), str), "Malformed agent result")
                # Reject nonserializable/nonfinite output before committing a journal entry.
                json.dumps(result, allow_nan=False)
                returned_result = result
                require(all(isinstance(result.get(key), list) for key in ("suggestions", "unresolved", "errors", "trace"))
                        and isinstance(result.get("timing"), dict) and isinstance(result.get("usage"), dict),
                        "Incomplete agent result contract")
                if result["status"] == "complete":
                    validate_suggestions(row["query"], result.get("suggestions"), focuses)
                    require(isinstance(result.get("initial_sources"), list)
                            and isinstance(result.get("initial_sources_sha256"), str)
                            and isinstance(result.get("focuses"), list), "Missing agent initialization provenance")
                    for previous in row["arms"].values():
                        prior = previous["result"]
                        if prior["status"] == "complete":
                            require(all(prior[key] == result[key] for key in
                                        ("initial_sources", "initial_sources_sha256", "focuses")),
                                    "Static/research initialization differs")
            except Exception as exc:
                failure = exc
                result = {"status": "runner_error", "suggestions": [], "unresolved": [], "trace": [],
                          "timing": {}, "usage": {}, "errors": [f"{type(exc).__name__}: {exc}"]}
                if before_retrieval is not None and retrieval is None:
                    try:
                        retrieval = retrieval_delta(before_retrieval, retrieval_metrics(corpus))
                    except Exception as metric_error:
                        result["errors"].append(f"Retrieval accounting failed: {type(metric_error).__name__}: {metric_error}")
            entry = {"result": result, "wall_seconds": time.perf_counter() - started}
            if retrieval is not None and "retrieval" in report:
                entry["retrieval_metrics"] = retrieval
                report["retrieval_metrics"] = retrieval_metrics(corpus)
            if failure is not None and returned_result is not None:
                entry["invalid_agent_result"] = returned_result
            append_event(out / "rows.jsonl", {"case_id": row["case_id"], "arm": arm, **entry})
            row["arms"][arm] = entry
            update_summaries(report)
            if failure is not None:
                report.update(status="failed", error=f"{type(failure).__name__}: {failure}")
            save(out / "report.json", report)
            print(json.dumps({"case": number, "planned_cases": len(report["rows"]), "arm": arm,
                              "status": result["status"], "seconds": entry["wall_seconds"]}), flush=True)
            if failure is not None:
                raise failure


def snapshot_file(source, destination, inputs):
    source, destination = Path(source).resolve(), Path(destination)
    expected = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    require(digest(destination) == expected and digest(source) == expected, "Input changed during snapshot")
    inputs[str(source)] = expected
    return expected


def freeze_sources(out):
    files = sorted((ROOT / "risk_layer").rglob("*.py")) + [Path(__file__).resolve(),
        ROOT / "tools/quality_research_model.py"]
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        hashes[relative.as_posix()] = snapshot_file(source, out / "source" / relative, {})
    return hashes


def check_bindings(inputs, sources):
    for filename, expected in inputs.items():
        require(digest(filename) == expected, "Bound input changed during experiment")
    for relative, expected in sources.items():
        require(digest(ROOT / relative) == expected, "Bound implementation changed during experiment")


def check_frozen(out, snapshots):
    for relative, expected in snapshots.items():
        require(digest(out / relative) == expected, "Frozen artifact changed during experiment")


def freeze_json(source, destination, inputs):
    source = Path(source).resolve()
    raw = source.read_bytes()
    value = parse_json(raw)
    sha = hashlib.sha256(raw).hexdigest()
    require(digest(source) == sha, "JSON input changed while it was read")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    inputs[str(source)] = sha
    return value, sha


def freeze_encoder(mode, model_path, out, inputs):
    """Copy and bind local encoder files before any runtime inference/import."""
    require(mode in ("bm25", "hybrid"), "Unknown retrieval mode")
    if mode == "bm25":
        require(model_path is None, "BM25 does not use --encoder-model")
        return None, {"mode": mode, "files": {}}
    require(model_path is not None, "Hybrid requires an explicit local --encoder-model")
    source = Path(model_path).resolve()
    require(source.is_dir() and not out.resolve().is_relative_to(source), "Invalid local encoder directory")
    destination = out / "inputs/encoder"
    # A manifest-free HF snapshot is identified by this complete pinned suffix.
    if source.parent.name == "snapshots" and source.parent.parent.name == "models--BAAI--bge-m3":
        destination = destination / source.parent.parent.name / "snapshots" / source.name
    files = sorted(path for path in source.rglob("*") if path.is_file())
    require(bool(files), "Local encoder directory is empty")
    hashes = {path.relative_to(source).as_posix(): snapshot_file(path, destination / path.relative_to(source), inputs)
              for path in files}
    return destination, {"mode": mode, "source_directory": str(source),
                         "snapshot_directory": destination.relative_to(out).as_posix(), "files": hashes}


def freeze_catalog(path, out, inputs, reference_hashes):
    if path is None:
        return None, None
    source = Path(path).resolve()
    raw = source.read_bytes()
    expected = hashlib.sha256(raw).hexdigest()
    reference = {value for name, value in reference_hashes.items() if Path(name).name in ("begriffe.yaml", "begriffe.yml")}
    require(not reference or reference == {expected}, "Catalog differs from the catalog bound by the live reference")
    require(digest(source) == expected, "Catalog changed while being read")
    destination = out / "inputs/catalog.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    require(digest(destination) == expected, "Catalog snapshot differs")
    inputs[str(source)] = expected
    return destination, {"snapshot": "inputs/catalog.yaml", "sha256": expected, "size_bytes": len(raw),
                         "reference_sha256": expected if reference else None}


def validate_catalog(metadata, provenance, path):
    if provenance is None:
        require(path is None and metadata is None, "Disabled catalog unexpectedly produced hints")
        return
    require(isinstance(metadata, dict) and metadata.get("format") in ("research_catalog/1", "research_catalog/2")
            and metadata.get("sha256") == provenance["sha256"] and type(metadata.get("size_bytes")) is int
            and metadata.get("size_bytes") == provenance["size_bytes"]
            and isinstance(metadata.get("path"), str) and Path(metadata["path"]).resolve() == path.resolve(),
            "Catalog identity differs from frozen YAML")


def validate_retrieval(metadata, provenance, encoder_path):
    require(isinstance(metadata, dict) and metadata.get("mode") == provenance["mode"], "Corpus retrieval mode differs")
    if provenance["mode"] == "bm25":
        require(encoder_path is None and not metadata.get("model_identity"), "BM25 unexpectedly uses an encoder")
        return
    if "format" in metadata or "query_vector" in metadata:
        vector = metadata.get("query_vector")
        require(metadata.get("format") == "research_cpu_hybrid/2"
                and isinstance(vector, dict) and type(vector.get("dimension")) is int
                and vector == {"dtype": "float32", "dimension": 1024, "byte_order": "little", "order": "C"},
                "Unknown hybrid query-vector format")
    model = metadata.get("model_identity")
    require(isinstance(model, dict) and model.get("files") == provenance["files"]
            and model.get("encoder") == "bge-m3" and model.get("device") == "cpu"
            and model.get("dtype") == "float32" and model.get("local_files_only") is True
            and type(model.get("cpu_threads")) is int and model["cpu_threads"] == 2
            and isinstance(metadata.get("model_path"), str)
            and Path(metadata["model_path"]).resolve() == encoder_path.resolve(),
            "Hybrid encoder identity/device differs from the frozen local files")


def retrieval_metrics(corpus):
    value = getattr(corpus, "retrieval_metrics", {})
    require(isinstance(value, dict), "Malformed retrieval metrics")
    return parse_json(json.dumps(value, allow_nan=False))


def retrieval_delta(before, after):
    require(set(before) == set(after), "Retrieval metric fields changed within an arm")
    result = {}
    for key, value in after.items():
        if key == "queries":
            require(isinstance(value, list) and isinstance(before[key], list)
                    and value[:len(before[key])] == before[key], "Retrieval query journal changed")
            result[key] = value[len(before[key]):]
        else:
            require(type(value) in (int, float) and type(before[key]) in (int, float)
                    and math.isfinite(value) and math.isfinite(before[key]) and value >= before[key] >= 0,
                    "Nonfinite or decreasing retrieval measurement")
            result[key] = value - before[key]
    return result


def local_base_url(value):
    parsed = urlsplit(value)
    require(parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
            and parsed.username is None and parsed.password is None and not parsed.query
            and not parsed.fragment and parsed.path in ("", "/")
            and parsed.port is not None, "Use an explicit loopback HTTP server URL without an API suffix")
    return value.rstrip("/")


def bind_model(args, out, inputs, config):
    model, model_sha = freeze_json(args.model_manifest, out / "model-manifest.json", inputs)
    launch, launch_sha = freeze_json(args.launch_manifest, out / "launch-manifest.json", inputs)
    require(model.get("format") == "quality_research_model/1" and model.get("status") == "complete"
            and isinstance(model.get("repo"), str) and isinstance(model.get("revision"), str)
            and len(model["revision"]) == 40 and isinstance(model.get("quantization"), str),
            "Expected a complete pinned research-model manifest")
    require(launch.get("format") == "quality_research_launch/1" and launch.get("status") == "healthy_handoff"
            and launch.get("model_manifest_sha256") == model_sha and launch.get("model") == model.get("model")
            and launch.get("base_url") == local_base_url(args.url) and launch.get("alias") == args.model
            and type(launch.get("pid")) is int and launch["pid"] > 0,
            "Launch manifest does not identify this healthy server/model")
    item = model["model"]
    model_path = Path(item["path"]).resolve()
    require(type(item.get("bytes")) is int and item["bytes"] > 0
            and model_path.stat().st_size == item["bytes"] and digest(model_path) == item.get("sha256"),
            "Declared GGUF file differs")
    inputs[str(model_path)] = item["sha256"]
    executable = Path(launch["executable"]).resolve()
    engine = launch["engine_files"]
    require(isinstance(engine, dict) and executable.name in engine, "Missing server binary hash")
    for name, expected in engine.items():
        require(name == Path(name).name and "/" not in name and "\\" not in name, "Invalid engine filename")
        path = executable.parent / name
        require(path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"],
                "Server binary/library differs")
        inputs[str(path)] = expected["sha256"]
    source = args.launch_manifest.resolve().parent / "source.py"
    require(digest(source) == launch["source_sha256"], "Launch source archive differs")
    snapshot_file(source, out / "launch-source.py", inputs)
    argv = launch["argv"]
    require(isinstance(argv, list) and all(isinstance(v, str) for v in argv)
            and argv and Path(argv[0]).resolve() == executable, "Invalid server argv")
    expected_args = {"-m": str(model_path), "--alias": args.model, "--ctx-size": str(config.context_tokens),
                     "--reasoning-budget": str(config.reasoning_budget_tokens), "--parallel": "1",
                     "--host": "127.0.0.1", "--port": str(urlsplit(args.url).port)}
    for flag, expected in expected_args.items():
        require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                and argv[argv.index(flag) + 1] == expected, "Server argument differs from benchmark protocol")
    setup_dir = args.launch_manifest.resolve().parent
    setup, setup_sha = freeze_json(setup_dir / "report.json", out / "model-setup-report.json", inputs)
    require(setup.get("format") == "quality_research_setup/1" and setup.get("status") == "pass"
            and setup.get("launch_manifest_sha256") == launch_sha, "Setup report does not bind this successful launch")
    props, props_sha = freeze_json(setup_dir / "props.response.json", out / "setup-props.json", inputs)
    require(setup.get("artifact_hashes", {}).get("props.response.json") == props_sha,
            "Original setup properties are not bound")
    return {"model_manifest_sha256": model_sha, "launch_manifest_sha256": launch_sha,
            "model": model, "launch": launch, "setup_report_sha256": setup_sha, "setup_props": props,
            "load_timing": "Server/model loading precedes this benchmark; consult the separately bound launch artifacts."}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Local metadata request must not redirect")


def validate_server(values, alias, provenance, config):
    require(alias in [item.get("id") for item in values["models"].get("data", []) if isinstance(item, dict)],
            "Server does not advertise the requested model alias")
    props, original = values["props"], provenance["setup_props"]
    ftypes = MODEL_FTYPE_BY_BUILD.get(props.get("build_info"), {})
    quantization = provenance["model"]["quantization"]
    require(quantization in ftypes, "Unsupported server build or canonical quantization for model_ftype mapping")
    require(props.get("model_alias") == alias and isinstance(props.get("model_path"), str)
            and Path(props["model_path"]).resolve() == Path(provenance["model"]["model"]["path"]).resolve()
            and props.get("model_ftype") == original.get("model_ftype") == ftypes[quantization], "Live model properties differ")
    require(type(props.get("total_slots")) is int and props["total_slots"] == 1
            and props.get("default_generation_settings", {}).get("n_ctx") == config.context_tokens,
            "Live server context/slot budget differs")
    require(isinstance(props.get("build_info"), str) and bool(props["build_info"])
            and props["build_info"] == original.get("build_info")
            and isinstance(props.get("chat_template"), str) and bool(props["chat_template"])
            and props["chat_template"] == original.get("chat_template"), "Live server build/template differs from setup")


def snapshot_server(base, alias, out, timeout, provenance, config):
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    values, hashes = {}, {}
    for name, route in (("models", "/v1/models"), ("props", "/props")):
        with opener.open(base + route, timeout=timeout) as response:
            raw = response.read(2_097_153)
        require(len(raw) <= 2_097_152, "Oversized server metadata")
        values[name] = parse_json(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        (out / ("server-" + name + ".json")).write_bytes(raw)
    validate_server(values, alias, provenance, config)
    return {"response_hashes": hashes, "model_ftype_mapping_version": MODEL_FTYPE_MAPPING_VERSION, **values}


def prepare_inputs(args, out, report):
    inputs = report["input_hashes"]
    reference, sha = freeze_json(args.reference, out / "reference.json", inputs)
    rows = select_cases(validate_reference(reference), args.case_id, args.case_limit)
    report.update(reference_sha256=sha, diagnostic_subset=bool(args.case_id or args.case_limit is not None),
                  rows=[{**{key: row[key] for key in CASE_FIELDS},
                         "baseline": {"suggestions": row["treffer"]["suggestions"]}, "arms": {}} for row in rows])
    paths = {"graph": out / "inputs/graph.sqlite", "index": out / "inputs/index"}
    index_files = sorted(path for path in args.index.rglob("*") if path.is_file())
    require({"meta.json", "vectors.npy"} <= {path.name for path in index_files}, "Incomplete research index")
    expected = reference["input_hashes"]
    for source, destination in [(args.graph, paths["graph"]),
            *((path, paths["index"] / path.relative_to(args.index)) for path in index_files)]:
        hashes = {value for name, value in expected.items() if Path(name).name == source.name}
        require(len(hashes) == 1, "Graph/index file is not uniquely bound by the live reference")
        require(snapshot_file(source, destination, inputs) == next(iter(hashes)),
                "Graph/index differs from the actual treffer reference")
    report["source_hashes"] = freeze_sources(out)
    paths["catalog"], catalog_provenance = freeze_catalog(args.catalog, out, inputs, reference["input_hashes"])
    if catalog_provenance is not None:
        report["catalog_provenance"] = catalog_provenance
    return paths


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference", type=Path, default=ROOT / "out/quality-recall-live/report.json")
    result.add_argument("--graph", type=Path, default=ROOT / "out/gpu-ab-20260907-final/artifacts/graph.sqlite")
    result.add_argument("--index", type=Path, default=ROOT / "out/quality-production-final/index")
    result.add_argument("--url", default="http://127.0.0.1:8089")
    result.add_argument("--model", required=True)
    result.add_argument("--model-manifest", type=Path, required=True)
    result.add_argument("--launch-manifest", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--case-limit", type=int)
    result.add_argument("--case-id", action="append", default=[])
    result.add_argument("--arms", default="static,research")
    result.add_argument("--timeout", type=float, default=180.)
    result.add_argument("--reuse-prompt-cache", action="store_true", help="Reuse prompt cache only after the first completion in each case/arm")
    result.add_argument("--finish-reserve", type=float, default=0., help="Explicit final-answer time reserve in seconds; zero preserves the legacy policy")
    result.add_argument("--retrieval", choices=("bm25", "hybrid"), default="bm25")
    result.add_argument("--encoder-model", type=Path, help="Pinned local BGE model directory; required for hybrid")
    result.add_argument("--catalog", type=Path, help="Optional reference-bound catalog hints; never source evidence")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    require(not args.out.exists(), "Choose a fresh output directory")
    require(math.isfinite(args.timeout) and args.timeout > 0, "timeout must be positive and finite")
    args.url = local_base_url(args.url)
    arms = parse_arms(args.arms)
    out = args.out.resolve()
    out.mkdir(parents=True)
    report = {"format": "quality_research_agent/1", "status": "preflight", "arms": arms,
              "rows": [], "input_hashes": {}, "source_hashes": {}, "timing": {},
              "diagnostic_subset": bool(args.case_id or args.case_limit is not None),
              "notes": ["All 162 reference cases are already examined development data, not independent evaluation.",
                        "Historical labels and previous proposals remain in this evaluator and never enter the agent.",
                        "A fresh agent session per case/arm uses identical initial sources and fixed budgets.",
                        "Technical failures and pending cases stay in the denominators; failed negatives are not successful abstentions.",
                        "Incomplete positive-case norm labels cannot establish suggestion precision.",
                        "Subset runs are diagnostics only and do not publish full-benchmark rates.",
                        "This private report includes queries, model requests/responses and source quotes. Do not publish it unchanged."]}
    corpus = None
    try:
        require(not any(name == "risk_layer" or name.startswith("risk_layer.") for name in sys.modules),
                "Runtime was imported before the source snapshot")
        paths = prepare_inputs(args, out, report)
        encoder_path, report["encoder_provenance"] = freeze_encoder(args.retrieval, args.encoder_model, out, report["input_hashes"])
        sys.path[:0] = [str(out / "source"), str(ROOT)]
        from risk_layer.research_agent import ResearchAgent, ResearchConfig
        from risk_layer.research_sources import ResearchCorpus
        from risk_layer.semantic_context import build_context_queries
        for name in ("risk_layer.research_agent", "risk_layer.research_sources", "risk_layer.semantic_context"):
            require(Path(sys.modules[name].__file__).resolve().is_relative_to(out / "source"), "Wrong frozen runtime import")
        config = ResearchConfig(timeout_seconds=args.timeout, reuse_prompt_cache=args.reuse_prompt_cache,
                                finish_reserve_seconds=args.finish_reserve)
        require(config.initial_sources == 6, "This comparison requires six fixed initial sources")
        report["agent_config"] = asdict(config)
        report["model_provenance"] = bind_model(args, out, report["input_hashes"], config)
        report["server"] = snapshot_server(args.url, args.model, out, args.timeout, report["model_provenance"], config)
        started = time.perf_counter()
        corpus = ResearchCorpus(paths["graph"], paths["index"], retrieval=args.retrieval, encoder_model_path=encoder_path,
                                catalog_path=paths["catalog"])
        report["corpus"] = corpus.identity_metadata()
        catalog_metadata = corpus.catalog_metadata()
        validate_catalog(catalog_metadata, report.get("catalog_provenance"), paths["catalog"])
        if catalog_metadata is not None:
            report["catalog_metadata"] = catalog_metadata
        report["retrieval"] = corpus.retrieval_metadata()
        validate_retrieval(report["retrieval"], report["encoder_provenance"], encoder_path)
        report["timing"]["corpus_load_seconds"] = time.perf_counter() - started
        report["snapshot_hashes"] = {path.relative_to(out).as_posix(): digest(path)
                                     for path in sorted(out.rglob("*")) if path.is_file()}
        protocol = {key: value for key, value in report.items() if key not in ("rows", "timing")}
        protocol.update(format="quality_research_agent_protocol/1", status="frozen_before_runs",
                        case_order=[row["case_id"] for row in report["rows"]], output_cap=5,
                        focus="original_sentence", server_url=args.url, model_alias=args.model)
        save(out / "protocol.json", protocol)
        report["protocol_sha256"] = digest(out / "protocol.json")
        check_bindings(report["input_hashes"], report["source_hashes"])
        check_frozen(out, report["snapshot_hashes"])
        started = time.perf_counter()
        corpus.prepare()
        report["timing"]["retrieval_prepare_seconds"] = time.perf_counter() - started
        report["retrieval_setup_metrics"] = retrieval_metrics(corpus)
        report["retrieval_metrics"] = retrieval_metrics(corpus)
        require(corpus.retrieval_metadata() == report["retrieval"], "Retrieval identity changed during prepare")
        require(corpus.catalog_metadata() == report.get("catalog_metadata"), "Catalog identity changed during prepare")
        require(args.retrieval == "hybrid" or "torch" not in sys.modules, "BM25 unexpectedly imported Torch")
        report["status"] = "running"
        update_summaries(report)
        save(out / "report.json", report)
        started = time.perf_counter()
        execute_cases(report, out, corpus, lambda: ResearchAgent(args.url, args.model, config),
                      lambda text: build_context_queries(text, mode="sentence"))
        report["timing"]["case_execution_seconds"] = time.perf_counter() - started
        require(corpus.retrieval_metadata() == report["retrieval"], "Retrieval identity changed during cases")
        require(corpus.catalog_metadata() == report.get("catalog_metadata"), "Catalog identity changed during cases")
        check_bindings(report["input_hashes"], report["source_hashes"])
        check_frozen(out, report["snapshot_hashes"])
        require(digest(out / "protocol.json") == report["protocol_sha256"], "Protocol changed during experiment")
        failures = sum(value["result"]["status"] != "complete" for row in report["rows"] for value in row["arms"].values())
        report.update(status="complete_with_errors" if failures else "complete", execution_complete=True,
                      technical_failures=failures, torch_imported="torch" in sys.modules)
        require(args.retrieval == "hybrid" or not report["torch_imported"], "BM25 unexpectedly imported Torch")
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", execution_complete=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if corpus is not None:
            try:
                corpus.close()
            except Exception as exc:
                report.update(status="failed", error=f"Corpus close failed: {type(exc).__name__}: {exc}")
        if report["rows"]:
            update_summaries(report)
        save(out / "report.json", report)
    print(json.dumps({"status": report["status"], "cases": len(report["rows"]),
                      "diagnostic_subset": report["diagnostic_subset"], "out": str(out)}), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
