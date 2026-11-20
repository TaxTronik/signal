"""Check the union of saved source50/agreement50 Top5 proposals via local JSON LLM.

Historical labels are used only for offline metrics, never in model prompts.
The check sees real neighboring source sentences and original norm passages.
This cannot recover candidates missing from both saved Top5 lists. Quote
grounding checks source presence, not legal correctness or model calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("source50", "agreement50")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_proposals(case, graph, build_context_queries, Passage):
    """Validate saved offsets/evidence and deduplicate only identical proposals."""
    contexts = {(c.focus_start, c.focus_end): c for c in build_context_queries(
        case["query"], mode="neighbors", neighbor_radius=2)}
    proposals = {}
    for variant in VARIANTS:
        for suggestion in case[variant]["suggestions"]:
            focus = (suggestion["start"], suggestion["end"])
            context = contexts.get(focus)
            if context is None or context.focus_text != suggestion["satz"]:
                raise ValueError("Saved focus is not an unchanged source sentence")
            prior_context = suggestion["kontext"]
            if case["query"][prior_context["start"]:prior_context["end"]] != prior_context["text"]:
                raise ValueError("Saved context differs from original query")
            for norm in suggestion["normen"]:
                key = (*focus, norm["id"])
                source = graph.node(norm["id"])
                if source is None or norm["titel"] != source["titel"]:
                    raise ValueError("Saved norm is absent or its title changed")
                passages = []
                for evidence in norm["evidenz"]["passagen"]:
                    start, end = evidence["quell_start"], evidence["quell_end"]
                    if (evidence["norm_id"] != norm["id"] or type(start) is not int
                            or type(end) is not int or not 0 <= start < end <= len(source["text"])
                            or source["text"][start:end] != evidence["text"]):
                        raise ValueError("Saved norm quote does not match original graph offsets")
                    passages.append(Passage(
                        evidence["id"], norm["id"], source["law"], source["para"],
                        source["titel"], start, end, evidence["text"]))
                if not passages:
                    raise ValueError("Saved proposal lacks source passages")
                signature = (norm["titel"], tuple(passages))
                if key in proposals and proposals[key]["signature"] != signature:
                    raise ValueError("Same focus/norm has different source packages across variants")
                if key not in proposals:
                    proposals[key] = {"context": context, "norm": norm,
                                      "passages": passages, "signature": signature}
    return proposals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Completed precision probe report")
    parser.add_argument("--graph", type=Path, help="Must match the precision report's graph input hash")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--llm-url", default="http://127.0.0.1:8087")
    parser.add_argument("--model", default="granite-4.1-8b-Q5_K_M.gguf")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--binary-file", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, help="Own-server exact argv/process metadata as JSON")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Choose a fresh output directory")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("case-limit must be positive")
    source_bytes = args.report.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    if source.get("status") != "complete":
        parser.error("Precision report must be complete")
    expected_graphs = {path: value for path, value in source["input_hashes"].items()
                       if Path(path).name == "graph.sqlite"}
    if len(expected_graphs) != 1:
        parser.error("Precision report must identify exactly one graph input")
    graph_path = args.graph or Path(next(iter(expected_graphs)))
    if digest(graph_path) != next(iter(expected_graphs.values())):
        parser.error("Graph does not match the precision report")
    cases = source["rows"]
    if len({case["case_id"] for case in cases}) != len(cases):
        parser.error("Duplicate case IDs")
    if args.case_id:
        missing = set(args.case_id) - {case["case_id"] for case in cases}
        if missing:
            parser.error(f"Unknown case IDs: {sorted(missing)}")
        cases = [case for case in cases if case["case_id"] in args.case_id]
    if args.case_limit:
        cases = cases[:args.case_limit]
    if not cases:
        parser.error("No cases selected")
    out = args.out.resolve()
    out.mkdir(parents=True)
    inputs = {str(path.resolve()): digest(path) for path in (graph_path, args.model_file, args.binary_file)}
    inputs[str(args.report.resolve())] = source_hash
    if digest(args.report) != source_hash:
        raise ValueError("Reference report changed after it was read")
    launch_manifest = None
    if args.launch_manifest:
        launch_bytes = args.launch_manifest.read_bytes()
        launch_manifest = json.loads(launch_bytes)
        inputs[str(args.launch_manifest.resolve())] = hashlib.sha256(launch_bytes).hexdigest()
        (out / "launch-manifest.json").write_bytes(launch_bytes)
    source_hashes = {}
    for path in sorted((ROOT / "risk_layer").rglob("*.py")) + [
            ROOT / "tools/_gpu_eval_metrics.py", Path(__file__)]:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[relative.as_posix()] = digest(target)
        if digest(path) != digest(target):
            raise ValueError("Source changed during snapshot")
    (out / "reference.json").write_bytes(source_bytes)
    sys.path[:0] = [str(out / "source"), str(ROOT)]
    from risk_layer.graph import NormGraph
    from risk_layer.norm_check import LocalNormChecker, NormCheckError, PROMPT_VERSION
    from risk_layer.passages import Passage
    from risk_layer.semantic_context import build_context_queries
    from tools._gpu_eval_metrics import summarize
    client = LocalNormChecker(args.llm_url, model=args.model, timeout=args.timeout,
                              max_tokens=args.max_tokens)
    result = {
        "format": "quality_norm_check_probe/1", "status": "running", "rows": [],
        "input_hashes": inputs, "source_hashes": source_hashes, "prompt_version": PROMPT_VERSION,
        "runtime": {"endpoint": client.base_url, "model_requested": args.model,
                    "model_file": str(args.model_file.resolve()), "binary_file": str(args.binary_file.resolve()),
                    "launch_manifest": launch_manifest,
                    "max_tokens": args.max_tokens, "temperature": 0, "seed": 0,
                    "context": "neighbors", "neighbor_radius": 2, "context_chars": 900},
        "diagnostic_subset": bool(args.case_id or args.case_limit or source.get("diagnostic_subset")),
        "notes": [
            "Previously seen historical cases, not a blind evaluation.",
            "Every check sees only focus, actual source context, norm identity/title and exact source passages.",
            "Gold labels and catalog anchors never enter prompts.",
            "Quote presence and a model judgment do not prove legal correctness or calibrated confidence.",
            "The union is limited to previously returned Top5 proposals; rejected suggestions are not refilled.",
            "Cases with backend errors are excluded from all metric comparisons and reported separately.",
            "Model/binary hashes identify operator-declared files; server metadata is recorded independently.",
        ],
    }
    with client._opener.open(client.base_url + "/models", timeout=args.timeout) as response:
        server_models = response.read(1_048_577)
    if len(server_models) > 1_048_576:
        raise ValueError("Oversized server model metadata")
    result["runtime"]["server_models"] = json.loads(server_models)
    result["runtime"]["server_models_sha256"] = hashlib.sha256(server_models).hexdigest()
    allowed_models = {item["id"] for item in result["runtime"]["server_models"].get("data", [])
                      if isinstance(item, dict) and isinstance(item.get("id"), str)}
    if not allowed_models:
        raise ValueError("Server did not advertise a model identity")
    if args.model not in allowed_models:
        raise ValueError("Requested model alias is not advertised by the server")
    result["runtime"]["allowed_response_models"] = sorted(allowed_models)
    save(out / "manifest.json", {k: v for k, v in result.items() if k != "rows"})
    graph = NormGraph(graph_path)
    started = time.perf_counter()
    try:
        for number, case in enumerate(cases, 1):
            proposals = prepare_proposals(case, graph, build_context_queries, Passage)
            assessments = {}
            errors = 0
            for key, proposal in proposals.items():
                context = proposal["context"]
                entry = {"case_id": case["case_id"], "focus_start": key[0], "focus_end": key[1],
                         "norm_id": key[2], "query_sha256": hashlib.sha256(case["query"].encode()).hexdigest()}
                before = time.perf_counter()
                try:
                    entry["check"] = client.check(context, key[2], proposal["norm"]["titel"], proposal["passages"])
                    if entry["check"]["model_response"] != args.model:
                        error = NormCheckError("Response model differs from advertised server model",
                                               raw_response=entry["check"]["raw_response"])
                        error.raw_http_sha256 = entry["check"]["raw_http_sha256"]
                        raise error
                    assessments[key] = entry["check"]
                except NormCheckError as exc:
                    errors += 1
                    entry.update(error=str(exc), raw_response=exc.raw_response,
                                 raw_response_sha256=exc.raw_response_sha256,
                                 raw_http_sha256=exc.raw_http_sha256)
                entry["seconds"] = time.perf_counter() - before
                with (out / "checks.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fields = ("case_id", "group", "query", "targets", "negative", "category", "uncertain", "dataset")
            row = {field: case.get(field) for field in fields}
            row.update(evaluation_eligible=errors == 0, check_errors=errors, checked_pairs=len(proposals))
            for variant in VARIANTS:
                row[variant] = case[variant]
                gated = []
                for suggestion in case[variant]["suggestions"]:
                    norms = []
                    for norm in suggestion["normen"]:
                        assessment = assessments.get((suggestion["start"], suggestion["end"], norm["id"]))
                        if assessment is not None and assessment["status"] == "passend":
                            norms.append({**norm, "norm_check": assessment})
                    if norms:
                        gated.append({**suggestion, "normen": norms})
                row[variant + "_checked"] = {"suggestions": gated}
            result["rows"].append(row)
            save(out / "report.json", result)
            print(f"{number}/{len(cases)} {case['case_id']}: {len(proposals)} checks, "
                  f"{sum(v['status'] == 'passend' for v in assessments.values())} passing, {errors} errors", flush=True)
        eligible = [row for row in result["rows"] if row["evaluation_eligible"]]
        variants = [*VARIANTS, *(name + "_checked" for name in VARIANTS)]
        result["summaries"] = {dataset: {
            name: summarize([row for row in eligible if row["dataset"] == dataset], name)
            for name in variants} for dataset in sorted({row["dataset"] for row in result["rows"]})}
        result["coverage"] = {"cases": len(cases), "eligible_cases": len(eligible),
                              "by_dataset": dict(Counter(row["dataset"] for row in eligible)),
                              "check_errors": sum(row["check_errors"] for row in result["rows"])}
        result["total_seconds"] = time.perf_counter() - started
        result["checks_sha256"] = digest(out / "checks.jsonl") if (out / "checks.jsonl").exists() else None
        if any(digest(path) != expected for path, expected in inputs.items()):
            raise ValueError("Input changed during generative checks")
        result["status"] = "complete" if len(eligible) == len(cases) else "incomplete"
    except Exception as exc:
        result["status"], result["error"] = "failed", repr(exc)
        raise
    finally:
        graph.close()
        save(out / "report.json", result)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
