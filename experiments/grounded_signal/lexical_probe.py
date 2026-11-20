"""Post-hoc lexical-source probe; only the operator starts local model calls.

Reuses accepted source-free facts from a terminal same-model primary run. It
does not generate facts, fit a model, select cases by success, or change the
frozen grounded_signal implementation. All private traces remain local.
"""
# ruff: noqa: E402 -- direct private-script execution needs the repository bootstrap
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.grounded_signal import runner, runtime
from experiments.grounded_signal.evaluation import evaluate, validate_answer
from experiments.grounded_signal.sources import SourceCorpus, render_package
from risk_layer.norm_check import _strict_json
from risk_layer.research_agent import ResearchAgent, ResearchConfig, _json
from tools.quality_research_agent import bind_model, snapshot_server

ARM = "structured_lexical"
PREPARE_FORMAT = "grounded_signal_lexical_inputs/1"
RUN_FORMAT = "grounded_signal_lexical_probe/1"
NOTE = ("Post-hoc exploratory source selection prompted by CPU retrieval results on known synthetic diagnostics; "
        "not independent gold validation, not evidence of LLM training, no new facts call.")


class ProbeError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ProbeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def value_sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def read(path):
    return _strict_json(Path(path).read_bytes())


def save(path, value):
    runner.save(path, value)


def inventory(directory, exclude=()):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.relative_to(directory).as_posix() not in exclude and "__pycache__" not in p.parts}


def check_inventory(directory, hashes, *, exclude=()):
    require(isinstance(hashes, dict), "Missing artifact inventory")
    require(all(isinstance(name, str) and not Path(name).is_absolute() and
                ".." not in Path(name).parts and (directory / name).resolve().is_relative_to(directory.resolve())
                for name in hashes), "Invalid artifact path")
    require(inventory(directory, exclude) == hashes, "Artifact inventory changed")


def snapshot(source, destination, bindings):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(sha(source) == digest, "Input changed while read")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    require(sha(destination) == digest, "Snapshot differs")
    bindings[str(source.resolve())] = digest
    return _strict_json(raw) if source.suffix == ".json" else None


def check_files(bindings):
    require(all(sha(path) == digest for path, digest in bindings.items()), "Bound original input changed")


def choose_cases(cases, learned, protocol):
    require(isinstance(cases, list) and len(cases) == 16, "Primary plan must contain all sixteen cases")
    ids = [case["case_id"] for case in cases]
    require(len(set(ids)) == 16 and protocol.get("case_ids") == ids, "Primary case plan differs")
    require(protocol.get("arms") == list(runner.ARMS) and protocol.get("top_k") == 3,
            "Expected the fixed three-arm top-three primary plan")
    selected = [case for case in cases if case.get("split") == "evaluation"]
    require(len(selected) == 8 and len({case["family_id"] for case in selected}) == 4,
            "Select all eight evaluation cases, without outcome filtering")
    require(len([case for case in cases if case.get("split") == "development"]) == 8,
            "Unexpected development denominator")
    require({c["family_id"] for c in selected}.isdisjoint(
        {c["family_id"] for c in cases if c["split"] == "development"}), "Family split leakage")
    require(isinstance(learned, list) and len(learned) == 16 and [row.get("case_id") for row in learned] == ids,
            "Saved lexical rankings have different cases/order")
    by_id = {row["case_id"]: row for row in learned}
    plans = {}
    for case in selected:
        row = by_id[case["case_id"]]
        ranking = row.get("unadapted")
        require(row.get("split") == "evaluation" and isinstance(ranking, list) and len(ranking) == 3,
                "Expected exactly the stored unadapted top three")
        norms = [entry.get("norm_id") for entry in ranking]
        require(all(isinstance(n, str) and n for n in norms) and len(set(norms)) == 3,
                "Duplicate or malformed lexical candidate")
        require(all(type(entry.get("score")) in (int, float) and math.isfinite(entry["score"])
                    for entry in ranking), "Nonfinite lexical score")
        require(ranking == sorted(ranking, key=lambda entry: (-entry["score"], entry["norm_id"])),
                "Stored lexical rank order differs")
        plans[case["case_id"]] = norms
    return deepcopy(selected), plans


def verify_primary_inputs(directory):
    path = directory / "input-manifest.json"
    raw = path.read_bytes()
    manifest = _strict_json(raw)
    require(manifest.get("format") == "grounded_signal_inputs/1", "Unknown primary input format")
    check_inventory(directory, manifest["artifacts"], exclude=("input-manifest.json",))
    require(sha(path) == hashlib.sha256(raw).hexdigest(), "Primary manifest changed while checked")
    return manifest, hashlib.sha256(raw).hexdigest()


def prepare(args):
    inputs, out = args.inputs.resolve(), args.out.resolve()
    require(not out.exists(), "Use a fresh lexical input directory")
    own_sha, code = sha(__file__), runner.source_hashes()
    primary_manifest, primary_manifest_sha = verify_primary_inputs(inputs)
    primary = read(inputs / "protocol.json")
    require(primary["source_hashes"] == code, "Primary implementation differs before preparation")
    out.mkdir(parents=True)
    originals: dict[str, str] = {}
    for name in ("input-manifest.json", "protocol.json", "cases.json", "learning-diagnostic.json"):
        snapshot(inputs / name, out / "primary" / name, originals)
    snapshot(inputs / "graph.sqlite", out / "graph.sqlite", originals)
    snapshot(Path(__file__).resolve(), out / "probe-source.py", originals)
    cases, plans = choose_cases(read(out / "primary/cases.json"),
                               read(out / "primary/learning-diagnostic.json"), primary)
    require(sha(out / "graph.sqlite") == primary["graph_sha256"], "Primary graph differs")
    packages, case_packages, units = {}, {}, {}
    with SourceCorpus(out / "graph.sqlite") as corpus:
        for case in cases:
            package = render_package(corpus.package(plans[case["case_id"]], max_chars=32000, reference_depth=0))
            require(package["status"] == "ready", "A stored lexical top-three whole package is unavailable")
            packages[package["package_id"]] = package
            case_packages[case["case_id"]] = package["package_id"]
            units.update({unit["unit_id"]: unit for unit in package["units"]})
    protocol = {"format": PREPARE_FORMAT, "post_hoc_exploratory": True, "notes": [NOTE],
                "arms": [ARM], "arm_policies": {ARM: {"source_mode": "automatic"}},
                "case_packages": {ARM: case_packages}, "case_ids": [case["case_id"] for case in cases],
                "primary_inputs": str(inputs), "primary_input_manifest_sha256": primary_manifest_sha,
                "primary_source_hashes": code, "probe_source_sha256": own_sha,
                "selection": "all eight evaluation cases; stored unadapted top3; no label-based package choice",
                "top_k": 3, "max_package_chars": 32000, "reference_depth": 0,
                "facts": "accepted source-free facts from the same model's terminal primary run; no new extraction",
                "original_bindings": originals}
    # Audit the full planned denominator and complete source groups before any model can run.
    evaluate(cases, {}, protocol=protocol, source_units=units, source_packages=packages)
    for name, value in (("cases.json", cases), ("packages.json", packages),
                        ("source-units.json", units), ("protocol.json", protocol)):
        save(out / name, value)
    check_files(originals)
    require(runner.source_hashes() == code and sha(__file__) == own_sha, "Implementation changed during preparation")
    check_inventory(inputs, primary_manifest["artifacts"], exclude=("input-manifest.json",))
    require(sha(inputs / "input-manifest.json") == primary_manifest_sha, "Primary manifest changed")
    save(out / "prepared-manifest.json", {"format": PREPARE_FORMAT, "artifacts": inventory(out),
                                          "post_hoc_exploratory": True})
    print(json.dumps({"status": "prepared", "cases": 8, "packages": len(packages), "model_calls": 0}), flush=True)


def expected_fact_request(text, alias, config):
    spans = runtime.case_spans(text)
    return {"model": alias, "messages": [{"role": "system", "content": runtime.FACT_PROMPT},
        {"role": "user", "content": _json({"original_case": text, "case_spans": spans})}],
        "tools": [{"type": "function", "function": {"name": "extract_facts",
                   "description": "Vollständige strukturierte Antwort auf die angegebene Teilaufgabe.",
                   "parameters": runtime.facts_schema(spans)}}],
        "tool_choice": "required",
        "parallel_tool_calls": False, "stream": False, "cache_prompt": False,
        "max_tokens": config.max_output_tokens, "temperature": config.temperature, "top_p": config.top_p,
        "top_k": config.top_k, "min_p": config.min_p, "presence_penalty": config.presence_penalty,
        "seed": config.seed, "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_budget_tokens": config.reasoning_budget_tokens}


def audit_fact_result(fact, *, text, alias, config):
    require(isinstance(fact, dict) and isinstance(fact.get("status"), str), "Missing inherited fact status")
    complete = fact["status"] == "complete"
    require(isinstance(fact.get("trace"), list), "Missing inherited fact trace")
    expected = expected_fact_request(text, alias, config)
    paths = [event.get("path") for event in fact["trace"]]
    require(paths == ["/apply-template", "/tokenize", "/v1/chat/completions"][:len(paths)],
            "Unexpected/repeated inherited fact requests")
    if complete:
        require(len(paths) == 3 and not fact.get("errors"), "Incomplete successful fact trace")
    else:
        require(fact.get("errors") and fact.get("answer") is None, "Failed facts cannot be accepted")
    parsed: dict[str, Any] = {}
    for event in fact["trace"]:
        require(event.get("kind") == "http" and event.get("request_sha256") == value_sha(event.get("request")),
                "Inherited request byte hash differs")
        path, request = event["path"], event["request"]
        if path in ("/apply-template", "/v1/chat/completions"):
            require(value_sha(request) == value_sha(expected), "Facts request is not the exact source-free original-case request")
        else:
            require("/apply-template" in parsed and value_sha(request) == value_sha({
                    "content": parsed["/apply-template"].get("prompt"), "add_special": False, "parse_special": True}),
                    "Inherited tokenizer prompt differs")
        raw = event.get("raw_response")
        if raw is None:
            require(not complete and (event.get("error") or event.get("response_sha256")),
                    "Missing response without technical failure")
            continue
        require(isinstance(raw, str) and hashlib.sha256(raw.encode()).hexdigest() == event.get("response_sha256"),
                "Inherited response byte hash differs")
        try:
            value = _strict_json(raw)
        except (ValueError, UnicodeError):
            require(not complete, "Invalid JSON in accepted facts")
            continue
        if isinstance(value, dict):
            parsed[path] = value
        else:
            require(not complete, "Accepted facts have non-object response")
    timing = fact.get("timing", {}).get("elapsed_seconds")
    require(type(timing) in (int, float) and math.isfinite(timing) and timing >= 0, "Invalid inherited fact timing")
    if not complete:
        return {"available": False, "status": fact["status"], "source_free_requests_checked": len(paths),
                "fact_sha256": value_sha(fact), "elapsed_seconds": timing}
    tokenized = parsed["/tokenize"].get("tokens")
    require(isinstance(tokenized, list) and all(type(t) is int for t in tokenized), "Invalid inherited native tokens")
    assert isinstance(tokenized, list)
    response = parsed["/v1/chat/completions"]
    usage = response.get("usage", {})
    require(response.get("model") == alias and response.get("truncated") is not True and
            type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] == len(tokenized) and
            type(usage.get("completion_tokens")) is int and 0 <= usage["completion_tokens"] <= config.max_output_tokens and
            len(tokenized) + config.max_output_tokens <= config.context_tokens and
            type(usage.get("prompt_tokens_details", {}).get("cached_tokens")) is int and
            usage["prompt_tokens_details"]["cached_tokens"] == 0, "Inherited model/token budget differs")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and choices[0].get("finish_reason") in ("stop", "tool_calls"),
            "Inherited facts did not finish")
    assert isinstance(choices, list)
    require(isinstance(choices[0].get("message"), dict) and
            choices[0]["message"].get("role", "assistant") == "assistant", "Inherited response is not assistant output")
    message = {k: v for k, v in {**choices[0]["message"], "role": "assistant"}.items()
               if k in ("role", "content", "tool_calls", "reasoning_content")}
    require(value_sha(message) == value_sha(fact.get("message")), "Inherited assistant message differs")
    calls = message.get("tool_calls")
    require(isinstance(calls, list) and len(calls) == 1 and calls[0].get("type") == "function" and
            isinstance(calls[0].get("id"), str) and bool(calls[0]["id"]) and
            calls[0].get("function", {}).get("name") == "extract_facts", "Invalid inherited fact tool")
    assert isinstance(calls, list)
    arguments = _strict_json(calls[0]["function"]["arguments"])
    require(value_sha(arguments) == value_sha(fact.get("arguments")) and
            value_sha(runtime.expand_facts(arguments, runtime.case_spans(text))) == value_sha(fact.get("answer")),
            "Inherited fact expansion differs")
    require(type(fact.get("usage", {}).get("model_calls")) is int and fact["usage"]["model_calls"] == 1 and
            type(fact["usage"].get("http_requests")) is int and fact["usage"]["http_requests"] == len(paths) and all(
        type(fact["usage"].get(key)) is int and fact["usage"].get(key) == usage[key] for key in ("prompt_tokens", "completion_tokens")),
        "Inherited usage differs")
    return {"available": True, "status": "complete", "source_free_requests_checked": len(paths),
            "fact_sha256": value_sha(fact), "elapsed_seconds": timing}


def model_identity(provenance):
    model = provenance.get("model", {})
    require(model.get("format") == "quality_research_model/1" and model.get("status") == "complete", "Invalid model receipt")
    item = model.get("model", {})
    require(type(item.get("bytes")) is int and item["bytes"] > 0 and isinstance(item.get("sha256"), str)
            and len(item["sha256"]) == 64, "Missing exact GGUF identity")
    return {key: model.get(key) for key in ("repo", "revision", "filename", "quantization")} | {
        "bytes": item["bytes"], "sha256": item["sha256"]}


def same_model(base, current):
    require(value_sha(model_identity(base)) == value_sha(model_identity(current)), "Inherited facts belong to a different GGUF/model")


def validate_base(base_results, base_protocol, primary_protocol, *, primary_manifest_sha):
    require(base_results.get("status") in ("complete", "incomplete"), "Base run must be terminal, not partial")
    ids = primary_protocol["case_ids"]
    require(base_protocol.get("case_ids") == ids and base_protocol.get("arms") == list(runner.ARMS),
            "Inherited primary run is a subset or different plan")
    require(base_protocol.get("input_manifest_sha256") == primary_manifest_sha and
            base_protocol.get("source_hashes") == primary_protocol["source_hashes"] == runner.source_hashes() and
            base_results.get("source_end_hashes") == primary_protocol["source_hashes"], "Inherited primary input/source identity differs")
    require(base_protocol.get("case_packages") == primary_protocol["case_packages"] and
            base_protocol.get("arm_policies") == primary_protocol["arm_policies"], "Inherited primary package plan differs")
    results = base_results.get("results")
    require(isinstance(results, dict) and set(results) == set(runner.ARMS), "Missing primary arms")
    for arm in runner.ARMS:
        require(isinstance(results[arm], list) and [row.get("case_id") for row in results[arm]] == ids,
                "Primary arm case denominator differs")
        require(all(isinstance(row.get("status"), str) and row["status"] not in ("", "running", "not_run")
                    for row in results[arm]), "Primary arm still contains unfinished rows")
    require(isinstance(base_results.get("fact_results"), dict) and set(base_results["fact_results"]) == set(ids),
            "Primary fact ledger is incomplete")
    require(all(isinstance(fact, dict) and isinstance(fact.get("status"), str) and
                fact["status"] not in ("", "running", "not_run") for fact in base_results["fact_results"].values()),
            "Primary fact extraction is unfinished")


def load_prepared(path):
    manifest = read(path / "prepared-manifest.json")
    require(manifest.get("format") == PREPARE_FORMAT, "Unknown lexical input format")
    check_inventory(path, manifest["artifacts"], exclude=("prepared-manifest.json",))
    protocol = read(path / "protocol.json")
    require(protocol.get("probe_source_sha256") == sha(__file__) and
            protocol.get("primary_source_hashes") == runner.source_hashes(), "Probe or primary source changed")
    return manifest, protocol, read(path / "cases.json"), read(path / "packages.json"), read(path / "source-units.json")


def execute_cases(cases, packages, units, facts, protocol, agent, out):
    results: dict[str, list[dict[str, Any]]] = {ARM: []}
    initial = time.monotonic()
    report = {"format": RUN_FORMAT, "status": "running", "post_hoc_exploratory": True, "notes": [NOTE],
              "planned_cases": len(cases), "results": results, "new_fact_calls": 0}

    def persist():
        report["evaluation"] = evaluate(cases, results, protocol=protocol, source_units=units, source_packages=packages)
        save(out / "report.json", report)

    persist()
    for number, case in enumerate(cases, 1):
        cid = case["case_id"]
        package = packages[protocol["case_packages"][ARM][cid]]
        inherited = deepcopy(facts[cid])
        original_sha = value_sha(inherited)
        call: dict[str, Any] = {"status": "upstream_failure", "errors": ["Inherited facts are unavailable; no new extraction"],
                "trace": [], "timing": {"elapsed_seconds": 0.0}, "usage": {"model_calls": 0}}
        fatal = None
        try:
            if inherited["status"] == "complete":
                call = runtime.decide(agent, case["text"], package, inherited, direct=False)
            require(value_sha(inherited) == original_sha, "Deciding modified the inherited fact ledger")
            if call["status"] == "complete":
                audit = validate_answer(call.get("answer"), text=case["text"],
                                        candidate_norm_ids=package["candidate_norm_ids"],
                                        source_units=units, available_unit_ids=package["unit_ids"])
                require(audit["valid"], "Independent attribution check failed: " + str(audit["errors"]))
                call["attribution_audit"] = audit
        except BaseException as exc:
            fatal = exc
            call["status"] = "probe_error"
            call.setdefault("errors", []).append(type(exc).__name__ + ": " + str(exc))
            if "answer" in call:
                call["rejected_answer"] = call.pop("answer")
        decision = call.get("timing", {}).get("elapsed_seconds", 0.0)
        inherited_seconds = facts[cid]["timing"]["elapsed_seconds"]
        row = {"case_id": cid, "source_package_id": package["package_id"],
               "candidate_norm_ids": package["candidate_norm_ids"], "status": call["status"],
               "answer": call.get("answer"), "error": call.get("errors", []),
               "usage": call.get("usage", {}), "inherited_fact_sha256": original_sha,
               "timing": {"elapsed_seconds": decision + inherited_seconds, "decision_seconds": decision,
                          "inherited_fact_seconds": inherited_seconds,
                          "scope": "standalone cost includes historical facts; only decision executed in this probe"}}
        results[ARM].append(row)
        save(out / f"case-{number:03}.json", {"row": row, "call": call})
        with (out / "rows.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(_json({"case_id": cid, "row": row}) + "\n")
            journal.flush()
        if fatal:
            report.update(status="failed", error=type(fatal).__name__ + ": " + str(fatal))
        persist()
        print(json.dumps({"case": number, "planned_cases": len(cases), "status": row["status"],
                          "decision_seconds": decision}), flush=True)
        if fatal:
            raise fatal
    report["execution_seconds"] = time.monotonic() - initial
    report["status"] = "complete" if all(row["status"] == "complete" for row in results[ARM]) else "incomplete"
    persist()
    return report


def run(args):
    prepared, base, out = args.prepared.resolve(), args.base_run.resolve(), args.out.resolve()
    # Fail early on a running base before any expensive asset validation.
    require(read(base / "results.json").get("status") in ("complete", "incomplete"), "Base run must be terminal")
    require(not out.exists(), "Use a fresh lexical result directory")
    manifest, prepared_protocol, cases, packages, units = load_prepared(prepared)
    prepared_manifest_sha = sha(prepared / "prepared-manifest.json")
    primary_inputs = Path(prepared_protocol["primary_inputs"])
    primary_manifest, primary_sha = verify_primary_inputs(primary_inputs)
    require(primary_sha == prepared_protocol["primary_input_manifest_sha256"], "Primary input receipt differs")
    primary_protocol = read(prepared / "primary/protocol.json")
    base_artifacts = inventory(base)
    base_protocol, base_results = read(base / "protocol.json"), read(base / "results.json")
    validate_base(base_results, base_protocol, primary_protocol, primary_manifest_sha=primary_sha)
    config = ResearchConfig(max_output_tokens=3072, timeout_seconds=300, reuse_prompt_cache=True)
    require(value_sha(base_protocol.get("configuration")) == value_sha(asdict(config)), "Primary generation configuration differs")
    base_model = read(base / "model-manifest.json")
    require(base_protocol["model_binding"]["model_manifest_sha256"] == sha(base / "model-manifest.json") and
            base_protocol["model_binding"]["model"] == base_model, "Inherited model receipt differs")
    alias = base_protocol["model_binding"]["launch"]["alias"]
    facts, fact_audits = {}, {}
    all_ids = primary_protocol["case_ids"]
    for case in cases:
        cid = case["case_id"]
        filename = f"facts-{all_ids.index(cid) + 1:03}.json"
        fact = read(base / filename)
        require(value_sha(fact) == value_sha(base_results["fact_results"][cid]), "Standalone inherited facts differ from primary ledger")
        fact_audits[cid] = audit_fact_result(fact, text=case["text"], alias=alias, config=config)
        facts[cid] = fact
    check_inventory(base, base_artifacts)
    out.mkdir(parents=True)
    originals: dict[str, str] = {}
    for name in ("protocol.json", "results.json", "model-manifest.json"):
        snapshot(base / name, out / "inherited" / name, originals)
    for case in cases:
        cid = case["case_id"]
        name = f"facts-{all_ids.index(cid) + 1:03}.json"
        snapshot(base / name, out / "inherited" / name, originals)
    snapshot(Path(__file__).resolve(), out / "probe-source.py", originals)
    snapshot(prepared / "prepared-manifest.json", out / "prepared-manifest.json", originals)
    save(out / "fact-audits.json", fact_audits)
    bound_model: dict[str, str] = {}
    provenance = bind_model(args, out, bound_model, config)
    same_model(base_protocol["model_binding"], provenance)
    server = snapshot_server(args.url, args.model, out, 30, provenance, config)
    run_protocol = {**prepared_protocol, "format": RUN_FORMAT, "prepared_directory": str(prepared),
                    "prepared_manifest_sha256": prepared_manifest_sha, "base_run": str(base),
                    "base_artifacts": base_artifacts, "base_protocol_sha256": sha(base / "protocol.json"),
                    "base_results_sha256": sha(base / "results.json"), "inherited_fact_audits": fact_audits,
                    "configuration": asdict(config), "model_binding": provenance, "server": server,
                    "model_input_hashes": bound_model, "run_original_bindings": originals,
                    "timing": "No new facts inference; decision latency and historical standalone fact cost recorded separately"}
    save(out / "protocol.json", run_protocol)
    preflight_artifacts = inventory(out)
    save(out / "preflight-manifest.json", {"artifacts": preflight_artifacts})
    check_files(originals)
    check_inventory(base, base_artifacts)
    check_inventory(prepared, manifest["artifacts"], exclude=("prepared-manifest.json",))
    require(sha(prepared / "prepared-manifest.json") == prepared_manifest_sha and
            runner.source_hashes() == prepared_protocol["primary_source_hashes"] and
            sha(__file__) == prepared_protocol["probe_source_sha256"], "Input/source changed before inference")
    agent = ResearchAgent(args.url, args.model, config)
    run_error = None
    try:
        report = execute_cases(cases, packages, units, facts, run_protocol, agent, out)
    except BaseException as exc:
        run_error = exc
        report = read(out / "report.json")
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
    verification_start = time.monotonic()
    try:
        check_files(originals)
        check_files(bound_model)
        require(sha(__file__) == prepared_protocol["probe_source_sha256"] and
                runner.source_hashes() == prepared_protocol["primary_source_hashes"], "Implementation changed during probe")
        check_inventory(base, base_artifacts)
        check_inventory(prepared, manifest["artifacts"], exclude=("prepared-manifest.json",))
        require(sha(prepared / "prepared-manifest.json") == prepared_manifest_sha, "Prepared receipt changed")
        check_inventory(primary_inputs, primary_manifest["artifacts"], exclude=("input-manifest.json",))
        require(sha(primary_inputs / "input-manifest.json") == primary_sha, "Primary receipt changed")
        require(all(sha(out / name) == digest for name, digest in preflight_artifacts.items()), "Preflight archive changed")
        report["end_verification"] = "pass"
    except BaseException as exc:
        report.update(status="failed", end_verification="failed", verification_error=type(exc).__name__ + ": " + str(exc))
        if run_error is None:
            run_error = exc
    report["end_verification_seconds"] = time.monotonic() - verification_start
    report["artifact_hashes"] = inventory(out, exclude=("report.json",))
    save(out / "report.json", report)
    if run_error is not None:
        raise run_error
    return 0 if report["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--inputs", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--prepared", type=Path, required=True)
    execute.add_argument("--base-run", type=Path, required=True)
    execute.add_argument("--url", required=True)
    execute.add_argument("--model", required=True)
    execute.add_argument("--model-manifest", type=Path, required=True)
    execute.add_argument("--launch-manifest", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
