"""Run the frozen 9B facts/lexical/decision path on every prepared full-test case."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from experiments.grounded_signal import runner, runtime
from experiments.grounded_signal.evaluation import validate_answer
from experiments.grounded_signal.lexical_probe import (
    check_files, check_inventory, inventory, sha, snapshot,
)
from risk_layer.research_agent import ResearchAgent, ResearchConfig
from tools.quality_research_agent import bind_model, snapshot_server


FORMAT = "grounded_signal_full_run/1"
PLANNED_CASES = 178


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protocol_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def elapsed(stage):
    value = stage["timing"]["elapsed_seconds"]
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Invalid recorded stage duration")
    return value


def skipped(status, message):
    return {"status": status, "errors": [message], "trace": [],
            "timing": {"elapsed_seconds": 0.0}, "usage": {"model_calls": 0, "http_requests": 0}}


def execute_cases(cases, packages, units, protocol, agent, out, report, persist):
    """No filtering, retries, historical facts or replacement model responses."""
    started = time.monotonic()
    for number, case in enumerate(cases, 1):
        cid = case["case_id"]
        package = packages[protocol["case_packages"][cid]]
        fatal = None
        fact_started = time.monotonic()
        try:
            fact = runtime.extract(agent, case["text"], thinking=True)
            elapsed(fact)
        except BaseException as exc:
            fatal = exc
            fact = skipped("execution_error", type(exc).__name__ + ": " + str(exc))
            fact["timing"]["elapsed_seconds"] = time.monotonic() - fact_started
        report["fact_results"][cid] = fact
        fact_path = out / f"facts-{number:03}.json"
        runner.save(fact_path, fact)
        fact_sha, fact_file_sha = protocol_hash(fact), sha(fact_path)
        persist()
        decision_started = time.monotonic()
        call = skipped("execution_error", "Decision did not start after an unexpected facts-stage error")
        if fatal is None:
            try:
                if fact["status"] != "complete":
                    call = skipped("upstream_failure", "Facts extraction did not complete")
                elif package["status"] != "ready":
                    require(package["status"] in ("overflow", "missing"), "Unknown package status")
                    call = skipped("source_package_" + package["status"], "Atomic whole-source package unavailable")
                else:
                    decision_fact = deepcopy(fact)
                    def check_facts():
                        require(protocol_hash(decision_fact) == protocol_hash(fact) ==
                                protocol_hash(report["fact_results"][cid]) == fact_sha
                                and sha(fact_path) == fact_file_sha, "Fresh facts ledger changed during decision")
                    check_facts()
                    decision_error: BaseException | None = None
                    try:
                        decision = runtime.decide(agent, case["text"], package, decision_fact, direct=False, thinking=True)
                        if isinstance(decision, dict):
                            call = decision
                    except BaseException as exc:
                        decision_error = exc
                    try:
                        check_facts()
                        call["fact_ledger_audit"] = {"status": "pass", "sha256": fact_sha}
                    except BaseException as exc:
                        call["fact_ledger_audit"] = {"status": "failed", "sha256": fact_sha, "error": str(exc)}
                        if decision_error is not None:
                            decision_error.add_note("Fresh facts ledger verification also failed: " + str(exc))
                            raise decision_error
                        raise
                    if decision_error is not None:
                        raise decision_error
                    require(isinstance(decision, dict), "Decision stage returned an invalid contract")
                    if call["status"] == "complete":
                        attribution = validate_answer(call["answer"], text=case["text"],
                            candidate_norm_ids=package["candidate_norm_ids"], source_units=units,
                            available_unit_ids=package["unit_ids"])
                        require(attribution["valid"], "Independent answer attribution failed: " + str(attribution["errors"]))
                        call["attribution_audit"] = attribution
                elapsed(call)
            except BaseException as exc:
                fatal = exc
                call["status"] = "execution_error"
                call.setdefault("errors", []).append(type(exc).__name__ + ": " + str(exc))
                call["timing"] = {"elapsed_seconds": time.monotonic() - decision_started}
                if "answer" in call:
                    call["rejected_answer"] = call.pop("answer")
        fact_time, decision_time = elapsed(fact), elapsed(call)
        row = {"case_id": cid, "status": call["status"],
               "answer": call.get("answer") if call["status"] == "complete" else None,
               "errors": call.get("errors", []), "fact_stage_status": fact["status"],
               "fresh_fact_sha256": fact_sha,
               "source_package_id": package["package_id"], "candidate_norm_ids": package["candidate_norm_ids"],
               "fact_usage": fact.get("usage", {}), "usage": call.get("usage", {}),
               "timing": {"fact_seconds": fact_time, "decision_seconds": decision_time,
                          "elapsed_seconds": fact_time + decision_time,
                          "scope": "Fresh facts plus decision; each physical stage is counted once."}}
        runner.save(out / f"decision-{number:03}.json", call)
        runner.save(out / f"case-{number:03}.json", {"row": row, "call": call})
        report["rows"].append(row)
        with (out / "rows.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
        report["execution_seconds"] = time.monotonic() - started
        if fatal is not None:
            report.update(status="failed", error=type(fatal).__name__ + ": " + str(fatal))
        persist()
        print(json.dumps({"case": number, "planned_cases": len(cases), "status": row["status"]}), flush=True)
        if fatal is not None:
            raise fatal
    report["executed_all"] = True
    report["all_rows_complete"] = all(row["status"] == "complete" for row in report["rows"])
    report["status"] = "complete" if report["all_rows_complete"] else "incomplete"
    report["execution_seconds"] = time.monotonic() - started
    persist()


def run(args):
    # Lazy import permits full_probe's CLI to delegate here without a cycle.
    from experiments.grounded_signal.full_probe import aggregate, load_prepared

    prepared, out = args.prepared.resolve(), args.out.resolve()
    require(not out.exists(), "Use a fresh full-run directory")
    manifest, protocol, cases, packages, units = load_prepared(prepared)
    require(len(cases) == PLANNED_CASES and len({c["case_id"] for c in cases}) == PLANNED_CASES
            and [c["case_id"] for c in cases] == protocol["case_ids"], "Every one of the 178 planned cases is required")
    require(set(protocol["case_packages"]) == set(protocol["case_ids"]), "Case package plan differs")
    prepared_sha = sha(prepared / "prepared-manifest.json")
    source_sha = sha(Path(__file__))
    config = ResearchConfig(max_output_tokens=3072, timeout_seconds=300, reuse_prompt_cache=True)
    run_protocol = {**protocol, "format": FORMAT, "prepared_format": protocol["format"],
                    "prepared_directory": str(prepared), "prepared_manifest_sha256": prepared_sha,
                    "configuration": asdict(config), "producer_sha256": source_sha,
                    "fact_policy": "One fresh source-free extraction for every case, including package overflow; no retries.",
                    "timing_scope": "Fresh serial facts and decisions; excludes prepared lexical retrieval and model/server startup."}
    report: dict[str, Any] = {"format": FORMAT, "status": "running", "planned_cases": len(cases),
                             "rows": [], "fact_results": {}, "executed_all": False, "all_rows_complete": False,
                             "protocol": run_protocol, "protocol_sha256": protocol_hash(run_protocol),
                             "execution_seconds": 0.0}
    out.mkdir(parents=True)
    originals: dict[str, str] = {}
    bound_model: dict[str, str] = {}
    frozen = None
    preflight_artifacts = None
    preflight_manifest_sha = None

    def persist():
        try:
            report["aggregate"] = aggregate(cases, report["rows"], packages, run_protocol)
        except BaseException as exc:
            report.update(status="failed", aggregate=None, aggregate_error=type(exc).__name__ + ": " + str(exc))
            runner.save(out / "report.json", report)
            raise
        runner.save(out / "report.json", report)

    def bindings_unchanged():
        check_inventory(prepared, manifest["artifacts"], exclude=("prepared-manifest.json",))
        require(sha(prepared / "prepared-manifest.json") == prepared_sha, "Prepared manifest changed")
        check_files(protocol["original_inputs"])
        check_files(originals)
        check_files(bound_model)
        require(runner.source_hashes() == protocol["source_hashes"] and sha(Path(__file__)) == source_sha,
                "Frozen implementation changed")
        if frozen is not None:
            require(frozen == protocol["source_hashes"], "Run source snapshot differs")
        if preflight_artifacts is not None:
            require(all(sha(out / name) == digest for name, digest in preflight_artifacts.items()),
                    "Run preflight snapshot changed")
            require(sha(out / "preflight-manifest.json") == preflight_manifest_sha,
                    "Run preflight manifest changed")

    failure = None
    setup_started = time.monotonic()
    try:
        persist()
        for name, target in (("prepared-manifest.json", "prepared-manifest.json"),
                             ("protocol.json", "prepared-protocol.json")):
            snapshot(prepared / name, out / target, originals)
        snapshot(Path(__file__).resolve(), out / "producer-source.py", originals)
        runner.save(out / "protocol.json", run_protocol)
        frozen = runner.freeze_code(out)
        bindings_unchanged()
        require(args.model == protocol["expected_model"]["alias"], "Requested model alias differs")
        provenance = bind_model(args, out, bound_model, config)
        require(provenance["model"]["model"]["sha256"] == protocol["expected_model"]["gguf_sha256"]
                and provenance["launch"]["alias"] == protocol["expected_model"]["alias"], "Bound model identity differs")
        server = snapshot_server(args.url, args.model, out, 30, provenance, config)
        run_protocol.update(model_binding=provenance, model_input_hashes=bound_model,
                            run_original_bindings=originals, server=server)
        report["protocol_sha256"] = protocol_hash(run_protocol)
        runner.save(out / "protocol.json", run_protocol)
        preflight_artifacts = inventory(out, exclude=("report.json",))
        runner.save(out / "preflight-manifest.json", {"artifacts": preflight_artifacts})
        preflight_manifest_sha = sha(out / "preflight-manifest.json")
        bindings_unchanged()
        report["setup_seconds"] = time.monotonic() - setup_started
        persist()
        execute_cases(cases, packages, units, run_protocol, ResearchAgent(args.url, args.model, config), out, report, persist)
    except BaseException as exc:
        failure = exc
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        try:
            persist()
        except BaseException:
            # The failed aggregation was already saved; retain the original error.
            pass
    verification_started = time.monotonic()
    try:
        bindings_unchanged()
        report["end_verification"] = "pass"
    except BaseException as exc:
        report.update(status="failed", end_verification="failed", verification_error=type(exc).__name__ + ": " + str(exc))
        if failure is None:
            failure = exc
    report["end_verification_seconds"] = time.monotonic() - verification_started
    report["artifact_hashes"] = inventory(out, exclude=("report.json",))
    runner.save(out / "report.json", report)
    if failure is not None:
        raise failure
    return 0 if report["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared", "model-manifest", "launch-manifest", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
