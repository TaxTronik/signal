from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from experiments.grounded_signal import full_run as probe


ALIAS = "signal-research-qwen35-9b-q6"
GGUF = "91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59"


def data(n=3):
    unit = {"unit_id": "original-unit", "norm_id": "norm:AO:1", "start": 0, "end": 5, "text": "Norm."}
    package = {"package_id": "p1", "status": "ready", "candidate_norm_ids": ["norm:AO:1"],
               "units": [unit], "unit_ids": [unit["unit_id"]], "complete_local_norms": True,
               "required_groups": [{"norm_id": "norm:AO:1", "unit_ids": [unit["unit_id"]]}]}
    cases = [{"case_id": f"c{i}", "text": f"Ein Fall {i}.", "suite": "test",
              "family_id": f"family{i}", "labels": {"secret": "LABEL_SENTINEL"}} for i in range(n)]
    protocol = {"format": "test_full_inputs/1", "case_ids": [c["case_id"] for c in cases],
                "case_packages": {c["case_id"]: "p1" for c in cases},
                "expected_model": {"alias": ALIAS, "gguf_sha256": GGUF}}
    return cases, {"p1": package}, {unit["unit_id"]: unit}, protocol


class FakeAgent:
    def __init__(self, url="unused", model=ALIAS, config=None):
        self.model = model
        self.config = config or probe.ResearchConfig(max_output_tokens=3072, timeout_seconds=300, reuse_prompt_cache=True)
        self.requests = []
        self.invalid_fact = False

    def _completion(self, request, deadline, result):
        self.requests.append(deepcopy(request))
        name = request["tools"][0]["function"]["name"]
        if name == "extract_facts":
            arguments = {"facts": [{"id": "f1", "span_id": "bad" if self.invalid_fact else "s1",
                "kind": "event", "polarity": "affirmed", "statement": "Ein genannter Fall."}], "unresolved": []}
        else:
            arguments = {"hints": {"norm:AO:1": {"status": "no_hint", "relation": "none",
                "evidence_unit_ids": [], "supported_fact_ids": [], "open_conditions": []}}, "unresolved": []}
        result["usage"].update(model_calls=1, http_requests=3, prompt_tokens=30, completion_tokens=20)
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "call", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def blank_report():
    return {"status": "running", "rows": [], "fact_results": {}, "executed_all": False,
            "all_rows_complete": False, "execution_seconds": 0.0}


def aggregate(cases, rows, packages, protocol):
    return {"planned": len(cases), "recorded": len(rows), "missing": len(cases) - len(rows),
            "statuses": dict(Counter(r["status"] for r in rows))}


def test_exact_runtime_two_cold_stages_and_labels_never_enter_requests(tmp_path):
    cases, packages, units, protocol = data()
    agent, report = FakeAgent(), blank_report()
    persisted = []
    probe.execute_cases(cases, packages, units, protocol, agent, tmp_path, report,
                        lambda: persisted.append(deepcopy(report)))
    assert report["executed_all"] and report["all_rows_complete"] and len(report["rows"]) == 3
    assert len(agent.requests) == 6
    for i, request in enumerate(agent.requests):
        assert request["cache_prompt"] is False and request["tool_choice"] == "required"
        assert request["max_tokens"] == 3072 and request["reasoning_budget_tokens"] == 768
        assert request["chat_template_kwargs"] == {"enable_thinking": True}
        assert "LABEL_SENTINEL" not in json.dumps(request)
        assert request["tools"][0]["function"]["name"] == ("extract_facts" if i % 2 == 0 else "decide_hints")
    assert len(persisted[0]["fact_results"]) == 1 and not persisted[0]["rows"]
    assert len(list(tmp_path.glob("facts-*.json"))) == len(list(tmp_path.glob("decision-*.json"))) == 3
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("package_status", ["overflow", "missing"])
def test_all_facts_even_when_every_package_is_unavailable(tmp_path, package_status):
    cases, packages, units, protocol = data()
    packages["p1"].update(status=package_status, units=[], unit_ids=[], complete_local_norms=False)
    agent, report = FakeAgent(), blank_report()
    probe.execute_cases(cases, packages, units, protocol, agent, tmp_path, report, lambda: None)
    assert len(agent.requests) == len(cases)
    assert report["executed_all"] and not report["all_rows_complete"]
    assert all(r["status"] == "source_package_" + package_status and r["answer"] is None
               and r["fact_stage_status"] == "complete" and r["timing"]["decision_seconds"] == 0 for r in report["rows"])


@pytest.mark.parametrize("package_status", ["ready", "overflow"])
def test_invalid_facts_are_upstream_failures_without_decisions(tmp_path, package_status):
    cases, packages, units, protocol = data()
    packages["p1"]["status"] = package_status
    agent, report = FakeAgent(), blank_report()
    agent.invalid_fact = True
    probe.execute_cases(cases, packages, units, protocol, agent, tmp_path, report, lambda: None)
    assert len(agent.requests) == 3 and report["status"] == "incomplete"
    assert all(r["status"] == "upstream_failure" and r["answer"] is None for r in report["rows"])


@pytest.mark.parametrize("error", [ValueError("contract"), RuntimeError("unexpected"), KeyboardInterrupt()])
def test_unexpected_failure_is_archived_and_reraised(tmp_path, monkeypatch, error):
    cases, packages, units, protocol = data()
    report = blank_report()
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(probe.runtime, "decide", fail)
    with pytest.raises(type(error)):
        probe.execute_cases(cases, packages, units, protocol, FakeAgent(), tmp_path, report, lambda: None)
    assert report["status"] == "failed" and not report["executed_all"]
    assert len(report["rows"]) == 1 and report["rows"][0]["status"] == "execution_error"
    assert report["rows"][0]["answer"] is None
    assert (tmp_path / "decision-001.json").is_file()
    assert aggregate(cases, report["rows"], packages, protocol)["missing"] == 2


def test_fact_exception_is_never_a_successful_empty_answer(tmp_path, monkeypatch):
    cases, packages, units, protocol = data()
    report = blank_report()
    def fail(*args, **kwargs):
        raise RuntimeError("facts failed")
    monkeypatch.setattr(probe.runtime, "extract", fail)
    with pytest.raises(RuntimeError, match="facts failed"):
        probe.execute_cases(cases, packages, units, protocol, FakeAgent(), tmp_path, report, lambda: None)
    assert report["fact_results"]["c0"]["status"] == "execution_error"
    assert report["rows"][0]["fact_stage_status"] == "execution_error"


def test_false_attribution_fails_run_and_preserves_rejected_answer(tmp_path, monkeypatch):
    cases, packages, units, protocol = data()
    monkeypatch.setattr(probe, "validate_answer", lambda *a, **k: {"valid": False, "errors": ["wrong_span"]})
    report = blank_report()
    with pytest.raises(ValueError, match="attribution"):
        probe.execute_cases(cases, packages, units, protocol, FakeAgent(), tmp_path, report, lambda: None)
    saved = json.loads((tmp_path / "decision-001.json").read_text())
    assert saved["status"] == "execution_error" and "rejected_answer" in saved and "answer" not in saved


@pytest.mark.parametrize("also_raises", [False, True])
def test_decision_cannot_mutate_original_facts_or_hide_a_prior_exception(tmp_path, monkeypatch, also_raises):
    cases, packages, units, protocol = data()
    report = blank_report()
    original_decide = probe.runtime.decide
    def mutate(agent, text, package, fact, **kwargs):
        result = original_decide(agent, text, package, fact, **kwargs)
        fact["answer"]["facts"][0]["statement"] = "Changed by decision"
        if also_raises:
            raise RuntimeError("original decision error")
        return result
    monkeypatch.setattr(probe.runtime, "decide", mutate)
    with pytest.raises(RuntimeError if also_raises else ValueError,
                       match="original decision error" if also_raises else "ledger changed"):
        probe.execute_cases(cases, packages, units, protocol, FakeAgent(), tmp_path, report, lambda: None)
    saved = json.loads((tmp_path / "facts-001.json").read_text())
    assert saved == report["fact_results"]["c0"]
    assert saved["answer"]["facts"][0]["statement"] == "Ein genannter Fall."
    decision = json.loads((tmp_path / "decision-001.json").read_text())
    assert decision["fact_ledger_audit"]["status"] == "failed"
    assert decision["fact_ledger_audit"]["sha256"] == probe.protocol_hash(saved)
    assert report["rows"][0]["answer"] is None and not report["executed_all"]


def test_fact_hash_and_file_are_bound_in_each_successful_decision(tmp_path):
    cases, packages, units, protocol = data(1)
    report = blank_report()
    probe.execute_cases(cases, packages, units, protocol, FakeAgent(), tmp_path, report, lambda: None)
    fact = json.loads((tmp_path / "facts-001.json").read_text())
    decision = json.loads((tmp_path / "decision-001.json").read_text())
    assert decision["fact_ledger_audit"] == {"status": "pass", "sha256": probe.protocol_hash(fact)}
    assert report["rows"][0]["fresh_fact_sha256"] == probe.protocol_hash(fact)


@pytest.fixture
def prepared_run(tmp_path, monkeypatch):
    cases, packages, units, protocol = data(178)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    original = tmp_path / "original.json"
    original.write_text('{}')
    code = {"experiments/grounded_signal/full_run.py": probe.sha(Path(probe.__file__))}
    protocol.update(source_hashes=code, original_inputs={str(original): probe.sha(original)})
    probe.runner.save(prepared / "protocol.json", protocol)
    probe.runner.save(prepared / "cases.json", cases)
    manifest = {"artifacts": probe.inventory(prepared)}
    probe.runner.save(prepared / "prepared-manifest.json", manifest)
    module = ModuleType("experiments.grounded_signal.full_probe")
    module.load_prepared = lambda _: (deepcopy(manifest), deepcopy(protocol), deepcopy(cases), deepcopy(packages), deepcopy(units))
    module.aggregate = aggregate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(probe.runner, "source_hashes", lambda: deepcopy(code))
    monkeypatch.setattr(probe.runner, "freeze_code", lambda _: deepcopy(code))
    monkeypatch.setattr(probe, "ResearchAgent", FakeAgent)
    counts = {"bind": 0, "server": 0}
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"fake model")
    def bind(args, out, bound, config):
        counts["bind"] += 1
        bound[str(model_file)] = probe.sha(model_file)
        return {"model": {"model": {"sha256": GGUF}}, "launch": {"alias": ALIAS}}
    def server(*args):
        counts["server"] += 1
        return {"status": "pass"}
    monkeypatch.setattr(probe, "bind_model", bind)
    monkeypatch.setattr(probe, "snapshot_server", server)
    args = SimpleNamespace(prepared=prepared, out=tmp_path / "run", url="http://127.0.0.1:8089",
                           model=ALIAS, model_manifest=tmp_path / "model.json", launch_manifest=tmp_path / "launch.json")
    return args, module, counts, original, model_file


def test_full_run_178_fresh_facts_decisions_and_final_receipts(prepared_run):
    args, _, counts, _, _ = prepared_run
    assert probe.run(args) == 0
    report = json.loads((args.out / "report.json").read_text())
    assert report["planned_cases"] == 178 and len(report["rows"]) == len(report["fact_results"]) == 178
    assert report["aggregate"]["planned"] == report["aggregate"]["recorded"] == 178
    assert report["executed_all"] and report["all_rows_complete"] and report["end_verification"] == "pass"
    assert report["protocol_sha256"] == probe.protocol_hash(report["protocol"])
    assert counts == {"bind": 1, "server": 1}
    assert "producer-source.py" in report["artifact_hashes"] and "facts-178.json" in report["artifact_hashes"]


@pytest.mark.parametrize("wrong", ["alias", "gguf"])
def test_model_preflight_failure_keeps_178_planned_and_zero_calls(prepared_run, monkeypatch, wrong):
    args, _, counts, _, _ = prepared_run
    if wrong == "alias":
        args.model = "other"
    else:
        monkeypatch.setattr(probe, "bind_model", lambda *a: {"model": {"model": {"sha256": "wrong"}}, "launch": {"alias": ALIAS}})
    with pytest.raises(ValueError, match="model"):
        probe.run(args)
    report = json.loads((args.out / "report.json").read_text())
    assert report["status"] == "failed" and not report["rows"] and not report["fact_results"]
    assert report["aggregate"] == {"planned": 178, "recorded": 0, "missing": 178, "statuses": {}}
    assert counts["server"] == 0 and not report["executed_all"]


@pytest.mark.parametrize("target", ["original", "model", "prepared", "source"])
def test_end_verification_still_runs_after_fatal_error(prepared_run, monkeypatch, target):
    args, _, _, original, model = prepared_run
    def fail(*a, **k):
        if target == "original":
            original.write_text('{"changed":true}')
        elif target == "model":
            model.write_bytes(b"changed")
        elif target == "prepared":
            (args.prepared / "extra.json").write_text('{}')
        else:
            monkeypatch.setattr(probe.runner, "source_hashes", lambda: {"changed": "yes"})
        raise RuntimeError("fatal case")
    monkeypatch.setattr(probe.runtime, "decide", fail)
    with pytest.raises(RuntimeError, match="fatal case"):
        probe.run(args)
    report = json.loads((args.out / "report.json").read_text())
    assert report["end_verification"] == "failed" and report["status"] == "failed"
    assert len(report["rows"]) == 1 and report["aggregate"]["missing"] == 177
    assert "verification_error" in report


def test_subset_not_allowed_before_inference(prepared_run):
    args, module, counts, _, _ = prepared_run
    original = module.load_prepared
    def load(path):
        manifest, protocol, cases, packages, units = original(path)
        return manifest, protocol, cases[:8], packages, units
    module.load_prepared = load
    with pytest.raises(ValueError, match="178"):
        probe.run(args)
    assert not args.out.exists() and counts == {"bind": 0, "server": 0}


def test_aggregation_failure_is_archived_rethrown_and_end_verified(prepared_run):
    args, module, _, _, _ = prepared_run
    def fail(*args):
        raise RuntimeError("aggregation contract failed")
    module.aggregate = fail
    with pytest.raises(RuntimeError, match="aggregation contract failed"):
        probe.run(args)
    report = json.loads((args.out / "report.json").read_text())
    assert report["status"] == "failed" and report["planned_cases"] == 178
    assert report["rows"] == [] and report["aggregate"] is None
    assert report["end_verification"] == "pass" and "aggregate_error" in report


def test_cli_has_no_case_filter_or_limit():
    with pytest.raises(SystemExit) as result:
        probe.main(["--help"])
    assert result.value.code == 0
