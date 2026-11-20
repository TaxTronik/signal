from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from experiments.grounded_signal import runner
from experiments.grounded_signal.sources import package_fingerprint


def harness(tmp_path, monkeypatch, *, arms=None, corrupt_answer=False):
    unit = {"unit_id": "u-stable", "norm_id": "norm:T:1", "start": 0, "end": 6, "text": "Regel."}
    unit["norm_text_sha256"] = hashlib.sha256(b"Regel.").hexdigest()
    package = {"status": "ready", "complete_local_norms": True, "candidate_norm_ids": ["norm:T:1"],
               "norm_ids": ["norm:T:1"], "unit_ids": ["u-stable"], "units": [unit],
               "required_groups": [{"norm_id": "norm:T:1", "unit_ids": ["u-stable"],
                                    "policy": "all_stored_parent_norm_text"}]}
    package["fingerprint"] = package_fingerprint(package)
    package["package_id"] = "pkg:" + package["fingerprint"]
    pid = package["package_id"]
    case = {"case_id": "case1", "family_id": "family1", "split": "evaluation", "text": "Ein Vorgang.",
            "prepared_norm_ids": ["norm:T:1"], "diagnostic_labels": {
                "annotation_status": "assistant_reviewed_synthetic", "hint_expected": True,
                "target_norm_ids": ["norm:T:1"], "allowed_hint_norm_ids": [], "no_hint_norm_ids": []}}
    protocol = {"arms": list(runner.ARMS), "case_ids": ["case1"], "family_packages": {"family1": pid},
                "arm_policies": {arm: {"source_mode": "automatic" if arm.endswith("retrieved") else "family_prepared"}
                                 for arm in runner.ARMS},
                "case_packages": {arm: {"case1": pid} for arm in runner.ARMS}}
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    runner.save(inputs / "input-manifest.json", {"artifacts": {}})
    args = SimpleNamespace(inputs=inputs, out=tmp_path / "run", arms=arms, case_id=None,
                           url="http://127.0.0.1:8089", model="test")
    monkeypatch.setattr(runner, "load_inputs", lambda directory: ([case], {pid: package}, {"u-stable": unit}, protocol))
    monkeypatch.setattr(runner, "artifact_hashes", lambda directory: {})
    monkeypatch.setattr(runner, "freeze_code", lambda out: {"source": "fixed"})
    monkeypatch.setattr(runner, "source_hashes", lambda: {"source": "fixed"})
    monkeypatch.setattr(runner, "bind_model", lambda *a: {})
    monkeypatch.setattr(runner, "snapshot_server", lambda *a: None)
    fact = {"id": "f1", "start": 0, "end": 12, "text": "Ein Vorgang.", "statement": "Ein Vorgang wird genannt.",
            "kind": "event", "polarity": "affirmed"}
    fact_result = {"status": "complete", "answer": {"facts": [fact], "unresolved": []},
                   "timing": {"elapsed_seconds": 2}, "usage": {"model_calls": 1}}
    facts_calls, decision_calls = [], []
    def extract(agent, text):
        facts_calls.append(text)
        return deepcopy(fact_result)
    def decide(agent, text, package, facts=None, *, direct=False):
        decision_calls.append({"text": text, "package": deepcopy(package), "facts": deepcopy(facts), "direct": direct})
        answer = {"facts": [deepcopy(fact)], "hints": [{"norm_id": "norm:T:1", "status": "hint",
                  "evidence_unit_ids": ["u-stable"], "supported_fact_ids": ["f1"],
                  "open_conditions": [], "relation": "procedure"}], "unresolved": []}
        if corrupt_answer:
            answer["facts"][0]["text"] = "Falsch."
        return {"status": "complete", "answer": answer, "errors": [], "trace": [{"actual": "request-and-answer"}],
                "timing": {"elapsed_seconds": 3}, "usage": {"model_calls": 1}}
    monkeypatch.setattr(runner.runtime, "extract", extract)
    monkeypatch.setattr(runner.runtime, "decide", decide)
    return args, facts_calls, decision_calls


def test_three_arm_run_reuses_facts_without_hiding_their_standalone_cost(tmp_path, monkeypatch):
    args, fact_calls, decision_calls = harness(tmp_path, monkeypatch)
    assert runner.run(args) == 0
    results = runner.read(args.out / "results.json")
    assert len(fact_calls) == 1 and len(decision_calls) == 3
    assert decision_calls[1]["facts"] == decision_calls[2]["facts"]
    assert decision_calls[0]["direct"] and not decision_calls[1]["direct"]
    assert results["results"]["direct_prepared"][0]["timing"]["elapsed_seconds"] == 3
    assert results["results"]["structured_prepared"][0]["timing"]["elapsed_seconds"] == 5
    assert results["status"] == "complete"
    assert "diagnostic_labels" not in decision_calls[0]["package"]


def test_selected_arm_is_explicit_in_protocol_and_does_not_extract_unused_facts(tmp_path, monkeypatch):
    args, facts, decisions = harness(tmp_path, monkeypatch, arms="direct_prepared")
    assert runner.run(args) == 0
    protocol = runner.read(args.out / "protocol.json")
    assert protocol["arms"] == ["direct_prepared"] and set(protocol["arm_policies"]) == {"direct_prepared"}
    assert facts == [] and len(decisions) == 1


def test_failed_independent_validation_preserves_actual_call_and_counts_failure(tmp_path, monkeypatch):
    args, _, _ = harness(tmp_path, monkeypatch, arms="direct_prepared", corrupt_answer=True)
    assert runner.run(args) == 1
    call = runner.read(args.out / "case-001-direct_prepared.json")["call"]
    assert call["trace"] == [{"actual": "request-and-answer"}]
    assert call["timing"]["elapsed_seconds"] == 3 and "rejected_answer" in call
    row = runner.read(args.out / "results.json")["results"]["direct_prepared"][0]
    assert row["answer"] is None and row["status"] == "invalid_input_or_answer"


def test_even_coordinated_input_manifest_changes_cannot_pass_end_binding(tmp_path, monkeypatch):
    args, _, _ = harness(tmp_path, monkeypatch, arms="direct_prepared")
    original = runner.runtime.decide
    def change_manifest(*a, **kw):
        result = original(*a, **kw)
        runner.save(args.inputs / "input-manifest.json", {"artifacts": {}, "changed": True})
        return result
    monkeypatch.setattr(runner.runtime, "decide", change_manifest)
    with pytest.raises(ValueError, match="changes during model run"):
        runner.run(args)
    assert runner.read(args.out / "results.json")["status"] == "running"


def test_loading_rejects_added_or_changed_prepared_files(tmp_path):
    runner.save(tmp_path / "input-manifest.json", {"artifacts": {}})
    (tmp_path / "unplanned.txt").write_text("added", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts changed"):
        runner.load_inputs(tmp_path)
