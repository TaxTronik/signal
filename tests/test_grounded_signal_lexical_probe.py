from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.grounded_signal import lexical_probe as probe


def config():
    return probe.ResearchConfig(max_output_tokens=3072, timeout_seconds=300, reuse_prompt_cache=True)


def event(path, request, response=None, error=None):
    result = {"kind": "http", "path": path, "request": request,
              "request_sha256": probe.value_sha(request), "elapsed_seconds": .1}
    if response is not None:
        raw = probe._json(response)
        result.update(raw_response=raw, response_sha256=hashlib.sha256(raw.encode()).hexdigest())
    if error:
        result["error"] = error
    return result


def facts(text="Ein Vorgang.", alias="same-model"):
    request = probe.expected_fact_request(text, alias, config())
    args = {"facts": [{"id": "f1", "span_id": "s1", "kind": "event", "polarity": "affirmed",
                       "statement": "Ein genannter Vorgang."}], "unresolved": []}
    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function",
                "function": {"name": "extract_facts", "arguments": probe._json(args)}}]}
    usage = {"prompt_tokens": 3, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 0}}
    trace = [event("/apply-template", request, {"prompt": "native prompt"}),
             event("/tokenize", {"content": "native prompt", "add_special": False, "parse_special": True}, {"tokens": [1, 2, 3]}),
             event("/v1/chat/completions", deepcopy(request), {"model": alias, "usage": usage,
                   "choices": [{"finish_reason": "tool_calls", "message": message}]})]
    return {"status": "complete", "trace": trace, "arguments": args, "message": message,
            "answer": probe.runtime.expand_facts(args, probe.runtime.case_spans(text)), "errors": [],
            "timing": {"elapsed_seconds": 2.0}, "usage": {**usage, "model_calls": 1, "http_requests": 3}}


def audit(fact, text="Ein Vorgang."):
    return probe.audit_fact_result(fact, text=text, alias="same-model", config=config())


def rewrite_response(fact, index, change):
    target = fact["trace"][index]
    response = json.loads(target["raw_response"])
    change(response)
    raw = probe._json(response)
    target.update(raw_response=raw, response_sha256=hashlib.sha256(raw.encode()).hexdigest())


def planned():
    cases = [{"case_id": f"c{i}", "family_id": f"family{i // 2}",
              "split": "development" if i < 8 else "evaluation", "text": f"Ein Vorgang {i}.",
              "prepared_norm_ids": ["n1"], "diagnostic_labels": {"target_norm_ids": ["n1"]}}
             for i in range(16)]
    learned = [{"case_id": c["case_id"], "split": c["split"], "unadapted": [
        {"norm_id": "n1", "score": .9}, {"norm_id": "n2", "score": .8}, {"norm_id": "n3", "score": .7}]}
        for c in cases]
    protocol = {"case_ids": [c["case_id"] for c in cases], "arms": list(probe.runner.ARMS), "top_k": 3,
                "source_hashes": {"source.py": "digest"}, "case_packages": {}, "arm_policies": {}}
    return cases, learned, protocol


def test_selection_is_all_evaluation_and_uses_unadapted_ranks_not_labels():
    cases, learned, protocol = planned()
    chosen, mapping = probe.choose_cases(cases, learned, protocol)
    assert len(chosen) == 8 and [c["case_id"] for c in chosen] == [f"c{i}" for i in range(8, 16)]
    cases[8]["diagnostic_labels"] = {"target_norm_ids": ["other"]}
    assert probe.choose_cases(cases, learned, protocol)[1] == mapping
    assert all(ids == ["n1", "n2", "n3"] for ids in mapping.values())


@pytest.mark.parametrize("modify", [
    lambda c, ranks, p: c.pop(),
    lambda c, ranks, p: p.update(top_k=4),
    lambda c, ranks, p: ranks[8]["unadapted"].pop(),
    lambda c, ranks, p: ranks[8]["unadapted"].reverse(),
    lambda c, ranks, p: ranks[8]["unadapted"][0].update(score=float("nan")),
    lambda c, ranks, p: ranks[8]["unadapted"][0].update(norm_id="n2"),
    lambda c, ranks, p: c[8].update(family_id="family0"),
])
def test_no_subset_rank_or_split_changes(modify):
    c, ranks, p = planned()
    modify(c, ranks, p)
    with pytest.raises(probe.ProbeError):
        probe.choose_cases(c, ranks, p)


def test_accepted_facts_reconstruct_exactly_without_model_calls():
    result = audit(facts())
    assert result["available"] and result["source_free_requests_checked"] == 3
    assert result["elapsed_seconds"] == 2.0


def test_expected_fact_request_matches_actual_frozen_runtime_extract():
    captured = []
    class CaptureAgent:
        model = "same-model"
        config = globals()["config"]()
        def _completion(self, request, deadline, result):
            captured.append(deepcopy(request))
            return facts()["message"]
    result = probe.runtime.extract(CaptureAgent(), "Ein Vorgang.")
    assert result["status"] == "complete"
    assert captured == [probe.expected_fact_request("Ein Vorgang.", "same-model", config())]
    assert captured[0]["tool_choice"] == "required"


@pytest.mark.parametrize("modify", [
    lambda f: f["answer"]["facts"][0].update(statement="Changed"),
    lambda f: f["answer"]["facts"][0].update(start=1),
    lambda f: f["trace"][0].update(request_sha256="0" * 64),
    lambda f: f["trace"][0].update(response_sha256="0" * 64),
    lambda f: f["trace"].append(deepcopy(f["trace"][-1])),
    lambda f: f["trace"].pop(),
    lambda f: f["timing"].update(elapsed_seconds=float("nan")),
])
def test_malformed_inherited_facts_rejected(modify):
    f = facts()
    modify(f)
    with pytest.raises(probe.ProbeError):
        audit(f)


def test_fact_request_cannot_contain_sources_or_foreign_case():
    for key in ("sources", "original_case"):
        f = facts()
        request = f["trace"][0]["request"]
        payload = json.loads(request["messages"][1]["content"])
        payload[key] = "Other content"
        request["messages"][1]["content"] = probe._json(payload)
        f["trace"][0]["request_sha256"] = probe.value_sha(request)
        with pytest.raises(probe.ProbeError, match="source-free"):
            audit(f)
    with pytest.raises(probe.ProbeError, match="source-free"):
        audit(facts(), text="Ein anderer Vorgang.")


@pytest.mark.parametrize("index,key,value", [(0, "cache_prompt", 0), (0, "seed", False), (1, "add_special", 0)])
def test_request_identity_is_type_exact(index, key, value):
    f = facts()
    f["trace"][index]["request"][key] = value
    f["trace"][index]["request_sha256"] = probe.value_sha(f["trace"][index]["request"])
    with pytest.raises(probe.ProbeError):
        audit(f)


def test_inherited_expansion_and_usage_are_type_exact():
    f = facts()
    f["answer"]["facts"][0]["start"] = False
    with pytest.raises(probe.ProbeError, match="expansion"):
        audit(f)
    f = facts()
    f["usage"]["model_calls"] = True
    with pytest.raises(probe.ProbeError, match="usage"):
        audit(f)


@pytest.mark.parametrize("modify", [
    lambda r: r.update(model="wrong-model"),
    lambda r: r.update(truncated=True),
    lambda r: r["usage"].update(prompt_tokens=4),
    lambda r: r["usage"].update(completion_tokens=3073),
    lambda r: r["usage"]["prompt_tokens_details"].update(cached_tokens=1),
    lambda r: r["choices"][0].update(finish_reason="length"),
    lambda r: r["choices"][0]["message"].update(role="user"),
])
def test_native_completion_budget_and_identity_guards(modify):
    f = facts()
    rewrite_response(f, 2, modify)
    with pytest.raises(probe.ProbeError):
        audit(f)


def test_real_transport_failure_shape_is_unavailable_not_a_rejection():
    request = probe.expected_fact_request("Ein Vorgang.", "same-model", config())
    f = {"status": "transport_error", "trace": [event("/apply-template", request, error="Timeout")],
         "errors": ["Timeout"], "timing": {"elapsed_seconds": 300.0}, "usage": {"model_calls": 0}}
    result = audit(f)
    assert result["available"] is False and result["status"] == "transport_error"
    f["answer"] = facts()["answer"]
    with pytest.raises(probe.ProbeError, match="cannot be accepted"):
        audit(f)


def model_provenance(digest="a" * 64):
    return {"model": {"format": "quality_research_model/1", "status": "complete", "repo": "example/model",
                      "revision": "b" * 40, "filename": "weights.gguf", "quantization": "Q6_K",
                      "model": {"path": "private-path", "bytes": 200, "sha256": digest}}}


def test_same_alias_is_not_enough_to_reuse_facts_from_other_weights():
    a, b = model_provenance(), model_provenance()
    probe.same_model(a, b)
    b["model"]["model"]["path"] = "moved-identical-file"
    probe.same_model(a, b)
    with pytest.raises(probe.ProbeError, match="different GGUF"):
        probe.same_model(a, model_provenance("c" * 64))


def base_objects():
    cases, _, primary = planned()
    base = deepcopy(primary) | {"input_manifest_sha256": "receipt"}
    result = {"status": "incomplete", "source_end_hashes": primary["source_hashes"],
              "results": {arm: [{"case_id": c["case_id"], "status": "complete"} for c in cases] for arm in probe.runner.ARMS},
              "fact_results": {c["case_id"]: {"status": "complete"} for c in cases}}
    return result, base, primary


def test_terminal_incomplete_full_base_is_accepted(monkeypatch):
    result, base, primary = base_objects()
    monkeypatch.setattr(probe.runner, "source_hashes", lambda: primary["source_hashes"])
    probe.validate_base(result, base, primary, primary_manifest_sha="receipt")


@pytest.mark.parametrize("modify", [
    lambda r, b, p: r.update(status="running"),
    lambda r, b, p: r.update(status="abandoned"),
    lambda r, b, p: r["results"]["direct_prepared"].pop(),
    lambda r, b, p: r["results"]["direct_prepared"][0].update(status="running"),
    lambda r, b, p: r["fact_results"].pop("c0"),
    lambda r, b, p: b.update(input_manifest_sha256="changed"),
])
def test_abandoned_partial_or_unbound_base_is_rejected(modify, monkeypatch):
    result, base, primary = base_objects()
    monkeypatch.setattr(probe.runner, "source_hashes", lambda: primary["source_hashes"])
    modify(result, base, primary)
    with pytest.raises(probe.ProbeError):
        probe.validate_base(result, base, primary, primary_manifest_sha="receipt")


def test_inventory_rejects_changes_and_escape(tmp_path):
    (tmp_path / "bound.txt").write_text("original")
    known = probe.inventory(tmp_path)
    probe.check_inventory(tmp_path, known)
    (tmp_path / "bound.txt").write_text("changed")
    with pytest.raises(probe.ProbeError):
        probe.check_inventory(tmp_path, known)
    with pytest.raises(probe.ProbeError, match="path"):
        probe.check_inventory(tmp_path, {"../outside": "digest"})


def test_run_refuses_running_base_before_any_model_binding(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    probe.save(base / "results.json", {"status": "running"})
    monkeypatch.setattr(probe, "bind_model", lambda *a: pytest.fail("Model binding must not start"))
    args = SimpleNamespace(prepared=tmp_path / "none", base_run=base, out=tmp_path / "out")
    with pytest.raises(probe.ProbeError, match="terminal"):
        probe.run(args)


def execution_fixture():
    from experiments.grounded_signal.sources import package_fingerprint
    text = "Eine erfundene Testregel."
    unit = {"unit_id": "u1", "norm_id": "n1", "start": 0, "end": len(text), "text": text,
            "norm_text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    package = {"status": "ready", "complete_local_norms": True, "candidate_norm_ids": ["n1"],
               "norm_ids": ["n1"], "unit_ids": ["u1"], "units": [unit],
               "required_groups": [{"norm_id": "n1", "unit_ids": ["u1"], "policy": "all_stored_parent_norm_text"}]}
    package["fingerprint"] = package_fingerprint(package)
    package["package_id"] = "pkg:" + package["fingerprint"]
    cases = [{"case_id": f"e{i}", "family_id": f"f{i//2}", "split": "evaluation",
              "text": f"Ein Testvorgang {i}.", "prepared_norm_ids": ["n1"],
              "diagnostic_labels": {"annotation_status": "assistant_reviewed_synthetic", "hint_expected": i % 2 == 0,
                                    "target_norm_ids": ["n1"] if i % 2 == 0 else [],
                                    "allowed_hint_norm_ids": [], "no_hint_norm_ids": [] if i % 2 == 0 else ["n1"]}}
             for i in range(8)]
    inherited = {case["case_id"]: facts(case["text"]) for case in cases}
    protocol = {"arms": [probe.ARM], "arm_policies": {probe.ARM: {"source_mode": "automatic"}},
                "case_packages": {probe.ARM: {case["case_id"]: package["package_id"] for case in cases}}}
    return cases, {package["package_id"]: package}, {"u1": unit}, inherited, protocol


def fake_decision(agent, text, package, inherited, direct=False):
    assert direct is False
    assert "diagnostic_labels" not in probe._json([package, inherited])
    return {"status": "complete", "answer": {"facts": deepcopy(inherited["answer"]["facts"]),
            "hints": [{"norm_id": "n1", "status": "hint", "evidence_unit_ids": ["u1"],
                       "supported_fact_ids": ["f1"], "open_conditions": [], "relation": "definition"}], "unresolved": []},
            "errors": [], "trace": [{"kind": "fake_completion_for_test"}],
            "timing": {"elapsed_seconds": .4}, "usage": {"model_calls": 1}}


def test_execution_reuses_facts_retains_failed_negative_and_records_standalone_time(tmp_path, monkeypatch):
    cases, packages, units, inherited, protocol = execution_fixture()
    inherited["e1"].update(status="invalid_facts", errors=["Invalid fact schema"])
    inherited["e1"].pop("answer")
    before = deepcopy(inherited)
    calls = []
    def decide(*args, **kwargs):
        calls.append(deepcopy(args[3]))
        return fake_decision(*args, **kwargs)
    monkeypatch.setattr(probe.runtime, "decide", decide)
    monkeypatch.setattr(probe.runtime, "extract", lambda *a, **k: pytest.fail("No new fact extraction allowed"))
    result = probe.execute_cases(cases, packages, units, inherited, protocol, object(), tmp_path)
    assert inherited == before and len(calls) == 7
    assert result["status"] == "incomplete" and result["new_fact_calls"] == 0
    metrics = result["evaluation"]["aggregate"]["arms"][probe.ARM]["all"]
    assert metrics["planned_cases"] == 8 and metrics["negative_cases_technically_unavailable"] == 1
    assert metrics["confirmed_no_hint_negative_cases"] == 0
    assert result["results"][probe.ARM][0]["timing"]["elapsed_seconds"] == 2.4
    assert result["results"][probe.ARM][0]["timing"]["decision_seconds"] == .4
    assert len((tmp_path / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 8
    assert len(list(tmp_path.glob("case-*.json"))) == 8


def test_unexpected_error_persists_failed_row_and_all_pending_denominators(tmp_path, monkeypatch):
    cases, packages, units, inherited, protocol = execution_fixture()
    count = 0
    def decide(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("Unexpected decoder failure")
        return fake_decision(*args, **kwargs)
    monkeypatch.setattr(probe.runtime, "decide", decide)
    with pytest.raises(RuntimeError, match="decoder"):
        probe.execute_cases(cases, packages, units, inherited, protocol, object(), tmp_path)
    report = probe.read(tmp_path / "report.json")
    assert report["status"] == "failed"
    metrics = report["evaluation"]["aggregate"]["arms"][probe.ARM]["all"]
    assert metrics["planned_cases"] == 8
    assert metrics["execution"] == {"complete": 1, "technical_error": 1, "not_run": 6}
    assert len(probe.read(tmp_path / "case-002.json")["call"]["errors"]) >= 1


def complete_fake_run(tmp_path, monkeypatch):
    cases, packages, units, inherited, execution_protocol = execution_fixture()
    primary = tmp_path / "primary"
    prepared = tmp_path / "prepared"
    base = tmp_path / "base"
    primary.mkdir()
    prepared.mkdir()
    base.mkdir()
    (prepared / "primary").mkdir()
    code = {"frozen.py": "fixed-source"}
    monkeypatch.setattr(probe.runner, "source_hashes", lambda: code)
    ids = [f"d{i}" for i in range(8)] + [case["case_id"] for case in cases]
    primary_protocol = {"case_ids": ids, "arms": list(probe.runner.ARMS), "source_hashes": code,
                        "case_packages": {arm: {cid: "primary-package" for cid in ids} for arm in probe.runner.ARMS},
                        "arm_policies": {arm: {"source_mode": "family_prepared"} for arm in probe.runner.ARMS}}
    probe.save(primary / "protocol.json", primary_protocol)
    probe.save(primary / "input-manifest.json", {"format": "grounded_signal_inputs/1", "artifacts": probe.inventory(primary)})
    primary_sha = probe.sha(primary / "input-manifest.json")
    probe.save(prepared / "primary/protocol.json", primary_protocol)
    prepared_protocol = {**execution_protocol, "primary_inputs": str(primary),
                         "primary_input_manifest_sha256": primary_sha, "primary_source_hashes": code,
                         "probe_source_sha256": probe.sha(probe.__file__)}
    for name, value in (("cases.json", cases), ("packages.json", packages), ("source-units.json", units),
                        ("protocol.json", prepared_protocol)):
        probe.save(prepared / name, value)
    probe.save(prepared / "prepared-manifest.json", {"format": probe.PREPARE_FORMAT, "artifacts": probe.inventory(prepared)})
    provenance = model_provenance() | {"launch": {"alias": "same-model"}}
    probe.save(base / "model-manifest.json", provenance["model"])
    provenance["model_manifest_sha256"] = probe.sha(base / "model-manifest.json")
    base_protocol = {**primary_protocol, "input_manifest_sha256": primary_sha,
                     "configuration": probe.asdict(config()), "model_binding": provenance}
    base_facts = {**{f"d{i}": {"status": "complete"} for i in range(8)}, **inherited}
    base_results = {"status": "incomplete", "source_end_hashes": code, "fact_results": base_facts,
                    "results": {arm: [{"case_id": cid, "status": "complete"} for cid in ids] for arm in probe.runner.ARMS}}
    probe.save(base / "protocol.json", base_protocol)
    probe.save(base / "results.json", base_results)
    for number, case in enumerate(cases, 9):
        probe.save(base / f"facts-{number:03}.json", inherited[case["case_id"]])
    monkeypatch.setattr(probe, "bind_model", lambda *args: deepcopy(provenance))
    monkeypatch.setattr(probe, "snapshot_server", lambda *args: {"fake_metadata_only": True})
    monkeypatch.setattr(probe, "ResearchAgent", lambda *args: object())
    monkeypatch.setattr(probe.runtime, "decide", fake_decision)
    return SimpleNamespace(prepared=prepared, base_run=base, out=tmp_path / "out", url="http://127.0.0.1:9999",
                           model="same-model", model_manifest=base / "model-manifest.json", launch_manifest=base / "unused.json")


def test_full_fake_run_binds_exact_saved_protocol_and_all_artifacts(tmp_path, monkeypatch):
    args = complete_fake_run(tmp_path, monkeypatch)
    assert probe.run(args) == 0
    report = probe.read(args.out / "report.json")
    protocol = probe.read(args.out / "protocol.json")
    assert report["end_verification"] == "pass"
    assert report["evaluation"]["aggregate"]["protocol_sha256"] == probe.value_sha(protocol)
    assert report["planned_cases"] == 8 and report["new_fact_calls"] == 0
    probe.check_inventory(args.out, report["artifact_hashes"], exclude=("report.json",))


def test_full_fake_run_still_verifies_bindings_after_unexpected_failure(tmp_path, monkeypatch):
    args = complete_fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(probe.runtime, "decide", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Unexpected failure")))
    with pytest.raises(RuntimeError, match="Unexpected failure"):
        probe.run(args)
    report = probe.read(args.out / "report.json")
    assert report["status"] == "failed" and report["end_verification"] == "pass"
    assert report["evaluation"]["aggregate"]["arms"][probe.ARM]["all"]["planned_cases"] == 8


def test_full_fake_run_detects_original_archive_change(tmp_path, monkeypatch):
    args = complete_fake_run(tmp_path, monkeypatch)
    def mutate(*args_inner, **kwargs):
        probe.save(args.base_run / "results.json", {"status": "tampered"})
        return fake_decision(*args_inner, **kwargs)
    monkeypatch.setattr(probe.runtime, "decide", mutate)
    with pytest.raises(probe.ProbeError, match="changed"):
        probe.run(args)
    report = probe.read(args.out / "report.json")
    assert report["status"] == "failed" and report["end_verification"] == "failed"
