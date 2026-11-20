"""Small CPU-only regressions for the independent private result auditor."""
import json
from pathlib import Path

import pytest


from experiments.grounded_signal import audit_grounded_results as m


def row(*, targets=(), allowed=(), no_hint=(), context=(), hints=(), unknown=(), candidates=None,
        complete=True, seconds=1.0):
    return {"target_norm_ids": set(targets), "allowed_hint_norm_ids": set(allowed),
            "no_hint_norm_ids": set(no_hint), "insufficient_context_norm_ids": set(context),
            "hints": set(hints), "unknown": set(unknown), "complete": complete,
            "candidates": set(candidates if candidates is not None else set(targets) | set(no_hint) | set(context)),
            "raw_status": "complete" if complete else "timeout", "package_status": "ready", "seconds": seconds,
            "split": "evaluation"}


def test_target_case_and_pair_counts_are_different():
    result = m.summarize([row(targets=("a", "b"), hints=("a", "b")), row(targets=("c",)), row(no_hint=("d",))])
    counts = result["evaluation_counts"]
    assert counts["known_target_norms_emitted"] == 2
    assert counts["known_target_norms"] == 3
    assert counts["positive_cases_with_known_target_hint"] == 1
    assert counts["positive_cases"] == 2
    assert counts["output_precision"] is None


def test_context_case_is_not_a_clear_negative():
    result = m.summarize([row(no_hint=("a",)), row(context=("b",), unknown=("b",))])
    assert result["evaluation_counts"]["negative_cases"] == 1
    assert result["clear_negatives"]["planned_cases"] == 1
    assert result["annotated_context_cases"]["planned_cases"] == 1
    assert result["evaluation_counts"]["confirmed_no_hint_negative_cases"] == 1
    assert result["evaluation_counts"]["context_annotated_norms_answered_insufficient"] == 1


def test_transport_failure_is_unknown_and_never_correct_negative():
    result = m.summarize([row(no_hint=("a",), unknown=("a",), complete=False)])
    assert result["clear_negatives"]["technical_cases"] == 1
    counts = result["evaluation_counts"]
    assert counts["confirmed_no_hint_negative_cases"] == 0
    assert counts["negative_cases_with_hint"] == 0
    assert counts["negative_case_output_upper_bound"] == 1


def test_unknown_candidate_never_rescues_target_missing_from_retrieval():
    result = m.summarize([row(targets=("missing",), candidates=("other",), unknown=("other",), complete=False)])
    counts = result["evaluation_counts"]
    assert counts["retrieval_missing_known_targets"] == 1
    assert counts["known_target_norms_unknown"] == 0
    assert counts["known_target_case_hit_upper_bound"] == 0


def test_allowed_context_and_unjudged_extras_are_not_false_by_default():
    result = m.summarize([row(targets=("a",), allowed=("b",), no_hint=("c",), context=("d",), hints=("a", "b", "c", "d", "e"))])
    assert result["evaluation_counts"]["diagnostic_negative_norms_emitted"] == 1
    assert result["evaluation_counts"]["unadjudicated_hint_norms"] == 1
    assert result["allowed_extra_hint_norm_pairs"] == 1
    assert result["context_annotated_norms_emitted_as_hint"] == 1
    assert result["otherwise_unjudged_hint_norm_pairs"] == 1


def test_legacy_matrix_can_be_recounted_with_its_original_denominators():
    rows = [row(context=("a",), hints=("a",))]
    archived = m.summarize(rows, "grounded_signal_evaluation/1")["evaluation_counts"]
    current = m.summarize(rows, "grounded_signal_evaluation/2")["evaluation_counts"]
    assert archived["negative_cases"] == archived["unadjudicated_hint_norms"] == 1
    assert current["negative_cases"] == current["unadjudicated_hint_norms"] == 0


def test_medians_preserve_failed_attempts_without_hiding_complete_median():
    result = m.summarize([row(seconds=2), row(seconds=4), row(seconds=300, complete=False)])
    assert result["standalone_timing"] == {"attempts": 3, "total_seconds": 306.0,
                                            "median_seconds": 4.0, "completed_median_seconds": 3.0}


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        m.parse(raw)


@pytest.mark.parametrize("a,b", [(False, 0), (True, 1), (1, 1.0)])
def test_canonical_equality_is_type_strict(a, b):
    with pytest.raises(ValueError):
        m.equal(a, b, "type")


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_duration_guard(value):
    with pytest.raises(ValueError):
        m.number(value)


def terminal():
    report = {"status": "complete", "results": {arm: [{"status": "complete"}] * 16 for arm in m.ARMS}}
    protocol = {"arms": list(m.ARMS), "model_interface": {"format": "fixed_norm_slots/2", "tool_choice": "required"}}
    return report, protocol


@pytest.mark.parametrize("marker", ["abandonment.json", "interruption.json"])
def test_abandoned_r1_cannot_be_relabelled_complete(tmp_path, marker):
    (tmp_path / marker).write_text("{}")
    with pytest.raises(ValueError, match="Abandoned"):
        m.terminal_guard(tmp_path, *terminal())


def test_full_terminal_r2_only(tmp_path):
    report, protocol = terminal()
    m.terminal_guard(tmp_path, report, protocol)
    report["results"][m.ARMS[0]].pop()
    with pytest.raises(ValueError, match="denominator"):
        m.terminal_guard(tmp_path, report, protocol)


@pytest.mark.parametrize("mutation", ["running", "old_schema", "partial_arms"])
def test_nonterminal_or_r1_adapter_rejected(tmp_path, mutation):
    report, protocol = terminal()
    if mutation == "running":
        report["status"] = "running"
    elif mutation == "old_schema":
        protocol["model_interface"]["format"] = "array/1"
    else:
        protocol["arms"].pop()
    with pytest.raises(ValueError):
        m.terminal_guard(tmp_path, report, protocol)


def event(request, response=None, error=False):
    result = {"kind": "http", "path": "/v1/chat/completions", "request": request, "request_sha256": m.digest(request)}
    if response is not None:
        raw = json.dumps(response)
        result.update(raw_response=raw, response_sha256=m.rawsha(raw.encode()))
    if error:
        result["error"] = "TimeoutError"
    return result


def test_physical_failed_http_is_attempt_not_accepted_call():
    call = {"status": "timeout", "usage": {"model_calls": 0, "http_requests": 1}, "trace": [event({}, error=True)]}
    counts, _ = m.http_accounting(call)
    assert counts["completion_requests"] == 1
    assert counts["completion_transport_errors"] == 1
    assert counts["completion_responses"] == counts["client_model_calls"] == 0


def test_rejected_completion_tokens_still_count_physically():
    call = {"status": "invalid_answer", "usage": {"model_calls": 1, "http_requests": 1},
            "trace": [event({}, {"usage": {"prompt_tokens": 15, "completion_tokens": 7}})]}
    counts, _ = m.http_accounting(call)
    assert counts["response_prompt_tokens"] == 15
    assert counts["response_completion_tokens"] == 7
    assert counts["client_model_calls"] == 1


def test_rawresponse_hash_and_request_types_bound():
    call = {"status": "timeout", "usage": {"model_calls": 0, "http_requests": 1},
            "trace": [event({"cache_prompt": False}, error=True)]}
    call["trace"][0]["request"]["cache_prompt"] = 0
    with pytest.raises(ValueError, match="Request hash"):
        m.http_accounting(call)


def test_response_without_accounting_is_explicitly_unavailable():
    call = {"status": "invalid_completion", "usage": {"model_calls": 0, "http_requests": 1}, "trace": [event({}, {"error": "bad"})]}
    counts, _ = m.http_accounting(call)
    assert counts["responses_without_usage"] == 1
    assert counts["response_prompt_tokens"] == 0


def test_invalid_usage_is_not_silently_counted_as_measured_zero():
    call = {"status": "invalid_completion", "usage": {"model_calls": 1, "http_requests": 1},
            "trace": [event({}, {"usage": {"prompt_tokens": True, "completion_tokens": 5}})]}
    counts, _ = m.http_accounting(call)
    assert counts["responses_with_invalid_usage"] == 1
    assert counts["client_model_calls"] == 1
    assert counts["completed_stage_calls"] == 0


def test_invalid_json_transport_response_preserves_failed_denominator():
    e = event({}, error=True)
    e.update(raw_response="not JSON", response_sha256=m.rawsha(b"not JSON"))
    call = {"status": "transport_error", "usage": {"model_calls": 0, "http_requests": 1}, "trace": [e]}
    counts, _ = m.http_accounting(call)
    assert counts["completion_responses"] == counts["response_payload_unavailable"] == 1
    assert counts["client_model_calls"] == 0


def test_original_span_binding_and_dynamic_fact_ids():
    spans = [{"id": "s1", "start": 0, "end": 5, "text": "Text."}]
    facts = [{"id": "f1", "span_id": "s1", "kind": "event", "polarity": "denied", "statement": "A model interpretation."}]
    out = m.expanded_facts(facts, spans, "Text.")
    assert out[0]["start"] == 0 and out[0]["text"] == "Text."
    spans[0]["start"] = False
    with pytest.raises(ValueError, match="Fact span"):
        m.expanded_facts(facts, spans, "Text.")


def test_bound_file_end_check_and_path_escape(tmp_path):
    file = tmp_path / "x.json"
    file.write_text("{}")
    b = m.BoundFiles()
    b.read(file)
    file.write_text('{"x":1}')
    with pytest.raises(ValueError):
        b.finish()
    with pytest.raises(ValueError, match="escapes"):
        m.local(tmp_path, "../escape")


def test_full_canonical_protocol_does_not_drop_unrecognized_fields():
    original = {"arms": ["a"], "extra": {"future_field": 1}}
    changed = {"arms": ["a"], "extra": {"future_field": 2}}
    assert m.digest(original) != m.digest(changed)


def test_markdown_does_not_emit_case_payload_or_ids():
    summary = m.summarize([row(targets=("private-norm",), hints=("private-norm",))])
    report = {"arms": {"direct_prepared": {"all": summary, "evaluation": summary}},
              "physical_execution": {"fact_calls": {"completion_requests": 1}, "decision_calls": {"direct_prepared": {"completion_requests": 1}},
                                     "total_recorded_wall_seconds": 3, "facts_seconds_once": 1, "decision_seconds": 2},
              "bindings": {"protocol_sha256": "a" * 64}, "artifact_receipts": {"results": "b" * 64}, "source_sha256": "c" * 64}
    assert "private-norm" not in m.markdown(report)


def test_no_experiment_or_model_imports():
    import ast
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(n.startswith(("experiments", "risk_layer", "torch", "transformers")) for n in imports)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.canonical(value))


def fake_call(case, config, *, package=None, fact=None, direct=False):
    spans = [{"id": "s1", "start": 0, "end": 5, "text": "Text."}]
    rawfact = {"id": "f1", "span_id": "s1", "kind": "event", "polarity": "affirmed", "statement": "Synthetic fact."}
    expanded = {"id": "f1", "start": 0, "end": 5, "text": "Text.", "kind": "event", "polarity": "affirmed", "statement": "Synthetic fact."}
    payload = {"original_case": case["text"], "case_spans": spans}
    arguments = {"unresolved": []}
    answer = {"facts": [expanded], "unresolved": []}
    name = "extract_facts" if package is None else "analyse_case" if direct else "decide_hints"
    if package is None or direct:
        arguments["facts"] = [rawfact]
    if package is not None:
        payload["original_sources"] = m.shown_sources(package)
        if not direct:
            payload.update(attributed_facts=fact["answer"]["facts"], unresolved_case_questions=[])
        arguments["hints"] = {norm: {"status": "no_hint", "relation": "none", "evidence_unit_ids": [],
                                     "supported_fact_ids": [], "open_conditions": []} for norm in package["candidate_norm_ids"]}
        answer["hints"] = [{"norm_id": norm, **hint} for norm, hint in arguments["hints"].items()]
    request = {"model": "synthetic", "messages": [{"role": "system", "content": "Synthetic fixture"},
                                                  {"role": "user", "content": m.canonical(payload).decode()}],
               "tools": [{"type": "function", "function": {"name": name, "parameters": {}}}],
               "tool_choice": "required", "cache_prompt": False, "parallel_tool_calls": False, "stream": False,
               "chat_template_kwargs": {"enable_thinking": True}, "max_tokens": config["max_output_tokens"],
               **{k: config[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed", "reasoning_budget_tokens")}}
    message = {"role": "assistant", "tool_calls": [{"type": "function", "id": "synthetic-call", "function": {"name": name, "arguments": m.canonical(arguments).decode()}}]}
    response = {"model": "synthetic", "choices": [{"finish_reason": "tool_calls", "message": message}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}}}
    trace = []
    for path, req, res in [("/apply-template", request, {"prompt": "P"}),
                           ("/tokenize", {"content": "P", "parse_special": True, "add_special": False}, {"tokens": [3] * 10}),
                           ("/v1/chat/completions", request, response)]:
        e = event(req, res)
        e["path"] = path
        trace.append(e)
    return {"status": "complete", "answer": answer, "arguments": arguments, "message": message,
            "trace": trace, "usage": {"model_calls": 1, "http_requests": 3, "prompt_tokens": 10, "completion_tokens": 5},
            "timing": {"elapsed_seconds": 1.0}, "errors": []}


def all_no_hint_counts(cases):
    positive = sum(bool(c["diagnostic_labels"]["target_norm_ids"]) for c in cases)
    targets = sum(len(c["diagnostic_labels"]["target_norm_ids"]) for c in cases)
    context = sum(bool(c["diagnostic_labels"]["insufficient_context_norm_ids"]) for c in cases)
    return {"planned_cases": len(cases), "positive_cases": positive, "negative_cases": len(cases) - positive,
            "execution": {"complete": len(cases)}, "source_package_status": {"ready": len(cases)},
            "positive_cases_with_any_hint": 0, "positive_cases_with_known_target_hint": 0, "known_target_case_recall": 0.0,
            "known_target_case_hit_upper_bound": 0, "known_target_norms": targets, "known_target_norms_emitted": 0,
            "known_target_norms_unknown": 0, "known_target_norm_recall": 0.0, "known_target_norm_hit_upper_bound": 0,
            "negative_cases_with_hint": 0, "negative_hint_norm_pairs": 0, "negative_cases_technically_unavailable": 0,
            "negative_cases_with_uncertain_norms": 0, "negative_case_output_upper_bound": 0,
            "negative_norm_output_upper_bound": 0, "confirmed_no_hint_negative_cases": len(cases) - positive - context,
            "context_annotated_cases": context, "context_annotated_norms": context * 2,
            "context_annotated_norms_answered_insufficient": 0, "diagnostic_negative_norms_emitted": 0,
            "unadjudicated_hint_norms": 0, "output_precision": None, "semantic_truth_verified": False,
            "retrieval_missing_known_targets": 0}


@pytest.fixture
def full_matrix(tmp_path):
    inputs, run = tmp_path / "inputs", tmp_path / "run"
    cases, packages, table, families = [], {}, {}, {}
    target_lengths = [1, 0, 1, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    for i, target_length in enumerate(target_lengths):
        family = f"family-{i // 2}"
        norms = [f"norm:{i // 2}:{k}" for k in range(3)]
        context = norms[:2] if i == 5 else []
        targets = norms[:target_length]
        case = {"case_id": f"case-{i}", "family_id": family, "split": "development" if i < 8 else "evaluation",
                "text": "Text.", "prepared_norm_ids": norms,
                "diagnostic_labels": {"target_norm_ids": targets, "allowed_hint_norm_ids": [],
                                      "no_hint_norm_ids": sorted(set(norms) - set(targets) - set(context)),
                                      "insufficient_context_norm_ids": context, "hint_expected": bool(targets),
                                      "annotation_status": "assistant_reviewed_synthetic"}}
        cases.append(case)
        if family in families:
            continue
        units = [{"unit_id": "unit:" + norm, "norm_id": norm, "start": 0, "end": 5, "text": "Rule.",
                  "norm_text_sha256": m.rawsha(b"Rule.")} for norm in norms]
        package = {"status": "ready", "candidate_norm_ids": norms, "norm_ids": norms, "complete_local_norms": True,
                   "unit_ids": [u["unit_id"] for u in units], "units": units, "reference_closure_complete": True,
                   "references": [], "limitations": [], "required_groups": [
                       {"norm_id": u["norm_id"], "unit_ids": [u["unit_id"]], "policy": "all_stored_parent_norm_text"} for u in units]}
        fingerprint = m.digest(package)
        package.update(fingerprint=fingerprint, package_id="pkg:" + fingerprint)
        families[family] = package["package_id"]
        packages[package["package_id"]] = package
        table.update({u["unit_id"]: u for u in units})
    protocol = {"arms": list(m.ARMS), "model_interface": {"format": "fixed_norm_slots/2", "tool_choice": "required"},
                "case_ids": [c["case_id"] for c in cases], "family_packages": families,
                "case_packages": {arm: {c["case_id"]: families[c["family_id"]] for c in cases} for arm in m.ARMS},
                "source_hashes": {"frozen.py": m.rawsha(b"# synthetic")}}
    for name, obj in {"protocol.json": protocol, "cases.json": cases, "packages.json": packages, "source-units.json": table}.items():
        write_json(inputs / name, obj)
    manifest = {"format": "grounded_signal_inputs/1", "artifacts": {p.name: m.rawsha(p.read_bytes()) for p in inputs.iterdir()}}
    write_json(inputs / "input-manifest.json", manifest)
    model = {"model": "synthetic"}
    write_json(run / "model-manifest.json", model)
    (run / "source").mkdir()
    (run / "source/frozen.py").write_bytes(b"# synthetic")
    config = {"max_output_tokens": 50, "context_tokens": 100, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
              "min_p": 0.0, "presence_penalty": 1.5, "seed": 0, "reasoning_budget_tokens": 20}
    protocol = {**protocol, "configuration": config, "input_manifest_sha256": m.rawsha((inputs / "input-manifest.json").read_bytes()),
                "model_binding": {"model": model, "model_manifest_sha256": m.rawsha((run / "model-manifest.json").read_bytes()), "launch": {"alias": "synthetic"}}}
    write_json(run / "protocol.json", protocol)
    results, facts = {arm: [] for arm in m.ARMS}, {}
    for index, case in enumerate(cases, 1):
        fact = fake_call(case, config)
        facts[case["case_id"]] = fact
        write_json(run / f"facts-{index:03}.json", fact)
        package = packages[families[case["family_id"]]]
        for arm in m.ARMS:
            direct = arm == "direct_prepared"
            call = fake_call(case, config, package=package, fact=fact, direct=direct)
            row_result = {"case_id": case["case_id"], "status": "complete", "source_package_id": package["package_id"],
                          "candidate_norm_ids": package["candidate_norm_ids"], "answer": call["answer"], "usage": call["usage"],
                          "fact_stage_status": "complete", "timing": {"decision_seconds": 1.0, "fact_seconds": 0.0 if direct else 1.0,
                                                                       "elapsed_seconds": 1.0 if direct else 2.0}}
            results[arm].append(row_result)
            write_json(run / f"case-{index:03}-{arm}.json", {"row": row_result, "call": call})
    write_json(run / "results.json", {"status": "complete", "results": results, "fact_results": facts,
                                      "source_end_hashes": protocol["source_hashes"], "elapsed_seconds": 65.0})
    stored = {"format": "grounded_signal_evaluation/1",
              "protocol_sha256": m.digest(protocol), "planned_cases_sha256": m.digest(cases), "results_sha256": m.digest(results),
              "source_packages_sha256": m.digest(packages), "source_units_sha256": m.digest(table), "planned_cases": 16,
              "planned_case_arm_rows": 48, "arms": {arm: {**{split: all_no_hint_counts([c for c in cases if split == "all" or c["split"] == split])
                                                              for split in ("all", "development", "evaluation")},
                                                         "timing": {"recorded_attempts": 16, "total_seconds": 16.0 if arm == "direct_prepared" else 32.0}}
                                                      for arm in m.ARMS}}
    write_json(run / "evaluation.json", {"aggregate": stored})
    return run, inputs


def test_full_independent_fake_matrix_counts_shared_facts_once(full_matrix):
    report = m.audit(*full_matrix)
    physical = report["physical_execution"]
    assert physical["facts_seconds_once"] == 16.0
    assert physical["decision_seconds"] == 48.0
    assert physical["total_recorded_wall_seconds"] == 65.0
    assert physical["fact_calls"]["completion_requests"] == 16
    assert sum(x["completion_requests"] for x in physical["decision_calls"].values()) == 48
    assert sum(x["all"]["standalone_timing"]["total_seconds"] for x in report["arms"].values()) == 80.0
    assert report["denominators"]["evaluation"]["target_norm_pairs"] == 6


@pytest.mark.parametrize("tamper", ["counter", "protocol", "time", "row", "source"])
def test_full_matrix_rejects_tamper(full_matrix, tamper):
    run, inputs = full_matrix
    if tamper in {"counter", "protocol"}:
        path = run / "evaluation.json"
        data = m.parse(path.read_bytes())
        if tamper == "counter":
            data["aggregate"]["arms"][m.ARMS[0]]["evaluation"]["known_target_norms_emitted"] = 1
        else:
            data["aggregate"]["protocol_sha256"] = "0" * 64
        write_json(path, data)
    elif tamper == "time":
        path = run / "results.json"
        data = m.parse(path.read_bytes())
        data["elapsed_seconds"] = 50.0
        write_json(path, data)
    elif tamper == "row":
        path = run / "case-001-direct_prepared.json"
        data = m.parse(path.read_bytes())
        data["row"]["status"] = "timeout"
        write_json(path, data)
    else:
        (run / "source/frozen.py").write_bytes(b"# changed")
    with pytest.raises(ValueError):
        m.audit(run, inputs)

