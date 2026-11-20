"""Pure-JSON audit tests: failure denominators, source ledgers and safe output."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

from tools import quality_research_report as audit


IDENTITY = {"source_fingerprint": "source-fingerprint", "graph_fingerprint": "graph-fingerprint",
            "index_fingerprint": "index-fingerprint", "citation_fingerprint": "citation-fingerprint"}
CONFIG = {"context_tokens": 16384, "initial_sources": 6, "seed": 0}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def page(query, norm="norm:TEST:1", text="Private original norm sentence."):
    value = {"norm_id": norm, "text": text, "source_start": 0, "source_end": len(text), "source_length": len(text),
             "text_sha256": audit.sha(text), "source_fingerprint": IDENTITY["source_fingerprint"],
             "graph_fingerprint": IDENTITY["graph_fingerprint"]}
    value["evidence_id"] = "ev:" + audit.sha(audit.canonical([IDENTITY, audit.sha(query), norm,
                                                             value["text_sha256"], 0, len(text)]))
    return value


def suggestion(query, source):
    return {"start": 0, "end": len(query), "satz": query, "normen": [{"id": source["norm_id"], "evidenz": {
        "fingerprint": source["source_fingerprint"], "anker": [], "passagen": [{"id": f"{source['norm_id']}@0:{len(source['text'])}",
        "norm_id": source["norm_id"], "quell_start": 0, "quell_end": len(source["text"]), "text": source["text"]}]}, "research": {
        "evidence_id": source["evidence_id"], "legal_validity_verified": False,
        "source_quote": {"start": 0, "end": len(query), "text": query},
        "norm_quote": {"start": 0, "end": len(source["text"]), "text": source["text"]}}}]}


def result(query, arm, *, status="complete", source=None, initial=None):
    source = source or page(query)
    initial = [page(query)] if initial is None else initial
    raw = audit.canonical({"model": "test-model", "choices": [{"finish_reason": "tool_calls"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}})
    request = {"model": "test-model", "messages": [{"role": "user", "content": query}], "cache_prompt": False}
    return {"status": status, "suggestions": [suggestion(query, source)] if status == "complete" else [],
            "allow_research": arm == "research", "config": CONFIG, "model": "test-model",
            "initial_sources": initial, "initial_sources_sha256": audit.sha(audit.canonical(initial)),
            "focuses": [{"id": "f0", "start": 0, "end": len(query), "text": query}],
            "unresolved": [], "errors": [], "usage": {"http_requests": 1, "model_calls": 1,
                "prompt_tokens": 10, "completion_tokens": 4}, "timing": {},
            "trace": [{"kind": "http", "path": "/v1/chat/completions", "request": request,
                       "request_sha256": audit.sha(audit.canonical(request)), "raw_response": raw,
                       "response_sha256": audit.sha(raw), "elapsed_seconds": 1.},
                      {"kind": "tool", "name": "finish", "result": {"accepted": status == "complete",
                       "suggestions": [suggestion(query, source)] if status == "complete" else []}, "elapsed_seconds": .1}]}


def row(name="private-case", *, negative=False):
    query = "Private original query for " + name
    return {"case_id": name, "query": query, "negative": negative, "dataset": "main",
            "targets": [] if negative else ["norm:TEST:1", "norm:TEST:2"], "baseline": {"suggestions": []}, "arms": {}}


def test_new_returned_pages_are_distinct_from_citation_metadata():
    value = row()
    new = page(value["query"], "norm:TEST:2", "Private second source text.")
    r = result(value["query"], "research", source=new)
    r["trace"].insert(1, {"kind": "tool", "name": "search_sources", "result": {"results": [new, new]}, "elapsed_seconds": .2})
    r["trace"].insert(2, {"kind": "tool", "name": "follow_citations", "result": {
        "citations": [{"norm_id": "norm:TEST:2"}, {"norm_id": "norm:TEST:3"}]}, "elapsed_seconds": .2})
    value["arms"]["research"] = {"result": r}
    assert audit.audit_sources(value, "research", IDENTITY) == {
        "initial_pages": 1, "initial_norms": 1, "new_pages": 1, "new_norms": 1, "citation_only_norms": 1,
        "selected_entries_using_new_pages": 1, "known_targets_newly_available": 1}


@pytest.mark.parametrize("tamper", ["unseen_page", "query_quote", "norm_quote", "identity", "initial_hash", "static_tool", "legal_proof", "output_passage", "boolean_focus"])
def test_source_presence_cannot_be_inferred_or_fabricated(tamper):
    value = row()
    r = result(value["query"], "static")
    norm = r["suggestions"][0]["normen"][0]
    if tamper == "unseen_page":
        norm["research"]["evidence_id"] = "ev:unseen"
    elif tamper == "query_quote":
        norm["research"]["source_quote"]["text"] = "invented"
    elif tamper == "norm_quote":
        norm["research"]["norm_quote"]["start"] = True
    elif tamper == "identity":
        r["initial_sources"][0]["source_fingerprint"] = "another corpus"
        r["initial_sources_sha256"] = audit.sha(audit.canonical(r["initial_sources"]))
    elif tamper == "initial_hash":
        r["initial_sources_sha256"] = "0" * 64
    elif tamper == "legal_proof":
        norm["research"]["legal_validity_verified"] = True
    elif tamper == "output_passage":
        norm["evidenz"]["passagen"][0]["text"] = "invented source"
    elif tamper == "boolean_focus":
        r["suggestions"][0]["start"] = False
    else:
        r["trace"].append({"kind": "tool", "name": "search_sources", "result": {"results": []}})
    value["arms"]["static"] = {"result": r}
    # Except for trace-consistency tests, keep final trace/output consistent so
    # the independently checked source obligation is the failing boundary.
    r["trace"][1]["result"]["suggestions"] = deepcopy(r["suggestions"])
    with pytest.raises(ValueError):
        audit.audit_sources(value, "static", IDENTITY)


def test_recovered_budget_and_finish_errors_remain_visible_after_success():
    r = result("Private query", "research")
    r["errors"] = [{"kind": "tool_error", "recoverable": True}, {"kind": "tool_error", "recoverable": True}]
    r["trace"] += [{"kind": "tool", "name": "read_source", "result": {"error": "ResearchBudgetExceeded"}, "elapsed_seconds": .1},
                   {"kind": "tool", "name": "finish", "result": {"error": "ValueError"}, "elapsed_seconds": .1}]
    value = audit.trace_counts(r)
    assert value["tool_result_errors"] == {"ResearchBudgetExceeded": 1, "ValueError": 1}
    assert value["repair_and_failure_counts"] == {"recoverable_errors": 2, "finish_validation_errors": 1}
    assert value["errors"] == {"tool_error": 2}


def test_context_budget_markers_have_counts_without_invented_durations():
    r = result("Private query", "research")
    marker = {"kind": "budget", "reason": "context_finish_reserve", "round": 2, "prompt_tokens_before_finish": 12000}
    r["trace"].insert(1, marker)
    r["errors"].append({"kind": "missing_tool_call", "recoverable": True})
    assert audit.trace_counts(r)["budget_events"] == {"context_finish_reserve": 1}
    assert audit.trace_counts(r)["errors"] == {"missing_tool_call": 1}
    assert "elapsed_seconds" not in marker
    assert audit.trace_counts(r)["tool_calls"] == {"finish": 1}
    # Actual HTTP/tool requests continue to require measured durations.
    r["trace"][0].pop("elapsed_seconds")
    with pytest.raises(ValueError, match="measured trace time"):
        audit.trace_counts(r)


@pytest.mark.parametrize("field,value", [("round", True), ("prompt_tokens_before_finish", -1),
                                         ("reason", "private unknown text"), ("elapsed_seconds", float("nan"))])
def test_malformed_context_budget_event_is_rejected(field, value):
    r = result("Private query", "research")
    marker = {"kind": "budget", "reason": "context_finish_reserve", "round": 2, "prompt_tokens_before_finish": 12000}
    marker[field] = value
    r["trace"].append(marker)
    with pytest.raises(ValueError):
        audit.trace_counts(r)


def quote_selection_row(query=None):
    """Two identical quotes in a nonzero-offset page must retain occurrence 2."""
    value = row()
    if query is not None:
        value["query"] = query
    source = page(value["query"], text="a" * 440)
    source.update(source_start=100, source_end=540, source_length=600)
    source["evidence_id"] = "ev:" + audit.sha(audit.canonical([IDENTITY, audit.sha(value["query"]),
        source["norm_id"], source["text_sha256"], 100, 540]))
    r = result(value["query"], "static", source=source, initial=[source])
    norm = r["suggestions"][0]["normen"][0]
    norm["evidenz"]["passagen"][0].update(id=f"{source['norm_id']}@100:540", quell_start=100, quell_end=540)
    norm["research"].update(reason="Selected supplied evidence", norm_quote={"start": 320, "end": 540, "text": "a" * 220})
    r["evidence_aliases"] = {"e1": source["evidence_id"]}
    r["quote_references"] = {f"e1q{i+1}": {"evidence_id": source["evidence_id"], "norm_id": source["norm_id"],
        "text": "a" * 220, "page_start": i * 220, "page_end": (i + 1) * 220, "source_start": 100 + i * 220}
        for i in range(2)}
    finding = {"focus_id": "f0", "norm_id": source["norm_id"], "quote_id": "e1q2", "source_quote": value["query"],
               "reason": "Selected supplied evidence"}
    expanded = {key: item for key, item in finding.items() if key != "quote_id"}
    expanded.update(evidence_id=source["evidence_id"], norm_quote="a" * 220, norm_quote_start=320)
    r["trace"][-1].update(arguments={"findings": [finding], "unresolved": []},
                           validated_arguments={"findings": [expanded], "unresolved": []})
    r["trace"][-1]["result"]["suggestions"] = deepcopy(r["suggestions"])
    value["arms"]["static"] = {"result": r}
    return value


def test_quote_id_host_expansion_preserves_the_selected_duplicate_occurrence():
    value = quote_selection_row()
    assert audit.audit_sources(value, "static", IDENTITY)["initial_pages"] == 1


@pytest.mark.parametrize("tamper", ["alias", "missing_segment", "segment_text", "segment_offset", "boolean_offset",
                                   "other_quote", "expanded_quote", "expanded_offset", "output_occurrence",
                                   "extra_finding", "unresolved", "missing_expansion"])
def test_quote_ledger_expansion_and_final_positions_are_independently_bound(tamper):
    value = quote_selection_row()
    r = value["arms"]["static"]["result"]
    finish = r["trace"][-1]
    quote = r["quote_references"]["e1q2"]
    if tamper == "alias":
        r["evidence_aliases"]["e1"] = "ev:unseen"
    elif tamper == "missing_segment":
        r["quote_references"].pop("e1q1")
    elif tamper == "segment_text":
        quote["text"] = "b" * 220
    elif tamper == "segment_offset":
        quote["source_start"] = 100
    elif tamper == "boolean_offset":
        r["quote_references"]["e1q1"]["page_start"] = False
    elif tamper == "other_quote":
        finish["arguments"]["findings"][0]["quote_id"] = "e1q1"
    elif tamper == "expanded_quote":
        finish["validated_arguments"]["findings"][0]["norm_quote"] = "b" * 220
    elif tamper == "expanded_offset":
        finish["validated_arguments"]["findings"][0]["norm_quote_start"] = 100
    elif tamper == "output_occurrence":
        # Still an exact original quote: only the selected occurrence is wrong.
        r["suggestions"][0]["normen"][0]["research"]["norm_quote"].update(start=100, end=320)
        finish["result"]["suggestions"] = deepcopy(r["suggestions"])
    elif tamper == "extra_finding":
        finish["arguments"]["findings"].append(deepcopy(finish["arguments"]["findings"][0]))
    elif tamper == "unresolved":
        r["unresolved"].append("Unexpected extra conclusion")
    else:
        finish.pop("validated_arguments")
    with pytest.raises(ValueError):
        audit.audit_sources(value, "static", IDENTITY)


def test_empty_quote_id_finish_remains_valid_and_failed_ledger_is_still_checked():
    value = quote_selection_row()
    r = value["arms"]["static"]["result"]
    r["suggestions"] = []
    r["trace"][-1].update(arguments={"findings": [], "unresolved": []},
                           validated_arguments={"findings": [], "unresolved": []},
                           result={"accepted": True, "suggestions": []})
    audit.audit_sources(value, "static", IDENTITY)
    r["status"] = "invalid_finish"
    r["trace"][-1]["result"] = {"error": "ValueError"}
    r["quote_references"]["e1q2"]["source_start"] = 100
    with pytest.raises(ValueError, match="page interval"):
        audit.audit_sources(value, "static", IDENTITY)


def case_quote_selection_row(*, duplicate=False):
    value = quote_selection_row("x" * 440)
    r = value["arms"]["static"]["result"]
    r["case_quote_references"] = {f"f0q{i+1}": {"focus_id": "f0", "text": "x" * 220,
        "start": i * 220, "end": (i + 1) * 220} for i in range(2)}
    finish = r["trace"][-1]
    finding = finish["arguments"]["findings"][0]
    finding.pop("source_quote")
    finding["source_quote_id"] = "f0q2"
    finish["validated_arguments"]["findings"][0].update(source_quote="x" * 220, source_quote_start=220)
    r["suggestions"][0]["normen"][0]["research"]["source_quote"] = {"start": 220, "end": 440, "text": "x" * 220}
    finish["result"]["suggestions"] = deepcopy(r["suggestions"])
    finish["merged_duplicate_findings"] = []
    if duplicate:
        extra = {**finding, "quote_id": "e1q1", "source_quote_id": "f0q1", "reason": "Additional same-norm finding"}
        finish["arguments"]["findings"].append(extra)
        finish["merged_duplicate_findings"].append(deepcopy(extra))
        r["errors"].append({"kind": "duplicate_norm_merged", "recoverable": True, "count": 1})
    return value


def test_case_quote_ids_preserve_occurrence_and_first_norm_evidence_after_deduplication():
    value = case_quote_selection_row(duplicate=True)
    r = value["arms"]["static"]["result"]
    assert audit.audit_sources(value, "static", IDENTITY)["initial_pages"] == 1
    assert audit.trace_counts(r)["repair_and_failure_counts"] == {"recoverable_errors": 1, "merged_duplicate_findings": 1}
    assert len(r["suggestions"][0]["normen"]) == 1
    assert r["suggestions"][0]["normen"][0]["research"]["source_quote"]["start"] == 220


@pytest.mark.parametrize("tamper", ["quote_position", "quote_partition", "expanded_position", "output_occurrence",
                                   "duplicate_unknown_source", "duplicate_unknown_norm_quote", "duplicate_removed_record",
                                   "duplicate_count", "boolean_offset"])
def test_case_quote_ledger_and_dropped_duplicates_cannot_hide_invalid_references(tamper):
    value = case_quote_selection_row(duplicate=True)
    r = value["arms"]["static"]["result"]
    finish = r["trace"][-1]
    if tamper == "quote_position":
        r["case_quote_references"]["f0q2"].update(start=0, end=220)
    elif tamper == "quote_partition":
        r["case_quote_references"].pop("f0q1")
    elif tamper == "expanded_position":
        finish["validated_arguments"]["findings"][0]["source_quote_start"] = 0
    elif tamper == "output_occurrence":
        r["suggestions"][0]["normen"][0]["research"]["source_quote"].update(start=0, end=220)
        finish["result"]["suggestions"] = deepcopy(r["suggestions"])
    elif tamper in ("duplicate_unknown_source", "duplicate_unknown_norm_quote"):
        key = "source_quote_id" if tamper == "duplicate_unknown_source" else "quote_id"
        finish["arguments"]["findings"][1][key] = "unseen"
        finish["merged_duplicate_findings"][0][key] = "unseen"
    elif tamper == "duplicate_removed_record":
        finish["merged_duplicate_findings"] = []
    elif tamper == "duplicate_count":
        r["errors"][0]["count"] = 2
    else:
        r["case_quote_references"]["f0q1"]["start"] = False
    with pytest.raises(ValueError):
        audit.audit_sources(value, "static", IDENTITY)
        audit.trace_counts(r)


@pytest.mark.parametrize("tamper", ["request", "response", "http_count"])
def test_trace_hash_and_request_counts_are_checked(tamper):
    r = result("Private query", "static")
    if tamper == "request":
        r["trace"][0]["request"]["model"] = "changed"
    elif tamper == "response":
        r["trace"][0]["raw_response"] += " "
    else:
        r["usage"]["http_requests"] += 1
    with pytest.raises(ValueError):
        audit.trace_counts(r)


def cache_dialog(*, reuse=False, legacy=False):
    r = result("Private query", "research")
    r["config"] = dict(CONFIG) if legacy else {**CONFIG, "reuse_prompt_cache": reuse, "finish_reserve_seconds": 0.}
    second = deepcopy(r["trace"][0])
    second["request"]["cache_prompt"] = reuse
    second["request_sha256"] = audit.sha(audit.canonical(second["request"]))
    raw = json.loads(second["raw_response"])
    raw["usage"]["prompt_tokens_details"]["cached_tokens"] = 4 if reuse else 0
    second["raw_response"] = audit.canonical(raw)
    second["response_sha256"] = audit.sha(second["raw_response"])
    r["trace"].insert(1, second)
    r["usage"].update(http_requests=2, model_calls=2, prompt_tokens=20, completion_tokens=8)
    return r


@pytest.mark.parametrize("reuse,legacy", [(False, True), (False, False), (True, False)])
def test_prompt_cache_policy_is_first_request_off_then_explicit_setting(reuse, legacy):
    r = cache_dialog(reuse=reuse, legacy=legacy)
    assert audit.trace_counts(r)["completion_finish_reasons"] == {"tool_calls": 2}
    # Starting another arm/run resets the count even when the same config is used.
    other = result("Private query", "static")
    other["config"] = deepcopy(r["config"])
    assert audit.trace_counts(other)["completion_finish_reasons"] == {"tool_calls": 1}
    if reuse:
        assert audit.trace_counts(r)["prompt_cache"] == {"cold_completions": 1, "reuse_requested_completions": 1,
            "accounted_completions": 2, "cached_prompt_tokens": 4}


@pytest.mark.parametrize("reuse,position,cache", [(True, 0, True), (True, 1, False), (False, 1, True),
                                                 (False, 0, 0), (True, 1, "true")])
def test_request_hash_cannot_authorize_wrong_or_nonboolean_cache_policy(reuse, position, cache):
    r = cache_dialog(reuse=reuse)
    event = r["trace"][position]
    event["request"]["cache_prompt"] = cache
    event["request_sha256"] = audit.sha(audit.canonical(event["request"]))
    with pytest.raises(ValueError, match="prompt-cache policy"):
        audit.trace_counts(r)


@pytest.mark.parametrize("policy", [{"reuse_prompt_cache": True}, {"finish_reserve_seconds": 0.},
    {"reuse_prompt_cache": 1, "finish_reserve_seconds": 0.}, {"reuse_prompt_cache": False, "finish_reserve_seconds": True},
    {"reuse_prompt_cache": False, "finish_reserve_seconds": -1.}, {"reuse_prompt_cache": False, "finish_reserve_seconds": float("nan")}])
def test_execution_policy_fields_are_optional_as_a_pair_and_strict_when_present(policy):
    with pytest.raises(ValueError, match="execution policy"):
        audit.execution_policy({**CONFIG, **policy})
    assert audit.execution_policy(CONFIG) == (False, 0.)


@pytest.mark.parametrize("cached", [None, True, -1, 11, "4"])
def test_cached_token_accounting_is_strict_without_dropping_technical_failures(cached):
    r = cache_dialog(reuse=True)
    event = r["trace"][1]
    raw = json.loads(event["raw_response"])
    if cached is None:
        raw["usage"].pop("prompt_tokens_details")
    else:
        raw["usage"]["prompt_tokens_details"]["cached_tokens"] = cached
    event["raw_response"] = audit.canonical(raw)
    event["response_sha256"] = audit.sha(event["raw_response"])
    with pytest.raises(ValueError, match="Prompt-cache accounting"):
        audit.trace_counts(r)
    r["status"] = "invalid_completion"
    r["suggestions"] = []
    r["trace"].pop()  # No accepted finish follows the invalid second completion.
    counts = audit.trace_counts(r)["prompt_cache"]
    assert counts["invalid_accounting_completions"] == 1
    assert counts["accounted_completions"] == 1


def test_cold_first_request_cannot_claim_cached_tokens_even_with_reuse_enabled():
    r = cache_dialog(reuse=True)
    raw = json.loads(r["trace"][0]["raw_response"])
    raw["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
    r["trace"][0]["raw_response"] = audit.canonical(raw)
    r["trace"][0]["response_sha256"] = audit.sha(r["trace"][0]["raw_response"])
    r["status"] = "invalid_completion"
    # An earlier invalid response cannot be hidden by a later failed attempt.
    with pytest.raises(ValueError, match="Prompt-cache accounting"):
        audit.trace_counts(r)


@pytest.mark.parametrize("remaining", [30., 50., 70.])
def test_time_finish_reserve_is_a_typed_unmeasured_decision_with_fixed_boundary(remaining):
    r = result("Private query", "research")
    r["config"] = {**CONFIG, "reuse_prompt_cache": False, "finish_reserve_seconds": 35.}
    marker = {"kind": "budget", "reason": "time_finish_reserve", "round": 1,
              "remaining_seconds": remaining, "finish_reserve_seconds": 35.}
    r["trace"].insert(0, marker)
    request = r["trace"][1]["request"]
    request.update(tools=[{"type": "function", "function": {"name": "finish"}}],
                   tool_choice={"type": "function", "function": {"name": "finish"}})
    r["trace"][1]["request_sha256"] = audit.sha(audit.canonical(request))
    assert audit.trace_counts(r)["budget_events"] == {"time_finish_reserve": 1}
    assert "elapsed_seconds" not in marker


@pytest.mark.parametrize("field,value", [("remaining_seconds", 29.999), ("remaining_seconds", 70.001),
    ("remaining_seconds", True), ("finish_reserve_seconds", 0.), ("finish_reserve_seconds", 36.), ("round", False)])
def test_time_finish_reserve_marker_cannot_change_declared_policy_or_legacy_guard(field, value):
    r = result("Private query", "research")
    r["config"] = {**CONFIG, "reuse_prompt_cache": False, "finish_reserve_seconds": 35.}
    marker = {"kind": "budget", "reason": "time_finish_reserve", "round": 1,
              "remaining_seconds": 50., "finish_reserve_seconds": 35.}
    marker[field] = value
    r["trace"].insert(0, marker)
    with pytest.raises(ValueError):
        audit.trace_counts(r)


def test_time_reserve_marker_must_actually_force_finish_and_may_fail_before_generation():
    r = result("Private query", "research")
    r["config"] = {**CONFIG, "reuse_prompt_cache": False, "finish_reserve_seconds": 35.}
    marker = {"kind": "budget", "reason": "time_finish_reserve", "round": 1,
              "remaining_seconds": 50., "finish_reserve_seconds": 35.}
    r["trace"].insert(0, marker)
    with pytest.raises(ValueError, match="finish-only"):
        audit.trace_counts(r)
    r["trace"] = [marker]
    r["usage"]["http_requests"] = 0
    with pytest.raises(ValueError, match="no completion request"):
        audit.trace_counts(r)
    r["status"] = "context_limit"
    assert audit.trace_counts(r)["budget_events"] == {"time_finish_reserve": 1}


def test_new_execution_settings_are_bound_in_protocol_and_kept_public(frozen_experiment):
    directory, report, events = frozen_experiment
    config = {**CONFIG, "reuse_prompt_cache": True, "finish_reserve_seconds": 35.}
    report["agent_config"] = config
    for row in report["rows"]:
        for entry in row["arms"].values():
            entry["result"]["config"] = config
    protocol = json.loads((directory / "protocol.json").read_bytes())
    protocol["agent_config"] = config
    write(directory / "protocol.json", protocol)
    report["protocol_sha256"] = audit.digest(directory / "protocol.json")
    write(directory / "report.json", report)
    (directory / "rows.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    value = audit.aggregate(directory / "report.json")
    assert value["configuration"]["reuse_prompt_cache"] is True
    assert value["configuration"]["finish_reserve_seconds"] == 35.


def test_invalid_response_shape_is_a_recorded_failed_attempt_not_a_reporter_crash():
    r = result("Private query", "static", status="transport_error")
    r["trace"][0]["raw_response"] = "[]"
    r["trace"][0]["response_sha256"] = audit.sha("[]")
    assert audit.trace_counts(r)["completion_finish_reasons"] == {}


def test_metrics_preserve_failed_and_pending_negative_denominators():
    values = [row("positive"), row("failed-negative", negative=True), row("pending-negative", negative=True)]
    values[0]["arms"]["research"] = {"result": result(values[0]["query"], "research")}
    values[1]["arms"]["research"] = {"result": result(values[1]["query"], "research", status="timeout")}
    counts = audit.metric_counts(values, "research", False)
    assert counts["cases"] == 3 and counts["negative_cases"] == 2
    assert counts["failed_negative_cases"] == counts["pending_negative_cases"] == 1
    assert counts["rates"]["negative_output_lower_bound"] == 0
    assert counts["rates"]["negative_output_upper_bound"] == 1
    assert audit.metric_counts(values, "research", True)["rates"] is None


def test_failed_origin_does_not_become_a_completed_research_gain():
    value = row()
    value["arms"]["static"] = {"result": result(value["query"], "static", status="timeout")}
    value["arms"]["research"] = {"result": result(value["query"], "research")}
    change = audit.transition([value], "static", "research")
    assert change["all_selected_output_changes"]["gained_hit_cases"] == 1
    assert change["both_complete_changes"]["gained_hit_cases"] == 0
    assert change["both_complete_changes"]["cases"] == 0
    assert change["unavailable"] == {"origin_cases": 1, "destination_cases": 0}


def test_positive_unlisted_norms_are_output_load_not_false_positive_labels():
    value = row()
    value["arms"]["static"] = {"result": result(value["query"], "static", source=page(value["query"], "norm:TEST:99"))}
    load = audit.output_load([value], "static")
    assert load["output_entries"] == load["unadjudicated_positive_case_norm_pairs"] == 1
    assert load["negative_entries"] == 0


def test_quantiles_include_failed_attempt_times_without_fabricated_empty_values():
    assert audit.quantiles([1., 2., 20.])["p95"] == pytest.approx(18.2)
    assert audit.quantiles([])["median"] is None
    with pytest.raises(ValueError):
        audit.quantiles([float("nan")])


@pytest.fixture
def frozen_experiment(tmp_path):
    input_hashes = {}
    for relative in ("inputs/graph.sqlite", "inputs/index/meta.json", "inputs/index/vectors.npy"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        input_hashes[str(path)] = audit.digest(path)
    source_hashes = {}
    for relative in ("tools/quality_research_agent.py", "risk_layer/research_agent.py", "risk_layer/research_sources.py"):
        path = tmp_path / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic frozen implementation")
        source_hashes[relative] = audit.digest(path)
    source_rows = []
    for dataset, positives, negatives in (("main", 102, 30), ("additional", 15, 15)):
        for i in range(positives + negatives):
            r = row(f"private-{dataset}-{i}", negative=i >= positives)
            r.update(dataset=dataset, group="private-group", category="test", uncertain=False)
            r.pop("arms")
            r["treffer"] = r.pop("baseline")
            source_rows.append(r)
    reference = {"format": "quality_recall_live/1", "status": "mismatch", "public_profile": "treffer",
                 "rows": source_rows, "source_audit": {"errors": []}, "input_hashes": input_hashes}
    write(tmp_path / "reference.json", reference)
    selected = [source_rows[i] for i in (0, 102, 132, 147)]
    rows = []
    for source in selected:
        r = {**{key: source[key] for key in audit.CASE_FIELDS}, "baseline": source["treffer"], "arms": {}}
        for arm in audit.ARMS:
            r["arms"][arm] = {"result": result(r["query"], arm), "wall_seconds": 2.}
        rows.append(r)
    model = {"repo": "test/model", "revision": "a" * 40, "filename": "model.gguf", "quantization": "Q6_K"}
    write(tmp_path / "model-manifest.json", model)
    model_sha = audit.digest(tmp_path / "model-manifest.json")
    launch = {"alias": "test-model", "model_manifest_sha256": model_sha}
    write(tmp_path / "launch-manifest.json", launch)
    report = {"format": "quality_research_agent/1", "status": "complete", "execution_complete": True,
              "technical_failures": 0, "diagnostic_subset": True, "rows": rows, "arms": list(audit.ARMS),
              "agent_config": CONFIG, "corpus": IDENTITY, "source_hashes": source_hashes, "input_hashes": input_hashes,
              "model_provenance": {"model": model, "launch": launch, "model_manifest_sha256": model_sha,
                                   "launch_manifest_sha256": audit.digest(tmp_path / "launch-manifest.json")},
              "server": {}, "reference_sha256": audit.digest(tmp_path / "reference.json"),
              "snapshot_hashes": {p.relative_to(tmp_path).as_posix(): audit.digest(p) for p in tmp_path.rglob("*") if p.is_file()},
              "timing": {"corpus_load_seconds": 1.}}
    protocol = {key: deepcopy(report[key]) for key in ("arms", "input_hashes", "source_hashes", "diagnostic_subset",
        "reference_sha256", "agent_config", "model_provenance", "server", "snapshot_hashes")}
    protocol.update(format="quality_research_agent_protocol/1", status="frozen_before_runs", output_cap=5,
                    focus="original_sentence", case_order=[r["case_id"] for r in rows], model_alias="test-model")
    write(tmp_path / "protocol.json", protocol)
    report["protocol_sha256"] = audit.digest(tmp_path / "protocol.json")
    report["summaries"] = {dataset: {arm: audit.metric_counts(
        [r for r in rows if dataset == "all" or r["dataset"] == dataset], arm, True)
        for arm in ("baseline", *audit.ARMS)} for dataset in ("all", *audit.DATASETS)}
    write(tmp_path / "report.json", report)
    events = [{"case_id": r["case_id"], "arm": arm, **r["arms"][arm]} for r in rows for arm in audit.ARMS]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return tmp_path, report, events


def test_aggregate_and_markdown_remain_private_payload_free(frozen_experiment):
    directory, report, events = frozen_experiment
    value = audit.aggregate(directory / "report.json")
    output = json.dumps(value, ensure_ascii=False) + audit.render(value)
    for case in report["rows"]:
        assert case["case_id"] not in output and case["query"] not in output
    assert "Private original norm sentence." not in output
    assert "norm:TEST:" not in output and str(directory) not in output
    assert value["scope"] == "diagnostic_subset" and value["attempted_case_arms"] == 8
    assert "diagnostische Teilmenge" in audit.render(value)
    assert "Präzision einzelner Vorschläge bleibt unbekannt" in audit.render(value)


def save_experiment(directory, report, events, protocol=None):
    if protocol is not None:
        write(directory / "protocol.json", protocol)
        report["protocol_sha256"] = audit.digest(directory / "protocol.json")
    write(directory / "report.json", report)
    (directory / "rows.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


@pytest.fixture
def catalog_experiment(frozen_experiment):
    directory, report, events = frozen_experiment
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import katalog_fingerprint_berechnen
    from risk_layer.research_catalog import ResearchCatalog

    catalog_path = directory / "inputs/catalog.yaml"
    write(catalog_path, {"version": "synthetic-1", "begriffe": [{"id": "private-concept",
        "begriff_kanonisch": "Private original", "norm_anker": ["§ 777 TEST"],
        "definitionsquelle": [{"typ": "privat", "fundstelle": "private catalog reference"}],
        "normketten": [{"glieder": ["§ 777 TEST", "§ 778 TEST"], "verknuepfung": "iVm", "hinweis": "private note"}]}]})
    fingerprint = katalog_fingerprint_berechnen(lade_katalog(catalog_path))
    catalog = ResearchCatalog(catalog_path, expected_fingerprint=fingerprint)
    write(directory / "inputs/index/meta.json", {"catalog_fingerprint": fingerprint})
    report["input_hashes"][str(directory / "inputs/index/meta.json")] = audit.digest(directory / "inputs/index/meta.json")
    report["input_hashes"]["historical/catalog/begriffe.yaml"] = audit.digest(catalog_path)
    reference = audit.parse((directory / "reference.json").read_bytes())
    reference["input_hashes"] = report["input_hashes"]
    write(directory / "reference.json", reference)
    report["reference_sha256"] = audit.digest(directory / "reference.json")
    for source in (Path(audit.__file__).parents[1] / "risk_layer").glob("*.py"):
        destination = directory / "source/risk_layer" / source.name
        shutil.copyfile(source, destination)
        report["source_hashes"]["risk_layer/" + source.name] = audit.digest(destination)
    report["catalog_metadata"] = catalog.metadata()
    report["catalog_provenance"] = {"snapshot": "inputs/catalog.yaml", "sha256": audit.digest(catalog_path),
        "size_bytes": catalog_path.stat().st_size, "reference_sha256": audit.digest(catalog_path)}
    for row in report["rows"]:
        for entry in row["arms"].values():
            result = entry["result"]
            result["catalog_metadata"] = catalog.metadata()
            result["catalog_hints"] = catalog.hints(row["query"])
            event = result["trace"][0]
            event["request"]["messages"][0]["content"] = audit.canonical({
                "SACHVERHALT": row["query"], "KATALOGHINWEISE": result["catalog_hints"]})
            event["request_sha256"] = audit.sha(audit.canonical(event["request"]))
    report["snapshot_hashes"].update({p.relative_to(directory).as_posix(): audit.digest(p)
        for p in directory.rglob("*") if p.is_file() and p.name not in ("report.json", "protocol.json", "rows.jsonl")})
    protocol = audit.parse((directory / "protocol.json").read_bytes())
    for key in ("catalog_metadata", "catalog_provenance", "input_hashes", "snapshot_hashes", "source_hashes", "reference_sha256"):
        protocol[key] = deepcopy(report[key])
    save_experiment(directory, report, events, protocol)
    return directory, report, events


def test_frozen_catalog_replay_is_separate_from_source_reads_and_public_payload(catalog_experiment):
    directory, report, _ = catalog_experiment
    before_modules = set(sys.modules)
    before_files = {path.relative_to(directory) for path in directory.rglob("*") if path.is_file()}
    before_bytecode_policy = sys.dont_write_bytecode
    value = audit.aggregate(directory / "report.json")
    assert not any(name.startswith("_research_catalog_audit_") for name in set(sys.modules) - before_modules)
    assert not any(name.split(".")[0] in ("torch", "spacy") for name in set(sys.modules) - before_modules)
    assert before_files == {path.relative_to(directory) for path in directory.rglob("*") if path.is_file()}
    assert sys.dont_write_bytecode == before_bytecode_policy
    assert value["catalog"]["enabled"] is True
    assert value["catalog"]["counts"] == {"arm_ledgers_with_hints": 8, "ledger_concept_cards": 8,
        "ledger_omitted_concept_cards": 0, "arm_ledgers": 8, "audited_request_payloads": 8, "unavailable_arm_ledgers": 0}
    for arm in audit.ARMS:
        assert value["datasets"]["all"][arm]["source_availability"]["new_pages"] == 0
        assert value["datasets"]["all"][arm]["source_availability"]["new_norms"] == 0
    public = audit.canonical(value) + audit.render(value)
    for private in ("private-concept", "Private original", "§ 777 TEST", "private catalog reference", "private note", str(directory)):
        assert private not in public
    assert "ungeprüfte Suchhinweise" in audit.render(value)
    assert value["catalog"]["yaml_sha256"] == report["catalog_provenance"]["sha256"]


@pytest.mark.parametrize("tamper", ["match", "anchor", "chain", "purpose", "omitted", "typed_offset", "typed_counter", "hint_id", "metadata",
                                   "missing_ledger", "message", "document", "protocol", "index", "dependency", "reference"])
def test_catalog_cards_require_original_text_yaml_policy_and_identical_user_payload(catalog_experiment, tamper):
    directory, report, events = catalog_experiment
    result = report["rows"][0]["arms"]["research"]["result"]
    hints = result["catalog_hints"]
    protocol = audit.parse((directory / "protocol.json").read_bytes())
    if tamper == "match":
        hints["concepts"][0]["matches"][0]["start"] += 1
    elif tamper == "anchor":
        hints["concepts"][0]["norm_anchors"].append("§ 999 TEST")
    elif tamper == "chain":
        hints["concepts"][0]["norm_chains"][0]["glieder"].reverse()
    elif tamper == "purpose":
        hints["purpose"] = "Verified source"
    elif tamper == "omitted":
        hints["omitted_concepts"] += 1
    elif tamper == "typed_offset":
        hints["concepts"][0]["matches"][0]["start"] = 0.0
    elif tamper == "typed_counter":
        hints["omitted_concepts"] = False
    elif tamper == "hint_id":
        hints["concepts"][0]["hint_id"] = "e1q1"
    elif tamper == "metadata":
        result["catalog_metadata"]["policy"]["source_evidence"] = True
    elif tamper == "missing_ledger":
        result.pop("catalog_hints")
    elif tamper in ("message", "document"):
        event = result["trace"][0]
        user = audit.parse(event["request"]["messages"][0]["content"])
        if tamper == "message":
            user["KATALOGHINWEISE"]["concepts"] = []
        else:
            user["SACHVERHALT"] = "A different original document"
        event["request"]["messages"][0]["content"] = audit.canonical(user)
        event["request_sha256"] = audit.sha(audit.canonical(event["request"]))
    elif tamper == "protocol":
        protocol.pop("catalog_metadata")
    elif tamper == "index":
        report["catalog_metadata"]["semantic_fingerprint"] = "0" * 64
        protocol["catalog_metadata"] = report["catalog_metadata"]
    elif tamper == "dependency":
        relative = "risk_layer/research_catalog.py"
        report["source_hashes"].pop(relative)
        protocol["source_hashes"].pop(relative)
    elif tamper == "reference":
        report["catalog_provenance"]["reference_sha256"] = None
        protocol["catalog_provenance"] = report["catalog_provenance"]
    save_experiment(directory, report, events, protocol)
    with pytest.raises(ValueError, match="[Cc]atalog"):
        audit.aggregate(directory / "report.json")


def test_catalog_empty_payload_is_explicit_and_early_failure_is_not_a_supplied_hint():
    value = result("unrelated text", "research")
    metadata = {"fingerprint": "catalog"}
    hints = {"concepts": []}
    value.update(catalog_metadata=metadata, catalog_hints=hints)
    value["trace"][0]["request"]["messages"][0]["content"] = audit.canonical({"SACHVERHALT": "unrelated text", "KATALOGHINWEISE": hints})
    assert audit.audit_catalog_result(value, metadata, hints, "unrelated text")["arm_ledgers"] == 1
    failed = {"status": "source_error", "trace": []}
    assert audit.audit_catalog_result(failed, metadata, hints, "unrelated text")["unavailable_arm_ledgers"] == 1
    with pytest.raises(ValueError, match="Disabled catalog"):
        audit.audit_catalog_result(value, None, None, "unrelated text")
    value.pop("catalog_metadata")
    value.pop("catalog_hints")
    with pytest.raises(ValueError, match="Disabled catalog leaked"):
        audit.audit_catalog_result(value, None, None, "unrelated text")


def test_actual_client_initial_payload_matches_reporter_catalog_contract(catalog_experiment):
    from risk_layer.research_agent import ResearchAgent, ResearchConfig

    _, report, _ = catalog_experiment
    row = report["rows"][0]
    metadata = report["catalog_metadata"]
    hints = row["arms"]["research"]["result"]["catalog_hints"]

    class Session:
        focuses = [{"id": "f0", "start": 0, "end": len(row["query"]), "text": row["query"]}]
        budget = {"used_calls": 0, "max_calls": 8}
        catalog_hints = hints

        def initial_sources(self, limit):
            return []

    class Corpus:
        def new_session(self, text):
            assert text == row["query"]
            return Session()

        def catalog_metadata(self):
            return metadata

    class CapturingAgent(ResearchAgent):
        def _loop(self, messages, session, sources, deadline, result):
            result["status"] = "complete"
            result["trace"].append({"kind": "http", "path": "/apply-template", "request": {"messages": deepcopy(messages)}})

    r = CapturingAgent("http://127.0.0.1:1", "unused-no-model-call", ResearchConfig()).run(row["query"], Corpus(), allow_research=True)
    assert r["status"] == "complete"
    assert audit.audit_catalog_result(r, metadata, hints, row["query"])["audited_request_payloads"] == 1


def test_catalog_hint_id_cannot_become_source_evidence_or_quote(catalog_experiment):
    directory, report, events = catalog_experiment
    value = report["rows"][0]["arms"]["research"]["result"]
    value["suggestions"][0]["normen"][0]["research"]["evidence_id"] = "k1"
    value["trace"][-1]["result"]["suggestions"] = value["suggestions"]
    save_experiment(directory, report, events)
    with pytest.raises(ValueError, match="source|Source|evidence"):
        audit.aggregate(directory / "report.json")


@pytest.fixture
def semantic_catalog_experiment(catalog_experiment):
    directory, report, events = catalog_experiment
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import BGE_M3_REVISION, EmbeddingIndex, HashEncoder
    from risk_layer.gii import NormKnoten
    from risk_layer.graph import NormGraph
    from risk_layer.research_catalog import ResearchCatalog

    catalog_path = directory / "inputs/catalog.yaml"
    write(catalog_path, {"version": "synthetic-2", "begriffe": [{"id": f"private-semantic-{i}",
        "begriff_kanonisch": f"Unspoken catalog concept {i}", "norm_anker": [f"§ {i} AO"],
        "paraphrasen": [f"Curated catalog paraphrase number {i}"]} for i in (1, 2)]})
    graph_path = directory / "inputs/graph.sqlite"
    graph_path.unlink()  # Replace only this test fixture's synthetic placeholder with a real small graph.
    nodes = [NormKnoten(id=f"norm:AO:{i}", law="AO", para=str(i), titel=f"Synthetic {i}",
                       text=f"Invented test source paragraph {i}.") for i in (1, 2)]
    nodes.append(NormKnoten(id="norm:TEST:1", law="TEST", para="1", titel="Synthetic source",
                           text="Private original norm sentence."))
    encoder = HashEncoder(dim=1024)
    encoder.name, encoder.revision = "bge-m3", BGE_M3_REVISION
    graph = NormGraph.build(nodes, graph_path)
    try:
        index = EmbeddingIndex.build(graph, encoder, katalog=lade_katalog(catalog_path), passagen=True)
        index.save(directory / "inputs/index")
    finally:
        graph.close()
    evidence = index.passage_evidence
    assert len(evidence.anchors) == 2
    vector = evidence.anchor_vectors[0].copy()
    vector_sha = audit.sha(vector.tobytes())
    catalog = ResearchCatalog(catalog_path, expected_fingerprint=index.catalog_fingerprint, evidence=evidence)
    report["catalog_metadata"] = catalog.metadata()
    report["catalog_provenance"].update(sha256=audit.digest(catalog_path), size_bytes=catalog_path.stat().st_size,
                                         reference_sha256=audit.digest(catalog_path))
    identity = {**IDENTITY, "index_fingerprint": evidence.fingerprint}
    report["corpus"] = identity
    encoder_path = directory / "inputs/encoder"
    encoder_path.mkdir()
    (encoder_path / "model.bin").write_bytes(b"synthetic encoder identity; no model load")
    files = {"model.bin": audit.digest(encoder_path / "model.bin")}
    report["encoder_provenance"] = {"mode": "hybrid", "source_directory": "private-model",
                                    "snapshot_directory": "inputs/encoder", "files": files}
    report["retrieval"] = {"mode": "hybrid", "format": "research_cpu_hybrid/2", "query_vector": audit.QUERY_VECTOR_FORMAT,
        "model_path": str(encoder_path), "model_identity": {"files": files, "encoder": "bge-m3", "revision": BGE_M3_REVISION,
        "device": "cpu", "dtype": "float32", "cpu_threads": 2, "local_files_only": True}, "policy": {"rrf_k": 60},
        "evidence_fingerprint": evidence.fingerprint, "source_fingerprint": IDENTITY["source_fingerprint"]}
    report["retrieval_setup_metrics"] = hybrid_measurement(1, setup=True)
    report["retrieval_setup_metrics"]["queries"][0]["query_vector_sha256"] = vector_sha
    measured = [report["retrieval_setup_metrics"]]

    def replace_ids(value, replacements):
        if isinstance(value, dict):
            return {key: replace_ids(item, replacements) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_ids(item, replacements) for item in value]
        return replacements.get(value, value) if isinstance(value, str) else value

    for number, event in enumerate(events, 2):
        row = next(row for row in report["rows"] if row["case_id"] == event["case_id"])
        entry = row["arms"][event["arm"]]
        r = entry["result"]
        replacements = {source["evidence_id"]: "ev:" + audit.sha(audit.canonical([
            identity, audit.sha(row["query"]), source["norm_id"], source["text_sha256"],
            source["source_start"], source["source_end"]])) for source in r["initial_sources"]}
        r = replace_ids(r, replacements)
        r.update(catalog_metadata=catalog.metadata(), catalog_hints=catalog.hints(row["query"], vector),
                 catalog_query_vector=vector.tolist())
        r["initial_sources_sha256"] = audit.sha(audit.canonical(r["initial_sources"]))
        r["trace"][0]["request"]["messages"][0]["content"] = audit.canonical({"SACHVERHALT": row["query"], "KATALOGHINWEISE": r["catalog_hints"]})
        r["trace"][0]["request_sha256"] = audit.sha(audit.canonical(r["trace"][0]["request"]))
        entry["result"] = event["result"] = r
        measurement = hybrid_measurement(number)
        measurement["queries"][0].update(query_sha256=audit.sha(row["query"]), characters=len(row["query"]),
                                           query_vector_sha256=vector_sha)
        entry["retrieval_metrics"] = event["retrieval_metrics"] = measurement
        measured.append(measurement)
    report["retrieval_metrics"] = {key: sum(part[key] for part in measured)
                                    for key in audit.RETRIEVAL_COUNTS | audit.RETRIEVAL_TIMES}
    report["retrieval_metrics"]["queries"] = [event for part in measured for event in part["queries"]]
    report["torch_imported"] = False
    for relative in ("inputs/graph.sqlite", "inputs/index/meta.json", "inputs/index/vectors.npy"):
        report["input_hashes"][str(directory / relative)] = audit.digest(directory / relative)
    report["input_hashes"]["historical/catalog/begriffe.yaml"] = audit.digest(catalog_path)
    reference = audit.parse((directory / "reference.json").read_bytes())
    reference["input_hashes"] = report["input_hashes"]
    write(directory / "reference.json", reference)
    report["reference_sha256"] = audit.digest(directory / "reference.json")
    report["snapshot_hashes"].update({p.relative_to(directory).as_posix(): audit.digest(p)
        for p in directory.rglob("*") if p.is_file() and p.name not in ("report.json", "protocol.json", "rows.jsonl")})
    protocol = audit.parse((directory / "protocol.json").read_bytes())
    for key in ("catalog_metadata", "catalog_provenance", "corpus", "retrieval", "encoder_provenance",
                "source_hashes", "input_hashes", "snapshot_hashes", "reference_sha256"):
        protocol[key] = deepcopy(report[key])
    save_experiment(directory, report, events, protocol)
    return directory, report, events


def test_semantic_catalog_replays_bound_anchors_and_each_arms_original_query_vector(semantic_catalog_experiment):
    directory, report, _ = semantic_catalog_experiment
    value = audit.aggregate(directory / "report.json")
    assert value["catalog"]["mode"] == "literal_and_semantic"
    assert value["retrieval"]["query_vector_journal"] is True
    for row in report["rows"]:
        for entry in row["arms"].values():
            hints = entry["result"]["catalog_hints"]
            assert hints["literal_concepts"] == 0 and hints["semantic_concepts"] == 2
            assert all(concept["matches"] == [] for concept in hints["concepts"])
    public = audit.canonical(value) + audit.render(value)
    assert "private-semantic" not in public and "Curated catalog paraphrase" not in public
    assert "catalog_query_vector" not in public and "query_vector_sha256" not in public


def test_failed_initial_semantic_query_remains_a_failure_without_invented_hints(semantic_catalog_experiment):
    directory, report, events = semantic_catalog_experiment
    entry = report["rows"][0]["arms"]["research"]
    r = entry["result"]
    for key in ("catalog_metadata", "catalog_hints", "catalog_query_vector"):
        r.pop(key)
    r.update(status="source_error", trace=[], suggestions=[], initial_sources=[],
             initial_sources_sha256=audit.sha(audit.canonical([])),
             usage={"http_requests": 0, "model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
             errors=[{"kind": "source_error", "message": "Initial query failed"}])
    entry["retrieval_metrics"]["queries"][0].update(status="failed", query_vector_sha256=None)
    entry["retrieval_metrics"]["failed_queries"] = 1
    report["retrieval_metrics"]["failed_queries"] = 1
    report.update(status="complete_with_errors", technical_failures=1)
    report["summaries"] = {dataset: {arm: audit.metric_counts(
        [row for row in report["rows"] if dataset == "all" or row["dataset"] == dataset], arm, True)
        for arm in ("baseline", *audit.ARMS)} for dataset in ("all", *audit.DATASETS)}
    save_experiment(directory, report, events)
    value = audit.aggregate(directory / "report.json")
    assert value["catalog"]["counts"]["unavailable_arm_ledgers"] == 1
    assert value["catalog"]["counts"]["arm_ledgers"] == 7
    assert value["datasets"]["all"]["research"]["metrics"]["failed_positive_cases"] == 1


@pytest.mark.parametrize("tamper", ["order", "score", "invented_span", "vector", "missing_vector", "wrong_first_query",
                                   "missing_checksum", "legacy_journal", "model_payload", "anchor_generation", "encoder_generation"])
def test_semantic_catalog_cannot_change_anchor_order_vector_or_retrieval_provenance(semantic_catalog_experiment, tamper):
    directory, report, events = semantic_catalog_experiment
    row = report["rows"][0]
    entry = row["arms"]["research"]
    r = entry["result"]
    protocol = audit.parse((directory / "protocol.json").read_bytes())
    if tamper == "order":
        r["catalog_hints"]["concepts"].reverse()
    elif tamper == "score":
        r["catalog_hints"]["concepts"][0]["semantic_matches"][0]["score"] -= 0.1
    elif tamper == "invented_span":
        r["catalog_hints"]["concepts"][0]["matches"] = [{"start": 0, "end": 7, "text": row["query"][:7], "methods": ["literal"]}]
    elif tamper == "vector":
        r["catalog_query_vector"] = r["catalog_query_vector"][1:] + r["catalog_query_vector"][:1]
    elif tamper == "missing_vector":
        r.pop("catalog_query_vector")
    elif tamper == "wrong_first_query":
        entry["retrieval_metrics"]["queries"][0]["query_sha256"] = audit.sha("different query")
    elif tamper == "missing_checksum":
        entry["retrieval_metrics"]["queries"][0].pop("query_vector_sha256")
    elif tamper == "legacy_journal":
        report["retrieval"].pop("format")
        report["retrieval"].pop("query_vector")
        protocol["retrieval"] = report["retrieval"]
    elif tamper == "model_payload":
        request = r["trace"][0]["request"]
        user = audit.parse(request["messages"][0]["content"])
        user["catalog_query_vector"] = r["catalog_query_vector"]
        request["messages"][0]["content"] = audit.canonical(user)
        r["trace"][0]["request_sha256"] = audit.sha(audit.canonical(request))
    elif tamper == "anchor_generation":
        report["catalog_metadata"]["semantic"]["anchor_vectors_sha256"] = "0" * 64
        protocol["catalog_metadata"] = report["catalog_metadata"]
    elif tamper == "encoder_generation":
        report["retrieval"]["model_identity"]["revision"] = "0" * 40
        protocol["retrieval"] = report["retrieval"]
    save_experiment(directory, report, events, protocol)
    with pytest.raises(ValueError, match="catalog|Catalog|hybrid"):
        audit.aggregate(directory / "report.json")


def test_budget_decisions_are_aggregated_without_adding_timing_samples(frozen_experiment):
    directory, report, events = frozen_experiment
    before = audit.aggregate(directory / "report.json")
    r = report["rows"][0]["arms"]["research"]["result"]
    r["trace"].append({"kind": "budget", "reason": "context_finish_reserve", "round": 2,
                       "prompt_tokens_before_finish": 12000})
    write(directory / "report.json", report)
    (directory / "rows.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    after = audit.aggregate(directory / "report.json")
    old, new = before["datasets"]["all"]["research"], after["datasets"]["all"]["research"]
    assert new["budget_events"] == {"context_finish_reserve": 1}
    assert new["wall_seconds"] == old["wall_seconds"]
    assert new["generation_request_seconds"] == old["generation_request_seconds"]
    assert new["client_measured_usage"] == old["client_measured_usage"]
    assert "Budgetentscheidungen ohne eigene Laufzeitmessung research" in audit.render(after)


def test_explicit_bm25_protocol_remains_encoder_free(frozen_experiment):
    directory, report, events = frozen_experiment
    report.update(retrieval={"mode": "bm25", "policy": "lexical"}, encoder_provenance={"mode": "bm25", "files": {}},
                  retrieval_setup_metrics={}, retrieval_metrics={}, torch_imported=False)
    for row in report["rows"]:
        for arm in report["arms"]:
            row["arms"][arm]["retrieval_metrics"] = {}
    for event in events:
        event["retrieval_metrics"] = {}
    protocol = json.loads((directory / "protocol.json").read_bytes())
    for key in ("retrieval", "encoder_provenance", "corpus"):
        protocol[key] = deepcopy(report[key])
    write(directory / "protocol.json", protocol)
    report["protocol_sha256"] = audit.digest(directory / "protocol.json")
    write(directory / "report.json", report)
    (directory / "rows.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    value = audit.aggregate(directory / "report.json")
    assert value["retrieval"]["mode"] == "bm25" and "encoder" not in value["retrieval"]
    report["torch_imported"] = True
    write(directory / "report.json", report)
    with pytest.raises(ValueError, match="unexpectedly used an encoder"):
        audit.aggregate(directory / "report.json")


def hybrid_measurement(number, *, setup=False, failed=False):
    event = {"number": number, "query_sha256": audit.sha("private retrieval query " + str(number)),
             "characters": 30, "limit": 1 if setup else 6, "status": "failed" if failed else "complete",
             "model_load_seconds": 0., "encode_seconds": .25, "retrieval_seconds": .125, "encode_count": 1}
    return {"model_load_count": int(setup), "model_validation_seconds": 1. if setup else 0.,
            "model_load_seconds": 2. if setup else 0., "query_count": 1, "encode_count": 1,
            "failed_queries": int(failed), "encode_seconds": .25, "retrieval_seconds": .125, "queries": [event]}


@pytest.mark.parametrize("failed,vector_sha", [(False, "a" * 64), (True, "a" * 64), (True, None)])
def test_hybrid_v2_query_journal_preserves_success_and_failure_vector_bindings(failed, vector_sha):
    value = hybrid_measurement(1, failed=failed)
    value["queries"][0]["query_vector_sha256"] = vector_sha
    assert audit.checked_retrieval_metrics(value, vector_journal=True)["query_count"] == 1
    with pytest.raises(ValueError, match="Malformed hybrid"):
        audit.checked_retrieval_metrics(value)


@pytest.mark.parametrize("vector_sha", [None, False, 1, "a" * 63, "A" * 64, []])
def test_completed_hybrid_v2_query_requires_a_valid_vector_checksum(vector_sha):
    value = hybrid_measurement(1)
    value["queries"][0]["query_vector_sha256"] = vector_sha
    with pytest.raises(ValueError, match="query-vector"):
        audit.checked_retrieval_metrics(value, vector_journal=True)


@pytest.mark.parametrize("tamper", [None, "vector", "dimension", "nan", "bool", "normalization", "query", "later", "limit", "failed"])
def test_semantic_catalog_uses_exactly_the_existing_first_original_query_vector(tamper):
    import numpy as np

    r = row()
    vector = np.zeros(1024, dtype=np.float32)
    vector[0] = 1
    measurement = hybrid_measurement(2)
    event = measurement["queries"][0]
    event.update(query_sha256=audit.sha(r["query"]), characters=len(r["query"]), limit=6,
                 query_vector_sha256=audit.sha(vector.tobytes()))
    entry = {"result": {"catalog_query_vector": vector.tolist()}, "retrieval_metrics": measurement}
    if tamper == "vector":
        entry["result"]["catalog_query_vector"] = np.roll(vector, 1).tolist()
    elif tamper == "dimension":
        entry["result"]["catalog_query_vector"].pop()
    elif tamper == "nan":
        entry["result"]["catalog_query_vector"][0] = float("nan")
    elif tamper == "bool":
        entry["result"]["catalog_query_vector"][0] = True
    elif tamper == "normalization":
        entry["result"]["catalog_query_vector"][0] = 0.5
    elif tamper == "query":
        event["query_sha256"] = audit.sha("A different follow-up search")
    elif tamper == "later":
        measurement["queries"].insert(0, {**event, "query_vector_sha256": "f" * 64})
    elif tamper == "limit":
        event["limit"] = 1
    elif tamper == "failed":
        event["status"] = "failed"
    if tamper is None:
        assert np.array_equal(audit.checked_catalog_vector(r, entry, 6), vector)
    else:
        with pytest.raises(ValueError, match="[Cc]atalog"):
            audit.checked_catalog_vector(r, entry, 6)


@pytest.fixture
def hybrid_experiment(frozen_experiment):
    directory, report, events = frozen_experiment
    encoder = directory / "inputs/encoder"
    encoder.mkdir()
    (encoder / "model.bin").write_bytes(b"synthetic encoder")
    files = {"model.bin": audit.digest(encoder / "model.bin")}
    source = directory / "source/risk_layer/research_hybrid.py"
    source.write_bytes(b"# synthetic CPU retrieval implementation")
    report["source_hashes"]["risk_layer/research_hybrid.py"] = audit.digest(source)
    report["snapshot_hashes"].update({"source/risk_layer/research_hybrid.py": audit.digest(source),
                                       "inputs/encoder/model.bin": files["model.bin"]})
    report["encoder_provenance"] = {"mode": "hybrid", "source_directory": "private-original-model",
                                     "snapshot_directory": "inputs/encoder", "files": files}
    report["retrieval"] = {"mode": "hybrid", "model_path": str(encoder), "model_identity": {
        "files": files, "encoder": "bge-m3", "revision": "b" * 40, "device": "cpu", "dtype": "float32",
        "cpu_threads": 2, "local_files_only": True}, "policy": {"rrf_k": 60},
        "evidence_fingerprint": IDENTITY["index_fingerprint"], "source_fingerprint": IDENTITY["source_fingerprint"]}
    report["retrieval_setup_metrics"] = hybrid_measurement(1, setup=True)
    measured = [report["retrieval_setup_metrics"]]
    for number, event in enumerate(events, 2):
        delta = hybrid_measurement(number)
        event["retrieval_metrics"] = delta
        target = next(row for row in report["rows"] if row["case_id"] == event["case_id"])
        target["arms"][event["arm"]]["retrieval_metrics"] = delta
        measured.append(delta)
    report["retrieval_metrics"] = {key: sum(part[key] for part in measured)
                                    for key in audit.RETRIEVAL_COUNTS | audit.RETRIEVAL_TIMES}
    report["retrieval_metrics"]["queries"] = [event for part in measured for event in part["queries"]]
    report["timing"]["retrieval_prepare_seconds"] = 2.5
    report["torch_imported"] = True
    protocol = json.loads((directory / "protocol.json").read_bytes())
    for key in ("retrieval", "encoder_provenance", "corpus", "source_hashes", "snapshot_hashes"):
        protocol[key] = deepcopy(report[key])
    write(directory / "protocol.json", protocol)
    report["protocol_sha256"] = audit.digest(directory / "protocol.json")
    write(directory / "report.json", report)
    (directory / "rows.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return directory, report, events


def test_hybrid_aggregate_binds_model_and_separates_prepare_from_case_measurements(hybrid_experiment):
    directory, report, _ = hybrid_experiment
    value = audit.aggregate(directory / "report.json")
    assert value["retrieval"]["mode"] == "hybrid" and value["retrieval"]["device"] == "cpu"
    assert value["retrieval"]["including_prepare"]["query_count"] == 9
    assert value["retrieval"]["prepare"]["query_count"] == 1
    assert value["datasets"]["all"]["research"]["cpu_retrieval"]["query_count"] == 4
    assert value["datasets"]["all"]["research"]["cpu_retrieval"]["model_load_seconds"] == 0
    assert value["audit"]["torch_imported"] is False and value["audit"]["benchmark_torch_imported"] is True
    output = json.dumps(value) + audit.render(value)
    assert "CPU" in output and "Probelauf" in output
    assert str(directory) not in output and "private-original-model" not in output
    assert all(event["query_sha256"] not in output for event in report["retrieval_metrics"]["queries"])


@pytest.mark.parametrize("tamper", ["file", "metadata", "total_count", "delta", "warmup_load", "query_order", "query_time", "missing_metric"])
def test_hybrid_provenance_and_all_measured_denominators_are_checked(hybrid_experiment, tamper):
    directory, report, events = hybrid_experiment
    if tamper == "file":
        (directory / "inputs/encoder/model.bin").write_bytes(b"changed")
    elif tamper == "metadata":
        report["retrieval"]["model_identity"]["device"] = "cuda:0"
    elif tamper == "total_count":
        report["retrieval_metrics"]["query_count"] += 1
    elif tamper == "delta":
        events[0]["retrieval_metrics"]["encode_count"] = 0
    elif tamper == "warmup_load":
        report["retrieval_setup_metrics"]["model_load_count"] = 0
    elif tamper == "query_order":
        report["retrieval_metrics"]["queries"].reverse()
    elif tamper == "query_time":
        report["retrieval_metrics"]["encode_seconds"] += 1
    else:
        events[0]["retrieval_metrics"].pop("failed_queries")
    write(directory / "report.json", report)
    (directory / "rows.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit.aggregate(directory / "report.json")


@pytest.mark.parametrize("tamper", ["protocol", "snapshot", "journal_gap", "journal_duplicate", "metric", "false_full", "baseline", "model_payload"])
def test_incomplete_or_tampered_experiment_cannot_be_published(frozen_experiment, tamper):
    directory, report, events = frozen_experiment
    if tamper == "protocol":
        (directory / "protocol.json").write_text('{}')
    elif tamper == "snapshot":
        (directory / "reference.json").write_text('{}')
    elif tamper == "journal_gap":
        events.pop(1)
    elif tamper == "journal_duplicate":
        events.append(events[0])
    elif tamper == "metric":
        report["summaries"]["all"]["research"]["hit5_cases"] += 1
    elif tamper == "false_full":
        report["diagnostic_subset"] = False
    elif tamper == "baseline":
        report["rows"][0]["baseline"]["suggestions"] = [suggestion(report["rows"][0]["query"], page(report["rows"][0]["query"]))]
    else:
        report["model_provenance"]["model"]["repo"] = report["rows"][0]["query"]
    write(directory / "report.json", report)
    (directory / "rows.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError):
        audit.aggregate(directory / "report.json")


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/secret", "source\\outside"])
def test_snapshot_paths_cannot_escape_private_archive(tmp_path, name):
    with pytest.raises(ValueError):
        audit.safe_path(tmp_path, name)
