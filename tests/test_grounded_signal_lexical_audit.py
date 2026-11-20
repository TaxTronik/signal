"""Model-free independent lexical-audit regressions."""
import ast
from copy import deepcopy
from pathlib import Path
import sqlite3

import pytest


from experiments.grounded_signal import audit_lexical_results as m


PATH = Path(m.__file__).resolve()


def terminal_case():
    return ({"format": "grounded_signal_lexical_probe/1", "status": "complete", "new_fact_calls": 0,
             "planned_cases": 8, "end_verification": "pass", "results": {m.ARM: [{"status": "complete"} for _ in range(8)]}},
            {"arms": [m.ARM]})


@pytest.mark.parametrize("change", ["running", "subset", "new_facts", "bool_zero", "incomplete_status", "end_failed"])
def test_terminal_guard_rejects_partial_or_unbound_execution(tmp_path, change):
    report, protocol = terminal_case()
    if change == "running":
        report["status"] = "running"
    elif change == "subset":
        report["results"][m.ARM].pop()
    elif change == "new_facts":
        report["new_fact_calls"] = 1
    elif change == "bool_zero":
        report["new_fact_calls"] = False
    elif change == "incomplete_status":
        report["results"][m.ARM][0]["status"] = "timeout"
    else:
        report["end_verification"] = "failed"
    with pytest.raises(ValueError):
        m.terminal(tmp_path, report, protocol)


def test_terminal_failures_stay_in_all_eight_cases(tmp_path):
    report, protocol = terminal_case()
    report["results"][m.ARM][0]["status"] = "upstream_failure"
    report["status"] = "incomplete"
    m.terminal(tmp_path, report, protocol)
    (tmp_path / "interruption.json").write_text("{}")
    with pytest.raises(ValueError, match="Interrupted"):
        m.terminal(tmp_path, report, protocol)


def test_schema_has_three_fixed_slots_and_same_norm_sources():
    units = [{"unit_id": f"original-{i}", "norm_id": f"norm:{i}", "start": 0, "end": 5, "text": "Rule."} for i in range(3)]
    package = {"candidate_norm_ids": [u["norm_id"] for u in units], "units": units,
               "required_groups": [{"norm_id": u["norm_id"], "unit_ids": [u["unit_id"]]} for u in units],
               "complete_local_norms": True, "reference_closure_complete": True, "references": [], "limitations": []}
    schema = m.expected_schema([], package, {"answer": {"facts": [{"id": "f1"}]}})
    slots = schema["properties"]["hints"]
    assert slots["required"] == ["norm:0", "norm:1", "norm:2"]
    for index, slot in enumerate(slots["properties"].values(), 1):
        branches = slot["oneOf"]
        assert branches[0]["properties"]["evidence_unit_ids"]["items"]["enum"] == [f"u{index}"]
        assert branches[0]["properties"]["supported_fact_ids"]["minItems"] == 1
        assert branches[1]["properties"]["relation"]["enum"] == ["none"]
    empty = m.expected_schema([], package, {"answer": {"facts": []}})
    for slot in empty["properties"]["hints"]["properties"].values():
        assert len(slot["oneOf"]) == 1
        assert slot["oneOf"][0]["properties"]["supported_fact_ids"]["maxItems"] == 0


def test_exact_span_plan_keeps_negation_text_and_original_offsets():
    text = "  Keine Rolle.\nEine zweite Frage?  "
    spans = m.span_plan(text)
    assert [s["text"] for s in spans] == ["Keine Rolle.", "Eine zweite Frage?"]
    assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)
    assert spans[0]["start"] == 2


def test_prompt_extraction_is_literal_and_does_not_execute_archived_source():
    source = 'FACT_PROMPT="fact"\nDECISION_PROMPT=("decision " "prompt")\nraise RuntimeError("must not execute")'
    assert m.prompts_from(source) == {"FACT_PROMPT": "fact", "DECISION_PROMPT": "decision prompt"}
    with pytest.raises((ValueError, TypeError)):
        m.prompts_from('FACT_PROMPT = open("private")\nDECISION_PROMPT="x"')


def model():
    return {"model": {"format": "quality_research_model/1", "status": "complete", "repo": "repo", "revision": "pin",
                      "filename": "model.gguf", "quantization": "Q6_K", "model": {"bytes": 100, "sha256": "a" * 64}}}


@pytest.mark.parametrize("change", ["quantization", "revision", "bytes", "sha256"])
def test_model_identity_binds_quantization_revision_and_weight_bytes(change):
    a, b = model(), model()
    if change in {"bytes", "sha256"}:
        b["model"]["model"][change] = 101 if change == "bytes" else "b" * 64
    else:
        b["model"][change] = "different"
    assert m.model_identity(a) != m.model_identity(b)


def package_for_graph():
    units = [{"unit_id": f"unit:{i}", "norm_id": f"norm:{i}", "start": 0, "end": 5, "text": "Rule.",
              "norm_text_sha256": m.common.rawsha(b"Rule.")} for i in range(3)]
    package = {"candidate_norm_ids": [u["norm_id"] for u in units], "norm_ids": [u["norm_id"] for u in units],
               "units": units, "unit_ids": [u["unit_id"] for u in units], "status": "ready", "complete_local_norms": True,
               "required_groups": [{"norm_id": u["norm_id"], "unit_ids": [u["unit_id"]], "policy": "all_stored_parent_norm_text"} for u in units]}
    package["fingerprint"] = m.digest(package)
    package["package_id"] = "pkg:" + package["fingerprint"]
    return package, {u["unit_id"]: u for u in units}


def test_original_graph_check_rejects_self_consistent_but_wrong_unit_content(tmp_path):
    graph = tmp_path / "graph.sqlite"
    con = sqlite3.connect(graph)
    con.execute("CREATE TABLE knoten (id TEXT, text TEXT)")
    con.executemany("INSERT INTO knoten VALUES (?,?)", [(f"norm:{i}", "Rule.") for i in range(3)])
    con.commit()
    con.close()
    package, table = package_for_graph()
    before = graph.read_bytes()
    assert m.original_sources(graph.resolve(), {package["package_id"]: package}, table)["source_characters"] == 15
    assert graph.read_bytes() == before
    # Both supplied tables and their self-reported fingerprints agree, but the graph does not.
    package["units"][0]["text"] = "Fake."
    package["units"][0]["norm_text_sha256"] = m.common.rawsha(b"Fake.")
    package["fingerprint"] = m.digest({k: v for k, v in package.items() if k not in {"fingerprint", "package_id"}})
    package["package_id"] = "pkg:" + package["fingerprint"]
    with pytest.raises(ValueError, match="original norm"):
        m.original_sources(graph.resolve(), {package["package_id"]: package}, table)


def test_original_source_cap_never_crops_whole_norms(tmp_path):
    package, table = package_for_graph()
    package["status"] = "overflow"
    package["complete_local_norms"] = False
    package["fingerprint"] = m.digest({k: v for k, v in package.items() if k not in {"fingerprint", "package_id"}})
    package["package_id"] = "pkg:" + package["fingerprint"]
    graph = tmp_path / "graph.sqlite"
    sqlite3.connect(graph).close()
    with pytest.raises(ValueError):
        m.original_sources(graph.resolve(), {package["package_id"]: package}, table)


def test_frozen_inventory_rejects_extra_files(tmp_path):
    p = tmp_path / "one.json"
    p.write_text("{}")
    bound = m.common.BoundFiles()
    expected = {"one.json": m.common.rawsha(p.read_bytes())}
    m.inventory(bound, tmp_path, expected)
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError):
        m.inventory(bound, tmp_path, expected)


def test_complete_model_prompt_cannot_change_its_required_tool_schema():
    text = "Text."
    spans = m.span_plan(text)
    request = {"messages": [{"role": "system", "content": "fact"}, {"role": "user", "content": m.common.canonical({"case_spans": spans}).decode()}],
               "tools": [{"type": "function", "function": {"name": "extract_facts",
                    "description": "Vollständige strukturierte Antwort auf die angegebene Teilaufgabe.", "parameters": m.expected_schema(spans, None, None)}}]}
    call = {"status": "complete", "trace": [{"path": "/apply-template", "request": request}]}
    m.check_prompt(call, text, {"FACT_PROMPT": "fact", "DECISION_PROMPT": "decision"})
    altered = deepcopy(call)
    altered["trace"][0]["request"]["tools"][0]["function"]["parameters"]["properties"]["facts"]["maxItems"] = 1
    with pytest.raises(ValueError, match="parameters"):
        m.check_prompt(altered, text, {"FACT_PROMPT": "fact", "DECISION_PROMPT": "decision"})


def test_imports_have_no_producer_evaluator_or_models():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    names = [f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert {n for n in names if n.startswith("experiments")} == {"experiments.grounded_signal.audit_grounded_results"}
    assert not any(n.startswith(("risk_layer", "torch", "transformers", "lexical_probe")) for n in names)
