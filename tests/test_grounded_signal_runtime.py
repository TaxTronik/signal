from copy import deepcopy
import json

import pytest

from experiments.grounded_signal import runtime as rt
from risk_layer.research_agent import ResearchConfig, ResearchError


TEXT = "Mara ist Geschäftsführerin. Steuern werden nicht bezahlt."


def package():
    return {"status": "ready", "package_id": "p1", "candidate_norm_ids": ["norm:T:1", "norm:T:2"],
            "units": [{"unit_id": "long-unit-a", "norm_id": "norm:T:1", "start": 0, "end": 17,
                       "text": "Vollständige Norm."},
                      {"unit_id": "long-unit-b", "norm_id": "norm:T:2", "start": 0, "end": 16,
                       "text": "Ein anderer Text."}], "limitations": ["Not legally complete"],
            "diagnostic_labels": {"target_norm_ids": ["norm:T:1"]}, "family_id": "must-not-appear"}


def raw_facts():
    return {"facts": [{"id": "f1", "span_id": "s1", "kind": "role", "polarity": "affirmed",
                       "statement": "Mara ist Geschäftsführerin."}], "unresolved": []}


def raw_hints():
    return {"hints": {"norm:T:1": {"status": "hint", "evidence_unit_ids": ["u1"],
                       "supported_fact_ids": ["f1"], "open_conditions": ["Welche weiteren Voraussetzungen gelten?"],
                       "relation": "duty"},
                      "norm:T:2": {"status": "no_hint", "evidence_unit_ids": [],
                       "supported_fact_ids": [], "open_conditions": [], "relation": "none"}}, "unresolved": []}


class FakeAgent:
    model = "local-test"
    config = ResearchConfig(max_output_tokens=3072, timeout_seconds=10, reuse_prompt_cache=True)

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def _completion(self, payload, deadline, result):
        self.requests.append(deepcopy(payload))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        result["usage"]["model_calls"] += 1
        return {"role": "assistant", "tool_calls": [{"id": "call1", "type": "function", "function": {
            "name": payload["tools"][0]["function"]["name"], "arguments": json.dumps(answer)}}]}


def test_fact_and_hint_stages_keep_original_spans_and_whole_evidence():
    agent = FakeAgent([raw_facts(), raw_hints()])
    original_package = package()
    before = deepcopy(original_package)
    facts = rt.extract(agent, TEXT)
    result = rt.decide(agent, TEXT, original_package, facts)
    assert facts["status"] == result["status"] == "complete"
    assert facts["answer"]["facts"][0]["text"] == "Mara ist Geschäftsführerin."
    assert facts["answer"]["facts"][0]["end"] == 27
    assert result["answer"]["hints"][0]["evidence_unit_ids"] == ["long-unit-a"]
    assert original_package == before
    for request in agent.requests:
        assert request["cache_prompt"] is False
        assert request["tool_choice"] == "required" and len(request["tools"]) == 1
        assert "diagnostic_labels" not in json.dumps(request)
        assert "must-not-appear" not in json.dumps(request)
    assert "original_sources" not in json.loads(agent.requests[0]["messages"][1]["content"])
    assert json.loads(agent.requests[1]["messages"][1]["content"])["original_case"] == TEXT


def test_direct_arm_has_same_sources_and_does_not_run_fact_call():
    agent = FakeAgent([{**raw_hints(), "facts": raw_facts()["facts"]}])
    result = rt.decide(agent, TEXT, package(), direct=True)
    assert result["status"] == "complete" and len(agent.requests) == 1
    assert result["answer"]["facts"][0]["text"] == "Mara ist Geschäftsführerin."
    request = agent.requests[0]
    assert list(request["tools"][0]["function"]["parameters"]["properties"])[0] == "facts"
    assert "attributed_facts" not in json.loads(request["messages"][1]["content"])


@pytest.mark.parametrize("mutation", [
    lambda h: h["hints"].pop("norm:T:2"),
    lambda h: h["hints"].update({"norm:T:other": deepcopy(h["hints"]["norm:T:1"])}),
    lambda h: h["hints"]["norm:T:1"].update(evidence_unit_ids=["u2"]),
    lambda h: h["hints"]["norm:T:1"].update(evidence_unit_ids=["u99"]),
    lambda h: h["hints"]["norm:T:1"].update(supported_fact_ids=["f8"]),
    lambda h: h["hints"]["norm:T:1"].update(supported_fact_ids=[]),
    lambda h: h["hints"]["norm:T:1"].update(relation="none"),
    lambda h: h["hints"]["norm:T:2"].update(relation="definition"),
    lambda h: h["hints"]["norm:T:1"].update(norm_id="norm:T:other"),
    lambda h: h["hints"]["norm:T:1"].update(evidence_unit_ids=["u1", "u1"]),
    lambda h: h["hints"]["norm:T:1"].update(supported_fact_ids=["f1", "f1"]),
    lambda h: h.update(hints=list(h["hints"].values())),
    lambda h: h.update(secret_reason="unsupported"),
])
def test_malformed_decisions_never_become_partial_or_empty_success(mutation):
    hints = raw_hints()
    mutation(hints)
    agent = FakeAgent([raw_facts(), hints])
    facts = rt.extract(agent, TEXT)
    result = rt.decide(agent, TEXT, package(), facts)
    assert result["status"] == "invalid_hints" and "answer" not in result


@pytest.mark.parametrize("mutation", [
    lambda f: f["facts"][0].update(span_id="s99"),
    lambda f: f["facts"].append(deepcopy(f["facts"][0])),
    lambda f: f["facts"][0].update(polarity="true"),
    lambda f: f["facts"][0].update(statement="x" * 181),
    lambda f: f.update(extra="ignored?"),
])
def test_invalid_facts_stop_downstream_inference(mutation):
    answer = raw_facts()
    mutation(answer)
    agent = FakeAgent([answer])
    facts = rt.extract(agent, TEXT)
    result = rt.decide(agent, TEXT, package(), facts)
    assert facts["status"] == "invalid_facts"
    assert result["status"] == "upstream_failure"
    assert len(agent.requests) == 1 and result["usage"]["model_calls"] == 0


def test_source_binding_does_not_claim_fact_statement_is_true():
    answer = raw_facts()
    answer["facts"][0]["statement"] = "Mara ist eine erfundene Steuerbehörde."
    result = rt.extract(FakeAgent([answer]), TEXT)
    assert result["status"] == "complete"
    assert result["answer"]["facts"][0]["statement"] != result["answer"]["facts"][0]["text"]
    assert "semantic_verified" not in result


def test_timeout_and_missing_tool_call_are_technical_failures():
    result = rt.extract(FakeAgent([ResearchError("timeout", "deadline")]), TEXT)
    assert result["status"] == "timeout" and "answer" not in result
    class ProseAgent(FakeAgent):
        def _completion(self, payload, deadline, result):
            return {"role": "assistant", "content": "I cannot decide"}
    result = rt.extract(ProseAgent([]), TEXT)
    assert result["status"] == "invalid_answer"


def test_case_spans_keep_full_long_sentences_and_negation():
    text = "  " + "Wort " * 500 + "nicht.\nNächster Satz!  "
    spans = rt.case_spans(text)
    assert len(spans) == 2 and spans[0]["text"].endswith("nicht.")
    assert len(spans[0]["text"]) > 2000
    for span in spans:
        assert span["text"] == text[span["start"]:span["end"]]


def test_renderer_uses_original_quotes_and_does_not_assert_applicability():
    agent = FakeAgent([raw_facts(), raw_hints()])
    answer = rt.decide(agent, TEXT, package(), rt.extract(agent, TEXT))["answer"]
    rendered = rt.render_hints(answer, {u["unit_id"]: u for u in package()["units"]})
    assert len(rendered) == 1 and rendered[0]["applicability_established"] is False
    assert rendered[0]["source_quotes"][0]["text"] == "Vollständige Norm."
    assert rendered[0]["case_quotes"] == ["Mara ist Geschäftsführerin."]


def accepts_schema(value, schema):
    """Independent evaluator for the deliberately small emitted schema subset."""
    assert set(schema) <= {"type", "properties", "required", "additionalProperties", "enum",
                           "items", "minItems", "maxItems", "minLength", "maxLength", "oneOf"}
    if "oneOf" in schema:
        return sum(accepts_schema(value, branch) for branch in schema["oneOf"]) == 1
    if "enum" in schema and value not in schema["enum"]:
        return False
    if schema.get("type") == "object":
        return (isinstance(value, dict) and set(schema.get("required", [])) <= value.keys()
                and (schema.get("additionalProperties", True) or value.keys() <= schema["properties"].keys())
                and all(accepts_schema(v, schema["properties"][k]) for k, v in value.items()))
    if schema.get("type") == "array":
        return (isinstance(value, list) and schema.get("minItems", 0) <= len(value) <= schema["maxItems"]
                and all(accepts_schema(item, schema["items"]) for item in value))
    if schema.get("type") == "string":
        return (isinstance(value, str) and schema.get("minLength", 0) <= len(value)
                <= schema.get("maxLength", float("inf")))
    raise AssertionError("Unexpected schema type")


def test_norm_slots_are_required_and_references_are_norm_specific():
    sources, _ = rt.model_sources(package())
    schema = rt.decision_schema(sources, ["f1"])
    slots = schema["properties"]["hints"]
    assert slots["required"] == ["norm:T:1", "norm:T:2"]
    assert slots["additionalProperties"] is False
    for nid, source_id in (("norm:T:1", "u1"), ("norm:T:2", "u2")):
        alternatives = slots["properties"][nid]["oneOf"]
        assert len(alternatives) == 2
        for branch in alternatives:
            assert "norm_id" not in branch["properties"]
            assert branch["properties"]["evidence_unit_ids"]["items"]["enum"] == [source_id]
        hint = alternatives[0]["properties"]
        assert hint["evidence_unit_ids"]["minItems"] == hint["supported_fact_ids"]["minItems"] == 1
    assert accepts_schema(raw_hints(), schema)


@pytest.mark.parametrize("mutation", [
    lambda h: h["hints"].pop("norm:T:2"),
    lambda h: h["hints"].update({"norm:T:3": h["hints"]["norm:T:1"]}),
    lambda h: h["hints"]["norm:T:1"].update(norm_id="norm:T:1"),
    lambda h: h["hints"]["norm:T:1"].update(evidence_unit_ids=["u2"]),
    lambda h: h["hints"]["norm:T:1"].update(evidence_unit_ids=[]),
    lambda h: h["hints"]["norm:T:1"].update(supported_fact_ids=[]),
    lambda h: h["hints"]["norm:T:1"].update(supported_fact_ids=["f2"]),
    lambda h: h["hints"]["norm:T:1"].update(relation="none"),
    lambda h: h["hints"]["norm:T:1"].update(status="insufficient_context"),
    lambda h: h["hints"]["norm:T:2"].update(relation="scope"),
])
def test_schema_itself_rejects_cardinality_references_and_status_relation(mutation):
    sources, _ = rt.model_sources(package())
    answer = raw_hints()
    mutation(answer)
    assert not accepts_schema(answer, rt.decision_schema(sources, ["f1"]))


@pytest.mark.parametrize("status", ["no_hint", "insufficient_context"])
def test_empty_fact_inventory_has_no_empty_enum_or_invented_fact(status):
    answer = raw_hints()
    for hint in answer["hints"].values():
        hint.update(status=status, relation="none", supported_fact_ids=[])
    agent = FakeAgent([{"facts": [], "unresolved": []}, answer])
    result = rt.decide(agent, TEXT, package(), rt.extract(agent, TEXT))
    assert result["status"] == "complete" and len(result["answer"]["hints"]) == 2
    schema = agent.requests[-1]["tools"][0]["function"]["parameters"]
    assert accepts_schema(answer, schema)
    assert '"enum": []' not in json.dumps(schema)
    for slot in schema["properties"]["hints"]["properties"].values():
        assert len(slot["oneOf"]) == 1
        assert slot["oneOf"][0]["properties"]["supported_fact_ids"]["maxItems"] == 0
    answer["hints"]["norm:T:1"].update(status="hint", relation="duty", supported_fact_ids=["f1"])
    assert not accepts_schema(answer, schema)


def test_direct_fact_references_still_require_the_actually_emitted_fact():
    answer = {**raw_hints(), "facts": raw_facts()["facts"]}
    answer["hints"]["norm:T:1"]["supported_fact_ids"] = ["f8"]
    agent = FakeAgent([answer])
    result = rt.decide(agent, TEXT, package(), direct=True)
    schema = agent.requests[0]["tools"][0]["function"]["parameters"]
    assert accepts_schema(answer, schema)  # Dynamic cross-field existence is a host invariant.
    assert result["status"] == "invalid_hints" and "answer" not in result


def test_slots_expand_in_source_order_even_if_model_object_order_differs():
    hints = raw_hints()
    hints["hints"] = dict(reversed(list(hints["hints"].items())))
    agent = FakeAgent([raw_facts(), hints])
    result = rt.decide(agent, TEXT, package(), rt.extract(agent, TEXT))
    assert result["status"] == "complete"
    assert [h["norm_id"] for h in result["answer"]["hints"]] == package()["candidate_norm_ids"]


def test_duplicate_raw_object_key_remains_an_error_before_slot_expansion():
    class DuplicateKeyAgent(FakeAgent):
        def _completion(self, payload, deadline, result):
            message = super()._completion(payload, deadline, result)
            call = message["tool_calls"][0]["function"]
            call["arguments"] = '{"hints": {}, "hints": {}, "unresolved": []}'
            return message
    facts = rt.extract(FakeAgent([raw_facts()]), TEXT)
    result = rt.decide(DuplicateKeyAgent([raw_hints()]), TEXT, package(), facts)
    assert result["status"] == "invalid_answer" and "answer" not in result
