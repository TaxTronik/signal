"""Exercise actual bounded orchestration with a scripted local transport."""

from copy import deepcopy
import json
import time

import pytest

from risk_layer.research_agent import ResearchAgent, ResearchConfig, ResearchError


def call(name="finish", arguments=None, call_id="c1"):
    return {"role": "assistant", "content": None, "reasoning_content": "Interne Modellüberlegung.",
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments if arguments is not None else {
                    "findings": [], "unresolved": []})}}]}


def finding(quote_id="e1q1", norm_id="norm:AO:42"):
    return {"focus_id": "f0", "norm_id": norm_id, "quote_id": quote_id,
            "source_quote_id": "f0q1", "reason": "Die Regel erklärt diesen Vorgang."}


class Session:
    focuses = [{"id": "f0", "start": 0, "end": 4, "text": "Text"}]

    def __init__(self):
        self.executed = []
        self.budget = {"used_calls": 0, "max_calls": 8}

    def initial_sources(self, limit):
        return [{"norm_id": "norm:AO:42", "evidence_id": "source1", "text": "Originalnorm"}]

    def execute(self, name, arguments):
        self.executed.append((name, arguments))
        self.budget["used_calls"] += 1
        return {"source": {"evidence_id": "source2", "text": "Gelesener Originaltext"}}

    def validate_finish(self, arguments):
        if arguments != {"findings": [], "unresolved": []}:
            raise ValueError("Norm quote is not a read source")
        return []


class Corpus:
    def __init__(self):
        self.sessions = []

    def new_session(self, text):
        session = Session()
        self.sessions.append(session)
        return session


class CatalogCorpus(Corpus):
    def __init__(self, concepts=None):
        super().__init__()
        self.metadata = {"format": "research_catalog/1", "path": "catalog.yaml", "sha256": "a" * 64,
                         "size_bytes": 100, "version": "test", "semantic_fingerprint": "c" * 64,
                         "policy": {}, "fingerprint": "b" * 64}
        self.hints = {"format": "research_catalog_hints/1", "status": "unverified_hint",
                      "purpose": "Ungeprüfte Suchhinweise aus dem Katalog, keine Belege. "
                                 "Genannte Normen müssen erst über die Quellenwerkzeuge gelesen werden.",
                      "catalog_fingerprint": self.metadata["fingerprint"],
                      "concepts": concepts if concepts is not None else [{
                          "hint_id": "k1", "status": "unverified_hint", "concept_id": "example",
                          "term": "Text", "matches": [{"start": 0, "end": 4, "text": "Text", "methods": ["literal"]}],
                          "norm_anchors": ["§ 42 AO"], "definition_references": [], "norm_chains": [],
                          "omitted": {"matches": 0, "norm_anchors": 0, "definition_references": 0, "norm_chains": 0}}],
                      "total_concepts": 1 if concepts is None else len(concepts), "omitted_concepts": 0}

    def catalog_metadata(self):
        return self.metadata

    def new_session(self, text):
        session = super().new_session(text)
        session.catalog_hints = self.hints
        return session


class SemanticCatalogCorpus(CatalogCorpus):
    def __init__(self):
        super().__init__()
        self.hints["format"] = "research_catalog_hints/2"
        self.metadata["format"] = "research_catalog/2"
        self.metadata["semantic"] = {"query_vector": {
            "dtype": "float32", "dimension": 2, "byte_order": "little", "order": "C"}}
        self.vector = [1., 0.]

    def new_session(self, text):
        session = super().new_session(text)
        session.catalog_query_vector = self.vector
        return session


class ScriptedAgent(ResearchAgent):
    def __init__(self, responses, config=None, prompt_tokens=100, finish_reason="tool_calls"):
        super().__init__("http://127.0.0.1:8089", "test-model", config)
        self.responses = iter(responses)
        self.requests = []
        self.prompt_tokens = prompt_tokens
        self.finish_reason = finish_reason

    def _post(self, path, payload, deadline, result):
        self.requests.append((path, deepcopy(payload)))
        if path == "/apply-template":
            assert payload["tools"] and payload["chat_template_kwargs"]["enable_thinking"] is True
            return {"prompt": "templated input"}
        if path == "/tokenize":
            assert payload == {"content": "templated input", "add_special": False, "parse_special": True}
            return {"tokens": [1] * self.prompt_tokens}
        assert path == "/v1/chat/completions"
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return {"model": self.model, "choices": [{"message": item, "finish_reason": self.finish_reason}],
                "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 100}}


def completions(agent):
    return [payload for path, payload in agent.requests if path == "/v1/chat/completions"]


class CacheAccountingAgent(ScriptedAgent):
    """Simulate measured cache hits, without removing any prompt tokens."""

    def _post(self, path, payload, deadline, result):
        response = super()._post(path, payload, deadline, result)
        if path == "/v1/chat/completions":
            response["usage"]["prompt_tokens_details"] = {
                "cached_tokens": self.prompt_tokens if payload["cache_prompt"] else 0}
        return response


def test_static_finish_is_valid_empty_result_and_has_no_research_tools():
    corpus, agent = Corpus(), ScriptedAgent([call()])
    result = agent.run("Text", corpus, allow_research=False)
    assert result["status"] == "complete" and result["suggestions"] == []
    assert result["errors"] == [] and corpus.sessions[0].executed == []
    request = completions(agent)[0]
    assert [tool["function"]["name"] for tool in request["tools"]] == ["finish"]
    assert request["tool_choice"]["function"]["name"] == "finish"
    assert request["reasoning_budget_tokens"] == 768 and request["max_tokens"] == 2048
    assert request["cache_prompt"] is False and request["parallel_tool_calls"] is False


@pytest.mark.parametrize("allow_research", [False, True])
def test_catalog_hints_are_separate_unverified_input_without_new_sources_or_budget(allow_research):
    base_agent, hint_agent = ScriptedAgent([call()]), ScriptedAgent([call()])
    base_corpus, hint_corpus = Corpus(), CatalogCorpus()
    before = base_agent.run("Text", base_corpus, allow_research)
    after = hint_agent.run("Text", hint_corpus, allow_research)
    assert before["status"] == after["status"] == "complete"
    assert after["catalog_hints"] == hint_corpus.hints
    assert after["catalog_metadata"] == hint_corpus.metadata
    base_request, hint_request = completions(base_agent)[0], completions(hint_agent)[0]
    base_user = json.loads(base_request["messages"][1]["content"])
    hint_user = json.loads(hint_request["messages"][1]["content"])
    assert hint_user.pop("KATALOGHINWEISE") == after["catalog_hints"]
    assert hint_user == base_user
    assert hint_request["messages"][0] == base_request["messages"][0]
    for key in ("tools", "max_tokens", "reasoning_budget_tokens", "tool_choice"):
        assert hint_request[key] == base_request[key]
    for key in ("initial_sources", "initial_sources_sha256", "source_budget", "evidence_aliases",
                "quote_references", "case_quote_references", "config"):
        assert after[key] == before[key]
    assert hint_corpus.sessions[0].executed == []
    assert all(event.get("name") == "finish" for event in after["trace"])


def test_enabled_catalog_without_matches_is_explicitly_visible_and_archived():
    corpus, agent = CatalogCorpus(concepts=[]), ScriptedAgent([call()])
    result = agent.run("Text", corpus)
    user = json.loads(completions(agent)[0]["messages"][1]["content"])
    assert user["KATALOGHINWEISE"] == result["catalog_hints"] == corpus.hints
    assert user["KATALOGHINWEISE"]["concepts"] == []
    assert user["KATALOGHINWEISE"]["status"] == "unverified_hint"
    assert "keine Belege" in user["KATALOGHINWEISE"]["purpose"]


def test_disabled_catalog_does_not_change_original_user_or_result_contract():
    class DisabledCorpus(CatalogCorpus):
        def catalog_metadata(self):
            raise AssertionError("Disabled hints must not fetch metadata")
    corpus = DisabledCorpus()
    corpus.hints = None
    baseline, disabled = ScriptedAgent([call()]), ScriptedAgent([call()])
    before, after = baseline.run("Text", Corpus()), disabled.run("Text", corpus)
    assert after.keys() == before.keys()
    assert "catalog_hints" not in after and "catalog_metadata" not in after
    assert completions(disabled) == completions(baseline)


def test_catalog_snapshots_cannot_be_mutated_by_later_corpus_changes():
    corpus, agent = CatalogCorpus(), ScriptedAgent([call()])
    original_hints, original_metadata = deepcopy(corpus.hints), deepcopy(corpus.metadata)
    result = agent.run("Text", corpus)
    corpus.hints["concepts"].clear()
    corpus.metadata["policy"]["changed_later"] = True
    assert result["catalog_hints"] == original_hints
    assert result["catalog_metadata"] == original_metadata
    assert json.loads(completions(agent)[0]["messages"][1]["content"])["KATALOGHINWEISE"] == original_hints


@pytest.mark.parametrize("reference", ["quote_id", "source_quote_id"])
def test_catalog_reference_cannot_be_used_as_source_or_fact_evidence(reference):
    proposed = finding()
    proposed[reference] = "k1"
    agent = ScriptedAgent([call(arguments={"findings": [proposed], "unresolved": []}, call_id=f"c{i}")
                           for i in range(2)])
    result = agent.run("Text", CatalogCorpus())
    assert result["status"] == "invalid_finish" and result["suggestions"] == []
    assert "k1" not in result["quote_references"] and "k1" not in result["case_quote_references"]
    assert result["source_budget"]["used_calls"] == 0


@pytest.mark.parametrize("change", ["missing_metadata", "fingerprint", "status", "concepts"])
def test_unbound_or_unmarked_catalog_hints_fail_before_any_model_request(change):
    corpus = CatalogCorpus()
    if change == "missing_metadata":
        corpus.metadata = None
    elif change == "fingerprint":
        corpus.metadata["fingerprint"] = "different"
    elif change == "status":
        corpus.hints["status"] = "verified_evidence"
    else:
        corpus.hints["concepts"] = "not a list"
    agent = ScriptedAgent([])
    result = agent.run("Text", corpus)
    assert result["status"] == "source_error" and agent.requests == []


def test_catalog_input_remains_subject_to_the_full_native_prompt_limit():
    agent = ScriptedAgent([], ResearchConfig(context_tokens=2100), prompt_tokens=100)
    result = agent.run("Text", CatalogCorpus())
    assert result["status"] == "context_limit" and completions(agent) == []
    assert "KATALOGHINWEISE" in agent.requests[0][1]["messages"][1]["content"]


def test_shared_catalog_vector_is_copied_only_to_private_result_not_model_input():
    corpus, agent = SemanticCatalogCorpus(), ScriptedAgent([call()])
    result = agent.run("Text", corpus)
    assert result["status"] == "complete" and result["catalog_query_vector"] == [1., 0.]
    user = json.loads(completions(agent)[0]["messages"][1]["content"])
    assert user["KATALOGHINWEISE"] == corpus.hints
    assert "catalog_query_vector" not in user and "semantic" not in user["KATALOGHINWEISE"]
    assert "query_vector" not in completions(agent)[0]["messages"][1]["content"]
    corpus.vector[0] = 0.
    assert result["catalog_query_vector"] == [1., 0.]
    assert result["source_budget"]["used_calls"] == 0


@pytest.mark.parametrize("version", [1, 2])
def test_literal_catalog_has_no_private_vector_field(version):
    corpus, agent = CatalogCorpus(), ScriptedAgent([call()])
    corpus.hints["format"] = f"research_catalog_hints/{version}"
    corpus.metadata["format"] = f"research_catalog/{version}"
    if version == 2:
        corpus.metadata["semantic"] = None
    result = agent.run("Text", corpus)
    assert result["status"] == "complete" and "catalog_query_vector" not in result


@pytest.mark.parametrize("vector", [None, [1, 0], [True, 0.], [1.], [1., 0., 0.],
                                    [float("nan"), 0.], [float("inf"), 0.], [2., 0.],
                                    [1., 1e-300], [1e300, 0.]])
def test_invalid_semantic_catalog_vector_stops_before_model_or_partial_catalog_ledger(vector):
    corpus, agent = SemanticCatalogCorpus(), ScriptedAgent([])
    corpus.vector = vector
    result = agent.run("Text", corpus)
    assert result["status"] == "source_error" and agent.requests == []
    assert "catalog_query_vector" not in result and "catalog_hints" not in result
    assert "catalog_metadata" not in result


@pytest.mark.parametrize(("field", "value"), [("dimension", True), ("dimension", 2.),
                                              ("dimension", 3), ("dtype", "float16"),
                                              ("byte_order", "big"), ("order", "F")])
def test_catalog_vector_specification_is_strictly_bound(field, value):
    corpus, agent = SemanticCatalogCorpus(), ScriptedAgent([])
    corpus.metadata["semantic"]["query_vector"][field] = value
    result = agent.run("Text", corpus)
    assert result["status"] == "source_error" and agent.requests == []


@pytest.mark.parametrize("value", [0, 1, None, "true", [], {}])
def test_prompt_cache_flag_requires_an_actual_boolean(value):
    with pytest.raises(ValueError, match="reuse_prompt_cache"):
        ResearchConfig(reuse_prompt_cache=value)


@pytest.mark.parametrize("value", [True, False, None, "30", -1, float("nan"),
                                  float("inf"), float("-inf")])
def test_finish_reserve_requires_a_finite_nonnegative_number(value):
    with pytest.raises(ValueError):
        ResearchConfig(finish_reserve_seconds=value)


@pytest.mark.parametrize("value", [0, 0.0, 0.5, 200])
def test_finish_reserve_accepts_finite_nonnegative_seconds(value):
    assert ResearchConfig(finish_reserve_seconds=value).finish_reserve_seconds == value


def test_cache_policy_changes_only_the_cache_flag_and_preserves_complete_history():
    responses = [call("search_sources", {"query": "eigene Suchbegriffe"}), call(call_id="c2")]
    default = CacheAccountingAgent(deepcopy(responses))
    cached = CacheAccountingAgent(deepcopy(responses), ResearchConfig(reuse_prompt_cache=True))
    assert default.run("Text", Corpus())["status"] == "complete"
    assert cached.run("Text", Corpus())["status"] == "complete"
    assert [request["cache_prompt"] for request in completions(default)] == [False, False]
    assert [request["cache_prompt"] for request in completions(cached)] == [False, True]
    for before, after in zip(completions(default), completions(cached), strict=True):
        assert {key: value for key, value in before.items() if key != "cache_prompt"} == {
            key: value for key, value in after.items() if key != "cache_prompt"}
    continuation = completions(cached)[1]
    assert len(continuation["messages"]) == 4
    assert continuation["messages"][-1]["role"] == "tool"
    assert continuation["messages"][-2]["reasoning_content"] == responses[0]["reasoning_content"]


def test_cache_is_cold_again_for_each_case_and_static_arm_on_the_same_agent():
    responses = [call("search_sources", {"query": "Suche"}), call(call_id="c2")]
    agent = CacheAccountingAgent(deepcopy(responses * 2) + [call()], ResearchConfig(reuse_prompt_cache=True))
    corpus = Corpus()
    for text, allow in (("Text", True), ("Anderer Text", True), ("Text", False)):
        assert agent.run(text, corpus, allow_research=allow)["status"] == "complete"
    assert len(corpus.sessions) == 3
    assert [request["cache_prompt"] for request in completions(agent)] == [False, True, False, True, False]


def test_repair_after_prose_reuses_the_current_run_cache():
    agent = CacheAccountingAgent([{"role": "assistant", "content": "Keine Vorschläge."},
                                  call(call_id="c2")], ResearchConfig(reuse_prompt_cache=True))
    assert agent.run("Text", Corpus())["status"] == "complete"
    assert [request["cache_prompt"] for request in completions(agent)] == [False, True]
    assert completions(agent)[1]["tool_choice"]["function"]["name"] == "finish"


@pytest.mark.parametrize("details", [None, [], {}, {"cached_tokens": True}, {"cached_tokens": -1},
                                    {"cached_tokens": 101}, {"cached_tokens": 0.0},
                                    {"cached_tokens": "0"}, {"cached_tokens": 1}])
def test_cache_enabled_rejects_missing_invalid_or_nonzero_cold_accounting(details):
    class ChangedAccounting(CacheAccountingAgent):
        def _post(self, path, payload, deadline, result):
            response = super()._post(path, payload, deadline, result)
            if path == "/v1/chat/completions":
                response["usage"]["prompt_tokens_details"] = details
            return response
    agent = ChangedAccounting([call()], ResearchConfig(reuse_prompt_cache=True))
    result = agent.run("Text", Corpus())
    assert result["status"] == "invalid_completion" and result["suggestions"] == []
    assert "cache accounting" in result["errors"][-1]["message"]


def test_cache_enabled_requires_accounting_on_a_continuation_too():
    class MissingContinuationAccounting(CacheAccountingAgent):
        def _post(self, path, payload, deadline, result):
            response = super()._post(path, payload, deadline, result)
            if path == "/v1/chat/completions" and payload["cache_prompt"]:
                del response["usage"]["prompt_tokens_details"]
            return response
    agent = MissingContinuationAccounting([call("search_sources", {"query": "Suche"}),
                                          call(call_id="c2")], ResearchConfig(reuse_prompt_cache=True))
    assert agent.run("Text", Corpus())["status"] == "invalid_completion"


def test_cache_enabled_allows_a_continuation_with_zero_actual_cache_hits():
    class NoHits(CacheAccountingAgent):
        def _post(self, path, payload, deadline, result):
            response = super()._post(path, payload, deadline, result)
            if path == "/v1/chat/completions":
                response["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
            return response
    agent = NoHits([call("search_sources", {"query": "Suche"}), call(call_id="c2")],
                   ResearchConfig(reuse_prompt_cache=True))
    assert agent.run("Text", Corpus())["status"] == "complete"


def test_cache_does_not_replace_full_prompt_accounting_with_uncached_tokens():
    class UncachedOnly(CacheAccountingAgent):
        def _post(self, path, payload, deadline, result):
            response = super()._post(path, payload, deadline, result)
            if path == "/v1/chat/completions" and payload["cache_prompt"]:
                response["usage"]["prompt_tokens"] = 0
            return response
    agent = UncachedOnly([call("search_sources", {"query": "Suche"}), call(call_id="c2")],
                         ResearchConfig(reuse_prompt_cache=True))
    result = agent.run("Text", Corpus())
    assert result["status"] == "invalid_completion"
    assert "native template" in result["errors"][-1]["message"]


def test_cached_continuation_still_counts_the_entire_prompt_before_generation():
    class GrowingPrompt(CacheAccountingAgent):
        def _post(self, path, payload, deadline, result):
            response = super()._post(path, payload, deadline, result)
            if path == "/v1/chat/completions":
                self.prompt_tokens = 6000
            return response
    agent = GrowingPrompt([call("search_sources", {"query": "Suche"})],
                           ResearchConfig(reuse_prompt_cache=True, context_tokens=7000))
    result = agent.run("Text", Corpus())
    assert result["status"] == "context_limit"
    assert len(completions(agent)) == 1
    assert result["timing"]["max_prompt_tokens"] == 6000
    assert any(payload.get("cache_prompt") is True for path, payload in agent.requests
               if path == "/apply-template")


@pytest.mark.parametrize(("seconds", "reserve", "forced", "reserve_trace"), [
    (400., 200., True, True), (400.01, 200., False, False),
    (30., 0., False, False), (29.99, 0., True, False),
    (29.99, 1., True, False), (180., 0., False, False),
])
def test_time_admission_uses_inclusive_double_reserve_and_preserves_old_thirty_seconds(
        monkeypatch, seconds, reserve, forced, reserve_trace):
    monkeypatch.setattr("risk_layer.research_agent.time.monotonic", lambda: 0.)
    agent = ScriptedAgent([call()], ResearchConfig(timeout_seconds=seconds, finish_reserve_seconds=reserve))
    result = agent.run("Text", Corpus())
    assert result["status"] == "complete"
    request = completions(agent)[0]
    assert (request["tool_choice"] != "auto") is forced
    events = [event for event in result["trace"] if event.get("reason") == "time_finish_reserve"]
    assert bool(events) is reserve_trace
    if reserve_trace:
        assert events == [{"kind": "budget", "reason": "time_finish_reserve", "round": 0,
                           "remaining_seconds": seconds, "finish_reserve_seconds": reserve}]


def test_time_reserve_after_research_requires_original_evidence_validation(monkeypatch):
    now, validated = [0.], []
    monkeypatch.setattr("risk_layer.research_agent.time.monotonic", lambda: now[0])

    def validate(session, arguments):
        validated.append(deepcopy(arguments))
        assert arguments["findings"][0]["norm_quote"] == "Originalnorm"
        assert arguments["findings"][0]["evidence_id"] == "source1"
        return [{"normen": [{"id": "norm:AO:42"}]}]

    monkeypatch.setattr(Session, "validate_finish", validate)

    class TimedAgent(ScriptedAgent):
        def _post(self, path, payload, deadline, result):
            response = super()._post(path, payload, deadline, result)
            if path == "/v1/chat/completions":
                now[0] += 200.
            return response

    agent = TimedAgent([call("search_sources", {"query": "Suche"}),
                        call(arguments={"findings": [finding()], "unresolved": []}, call_id="c2")],
                       ResearchConfig(timeout_seconds=600., finish_reserve_seconds=200.))
    result = agent.run("Text", Corpus())
    assert result["status"] == "complete" and result["suggestions"] == [{"normen": [{"id": "norm:AO:42"}]}]
    assert len(validated) == 1
    assert completions(agent)[0]["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in completions(agent)[1]["tools"]] == ["finish"]
    assert result["trace"][1] == {"kind": "budget", "reason": "time_finish_reserve", "round": 1,
                                  "remaining_seconds": 400., "finish_reserve_seconds": 200.}


@pytest.mark.parametrize("failure", ["transport", "late_completion", "late_validation", "invalid_finish"])
def test_time_reserve_never_fabricates_an_empty_success(monkeypatch, failure):
    now = [0.]
    monkeypatch.setattr("risk_layer.research_agent.time.monotonic", lambda: now[0])
    if failure == "late_validation":
        def validate(session, arguments):
            now[0] = 101.
            return [{"normen": [{"id": "norm:AO:42"}]}]
        monkeypatch.setattr(Session, "validate_finish", validate)

    class FailingAgent(ScriptedAgent):
        def _post(self, path, payload, deadline, result):
            response = super()._post(path, payload, deadline, result)
            if path == "/v1/chat/completions" and failure == "late_completion":
                now[0] = 101.
            return response

    if failure == "transport":
        responses = [ResearchError("timeout", "No response within deadline")]
    elif failure == "invalid_finish":
        responses = [call(arguments={"findings": [finding("unknown")], "unresolved": []},
                          call_id=f"c{number}") for number in range(2)]
    else:
        responses = [call()]
    agent = FailingAgent(responses, ResearchConfig(timeout_seconds=100., finish_reserve_seconds=50.))
    result = agent.run("Text", Corpus())
    assert result["status"] == ("invalid_finish" if failure == "invalid_finish" else "timeout")
    assert result["suggestions"] == [] and result["errors"]
    assert all(not event.get("result", {}).get("accepted") for event in result["trace"])


def test_research_executes_model_selected_query_and_returns_source_to_model():
    corpus = Corpus()
    agent = ScriptedAgent([call("search_sources", {"query": "eigene neue Suchbegriffe"}), call(call_id="c2")])
    result = agent.run("Text", corpus)
    assert result["status"] == "complete"
    assert corpus.sessions[0].executed == [("search_sources", {"query": "eigene neue Suchbegriffe"})]
    request = completions(agent)[1]
    assert request["messages"][-1]["role"] == "tool"
    assert "Gelesener Originaltext" in request["messages"][-1]["content"]
    assert request["messages"][-2]["reasoning_content"] == "Interne Modellüberlegung."
    assert result["usage"]["model_calls"] == 2 and result["usage"]["completion_tokens"] == 200


def test_modes_use_identical_start_sources_and_new_sessions():
    corpus, agent = Corpus(), ScriptedAgent([call(), call()])
    static = agent.run("Text", corpus, False)
    research = agent.run("Text", corpus, True)
    assert len(corpus.sessions) == 2 and corpus.sessions[0] is not corpus.sessions[1]
    assert static["initial_sources"] == research["initial_sources"]
    assert static["initial_sources_sha256"] == research["initial_sources_sha256"]
    assert static["focuses"] == research["focuses"]


def test_missing_finish_is_a_failure_not_valid_abstention():
    result = ScriptedAgent([{"role": "assistant", "content": "Keine Vorschläge."}] * 2).run("Text", Corpus())
    assert result["status"] == "invalid_completion" and result["errors"]


def test_prose_gets_one_explicit_finish_repair_without_becoming_an_empty_success():
    agent = ScriptedAgent([{"role": "assistant", "content": "Keine Vorschläge."}, call(call_id="c2")])
    result = agent.run("Text", Corpus())
    assert result["status"] == "complete"
    assert result["errors"][0]["kind"] == "missing_tool_call"
    assert completions(agent)[1]["tool_choice"]["function"]["name"] == "finish"


def test_model_sources_have_short_references_but_original_trace_identity_survives():
    from risk_layer.research_agent import _ModelSources
    model = _ModelSources()
    model.focus_view(Session.focuses)
    source = {"evidence_id": "ev:longhash", "norm_id": "norm:AO:42", "text": "Originalnorm",
              "text_sha256": "secret-long-digest", "source_fingerprint": "corpus-fingerprint",
              "source_start": 50, "source_end": 62, "next_cursor": 62,
              "requested_id": "norm:AO:42:abs2", "fallback": True}
    initial = deepcopy(source)
    view = model.view(source)
    assert source == initial
    assert view["evidence_id"] == "e1" and view["quotes"] == [{"quote_id": "e1q1", "text": source["text"]}]
    assert "text_sha256" not in view and "source_fingerprint" not in view
    assert view["requested_id"] == "norm:AO:42:abs2" and view["fallback"] is True
    again = model.view(source)
    assert again["evidence_id"] == "e1" and again["already_supplied"] is True and "quotes" not in again
    arguments = {"findings": [finding()], "unresolved": []}
    assert model.expand_finish(arguments)["findings"][0]["evidence_id"] == "ev:longhash"
    assert model.expand_finish(arguments)["findings"][0]["norm_quote"] == "Originalnorm"
    assert arguments["findings"][0]["quote_id"] == "e1q1"
    with pytest.raises(ValueError, match="Unknown quote_id"):
        model.expand_finish({"findings": [finding("ev:longhash")]})
    with pytest.raises(ValueError, match="Unknown quote_id"):
        model.expand_finish({"findings": [finding("e2q1")]})
    with pytest.raises(ValueError, match="different norm_id"):
        model.expand_finish({"findings": [finding(norm_id="norm:AO:43")]})


def test_short_evidence_reference_is_expanded_only_before_original_source_validation():
    class ValidatingSession(Session):
        def validate_finish(self, arguments):
            assert arguments["findings"][0]["evidence_id"] == "source1"
            assert arguments["findings"][0]["norm_quote"] == "Originalnorm"
            return []
    class ValidatingCorpus:
        def new_session(self, text):
            return ValidatingSession()
    agent = ScriptedAgent([call(arguments={"findings": [finding()], "unresolved": []})])
    result = agent.run("Text", ValidatingCorpus())
    assert result["status"] == "complete" and result["evidence_aliases"] == {"e1": "source1"}
    assert result["initial_sources"][0]["evidence_id"] == "source1"
    assert result["trace"][-1]["validated_arguments"]["findings"][0]["evidence_id"] == "source1"
    assert "source1" not in completions(agent)[0]["messages"][1]["content"]


@pytest.mark.parametrize("text", ["Originalnorm", "Ein unveränderter Originaltext.\n" * 150,
                                  "Sonderzeichen (§ 2) – bleiben\tgenau erhalten. " * 40, "a" * 900])
def test_selectable_quote_units_cover_every_original_character_and_fit_the_validator(text):
    from risk_layer.research_agent import _ModelSources
    model = _ModelSources()
    model.focus_view(Session.focuses)
    source = {"evidence_id": "source", "norm_id": "norm:AO:42", "text": text}
    view = model.view(source)
    assert "".join(item["text"] for item in view["quotes"]) == text
    assert all(8 <= len(item["text"]) <= 240 for item in view["quotes"])
    for item in view["quotes"]:
        expanded = model.expand_finish({"findings": [finding(item["quote_id"])]})["findings"][0]
        assert expanded["norm_quote"] == item["text"] and expanded["norm_quote"] in text


def test_exhausted_individual_source_tool_is_removed_from_native_tool_choices():
    class ExhaustedSession(Session):
        def __init__(self):
            super().__init__()
            self.budget.update(per_tool={"search_sources": 3}, max_per_tool={"search_sources": 3})
    class ExhaustedCorpus:
        def new_session(self, text):
            return ExhaustedSession()
    agent = ScriptedAgent([call()])
    assert agent.run("Text", ExhaustedCorpus())["status"] == "complete"
    assert "search_sources" not in [tool["function"]["name"] for tool in completions(agent)[0]["tools"]]


def test_context_growth_forces_finish_early_with_all_existing_sources_intact():
    agent = ScriptedAgent([call()], prompt_tokens=11000)
    result = agent.run("Text", Corpus())
    assert result["status"] == "complete"
    assert completions(agent)[0]["tool_choice"]["function"]["name"] == "finish"
    assert "Originalnorm" in completions(agent)[0]["messages"][1]["content"]
    assert any(event.get("reason") == "context_finish_reserve" for event in result["trace"])


def test_case_reference_cannot_invent_or_rewrite_a_fact():
    from risk_layer.research_agent import _ModelSources
    model = _ModelSources()
    original = "Originaler Sachverhalt bleibt erhalten. " * 20
    focuses = [{"id": "f0", "start": 7, "end": 7 + len(original), "text": original}]
    view = model.focus_view(focuses)
    assert "".join(item["text"] for item in view[0]["quotes"]) == original
    assert model.case_quotes["f0q2"]["start"] > 7
    model.view({"evidence_id": "real", "norm_id": "norm:AO:42", "text": "Originale Normregelung"})
    chosen = {**finding(), "source_quote_id": "f0q2"}
    expanded = model.expand_finish({"findings": [chosen]})["findings"][0]
    assert expanded["source_quote"] == model.case_quotes["f0q2"]["text"]
    assert expanded["source_quote_start"] == model.case_quotes["f0q2"]["start"]
    with pytest.raises(ValueError, match="Unknown source_quote_id"):
        model.expand_finish({"findings": [{**chosen, "source_quote_id": "fabricated"}]})


def test_duplicate_norms_keep_first_evidence_and_archive_every_validated_extra():
    from risk_layer.research_agent import _ModelSources
    model = _ModelSources()
    model.focus_view(Session.focuses)
    model.view({"evidence_id": "real", "norm_id": "norm:AO:42", "text": "Originale Normregelung. " * 30})
    first, second = finding("e1q1"), finding("e1q2")
    expanded = model.expand_finish({"findings": [first, second]})["findings"]
    assert len(expanded) == 1 and expanded[0]["norm_quote"] == model.quotes["e1q1"]["text"]
    assert model.merged_duplicates == [second]
    with pytest.raises(ValueError, match="Unknown quote_id"):
        model.expand_finish({"findings": [first, finding("invented")]})


def test_request_trace_keeps_exact_sent_body_after_mutating_later_payload():
    import hashlib
    import io
    from types import SimpleNamespace
    from risk_layer.research_agent import _json
    agent = ResearchAgent("http://127.0.0.1:8089", "test")
    agent._opener = SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(b'{"prompt":"rendered"}'))
    payload = {"tools": [{"name": "source"}], "messages": [{"content": "original"}]}
    result = {"trace": [], "usage": {"http_requests": 0}, "timing": {"generation_seconds": 0.}}
    agent._post("/apply-template", payload, time.monotonic() + 10, result)
    payload["tools"] = [{"name": "finish"}]
    payload["messages"][0]["content"] = "changed"
    event = result["trace"][0]
    assert event["request"]["tools"] == [{"name": "source"}]
    assert event["request"]["messages"][0]["content"] == "original"
    assert hashlib.sha256(_json(event["request"]).encode()).hexdigest() == event["request_sha256"]


def test_output_truncation_never_uses_apparently_complete_finish():
    result = ScriptedAgent([call()], finish_reason="length").run("Text", Corpus())
    assert result["status"] == "output_limit"


def test_unread_quote_rejected_and_repair_is_explicitly_recorded():
    agent = ScriptedAgent([call(arguments={"findings": [finding("not-supplied")], "unresolved": []}),
                           call(call_id="c2")])
    result = agent.run("Text", Corpus())
    assert result["status"] == "complete"
    assert len(result["errors"]) == 1 and result["errors"][0]["recoverable"] is True
    assert completions(agent)[1]["tool_choice"]["function"]["name"] == "finish"
    assert "Unknown quote_id" in completions(agent)[1]["messages"][-1]["content"]


def test_two_invalid_finishes_are_a_failure():
    invalid = {"findings": [{"norm_quote": "erfunden"}], "unresolved": []}
    result = ScriptedAgent([call(arguments=invalid), call(arguments=invalid, call_id="c2")]).run("Text", Corpus())
    assert result["status"] == "invalid_finish"


@pytest.mark.parametrize("name,mode", [("delete_file", True), ("read_source", False)])
def test_unknown_or_static_forbidden_tools_never_execute(name, mode):
    corpus = Corpus()
    result = ScriptedAgent([call(name, {"norm_id": "norm:AO:42"})]).run("Text", corpus, mode)
    assert result["status"] == "invalid_completion" and corpus.sessions[0].executed == []


def test_repeated_tool_id_rejected_before_execution():
    corpus = Corpus()
    result = ScriptedAgent([call("read_source", {"norm_id": "norm:AO:42"}), call()]).run("Text", corpus)
    assert result["status"] == "invalid_completion" and len(corpus.sessions[0].executed) == 1


def test_tool_cap_forces_finish_without_discarding_sources():
    config = ResearchConfig(max_tool_calls=1)
    agent = ScriptedAgent([call("read_source", {"norm_id": "norm:AO:42"}), call(call_id="c2")], config)
    result = agent.run("Text", Corpus())
    assert result["status"] == "complete"
    assert completions(agent)[1]["tool_choice"]["function"]["name"] == "finish"


def test_context_limit_is_measured_before_inference_with_answer_reserve():
    config = ResearchConfig(context_tokens=2100)
    agent = ScriptedAgent([call()], config, prompt_tokens=100)
    result = agent.run("Text", Corpus())
    assert result["status"] == "context_limit" and completions(agent) == []


def test_total_output_budget_forces_final_call():
    config = ResearchConfig(total_output_tokens=2048)
    agent = ScriptedAgent([call()], config)
    assert agent.run("Text", Corpus())["status"] == "complete"
    assert completions(agent)[0]["tool_choice"]["function"]["name"] == "finish"


def test_expired_deadline_never_attempts_http():
    agent = ResearchAgent("http://localhost:8089/v1", "test")
    with pytest.raises(ResearchError) as exc:
        agent._post("/tokenize", {}, time.monotonic() - 1, {})
    assert exc.value.status == "timeout"


@pytest.mark.parametrize("url", ["https://example.com", "http://example.com", "http://127.0.0.1:8089/other",
                                 "http://name:password@localhost:8089", "http://localhost:8089?proxy=x"])
def test_transport_is_loopback_only_without_url_credentials_or_redirect_paths(url):
    with pytest.raises(ValueError):
        ResearchAgent(url, "test")


@pytest.mark.parametrize("kwargs", [{"reasoning_budget_tokens": 0}, {"reasoning_budget_tokens": 2048},
                                    {"max_tool_calls": 9}, {"initial_sources": 7}, {"timeout_seconds": float("nan")},
                                    {"max_rounds": True}, {"total_output_tokens": 1024}, {"temperature": float("inf")}])
def test_invalid_budget_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ResearchConfig(**kwargs)


def test_transport_failure_keeps_explicit_failure_and_empty_technical_result():
    agent = ScriptedAgent([ResearchError("transport_error", "service stopped")])
    result = agent.run("Text", Corpus())
    assert result["status"] == "transport_error"
    assert result["errors"][-1]["recoverable"] is False


@pytest.mark.parametrize("tamper", ["model", "prompt_tokens", "truncated", "completion_tokens"])
def test_server_identity_and_actual_accounting_must_match_native_preflight(tamper):
    class ChangedServer(ScriptedAgent):
        def _post(self, path, payload, deadline, result):
            value = super()._post(path, payload, deadline, result)
            if path == "/v1/chat/completions":
                if tamper == "model":
                    value["model"] = "other-model"
                elif tamper == "prompt_tokens":
                    value["usage"]["prompt_tokens"] += 1
                elif tamper == "completion_tokens":
                    value["usage"]["completion_tokens"] = payload["max_tokens"] + 1
                else:
                    value["truncated"] = True
            return value
    result = ChangedServer([call()]).run("Text", Corpus())
    assert result["status"] == "invalid_completion"


def bounded_page(start, end, length):
    return {"norm_id": "norm:TEST:1", "evidence_id": "test-page", "text": "x" * (end - start),
            "source_start": start, "source_end": end, "source_length": length,
            "next_cursor": end if end < length else None,
            "complete": start == 0 and end == length}


@pytest.mark.parametrize("start,end,length,cursor", [
    (0, 100, 100, None), (0, 100, 10000, 0), (0, 4000, 10000, None),
    (100, 200, 1000, 0), (4900, 5100, 10000, 3000),
    (9900, 10000, 10000, 6000), (200, 1000, 1000, 0),
    (6000, 10000, 10000, None), (10, 4011, 9000, None),
    (0, 1, 1, None), (1, 2, 100, 0), (0, 3999, 4000, 0),
    (2000, 2001, 10000, 1), (5000, 5001, 6000, 2000),
])
def test_source_context_is_bounded_contains_the_page_and_never_promises_a_noop(start, end, length, cursor):
    from risk_layer.research_agent import _ModelSources
    original = bounded_page(start, end, length)
    before = deepcopy(original)
    view = _ModelSources().view(original)
    assert original == before
    assert view["complete"] is original["complete"]
    assert view["next_cursor"] == original["next_cursor"]
    assert view["context_cursor"] == cursor
    assert "".join(q["text"] for q in view["quotes"]) == original["text"]
    if cursor is not None:
        context_end = min(length, cursor + 4000)
        assert 0 <= cursor <= start < end <= context_end <= length
        assert context_end - cursor <= 4000
        assert cursor < start or context_end > end


@pytest.mark.parametrize("field,value", [
    ("source_start", True), ("source_start", -1), ("source_start", 101),
    ("source_end", 100), ("source_end", 201),
    ("source_length", 199), ("source_length", 1000.),
    ("source_length", float("nan")), ("complete", 0), ("complete", "false"),
    ("complete", True),
])
def test_context_metadata_rejects_invalid_bounds_before_registering_a_source(field, value):
    from risk_layer.research_agent import _ModelSources
    model = _ModelSources()
    source = bounded_page(100, 200, 1000)
    source[field] = value
    with pytest.raises(ValueError):
        model.view(source)
    assert model.aliases == {} and model.quotes == {}


def test_incomplete_legacy_metadata_does_not_invent_extent_or_completeness():
    from risk_layer.research_agent import _ModelSources
    source = {"evidence_id": "legacy", "text": "Original source text", "source_start": 90,
              "source_end": 110, "next_cursor": 110}
    view = _ModelSources().view(source)
    assert "complete" not in view and "context_cursor" not in view
    assert view["next_cursor"] == 110


def test_repeated_page_metadata_describes_the_page_and_does_not_assert_global_unread_text():
    from risk_layer.research_agent import _ModelSources
    model = _ModelSources()
    source = bounded_page(4900, 5100, 10000)
    initial = model.view(source)
    quotes = deepcopy(model.quotes)
    repeated = model.view(source)
    assert repeated["already_supplied"] is True and "quotes" not in repeated
    assert repeated["complete"] is False and repeated["context_cursor"] == 3000
    assert repeated["next_cursor"] == initial["next_cursor"] == 5100
    assert model.quotes == quotes
    assert set(initial) == {"norm_id", "source_start", "source_end", "source_length", "next_cursor",
                            "complete", "context_cursor", "evidence_id", "quotes"}


def test_context_handoff_versions_the_tool_view_without_running_an_automatic_read():
    import risk_layer.research_agent as module
    class ExactSession(Session):
        def initial_sources(self, limit):
            return [bounded_page(4900, 5100, 10000)]
    class ExactCorpus(Corpus):
        def new_session(self, text):
            self.sessions.append(ExactSession())
            return self.sessions[-1]
    corpus = ExactCorpus()
    agent = ScriptedAgent([call()])
    result = agent.run("Text", corpus)
    assert result["status"] == "complete" and result["prompt_version"] == "signal/local-research/v6"
    assert corpus.sessions[0].executed == [] and result["source_budget"]["used_calls"] == 0
    assert result["initial_sources"] == [bounded_page(4900, 5100, 10000)]
    assert result["initial_sources_sha256"] == module._sha(module._json(result["initial_sources"]))
    request = completions(agent)[0]
    source = json.loads(request["messages"][1]["content"])["STARTQUELLEN"][0]
    assert source["context_cursor"] == 3000 and source["complete"] is False
    read = next(t["function"] for t in request["tools"] if t["function"]["name"] == "read_source")
    assert "context_cursor" in read["description"] and "4000" in read["description"]
    assert set(read["parameters"]["properties"]) == {"norm_id", "cursor"}
    assert request["max_tokens"] == 2048 and result["usage"]["model_calls"] == 1


def test_context_cursor_uses_the_actual_unchanged_reader_and_only_then_registers_evidence(tmp_path):
    from risk_layer.embedding import EmbeddingIndex, HashEncoder
    from risk_layer.gii import NormKnoten
    from risk_layer.graph import NormGraph
    from risk_layer.research_agent import _ModelSources
    from risk_layer.research_sources import ResearchCorpus
    text = "Vorheriger Originalkontext. " * 200 + "Die genaue Regel folgt. " * 100
    graph_path, index_path = tmp_path / "graph.sqlite", tmp_path / "index"
    graph = NormGraph.build([NormKnoten(id="norm:TEST:1", law="TEST", para="1", titel="Regel", text=text)], graph_path)
    try:
        EmbeddingIndex.build(graph, HashEncoder(dim=16), passagen=True).save(index_path)
    finally:
        graph.close()
    corpus = ResearchCorpus(graph_path, index_path)
    try:
        session = corpus.new_session("Die genaue Regel folgt.")
        initial = session.initial_sources(limit=1)[0]
        before = deepcopy(session.budget)
        view = _ModelSources().view(initial)
        assert session.budget == before and before["used_calls"] == 0
        cursor = view["context_cursor"]
        assert type(cursor) is int
        output = session.execute("read_source", {"norm_id": view["norm_id"], "cursor": cursor})
        page = output["source"]
        assert page["source_start"] == cursor
        assert page["source_end"] == min(len(text), cursor + 4000)
        assert page["text"] == text[page["source_start"]:page["source_end"]]
        assert page["source_start"] <= initial["source_start"] < initial["source_end"] <= page["source_end"]
        assert page["evidence_id"] != initial["evidence_id"]
        assert session.budget["used_calls"] == 1 and session.budget["per_tool"]["read_source"] == 1
    finally:
        corpus.close()
