"""Local JSON transport and quote provenance do not establish model accuracy."""

import hashlib
import json
from types import SimpleNamespace
import urllib.error

import pytest

from risk_layer.norm_check import LocalNormChecker, NormCheckError, RESPONSE_SCHEMA, SYSTEM_PROMPT
from risk_layer.passages import Passage
from risk_layer.semantic_context import build_context_queries


def example():
    context, = build_context_queries("Die Belege fehlen vollständig.", mode="paragraph")
    passage = Passage("norm:AO:1@10:51", "norm:AO:1", "AO", "1", "Titel", 10, 51,
                      "Die Belege müssen vollständig vorliegen.")
    # Offsets use the actual Unicode character count, independently of an ID.
    passage = Passage(passage.passage_id, passage.norm_id, passage.law, passage.para,
                      passage.title, 10, 10 + len(passage.text), passage.text)
    answer = {"status": "passend", "source_quote": "Belege fehlen vollständig",
              "norm_quote": "Belege müssen vollständig vorliegen", "begruendung": "Konkrete Pflichtbeschreibung."}
    return context, passage, answer


def fake_client(answer, *, finish="stop", envelope=None):
    client = LocalNormChecker()
    calls = []
    raw = json.dumps(envelope if envelope is not None else {
        "model": "granite-test-alias", "usage": {"prompt_tokens": 80, "completion_tokens": 42},
        "choices": [{"finish_reason": finish, "message": {"content": json.dumps(answer, ensure_ascii=False)}}]
    }, ensure_ascii=False).encode("utf-8")
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, limit):
            return raw[:limit]
    def open_response(request, timeout):
        calls.append((request, timeout))
        return Response()
    client._opener = SimpleNamespace(open=open_response)
    return client, calls, raw


def test_valid_positive_preserves_exact_source_and_norm_offsets_and_fixed_request():
    context, passage, answer = example()
    client, calls, raw = fake_client(answer)
    result = client.check(context, passage.norm_id, passage.title, [passage])
    assert result["status"] == "passend" and result["quotes_grounded"]
    assert context.query_text[result["source_evidence"]["start"]:result["source_evidence"]["end"]] == answer["source_quote"]
    evidence = result["norm_evidence"]
    assert passage.text[evidence["start"] - passage.start:evidence["end"] - passage.start] == answer["norm_quote"]
    request, timeout = calls[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert timeout == 60 and payload["temperature"] == payload["seed"] == 0
    assert payload["max_tokens"] == 256 and not payload["stream"]
    assert payload["response_format"]["json_schema"]["schema"] == RESPONSE_SCHEMA
    assert payload["messages"][0]["content"] == SYSTEM_PROMPT
    assert "FOKUS" in payload["messages"][1]["content"]
    assert result["request_sha256"] == hashlib.sha256(request.data).hexdigest()
    assert result["raw_http_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["raw_response_sha256"] == hashlib.sha256(result["raw_response"].encode()).hexdigest()
    assert result["kalibriert"] is False
    assert result["model_requested"] == "granite-4.1-8b-Q5_K_M.gguf"
    assert result["model_response"] == "granite-test-alias" and result["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 80, "completion_tokens": 42}


@pytest.mark.parametrize(("field", "quote"), [
    ("source_quote", ""), ("norm_quote", ""), ("source_quote", "   "),
    ("norm_quote", "Titel"), ("norm_quote", "Erfundene Katalogparaphrase"),
    ("source_quote", "Belege  fehlen vollständig"), ("source_quote", "belege fehlen vollständig"),
])
def test_positive_without_two_exact_nonempty_source_quotes_is_unclear(field, quote):
    context, passage, answer = example()
    answer[field] = quote
    client, _, _ = fake_client(answer)
    result = client.check(context, passage.norm_id, passage.title, [passage])
    assert result["status"] == "unklar" and result["model_status"] == "passend"
    assert not result["quotes_grounded"]


@pytest.mark.parametrize("status", ["unklar", "widerspruch"])
def test_nonpositive_answers_need_no_forced_quote(status):
    context, passage, _ = example()
    client, _, _ = fake_client({"status": status, "source_quote": "", "norm_quote": "", "begruendung": "Information fehlt."})
    result = client.check(context, passage.norm_id, passage.title, [passage])
    assert result["status"] == status and not result["quotes_grounded"]


@pytest.mark.parametrize("answer", [
    {}, [], None, {"status": "passend"},
    {"status": "sicher", "source_quote": "", "norm_quote": "", "begruendung": ""},
    {"status": "unklar", "source_quote": None, "norm_quote": "", "begruendung": ""},
    {"status": "unklar", "source_quote": "", "norm_quote": "", "begruendung": "", "confidence": 1},
])
def test_schema_failures_are_backend_errors_not_abstention(answer):
    context, passage, _ = example()
    client, _, _ = fake_client(answer)
    with pytest.raises(NormCheckError, match="Schema"):
        client.check(context, passage.norm_id, passage.title, [passage])


@pytest.mark.parametrize("finish", ["length", "error", None])
def test_incomplete_generation_cannot_supply_positive_judgment(finish):
    context, passage, answer = example()
    client, _, _ = fake_client(answer, finish=finish)
    with pytest.raises(NormCheckError, match="beendete"):
        client.check(context, passage.norm_id, passage.title, [passage])


@pytest.mark.parametrize("url", [
    "https://example.com", "http://example.com", "http://127.0.0.1.evil.example",
    "http://127.0.0.1@external.test", "http://user@localhost", "http://localhost/redirect",
    "http://localhost/?remote=external", "http://localhost/#fragment",
])
def test_constructor_rejects_nonlocal_or_ambiguous_endpoints(url):
    with pytest.raises(ValueError, match="Loopback"):
        LocalNormChecker(url)


@pytest.mark.parametrize("url", ["http://localhost:8099", "http://127.0.0.1:8080/v1/", "http://[::1]:8080"])
def test_loopback_constructor_never_starts_server_or_model(url):
    assert LocalNormChecker(url).base_url.endswith("/v1")


def test_transport_error_is_distinct_from_unknown_model_answer():
    context, passage, answer = example()
    client, _, _ = fake_client(answer)
    def fail(*a, **kw):
        raise urllib.error.URLError("offline")
    client._opener = SimpleNamespace(open=fail)
    with pytest.raises(NormCheckError, match="nicht erreichbar"):
        client.check(context, passage.norm_id, passage.title, [passage])


def test_foreign_norm_evidence_is_rejected_before_http():
    context, passage, answer = example()
    client, calls, _ = fake_client(answer)
    with pytest.raises(ValueError, match="derselben Norm"):
        client.check(context, "norm:AO:2", passage.title, [passage])
    assert calls == []


@pytest.mark.parametrize("envelope", [
    {"choices": "x"}, {"choices": [{}]}, {"choices": ["x"]},
    {"choices": [{"message": None}]}, {"choices": [{"message": {"content": None}}]},
])
def test_malformed_response_envelope_retains_received_body_hash(envelope):
    context, passage, answer = example()
    client, _, raw = fake_client(answer, envelope=envelope)
    with pytest.raises(NormCheckError) as caught:
        client.check(context, passage.norm_id, passage.title, [passage])
    assert caught.value.raw_http_sha256 == hashlib.sha256(raw).hexdigest()


def test_token_limit_failure_keeps_raw_model_text_for_audit():
    context, passage, answer = example()
    client, _, _ = fake_client(answer, finish="length")
    with pytest.raises(NormCheckError) as caught:
        client.check(context, passage.norm_id, passage.title, [passage])
    assert json.loads(caught.value.raw_response) == answer
    assert caught.value.raw_response_sha256


def test_quote_cannot_bridge_unrelated_source_passages():
    context, passage, answer = example()
    first = Passage("p1", passage.norm_id, "AO", "1", "Titel", 0, 3, "abc")
    second = Passage("p2", passage.norm_id, "AO", "1", "Titel", 30, 33, "def")
    answer["norm_quote"] = "abc\n\ndef"
    client, _, _ = fake_client(answer)
    result = client.check(context, passage.norm_id, passage.title, [first, second])
    assert result["status"] == "unklar" and result["norm_evidence"] is None


def test_full_context_quote_can_supply_real_reference_facts_for_a_distinct_focus():
    text = "Die Geschäftsführerin verwaltet die Firma. Die Belege fehlen vollständig."
    context = build_context_queries(text, mode="paragraph")[1]
    _, passage, answer = example()
    answer["source_quote"] = "Die Geschäftsführerin"
    client, _, _ = fake_client(answer)
    result = client.check(context, passage.norm_id, passage.title, [passage])
    assert result["status"] == "passend"
    assert result["source_evidence"]["start"] < context.focus_start


def test_repeated_quote_uses_first_original_occurrence_deterministically():
    text = "Belege fehlen vollständig; Belege fehlen vollständig."
    context = build_context_queries(text, mode="paragraph")[0]
    _, passage, answer = example()
    client, _, _ = fake_client(answer)
    result = client.check(context, passage.norm_id, passage.title, [passage])
    assert result["source_evidence"]["start"] == 0


def test_redirect_handler_never_forwards_request_payload():
    from risk_layer.norm_check import _NoRedirect
    assert _NoRedirect().redirect_request(None, None, 302, "Found", {}, "http://external.test") is None


@pytest.mark.parametrize("content", [
    '{"status":"unklar","status":"passend","source_quote":"","norm_quote":"","begruendung":""}',
    '{"status":"unklar","source_quote":"","norm_quote":"","begruendung":NaN}',
    '{"status":"unklar","source_quote":"","norm_quote":"","begruendung":Infinity}',
])
def test_ambiguous_or_nonstandard_json_is_rejected_with_raw_audit(content):
    context, passage, answer = example()
    client, _, _ = fake_client(answer, envelope={"choices": [
        {"finish_reason": "stop", "message": {"content": content}}]})
    with pytest.raises(NormCheckError, match="ungültige JSON") as caught:
        client.check(context, passage.norm_id, passage.title, [passage])
    assert caught.value.raw_response == content


def test_envelope_duplicate_keys_and_nonfinite_constants_are_rejected():
    from risk_layer.norm_check import _strict_json
    for value in ('{"choices":[],"choices":[]}', '{"usage":NaN}'):
        with pytest.raises(ValueError):
            _strict_json(value)
