"""Exercise optional factual gates through the real passage runtime, model-free."""

from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from risk_layer.catalog import Katalog
from risk_layer.embedding import EmbeddingIndex
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.reranker import Qwen3Reranker
from risk_layer.semantic_checks import (
    CHECK_INSTRUCTION_VERSION, CONTRADICTION_INSTRUCTION, SUPPORT_INSTRUCTION,
)
from risk_layer.semantic_search import SemanticResourcesUnavailable, passage_suggestions, PROFILE_CONFIG


@pytest.fixture(autouse=True)
def experimental_binary_checks(monkeypatch):
    # The Qwen two-question probe proved unsuitable in the real GPU experiment.
    # Test its generic runtime mechanics without enabling it in public profiles.
    for profile in ("praezision", "dokument"):
        monkeypatch.setitem(PROFILE_CONFIG, profile, {
            **PROFILE_CONFIG[profile], "checks": True, "agreement": False})


class Encoder:
    name = "precision-test"
    revision = None
    dim = 2

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.tile(np.array([1., 0.], dtype=np.float32), (len(texts), 1))


class Scores:
    def __init__(self, value=.9):
        self.value = value
        self.calls = []

    def score(self, pairs):
        self.calls.append(list(pairs))
        return [self.value(q, d) if callable(self.value) else self.value for q, d in pairs]


class CheckingScores(Scores):
    def __init__(self, value=.9, support=.8, contradiction=.1):
        super().__init__(value)
        self.vorauswahl = Scores(.4)
        self.support = support
        self.contradiction = contradiction
        self.checks = []

    def score_with_instruction(self, pairs, instruction, *, max_length):
        self.checks.append((list(pairs), instruction, max_length))
        assert instruction in (SUPPORT_INSTRUCTION, CONTRADICTION_INSTRUCTION)
        value = self.support if instruction == SUPPORT_INSTRUCTION else self.contradiction
        return [value(q, d) if callable(value) else value for q, d in pairs]


@pytest.fixture
def resources(tmp_path):
    nodes = [NormKnoten(id=f"norm:AO:{i}", law="AO", para=str(i),
                       titel=f"Norm {i}", text=f"Die eigenständige Regelung {i} betrifft den Vorgang.")
             for i in range(1, 9)]
    graph = NormGraph.build(nodes, tmp_path / "graph.sqlite")
    catalog = Katalog("test", [])
    encoder = Encoder()
    index = EmbeddingIndex.build(graph, encoder, katalog=catalog, passagen=True)
    encoder.calls.clear()
    try:
        yield SimpleNamespace(graph=graph, catalog=catalog, encoder=encoder, index=index)
    finally:
        graph.close()


def run(resources, reranker, *, profile="praezision",
        text="Der geschilderte Vorgang muss geprüft werden.", **overrides):
    kwargs = dict(profile=profile, laws=(), index=resources.index,
                  encoder=resources.encoder, reranker=reranker, trigger=None,
                  graph=resources.graph, chains={}, catalog=resources.catalog)
    kwargs.update(overrides)
    return passage_suggestions(text, **kwargs)


@pytest.mark.parametrize(("support", "contradiction", "hit"), [
    (.8, .1, True), (.1, .8, False), (.1, .1, False), (.8, .8, False),
    (.499999, .1, False), (.5, .499999, True), (.5, .5, False),
])
def test_only_unopposed_raw_support_reaches_output(resources, support, contradiction, hit):
    ranker = CheckingScores(support=support, contradiction=contradiction)
    rows = run(resources, ranker)
    assert bool(rows) == hit
    assert len(ranker.checks) == 2
    assert all(max_length == 2048 for _, _, max_length in ranker.checks)
    if hit:
        assert len(rows[0]["normen"]) == 5
        for norm in rows[0]["normen"]:
            assert norm["pruefung"] == {
                "status": "modell_gestuetzt", "version": CHECK_INSTRUCTION_VERSION,
                "unterstuetzung_score": support, "widerspruch_score": contradiction,
                "kalibriert": False,
            }
            assert norm["gestuetzt"] and norm["im_korpus"]
            assert norm["evidenz"]["passagen"]


@pytest.mark.parametrize(("score", "hit"), [(.499999, False), (.5, True)])
def test_raw_relevance_threshold_precedes_checks_without_similarity_rescue(resources, score, hit):
    ranker = CheckingScores(value=score)
    assert bool(run(resources, ranker)) == hit
    assert bool(ranker.checks) == hit


def test_all_relevance_survivors_are_checked_before_filling_top_five(resources):
    ranker = CheckingScores(
        value=lambda q, d: 1 - int(d.splitlines()[0].split(":")[-1]) / 100,
        support=lambda q, d: .8 if d.splitlines()[0] in {
            "norm:AO:6", "norm:AO:7", "norm:AO:8"} else .1,
    )
    result = run(resources, ranker)
    assert [n["id"] for n in result[0]["normen"]] == ["norm:AO:6", "norm:AO:7", "norm:AO:8"]
    assert [len(pairs) for pairs, _, _ in ranker.checks] == [8, 8]


def test_same_document_context_is_checked_separately_for_each_original_focus(resources):
    text = "Die erste Aussage beschreibt die Person. Die zweite beschreibt den Vorgang."
    ranker = CheckingScores(support=lambda q, d: (
        .8 if "erste Aussage" in q.split("\n\nORIGINALKONTEXT", 1)[0] else .1))
    result = run(resources, ranker, profile="dokument", text=text)
    assert len(result) == 1
    assert "erste Aussage" in result[0]["satz"]
    assert text[result[0]["start"]:result[0]["end"]] == result[0]["satz"]
    assert len(ranker.checks) == 4
    support_inputs = [pairs[0][0] for pairs, instruction, _ in ranker.checks
                      if instruction == SUPPORT_INSTRUCTION]
    assert len(set(support_inputs)) == 2
    assert support_inputs[0].split("\n\nORIGINALKONTEXT", 1)[1] == \
        support_inputs[1].split("\n\nORIGINALKONTEXT", 1)[1]
    assert len(resources.encoder.calls) == 1 and len(resources.encoder.calls[0]) == 1


def test_context_cut_through_focus_cannot_create_positive_assessment(resources):
    text = "Die " + "umfangreiche " * 100 + "Buchführung fehlt."
    ranker = CheckingScores()
    assert run(resources, ranker, profile="dokument", text=text) == []
    assert ranker.checks == []


@pytest.mark.parametrize("missing", ["preselector", "instruction", "index", "encoder", "graph"])
def test_explicit_precision_profile_never_silently_degrades(resources, missing):
    ranker = CheckingScores()
    kwargs = {}
    if missing == "preselector":
        ranker.vorauswahl = None
    elif missing == "instruction":
        ranker.score_with_instruction = None
    else:
        kwargs[missing] = None
    with pytest.raises(SemanticResourcesUnavailable):
        run(resources, ranker, **kwargs)
    assert ranker.calls == [] and ranker.checks == [] and resources.encoder.calls == []


@pytest.mark.parametrize("stage", ["support", "contradiction"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01, True, "0.8"])
def test_invalid_check_output_is_a_backend_failure_not_a_positive_or_negative(resources, stage, invalid):
    ranker = CheckingScores(**{stage: invalid})
    with pytest.raises(RuntimeError, match="Prüf|Prüfs|Reranker"):
        run(resources, ranker)


def test_incomplete_check_output_does_not_mean_no_match(resources, monkeypatch):
    ranker = CheckingScores()
    monkeypatch.setattr(ranker, "score_with_instruction", lambda *a, **kw: [])
    with pytest.raises(RuntimeError, match="Unvollständige"):
        run(resources, ranker)


def test_check_model_failure_is_not_swallowed(resources, monkeypatch):
    ranker = CheckingScores()
    def fail(*a, **kw):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(ranker, "score_with_instruction", fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        run(resources, ranker)


@pytest.mark.parametrize("profile", ["normpaket", "kontext", "kaskade"])
def test_existing_profiles_never_call_new_questions_or_add_check_metadata(resources, profile):
    ranker = CheckingScores()
    result = run(resources, ranker, profile=profile)
    assert result
    assert ranker.checks == []
    assert all("pruefung" not in n for row in result for n in row["normen"])


def _fake_qwen(monkeypatch):
    """Inspect tokenizer/forward protocol without Torch or model dependencies."""
    class Tensor(np.ndarray):
        def float(self):
            return self.astype(np.float32)

    class Inputs(dict):
        def to(self, device):
            return self

    calls, forwards = [], []
    class Tokenizer:
        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return SimpleNamespace(input_ids=[10, 11])

        def pad(self, inputs, **kwargs):
            return Inputs(inputs)

    def model(**kwargs):
        forwards.append(kwargs)
        return SimpleNamespace(logits=np.array([[[0., 2.]]] * len(kwargs["input_ids"])).view(Tensor))

    def softmax(value, dim):
        exp = np.exp(value - value.max(axis=dim, keepdims=True))
        return exp / exp.sum(axis=dim, keepdims=True)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext, softmax=softmax))
    ranker = Qwen3Reranker(device="cpu")
    ranker._model, ranker._tok, ranker._dev = model, Tokenizer(), "cpu"
    ranker._pre, ranker._suf, ranker._no, ranker._yes = [1], [2], 0, 1
    return ranker, calls, forwards


def test_legacy_qwen_score_keeps_exact_prompt_budget_and_forward(monkeypatch):
    ranker, calls, forwards = _fake_qwen(monkeypatch)
    scores = ranker.score([("Fokus", "Quelle")])
    assert calls == [(f"<Instruct>: {Qwen3Reranker.INSTRUCTION}\n<Query>: Fokus\n<Document>: Quelle", {
        "add_special_tokens": False, "truncation": True, "max_length": 1024,
    })]
    assert scores == pytest.approx([1 / (1 + np.exp(-2))])
    assert forwards[0]["logits_to_keep"] == 1 and forwards[0]["use_cache"] is False


def test_factual_qwen_refuses_overlength_instead_of_truncating_qualifying_text(monkeypatch):
    ranker, calls, forwards = _fake_qwen(monkeypatch)
    with pytest.raises(RuntimeError, match="Tokenbudget"):
        ranker.score_with_instruction([("Fokus", "Quelle")], SUPPORT_INSTRUCTION, max_length=3)
    assert calls[0][1] == {"add_special_tokens": False, "truncation": False}
    assert forwards == []


def test_alternating_check_instructions_do_not_mutate_default_query(monkeypatch):
    ranker, calls, _ = _fake_qwen(monkeypatch)
    for instruction in (SUPPORT_INSTRUCTION, CONTRADICTION_INSTRUCTION):
        ranker.score_with_instruction([("Fokus", "Quelle")], instruction)
    ranker.score([("Fokus", "Quelle")])
    assert calls[0][0].startswith(f"<Instruct>: {SUPPORT_INSTRUCTION}\n")
    assert calls[1][0].startswith(f"<Instruct>: {CONTRADICTION_INSTRUCTION}\n")
    assert calls[2][0].startswith(f"<Instruct>: {Qwen3Reranker.INSTRUCTION}\n")
