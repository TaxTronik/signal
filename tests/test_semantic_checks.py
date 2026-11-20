"""Input provenance and conservative state handling, without loading a model."""

from dataclasses import replace
import math

import pytest

from risk_layer.semantic_checks import classify, format_focus_context
from risk_layer.semantic_context import ContextQuery, build_context_queries


@pytest.mark.parametrize(("support", "contradiction", "expected"), [
    (.99, .01, "modell_gestuetzt"),
    (.01, .99, "widerspruch"),
    (.01, .01, "unklar"),
    (.99, .99, "unklar"),
    (.5, .499999, "modell_gestuetzt"),
    (.499999, .5, "widerspruch"),
    (.5, .5, "unklar"),
    (0, 1, "widerspruch"),
    (1, 0, "modell_gestuetzt"),
])
def test_missing_support_and_conflicting_answers_are_not_legal_verdicts(
        support, contradiction, expected):
    assert classify(support, contradiction) == expected


@pytest.mark.parametrize("invalid", [
    math.nan, math.inf, -math.inf, -0.001, 1.001, True, False, "0.7", None,
])
@pytest.mark.parametrize("field", ["support", "contradiction"])
def test_invalid_model_scores_cannot_silently_produce_a_state(invalid, field):
    scores = {"support": .75, "contradiction": .1, field: invalid}
    with pytest.raises(ValueError, match="Prüfscore"):
        classify(scores["support"], scores["contradiction"])


def test_same_context_has_distinct_focus_inputs_and_preserves_unicode_source():
    text = "Die Geschäftsführerin prüft die Bücher.\nSie meldet sämtliche Beträge."
    contexts = build_context_queries(text, mode="paragraph")
    assert len(contexts) == 2
    assert contexts[0].query_text == contexts[1].query_text == text
    inputs = [format_focus_context(context) for context in contexts]
    assert inputs[0] != inputs[1]
    for context, model_input in zip(contexts, inputs, strict=True):
        assert f"]:\n{context.focus_text}\n\nORIGINALKONTEXT" in model_input
        assert model_input.endswith(text)
        assert f"Originalzeichen {context.focus_start}:{context.focus_end}" in model_input


def test_centered_context_crop_does_not_replace_full_focus():
    text = "Die " + "umfangreiche " * 20 + "Buchführung fehlt."
    context, = build_context_queries(text, mode="paragraph", max_chars=45)
    assert context.context_truncated and context.focus_truncated
    model_input = format_focus_context(context)
    assert context.focus_text in model_input
    assert model_input.endswith(context.query_text)
    assert "Kontextbereich gekürzt: ja; Fokus im Kontext gekürzt: ja" in model_input


def test_sentence_context_preserves_external_whitespace_without_moving_focus():
    context, = build_context_queries("  Die Bücher fehlen.  ", mode="sentence")
    assert context.query_text != context.focus_text
    model_input = format_focus_context(context)
    assert model_input.endswith(context.query_text)
    assert "Fokus im Kontext gekürzt: nein" in model_input


@pytest.mark.parametrize("change", [
    {"focus_start": -1},
    {"focus_start": True},
    {"focus_end": 100},
    {"context_start": 0.0},
    {"context_end": 100},
    {"focus_text": "xxxxxxxxxxxxxxxxxx"},
    {"query_text": "xxxxxxxxxxxxxxxxxx"},
    {"query_text": ""},
    {"focus_truncated": True},
    {"context_truncated": 1},
])
def test_malformed_context_is_rejected_instead_of_claiming_original_evidence(change):
    context, = build_context_queries("Die Bücher fehlen.", mode="paragraph")
    with pytest.raises(ValueError):
        format_focus_context(replace(context, **change))


def test_same_length_changed_source_overlap_is_rejected():
    context, = build_context_queries("Die Bücher fehlen.", mode="paragraph")
    altered = replace(context, query_text="X" + context.query_text[1:])
    with pytest.raises(ValueError, match="Originaloffsets"):
        format_focus_context(altered)


def test_unrelated_disjoint_source_spans_are_not_accepted_as_focus_context():
    context = ContextQuery(0, 3, "abc", "def", 10, 13, "neighbors", True, True)
    with pytest.raises(ValueError, match="Originaloffsets"):
        format_focus_context(context)
