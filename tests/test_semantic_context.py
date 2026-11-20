"""Coherent semantic context preserves source spans and connected facts."""

from types import SimpleNamespace

import pytest

from risk_layer.semantic_context import build_context_queries


def assert_source_spans(text, contexts):
    for context in contexts:
        assert text[context.focus_start:context.focus_end] == context.focus_text
        assert text[context.context_start:context.context_end] == context.query_text


@pytest.mark.parametrize("abbreviation", ["lt", "Lt", "bzgl", "insb"])
def test_abbreviation_continuation_keeps_following_phrase_in_focus(abbreviation):
    text = f"  Der Betrag {abbreviation}. Rechnung ist bezahlt. Der Rest bleibt offen."
    contexts = build_context_queries(text, mode="coherent")
    assert [c.focus_text for c in contexts] == [
        f"Der Betrag {abbreviation}. Rechnung ist bezahlt", "Der Rest bleibt offen",
    ]
    assert all(c.query_text == text.strip() for c in contexts)
    assert_source_spans(text, contexts)


def test_legacy_sentence_and_paragraph_modes_keep_their_original_boundaries():
    text = "Der Betrag lt. Rechnung ist bezahlt; der Rest bleibt offen."
    expected = ["Der Betrag lt", "Rechnung ist bezahlt", "der Rest bleibt offen"]
    for mode in ("sentence", "neighbors", "paragraph"):
        assert [c.focus_text for c in build_context_queries(text, mode=mode)] == expected


def test_list_intro_items_and_numeric_value_remain_one_original_focus():
    text = "Erfasst wurden:\nMaterial 740 EUR; Fahrtkosten 86 EUR; Portokosten: 9,50."
    context, = build_context_queries(text, mode="coherent")
    assert context.focus_text == text[:-1]
    assert context.query_text == text
    assert_source_spans(text, [context])


def test_colon_value_is_not_discarded_as_an_ineligible_numeric_fragment():
    text = "Der Wiederbeschaffungspreis: 730,00. Der Verkaufspreis: 990,00."
    contexts = build_context_queries(text, mode="coherent")
    assert [c.focus_text for c in contexts] == [
        "Der Wiederbeschaffungspreis: 730,00", "Der Verkaufspreis: 990,00",
    ]
    assert all(c.query_text == text for c in contexts)
    assert_source_spans(text, contexts)


def test_comparison_keeps_distinct_foci_with_both_amounts_in_context():
    text = ("Die Gesellschaft zahlt der Inhaberin monatlich 4.200 Euro Miete. "
            "Andere Mieter zahlen fuer vergleichbare Raeume monatlich 2.600 Euro.")
    contexts = build_context_queries(text, mode="coherent")
    assert len(contexts) == 2
    assert contexts[0].focus_end < contexts[1].focus_start
    assert all("4.200" in c.query_text and "2.600" in c.query_text for c in contexts)
    assert all(not c.context_truncated and not c.focus_truncated for c in contexts)
    assert_source_spans(text, contexts)


@pytest.mark.parametrize("separator", ["\n\n", "\r\n \t\r\n"])
@pytest.mark.parametrize("cross_paragraphs", [False, True])
def test_blank_paragraphs_never_merge_foci(separator, cross_paragraphs):
    first, second = "Der Betrag lt.", "Rechnung liegt noch nicht vor."
    text = first + separator + second
    contexts = build_context_queries(text, mode="coherent", cross_paragraphs=cross_paragraphs)
    assert [c.focus_text for c in contexts] == [first[:-1], second[:-1]]
    if cross_paragraphs:
        assert all(c.query_text == text for c in contexts)
    else:
        assert [c.query_text for c in contexts] == [first, second]
    assert_source_spans(text, contexts)


def test_pdf_linebreak_after_abbreviation_is_preserved_inside_focus():
    text = "Der Anspruch lt.\r\nVertrag bleibt unveraendert."
    context, = build_context_queries(text, mode="coherent")
    assert context.focus_text == text[:-1]
    assert_source_spans(text, [context])


def test_real_sentence_ending_is_not_merged_by_a_word_suffix():
    text = "Die Ware wurde abgeholt. Rechnung ist bezahlt."
    contexts = build_context_queries(text, mode="coherent")
    assert [c.focus_text for c in contexts] == [
        "Die Ware wurde abgeholt", "Rechnung ist bezahlt",
    ]


def test_oversized_connected_focus_is_not_silently_shortened():
    text = "Als Kosten erfasst: " + "; ".join(f"Position {i} zu 80 Euro" for i in range(8)) + "."
    context, = build_context_queries(text, mode="coherent", max_chars=80)
    assert context.focus_text == text[:-1]
    assert len(context.query_text) <= 80
    assert context.focus_truncated and context.context_truncated
    assert_source_spans(text, [context])


def test_context_budget_preserves_a_fitting_focus_and_reports_missing_neighbors():
    text = "Vorheriger Kontext ist sehr umfangreich. Der Betrag fehlt. Danach folgt weiterer Kontext."
    contexts = build_context_queries(text, mode="coherent", max_chars=40)
    assert all(len(c.query_text) <= 40 and c.context_truncated for c in contexts)
    assert all(c.focus_text in c.query_text and not c.focus_truncated for c in contexts)
    assert_source_spans(text, contexts)


@pytest.mark.parametrize("kind", ["wrong_text", "overlap", "bool_offset"])
def test_invalid_injected_spans_are_rejected_before_continuation_merging(kind):
    text = "Der Betrag lt. Rechnung bleibt offen."
    first = SimpleNamespace(text=text[:13], start=0, end=13)
    second = SimpleNamespace(text=text[14:-1], start=14, end=len(text) - 1)
    if kind == "wrong_text":
        first.text = "Erfundener Inhalt"
    elif kind == "overlap":
        second.start = 10
        second.text = text[10:-1]
    else:
        first.start = False
    with pytest.raises(ValueError, match="offsettreu"):
        build_context_queries(text, mode="coherent", sentence_splitter=lambda _: [first, second])


@pytest.mark.parametrize("space", ["\u00a0", "\u202f"])
def test_protected_spaces_preserve_abbreviation_and_list_continuations(space):
    text = f"Der Betrag lt.{space}Rechnung:{space}740 EUR;{space}bereits bezahlt: 90 EUR."
    context, = build_context_queries(text, mode="coherent")
    assert context.focus_text == text[:-1]
    assert context.query_text == text
    assert_source_spans(text, [context])


@pytest.mark.parametrize("space", ["\u00a0", "\u202f"])
def test_unicode_blank_crlf_line_is_a_focus_boundary_even_without_punctuation(space):
    first, second = "Ein erster Sachverhalt", "Ein zweiter Sachverhalt"
    text = first + "\r\n" + space + "\r\n" + second
    for cross_paragraphs in (False, True):
        contexts = build_context_queries(text, mode="coherent", cross_paragraphs=cross_paragraphs)
        assert [c.focus_text for c in contexts] == [first, second]
        assert [c.query_text for c in contexts] == ([text, text] if cross_paragraphs else [first, second])
        assert_source_spans(text, contexts)
    # The optional repair does not change the legacy paragraph detector.
    legacy = build_context_queries(text, mode="paragraph")
    assert all(c.query_text == text for c in legacy)


@pytest.mark.parametrize(("opening", "closing"), [("»", "«"), ("“", "”")])
def test_quoted_colon_and_internal_full_stop_keep_balanced_quote_and_next_sentence(opening, closing):
    quoted = f"{opening}Zitat: Der Eingang ist bestaetigt.{closing}"
    text = quoted + "\r\nDie Bearbeitung beginnt morgen."
    contexts = build_context_queries(text, mode="coherent")
    assert [c.focus_text for c in contexts] == [quoted, "Die Bearbeitung beginnt morgen"]
    assert all(c.query_text == text for c in contexts)
    assert_source_spans(text, contexts)


def test_semicolon_after_a_real_sentence_end_does_not_join_independent_foci():
    text = "Die Ware ist angekommen.; Die Rechnung bleibt offen."
    contexts = build_context_queries(text, mode="coherent")
    assert [c.focus_text for c in contexts] == ["Die Ware ist angekommen", "Die Rechnung bleibt offen"]
    assert_source_spans(text, contexts)
