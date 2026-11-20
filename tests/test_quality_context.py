"""Originalspans, begrenzter Retrieval-Kontext und explizite Herkunfts-Suchraeume."""

from dataclasses import dataclass

import pytest

from risk_layer.muster import saetze_mit_offset
from risk_layer.semantic_context import build_context_queries as runtime_context_queries
from tools._quality_context import build_context_queries as research_context_queries
from tools._quality_context import source_workflow_laws


@pytest.fixture(params=[runtime_context_queries, research_context_queries],
                ids=["runtime", "research"])
def build_context_queries(request):
    return request.param


def _assert_offsets(text, queries):
    for query in queries:
        assert text[query.focus_start:query.focus_end] == query.focus_text
        assert text[query.context_start:query.context_end] == query.query_text


def test_sentence_baseline_preserves_real_encoder_inputs_and_focus_offsets(build_context_queries):
    text = "  Die Belege sind vorhanden.\n Am 03. Mai wurde alles erfasst.  "
    result = build_context_queries(text, mode="sentence", max_chars=5)
    expected = [s for s in saetze_mit_offset(text) if len(s.text.strip()) >= 12]
    assert [q.query_text for q in result] == [s.text for s in expected]
    assert [q.focus_text for q in result] == [s.text.strip() for s in expected]
    assert all(not q.context_truncated for q in result)
    _assert_offsets(text, result)


def test_neighbors_preserve_short_reference_sentence_as_context(build_context_queries):
    text = "Die Tochter erzielt Verluste. Ja. Die Mutter uebernimmt diese."
    result = build_context_queries(text, mode="neighbors")
    assert len(result) == 2
    assert result[0].query_text == "Die Tochter erzielt Verluste. Ja"
    assert result[1].query_text == "Ja. Die Mutter uebernimmt diese"
    assert result[1].focus_text == "Die Mutter uebernimmt diese"
    _assert_offsets(text, result)


@pytest.mark.parametrize("separator", ["\n\n", "\n \t\n", "\r\n\r\n", "\r\n \t\r\n"])
@pytest.mark.parametrize("mode", ["neighbors", "paragraph"])
def test_context_does_not_cross_blank_paragraphs(separator, mode, build_context_queries):
    first = "Die Belege sind vollstaendig. Alle Zahlen stimmen."
    second = "Die Tochter hat grosse Verluste. Diese werden ausgeglichen."
    text = first + separator + second
    result = build_context_queries(text, mode=mode, neighbor_radius=3)
    assert len(result) == 4
    assert all(q.context_end <= len(first) for q in result[:2])
    assert all(q.context_start >= len(first + separator) for q in result[2:])
    _assert_offsets(text, result)


def test_single_linebreak_is_retained_and_paragraph_query_includes_punctuation(build_context_queries):
    text = "Die Tochter erzielt\nVerluste. Die Mutter uebernimmt diese."
    result = build_context_queries(text, mode="paragraph")
    assert len(result) == 2
    assert all(q.query_text == text for q in result)
    _assert_offsets(text, result)


def test_explicit_cross_paragraphs_switch_allows_document_context(build_context_queries):
    text = "Die Tochter erzielt Verluste.\n\nDie Mutter uebernimmt diese."
    result = build_context_queries(text, mode="paragraph", cross_paragraphs=True)
    assert len(result) == 2
    assert all(q.query_text == text for q in result)


def test_context_budget_contains_short_focus_and_clamps_to_paragraph_bounds(build_context_queries):
    text = "Vorheriger Kontext mit vielen Worten. Die Belege fehlen. Nachfolgender Kontext."
    result = build_context_queries(text, mode="paragraph", max_chars=35)
    for query in result:
        assert len(query.query_text) <= 35
        if len(query.focus_text) <= 35:
            assert query.focus_text in query.query_text
            assert not query.focus_truncated
    assert all(q.context_truncated for q in result)
    _assert_offsets(text, result)


def test_long_focus_keeps_original_span_and_reports_its_center_crop(build_context_queries):
    text = "Die " + "umfangreiche " * 20 + "Buchfuehrung fehlt."
    result = build_context_queries(text, mode="neighbors", max_chars=45)
    assert len(result) == 1
    query = result[0]
    assert len(query.query_text) <= 45
    assert query.focus_text == text[:-1]
    assert query.context_truncated and query.focus_truncated
    _assert_offsets(text, result)


def test_center_crop_over_pdf_whitespace_keeps_a_nonempty_original_query(build_context_queries):
    text = "Buchfuehrung" + " " * 100 + "fehlt"
    result = build_context_queries(text, mode="paragraph", max_chars=15)
    assert result[0].query_text.strip()
    assert len(result[0].query_text) <= 15
    assert result[0].focus_truncated
    _assert_offsets(text, result)


def test_injected_splitter_offsets_are_relative_to_each_paragraph(build_context_queries):
    @dataclass
    class Sentence:
        text: str
        start: int
        end: int

    text = "Erster Kontext vorhanden.\n\nZweiter Kontext vorhanden."
    def whole_paragraph(value):
        return [Sentence(value, 0, len(value))]
    result = build_context_queries(text, mode="neighbors", sentence_splitter=whole_paragraph)
    assert len(result) == 2
    assert result[1].focus_start == text.index("Zweiter")
    _assert_offsets(text, result)


def test_bad_splitter_offsets_are_not_silently_accepted(build_context_queries):
    with pytest.raises(ValueError, match="offsettreu"):
        build_context_queries("Die Belege fehlen.", sentence_splitter=lambda _: [
            type("Wrong", (), {"text": "Erfundener Inhalt", "start": 0, "end": 4})()])


@pytest.mark.parametrize("text", ["", " \n\n ", "12345678901234567890.", "Kurz."])
def test_empty_short_and_non_alphabetic_focus_is_skipped(text, build_context_queries):
    assert build_context_queries(text) == []


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"max_chars": 0}, {"neighbor_radius": -1}, {"min_focus_chars": 0},
])
def test_invalid_options_fail_explicitly(kwargs, build_context_queries):
    with pytest.raises(ValueError):
        build_context_queries("Die Belege fehlen.", **kwargs)


@pytest.mark.parametrize(("source", "laws"), [
    ("AO-Klausur, Aufg. 1a", ("AO",)),
    ("ESt-I-Klausur, Aufg. 1b Nr. 3", ("EStG", "EStDV", "AO")),
    ("ESt-II-Klausur, Aufg. 2", ("EStG", "EStDV", "AO")),
    ("KSt-Klausur (segmentiert_grob)", ("KStG", "KStDV", "AO")),
    ("USt-II-Klausur, Nr. 5", ("UStG", "UStDV", "AO")),
    ("GewSt-Klausur (segmentiert_grob)", ("GewStG", "GewStDV", "AO")),
    ("  ust–i–klausur  ", ("UStG", "UStDV", "AO")),
])
def test_workflow_scope_uses_only_explicit_document_type(source, laws):
    assert source_workflow_laws(source) == laws


@pytest.mark.parametrize("source", [
    "", "Prod-Katalog ao_gestaltungsmissbrauch", "Kap 4.3.2 (UStG:2)",
    "Ziel: norm:AO:42", "Ein Hinweis auf AO-Klausur", "ESt-Klausurloesung",
])
def test_unknown_or_label_like_source_does_not_create_scope(source):
    assert source_workflow_laws(source) is None


def test_solution_comments_do_not_expand_source_workflow_scope():
    assert source_workflow_laws("USt-I-Klausur, Nr. 5 — EStG:20-Zuordnung korrigiert") == (
        "UStG", "UStDV", "AO")
