"""Verify frozen proposal provenance before any local generative request."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from risk_layer.passages import Passage
from risk_layer.semantic_context import build_context_queries
from tools.quality_norm_check_probe import prepare_proposals


def fixture_case():
    text = "Die Geschäftsführerin verwaltet die Firma. Die Belege fehlen vollständig."
    context = build_context_queries(text, mode="sentence")[1]
    source = {"titel": "Buchführung", "law": "AO", "para": "1", "text": "Die Belege müssen vollständig vorliegen."}
    norm = {"id": "norm:AO:1", "titel": source["titel"], "evidenz": {"passagen": [{
        "id": "norm:AO:1@0:40", "norm_id": "norm:AO:1", "quell_start": 0,
        "quell_end": len(source["text"]), "text": source["text"]}],
        "anker": ["Katalogparaphrase, die nicht zum Prüfer gelangen darf."]}}
    suggestion = {"start": context.focus_start, "end": context.focus_end,
                  "satz": context.focus_text, "kontext": {
                      "start": context.context_start, "end": context.context_end, "text": context.query_text},
                  "normen": [norm]}
    case = {"case_id": "case-1", "query": text, "targets": ["secret_gold_target"],
            "source50": {"suggestions": [suggestion]},
            "agreement50": {"suggestions": [deepcopy(suggestion)]}}
    graph = SimpleNamespace(node=lambda norm_id: source if norm_id == "norm:AO:1" else None)
    return case, graph


def test_union_deduplicates_identical_focus_norm_and_uses_real_neighbor_context_only():
    case, graph = fixture_case()
    result = prepare_proposals(case, graph, build_context_queries, Passage)
    assert len(result) == 1
    proposal = next(iter(result.values()))
    assert "Geschäftsführerin" in proposal["context"].query_text
    assert "Geschäftsführerin" not in proposal["context"].focus_text
    assert proposal["passages"][0].text == graph.node("norm:AO:1")["text"]
    assert all("Katalogparaphrase" not in p.text and "secret_gold_target" not in p.text
               for p in proposal["passages"])


@pytest.mark.parametrize("corruption", ["focus", "context", "norm_text", "norm_offset", "norm_title", "norm_id"])
def test_stale_or_fabricated_input_cannot_reach_checker(corruption):
    case, graph = fixture_case()
    suggestion = case["source50"]["suggestions"][0]
    norm = suggestion["normen"][0]
    if corruption == "focus":
        suggestion["satz"] = "Erfundener Inhalt"
    elif corruption == "context":
        suggestion["kontext"]["text"] = "Erfundener Kontext"
    elif corruption == "norm_text":
        norm["evidenz"]["passagen"][0]["text"] = "Erfundene Normpassage"
    elif corruption == "norm_offset":
        norm["evidenz"]["passagen"][0]["quell_start"] = True
    elif corruption == "norm_title":
        norm["titel"] = "Anderer Normtitel"
    else:
        norm["id"] = "norm:AO:2"
    with pytest.raises(ValueError):
        prepare_proposals(case, graph, build_context_queries, Passage)


def test_different_valid_packages_for_same_focus_norm_are_not_arbitrarily_collapsed():
    case, graph = fixture_case()
    passage = case["agreement50"]["suggestions"][0]["normen"][0]["evidenz"]["passagen"][0]
    passage["text"] = passage["text"][4:]
    passage["quell_start"] = 4
    with pytest.raises(ValueError, match="different source packages"):
        prepare_proposals(case, graph, build_context_queries, Passage)


def test_original_report_labels_do_not_change_prepared_inputs():
    case, graph = fixture_case()
    first = prepare_proposals(case, graph, build_context_queries, Passage)
    case["targets"] = ["completely_different_gold"]
    case["negative"] = True
    second = prepare_proposals(case, graph, build_context_queries, Passage)
    assert first == second
