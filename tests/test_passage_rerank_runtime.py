"""Source-window reranking through real index/graph paths, without models."""

from types import SimpleNamespace

import numpy as np
import pytest

from risk_layer.catalog import Katalog
from risk_layer.embedding import EmbeddingIndex
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.models import Katalogeintrag
from risk_layer.semantic_search import PROFILE_CONFIG, SemanticResourcesUnavailable, passage_suggestions


class Encoder:
    name = "passage-runtime-test"
    revision = None
    dim = 2

    def encode(self, texts):
        return np.tile(np.array([1., 0.], dtype=np.float32), (len(texts), 1))


class Scores:
    def __init__(self, value=.8):
        self.value = value
        self.calls = []

    def score(self, pairs):
        self.calls.append(list(pairs))
        return [self.value(query, document) if callable(self.value) else self.value
                for query, document in pairs]


class Ranker(Scores):
    def __init__(self, value=.8, preselector=.8):
        super().__init__(value)
        self.vorauswahl = Scores(preselector)
        self.instruction_calls = []

    def score_with_instruction(self, pairs, instruction, *, max_length):
        self.instruction_calls.append((list(pairs), instruction, max_length))
        return [self.value(query, document) if callable(self.value) else self.value
                for query, document in pairs]


@pytest.fixture
def resources(tmp_path, monkeypatch):
    text = ("  ALPHA_PEAK steht ausschließlich am Beginn der Originalquelle. "
            + "Der weitere Abschnitt beschreibt allgemeine Voraussetzungen und Verfahren. " * 38)
    nodes = [
        NormKnoten(id="norm:AO:1", law="AO", para="1", titel="Erste Quellnorm", text=text),
        NormKnoten(id="norm:AO:2", law="AO", para="2", titel="Zweite Quellnorm",
                   text="Die zweite Regelung betrifft einen anderen Sachverhalt. " * 24),
        NormKnoten(id="norm:AO:3", law="AO", para="3", titel="Dritte Quellnorm",
                   text="Die dritte Regelung benennt eine andere Voraussetzung. " * 24),
    ]
    graph = NormGraph.build(nodes, tmp_path / "graph.sqlite")
    catalog = Katalog("test", [Katalogeintrag(
        id="anchor", begriff_kanonisch="Quellbegriff", norm_anker=("§ 1 AO",),
        paraphrasen=("CATALOG_ANCHOR_ONLY ist eine vorhandene Katalogparaphrase.",))])
    encoder = Encoder()
    index = EmbeddingIndex.build(graph, encoder, katalog=catalog, passagen=True)
    monkeypatch.setitem(PROFILE_CONFIG, "kaskade", {
        **PROFILE_CONFIG["kaskade"], "passage_rerank": True, "passage_pool": 8,
        "shortlist": 2, "agreement": False, "passage_shortlist": "max",
    })
    try:
        yield SimpleNamespace(graph=graph, catalog=catalog, encoder=encoder, index=index)
    finally:
        graph.close()


def run(resources, ranker, text="Der ALPHA_PEAK Sachverhalt soll untersucht werden."):
    return passage_suggestions(text, profile="kaskade", laws=(), index=resources.index,
        encoder=resources.encoder, reranker=ranker, trigger=None, graph=resources.graph,
        chains={}, catalog=resources.catalog)


def test_norm_maximum_drives_shortlist_instead_of_mean_or_package_score(resources, monkeypatch):
    monkeypatch.setitem(PROFILE_CONFIG, "kaskade", {**PROFILE_CONFIG["kaskade"], "shortlist": 1})
    def score(_query, document):
        norm_id = document.splitlines()[0]
        if norm_id == "norm:AO:1":
            return .99 if "ALPHA_PEAK" in document else .01
        return .8
    ranker = Ranker(preselector=score)
    result = run(resources, ranker)
    assert [norm["id"] for norm in result[0]["normen"]] == ["norm:AO:1"]
    assert len(ranker.vorauswahl.calls) == 1  # No second, whole-package BGE ranking.
    passage_pairs = ranker.vorauswahl.calls[0]
    assert {document.splitlines()[0] for _, document in passage_pairs} == {
        "norm:AO:1", "norm:AO:2", "norm:AO:3"}
    alpha_scores = [score(query, document) for query, document in passage_pairs
                    if document.startswith("norm:AO:1\n")]
    assert len(alpha_scores) > 1 and sum(alpha_scores) / len(alpha_scores) < .8
    assert all("CATALOG_ANCHOR_ONLY" not in document for _, document in passage_pairs)
    assert len(ranker.calls) == 1 and len(ranker.calls[0]) == 1
    assert "ALPHA_PEAK" in ranker.calls[0][0][1]


def test_norm_shortlist_caps_distinct_norms_not_individual_windows(resources):
    ranker = Ranker(preselector=lambda _q, d: .9 if d.startswith("norm:AO:1\n") else .8)
    result = run(resources, ranker)
    assert [norm["id"] for norm in result[0]["normen"]] == ["norm:AO:1", "norm:AO:2"]
    assert [document.splitlines()[0] for _, document in ranker.calls[0]] == ["norm:AO:1", "norm:AO:2"]
    assert sum(document.startswith("norm:AO:1\n") for _, document in ranker.vorauswahl.calls[0]) > 1


def test_output_evidence_and_focus_offsets_remain_in_the_original_sources(resources):
    source_windows = [p for p in resources.index.passage_evidence.passages if p.norm_id == "norm:AO:1"]
    assert any(left.start < right.start < left.end
               for left, right in zip(source_windows, source_windows[1:]))
    text = "  Der ALPHA_PEAK Sachverhalt soll untersucht werden. Ein weiterer Vorgang muss erklärt werden."
    ranker = Ranker(preselector=lambda _q, d: .99 if "ALPHA_PEAK" in d else .6)
    result = run(resources, ranker, text)
    assert len(result) == 2
    for suggestion in result:
        assert text[suggestion["start"]:suggestion["end"]] == suggestion["satz"]
        context = suggestion["kontext"]
        assert text[context["start"]:context["end"]] == context["text"]
        for norm in suggestion["normen"]:
            evidence = norm["evidenz"]
            assert evidence["fingerprint"] == resources.index.passage_evidence.source_fingerprint
            for passage in evidence["passagen"]:
                source = resources.graph.node(norm["id"])["text"]
                assert passage["norm_id"] == norm["id"]
                assert source[passage["quell_start"]:passage["quell_end"]] == passage["text"]
                assert passage["quell_start"] < passage["quell_end"]
            spans = sorted((p["quell_start"], p["quell_end"]) for p in evidence["passagen"])
            assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_configured_instruction_receives_selected_packages_and_exact_focus(resources, monkeypatch):
    instruction = "Return source relevance for this exact query."
    monkeypatch.setitem(PROFILE_CONFIG, "kaskade", {**PROFILE_CONFIG["kaskade"], "instruction": instruction})
    ranker = Ranker(value=.5)
    text = "Der ALPHA_PEAK Sachverhalt soll untersucht werden."
    result = run(resources, ranker, text)
    assert result and ranker.calls == []
    assert len(ranker.instruction_calls) == 1
    pairs, actual_instruction, budget = ranker.instruction_calls[0]
    assert actual_instruction == instruction and budget == 2048
    focus = text[result[0]["start"]:result[0]["end"]]
    assert len(pairs) == 2 and all(query == focus for query, _ in pairs)
    assert all(norm["rerank"] == .5 for norm in result[0]["normen"])
    assert all("pruefung" not in norm for norm in result[0]["normen"])


def test_repeated_focus_keeps_each_original_span(resources):
    sentence = "Der ALPHA_PEAK Sachverhalt soll untersucht werden."
    text = sentence + " " + sentence
    ranker = Ranker()
    result = run(resources, ranker, text)
    assert len(result) == 2 and result[0]["start"] < result[1]["start"]
    assert result[0]["satz"] == result[1]["satz"]
    assert all(text[row["start"]:row["end"]] == row["satz"] for row in result)
    assert result[0]["normen"] == result[1]["normen"]


def test_configured_instruction_requires_backend_capability_before_any_scoring(resources, monkeypatch):
    monkeypatch.setitem(PROFILE_CONFIG, "kaskade", {
        **PROFILE_CONFIG["kaskade"], "instruction": "A declared relevance instruction."})
    ranker = Ranker()
    monkeypatch.setattr(ranker, "score_with_instruction", None)
    with pytest.raises(SemanticResourcesUnavailable, match="Instruktions"):
        run(resources, ranker)
    assert ranker.vorauswahl.calls == [] and ranker.calls == []


def test_invalid_passage_score_fails_before_qwen_instead_of_becoming_abstention(resources):
    ranker = Ranker(preselector=float("nan"))
    with pytest.raises(RuntimeError, match="Nicht-endliche Passagen-Vorauswahl"):
        run(resources, ranker)
    assert ranker.calls == [] and ranker.instruction_calls == []


def test_incomplete_passage_scores_fail_before_qwen(resources, monkeypatch):
    ranker = Ranker()
    monkeypatch.setattr(ranker.vorauswahl, "score", lambda pairs: [.8])
    with pytest.raises(RuntimeError, match="Unvollständige Passagen-Vorauswahl"):
        run(resources, ranker)
    assert ranker.calls == []


def test_norm_without_selected_source_windows_never_falls_back_to_old_package(resources, monkeypatch):
    from risk_layer import passage_selection
    original = passage_selection.select_passage_candidates
    def select(evidence, norm_id, vector, query, **kwargs):
        return [] if norm_id == "norm:AO:1" else original(evidence, norm_id, vector, query, **kwargs)
    monkeypatch.setattr(passage_selection, "select_passage_candidates", select)
    ranker = Ranker()
    result = run(resources, ranker)
    assert result
    assert all(norm["id"] != "norm:AO:1" for sentence in result for norm in sentence["normen"])
    assert all(not document.startswith("norm:AO:1\n") for calls in
               (ranker.vorauswahl.calls, ranker.calls) for pairs in calls for _, document in pairs)


@pytest.fixture
def anchored_resources(tmp_path, monkeypatch):
    # Synthetic source strings test ranking/provenance, not legal applicability.
    short_source = "Die kurze Originalquelle beschreibt einen abstrakten Vorgang."
    long_source = ("EARLY_WINDOW beschreibt den Beginn der anderen Quelle. "
                   + "Dieser Quellabschnitt enthaelt weitere unabhaengige Angaben. " * 35
                   + "LATE_WINDOW beschreibt den gesuchten Teil der anderen Quelle.")
    anchor = "Die Lieferkiste bleibt unbeachtet im Nebenraum."
    graph = NormGraph.build([
        NormKnoten(id="norm:AO:11", law="AO", para="11", titel="Kurze Testquelle", text=short_source),
        NormKnoten(id="norm:AO:12", law="AO", para="12", titel="Lange Testquelle", text=long_source),
    ], tmp_path / "anchor-graph.sqlite")
    catalog = Katalog("test", [Katalogeintrag(
        id="synthetic_anchor", begriff_kanonisch="Testparaphrase", norm_anker=("§ 11 AO",),
        paraphrasen=(anchor,))])
    encoder = Encoder()
    index = EmbeddingIndex.build(graph, encoder, katalog=catalog, passagen=True)
    monkeypatch.setitem(PROFILE_CONFIG, "kaskade", {
        **PROFILE_CONFIG["kaskade"], "passage_rerank": True, "passage_pool": 8,
        "shortlist": 1, "agreement": False, "passage_shortlist": "package",
    })
    try:
        yield SimpleNamespace(graph=graph, catalog=catalog, encoder=encoder, index=index, anchor=anchor)
    finally:
        graph.close()


def anchored_preselection(_query, document):
    if "Die Lieferkiste bleibt unbeachtet im Nebenraum." in document:
        return .99  # The existing catalog bridge is present only in the package.
    if document.startswith("norm:AO:11\n"):
        return .01  # The abstract short source alone is weakly ranked.
    return .9 if "LATE_WINDOW" in document else .1


@pytest.mark.parametrize(("mode", "expected_norm", "bge_calls"), [
    ("package", "norm:AO:11", 2), ("max", "norm:AO:12", 1),
])
def test_package_shortlist_restores_catalog_bridge_lost_by_source_maximum(
        anchored_resources, monkeypatch, mode, expected_norm, bge_calls):
    monkeypatch.setitem(PROFILE_CONFIG, "kaskade", {
        **PROFILE_CONFIG["kaskade"], "passage_shortlist": mode})
    ranker = Ranker(preselector=anchored_preselection)
    result = run(anchored_resources, ranker, anchored_resources.anchor)
    assert [n["id"] for n in result[0]["normen"]] == [expected_norm]
    assert len(ranker.vorauswahl.calls) == bge_calls
    assert all(anchored_resources.anchor not in d for _, d in ranker.vorauswahl.calls[0])
    assert [d.splitlines()[0] for _, d in ranker.calls[0]] == [expected_norm]
    if mode == "package":
        assert len(ranker.vorauswahl.calls[1]) == 2
        assert any(anchored_resources.anchor in d for _, d in ranker.vorauswahl.calls[1])


def test_package_ranking_keeps_neural_source_choice_and_labels_anchors_separately(
        anchored_resources, monkeypatch):
    monkeypatch.setitem(PROFILE_CONFIG, "kaskade", {**PROFILE_CONFIG["kaskade"], "shortlist": 2})
    ranker = Ranker(preselector=anchored_preselection)
    result = run(anchored_resources, ranker, anchored_resources.anchor)
    norms = {n["id"]: n for n in result[0]["normen"]}
    assert set(norms) == {"norm:AO:11", "norm:AO:12"}
    assert "LATE_WINDOW" in norms["norm:AO:12"]["evidenz"]["passagen"][0]["text"]
    assert norms["norm:AO:11"]["evidenz"]["anker"] == [anchored_resources.anchor]
    assert len(norms["norm:AO:11"]["evidenz"]["passagen"]) == 1
    for norm in norms.values():
        for source in norm["evidenz"]["passagen"]:
            original = anchored_resources.graph.node(norm["id"])["text"]
            assert original[source["quell_start"]:source["quell_end"]] == source["text"]
            assert anchored_resources.anchor not in source["text"]
    # Both complete packages are ranked before the final model sees them.
    assert ranker.calls[0] == ranker.vorauswahl.calls[1]


def test_invalid_second_stage_package_score_fails_before_final_reranking(anchored_resources):
    def score(query, document):
        return float("nan") if anchored_resources.anchor in document else anchored_preselection(query, document)
    ranker = Ranker(preselector=score)
    with pytest.raises(RuntimeError, match="Nicht-endliche Reranker-Vorauswahl"):
        run(anchored_resources, ranker, anchored_resources.anchor)
    assert len(ranker.vorauswahl.calls) == 2
    assert ranker.calls == [] and ranker.instruction_calls == []
