"""Runtime profiles preserve evidence, scope, and actual two-stage scoring."""
from types import SimpleNamespace

import numpy as np
import pytest

from risk_layer.catalog import Katalog
from risk_layer.embedding import EmbeddingIndex
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.pipeline import AnalyseOptionen, Pipeline
from risk_layer.models import Katalogeintrag
from risk_layer.reranker import CascadedQwenReranker, reranker_oeffnen
from risk_layer.semantic_search import SemanticResourcesUnavailable


class Encoder:
    name = "semantic-test"
    revision = None
    dim = 2

    def encode(self, texts):
        return np.tile(np.array([1., 0.], dtype=np.float32), (len(texts), 1))


class Reranker:
    def __init__(self, value=.2):
        self.value = value
        self.calls = []

    def score(self, pairs):
        self.calls.append(list(pairs))
        return [self.value] * len(pairs)


class InstructionReranker(Reranker):
    """Model-free backend with separately observable BGE and instructed Qwen."""

    def __init__(self, value=.7, preselection=.4):
        super().__init__(value)
        self.instruction_calls = []
        self.events = []
        owner = self

        class Preselection(Reranker):
            def score(self, pairs):
                self.calls.append(list(pairs))
                owner.events.append("bge")
                return [preselection(q, d) if callable(preselection) else preselection for q, d in pairs]

        self.vorauswahl = Preselection()

    def score_with_instruction(self, pairs, instruction, *, max_length):
        self.instruction_calls.append((list(pairs), instruction, max_length))
        self.events.append("qwen")
        return [self.value(q, d) if callable(self.value) else self.value for q, d in pairs]


@pytest.fixture
def resources(tmp_path):
    nodes = [NormKnoten(id=f"norm:AO:{i}", law="AO", para=str(i),
                       titel=f"Norm {i}", text=f"Die Regelung {i} enthält einen eigenständigen Tatbestand.")
             for i in range(1, 25)]
    nodes.append(NormKnoten(id="norm:EStG:9", law="EStG", para="9", titel="Anderes Gesetz",
                           text="Diese besondere Regelung gehört zum anderen Rechtsgebiet."))
    graph = NormGraph.build(nodes, tmp_path / "graph.sqlite")
    catalog = Katalog("test", [])
    encoder = Encoder()
    index = EmbeddingIndex.build(graph, encoder, katalog=catalog, passagen=True)
    try:
        yield SimpleNamespace(graph=graph, catalog=catalog, encoder=encoder, index=index)
    finally:
        graph.close()


def run(resources, profile="normpaket", *, laws=(), reranker=None, index=None,
        text="Der geschilderte Vorgang muss geprüft werden.", trigger=None):
    return Pipeline(resources.catalog).run(
        text, optionen=AnalyseOptionen(semantik_profil=profile, rechtsgebiete=laws),
        graph=resources.graph, embedding=index or resources.index,
        embedding_encoder=resources.encoder, reranker=reranker or Reranker(), trigger=trigger)


def test_source_evidence_offsets_are_not_input_offsets(resources):
    text = "  Ein anderer Sachverhalt steht hier zur Prüfung.  "
    report = run(resources, text=text)
    suggestion = report.semantik_vorschlaege[0]
    assert text[suggestion["start"]:suggestion["end"]] == suggestion["satz"]
    for norm in suggestion["normen"]:
        assert norm["evidenz"]["fingerprint"]
        for passage in norm["evidenz"]["passagen"]:
            source = resources.graph.node(passage["norm_id"])["text"]
            assert source[passage["quell_start"]:passage["quell_end"]] == passage["text"]


def test_scope_is_applied_before_candidate_limit(resources):
    ranker = Reranker()
    result = run(resources, laws=("EStG",), reranker=ranker)
    # AO occupies all global top20 slots; explicit scope must still reach EStG.
    assert [n["id"] for n in result.semantik_vorschlaege[0]["normen"]] == ["norm:EStG:9"]
    assert len(ranker.calls[0]) == 1
    with pytest.raises(ValueError, match="Rechtsgebiete"):
        run(resources, profile="standard", laws=("EStG",))


def test_context_is_shared_but_focus_offsets_stay_exact(resources):
    text = "Die erste Aussage beschreibt die Person. Die zweite beschreibt den Vorgang."
    ranker = Reranker()
    report = run(resources, profile="kontext", text=text, reranker=ranker)
    assert len(ranker.calls) == 1  # both foci have the same context, scored once
    assert len(report.semantik_vorschlaege) == 2
    for suggestion in report.semantik_vorschlaege:
        assert text[suggestion["start"]:suggestion["end"]] == suggestion["satz"]
        context = suggestion["kontext"]
        assert text[context["start"]:context["end"]] == context["text"]
        assert ranker.calls[0][0][0] == context["text"]


def test_missing_evidence_or_cascade_cannot_silently_degrade(resources):
    old = EmbeddingIndex.build(resources.graph, resources.encoder, katalog=resources.catalog)
    with pytest.raises(SemanticResourcesUnavailable, match="Passagen-Index"):
        run(resources, index=old)
    with pytest.raises(SemanticResourcesUnavailable, match="Kaskade"):
        run(resources, profile="kaskade")


@pytest.mark.parametrize(("score", "hit"), [(.009999, False), (.01, True), (.1, True)])
@pytest.mark.parametrize("profile", ["normpaket", "kontext"])
def test_profile_uses_raw_weak_rescue_floor(resources, score, hit, profile):
    result = run(resources, profile=profile, reranker=Reranker(score))
    assert bool(result.semantik_vorschlaege) == hit


def test_cascade_scores_only_real_shortlisted_pairs(resources):
    final = Reranker(.7)
    final.vorauswahl = Reranker(.4)
    result = run(resources, profile="kaskade", reranker=final)
    assert len(final.vorauswahl.calls[0]) == 25
    assert len(final.calls[0]) == 20
    assert set(final.calls[0]) <= set(final.vorauswahl.calls[0])
    assert len(result.semantik_vorschlaege[0]["normen"]) == 5
    assert all(n["rerank"] == .7 for n in result.semantik_vorschlaege[0]["normen"])


@pytest.mark.parametrize(("score", "hit"), [(.1, False), (.4999, False), (.5, True)])
def test_cascade_uses_raw_half_threshold_without_similarity_rescue(resources, monkeypatch, score, hit):
    hybrid_topk = resources.index.hybrid_topk
    monkeypatch.setattr(resources.index, "hybrid_topk", lambda *a, **kw: [
        {**candidate, "sim": .99} for candidate in hybrid_topk(*a, **kw)])
    final = Reranker(score)
    final.vorauswahl = Reranker(.9)
    result = run(resources, profile="kaskade", reranker=final)
    assert len(final.calls[0]) == 20
    assert bool(result.semantik_vorschlaege) == hit
    if hit:
        assert all(n["rerank"] == .5 and n["sim"] == .99
                   for n in result.semantik_vorschlaege[0]["normen"])


def test_incomplete_reranker_output_is_not_a_negative_result(resources):
    ranker = Reranker()
    ranker.score = lambda pairs: []
    with pytest.raises(RuntimeError, match="Unvollständige"):
        run(resources, reranker=ranker)


def test_operator_selects_cascade_model_without_loading_weights(monkeypatch):
    monkeypatch.setenv("RISK_LAYER_RERANKER", "bge-qwen3-reranker-4b")
    ranker = reranker_oeffnen()
    assert isinstance(ranker, CascadedQwenReranker)
    assert ranker._model is None
    assert ranker.vorauswahl._modell is None
    assert reranker_oeffnen("aus") is None


def test_trigger_without_source_evidence_never_reaches_reranker(resources):
    trigger = SimpleNamespace(kandidaten=lambda *a, **kw: [{
        "norm_id": "norm:AO:0", "zitat": "§ 0 AO", "titel": "Veralteter Trigger"}])
    ranker = Reranker()
    report = run(resources, trigger=trigger, reranker=ranker)
    assert all(document != "norm:AO:0" for call in ranker.calls for _, document in call)
    assert all(n["evidenz"]["passagen"] for s in report.semantik_vorschlaege for n in s["normen"])


def test_title_drift_invalidates_profile_even_if_text_fingerprint_matches(resources):
    before = resources.graph.fingerprint()
    resources.graph.con.execute("UPDATE knoten SET titel=? WHERE id=?",
                                ("Eine geänderte amtliche Überschrift", "norm:AO:1"))
    resources.graph.con.commit()
    assert resources.graph.fingerprint() == before
    with pytest.raises(SemanticResourcesUnavailable, match="Evidenz|Stand"):
        run(resources)


def test_catalog_drift_invalidates_profile_without_changing_catalog_version(resources):
    changed = Katalog("test", [Katalogeintrag(
        id="neu", begriff_kanonisch="Neuer Begriff", norm_anker=("§ 1 AO",),
        paraphrasen=("Eine neue kuratierte Umschreibung.",))])
    with pytest.raises(SemanticResourcesUnavailable, match="Evidenz|Katalog"):
        Pipeline(changed).run(
            "Der geschilderte Vorgang muss geprüft werden.",
            optionen=AnalyseOptionen(semantik_profil="normpaket"), graph=resources.graph,
            embedding=resources.index, embedding_encoder=resources.encoder, reranker=Reranker())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("stage", ["final", "preselection"])
def test_nonfinite_model_scores_fail_instead_of_becoming_results(resources, value, stage):
    ranker = Reranker(value if stage == "final" else .2)
    if stage == "preselection":
        ranker.vorauswahl = Reranker(value)
    with pytest.raises(RuntimeError, match="Reranker"):
        run(resources, profile="kaskade" if stage == "preselection" else "normpaket",
            reranker=ranker)


def test_scope_cannot_be_cut_off_by_larger_trigger_index(resources, monkeypatch):
    # A stale/larger trigger index may contain more norms than either vector
    # index; none of those foreign entries may consume the scoped limit.
    rows = [{"norm_id": f"norm:AO:alt{i}", "zitat": "AO", "titel": "Alter Eintrag"}
            for i in range(100)]
    # Even a still-valid ID can carry outdated display metadata in this index.
    rows.append({"norm_id": "norm:EStG:9", "zitat": "Altes Zitat", "titel": "Alter Titel"})
    trigger = SimpleNamespace(kandidaten=lambda _text, k=8: rows[:k])
    monkeypatch.setattr(resources.index, "_topk", lambda *a, **kw: [])
    report = run(resources, laws=("EStG",), trigger=trigger)
    assert [(n["id"], n["zitat"], n["titel"]) for n in report.semantik_vorschlaege[0]["normen"]] == [
        ("norm:EStG:9", "§ 9 EStG", "Anderes Gesetz")]


def test_invalid_query_vectors_fail_before_reranking(resources, monkeypatch):
    ranker = Reranker()
    monkeypatch.setattr(resources.encoder, "encode", lambda texts: np.full((len(texts), 2), np.nan))
    with pytest.raises(RuntimeError, match="Embedding-Vektoren"):
        run(resources, reranker=ranker)
    assert ranker.calls == []


@pytest.mark.parametrize("profile", ["praezision", "dokument"])
@pytest.mark.parametrize(("bge_score", "sim", "hit"), [
    (.009999, .99, False), (.01, .55, True), (.01, .549999, False),
    (.099999, .4, False), (.1, .4, True),
])
def test_public_precision_profiles_require_actual_model_agreement(
        resources, monkeypatch, profile, bge_score, sim, hit):
    ranker = Reranker(.7)
    ranker.vorauswahl = Reranker(bge_score)
    original = resources.index.hybrid_topk
    monkeypatch.setattr(resources.index, "hybrid_topk", lambda *a, **kw: [
        {**candidate, "sim": sim} for candidate in original(*a, **kw)])
    result = run(resources, profile=profile, reranker=ranker)
    # A binary relevance model is not required to answer contradiction questions.
    if hit:
        # The fixture contains 25 candidates: precision caps at 20, document at 50.
        assert len(ranker.calls[0]) == (20 if profile == "praezision" else 25)
    else:
        assert ranker.calls == []  # expensive Qwen work cannot change this gate
    assert bool(result.semantik_vorschlaege) == hit


@pytest.fixture
def treffer_resources(tmp_path):
    # Invented prose tests selection/provenance; it does not assign legal gold.
    anchor = "Die Lieferkiste bleibt unbeachtet im Nebenraum."
    long_source = ("EARLY_WINDOW beschreibt den Beginn der Testquelle. "
                   + "Dieser Quellabschnitt enthaelt weitere unabhaengige Angaben. " * 35
                   + "LATE_WINDOW beschreibt den gesuchten Teil der Testquelle.")
    nodes = [NormKnoten(
        id=f"norm:AO:{i}", law="AO", para=str(i), titel=f"Synthetische Quelle {i}",
        text=long_source if i == 1 else f"Die kurze Originalquelle {i} beschreibt einen abstrakten Vorgang.")
        for i in range(1, 26)]
    catalog = Katalog("test", [Katalogeintrag(
        id="synthetic_bridge", begriff_kanonisch="Testparaphrase", norm_anker=("§ 24 AO",),
        paraphrasen=(anchor,))])
    graph = NormGraph.build(nodes, tmp_path / "treffer-graph.sqlite")
    encoder = Encoder()
    index = EmbeddingIndex.build(graph, encoder, katalog=catalog, passagen=True)
    try:
        yield SimpleNamespace(graph=graph, catalog=catalog, encoder=encoder, index=index, anchor=anchor)
    finally:
        graph.close()


def test_public_treffer_pipeline_ranks_sources_then_anchor_aware_packages(treffer_resources):
    resources = treffer_resources

    def bge(_query, document):
        if resources.anchor in document:
            return .99
        if document.startswith("norm:AO:24\n"):
            return .01  # All other norms win if source maxima replace packages.
        if document.startswith("norm:AO:1\n"):
            return .9 if "LATE_WINDOW" in document else .1
        return .8

    def qwen(_query, document):
        if resources.anchor in document:
            return .95
        return .9 if document.startswith("norm:AO:1\n") else .6

    ranker = InstructionReranker(value=qwen, preselection=bge)
    text = "  Der geschilderte Vorgang muss geprüft werden.  "
    report = run(resources, profile="treffer", reranker=ranker, text=text)
    assert ranker.events == ["bge", "bge", "qwen"]
    sources, packages = ranker.vorauswahl.calls
    assert len(sources) > 25 and len(packages) == 25
    assert all(resources.anchor not in document for _, document in sources)
    weak_source = [(q, d) for q, d in sources if d.startswith("norm:AO:24\n")]
    assert len(weak_source) == 1 and bge(*weak_source[0]) == .01
    assert sum(any(d.startswith(f"norm:AO:{i}\n") and bge(q, d) > .01 for q, d in sources)
               for i in range(1, 26) if i != 24) == 24
    anchored_package = next((q, d) for q, d in packages if d.startswith("norm:AO:24\n"))
    assert resources.anchor in anchored_package[1] and bge(*anchored_package) == .99

    assert ranker.calls == []  # The legacy instruction path must not be used.
    assert len(ranker.instruction_calls) == 1
    pairs, instruction, budget = ranker.instruction_calls[0]
    assert len(pairs) == 20 and len({d.splitlines()[0] for _, d in pairs}) == 20
    assert pairs == sorted(packages, key=lambda pair: (-bge(*pair), pair[1].splitlines()[0]))[:20]
    assert pairs[0] == anchored_package
    assert instruction == ("Judge whether the legal provision (Document) matches the legal concept "
                           "or factual situation described in the Query.")
    assert budget == 2048
    suggestion, = report.semantik_vorschlaege
    focus = text[suggestion["start"]:suggestion["end"]]
    assert focus == suggestion["satz"] and all(query == focus for query, _ in pairs)
    assert len(suggestion["normen"]) == 5
    norms = {n["id"]: n for n in suggestion["normen"]}
    assert norms["norm:AO:24"]["evidenz"]["anker"] == [resources.anchor]
    assert "LATE_WINDOW" in norms["norm:AO:1"]["evidenz"]["passagen"][0]["text"]
    for norm in norms.values():
        assert norm["evidenz"]["fingerprint"] == resources.index.passage_evidence.source_fingerprint
        for passage in norm["evidenz"]["passagen"]:
            original = resources.graph.node(norm["id"])["text"]
            assert original[passage["quell_start"]:passage["quell_end"]] == passage["text"]
            assert resources.anchor not in passage["text"]


@pytest.mark.parametrize(("score", "hit"), [(.1, False), (.4999, False), (.5, True)])
def test_public_treffer_pipeline_uses_raw_half_without_similarity_rescue(resources, monkeypatch, score, hit):
    hybrid_topk = resources.index.hybrid_topk
    monkeypatch.setattr(resources.index, "hybrid_topk", lambda *a, **kw: [
        {**candidate, "sim": .99} for candidate in hybrid_topk(*a, **kw)])
    ranker = InstructionReranker(value=score, preselection=.009999)
    result = run(resources, profile="treffer", reranker=ranker)
    assert len(ranker.instruction_calls[0][0]) == 20
    assert bool(result.semantik_vorschlaege) == hit
    if hit:
        assert all(n["rerank"] == .5 and n["sim"] == .99
                   for n in result.semantik_vorschlaege[0]["normen"])


def test_public_treffer_pipeline_applies_scope_before_both_bge_stages(resources):
    ranker = InstructionReranker()
    result = run(resources, profile="treffer", laws=("EStG",), reranker=ranker)
    assert ranker.events == ["bge", "bge", "qwen"]
    for pairs in [*ranker.vorauswahl.calls, ranker.instruction_calls[0][0]]:
        assert len(pairs) == 1 and pairs[0][1].startswith("norm:EStG:9\n")
    assert [n["id"] for n in result.semantik_vorschlaege[0]["normen"]] == ["norm:EStG:9"]


def test_public_treffer_pipeline_requires_instruction_support_before_scoring(resources):
    ranker = Reranker(.7)
    ranker.vorauswahl = Reranker(.9)
    with pytest.raises(SemanticResourcesUnavailable, match="Instruktionsunterstützung"):
        run(resources, profile="treffer", reranker=ranker)
    assert ranker.calls == [] and ranker.vorauswahl.calls == []
