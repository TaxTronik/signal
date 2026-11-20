"""Schicht 1.7: RRF-Fusion (Embedding × Trigger-Index) + sim-Rettung (Policy P2).

Verdrahtung der im Eval-Harness bewiesenen Optimierung (benchmarks/
harness-fusion-policy.json: R@5 0,560 → 0,840 gegen den Ist-Default):

  * Die RRF-Mathematik in ``pipeline.rrf_fusion`` ist IDENTISCH zur Referenz-
    Implementierung im Harness (tools/eval_harness.py) — gleiche Eingabe ⇒
    gleiche Reihenfolge/Scores (hier direkt gegeneinander geprüft).
  * Rettungsregel ``reranker.filtere_mit_rettung``: Kandidat überlebt mit
    rerank ≥ Schwelle ODER sim ≥ Rettung; Trigger-only (sim=None) wird NIE
    über sim gerettet; deterministische Ordnung.
  * Env-Schalter: RISK_LAYER_FUSION=aus (Fusion aus, Rettung bleibt) und
    RISK_LAYER_SIM_RETTUNG (0 = Rettung aus = altes P0).
  * Degradationspfade: ohne Reranker bleibt der alte Embedding-only-Pfad
    (Floor 0,50, Top-5, KEIN Trigger-Zugriff); ohne Trigger keine Fusion,
    aber Rerank + Rettung.

Muster wie tests/test_pipeline_restbefunde.py: Stubs statt Modelle, Reranker
über monkeypatch auf ``pipeline.rerank_normen`` (kein Modell-Download).
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import risk_layer.pipeline as P
from risk_layer.catalog import lade_katalog
from risk_layer.embedding import EmbeddingIndex, HashEncoder
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.pipeline import Pipeline, fusion_aktiviert, rrf_fusion
from risk_layer.reranker import SIM_RETTUNG, filtere_mit_rettung, sim_rettung

from tools.eval_harness import fusioniere_treffer, konfig_kandidaten


@pytest.fixture(autouse=True)
def _saubere_env(monkeypatch):
    """Defaults der Schalter — Entwickler-Umgebungen dürfen nicht reinfunken."""
    monkeypatch.delenv("RISK_LAYER_RERANK_SCHWELLE", raising=False)
    monkeypatch.delenv("RISK_LAYER_SIM_RETTUNG", raising=False)
    monkeypatch.delenv("RISK_LAYER_FUSION", raising=False)


# --- RRF-Mathe: Parität zur Harness-Referenz ---------------------------------

def _dense(*paare):
    return [{"norm_id": nid, "sim": sim, "zitat": f"§ {nid}", "titel": nid}
            for nid, sim in paare]


def _trigger(*ids):
    return [{"norm_id": nid, "zitat": f"§ {nid}", "titel": nid, "score": 1.0,
             "trigger": [], "geprueft": True, "quelle": "trigger_index"}
            for nid in ids]


def test_rrf_fusion_identisch_zum_harness():
    dense = _dense(("norm:AO:42", 0.81), ("norm:AO:162", 0.7),
                   ("norm:AO:146", 0.6))
    trigger = _trigger("norm:AO:162", "norm:HGB:257", "norm:EStG:4")
    for k_rrf in (20, 60):
        fused = rrf_fusion(dense, trigger, k_rrf)
        referenz = fusioniere_treffer(dense, [c["norm_id"] for c in trigger],
                                      k_rrf)
        # Gleiche Eingabe ⇒ gleiche Reihenfolge UND gleiche (gerundete) Scores.
        assert [n["norm_id"] for n in fused] == \
            [t["norm_id"] for t in referenz], k_rrf
        assert [n["rrf"] for n in fused] == [t["rrf"] for t in referenz], k_rrf
    # Dense-sim bleibt erhalten; Trigger-only trägt sim=None (Harness: 0,0 —
    # beide Werte können die sim-Rettung nie auslösen).
    by_id = {n["norm_id"]: n for n in rrf_fusion(dense, trigger, 60)}
    assert by_id["norm:AO:42"]["sim"] == 0.81
    assert by_id["norm:HGB:257"]["sim"] is None


def test_rrf_fusion_tiebreak_wie_harness():
    # Beide Rang 1 in je einer Liste ⇒ RRF-Gleichstand ⇒ norm_id entscheidet
    # (exakt der deterministische Tie-Breaker des Harness).
    fused = rrf_fusion(_dense(("y", 0.9)), _trigger("x"), 60)
    assert [n["norm_id"] for n in fused] == ["x", "y"]
    assert [t["norm_id"] for t in fusioniere_treffer(
        _dense(("y", 0.9)), ["x"], 60)] == ["x", "y"]
    assert rrf_fusion([], [], 60) == []


def test_policy_paritaet_zum_harness_p2():
    # Dieselben Kandidaten + Scores durch Harness-Policy P2 (konfig_kandidaten)
    # und durch die Produktions-Funktion filtere_mit_rettung ⇒ gleiche Liste.
    roh = [{"norm_id": "norm:AO:1", "sim": 0.9},
           {"norm_id": "norm:AO:2", "sim": 0.8},
           {"norm_id": "norm:AO:3", "sim": 0.56},
           {"norm_id": "norm:AO:4", "sim": 0.4}]
    scores = {("q", "norm:AO:1"): 0.02, ("q", "norm:AO:2"): 0.9,
              ("q", "norm:AO:3"): 0.04, ("q", "norm:AO:4"): 0.4}
    referenz = konfig_kandidaten(roh, floor=0.0, topk=10, query="q",
                                 rerank_scores=scores, schwelle=0.10,
                                 policy="P2", hoch=0.55)
    kand = [{**n, "rerank": scores[("q", n["norm_id"])]} for n in roh]
    assert [n["norm_id"] for n in filtere_mit_rettung(kand, 0.10, 0.55)] == \
        referenz == ["norm:AO:2", "norm:AO:4", "norm:AO:3", "norm:AO:1"]


# --- Rettungsregel + deterministische Ordnung ---------------------------------

def test_filtere_mit_rettung_regeln():
    kand = [
        {"norm_id": "A", "sim": 0.30, "rerank": 0.50},   # über Schwelle
        {"norm_id": "B", "sim": 0.56, "rerank": 0.04},   # unter Schwelle, sim rettet
        {"norm_id": "C", "sim": None, "rerank": 0.04},   # Trigger-only: KEINE Rettung
        {"norm_id": "D", "sim": 0.54, "rerank": 0.09},   # unter beidem ⇒ fällt
    ]
    assert [n["norm_id"] for n in filtere_mit_rettung(kand, 0.10, 0.55)] == \
        ["A", "B"]
    # Fehlender rerank-Score = -inf (kein Phantom-Default): nur sim kann retten.
    ohne_score = [{"norm_id": "E", "sim": 0.56}, {"norm_id": "F", "sim": 0.40}]
    assert [n["norm_id"] for n in filtere_mit_rettung(ohne_score, 0.10, 0.55)] \
        == ["E"]
    assert filtere_mit_rettung([], 0.10, 0.55) == []


def test_filtere_mit_rettung_deterministische_ordnung():
    kand = [{"norm_id": "B", "sim": 0.60, "rerank": 0.5},
            {"norm_id": "A", "sim": 0.60, "rerank": 0.5},
            {"norm_id": "C", "sim": 0.70, "rerank": 0.5},
            {"norm_id": "D", "sim": 0.90, "rerank": 0.8}]
    out = filtere_mit_rettung(kand, 0.10, 0.55)
    # rerank absteigend, dann sim absteigend, dann norm_id — und unabhängig
    # von der Eingabe-Reihenfolge.
    assert [n["norm_id"] for n in out] == ["D", "C", "A", "B"]
    assert filtere_mit_rettung(list(reversed(kand)), 0.10, 0.55) == out


# --- Env-Schalter --------------------------------------------------------------

def test_sim_rettung_default_und_env(monkeypatch):
    assert sim_rettung() == SIM_RETTUNG == 0.55
    monkeypatch.setenv("RISK_LAYER_SIM_RETTUNG", "0.60")
    assert sim_rettung() == 0.60
    monkeypatch.setenv("RISK_LAYER_SIM_RETTUNG", "quatsch")
    assert sim_rettung() == 0.55                            # robust gegen Müll
    monkeypatch.setenv("RISK_LAYER_SIM_RETTUNG", "0")
    assert sim_rettung() == 0.0


def test_rettung_null_ist_altes_p0():
    # sim_rettung=0 schaltet die Rettung AB (altes P0) — sonst würde sim>=0
    # jeden Kandidaten retten und die Abstention komplett aushebeln.
    kand = [{"norm_id": "A", "sim": 0.9, "rerank": 0.04},
            {"norm_id": "B", "sim": 0.3, "rerank": 0.50}]
    assert [n["norm_id"] for n in filtere_mit_rettung(kand, 0.10, 0.0)] == ["B"]


def test_fusion_aktiviert_env(monkeypatch):
    assert fusion_aktiviert() is True                       # Default: an
    monkeypatch.setenv("RISK_LAYER_FUSION", "aus")
    assert fusion_aktiviert() is False
    monkeypatch.setenv("RISK_LAYER_FUSION", "an")
    assert fusion_aktiviert() is True


# --- Pipeline-Verdrahtung (Stubs statt Modelle) ---------------------------------

class EmbeddingStub:
    def __init__(self, marken):
        self._m = marken
        self.aufrufe: list[dict] = []

    def kandidaten_pro_satz(self, text, encoder, k=5, min_sim=0.5,
                            auch_ohne_treffer=False):
        self.aufrufe.append({"k": k, "min_sim": min_sim})
        return self._m


class TriggerStub:
    def __init__(self, kandidaten):
        self._k = kandidaten
        self.aufrufe: list[dict] = []

    def kandidaten(self, text, k=8, **kw):
        self.aufrufe.append({"text": text, "k": k})
        return self._k[:k]


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    p = tmp_path_factory.mktemp("g") / "g.sqlite"
    knoten = [NormKnoten(id="norm:KStG:14", law="KStG", para="14", absatz=None,
                         nummer=None, titel="Organgesellschaft",
                         text="Verpflichtung zur Gewinnabführung.")]
    return NormGraph.build(knoten, p)


def _marken(sim=0.6):
    return [{"start": 0, "end": 9, "text": "Ein Satz.",
             "normen": [{"norm_id": "norm:KStG:14", "zitat": "§ 14 KStG",
                         "titel": "Organgesellschaft", "sim": sim}]}]


def _rerank_fake(monkeypatch, scores):
    """rerank_normen-Ersatz: Scores je norm_id anhängen + Pool festhalten."""
    gesehen: list[list[dict]] = []

    def fake(rr, q, normen, doc_fn, k):
        gesehen.append([dict(n) for n in normen])
        out = [{**n, "rerank": scores[n["norm_id"]]} for n in normen]
        out.sort(key=lambda n: -n["rerank"])
        return out[:k]

    monkeypatch.setattr(P, "rerank_normen", fake)
    return gesehen


def test_pipeline_fusion_mischt_trigger_in_den_rerank_pool(monkeypatch, graph):
    emb = EmbeddingStub(_marken())
    trig = TriggerStub(_trigger("norm:AO:42"))
    gesehen = _rerank_fake(monkeypatch,
                           {"norm:KStG:14": 0.5, "norm:AO:42": 0.2})
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, trigger=trig, embedding=emb,
        embedding_encoder=object(), reranker=object())
    # Mit Reranker: breiter Pool (Top-20, Floor 0,35) — wie bisher.
    assert emb.aufrufe == [{"k": 20, "min_sim": 0.35}]
    # Trigger wird je SATZ mit Top-20 gefragt (derselbe Text wie das Embedding).
    assert trig.aufrufe == [{"text": "Ein Satz.", "k": 20}]
    # Der Rerank-Pool ist die RRF-fusionierte Liste; Trigger-only trägt sim=None
    # (Tie Rang 1/Rang 1 ⇒ norm_id-Ordnung wie im Harness: AO vor KStG).
    pool = gesehen[0]
    assert [n["norm_id"] for n in pool] == ["norm:AO:42", "norm:KStG:14"]
    assert pool[0]["sim"] is None and pool[1]["sim"] == 0.6
    # Beide über der Schwelle ⇒ beide im Vorschlag, rerank-absteigend; der
    # Trigger-only-Vorschlag zeigt sim=null (sichtbarer Score-Unterschied).
    normen = rep.semantik_vorschlaege[0]["normen"]
    assert [(n["id"], n["rerank"]) for n in normen] == \
        [("norm:KStG:14", 0.5), ("norm:AO:42", 0.2)]
    assert normen[1]["sim"] is None


def test_pipeline_trigger_only_unter_schwelle_faellt(monkeypatch, graph):
    # Trigger-only (sim=None) unter der Schwelle wird NICHT gerettet —
    # Negativ-Abwehr gegen Trigger-Rauschen bleibt (exakt wie im Harness).
    emb = EmbeddingStub(_marken())
    trig = TriggerStub(_trigger("norm:AO:42"))
    _rerank_fake(monkeypatch, {"norm:KStG:14": 0.5, "norm:AO:42": 0.04})
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, trigger=trig, embedding=emb,
        embedding_encoder=object(), reranker=object())
    assert [n["id"] for n in rep.semantik_vorschlaege[0]["normen"]] == \
        ["norm:KStG:14"]


def test_pipeline_rettung_haelt_embedding_kandidat_unter_schwelle(monkeypatch, graph):
    # Kandidat unter der Rerank-Schwelle, aber sim 0,56 >= 0,55 ⇒ bleibt
    # (DIE Regel, die im Harness +28 pp Recall gebracht hat).
    emb = EmbeddingStub(_marken(sim=0.56))
    _rerank_fake(monkeypatch, {"norm:KStG:14": 0.04})
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, embedding=emb,
        embedding_encoder=object(), reranker=object())
    norm = rep.semantik_vorschlaege[0]["normen"][0]
    assert norm["id"] == "norm:KStG:14"
    assert norm["rerank"] == 0.04 and norm["sim"] == 0.56


def test_pipeline_fusion_env_aus_rettung_bleibt(monkeypatch, graph):
    # RISK_LAYER_FUSION=aus: kein Trigger-Zumisch (Pool = nur Embedding),
    # die sim-Rettung wirkt aber weiter (eigener Schalter).
    monkeypatch.setenv("RISK_LAYER_FUSION", "aus")
    emb = EmbeddingStub(_marken(sim=0.56))
    trig = TriggerStub(_trigger("norm:AO:42"))
    gesehen = _rerank_fake(monkeypatch, {"norm:KStG:14": 0.04})
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, trigger=trig, embedding=emb,
        embedding_encoder=object(), reranker=object())
    assert trig.aufrufe == []                       # Trigger nie gefragt
    assert [n["norm_id"] for n in gesehen[0]] == ["norm:KStG:14"]
    assert [n["id"] for n in rep.semantik_vorschlaege[0]["normen"]] == \
        ["norm:KStG:14"]                            # über die Rettung gehalten


def test_pipeline_ohne_reranker_alter_pfad(monkeypatch, graph):
    # Degradationspfad: OHNE Reranker bleibt der heutige Embedding-only-Pfad
    # unverändert (Floor 0,50, Top-5) — und der Trigger wird NICHT zugemischt
    # (Fusion ohne Reranker ist gemessen schädlich: MRR 0,405).
    emb = EmbeddingStub(_marken())
    trig = TriggerStub(_trigger("norm:AO:42"))
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, trigger=trig, embedding=emb,
        embedding_encoder=object(), reranker=None)
    assert emb.aufrufe == [{"k": 5, "min_sim": 0.50}]
    assert trig.aufrufe == []
    normen = rep.semantik_vorschlaege[0]["normen"]
    assert [n["id"] for n in normen] == ["norm:KStG:14"]
    assert normen[0]["rerank"] is None              # ungescort durchgereicht


def test_pipeline_ohne_trigger_rerank_und_rettung_ohne_fusion(monkeypatch, graph):
    # Degradationspfad: Reranker da, Trigger fehlt ⇒ keine Fusion, aber
    # Rerank + Rettung laufen normal.
    emb = EmbeddingStub(_marken(sim=0.56))
    gesehen = _rerank_fake(monkeypatch, {"norm:KStG:14": 0.04})
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, trigger=None, embedding=emb,
        embedding_encoder=object(), reranker=object())
    assert emb.aufrufe == [{"k": 20, "min_sim": 0.35}]
    assert [n["norm_id"] for n in gesehen[0]] == ["norm:KStG:14"]
    assert [n["id"] for n in rep.semantik_vorschlaege[0]["normen"]] == \
        ["norm:KStG:14"]


@pytest.mark.parametrize(("score", "erwartet"), [(0.5, True), (0.04, False)])
def test_pipeline_fusion_prueft_trigger_auch_ohne_dense_treffer(
        monkeypatch, graph, score, erwartet):
    # Echter Embedding-Pfad: alle Kosinuswerte liegen unter dem Floor. Der
    # lexikalische Kandidat muss trotzdem vom Reranker beurteilt werden.
    encoder = HashEncoder(dim=64)
    embedding = EmbeddingIndex(
        np.zeros((1, 64), dtype=np.float32), ["norm:KStG:14"],
        ["§ 14 KStG"], ["Organgesellschaft"], encoder.name)
    trigger = TriggerStub(_trigger("norm:KStG:14"))
    gesehen = _rerank_fake(monkeypatch, {"norm:KStG:14": score})
    text = "  Der ganze Gewinn wird an die Mutter abgeführt.  "

    rep = Pipeline(lade_katalog()).run(
        text, graph=graph, trigger=trigger, embedding=embedding,
        embedding_encoder=encoder, reranker=object())

    assert trigger.aufrufe == [{"text": text.strip().rstrip("."), "k": 20}]
    assert len(gesehen) == 1
    assert [(n["norm_id"], n["sim"]) for n in gesehen[0]] == [
        ("norm:KStG:14", None)]
    assert bool(rep.semantik_vorschlaege) is erwartet
    if erwartet:
        vorschlag = rep.semantik_vorschlaege[0]
        assert text[vorschlag["start"]:vorschlag["end"]] == vorschlag["satz"]
        assert vorschlag["normen"][0]["sim"] is None


def test_pipeline_leerer_dense_pool_ohne_reranker_fragt_keine_trigger(graph):
    encoder = HashEncoder(dim=64)
    embedding = EmbeddingIndex(
        np.zeros((1, 64), dtype=np.float32), ["norm:KStG:14"],
        ["§ 14 KStG"], ["Organgesellschaft"], encoder.name)
    trigger = TriggerStub(_trigger("norm:KStG:14"))

    rep = Pipeline(lade_katalog()).run(
        "Der ganze Gewinn wird an die Mutter abgeführt.", graph=graph,
        trigger=trigger, embedding=embedding, embedding_encoder=encoder)

    assert trigger.aufrufe == []
    assert rep.semantik_vorschlaege == []


@pytest.mark.parametrize(("sim", "rerank", "erwartet"), [
    (0.5496, 0.01, False),       # Rundung darf die sim-Rettung nicht aktivieren.
    (0.4, 0.0996, False),        # Rundung darf die Rerank-Abstention nicht umgehen.
    (0.55555, 0.12345, True),    # Nur die Anzeige wird auf drei Stellen gerundet.
])
def test_pipeline_entscheidet_mit_ungerundeten_scores(graph, sim, rerank, erwartet):
    class Encoder:
        name = "konstant"

        def encode(self, texte):
            return np.tile([1.0, 0.0], (len(texte), 1))

    class Reranker:
        def score(self, paare):
            return [rerank] * len(paare)

    encoder = Encoder()
    embedding = EmbeddingIndex(
        np.array([[sim, np.sqrt(1.0 - sim ** 2)]]), ["norm:KStG:14"],
        ["§ 14 KStG"], ["Organgesellschaft"], encoder.name)

    rep = Pipeline(lade_katalog()).run(
        "Der ganze Gewinn wird an die Mutter abgeführt.", graph=graph,
        embedding=embedding, embedding_encoder=encoder, reranker=Reranker())

    assert bool(rep.semantik_vorschlaege) is erwartet
    if erwartet:
        norm = rep.semantik_vorschlaege[0]["normen"][0]
        assert norm["sim"] == 0.556
        assert norm["rerank"] == 0.123
