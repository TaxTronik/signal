"""Pipeline-Restbefunde: Prompt-Dedup, einmalige Norm-Prüfung, defensiver Rerank-Default.

Drei punktuelle Verhaltens-/Robustheitsgarantien der Pipeline:
  * ``_kandidaten_normen`` dedupliziert Trigger-Kandidaten gegen bereits zitierte
    Normen – dieselbe Norm darf nicht doppelt in den LLM-Prompt.
  * ``_pruefe_norm_vorschlag`` läuft je LLM-Kandidat genau EINMAL (keine doppelten
    Graph-Lookups für behaltene Karten-Überlapper im Abweichungs-Modus).
  * Semantik-Vorschläge ohne ``rerank``-Score fallen DURCH (kein Phantom-Default
    1.0, der die Abstention aushebeln würde).
Zusätzlich eine Regression für den in ``_streit_verstaerker`` extrahierten Block.
"""

import pytest

from risk_layer.catalog import lade_katalog
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.pipeline import Pipeline, _kandidaten_normen
from risk_layer.schicht2 import Kandidat


class StubProvider:
    def __init__(self, kandidaten):
        self._k = kandidaten

    def kandidaten(self, text, bekannte, kontext_normen=None, abweichung=False):
        return [k.__dict__ if hasattr(k, "__dict__") else k for k in self._k]


class GraphStub:
    """Bestätigt jede Norm – reicht für die Prompt-Zeilen von _kandidaten_normen."""

    def aufloesen(self, norm_id, **kw):
        return {"gefunden": True, "titel": "Titel"}


class TriggerStub:
    def __init__(self, kandidaten):
        self._k = kandidaten

    def kandidaten(self, text, k=8, **kw):
        return self._k[:k]


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    p = tmp_path_factory.mktemp("g") / "g.sqlite"
    knoten = [
        NormKnoten(id="norm:KStG:14", law="KStG", para="14", absatz=None, nummer=None,
                   titel="Organgesellschaft", text="Verpflichtung zur Gewinnabführung."),
    ]
    return NormGraph.build(knoten, p)


# --- _kandidaten_normen: Dedup Zitat- vs. Trigger-Zweig ---------------------

def test_kandidaten_normen_trigger_dedupliziert_gegen_zitate():
    # § 42 AO ist im Text ZITIERT (Zitat-Zweig) UND kommt zusätzlich (doppelt!)
    # aus dem Trigger-Index – er darf trotzdem nur EINMAL im Prompt landen.
    text = "Nach § 42 AO ist der Vorgang zu würdigen."
    trigger = TriggerStub([
        {"norm_id": "norm:AO:42", "zitat": "§ 42 AO", "titel": "Gestaltungsmissbrauch"},
        {"norm_id": "norm:AO:42", "zitat": "§ 42 AO", "titel": "Gestaltungsmissbrauch"},
        {"norm_id": "norm:HGB:257", "zitat": "§ 257 HGB", "titel": "Aufbewahrung"},
    ])
    zeilen, seen = _kandidaten_normen(text, GraphStub(), trigger)
    assert sum("§ 42 AO" in z for z in zeilen) == 1          # keine Prompt-Dublette
    assert len(zeilen) == len(set(zeilen)) == 2
    assert {"norm:AO:42", "norm:HGB:257"} <= seen            # beide gelten als gestützt


def test_kandidaten_normen_trigger_ohne_zitate_bleibt_vollstaendig():
    trigger = TriggerStub([
        {"norm_id": "norm:AO:146", "zitat": "§ 146 AO", "titel": "Ordnungsvorschriften"},
        {"norm_id": "norm:HGB:257", "zitat": "§ 257 HGB", "titel": "Aufbewahrung"},
    ])
    zeilen, seen = _kandidaten_normen("Text ohne Normzitat.", None, trigger)
    assert zeilen == ["§ 146 AO – Ordnungsvorschriften", "§ 257 HGB – Aufbewahrung"]
    assert seen == {"norm:AO:146", "norm:HGB:257"}


# --- _pruefe_norm_vorschlag: genau EIN Graph-Durchlauf je Kandidat ----------

def _zaehlende_pruefung(monkeypatch):
    import risk_layer.pipeline as P
    original = P._pruefe_norm_vorschlag
    aufrufe: list[str] = []

    def zaehlend(nv, graph, gestuetzt_paras=None):
        aufrufe.append(nv)
        return original(nv, graph, gestuetzt_paras)

    monkeypatch.setattr(P, "_pruefe_norm_vorschlag", zaehlend)
    return aufrufe


def test_norm_vorschlag_bei_behaltenem_ueberlapper_nur_einmal_geprueft(monkeypatch, graph):
    aufrufe = _zaehlende_pruefung(monkeypatch)
    # Kandidat überlappt die Katalog-Karte nur TEILWEISE (nicht ko-extensiv) und
    # trägt ein fremdes Konzept -> wird BEHALTEN und getaggt. Vorher liefen die
    # Graph-Lookups dafür doppelt (Abweichungs-Vermerk + Risikomarkierung).
    text = "Die organisatorische Eingliederung ist im Streitfall gegeben."
    span = "Eingliederung ist im Streitfall"
    kand = [Kandidat(span_text=span, char_offset=text.index(span),
                     begruendung_kurz="kapitalertragsteuerliche Zurechnung offen",
                     konfidenz="hoch", norm_vorschlag="§ 14 KStG")]
    rep = Pipeline(lade_katalog()).run(text, provider=StubProvider(kand),
                                       graph=graph, abweichung=True)
    assert aufrufe.count("§ 14 KStG") == 1                   # vorher: 2
    # Verhalten unverändert: Abweichung vermerkt UND Marke trägt den Vorschlag.
    karte = next(k for k in rep.karten if k.eintrag.id == "ust_organisatorische_eingliederung")
    assert karte.abweichung and karte.abweichung[0]["llm_norm_id"] == "norm:KStG:14"
    risiko = next(r for r in rep.risiken if r.quelle == "llm")
    assert risiko.norm_vorschlag[0]["id"] == "norm:KStG:14"
    assert risiko.norm_vorschlag[0]["im_korpus"] is True
    assert risiko.signale["ueberlagert_karte"] == "ust_organisatorische_eingliederung"


def test_norm_vorschlag_ohne_ueberlapp_weiterhin_einmal_geprueft(monkeypatch, graph):
    aufrufe = _zaehlende_pruefung(monkeypatch)
    text = "Der Vorgang erscheint insgesamt ungewöhnlich strukturiert."
    span = "ungewöhnlich strukturiert"
    kand = [Kandidat(span_text=span, char_offset=text.index(span),
                     begruendung_kurz="unklare Einordnung", konfidenz="niedrig",
                     norm_vorschlag="§ 14 KStG")]
    rep = Pipeline(lade_katalog()).run(text, provider=StubProvider(kand), graph=graph)
    assert aufrufe.count("§ 14 KStG") == 1
    risiko = next(r for r in rep.risiken if r.quelle == "llm")
    assert risiko.norm_vorschlag[0]["id"] == "norm:KStG:14"


# --- Semantik-Vorschläge: fehlender rerank-Score fällt durch ----------------

class EmbeddingStub:
    def __init__(self, marken):
        self._m = marken

    def kandidaten_pro_satz(self, text, encoder, k=5, min_sim=0.5):
        return self._m


def _embedding_marken():
    return [{"start": 0, "end": 9, "text": "Ein Satz.",
             "normen": [{"norm_id": "norm:KStG:14", "zitat": "§ 14 KStG",
                         "titel": "Organgesellschaft", "sim": 0.6}]}]


def test_semantik_ohne_rerank_score_wird_nicht_durchgelassen(monkeypatch, graph):
    import risk_layer.pipeline as P
    monkeypatch.delenv("RISK_LAYER_RERANK_SCHWELLE", raising=False)
    # Simulierter künftiger score-loser Pfad: rerank_normen liefert KEIN "rerank".
    # Defensive Erwartung: NICHT durchlassen (kein Phantom-Default 1.0).
    monkeypatch.setattr(P, "rerank_normen",
                        lambda rr, q, normen, doc_fn, k: [dict(n) for n in normen])
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, embedding=EmbeddingStub(_embedding_marken()),
        embedding_encoder=object(), reranker=object())
    assert rep.semantik_vorschlaege == []


def test_semantik_mit_rerank_score_passiert_die_schwelle(monkeypatch, graph):
    import risk_layer.pipeline as P
    monkeypatch.delenv("RISK_LAYER_RERANK_SCHWELLE", raising=False)
    monkeypatch.setattr(P, "rerank_normen",
                        lambda rr, q, normen, doc_fn, k: [{**n, "rerank": 0.5} for n in normen])
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, embedding=EmbeddingStub(_embedding_marken()),
        embedding_encoder=object(), reranker=object())
    assert rep.semantik_vorschlaege
    norm = rep.semantik_vorschlaege[0]["normen"][0]
    assert norm["id"] == "norm:KStG:14" and norm["rerank"] == 0.5


def test_semantik_unter_schwelle_bleibt_abstention(monkeypatch, graph):
    import risk_layer.pipeline as P
    monkeypatch.delenv("RISK_LAYER_RERANK_SCHWELLE", raising=False)
    monkeypatch.setattr(P, "rerank_normen",
                        lambda rr, q, normen, doc_fn, k: [{**n, "rerank": 0.05} for n in normen])
    rep = Pipeline(lade_katalog()).run(
        "Ein Satz.", graph=graph, embedding=EmbeddingStub(_embedding_marken()),
        embedding_encoder=object(), reranker=object())
    assert rep.semantik_vorschlaege == []                    # 0.05 < Schwelle 0.10


# --- Regression: Streit-Block nach Extraktion unverändert -------------------

def test_streit_verstaerker_nach_extraktion_unveraendert():
    text = "Die organisatorische Eingliederung ist streitig."
    rep = Pipeline(lade_katalog()).run(text, streit=True)
    marke = next(r for r in rep.risiken if r.quelle == "streit")
    assert marke.ist_streitig and marke.streit_signal == "streitig"
    assert marke.matched_text == "streitig"
    karte = next(k for k in rep.karten if k.eintrag.id == "ust_organisatorische_eingliederung")
    assert karte.ist_streitig and karte.streit_signal == "streitig"
