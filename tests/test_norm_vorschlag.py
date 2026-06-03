"""Norm-Vorschlag (LLM-Einordnung) – Existenzprüfung im Graph, kein Mapping.

Diese Tests isolieren den LLM-Pfad: die Kandidaten-Spans liegen bewusst auf
NEUTRALEM Text, den weder der wörtliche Katalog noch die Beschreibungs-Muster
(Schicht 1b) beanspruchen – sonst würde der deterministische Treffer den
LLM-Kandidaten (korrekt) deduplizieren.
"""

import pytest

from risk_layer.catalog import lade_katalog
from risk_layer.graph import NormGraph
from risk_layer.gii import NormKnoten
from risk_layer.pipeline import Pipeline
from risk_layer.schicht2 import Kandidat


class StubProvider:
    def __init__(self, kandidaten):
        self._k = kandidaten

    def kandidaten(self, text, bekannte, kontext_normen=None, abweichung=False):
        return [k.__dict__ if hasattr(k, "__dict__") else k for k in self._k]


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    p = tmp_path_factory.mktemp("g") / "g.sqlite"
    knoten = [
        NormKnoten(id="norm:AO:42", law="AO", para="42", absatz=None, nummer=None,
                   titel="Gestaltungsmissbrauch", text="Missbrauch von Gestaltungen."),
    ]
    return NormGraph.build(knoten, p)


def test_vorschlag_existiert_wird_bestaetigt(graph):
    text = "Im Gespräch ging es um eine besondere Konstruktion."
    span = "eine besondere Konstruktion"
    kand = [Kandidat(span_text=span, char_offset=text.index(span),
                     begruendung_kurz="ungewöhnlich?", konfidenz="hoch",
                     norm_vorschlag="§ 42 AO")]
    rep = Pipeline(lade_katalog()).run(text, provider=StubProvider(kand), graph=graph)
    risiko = next(r for r in rep.risiken if r.quelle == "llm")
    assert risiko.norm_vorschlag and risiko.norm_vorschlag[0]["im_korpus"] is True
    assert risiko.norm_vorschlag[0]["id"] == "norm:AO:42"


def test_ohne_graph_kein_falsches_bestaetigen():
    text = "Der Vorgang erscheint insgesamt ungewöhnlich."
    span = "insgesamt ungewöhnlich"
    kand = [Kandidat(span_text=span, char_offset=text.index(span),
                     begruendung_kurz="unklar", konfidenz="niedrig",
                     norm_vorschlag="§ 9999 ZZZ")]
    rep = Pipeline(lade_katalog()).run(text, provider=StubProvider(kand))
    risiko = next(r for r in rep.risiken if r.quelle == "llm")
    assert risiko.norm_vorschlag == [] or risiko.norm_vorschlag[0]["im_korpus"] is False


def test_norm_vorschlag_kandidat_dataclass_feld():
    k = Kandidat(span_text="x", char_offset=0, begruendung_kurz="y",
                 konfidenz="niedrig", norm_vorschlag="§ 42 AO")
    assert k.norm_vorschlag == "§ 42 AO"
