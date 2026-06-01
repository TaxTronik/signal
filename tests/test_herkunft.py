"""Layer-/Herkunfts-Kennzeichnung – der epistemische Vertrag der Haftungssoftware.

Jede Markierung muss ausweisen, WOHER sie stammt: deterministisch wörtlich
(Schicht 1) vs. interpretiert (Schicht 2 LLM/Heuristik) vs. Berater.
"""

import json

from risk_layer.catalog import lade_katalog
from risk_layer.llm import LlamaServerProvider
from risk_layer.models import Status
from risk_layer.pipeline import Pipeline


def _karte(report, begriff_id):
    return next(k for k in report.karten if k.eintrag.id == begriff_id)


def test_schicht1_treffer_ist_woertlich_keine_interpretation():
    rep = Pipeline(lade_katalog()).run("Die organisatorische Eingliederung ist gegeben.")
    h = _karte(rep, "ust_organisatorische_eingliederung").herkunft()
    assert h["schicht"] == "1"
    assert h["interpretation"] is False
    assert "wörtlich" in h["kurz"]


def test_luecke_ist_woertlich_ohne_definition():
    rep = Pipeline(lade_katalog()).run("Es bedarf einer Gesamtwürdigung der tatsächlichen Verhältnisse.")
    luecke = next(k for k in rep.karten if k.status is Status.LUECKE)
    h = luecke.herkunft()
    assert h["schicht"] == "1" and h["interpretation"] is False
    assert "ohne Definition" in h["kurz"]      # „wörtlich im Gesetz, aber nicht erklärt"


class _MockLlama(LlamaServerProvider):
    def __init__(self):
        super().__init__(modell="mock", seed=0, pruefe_verfuegbar=False)

    SPAN = "dieselbe Person kann jede Entscheidung verhindern"

    def _infer(self, segment_text, bekannte):
        off = segment_text.find(self.SPAN)
        return json.dumps({"kandidaten": [{
            "span_text": self.SPAN,
            "char_offset": off, "begruendung_kurz": "organisatorische Eingliederung",
            "konfidenz": "hoch"}]})


def test_schicht2_llm_ist_interpretation_und_als_layer2_gekennzeichnet():
    txt = "Im Konzern führt eine Holding, und dieselbe Person kann jede Entscheidung verhindern."
    rep = Pipeline(lade_katalog()).run(txt, provider=_MockLlama())
    r = next(r for r in rep.risiken if r.status is Status.UNKNOWN_RISIKO)
    h = r.herkunft()
    assert r.quelle == "llm"
    assert h["schicht"] == "2" and h["interpretation"] is True
    assert h["konfidenz"] == "hoch"
    # Titel = das interpretierte Konzept; Beleg = wörtlich im Text
    assert r.titel == "organisatorische Eingliederung"
    assert txt[r.start:r.end] == r.matched_text
