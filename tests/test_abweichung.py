"""Abweichungs-Check: LLM wird über Katalog-Treffer gelegt; abweichende Norm = Hinweis.

Der Katalog bleibt autoritativ – das LLM liefert nur eine unabhängige Zweitmeinung,
deren Abweichung sichtbar gemacht wird (zu prüfen, keine Übersteuerung).
"""

import json

from risk_layer.catalog import lade_katalog
from risk_layer.llm import LlamaServerProvider
from risk_layer.pipeline import Pipeline

TEXT = "Die organisatorische Eingliederung ist im Streitfall gegeben."


class _Mock(LlamaServerProvider):
    def __init__(self):
        super().__init__(modell="mock", seed=0, pruefe_verfuegbar=False)

    def _infer(self, segment_text, bekannte):
        s = "organisatorische Eingliederung"
        return json.dumps({"kandidaten": [{
            "span_text": s, "char_offset": segment_text.index(s),
            "begruendung_kurz": "andere Einordnung", "konfidenz": "hoch",
            "norm_vorschlag": "§ 14 KStG",      # abweichend vom Katalog (§ 2 UStG)
        }]})


def test_abweichung_wird_an_katalogkarte_vermerkt():
    rep = Pipeline(lade_katalog()).run(TEXT, provider=_Mock(), abweichung=True)
    karte = next(k for k in rep.karten if k.eintrag.id == "ust_organisatorische_eingliederung")
    assert karte.abweichung, "Abweichung sollte vermerkt sein"
    assert karte.abweichung[0]["llm_norm_id"] == "norm:KStG:14"
    assert "norm:UStG:2" in karte.abweichung[0]["katalog_norm"]
    # die überlappende Stelle erzeugt KEINE separate unknown-Marke
    assert not any("organisatorische Eingliederung" in r.matched_text for r in rep.risiken)


def test_ohne_abweichungsmodus_kein_vermerk_und_kein_doppel():
    rep = Pipeline(lade_katalog()).run(TEXT, provider=_Mock(), abweichung=False)
    karte = next(k for k in rep.karten if k.eintrag.id == "ust_organisatorische_eingliederung")
    assert karte.abweichung == []
    assert rep.risiken == []      # überlappender Kandidat wird verworfen (kein Doppel)
