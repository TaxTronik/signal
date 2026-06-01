"""Edge-Case: Text umschreibt die Begriffe, benennt sie aber NICHT.

Zeigt die Arbeitsteilung der Architektur:
  * Schicht 1 (deterministisch) findet per Konstruktion NICHTS – die Begriffe
    stehen nicht wörtlich im Text (recall=100 % gilt für GENANNTE Begriffe).
  * Schicht 2 (LLM, hier gemockt) erkennt die umschriebenen Stellen; nach dem
    Span-Gate werden sie als unknown_risiko eskaliert.

Das ist genau die Daseinsberechtigung von Schicht 2 (/LF90/, /LF130/).
"""

import json

from risk_layer.catalog import lade_katalog
from risk_layer.llm import LlamaServerProvider
from risk_layer.models import Status
from risk_layer.pipeline import Pipeline


def _text():
    return open("examples/beispiel_beschreibung.txt", encoding="utf-8").read()


def test_schicht1_findet_nichts_im_beschreibungstext():
    report = Pipeline(lade_katalog()).run(_text())
    assert report.karten == []          # kein wörtlicher Katalogbegriff vorhanden


class MockBeschreibungsLlama(LlamaServerProvider):
    """Simuliert ein LLM, das die UMSCHRIEBENEN Stellen erkennt."""

    SPANS = [
        "alle Anteile und damit sämtliche Stimmen",   # ~ finanzielle Eingliederung
        "von derselben Person geführt",               # ~ organisatorische Eingliederung
        "liefert ihre Erzeugnisse fast vollständig",  # ~ wirtschaftliche Eingliederung
    ]

    def __init__(self):
        super().__init__(modell="mock", seed=0, pruefe_verfuegbar=False)

    def _infer(self, segment_text: str, bekannte_begriffe) -> str:
        kand = []
        for s in self.SPANS:
            off = segment_text.find(s)
            if off >= 0:
                kand.append({"span_text": s, "char_offset": off,
                             "begruendung_kurz": "umschriebener unbestimmter Rechtsbegriff",
                             "konfidenz": "hoch"})
        return json.dumps({"kandidaten": kand})


def test_schicht2_eskaliert_umschriebene_stellen():
    report = Pipeline(lade_katalog()).run(_text(), provider=MockBeschreibungsLlama())
    unknown = [r for r in report.risiken if r.status is Status.UNKNOWN_RISIKO]
    # die umschriebenen Stellen werden – nach Span-Verifikation – eskaliert
    assert unknown, "Schicht 2 sollte umschriebene Begriffe als unknown_risiko surfacen"
    treffer_texte = " ".join(r.matched_text for r in unknown)
    assert "sämtliche Stimmen" in treffer_texte or "derselben Person" in treffer_texte
    # jede eskalierte Stelle steht wörtlich im Text (Span-Treue /AB30/)
    t = _text()
    for r in unknown:
        assert t[r.start:r.end] == r.matched_text
