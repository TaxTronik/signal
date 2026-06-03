"""Smart-Windowing (Coverage-Preserving LLM-Gating).

Schicht 2 läuft absatzweise; deterministisch GESÄTTIGTE Fenster werden
übersprungen, blinde/teilbedeckte/streitige laufen durchs LLM. Geprüft werden
die drei Leitplanken: blind wird NIE übersprungen, 'voll abgedeckt' ist
konservativ, neue Spans gelten nur im Hauptfenster (Offsets korrekt geshiftet).
"""

import json

from risk_layer.catalog import lade_katalog
from risk_layer.llm import LlamaServerProvider
from risk_layer.pipeline import (
    Pipeline, _absatz_grenzen, _coverage_maske, _haupt_fenster, _mit_kontext,
    _wort_unbedeckt,
)


def test_absatz_grenzen_und_buendelung():
    text = "Erster Absatz hier.\n\nZweiter Absatz da.\n\nDritter Block."
    abs_ = _absatz_grenzen(text)
    assert len(abs_) == 3
    assert text[abs_[0][0]:abs_[0][1]] == "Erster Absatz hier."
    assert text[abs_[2][0]:abs_[2][1]] == "Dritter Block."
    # kleines Budget -> jeder Absatz ein Fenster; großes Budget -> alles eins
    assert len(_haupt_fenster(abs_, max_chars=5)) == 3
    assert _haupt_fenster(abs_, max_chars=10_000) == [(abs_[0][0], abs_[2][1])]


def test_mit_kontext_erweitert_um_nachbarabsaetze():
    abs_ = [(0, 10), (12, 20), (22, 30), (32, 40)]
    assert _mit_kontext(22, 30, abs_) == (12, 40)   # mittleres Fenster: beide Nachbarn
    assert _mit_kontext(0, 10, abs_) == (0, 20)      # erstes: nur rechter Nachbar


def test_coverage_maske_konservativ():
    text = "abcdef ghijk"
    maske = _coverage_maske(len(text), [(0, 6)])      # nur "abcdef" abgedeckt
    assert not _wort_unbedeckt(0, 6, maske, text)     # gesättigt
    assert _wort_unbedeckt(0, len(text), maske, text)  # "ghijk" offen -> nicht gesättigt


class _WinMock(LlamaServerProvider):
    """Markiert eine feste Phrase, sobald sie im (Fenster-)Text steht."""
    PHRASE = "ungewöhnliche Briefkastenkonstruktion"

    def __init__(self):
        super().__init__(modell="mock", seed=0, pruefe_verfuegbar=False)

    def _infer(self, segment_text, bekannte_begriffe):
        if self.PHRASE in segment_text:
            return json.dumps({"kandidaten": [{
                "span_text": self.PHRASE, "char_offset": segment_text.index(self.PHRASE),
                "begruendung_kurz": "auffällige Gestaltung", "konfidenz": "hoch"}]})
        return json.dumps({"kandidaten": []})


def test_smart_windowing_skippt_gesaettigt_und_shiftet_offset():
    # Absatz 1 = exakt ein Katalogbegriff (deterministisch gesättigt -> Skip).
    # Absatz 2 = blind, enthält die LLM-Phrase -> muss am ORIGINAL-Offset landen.
    text = ("organisatorische Eingliederung\n\n"
            "Hier liegt eine ungewöhnliche Briefkastenkonstruktion vor.")
    # kleines Budget -> Absätze bleiben getrennte Fenster (sonst mergen die kurzen)
    rep = Pipeline(lade_katalog()).run(text, provider=_WinMock(),
                                       smart_windowing=True, fenster_max_chars=20)

    prov = rep.schicht2_info["windowing"]
    assert prov[0]["state"] == "covered_full" and prov[0]["llm"] is False  # Leitplanke 2
    assert any(p["llm"] for p in prov)                                      # Leitplanke 1

    funde = [r for r in rep.risiken if r.matched_text == _WinMock.PHRASE]
    assert funde, "blinde LLM-Phrase muss als Risiko erscheinen"
    assert funde[0].start == text.index(_WinMock.PHRASE)   # korrekt auf Original geshiftet


def test_smart_windowing_blind_wird_nie_uebersprungen():
    # Ein rein blinder Text (kein Katalog-/Trigger-Treffer) MUSS komplett ans LLM.
    text = "Eine ungewöhnliche Briefkastenkonstruktion ohne jeden Fachbegriff."
    rep = Pipeline(lade_katalog()).run(text, provider=_WinMock(), smart_windowing=True)
    prov = rep.schicht2_info["windowing"]
    assert all(p["llm"] for p in prov)                     # nichts übersprungen
    assert any(r.matched_text == _WinMock.PHRASE for r in rep.risiken)
