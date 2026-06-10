"""AnalyseOptionen (pipeline.py): Single Source of Truth für run()-Defaults.

Hintergrund (Review-Fundstelle): CLI und Web mussten die run()-Defaults per
Konvention synchron halten – CLI- und /v1-Ergebnisse divergierten still. Die
Dataclass bündelt die Lauf-Optionen; diese Tests nageln fest:
  (a) Defaults über optionen= sind IDENTISCH zum Alt-Verhalten (Einzel-Kwargs),
  (b) Mischen von optionen + Einzel-Kwargs ist ein harter ValueError
      (kein stilles Gewinnen einer Seite),
  (c) die Dataclass-Defaults selbst = die dokumentierten Defaults.
"""

import dataclasses

import pytest

from risk_layer.catalog import lade_katalog
from risk_layer.pipeline import _FENSTER_MAX_CHARS, AnalyseOptionen, Pipeline

# Fixture-Text mit bekanntem Katalog-Treffer (vgl. test_acceptance) UND einem
# Streit-Signal ("streitig"), damit auch eine Nicht-Default-Option sichtbar wirkt.
TEXT = ("Die organisatorische Eingliederung ist zu prüfen. "
        "Die Kassenführung ist zwischen den Beteiligten streitig.")
# Fester Zeitstempel: erzeugt_am ist sonst die einzige nicht-deterministische
# Stelle eines Laufs – mit Fixwert sind ganze RunReports vergleichbar.
TS = "2026-01-01T00:00:00+00:00"


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(lade_katalog())


def test_defaults_identisch_mit_und_ohne_optionen(pipeline):
    # Gleicher Lauf einmal über die Übergangs-API, einmal über optionen= –
    # die Reports müssen FELDGENAU identisch sein (RunReport ist eine Dataclass,
    # __eq__ vergleicht rekursiv alle Karten/Risiken/Spans).
    alt = pipeline.run(TEXT, erzeugt_am=TS)
    neu = pipeline.run(TEXT, optionen=AnalyseOptionen(erzeugt_am=TS))
    assert alt == neu
    assert alt.karten, "Fixture-Text muss einen Katalog-Treffer liefern"


def test_gesetzte_optionen_wirken_wie_einzel_kwargs(pipeline):
    # Auch NICHT-Default-Werte müssen über beide Wege dasselbe Ergebnis liefern.
    alt = pipeline.run(TEXT, erzeugt_am=TS, streit=True, unknown=True)
    neu = pipeline.run(TEXT, optionen=AnalyseOptionen(erzeugt_am=TS, streit=True,
                                                      unknown=True))
    assert alt == neu
    # Plausibilität: streit=True kam wirklich an (Brennpunkt-Marke vorhanden).
    assert any(r.quelle == "streit" for r in neu.risiken)


def test_misch_verbot_wirft_value_error(pipeline):
    with pytest.raises(ValueError, match="streit"):
        pipeline.run(TEXT, optionen=AnalyseOptionen(), streit=True)


def test_misch_verbot_auch_bei_explizitem_default_wert(pipeline):
    # Sogar das explizite Setzen des DEFAULT-Werts gilt als Mischen: run() kann
    # Absicht nicht von Versehen unterscheiden – stilles Gewinnen wäre genau die
    # Divergenz-Falle, die AnalyseOptionen beseitigt.
    with pytest.raises(ValueError, match="unknown"):
        pipeline.run(TEXT, optionen=AnalyseOptionen(), unknown=False)


def test_ressourcen_neben_optionen_erlaubt(pipeline):
    # Ressourcen-Objekte (graph/berater/provider/…) sind KEINE Optionen – sie
    # haben einen anderen Lebenszyklus und dürfen zusammen mit optionen= kommen.
    rep = pipeline.run(TEXT, optionen=AnalyseOptionen(erzeugt_am=TS),
                       graph=None, provider=None, berater=None)
    assert rep.karten


def test_dataclass_defaults_sind_die_dokumentierten():
    # Die EXAKTEN bisherigen run()-Defaults – wer hier etwas ändert, ändert das
    # Verhalten ALLER Aufrufer (CLI, Web, /v1) und muss es bewusst tun.
    erwartet = {
        "erzeugt_am": None,
        "unknown": False,
        "nutzer": None,
        "abweichung": False,
        "detektor": False,
        "streit": False,
        "smart_windowing": False,
        "fenster_max_chars": _FENSTER_MAX_CHARS,
    }
    ist = {f.name: f.default for f in dataclasses.fields(AnalyseOptionen)}
    assert ist == erwartet
    assert _FENSTER_MAX_CHARS == 600          # dokumentierter Windowing-Default
