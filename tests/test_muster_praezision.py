"""Wortbezogene Ausnahmen vermeiden Fehlalarme ohne andere Signale zu verlieren."""

import pytest

from risk_layer.catalog import Katalog, lade_katalog
from risk_layer.models import Katalogeintrag
from risk_layer.muster import MusterMatcher
from risk_layer.pipeline import Pipeline


KASSENMANGEL = "ao_buchfuehrungs_kassenmangel"


@pytest.fixture(scope="module")
def matcher():
    return MusterMatcher(lade_katalog())


@pytest.mark.parametrize("text", [
    "Die Belege sind vollständig und fehlerfrei.",
    "Die Belege enthalten fehlerfreie Angaben.",
    "Es handelt sich um mangelfreie Belege.",
    "Die Belege sind mangelfrei.",
    "Die Belege sind mängelfrei.",
    "Bei den mängelfreien Belegen ist alles vollständig.",
    "Die Belege sind FEHLERFREI.",
    "Die Belege weisen keine Mängel auf.",
    "Die Buchführung ist ohne Mängel.",
    "Die Belege sind frei von Mängeln.",
    "Das Kassenbuch weist keinen erkennbaren Fehler auf.",
    "Bei den Aufzeichnungen zeigen sich keinerlei wesentliche Mängel.",
])
def test_saubere_befunde_loesen_keinen_mangel_aus(matcher, text):
    assert KASSENMANGEL not in {t.begriff_id for t in matcher.match(text)}


@pytest.mark.parametrize("text", [
    "Die Belege fehlen.",
    "Die Belege fehlten.",
    "Ein Beleg fehle nach Auskunft des Betriebs.",
    "Die fehlenden Belege werden nachgereicht.",
    "Die Belege enthalten einen Fehler.",
    "Die Belege sind fehlerhaft.",
    "Die Belege sind mangelhaft.",
    "Die Belege weisen einen Mangel auf.",
    "Die Belege weisen Mängel auf.",
    "Die Belege sind mit Mängeln behaftet.",
    "Die Belege sind fehlerfrei, weitere Aufzeichnungen fehlen.",
    "Die Belege sind mangelfrei, weitere Aufzeichnungen sind mangelhaft.",
    "Die Belege sind mängelfrei, weitere Aufzeichnungen weisen Mängel auf.",
    "Die Belege sind nicht fehlerfrei.",
    "Die Belege sind nicht mangelfrei.",
    "Die Belege sind nicht mängelfrei.",
    "Die Belege sind keineswegs fehlerfrei.",
    "Die Belege sind keineswegs mangelfrei.",
    "Die Belege sind keineswegs mängelfrei.",
    "Es handelt sich um nicht fehlerfreie Belege.",
    "Die keineswegs mangelfreien Belege liegen vor.",
    "Die Belege sind nicht\n  mängelfrei.",
    "Die Belege sind nicht immer fehlerfrei.",
    "Die Belege sind nicht immer mangelfrei.",
    "Die Belege sind nicht immer mängelfrei.",
    "Die Belege sind nicht durchgängig fehlerfrei.",
    "Die Belege sind nicht frei von Mängeln.",
    "Die Belege sind nicht ohne Fehler.",
    "Die Belege weisen keine Mängel auf, aber weitere Aufzeichnungen fehlen.",
])
def test_mangelhinweise_bleiben_erkannt(matcher, text):
    assert KASSENMANGEL in {t.begriff_id for t in matcher.match(text)}


def test_ausgeschlossene_woerter_erweitern_den_fund_nicht(matcher):
    text = "İ: Fehlerfreie Belege fehlen trotz mangelfreier Zusatzangaben."
    treffer = [t for t in matcher.match(text) if t.begriff_id == KASSENMANGEL]
    assert len(treffer) == 1
    assert treffer[0].start == text.index("Belege")
    assert treffer[0].end == text.index("fehlen") + len("fehl")
    assert text[treffer[0].start:treffer[0].end] == treffer[0].matched_text


def test_ausschluss_gilt_nur_fuer_seine_spec():
    katalog = Katalog("test", [Katalogeintrag(
        id="test", begriff_kanonisch="Test", ausloeser=(
            {"gruppen": [["beleg"], ["fehl"]], "ausschlusswoerter": ["fehlerfrei"]},
            {"gruppen": [["beleg"], ["fehlerfrei"]]},
        ),
    )])
    treffer = MusterMatcher(katalog).match("Die Belege sind fehlerfrei.")
    assert len(treffer) == 1
    assert treffer[0].matched_text == "Belege sind fehlerfrei"


def test_ausschluss_blockiert_keine_explizite_phrase_davor():
    katalog = Katalog("test", [Katalogeintrag(
        id="test", begriff_kanonisch="Test", ausloeser=({
            "gruppen": [["beleg"], ["nicht fehlerfrei"]],
            "ausschlusswoerter": ["fehlerfrei"],
        },),
    )])
    treffer = MusterMatcher(katalog).match("Der Beleg ist nicht fehlerfrei.")
    assert len(treffer) == 1
    assert treffer[0].matched_text == "Beleg ist nicht fehlerfrei"


@pytest.mark.parametrize("text, erwartet", [
    ("Die Belege sind vollständig und fehlerfrei.", False),
    ("Die Belege weisen Mängel auf.", True),
    ("Die Belege sind fehlerfrei, weitere Aufzeichnungen fehlen.", True),
    ("Die Belege sind nicht fehlerfrei.", True),
    ("Die Belege weisen keine Mängel auf.", False),
    ("Die Belege sind nicht durchgängig fehlerfrei.", True),
])
def test_pipeline_uebernimmt_die_korrigierten_muster(text, erwartet):
    report = Pipeline(lade_katalog()).run(text)
    assert (KASSENMANGEL in {k.eintrag.id for k in report.karten}) is erwartet
