"""Tests für den Schicht-1-Matcher: Morphologie-Toleranz & Determinismus."""

import pytest

from risk_layer import ingest as ing
from risk_layer.catalog import lade_katalog
from risk_layer.matcher import Schicht1Matcher


@pytest.fixture(scope="module")
def matcher():
    return Schicht1Matcher(lade_katalog())


def _ids(matcher, text):
    res = ing.ingest(text)
    return [t.begriff_id for t in matcher.match(res)]


def test_kanonische_form_wird_erkannt(matcher):
    ids = _ids(matcher, "Die organisatorische Eingliederung ist gegeben.")
    assert "ust_organisatorische_eingliederung" in ids


def test_flexion_wird_erkannt(matcher):
    # gebeugte/prädikative Formen
    ids = _ids(matcher, "Die Gesellschaft ist organisatorisch eingegliedert.")
    assert "ust_organisatorische_eingliederung" in ids


def test_alias_wird_erkannt(matcher):
    ids = _ids(matcher, "Es bestand eine personelle Verflechtung.")
    assert "ust_organisatorische_eingliederung" in ids


def test_steuerartenuebergreifend_mehrfachtreffer(matcher):
    # 'finanzielle Eingliederung' lebt in UStG UND KStG → beide Norm-Anker
    ids = _ids(matcher, "Die finanzielle Eingliederung lag vor.")
    assert "ust_finanzielle_eingliederung" in ids


def test_offset_treue(matcher):
    text = "Hier liegt ein schädlicher Beteiligungserwerb vor."
    res = ing.ingest(text)
    treffer = matcher.match(res)
    assert treffer, "mindestens ein Treffer erwartet"
    for t in treffer:
        # Offsets verweisen wörtlich auf den Originaltext (/LF20/, /LF70/)
        assert text[t.start:t.end] == t.matched_text
        assert t.matched_text  # nicht leer


def test_determinismus_gleiche_eingabe_gleiche_ausgabe(matcher):
    text = "organisatorische Eingliederung und finanzielle Eingliederung"
    res1 = ing.ingest(text)
    res2 = ing.ingest(text)
    a = [t.as_dict() for t in matcher.match(res1)]
    b = [t.as_dict() for t in matcher.match(res2)]
    assert a == b


def test_kein_treffer_bei_irrelevantem_text(matcher):
    ids = _ids(matcher, "Heute scheint die Sonne und der Kaffee ist gut.")
    assert ids == []
