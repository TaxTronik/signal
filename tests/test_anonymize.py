"""Anonymisierung/Abstraktion (§ 203 StGB)."""

from risk_layer.anonymize import anonymisiere


def test_firmen_kompakt_und_spaced_werden_ersetzt():
    r = anonymisiere("Die M-GmbH und die Alpha SE sowie die Beta GmbH handeln.")
    assert "M-GmbH" not in r.abstrahiert
    assert "Alpha SE" not in r.abstrahiert
    assert "Beta GmbH" not in r.abstrahiert
    assert "[Gesellschaft A]" in r.abstrahiert


def test_platzhalter_sind_stabil():
    r = anonymisiere("Die M-GmbH zahlt. Später zahlt die M-GmbH erneut.")
    # gleiche Firma → gleicher Platzhalter
    assert r.abstrahiert.count("[Gesellschaft A]") == 2
    assert sum(1 for e in r.ersetzungen if e.typ == "firma") == 1


def test_rechtsbegriffe_bleiben_erhalten():
    r = anonymisiere("Die organisatorische Eingliederung der M-GmbH ist strittig.")
    assert "organisatorische Eingliederung" in r.abstrahiert


def test_betrag_und_datum_werden_ersetzt():
    r = anonymisiere("Am 1. April 2026 wurden 1.250.000 EUR überwiesen.")
    typen = {e.typ for e in r.ersetzungen}
    assert "datum" in typen and "betrag" in typen
    assert "2026" not in r.abstrahiert or "[Datum" in r.abstrahiert


def test_steuernummer_und_iban():
    r = anonymisiere("StNr 151/815/08153, IBAN DE89 3704 0044 0532 0130 00.")
    typen = {e.typ for e in r.ersetzungen}
    assert "steuernr" in typen and "iban" in typen
