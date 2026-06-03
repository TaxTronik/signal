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


def test_email_und_telefon_werden_ersetzt():
    # § 203: E-Mail/Telefon sind identifizierende Angaben und muessen geschwaerzt werden.
    r = anonymisiere("Kontakt: max.mustermann@example.com oder 0151 23456789.")
    typen = {e.typ for e in r.ersetzungen}
    assert "email" in typen and "telefon" in typen
    assert "max.mustermann@example.com" not in r.abstrahiert
    assert "0151 23456789" not in r.abstrahiert


def test_telefon_stiehlt_keine_betraege_oder_daten():
    # Telefon laeuft zuletzt und hat kein '.'/',' in der Klasse -> Betrag/Datum bleiben deren Muster.
    r = anonymisiere("Am 1. April 2026 wurden 1.250.000 EUR gezahlt.")
    typen = {e.typ for e in r.ersetzungen}
    assert "telefon" not in typen
    assert "datum" in typen and "betrag" in typen


def test_volle_anschrift_wird_anonymisiert():
    # § 203: Strasse + Hausnummer + PLZ + Ort sind identifizierend.
    r = anonymisiere("Wohnhaft Johannes-Gutwein Str. 8, 36304 Alsfeld, seit 2020.")
    assert "Johannes-Gutwein Str. 8" not in r.abstrahiert
    assert "36304 Alsfeld" not in r.abstrahiert
    assert "adresse" in {e.typ for e in r.ersetzungen}


def test_angehaengte_strasse_und_plz_standalone():
    r = anonymisiere("Kanzlei in der Hauptstraße 5; Sitz in 60311 Frankfurt.")
    assert "Hauptstraße 5" not in r.abstrahiert
    assert "60311 Frankfurt" not in r.abstrahiert


def test_adresse_stiehlt_keine_rechtsbegriffe_oder_betraege():
    r = anonymisiere("Die organisatorische Eingliederung; gezahlt wurden 1.250.000 EUR.")
    typen = {e.typ for e in r.ersetzungen}
    assert "organisatorische Eingliederung" in r.abstrahiert
    assert "betrag" in typen and "adresse" not in typen
