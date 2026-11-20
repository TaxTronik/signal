"""Schicht 1b – Beschreibungs-Muster + Streit-Verstärker.

Sichert die Katalog-Optimierung gegen Regression: umschriebene/negative
Sachverhalte werden deterministisch erkannt (ohne Modell), die "Kasse"-Präzision
hält (kein § 257-/Unterstützungskasse-Fehlgriff), und der AO-Edge-Case liefert
genau die fachlich erwarteten Begriffe samt Norm-Ankern.
"""

import pytest

from risk_layer.catalog import lade_katalog
from risk_layer.muster import (MusterMatcher, StreitDetektor, _glied_funde,
                               saetze_mit_offset)
from risk_layer.pipeline import Pipeline

EDGE = (
    "Der Prüfer entdeckt, dass in mehreren Jahren Bareinnahmen nicht in der Kasse\n"
    "erscheinen; Belege fehlen, die Kassenführung ist nicht nachvollziehbar. Weil sich\n"
    "die wirklichen Umsätze nicht mehr feststellen lassen, setzt das Finanzamt die\n"
    "Bemessungsgrundlagen anhand allgemeiner Erfahrungssätze kurzerhand selbst an.\n\n"
    "Die Steuerbescheide waren längst nicht mehr anfechtbar. Nun werden dem Amt\n"
    "Umstände bekannt, die ihm bei der ursprünglichen Veranlagung nicht vorlagen. Ob\n"
    "der Unternehmer die unvollständige Erklärung hätte vermeiden müssen, weil er die\n"
    "Aufzeichnungen erkennbar nachlässig geführt hat, ist zwischen den Beteiligten\n"
    "streitig.\n\n"
    "Die Gesellschaft hatte ihre Lizenzrechte auf eine eigens dafür gegründete Firma\n"
    "in einem Niedrigsteuerland übertragen. Diese Firma verfügt über kein eigenes\n"
    "Personal und über keine Geschäftsräume; sie leitet die eingehenden Zahlungen\n"
    "lediglich weiter. Tatsächlich wurde alles unverändert vom inländischen Betrieb\n"
    "aus gesteuert, der die Rechte auch weiterhin nutzte und über sie verfügte.\n\n"
    "Schließlich soll der Geschäftsführer persönlich für die offenen Beträge einstehen,\n"
    "weil er die fälligen Abgaben trotz ausreichender Mittel über Monate nicht an das\n"
    "Finanzamt weitergeleitet, sondern andere Gläubiger bevorzugt bedient hat.\n"
)


# --- Bausteine ---------------------------------------------------------------

def test_glied_wortanfang_praezision():
    # "kasse" trifft Kassenführung (Wortanfang), NICHT Unterstützungskasse
    assert _glied_funde("Die Kassenführung", "kasse")
    assert not _glied_funde("Eine Unterstützungskasse", "kasse")
    # Stamm-/Präfixtoleranz und Mehrwort-Phrasen (auch über Whitespace/Umbruch)
    assert _glied_funde("Die Belege fehlen.", "fehl")
    assert _glied_funde("ist nicht  nachvollziehbar", "nicht nachvollziehbar")
    assert _glied_funde("nicht in der\nKasse", "nicht in der kasse")


def test_glied_funde_offsets_stabil_bei_laengenaendernder_lowercase():
    # "İ".lower() ist ZWEI Codepoints – mit der alten satz.lower()-Variante wären
    # alle Offsets danach um 1 verschoben. Der Fund muss exakt auf das Originalwort
    # zeigen (Offset-Treue = load-bearing Integrationsgarantie).
    satz = "İstanbul: die Belege fehlen offenkundig"
    funde = _glied_funde(satz, "belege")
    assert funde, "‚belege‘ sollte gefunden werden"
    s, e = funde[0]
    assert satz[s:e].lower() == "belege"


def test_saetze_zerlegen_ueber_zeilenumbruch():
    # Ein einzelnes \n mitten im Satz darf die Ko-Okkurrenz nicht zerreißen
    saetze = saetze_mit_offset("Belege fehlen, die Kasse\nist nicht nachvollziehbar.")
    assert len(saetze) == 1
    # Doppelter Umbruch (Absatz) trennt hingegen
    assert len(saetze_mit_offset("Erster Teil ohne Punkt\n\nzweiter Teil")) == 2


def test_abkuerzungen_zerreissen_satz_nicht():
    # gem. / Abs. / Nr. sind KEINE Satzenden – der Norm-Zitat-Satz bleibt EIN Satz
    text = "Die Kassenführung ist gem. § 146 Abs. 1 Nr. 1 AO nicht nachvollziehbar."
    saetze = saetze_mit_offset(text)
    assert len(saetze) == 1, [s.text for s in saetze]
    # Offset-Treue bleibt erhalten (kein Längenverlust)
    for s in saetze:
        assert text[s.start:s.end] == s.text
    # Einzelbuchstaben-Abkürzungen (z. B. / i. S. d.) trennen ebenfalls nicht,
    # obwohl danach ein Großbuchstabe folgt
    assert len(saetze_mit_offset("Es fehlen z. B. Belege und Aufzeichnungen.")) == 1
    assert len(saetze_mit_offset("Unternehmer i. S. d. § 2 UStG ist der Wirt.")) == 1


def test_abzgl_vor_betrag_ist_keine_satzgrenze():
    # Im echten GPU-Klausurlauf erzeugte „abzgl. 1 % Nebenkosten“ sonst ein
    # isoliertes Fragment mit sachfremden semantischen Vorschlägen.
    text = "Veräußerung für 45 EUR je Stück abzgl. 1 % Nebenkosten."
    saetze = saetze_mit_offset(text)
    assert len(saetze) == 1
    assert text[saetze[0].start:saetze[0].end] == text.rstrip(".")
    report = Pipeline(lade_katalog()).run(
        "Die Belege über 100 EUR abzgl. 10 EUR Rabatt fehlen.")
    assert {k.eintrag.id for k in report.karten} == {"ao_buchfuehrungs_kassenmangel"}


def test_geb_vor_jahreszahl_ist_keine_satzgrenze():
    text = "Die Belege für das Kind (geb. 2007) fehlen."
    assert len(saetze_mit_offset(text)) == 1
    report = Pipeline(lade_katalog()).run(text)
    assert {k.eintrag.id for k in report.karten} == {"ao_buchfuehrungs_kassenmangel"}


def test_echte_satzgrenzen_trennen_weiterhin():
    text = "Die Belege fehlen vollständig. Der Bescheid wurde aufgehoben."
    assert len(saetze_mit_offset(text)) == 2
    # Punkt am Zeilenende ist ebenfalls Satzgrenze
    assert len(saetze_mit_offset("Die Belege fehlen.\nder nächste beginnt klein")) == 2
    # harte Trenner (!?;:) trennen wie bisher immer
    assert len(saetze_mit_offset("Belege fehlen; die Kasse stimmt nicht")) == 2


@pytest.mark.parametrize("text", [
    "Alle Belege sind vorhanden. 2025 fehlten nur die Bürostühle.",
    "„Alle Belege sind vorhanden.“ Fehlende Stühle wurden nachbestellt.",
    '"Alle Belege sind vorhanden." Fehlende Stühle wurden nachbestellt.',
    "(Alle Belege sind vorhanden.) Fehlende Stühle wurden nachbestellt.",
    "Die Belege reichen bis zum 1. Jan. Fehlende Stühle wurden nachbestellt.",
    "Die Belege reichen bis zum 1. 3. Fehlende Stühle wurden nachbestellt.",
])
def test_getrennte_saetze_erzeugen_keinen_kassenmangel(text):
    # Objekt und Mangel gehören zu verschiedenen Aussagen. Jahreszahlen und
    # schließende Anführungszeichen dürfen die Satzgrenze nicht verschlucken.
    saetze = saetze_mit_offset(text)
    assert len(saetze) == 2
    assert all(text[s.start:s.end] == s.text for s in saetze)
    assert not Pipeline(lade_katalog()).run(text).karten


@pytest.mark.parametrize("datum", [
    "1. März", "12. September 2025", "1. Jan. 2025",
    "01.03.2025", "1. 3. 2025", "1. 3.", "1. Jan.",
])
def test_datum_zerreisst_beschreibungsmuster_nicht(datum):
    text = f"Die Belege sind am {datum} nicht nachvollziehbar. Danach endet die Prüfung."
    saetze = saetze_mit_offset(text)
    assert len(saetze) == 2
    assert all(text[s.start:s.end] == s.text for s in saetze)
    karten = Pipeline(lade_katalog()).run(text).karten
    assert {k.eintrag.id for k in karten} == {"ao_buchfuehrungs_kassenmangel"}
    for karte in karten:
        t = karte.treffer
        assert text[t.start:t.end] == t.matched_text


def test_kassenmangel_feuert_trotz_gem_abkuerzung():
    # Reproduktion des Befunds: der Satz zerfiel an "gem." / "Abs." in drei
    # Teile, die Ko-Okkurrenz (Kassenführung + nicht nachvollziehbar) riss ab.
    mm = MusterMatcher(lade_katalog())
    ids = {t.begriff_id for t in mm.match(
        "Die Kassenführung ist gem. § 146 Abs. 1 AO nicht nachvollziehbar.")}
    assert "ao_buchfuehrungs_kassenmangel" in ids


# --- StreitDetektor ----------------------------------------------------------

def test_streit_signale():
    signale = {f.signal for f in StreitDetektor().funde(EDGE)}
    assert "streitig" in signale
    assert "nachlässig" in signale
    assert any("hätte" in s for s in signale)


def test_streit_offsets_treu():
    for f in StreitDetektor().funde(EDGE):
        assert EDGE[f.start:f.end] == f.text


# --- MusterMatcher (gegen den echten Katalog) --------------------------------

def test_kassenmangel_nicht_ohne_mangelwort():
    mm = MusterMatcher(lade_katalog())
    # "Kasse" allein (ohne Mangel-/Abwesenheitswort) darf NICHT feuern
    ids = {t.begriff_id for t in mm.match("Das Geld liegt in der Kasse des Betriebs.")}
    assert "ao_buchfuehrungs_kassenmangel" not in ids
    # mit Mangelwort feuert es
    ids2 = {t.begriff_id for t in
            mm.match("Die Belege fehlen und die Kasse ist nicht nachvollziehbar.")}
    assert "ao_buchfuehrungs_kassenmangel" in ids2


def test_muster_offsets_treu():
    mm = MusterMatcher(lade_katalog())
    for t in mm.match(EDGE):
        assert EDGE[t.start:t.end] == t.matched_text


# --- Pipeline-Integration: der vollständige Edge-Case ------------------------

def test_edge_case_erkennt_alle_kernbegriffe():
    rep = Pipeline(lade_katalog()).run(EDGE)
    muster_ids = {k.eintrag.id for k in rep.karten if k.via == "muster"}
    erwartet = {
        "ao_buchfuehrungs_kassenmangel",      # § 146/146a/158
        "ao_schaetzung",                       # § 162
        "ao_neue_tatsachen",                   # § 173 Abs. 1
        "ao_verschuldensgrad",                 # § 173 Abs. 2 / 153 / 169 / 370 / 378
        "ao_gestaltungsmissbrauch",            # § 42
        "astg_hinzurechnung",                  # §§ 7/8/1 AStG
        "ao_wirtschaftliches_eigentum",        # § 39 Abs. 2
        "ao_vertreterhaftung",                 # § 69 / 34
    }
    fehlend = erwartet - muster_ids
    assert not fehlend, f"nicht erkannt: {fehlend}"


def test_edge_case_streit_verstaerkt_verschulden():
    # Streit-Verstärker ist optional (Default aus, damit der reine MVP-Lauf sauber
    # bleibt); im Web-Pfad ist er an. Hier explizit aktivieren.
    rep = Pipeline(lade_katalog()).run(EDGE, streit=True)
    verschulden = [k for k in rep.karten if k.eintrag.id == "ao_verschuldensgrad"]
    assert verschulden and verschulden[0].ist_streitig
    assert verschulden[0].streit_signal


def test_muster_karte_traegt_muster_herkunft():
    rep = Pipeline(lade_katalog()).run(EDGE)
    muster = [k for k in rep.karten if k.via == "muster"]
    assert muster
    h = muster[0].herkunft()
    assert h["schicht"] == "1b"
    assert h["interpretation"] is False     # deterministisch, kein Modellurteil
