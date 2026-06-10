"""Anonymisierung/Abstraktion (§ 203 StGB)."""

import logging
import sys

import pytest

from risk_layer import anonymize
from risk_layer.anonymize import anonymisiere

# Referenz auf den ECHTEN Loader, bevor die autouse-Fixture ihn wegpatcht –
# der Warning-Test (unten) prüft das einmalige Logging am Original.
_ECHTER_LOADER = anonymize._lade_nlp


@pytest.fixture(autouse=True)
def _kein_spacy(monkeypatch):
    """Kein Test lädt implizit das echte spaCy-Modell.

    Auf Entwickler-Maschinen MIT installiertem spaCy würde sonst jeder
    anonymisiere()-Aufruf das Modell laden (langsam, umgebungsabhängiges
    Testverhalten). NER-Tests injizieren ihren Fake über denselben Seam –
    test-lokale monkeypatches überschreiben diese Fixture.
    """
    monkeypatch.setattr("risk_layer.anonymize._lade_nlp", lambda: None)
    monkeypatch.delenv("RISK_LAYER_NER", raising=False)


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


def test_betrag_ohne_tausendertrenner_wird_ersetzt():
    # § 203-Gate: „50000 EUR" / „1500 Euro" sind genauso identifizierend wie
    # getrennte Schreibweisen – der Währungskontext ist Pflicht (kein Zahlenfressen).
    for text, betrag in [("Es wurden 50000 EUR überwiesen.", "50000 EUR"),
                         ("Vereinbart waren 1500 Euro.", "1500 Euro"),
                         ("Es wurden 1.500 EUR gezahlt.", "1.500 EUR"),
                         ("Kaufpreis: 1 500 000 €.", "1 500 000 €")]:
        r = anonymisiere(text)
        assert betrag not in r.abstrahiert, text
        assert "betrag" in {e.typ for e in r.ersetzungen}, text


def test_betrag_frisst_keine_paragraphen_oder_nummern():
    # Nackte Zahlen OHNE Währungskontext bleiben unangetastet – Norm-Zitate
    # („§ 146 AO") und Ordnungsnummern („Nr. 21") sind kein Geheimnis.
    r = anonymisiere("Nach § 146 AO und § 162 Abs. 2 Nr. 21 AO ist zu schätzen.")
    assert r.abstrahiert == "Nach § 146 AO und § 162 Abs. 2 Nr. 21 AO ist zu schätzen."
    assert r.ersetzungen == []


def test_person_mit_bindestrich_doppelname():
    r = anonymisiere("Herr Dr. Mueller-Lüdenscheid hat unterschrieben.")
    assert "Mueller" not in r.abstrahiert and "Lüdenscheid" not in r.abstrahiert
    assert "[Person A] hat unterschrieben." in r.abstrahiert


def test_firma_mit_mehreren_namenswoertern():
    # Mehrwortige Firmennamen (inkl. „und"/„&") vollständig erfassen – aber den
    # Satzanfang-Artikel („Die") NICHT mitfressen.
    r = anonymisiere("Die Mueller und Schulze GmbH zahlt an die Maier & Huber KG.")
    assert "Mueller" not in r.abstrahiert and "Schulze" not in r.abstrahiert
    assert "Maier" not in r.abstrahiert and "Huber" not in r.abstrahiert
    assert r.abstrahiert.startswith("Die [Gesellschaft A]")
    assert sum(1 for e in r.ersetzungen if e.typ == "firma") == 2


def test_generische_rechtsform_saetze_bleiben():
    # „Eine GmbH ist …" ist Rechtswissen, kein Mandantengeheimnis.
    r = anonymisiere("Eine GmbH ist eine Kapitalgesellschaft.")
    assert r.abstrahiert == "Eine GmbH ist eine Kapitalgesellschaft."


# --- Optionale NER-Schicht (Fake-NLP über den Lade-Seam, KEIN Modell-Download) -

class _FakeSpan:
    """Span-ähnliches Objekt mit der spaCy-Schnittstelle, die _ner_pass nutzt."""

    def __init__(self, text: str, start_char: int, end_char: int, label_: str):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label_


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNlp:
    """Deterministischer NER-Ersatz: findet konfigurierte Namen per str.find."""

    def __init__(self, namen: list[tuple[str, str]]):
        self._namen = namen    # [(Wortlaut, spaCy-Label), …]

    def __call__(self, text: str) -> _FakeDoc:
        ents = []
        for name, label in self._namen:
            pos = text.find(name)
            while pos != -1:
                ents.append(_FakeSpan(text[pos:pos + len(name)], pos, pos + len(name), label))
                pos = text.find(name, pos + len(name))
        ents.sort(key=lambda e: e.start_char)
        return _FakeDoc(ents)


def _mit_fake_ner(monkeypatch, namen: list[tuple[str, str]]) -> None:
    fake = _FakeNlp(namen)
    monkeypatch.setattr("risk_layer.anonymize._lade_nlp", lambda: fake)


def test_ner_person_ohne_anrede_wird_ersetzt(monkeypatch):
    # (a) Die dokumentierte Regex-Grenze: ohne Anrede greift nur die NER-Schicht.
    _mit_fake_ner(monkeypatch, [("Max Mustermann", "PER")])
    r = anonymisiere("Max Mustermann verkaufte das Grundstück.")
    assert r.abstrahiert == "[Person A] verkaufte das Grundstück."
    assert any(e.typ == "person" and e.original == "Max Mustermann" for e in r.ersetzungen)


def test_ner_gleiche_person_gleicher_platzhalter(monkeypatch):
    # (b) Stabile Nummerierung pro Dokument: gleiche Entität → gleicher Platzhalter.
    _mit_fake_ner(monkeypatch, [("Max Mustermann", "PER")])
    r = anonymisiere("Max Mustermann kam. Danach ging Max Mustermann wieder.")
    assert r.abstrahiert.count("[Person A]") == 2
    assert sum(1 for e in r.ersetzungen if e.typ == "person") == 1


def test_ner_fasst_regex_platzhalter_nicht_an(monkeypatch):
    # (c) Der Regex-Pass bleibt führend: ein NER-Treffer AUF einem bereits
    # vergebenen Platzhalter („[Person A]") ist tabu und wird übersprungen.
    _mit_fake_ner(monkeypatch, [("[Person A]", "PER"), ("Max Mustermann", "PER")])
    r = anonymisiere("Herr Mueller traf Max Mustermann.")
    assert r.abstrahiert == "[Person A] traf [Person B]."
    assert {e.original for e in r.ersetzungen} == {"Herr Mueller", "Max Mustermann"}


def test_ner_org_loc_und_fortlaufende_nummerierung(monkeypatch):
    # ORG → [Gesellschaft …], LOC → [Ort …]; der gemeinsame Zähler führt die
    # Regex-Nummerierung fort (Beta GmbH = A, NER-Fund = B).
    _mit_fake_ner(monkeypatch, [("Mustermann-Gruppe", "ORG"), ("Hintertupfingen", "LOC")])
    r = anonymisiere("Die Beta GmbH kooperiert mit der Mustermann-Gruppe in Hintertupfingen.")
    assert "[Gesellschaft A]" in r.abstrahiert     # Regex: Beta GmbH
    assert "[Gesellschaft B]" in r.abstrahiert     # NER:   Mustermann-Gruppe
    assert "[Ort A]" in r.abstrahiert              # NER:   LOC
    assert "Mustermann" not in r.abstrahiert and "Hintertupfingen" not in r.abstrahiert


def test_ner_misc_wird_nicht_ersetzt(monkeypatch):
    # MISC ist zu unscharf (Gesetzesnamen, Produkte) – bewusst kein Ersatz.
    _mit_fake_ner(monkeypatch, [("Umsatzsteuergesetz", "MISC")])
    r = anonymisiere("Das Umsatzsteuergesetz gilt.")
    assert r.abstrahiert == "Das Umsatzsteuergesetz gilt."
    assert r.ersetzungen == []


def test_ohne_spacy_exakt_bisheriges_verhalten():
    # (d) Loader liefert None (autouse-Fixture) → reiner Regex-Pfad, unverändert.
    text = "Max Mustermann verkaufte das Grundstück."
    r = anonymisiere(text)
    assert r.abstrahiert == text
    assert r.ersetzungen == []


def test_env_schalter_deaktiviert_ner(monkeypatch):
    # (e) RISK_LAYER_NER=aus: Schicht aus, Loader darf NICHT einmal aufgerufen
    # werden (Determinismus-kritische Läufe sollen das Modell nie laden).
    monkeypatch.setenv("RISK_LAYER_NER", "aus")

    def _nicht_laden():
        raise AssertionError("RISK_LAYER_NER=aus darf den Loader nicht aufrufen")

    monkeypatch.setattr("risk_layer.anonymize._lade_nlp", _nicht_laden)
    text = "Max Mustermann verkaufte das Grundstück."
    r = anonymisiere(text)
    assert r.abstrahiert == text
    assert r.ersetzungen == []


# --- ner_status(): Engine-Status der NER-Schicht (OHNE Modell-Laden) ---------
# Semantik: "verfuegbar" ist bewusst eine KÖNNTE-Aussage (Env/Cache/find_spec),
# kein Aktiv-Beweis – das Laden beim ersten anonymisiere() kann trotzdem
# scheitern. Alle Zweige werden hier ohne echtes spaCy geprüft.

def test_ner_status_env_aus(monkeypatch):
    # Determinismus-Schalter gewinnt vor allem anderen – und nennt sich selbst.
    monkeypatch.setenv("RISK_LAYER_NER", "aus")
    assert anonymize.ner_status() == {
        "verfuegbar": False, "modell": None, "grund": "deaktiviert (RISK_LAYER_NER)"}


def test_ner_status_laedt_nie_das_modell(monkeypatch):
    # ner_status ist eine reine Status-Abfrage: der Lade-Seam darf NIE
    # angefasst werden (weder Ladezeit noch Seiteneffekte auf den Cache).
    def _nicht_laden():
        raise AssertionError("ner_status darf _lade_nlp nicht aufrufen")
    monkeypatch.setattr("risk_layer.anonymize._lade_nlp", _nicht_laden)
    monkeypatch.setattr(anonymize, "_NLP", anonymize._UNGELADEN)
    anonymize.ner_status()    # darf nicht werfen


def test_ner_status_cache_geladen_nennt_tatsaechliches_modell(monkeypatch):
    # Modul-Cache befüllt → härteste Evidenz: True + TATSÄCHLICHER Modellname
    # (aus nlp.meta, wie ihn spacy.load liefert), kein find_spec-Raten.
    class _Nlp:
        meta = {"lang": "de", "name": "core_news_lg"}
    monkeypatch.setattr(anonymize, "_NLP", _Nlp())
    assert anonymize.ner_status() == {
        "verfuegbar": True, "modell": "de_core_news_lg", "grund": None}


def test_ner_status_cache_fehlgeschlagen(monkeypatch):
    # Laden wurde versucht und scheiterte (Cache = None): ehrlich False mit
    # Install-Hinweis – auch wenn find_spec etwas anderes prognostizieren würde.
    monkeypatch.setattr(anonymize, "_NLP", None)
    st = anonymize.ner_status()
    assert st["verfuegbar"] is False and st["modell"] is None
    assert "pip install" in st["grund"]


def test_ner_status_prognose_ueber_find_spec(monkeypatch):
    # Noch nie geladen: Prognose über die Installations-Metadaten, in der
    # Modellpräferenz lg → sm (hier: nur sm installiert).
    monkeypatch.setattr(anonymize, "_NLP", anonymize._UNGELADEN)
    vorhanden = {"spacy", "de_core_news_sm"}
    monkeypatch.setattr(anonymize, "_find_spec",
                        lambda name: object() if name in vorhanden else None)
    assert anonymize.ner_status() == {
        "verfuegbar": True, "modell": "de_core_news_sm", "grund": None}


def test_ner_status_nichts_installiert(monkeypatch):
    # Weder spaCy noch Modelle: False + Abhilfe (derselbe Hinweis wie im Log).
    monkeypatch.setattr(anonymize, "_NLP", anonymize._UNGELADEN)
    monkeypatch.setattr(anonymize, "_find_spec", lambda name: None)
    st = anonymize.ner_status()
    assert st == {"verfuegbar": False, "modell": None,
                  "grund": anonymize._NER_INSTALL_HINWEIS}


def test_ner_status_spacy_ohne_modelle(monkeypatch):
    # spaCy installiert, aber kein deutsches Modell → ebenfalls False + Abhilfe.
    monkeypatch.setattr(anonymize, "_NLP", anonymize._UNGELADEN)
    monkeypatch.setattr(anonymize, "_find_spec",
                        lambda name: object() if name == "spacy" else None)
    st = anonymize.ner_status()
    assert st["verfuegbar"] is False and "spacy download" in st["grund"]


def test_ner_status_kaputter_import_zaehlt_als_nicht_verfuegbar(monkeypatch):
    # find_spec wirft ValueError, wenn ein Paket in sys.modules auf None steht
    # (kaputter Import) – der Status darf daran nicht crashen.
    monkeypatch.setattr(anonymize, "_NLP", anonymize._UNGELADEN)

    def _kaputt(name):
        raise ValueError(f"{name}.__spec__ is None")
    monkeypatch.setattr(anonymize, "_find_spec", _kaputt)
    st = anonymize.ner_status()
    assert st["verfuegbar"] is False


def test_loader_warnt_einmalig_ohne_spacy(monkeypatch, caplog):
    # Degradations-Philosophie: fehlendes spaCy → sichtbares Warning mit
    # Install-Hinweis, aber dank Modul-Cache nur EINMAL pro Prozess.
    monkeypatch.setitem(sys.modules, "spacy", None)              # Import schlägt fehl
    monkeypatch.setattr(anonymize, "_NLP", anonymize._UNGELADEN)  # Cache zurücksetzen
    with caplog.at_level(logging.WARNING, logger="risk_layer.anonymize"):
        assert _ECHTER_LOADER() is None
        assert _ECHTER_LOADER() is None    # zweiter Aufruf: still, aus dem Cache
    warnungen = [r for r in caplog.records if "NER-Schicht inaktiv" in r.getMessage()]
    assert len(warnungen) == 1
    assert "pip install '.[nlp]'" in warnungen[0].getMessage()
