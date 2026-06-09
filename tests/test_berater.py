"""Berater-Definitionen: Datenmodell, Persistenz, Lebenszyklus (Erweiterung §4)."""

import pytest

from risk_layer.berater import BeraterFehler, BeraterStore
from risk_layer.models import Status


def test_definiere_personal_und_reload(tmp_path):
    store = BeraterStore(tmp_path)
    d = store.definiere(
        "Betriebsaufspaltung",
        "Aufteilung in Besitz- und Betriebsunternehmen mit Verflechtung.",
        autor="rey", datum="2026-05-31", norm_anker=("§ 15 EStG",),
    )
    assert d.scope == "personal"
    # geankert_an wird via Zitat-Parser gesetzt (Schema-Feld)
    assert d.geankert_an == ("norm:EStG:15",)
    # frisch geladen wiederauffindbar
    store2 = BeraterStore(tmp_path)
    gefunden = store2.finde("Betriebsaufspaltung", nutzer="rey")
    assert gefunden is not None and gefunden.autor == "rey"


def test_markierungen_erkennt_begriff_im_text(tmp_path):
    store = BeraterStore(tmp_path)
    store.definiere("Betriebsaufspaltung", "…", autor="rey", datum="2026-05-31")
    marks = store.markierungen("Hier liegt eine Betriebsaufspaltung vor.", nutzer="rey")
    assert len(marks) == 1
    assert marks[0].status is Status.BERATER_DEFINIERT
    assert marks[0].provenienz["autor"] == "rey"


def test_markierungen_ueberspringt_belegte(tmp_path):
    store = BeraterStore(tmp_path)
    store.definiere("Betriebsaufspaltung", "…", autor="rey", datum="2026-05-31")
    text = "Eine Betriebsaufspaltung."
    s = text.index("Betriebsaufspaltung")
    marks = store.markierungen(text, belegte_spans=[(s, s + 18)], nutzer="rey")
    assert marks == []


def test_geteilt_erst_nach_freigabe_kanonisch(tmp_path):
    store = BeraterStore(tmp_path)
    d = store.definiere("Quotennießbrauch", "…", autor="a", datum="d", scope="geteilt")
    # Entwurf ist nicht kanonisch → nicht als geteilte Definition sichtbar
    assert store.finde("Quotennießbrauch") is None
    store.setze_review_status(d.id, "geprüft", reviewer="b")
    store.setze_review_status(d.id, "freigegeben", reviewer="b")
    gefunden = store.finde("Quotennießbrauch")
    assert gefunden is not None and gefunden.review_status == "freigegeben"


def test_lebenszyklus_nur_vorwaerts(tmp_path):
    store = BeraterStore(tmp_path)
    d = store.definiere("Testbegriff", "…", autor="a", datum="d", scope="geteilt")
    store.setze_review_status(d.id, "geprüft", reviewer="b")
    with pytest.raises(BeraterFehler):
        store.setze_review_status(d.id, "entwurf", reviewer="b")  # rückwärts


def test_annotation_an_katalogkarte(tmp_path):
    from risk_layer.catalog import lade_katalog
    from risk_layer.pipeline import Pipeline
    store = BeraterStore(tmp_path)
    store.definiere("organisatorische Eingliederung", "Eigene Notiz: bei GF-Wechsel sofort prüfen.",
                    autor="rey", datum="2026-05-31", katalog_id="ust_organisatorische_eingliederung")
    # Annotation hängt an der Katalogkarte, NICHT als eigenständige Markierung
    rep = Pipeline(lade_katalog()).run(
        "Die organisatorische Eingliederung ist gegeben.", berater=store, nutzer="rey")
    karte = next(k for k in rep.karten if k.eintrag.id == "ust_organisatorische_eingliederung")
    assert karte.berater_notizen
    assert "Eigene Notiz" in karte.berater_notizen[0]["text"]
    # keine eigenständige berater_definiert-Markierung dafür
    assert not any(r.titel == "organisatorische Eingliederung" for r in rep.berater)


def test_personal_nur_fuer_autor_sichtbar(tmp_path):
    store = BeraterStore(tmp_path)
    store.definiere("Sondertatbestand", "…", autor="rey", datum="d", scope="personal")
    assert store.finde("Sondertatbestand", nutzer="rey") is not None
    assert store.finde("Sondertatbestand", nutzer="andere") is None


def test_personal_gleicher_begriff_zweier_autoren_koexistiert(tmp_path):
    # Befund 1: Nutzer B darf die gleichnamige personal-Definition von Nutzer A
    # nicht still löschen – beide leben in EINER Datei.
    store = BeraterStore(tmp_path)
    store.definiere("Sondertatbestand", "Sicht A", autor="alice", datum="d")
    store.definiere("Sondertatbestand", "Sicht B", autor="bob", datum="d")
    store2 = BeraterStore(tmp_path)   # nach Reload überleben beide
    a = store2.finde("Sondertatbestand", nutzer="alice")
    b = store2.finde("Sondertatbestand", nutzer="bob")
    assert a is not None and a.definition_text == "Sicht A"
    assert b is not None and b.definition_text == "Sicht B"
    # eigener Re-Define ersetzt weiterhin NUR den eigenen Eintrag
    store2.definiere("Sondertatbestand", "Sicht A v2", autor="alice", datum="d")
    assert store2.finde("Sondertatbestand", nutzer="alice").definition_text == "Sicht A v2"
    assert store2.finde("Sondertatbestand", nutzer="bob").definition_text == "Sicht B"
    assert sum(1 for d in store2.alle() if d.id == "sondertatbestand") == 2


def test_definiere_geteilt_ueberschreibt_keine_freigabe(tmp_path):
    # Befund 2: erneutes definiere(scope=geteilt) darf einen geprüften/
    # freigegebenen Eintrag nicht durch einen neuen Entwurf ersetzen –
    # das hebelte die Nur-vorwärts-Invariante des Reviews aus.
    store = BeraterStore(tmp_path)
    d = store.definiere("Quotennießbrauch", "v1", autor="a", datum="d", scope="geteilt")
    # entwurf → entwurf bleibt erlaubt
    store.definiere("Quotennießbrauch", "v2", autor="a", datum="d", scope="geteilt")
    store.setze_review_status(d.id, "geprüft", reviewer="r")
    with pytest.raises(BeraterFehler, match="Lebenszyklus"):
        store.definiere("Quotennießbrauch", "v3", autor="a", datum="d", scope="geteilt")
    store.setze_review_status(d.id, "freigegeben", reviewer="r")
    with pytest.raises(BeraterFehler, match="Lebenszyklus"):
        store.definiere("Quotennießbrauch", "v4", autor="b", datum="d", scope="geteilt")
    # der freigegebene Stand bleibt unangetastet
    assert store.finde("Quotennießbrauch").definition_text == "v2"


def test_zwei_store_instanzen_verlieren_keine_updates(tmp_path):
    # Befund 3: zwei Store-Instanzen auf derselben Datei (z. B. zwei Prozesse) –
    # der zweite Save darf den ersten nicht überschreiben (frisch laden + mergen).
    s1 = BeraterStore(tmp_path)
    s2 = BeraterStore(tmp_path)        # beide laden VOR den Writes
    s1.definiere("Begriff Eins", "…", autor="a", datum="d")
    s2.definiere("Begriff Zwei", "…", autor="b", datum="d")
    ids = {d.id for d in BeraterStore(tmp_path).alle()}
    assert {"begriff_eins", "begriff_zwei"} <= ids


def test_laden_root_mapping_statt_liste(tmp_path):
    # Befund 5: Mapping statt Liste als Root → BeraterFehler statt TypeError
    (tmp_path / "personal.yaml").write_text("a: b\n", encoding="utf-8")
    with pytest.raises(BeraterFehler):
        BeraterStore(tmp_path)


def test_laden_fehlende_pflichtschluessel(tmp_path):
    # Befund 5: fehlendes 'id' → BeraterFehler mit Datei-Kontext statt KeyError
    (tmp_path / "personal.yaml").write_text("- begriff: x\n", encoding="utf-8")
    with pytest.raises(BeraterFehler) as exc:
        BeraterStore(tmp_path)
    assert "personal.yaml" in str(exc.value)


def test_laden_definitionsquelle_als_string(tmp_path):
    # Befund 5: String statt dict in definitionsquelle → BeraterFehler statt AttributeError
    (tmp_path / "personal.yaml").write_text(
        "- id: x\n  begriff: X\n  definitionsquelle: ['nur ein string']\n",
        encoding="utf-8")
    with pytest.raises(BeraterFehler):
        BeraterStore(tmp_path)


def test_laden_kaputte_kuratierung(tmp_path):
    # Befund 5: Kuratierungs-Record ohne Pflichtschlüssel → BeraterFehler statt KeyError
    (tmp_path / "norm_kuratierung.yaml").write_text("- norm_id: n\n", encoding="utf-8")
    with pytest.raises(BeraterFehler):
        BeraterStore(tmp_path)


def test_markierungen_dedupliziert_personal_vor_geteilt(tmp_path):
    # Befund 6: zwei sichtbare Definitionen desselben Begriffs ergeben EINE Marke
    # je Stelle; Präzedenz personal vor geteilt.
    store = BeraterStore(tmp_path)
    g = store.definiere("Betriebsaufspaltung", "geteilte Sicht",
                        autor="a", datum="d", scope="geteilt")
    store.setze_review_status(g.id, "geprüft", reviewer="r")
    store.setze_review_status(g.id, "freigegeben", reviewer="r")
    store.definiere("Betriebsaufspaltung", "meine Sicht", autor="rey", datum="d")
    marks = store.markierungen("Hier liegt eine Betriebsaufspaltung vor.", nutzer="rey")
    assert len(marks) == 1
    assert marks[0].provenienz["scope"] == "personal"
    # ohne eigene personal-Definition greift die kanonische geteilte
    marks_b = store.markierungen("Hier liegt eine Betriebsaufspaltung vor.", nutzer="bob")
    assert len(marks_b) == 1 and marks_b[0].provenienz["scope"] == "geteilt"


def test_markierungen_mutiert_belegte_spans_nicht(tmp_path):
    store = BeraterStore(tmp_path)
    store.definiere("Betriebsaufspaltung", "…", autor="rey", datum="d")
    spans: list[tuple[int, int]] = []
    store.markierungen("Eine Betriebsaufspaltung.", belegte_spans=spans, nutzer="rey")
    assert spans == []   # Caller-Liste bleibt unangetastet


def test_nutzer_none_blendet_personal_aus(tmp_path):
    # Befund 7: kein Nutzerkontext (None) = nur kanonisches Wissen –
    # personal-Einträge ALLER Autoren bleiben ausgeblendet.
    store = BeraterStore(tmp_path)
    store.definiere("Sondertatbestand", "…", autor="rey", datum="d")
    assert store.finde("Sondertatbestand") is None
    assert store.markierungen("Der Sondertatbestand greift.") == []
    assert store.finde("Sondertatbestand", nutzer="rey") is not None


def test_multiplikationszeichen_ist_wortgrenze(tmp_path):
    # Befund 14: × (U+00D7) und ÷ (U+00F7) sind Begriffs-GRENZEN, keine Wortzeichen
    store = BeraterStore(tmp_path)
    store.definiere("Betriebsaufspaltung", "…", autor="rey", datum="d")
    marks = store.markierungen("Variante 2×Betriebsaufspaltung÷2 prüfen.", nutzer="rey")
    assert len(marks) == 1 and marks[0].titel == "Betriebsaufspaltung"


def test_markierungen_ignoriert_leeren_begriff(tmp_path):
    # Defensive: ein (manuell eingeschleuster) leerer Begriff darf keine
    # Null-Breite-Spurmarken erzeugen – re.escape('') matcht sonst ueberall.
    from risk_layer.berater import BeraterDefinition
    store = BeraterStore(tmp_path)
    store.definiere("Betriebsaufspaltung", "…", autor="rey", datum="d")
    store._eintraege.append(BeraterDefinition(id="leer", begriff="",
                                              definition_text="x", scope="personal", autor="rey"))
    marks = store.markierungen("Eine Betriebsaufspaltung und mehr Text.", nutzer="rey")
    assert len(marks) == 1 and marks[0].titel == "Betriebsaufspaltung"
