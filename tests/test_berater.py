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
