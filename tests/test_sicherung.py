"""Festwissen-Sicherung (sicherung.py, PS-861 L5) – Zip-Roundtrip, Whitelist,
Zip-Slip-Abwehr und Überschreib-Schutz.

Gesichert wird NUR das nicht-regenerierbare Festwissen (Berater-YAMLs aus
berater.py); Korpus/Indizes sind Build-Artefakte und bewusst nicht enthalten.
"""

import zipfile

import pytest

from risk_layer.berater import BeraterStore
from risk_layer.sicherung import (FESTWISSEN_DATEIEN, SicherungsFehler,
                                  sichere_festwissen, stelle_wieder_her,
                                  zeitstempel_dateiname)


def _store_mit_inhalt(verzeichnis) -> BeraterStore:
    """Echter Store mit allen drei Festwissens-Dateien (personal/geteilt/Kuratierung)."""
    store = BeraterStore(verzeichnis)
    store.definiere("Betriebsaufspaltung", "Besitz- und Betriebsunternehmen.",
                    autor="rey", datum="2026-06-10", scope="personal")
    store.definiere("Organschaft", "Eingliederung in den Organträger.",
                    autor="rey", datum="2026-06-10", scope="geteilt")
    store.kuratiere_norm("ust_organschaft", "§ 2 UStG", "ergaenzen",
                         scope="geteilt", autor="rey", datum="2026-06-10")
    return store


# -- zeitstempel_dateiname ------------------------------------------------------

def test_zeitstempel_dateiname_iso_wird_windows_sicher():
    # ':' ist unter Windows verboten – muss ersetzt werden, Rest bleibt lesbar
    s = zeitstempel_dateiname("2026-06-10T12:00:00")
    assert ":" not in s and s == "2026-06-10T12-00-00"


def test_zeitstempel_dateiname_leerer_wert_faellt_laut():
    with pytest.raises(SicherungsFehler):
        zeitstempel_dateiname("::")


# -- sichern --------------------------------------------------------------------

def test_sichern_roundtrip_und_deterministische_reihenfolge(tmp_path):
    berater = tmp_path / "berater"
    _store_mit_inhalt(berater)
    pfad = sichere_festwissen(berater, tmp_path / "sich", "2026-06-10T12:00:00")
    assert pfad.name == "festwissen_2026-06-10T12-00-00.zip"
    with zipfile.ZipFile(pfad) as zf:
        # FESTE Reihenfolge (FESTWISSEN_DATEIEN), nicht Verzeichnis-Zufall
        assert zf.namelist() == list(FESTWISSEN_DATEIEN)
        for name in FESTWISSEN_DATEIEN:
            assert zf.read(name) == (berater / name).read_bytes()


def test_sichern_leerer_store_ist_gueltiges_leeres_archiv(tmp_path):
    berater = tmp_path / "leer"
    berater.mkdir()
    pfad = sichere_festwissen(berater, tmp_path / "sich", "2026-01-01T00:00:00")
    with zipfile.ZipFile(pfad) as zf:
        assert zf.namelist() == []                       # gültig, aber leer
        assert b"leerer Berater-Store" in zf.comment     # selbsterklärender Hinweis
    # Restore aus dem leeren Archiv: kein Fehler, nichts wiederhergestellt
    assert stelle_wieder_her(pfad, tmp_path / "ziel") == []


def test_sichern_fehlendes_berater_verzeichnis_faellt_laut(tmp_path):
    """Ein Tippfehler im Pfad darf KEIN leeres 'Backup' erzeugen."""
    with pytest.raises(SicherungsFehler, match="existiert nicht"):
        sichere_festwissen(tmp_path / "gibtsnicht", tmp_path / "sich", "2026")


def test_sichern_nur_teilbestand(tmp_path):
    """Nur personal.yaml vorhanden → Archiv enthält genau diese (kein Fehler)."""
    berater = tmp_path / "berater"
    BeraterStore(berater).definiere("X", "y", autor="a", datum="d")
    pfad = sichere_festwissen(berater, tmp_path / "sich", "2026-06-10")
    with zipfile.ZipFile(pfad) as zf:
        assert zf.namelist() == ["personal.yaml"]


# -- wiederherstellen ------------------------------------------------------------

def test_wiederherstellen_roundtrip_store_bleibt_ladbar(tmp_path):
    berater = tmp_path / "berater"
    vorher = _store_mit_inhalt(berater)
    pfad = sichere_festwissen(berater, tmp_path / "sich", "2026-06-10")

    ziel = tmp_path / "restore"
    dateien = stelle_wieder_her(pfad, ziel)
    assert dateien == list(FESTWISSEN_DATEIEN)
    # semantischer Roundtrip: der wiederhergestellte Store lädt und trägt
    # denselben Bestand (das ist der Restore-Beleg, den L5 verlangt)
    nachher = BeraterStore(ziel)
    assert {d.id for d in nachher.alle()} == {d.id for d in vorher.alle()}
    assert len(nachher.kuratierungen(nutzer="rey")) == 1


def test_wiederherstellen_nicht_leeres_ziel_braucht_force(tmp_path):
    berater = tmp_path / "berater"
    _store_mit_inhalt(berater)
    pfad = sichere_festwissen(berater, tmp_path / "sich", "2026-06-10")
    ziel = tmp_path / "ziel"
    ziel.mkdir()
    (ziel / "personal.yaml").write_text("- bestehend", encoding="utf-8")
    with pytest.raises(SicherungsFehler, match="force"):
        stelle_wieder_her(pfad, ziel)
    # mit force: ausdrückliche Entscheidung → überschreibt
    assert "personal.yaml" in stelle_wieder_her(pfad, ziel, force=True)
    assert "Betriebsaufspaltung" in (ziel / "personal.yaml").read_text(encoding="utf-8")


def test_wiederherstellen_fehlendes_archiv_faellt_laut(tmp_path):
    with pytest.raises(SicherungsFehler, match="nicht gefunden"):
        stelle_wieder_her(tmp_path / "fehlt.zip", tmp_path / "ziel")


def test_wiederherstellen_kein_zip_faellt_laut(tmp_path):
    kaputt = tmp_path / "kaputt.zip"
    kaputt.write_text("kein zip", encoding="utf-8")
    with pytest.raises(SicherungsFehler, match="Zip"):
        stelle_wieder_her(kaputt, tmp_path / "ziel")


def test_wiederherstellen_whitelist_blockt_fremden_inhalt(tmp_path):
    """NUR die erwarteten YAML-Namen werden entpackt – alles andere bricht ab
    (ein Restore darf kein Einfallstor für beliebige Dateien sein)."""
    arch = tmp_path / "fremd.zip"
    with zipfile.ZipFile(arch, "w") as zf:
        zf.writestr("personal.yaml", "[]")
        zf.writestr("extra.txt", "schadcode?")
    ziel = tmp_path / "ziel"
    with pytest.raises(SicherungsFehler, match="unerwarteter Archiv-Inhalt"):
        stelle_wieder_her(arch, ziel)
    # NICHTS wurde teilweise entpackt (Validierung VOR der Extraktion)
    assert not ziel.exists() or list(ziel.iterdir()) == []


@pytest.mark.parametrize("boese", [
    "../boese.yaml",                 # klassischer Zip-Slip
    "..\\boese.yaml",                # Windows-Variante
    "/etc/personal.yaml",            # absoluter Pfad
    "unter/personal.yaml",           # Unterverzeichnis (nicht whitelisted)
])
def test_wiederherstellen_zip_slip_abgewehrt(tmp_path, boese):
    arch = tmp_path / "slip.zip"
    with zipfile.ZipFile(arch, "w") as zf:
        zf.writestr(boese, "raus aus dem Ziel")
    with pytest.raises(SicherungsFehler):
        stelle_wieder_her(arch, tmp_path / "ziel")
    assert not (tmp_path / "boese.yaml").exists()
