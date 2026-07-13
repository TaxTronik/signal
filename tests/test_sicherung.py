"""Manifestierte, atomare Festwissen-Sicherung und transaktionaler Restore."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest

import risk_layer.sicherung as sicherung_modul
from risk_layer.berater import BeraterStore
from risk_layer.dateisperre import (
    DateisperrenFehler,
    festwissen_sperre,
    festwissen_sperrpfad,
    restore_transaktion_beginnen,
)
from risk_layer.sicherung import (
    ARCHIV_FORMAT,
    ARCHIV_FORMAT_VERSION,
    FESTWISSEN_DATEIEN,
    MANIFEST_DATEI,
    SicherungsFehler,
    bereinige_sicherungen,
    pruefe_sicherung,
    sichere_festwissen,
    stelle_wieder_her,
    stelle_wieder_her_mit_sicherung,
    zeitstempel_dateiname,
)


def _store_mit_inhalt(verzeichnis: Path) -> BeraterStore:
    store = BeraterStore(verzeichnis)
    store.definiere(
        "Betriebsaufspaltung",
        "Besitz- und Betriebsunternehmen.",
        autor="rey",
        datum="2026-06-10",
        scope="personal",
    )
    store.definiere(
        "Organschaft",
        "Eingliederung in den Organträger.",
        autor="rey",
        datum="2026-06-10",
        scope="geteilt",
    )
    store.kuratiere_norm(
        "ust_organschaft",
        "§ 2 UStG",
        "ergaenzen",
        scope="geteilt",
        autor="rey",
        datum="2026-06-10",
    )
    return store


def _mitglieder(archiv: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archiv) as zf:
        return {info.filename: zf.read(info) for info in zf.infolist()}


def _schreibe_mitglieder(archiv: Path, mitglieder: dict[str, bytes]) -> None:
    with zipfile.ZipFile(archiv, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, daten in mitglieder.items():
            zf.writestr(name, daten)


def _aendere_manifest(archiv: Path, aenderung) -> None:
    mitglieder = _mitglieder(archiv)
    manifest = json.loads(mitglieder[MANIFEST_DATEI])
    aenderung(manifest)
    mitglieder[MANIFEST_DATEI] = (
        json.dumps(manifest, ensure_ascii=False) + "\n"
    ).encode()
    _schreibe_mitglieder(archiv, mitglieder)


# -- Zeitstempel und Lock -----------------------------------------------------


def test_zeitstempel_dateiname_iso_wird_windows_sicher():
    assert zeitstempel_dateiname("2026-06-10T12:00:00") == "2026-06-10T12-00-00"


def test_zeitstempel_dateiname_leerer_wert_faellt_laut():
    with pytest.raises(SicherungsFehler):
        zeitstempel_dateiname("::")


def test_lockpfad_liegt_neben_store_und_zweiter_lock_timeout(tmp_path):
    store = tmp_path / "berater"
    assert festwissen_sperrpfad(store) == tmp_path / ".berater.festwissen.lock"
    with festwissen_sperre(store):
        with pytest.raises(DateisperrenFehler):
            with festwissen_sperre(store, timeout=0.01, poll_intervall=0.002):
                pytest.fail("zweite exklusive Sperre darf nicht erworben werden")


def test_lock_blockt_praeparierte_symlink_datei(tmp_path):
    store = tmp_path / "berater"
    fremd = tmp_path / "fremd"
    fremd.write_bytes(b"nicht anfassen")
    lock = festwissen_sperrpfad(store)
    try:
        lock.symlink_to(fremd)
    except OSError:
        pytest.skip("Symlinks sind fuer diesen Windows-Nutzer nicht erlaubt")
    with pytest.raises(DateisperrenFehler, match="symbolischer Link|Reparse"):
        with festwissen_sperre(store):
            pass
    assert fremd.read_bytes() == b"nicht anfassen"


def test_lock_blockt_symlink_in_elternkette(tmp_path):
    echt = tmp_path / "echt"
    echt.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(echt, target_is_directory=True)
    except OSError:
        pytest.skip("Verzeichnis-Symlinks sind fuer diesen Nutzer nicht erlaubt")
    with pytest.raises(DateisperrenFehler, match="Symlink-/Reparse-Komponente"):
        with festwissen_sperre(alias / "berater"):
            pass


def test_lock_prueft_symlink_elternkette_vor_mkdir(tmp_path):
    echt = tmp_path / "echt"
    echt.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(echt, target_is_directory=True)
    except OSError:
        pytest.skip("Verzeichnis-Symlinks sind fuer diesen Nutzer nicht erlaubt")
    mit_luecke = alias / "darf-nicht-entstehen" / "berater"
    with pytest.raises(DateisperrenFehler, match="Symlink-/Reparse-Komponente"):
        with festwissen_sperre(mit_luecke):
            pass
    assert not (echt / "darf-nicht-entstehen").exists()


def test_lock_blockt_nichtregulaeren_lockpfad(tmp_path):
    store = tmp_path / "berater"
    festwissen_sperrpfad(store).mkdir()
    with pytest.raises(DateisperrenFehler, match="keine regulaere Datei"):
        with festwissen_sperre(store):
            pass


def test_lock_blockt_hardlink_ohne_fremde_datei_zu_veraendern(tmp_path):
    store = tmp_path / "berater"
    fremd = tmp_path / "fremd"
    fremd.write_bytes(b"nicht anfassen")
    try:
        os.link(fremd, festwissen_sperrpfad(store))
    except OSError:
        pytest.skip("Hardlinks werden auf diesem Dateisystem nicht unterstuetzt")
    with pytest.raises(DateisperrenFehler, match="Hardlinks"):
        with festwissen_sperre(store):
            pass
    assert fremd.read_bytes() == b"nicht anfassen"


@pytest.mark.skipif(os.name == "nt", reason="POSIX-Modusbits")
def test_lockdatei_hat_restriktive_rechte(tmp_path):
    store = tmp_path / "berater"
    with festwissen_sperre(store):
        pass
    assert festwissen_sperrpfad(store).stat().st_mode & 0o777 == 0o600


# -- Backup ------------------------------------------------------------------


def test_sichern_manifest_hashes_metadaten_und_deterministische_reihenfolge(tmp_path):
    berater = tmp_path / "berater"
    _store_mit_inhalt(berater)
    git_hash = "a" * 40
    release_hash = "b" * 64
    pfad = sichere_festwissen(
        berater,
        tmp_path / "sich",
        "2026-06-10T12:00:00",
        release_id="v2.0.0",
        git_commit=git_hash,
        release_manifest_sha256=release_hash,
    )
    assert pfad.name == "festwissen_2026-06-10T12-00-00.zip"
    with zipfile.ZipFile(pfad) as zf:
        assert zf.namelist() == [MANIFEST_DATEI, *FESTWISSEN_DATEIEN]
        manifest = json.loads(zf.read(MANIFEST_DATEI))
        assert manifest["format"] == ARCHIV_FORMAT
        assert manifest["format_version"] == ARCHIV_FORMAT_VERSION
        assert manifest["created_at"] == "2026-06-10T12:00:00"
        assert manifest["release"] == {
            "id": "v2.0.0",
            "package_version": "0.1.0",
            "git_commit": git_hash,
            "release_manifest_sha256": release_hash,
        }
        assert manifest["counts"] == {
            "definitionen": 2,
            "norm_kuratierungen": 1,
        }
        for eintrag in manifest["files"]:
            daten = zf.read(eintrag["name"])
            assert eintrag["size"] == len(daten)
            assert eintrag["sha256"] == hashlib.sha256(daten).hexdigest()
            assert daten == (berater / eintrag["name"]).read_bytes()

    info = pruefe_sicherung(pfad)
    assert info["legacy"] is False
    assert info["release"]["id"] == "v2.0.0"


def test_sichern_leerer_store_ist_expliziter_manifestierter_snapshot(tmp_path):
    berater = tmp_path / "leer"
    berater.mkdir()
    pfad = sichere_festwissen(berater, tmp_path / "sich", "2026-01-01T00:00:00")
    with zipfile.ZipFile(pfad) as zf:
        assert zf.namelist() == [MANIFEST_DATEI]
        manifest = json.loads(zf.read(MANIFEST_DATEI))
        assert manifest["files"] == []
        assert manifest["counts"] == {
            "definitionen": 0,
            "norm_kuratierungen": 0,
        }
    assert stelle_wieder_her(pfad, tmp_path / "ziel") == []


def test_sichern_fehlendes_oder_verlinktes_beraterverzeichnis_faellt_laut(tmp_path):
    with pytest.raises(SicherungsFehler, match="existiert nicht|unsicher"):
        sichere_festwissen(tmp_path / "fehlt", tmp_path / "sich", "2026")
    if hasattr(os, "symlink"):
        echt = tmp_path / "echt"
        echt.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(echt, target_is_directory=True)
        except OSError:
            return
        with pytest.raises(SicherungsFehler, match="unsicher"):
            sichere_festwissen(link, tmp_path / "sich", "2026")


def test_sichern_nur_teilbestand_manifestiert_genau_diesen(tmp_path):
    berater = tmp_path / "berater"
    BeraterStore(berater).definiere("X", "y", autor="a", datum="d")
    pfad = sichere_festwissen(berater, tmp_path / "sich", "2026-06-10")
    with zipfile.ZipFile(pfad) as zf:
        assert zf.namelist() == [MANIFEST_DATEI, "personal.yaml"]
        manifest = json.loads(zf.read(MANIFEST_DATEI))
        assert [f["name"] for f in manifest["files"]] == ["personal.yaml"]


@pytest.mark.parametrize(
    ("name", "inhalt"),
    [
        ("personal.yaml", "root: mapping"),
        ("geteilt.yaml", "- id: ohne-begriff"),
        ("norm_kuratierung.yaml", "- katalog_id: x"),
    ],
)
def test_sichern_blockt_semantisch_ungueltige_quelle(tmp_path, name, inhalt):
    berater = tmp_path / "berater"
    berater.mkdir()
    (berater / name).write_text(inhalt, encoding="utf-8")
    ziel = tmp_path / "sich"
    with pytest.raises(SicherungsFehler, match="Backup-Quelle"):
        sichere_festwissen(berater, ziel, "2026")
    assert not list(ziel.glob("*.zip"))


def test_sichern_validiert_release_hashes(tmp_path):
    berater = tmp_path / "berater"
    berater.mkdir()
    with pytest.raises(SicherungsFehler, match="git_commit"):
        sichere_festwissen(berater, tmp_path / "sich", "2026", git_commit="kurz")
    with pytest.raises(SicherungsFehler, match="release_manifest_sha256"):
        sichere_festwissen(
            berater,
            tmp_path / "sich",
            "2026",
            release_manifest_sha256="nicht-sha",
        )


def test_sichern_ueberschreibt_keine_zeitstempel_kollision(tmp_path):
    berater = tmp_path / "berater"
    BeraterStore(berater).definiere("Alt", "alt", autor="a", datum="d")
    archiv = sichere_festwissen(berater, tmp_path / "sich", "2026")
    vorher = archiv.read_bytes()
    BeraterStore(berater).definiere("Neu", "neu", autor="a", datum="d")
    with pytest.raises(SicherungsFehler, match="nicht ueberschrieben"):
        sichere_festwissen(berater, tmp_path / "sich", "2026")
    assert archiv.read_bytes() == vorher


def test_semantische_pruefung_hinterlaesst_keine_temp_lockdatei(
    tmp_path, monkeypatch
):
    import risk_layer.sicherung as sicherung

    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("X", "y", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    vorher = {p.name for p in tmp_path.iterdir()}
    monkeypatch.setattr(sicherung.tempfile, "tempdir", str(tmp_path))
    pruefe_sicherung(archiv)
    assert {p.name for p in tmp_path.iterdir()} == vorher


def test_snapshot_blockt_fortlaufende_aenderung(tmp_path, monkeypatch):
    import risk_layer.sicherung as sicherung

    berater = tmp_path / "berater"
    berater.mkdir()
    zaehler = iter(range(20))

    def instabil(_verzeichnis):
        return {"personal.yaml": str(next(zaehler)).encode()}

    monkeypatch.setattr(sicherung, "_lese_quelle", instabil)
    monkeypatch.setattr(sicherung, "_SNAPSHOT_VERSUCHE", 2)
    with pytest.raises(SicherungsFehler, match="konsistenter Snapshot"):
        sicherung.sichere_festwissen(berater, tmp_path / "sich", "2026")


# -- Restore/Verify -----------------------------------------------------------


def test_wiederherstellen_roundtrip_store_bleibt_ladbar(tmp_path):
    berater = tmp_path / "berater"
    vorher = _store_mit_inhalt(berater)
    pfad = sichere_festwissen(berater, tmp_path / "sich", "2026-06-10")
    ziel = tmp_path / "restore"
    assert stelle_wieder_her(pfad, ziel) == list(FESTWISSEN_DATEIEN)
    nachher = BeraterStore(ziel)
    assert {d.id for d in nachher.alle()} == {d.id for d in vorher.alle()}
    assert len(nachher.kuratierungen(nutzer="rey")) == 1


def test_force_stellt_exakten_snapshot_her_und_behaelt_unverwaltetes(tmp_path):
    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("Neu", "neu", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")

    ziel = tmp_path / "ziel"
    _store_mit_inhalt(ziel)
    (ziel / "sicherungen").mkdir()
    (ziel / "sicherungen" / "alt.zip").write_bytes(b"behalten")
    assert stelle_wieder_her(archiv, ziel, force=True) == ["personal.yaml"]
    assert (ziel / "personal.yaml").is_file()
    assert not (ziel / "geteilt.yaml").exists()
    assert not (ziel / "norm_kuratierung.yaml").exists()
    assert (ziel / "sicherungen" / "alt.zip").read_bytes() == b"behalten"
    assert {d.id for d in BeraterStore(ziel).alle()} == {"neu"}


def test_v2_leersnapshot_leert_verwaltete_dateien_exakt(tmp_path):
    leer = tmp_path / "leer"
    leer.mkdir()
    archiv = sichere_festwissen(leer, tmp_path / "sich", "2026")
    ziel = tmp_path / "ziel"
    _store_mit_inhalt(ziel)
    (ziel / "behalten.txt").write_text("ja", encoding="utf-8")
    assert stelle_wieder_her(archiv, ziel, force=True) == []
    assert all(not (ziel / name).exists() for name in FESTWISSEN_DATEIEN)
    assert (ziel / "behalten.txt").read_text(encoding="utf-8") == "ja"


def test_wiederherstellen_nicht_leeres_ziel_braucht_force_und_bleibt_unveraendert(tmp_path):
    quelle = tmp_path / "quelle"
    _store_mit_inhalt(quelle)
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    ziel = tmp_path / "ziel"
    ziel.mkdir()
    alt = b"- bestehend"
    (ziel / "personal.yaml").write_bytes(alt)
    with pytest.raises(SicherungsFehler, match="force"):
        stelle_wieder_her(archiv, ziel)
    assert (ziel / "personal.yaml").read_bytes() == alt


def test_promotion_fehler_rollt_vollstaendig_zurueck(tmp_path, monkeypatch):
    import risk_layer.sicherung as sicherung

    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("Neu", "neu", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    ziel = tmp_path / "ziel"
    _store_mit_inhalt(ziel)
    vorher = {name: (ziel / name).read_bytes() for name in FESTWISSEN_DATEIEN}

    echt = sicherung.os.replace
    einmal = {"ausgeloest": False}

    def fehler_beim_aktivieren(src, dst):
        src_p, dst_p = Path(src), Path(dst)
        if (
            not einmal["ausgeloest"]
            and src_p.name.startswith(f".{ziel.name}.restore-")
            and dst_p == ziel
        ):
            einmal["ausgeloest"] = True
            raise OSError("simulierter Promote-Fehler")
        return echt(src, dst)

    monkeypatch.setattr(sicherung.os, "replace", fehler_beim_aktivieren)
    with pytest.raises(SicherungsFehler, match="alter Stand wurde wiederhergestellt"):
        stelle_wieder_her(archiv, ziel, force=True)
    assert einmal["ausgeloest"]
    assert {name: (ziel / name).read_bytes() for name in FESTWISSEN_DATEIEN} == vorher
    assert {d.id for d in BeraterStore(ziel).alle()} == {
        "betriebsaufspaltung",
        "organschaft",
    }


def test_wiederherstellen_fehlendes_archiv_und_kein_zip_fallen_laut(tmp_path):
    with pytest.raises(SicherungsFehler, match="nicht gefunden"):
        stelle_wieder_her(tmp_path / "fehlt.zip", tmp_path / "ziel")
    kaputt = tmp_path / "kaputt.zip"
    kaputt.write_text("kein zip", encoding="utf-8")
    with pytest.raises(SicherungsFehler, match="Zip"):
        stelle_wieder_her(kaputt, tmp_path / "ziel")


def test_manifestloses_leeres_legacy_archiv_wird_abgelehnt(tmp_path):
    archiv = tmp_path / "leer.zip"
    with zipfile.ZipFile(archiv, "w"):
        pass
    with pytest.raises(SicherungsFehler, match="leeres manifestloses Legacy-Archiv"):
        pruefe_sicherung(archiv)


def test_manifestloses_nichtleeres_legacy_archiv_bleibt_importierbar(tmp_path):
    archiv = tmp_path / "legacy.zip"
    with zipfile.ZipFile(archiv, "w") as zf:
        zf.writestr("personal.yaml", "[]")
    info = pruefe_sicherung(archiv)
    assert info["legacy"] is True
    assert info["files"][0]["name"] == "personal.yaml"
    assert stelle_wieder_her(archiv, tmp_path / "ziel") == ["personal.yaml"]


def test_verify_blockt_hashmanipulation_ohne_ziel_mutation(tmp_path):
    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("X", "y", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    mitglieder = _mitglieder(archiv)
    mitglieder["personal.yaml"] += b"\n# manipuliert"
    _schreibe_mitglieder(archiv, mitglieder)
    ziel = tmp_path / "ziel"
    ziel.mkdir()
    (ziel / "behalten.txt").write_text("ja", encoding="utf-8")
    with pytest.raises(SicherungsFehler, match="Groessenpruefung|SHA-256"):
        stelle_wieder_her(archiv, ziel, force=True)
    assert (ziel / "behalten.txt").read_text(encoding="utf-8") == "ja"


def test_verify_blockt_semantik_auch_mit_passendem_hash(tmp_path):
    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("X", "y", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    mitglieder = _mitglieder(archiv)
    boese = b"- id: ohne-begriff\n"
    mitglieder["personal.yaml"] = boese
    manifest = json.loads(mitglieder[MANIFEST_DATEI])
    eintrag = next(f for f in manifest["files"] if f["name"] == "personal.yaml")
    eintrag["size"] = len(boese)
    eintrag["sha256"] = hashlib.sha256(boese).hexdigest()
    eintrag["records"] = 1
    mitglieder[MANIFEST_DATEI] = json.dumps(manifest).encode()
    _schreibe_mitglieder(archiv, mitglieder)
    with pytest.raises(SicherungsFehler, match="semantisch"):
        pruefe_sicherung(archiv)


@pytest.mark.parametrize(
    "boese",
    ["extra.txt", "../boese.yaml", "..\\boese.yaml", "/etc/personal.yaml", "unter/personal.yaml"],
)
def test_wiederherstellen_whitelist_und_zip_slip_abgewehrt(tmp_path, boese):
    archiv = tmp_path / "fremd.zip"
    with zipfile.ZipFile(archiv, "w") as zf:
        zf.writestr("personal.yaml", "[]")
        zf.writestr(boese, "schadcode?")
    ziel = tmp_path / "ziel"
    with pytest.raises(SicherungsFehler, match="unerwarteter Archiv-Inhalt"):
        stelle_wieder_her(archiv, ziel)
    assert not ziel.exists()
    assert not (tmp_path / "boese.yaml").exists()


def test_wiederherstellen_blockt_doppelte_member(tmp_path):
    archiv = tmp_path / "doppelt.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archiv, "w") as zf:
            zf.writestr("personal.yaml", "[]")
            zf.writestr("personal.yaml", "[]")
    with pytest.raises(SicherungsFehler, match="doppelte Archiv-Eintraege"):
        stelle_wieder_her(archiv, tmp_path / "ziel")


def test_manifest_und_mitglieder_muessen_exakt_uebereinstimmen(tmp_path):
    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("X", "y", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    mitglieder = _mitglieder(archiv)
    del mitglieder["personal.yaml"]
    _schreibe_mitglieder(archiv, mitglieder)
    with pytest.raises(SicherungsFehler, match="stimmen nicht ueberein"):
        pruefe_sicherung(archiv)


@pytest.mark.parametrize(
    ("attribut", "wert", "meldung"),
    [
        ("format_version", 999, "Manifest-Version"),
        ("schema_version", 999, "Festwissen-Schema"),
        ("created_at", "", "created_at"),
    ],
)
def test_manifest_version_schema_und_zeit_werden_validiert(
    tmp_path, attribut, wert, meldung
):
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    _aendere_manifest(archiv, lambda m: m.__setitem__(attribut, wert))
    with pytest.raises(SicherungsFehler, match=meldung):
        pruefe_sicherung(archiv)


@pytest.mark.parametrize("bereich", ["top", "release", "file", "counts"])
def test_manifest_blockt_unbekannte_schluessel(tmp_path, bereich):
    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("X", "y", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")

    def unbekannt(manifest):
        ziel = {
            "top": manifest,
            "release": manifest["release"],
            "file": manifest["files"][0],
            "counts": manifest["counts"],
        }[bereich]
        ziel["unerwartet"] = "nein"

    _aendere_manifest(archiv, unbekannt)
    with pytest.raises(SicherungsFehler, match="Schema"):
        pruefe_sicherung(archiv)


def test_manifest_blockt_nan_und_infinity(tmp_path):
    quelle = tmp_path / "quelle"
    quelle.mkdir()
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    mitglieder = _mitglieder(archiv)
    text = mitglieder[MANIFEST_DATEI].decode("utf-8")
    text = text.replace('"format_version": 2', '"format_version": NaN')
    mitglieder[MANIFEST_DATEI] = text.encode("utf-8")
    _schreibe_mitglieder(archiv, mitglieder)
    with pytest.raises(SicherungsFehler, match="nicht-endliche JSON-Zahl"):
        pruefe_sicherung(archiv)


def test_wiederherstellen_begrenzt_archiv_dateianzahl_vor_lesen(tmp_path, monkeypatch):
    archiv = tmp_path / "zu-viele.zip"
    with zipfile.ZipFile(archiv, "w") as zf:
        zf.writestr("personal.yaml", "[]")
        zf.writestr("geteilt.yaml", "[]")
    monkeypatch.setattr("risk_layer.sicherung._MAX_ARCHIV_DATEIEN", 1)
    monkeypatch.setattr(
        zipfile.ZipFile,
        "read",
        lambda *args, **kwargs: pytest.fail("Lesen vor Limitpruefung"),
    )
    with pytest.raises(SicherungsFehler, match="Dateien/Eintraege"):
        stelle_wieder_her(archiv, tmp_path / "ziel")


def test_wiederherstellen_begrenzt_unkomprimierte_einzelgroesse(tmp_path, monkeypatch):
    archiv = tmp_path / "zu-gross.zip"
    with zipfile.ZipFile(archiv, "w") as zf:
        zf.writestr("personal.yaml", "1234")
    monkeypatch.setattr("risk_layer.sicherung._MAX_DATEI_UNKOMPRIMIERT", 3)
    with pytest.raises(SicherungsFehler, match="personal\\.yaml.*unkomprimiert"):
        stelle_wieder_her(archiv, tmp_path / "ziel")


def test_wiederherstellen_begrenzt_unkomprimierte_gesamtgroesse(tmp_path, monkeypatch):
    archiv = tmp_path / "gesamt-zu-gross.zip"
    with zipfile.ZipFile(archiv, "w") as zf:
        zf.writestr("personal.yaml", "1234")
        zf.writestr("geteilt.yaml", "5678")
    monkeypatch.setattr("risk_layer.sicherung._MAX_ARCHIV_UNKOMPRIMIERT", 7)
    with pytest.raises(SicherungsFehler, match="insgesamt"):
        stelle_wieder_her(archiv, tmp_path / "ziel")


def test_wiederherstellen_blockt_extremes_kompressionsverhaeltnis(tmp_path, monkeypatch):
    archiv = tmp_path / "zip-bomb.zip"
    with zipfile.ZipFile(archiv, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("personal.yaml", "A" * 1000)
    monkeypatch.setattr("risk_layer.sicherung._MAX_KOMPRESSIONSVERHAELTNIS", 2)
    with pytest.raises(SicherungsFehler, match="Kompressionsverhaeltnis"):
        stelle_wieder_her(archiv, tmp_path / "ziel")


@pytest.mark.skipif(os.name == "nt", reason="POSIX-Modusbits")
def test_backup_und_restore_dateien_haben_restriktive_rechte(tmp_path):
    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("X", "y", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "sich", "2026")
    assert archiv.stat().st_mode & 0o777 == 0o600
    ziel = tmp_path / "ziel"
    stelle_wieder_her(archiv, ziel)
    assert ziel.stat().st_mode & 0o777 == 0o700
    assert (ziel / "personal.yaml").stat().st_mode & 0o777 == 0o600


def test_sichern_folgt_keinem_yamllink(tmp_path):
    berater = tmp_path / "berater"
    berater.mkdir()
    geheim = tmp_path / "geheim.txt"
    geheim.write_text("nicht ins Backup", encoding="utf-8")
    try:
        (berater / "personal.yaml").symlink_to(geheim)
    except OSError:
        pytest.skip("Symlinks sind fuer diesen Windows-Nutzer nicht erlaubt")

    with pytest.raises(SicherungsFehler, match="Link|Reparse"):
        sichere_festwissen(berater, tmp_path / "backup", "2026")


def test_sichern_und_restore_blocken_umgelenktes_ziel(tmp_path):
    berater = tmp_path / "berater"
    _store_mit_inhalt(berater)
    echtes_backup = tmp_path / "echtes-backup"
    echtes_backup.mkdir()
    backup_link = tmp_path / "backup-link"
    try:
        backup_link.symlink_to(echtes_backup, target_is_directory=True)
    except OSError:
        pytest.skip("Verzeichnis-Symlinks sind fuer diesen Windows-Nutzer nicht erlaubt")

    with pytest.raises(SicherungsFehler, match="Symlink|Reparse"):
        sichere_festwissen(berater, backup_link, "2026")

    archiv = sichere_festwissen(berater, echtes_backup, "2026")
    echtes_ziel = tmp_path / "echtes-ziel"
    echtes_ziel.mkdir()
    ziel_link = tmp_path / "ziel-link"
    ziel_link.symlink_to(echtes_ziel, target_is_directory=True)
    with pytest.raises(SicherungsFehler, match="Symlink|Reparse"):
        stelle_wieder_her(archiv, ziel_link, force=True)


def test_vorher_sicherung_und_restore_teilen_exakt_einen_live_lock(
    tmp_path, monkeypatch
):
    quelle = tmp_path / "quelle"
    BeraterStore(quelle).definiere("Alt", "Snapshot", autor="a", datum="d")
    archiv = sichere_festwissen(quelle, tmp_path / "quelle-backup", "quelle")

    live = tmp_path / "live"
    store = BeraterStore(live)
    store.definiere("Eins", "live", autor="a", datum="d")
    store.definiere("Zwei", "live", autor="a", datum="d")
    original = sicherung_modul._sicherung_publizieren

    def pruefe_lock(dateien, *args, **kwargs):
        with pytest.raises(DateisperrenFehler, match="nicht innerhalb"):
            with festwissen_sperre(live, timeout=0):
                pass
        return original(dateien, *args, **kwargs)

    monkeypatch.setattr(sicherung_modul, "_sicherung_publizieren", pruefe_lock)
    dateien, vorher = stelle_wieder_her_mit_sicherung(
        archiv,
        live,
        tmp_path / "offhost",
        "2026-pre-restore",
        force=True,
    )

    assert dateien == ["personal.yaml"]
    assert vorher is not None
    assert pruefe_sicherung(vorher)["counts"]["definitionen"] == 2
    assert {eintrag.id for eintrag in BeraterStore(live).alle()} == {"alt"}


def test_backup_retention_ist_dry_run_und_behaelt_mindestanzahl(tmp_path):
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    ziel = tmp_path / "backups"
    now = 2_000_000_000.0
    archive = [
        sichere_festwissen(quelle, ziel, f"2026-0{index}")
        for index in range(1, 5)
    ]
    for index, pfad in enumerate(archive):
        alter = now - (120 - index) * 86400
        os.utime(pfad, (alter, alter))

    plan = bereinige_sicherungen(
        ziel, keep_last=2, keep_days=30, apply=False, now=now
    )
    assert len(plan["candidates"]) == 2
    assert all(pfad.exists() for pfad in archive)

    ergebnis = bereinige_sicherungen(
        ziel, keep_last=2, keep_days=30, apply=True, now=now
    )
    assert len(ergebnis["deleted"]) == 2
    assert [pfad.exists() for pfad in archive] == [False, False, True, True]


def test_backup_retention_bricht_vor_loeschung_bei_korruptem_archiv_ab(tmp_path):
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    ziel = tmp_path / "backups"
    gut = sichere_festwissen(quelle, ziel, "gut")
    korrupt = ziel / "festwissen_korrupt.zip"
    korrupt.write_bytes(b"kein zip")
    with pytest.raises((SicherungsFehler, zipfile.BadZipFile)):
        bereinige_sicherungen(ziel, keep_last=1, keep_days=1, apply=True, now=2e9)
    assert gut.exists()


@pytest.mark.parametrize("neuer_stand_schon_aktiv", [False, True])
def test_restore_journal_repariert_prozessabbruch_beim_naechsten_zugriff(
    tmp_path, neuer_stand_schon_aktiv
):
    ziel = tmp_path / "berater"
    BeraterStore(ziel).definiere("Alt", "sicher", autor="a", datum="d")
    neu = tmp_path / "neu"
    BeraterStore(neu).definiere("Neu", "restore", autor="a", datum="d")
    staging = tmp_path / f".{ziel.name}.restore-crash"
    rollback = tmp_path / f".{ziel.name}.vor-restore-crash"
    shutil.copytree(neu, staging)

    with festwissen_sperre(ziel):
        restore_transaktion_beginnen(ziel, staging, rollback)
        os.replace(ziel, rollback)
        if neuer_stand_schon_aktiv:
            os.replace(staging, ziel)

    # Der erste beliebige Store-Zugriff nach dem Crash erwirbt denselben Lock
    # und rollt auf den nachweislich vollstaendigen alten Stand zurueck.
    assert {eintrag.id for eintrag in BeraterStore(ziel).alle()} == {"alt"}
    assert not rollback.exists()
    assert not staging.exists()
    assert not (tmp_path / f".{ziel.name}.restore-journal.json").exists()
