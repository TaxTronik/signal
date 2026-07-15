"""Manifestierte, atomare Festwissen-Sicherung und transaktionaler Restore."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from types import SimpleNamespace
import zipfile
from pathlib import Path

import pytest

import risk_layer.dateisperre as dateisperre_modul
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
    bereinige_sicherungen as _bereinige_sicherungen,
    pruefe_sicherung,
    sichere_festwissen as _sichere_festwissen,
    stelle_wieder_her,
    stelle_wieder_her_mit_sicherung as _stelle_wieder_her_mit_sicherung,
    zeitstempel_dateiname,
)


BACKUP_TARGET_ID = "test-offhost"


def _provisioniere_backup_ziel(path: Path, target_id: str = BACKUP_TARGET_ID) -> Path:
    if path.is_symlink():
        return path
    path.mkdir(parents=True, exist_ok=True)
    sentinel = path / sicherung_modul.BACKUP_TARGET_FILENAME
    if not sentinel.exists():
        sentinel.write_text(
            json.dumps(
                {
                    "schema": sicherung_modul.BACKUP_TARGET_SCHEMA,
                    "target_id": target_id,
                }
            ),
            encoding="utf-8",
        )
        if os.name != "nt":
            sentinel.chmod(0o600)
    return path


def sichere_festwissen(berater, ziel, zeitstempel, **kwargs):
    ziel_pfad = _provisioniere_backup_ziel(Path(ziel))
    kwargs.setdefault("backup_target_id", BACKUP_TARGET_ID)
    return _sichere_festwissen(berater, ziel_pfad, zeitstempel, **kwargs)


def stelle_wieder_her_mit_sicherung(archiv, ziel, backup_ziel, zeitstempel, **kwargs):
    backup_pfad = _provisioniere_backup_ziel(Path(backup_ziel))
    kwargs.setdefault("backup_target_id", BACKUP_TARGET_ID)
    return _stelle_wieder_her_mit_sicherung(
        archiv,
        ziel,
        backup_pfad,
        zeitstempel,
        **kwargs,
    )


def bereinige_sicherungen(ziel, **kwargs):
    kwargs.setdefault("backup_target_id", BACKUP_TARGET_ID)
    return _bereinige_sicherungen(ziel, **kwargs)


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


@pytest.mark.parametrize(
    "value",
    [
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        3601,
        pytest.param(10**10000, id="huge-int"),
    ],
)
def test_festwissen_lock_timeout_ist_endlich_nichtnegativ_und_begrenzt(
    tmp_path: Path,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="timeout"):
        with festwissen_sperre(tmp_path / "berater", timeout=value):
            pass


def test_festwissen_lock_timeout_null_bleibt_nonblocking(tmp_path: Path) -> None:
    store = tmp_path / "berater"
    with festwissen_sperre(store, timeout=0):
        with pytest.raises(DateisperrenFehler, match="nicht innerhalb"):
            with festwissen_sperre(store, timeout=0):
                pass


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf"), 61])
def test_festwissen_lock_pollintervall_ist_endlich_positiv_und_begrenzt(
    tmp_path: Path,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="poll_intervall"):
        with festwissen_sperre(tmp_path / "berater", poll_intervall=value):
            pass


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
    backup = _provisioniere_backup_ziel(tmp_path / "sich")
    with pytest.raises(SicherungsFehler, match="konsistenter Snapshot"):
        sicherung.sichere_festwissen(
            berater,
            backup,
            "2026",
            backup_target_id=BACKUP_TARGET_ID,
        )


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema":"taxtronik.risk-layer-backup-target/v1","target_id":"falsch"}',
        (
            '{"schema":"taxtronik.risk-layer-backup-target/v1",'
            '"target_id":"test-offhost","target_id":"test-offhost"}'
        ),
        (
            '{"schema":"taxtronik.risk-layer-backup-target/v1",'
            '"target_id":"test-offhost","extra":true}'
        ),
    ],
)
def test_backup_mutation_braucht_strikten_identitaets_sentinel(
    tmp_path: Path,
    raw: str,
) -> None:
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    ziel = tmp_path / "backups"
    ziel.mkdir()
    sentinel = ziel / sicherung_modul.BACKUP_TARGET_FILENAME
    sentinel.write_text(raw, encoding="utf-8")
    if os.name != "nt":
        sentinel.chmod(0o600)

    with pytest.raises(SicherungsFehler, match="target_id|Doppelter|Schluessel"):
        _sichere_festwissen(
            quelle,
            ziel,
            "sentinel",
            backup_target_id=BACKUP_TARGET_ID,
        )
    assert not list(ziel.glob("festwissen_*.zip"))


def test_backup_mutation_erzeugt_fehlenden_sentinel_nie_selbst(tmp_path: Path) -> None:
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    ziel = tmp_path / "backups"
    ziel.mkdir()

    with pytest.raises(SicherungsFehler, match="Sentinel"):
        _sichere_festwissen(
            quelle,
            ziel,
            "missing",
            backup_target_id=BACKUP_TARGET_ID,
        )
    assert not (ziel / sicherung_modul.BACKUP_TARGET_FILENAME).exists()


def test_backup_ziel_sentinel_darf_keinen_hardlink_besitzen(tmp_path: Path) -> None:
    ziel = _provisioniere_backup_ziel(tmp_path / "backups")
    sentinel = ziel / sicherung_modul.BACKUP_TARGET_FILENAME
    try:
        os.link(sentinel, ziel / "sentinel-hardlink.json")
    except OSError:
        pytest.skip("Hardlinks werden auf diesem Test-Dateisystem nicht unterstuetzt")

    with pytest.raises(SicherungsFehler, match="Unsicherer.*Sentinel"):
        sicherung_modul.pruefe_backup_ziel(ziel, BACKUP_TARGET_ID)


def test_backup_ziel_sentinel_prueft_finalen_reparse_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ziel = _provisioniere_backup_ziel(tmp_path / "backups")
    sentinel = ziel / sicherung_modul.BACKUP_TARGET_FILENAME
    real_lstat = os.lstat
    sentinel_calls = 0

    def final_reparse(path):
        nonlocal sentinel_calls
        info = real_lstat(path)
        if Path(path) == sentinel:
            sentinel_calls += 1
            if sentinel_calls == 4:
                return SimpleNamespace(
                    st_dev=info.st_dev,
                    st_ino=info.st_ino,
                    st_size=info.st_size,
                    st_mtime_ns=info.st_mtime_ns,
                    st_mode=info.st_mode,
                    st_nlink=info.st_nlink,
                    st_file_attributes=getattr(info, "st_file_attributes", 0)
                    | getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
                )
        return info

    monkeypatch.setattr(sicherung_modul.os, "lstat", final_reparse)
    with pytest.raises(SicherungsFehler, match="ausgetauscht"):
        sicherung_modul._backup_target_lesen(sentinel)
    assert sentinel_calls == 4


def test_backup_target_wird_unmittelbar_vor_publikation_neu_validiert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    ziel = _provisioniere_backup_ziel(tmp_path / "backups")
    sentinel = ziel / sicherung_modul.BACKUP_TARGET_FILENAME
    original = sicherung_modul.pruefe_backup_ziel
    calls = 0

    def austausch(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            sentinel.write_text(
                json.dumps(
                    {
                        "schema": sicherung_modul.BACKUP_TARGET_SCHEMA,
                        "target_id": "anderes-ziel",
                    }
                ),
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(sicherung_modul, "pruefe_backup_ziel", austausch)
    with pytest.raises(SicherungsFehler, match="andere target_id"):
        _sichere_festwissen(
            quelle,
            ziel,
            "recheck",
            backup_target_id=BACKUP_TARGET_ID,
        )

    assert calls == 1
    assert not list(ziel.glob("festwissen_*.zip"))
    assert not list(ziel.glob(".*.tmp"))


def test_backup_target_wird_nach_temp_write_an_publish_grenze_neu_validiert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    ziel = _provisioniere_backup_ziel(tmp_path / "backups")
    sentinel = ziel / sicherung_modul.BACKUP_TARGET_FILENAME
    original = sicherung_modul._restriktive_rechte

    def tausche_nach_write(path: Path, *, verzeichnis: bool = False) -> None:
        original(path, verzeichnis=verzeichnis)
        if path.suffix == ".tmp":
            sentinel.write_text(
                json.dumps(
                    {
                        "schema": sicherung_modul.BACKUP_TARGET_SCHEMA,
                        "target_id": "anderes-ziel",
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(sicherung_modul, "_restriktive_rechte", tausche_nach_write)
    with pytest.raises(SicherungsFehler, match="andere target_id"):
        _sichere_festwissen(
            quelle,
            ziel,
            "late-target-swap",
            backup_target_id=BACKUP_TARGET_ID,
        )

    assert not list(ziel.glob("festwissen_*.zip"))
    assert not list(ziel.glob(".*.tmp"))


def test_backup_temp_ist_vor_publish_an_erzeugten_inode_gebunden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    ziel = _provisioniere_backup_ziel(tmp_path / "backups")
    original = sicherung_modul._restriktive_rechte

    def tausche_temp(path: Path, *, verzeichnis: bool = False) -> None:
        original(path, verzeichnis=verzeichnis)
        if path.suffix == ".tmp":
            ersatz = path.with_suffix(".ersatz")
            ersatz.write_bytes(b"fremd")
            os.replace(ersatz, path)

    monkeypatch.setattr(sicherung_modul, "_restriktive_rechte", tausche_temp)
    with pytest.raises(SicherungsFehler, match="Temp-Datei.*ausgetauscht"):
        _sichere_festwissen(
            quelle,
            ziel,
            "temp-swap",
            backup_target_id=BACKUP_TARGET_ID,
        )

    assert not list(ziel.glob("festwissen_*.zip"))
    assert not list(ziel.glob(".*.tmp"))


@pytest.mark.parametrize(
    "yaml_text",
    [
        "&zyklus\n- *zyklus\n",
        "- &geteilt\n  id: eins\n- *geteilt\n",
    ],
)
def test_yaml_alias_und_zyklen_werden_vor_semantik_abgelehnt(
    tmp_path: Path,
    yaml_text: str,
) -> None:
    quelle = tmp_path / "berater"
    quelle.mkdir()
    (quelle / "personal.yaml").write_text(yaml_text, encoding="utf-8")

    with pytest.raises(SicherungsFehler, match="Alias|Zyklus"):
        sichere_festwissen(quelle, tmp_path / "backups", "yaml-alias")


def test_yaml_graph_hat_harte_tiefen_und_knotengrenzen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quelle = tmp_path / "berater"
    quelle.mkdir()
    personal = quelle / "personal.yaml"
    personal.write_text("- a:\n    b:\n      c: 1\n", encoding="utf-8")
    monkeypatch.setattr(sicherung_modul, "_MAX_YAML_TIEFE", 2)
    with pytest.raises(SicherungsFehler, match="Tiefe"):
        sichere_festwissen(quelle, tmp_path / "tiefe", "tiefe")

    personal.write_text("- a\n- b\n- c\n- d\n", encoding="utf-8")
    monkeypatch.setattr(sicherung_modul, "_MAX_YAML_TIEFE", 64)
    monkeypatch.setattr(sicherung_modul, "_MAX_YAML_KNOTEN", 3)
    with pytest.raises(SicherungsFehler, match="Knoten"):
        sichere_festwissen(quelle, tmp_path / "knoten", "knoten")


def test_yaml_recursionerror_wird_als_sicherungsfehler_normalisiert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    monkeypatch.setattr(
        sicherung_modul.yaml,
        "safe_load",
        lambda _raw: (_ for _ in ()).throw(RecursionError("zu tief")),
    )

    with pytest.raises(SicherungsFehler, match="UTF-8/YAML"):
        sichere_festwissen(quelle, tmp_path / "backups", "recursion")


def test_manifest_json_recursionerror_wird_normalisiert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sicherung_modul.json,
        "loads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RecursionError("zu tief")),
    )
    with pytest.raises(SicherungsFehler, match="Manifest ist ungueltig"):
        sicherung_modul._json_ohne_doppelte_keys(b"{}", Path("backup.zip"))


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
            with festwissen_sperre(live, timeout=0.001, poll_intervall=0.0005):
                pass
        return original(dateien, *args, **kwargs)

    monkeypatch.setattr(sicherung_modul, "_sicherung_publizieren", pruefe_lock)
    dateien, vorher, info = stelle_wieder_her_mit_sicherung(
        archiv,
        live,
        tmp_path / "offhost",
        "2026-pre-restore",
        force=True,
    )
    assert info["format"] == "risk-layer-festwissen"

    assert dateien == ["personal.yaml"]
    assert vorher is not None
    assert pruefe_sicherung(vorher)["counts"]["definitionen"] == 2
    assert {eintrag.id for eintrag in BeraterStore(live).alle()} == {"alt"}


def test_restore_nutzt_nach_pfadaustausch_nur_bestaetigten_archiv_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bestaetigt = tmp_path / "bestaetigt"
    BeraterStore(bestaetigt).definiere("Alt", "bestaetigt", autor="a", datum="d")
    archiv = sichere_festwissen(bestaetigt, tmp_path / "backups-a", "a")
    expected = hashlib.sha256(archiv.read_bytes()).hexdigest()

    austausch = tmp_path / "austausch"
    BeraterStore(austausch).definiere("Neu", "ausgetauscht", autor="a", datum="d")
    anderes_archiv = sichere_festwissen(austausch, tmp_path / "backups-b", "b")
    original = sicherung_modul._lade_archiv_quelle
    ausgetauscht = False

    def tausche_nach_snapshot(quelle, archiv_name):
        nonlocal ausgetauscht
        if not ausgetauscht and not isinstance(quelle, Path):
            os.replace(anderes_archiv, archiv_name)
            ausgetauscht = True
        return original(quelle, archiv_name)

    monkeypatch.setattr(sicherung_modul, "_lade_archiv_quelle", tausche_nach_snapshot)
    dateien, vorher, _info = stelle_wieder_her_mit_sicherung(
        archiv,
        tmp_path / "live",
        tmp_path / "pre-restore",
        "race",
        force=True,
        expected_sha256=expected,
    )

    assert ausgetauscht is True
    assert dateien == ["personal.yaml"]
    assert vorher is None
    assert {eintrag.id for eintrag in BeraterStore(tmp_path / "live").alle()} == {
        "alt"
    }


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


def test_retention_validiert_target_unmittelbar_vor_jeder_loeschung(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("X", "Y", autor="a", datum="d")
    ziel = tmp_path / "backups"
    archive = [
        sichere_festwissen(quelle, ziel, f"retention-{index}")
        for index in range(2)
    ]
    for index, pfad in enumerate(archive):
        os.utime(pfad, (index + 1, index + 1))
    sentinel = ziel / sicherung_modul.BACKUP_TARGET_FILENAME
    original = sicherung_modul.pruefe_backup_ziel
    calls = 0

    def austausch(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            sentinel.write_text(
                json.dumps(
                    {
                        "schema": sicherung_modul.BACKUP_TARGET_SCHEMA,
                        "target_id": "anderes-ziel",
                    }
                ),
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(sicherung_modul, "pruefe_backup_ziel", austausch)
    with pytest.raises(SicherungsFehler, match="andere target_id"):
        _bereinige_sicherungen(
            ziel,
            backup_target_id=BACKUP_TARGET_ID,
            keep_last=1,
            keep_days=1,
            apply=True,
            now=2_000_000_000.0,
        )

    assert calls == 1
    assert all(path.exists() for path in archive)


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


def test_restore_journal_recursionerror_wird_normalisiert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ziel = tmp_path / "berater"
    ziel.mkdir()
    staging = tmp_path / ".berater.restore-test"
    staging.mkdir()
    rollback = tmp_path / ".berater.vor-restore-test"
    with festwissen_sperre(ziel):
        restore_transaktion_beginnen(ziel, staging, rollback)
    monkeypatch.setattr(
        dateisperre_modul.json,
        "loads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RecursionError("zu tief")),
    )

    with pytest.raises(DateisperrenFehler, match="gueltiges JSON"):
        with festwissen_sperre(ziel):
            pass
