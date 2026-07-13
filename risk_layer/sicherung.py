"""Transaktionale Sicherung des nicht-regenerierbaren Festwissens.

Neue Archive tragen ein versioniertes Manifest mit Release-/Schema-Metadaten,
Groessen, Record-Zahlen und SHA-256 je YAML. Backup und Restore verwenden die
gemeinsame Interprozess-Sperre. Vor einer Promotion wird das Archiv komplett in
einem Staging-Verzeichnis materialisiert und als echter ``BeraterStore``
validiert; das Live-Verzeichnis wird erst danach praxisnah atomar umgeschaltet.

Manifestlose Altarchive bleiben als bewusster Legacy-Import lesbar, sofern sie
nicht leer sind und ihre YAMLs semantisch valide sind. Neue Archive sind immer
v2; ein v2-Manifest darf eine explizit leere Dateiliste enthalten und damit bei
``force=True`` einen Store bewusst leeren.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .berater import BeraterFehler, BeraterStore
from .dateisperre import (
    DateisperrenFehler,
    festwissen_sperre,
    restore_transaktion_beenden,
    restore_transaktion_beginnen,
)

FESTWISSEN_DATEIEN: tuple[str, ...] = (
    "personal.yaml",
    "geteilt.yaml",
    "norm_kuratierung.yaml",
)
ARCHIV_PREFIX = "festwissen_"
MANIFEST_DATEI = "manifest.json"
ARCHIV_FORMAT = "risk-layer-festwissen"
ARCHIV_FORMAT_VERSION = 2
FESTWISSEN_SCHEMA_VERSION = 1

_MAX_ARCHIV_DATEIEN = len(FESTWISSEN_DATEIEN) + 1
_MAX_DATEI_UNKOMPRIMIERT = 256 * 1024 * 1024
_MAX_ARCHIV_UNKOMPRIMIERT = 512 * 1024 * 1024
_MAX_KOMPRESSIONSVERHAELTNIS = 500
_SNAPSHOT_VERSUCHE = 5
_LOCK_TIMEOUT = 10.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class SicherungsFehler(ValueError):
    """Bedienungs-, Integritaets- oder Strukturfehler bei Backup/Restore."""


def zeitstempel_dateiname(zeitstempel: str) -> str:
    """Macht einen (ISO-)Zeitstempel unter Windows und Unix dateisicher."""
    s = "".join(
        c if (c.isalnum() or c in "._-") else "-"
        for c in str(zeitstempel).strip()
    ).strip("-.")
    if not s:
        raise SicherungsFehler(
            f"Zeitstempel ergibt keinen Dateinamen: {zeitstempel!r}"
        )
    return s


def _restriktive_rechte(pfad: Path, *, verzeichnis: bool = False) -> None:
    """Setzt POSIX-Rechte restriktiv; Windows uebernimmt die Verzeichnis-ACL."""
    if os.name != "nt":
        os.chmod(pfad, 0o700 if verzeichnis else 0o600)


def _fsync_verzeichnis(pfad: Path) -> None:
    """Persistiert Verzeichnis-Metadaten soweit die Plattform das unterstuetzt."""
    if os.name == "nt":
        return
    fd = os.open(pfad, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256(daten: bytes) -> str:
    return hashlib.sha256(daten).hexdigest()


def _ist_link_oder_reparse(pfad: Path) -> bool:
    try:
        info = os.lstat(pfad)
    except FileNotFoundError:
        return False
    attribut = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(info.st_mode) or bool(
        attribut and getattr(info, "st_file_attributes", 0) & attribut
    )


def _pruefe_verzeichniskette(pfad: Path) -> None:
    """Verweigert existierende Symlink-/Reparse-Komponenten eines Pfads."""
    absolut = Path(os.path.abspath(pfad))
    for teil in reversed((absolut, *absolut.parents)):
        try:
            info = os.lstat(teil)
        except FileNotFoundError:
            continue
        attribut = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(info.st_mode) or bool(
            attribut and getattr(info, "st_file_attributes", 0) & attribut
        ):
            raise SicherungsFehler(
                f"unsichere Symlink-/Reparse-Komponente im Pfad: {teil}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise SicherungsFehler(f"Pfadkomponente ist kein Verzeichnis: {teil}")


def _lese_regulaere_datei(pfad: Path) -> bytes:
    """Liest eine YAML begrenzt, ohne Link-Folgen und mit TOCTOU-Pruefung."""
    if _ist_link_oder_reparse(pfad):
        raise SicherungsFehler(
            f"Unsichere Backup-Quelle: {pfad} ist ein Link/Reparse-Point"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(pfad, flags)
    except OSError as exc:
        raise SicherungsFehler(f"Backup-Quelle nicht sicher lesbar: {pfad}: {exc}") from exc
    try:
        vorher = os.fstat(descriptor)
        nachher_pfad = os.lstat(pfad)
        if (
            not stat.S_ISREG(vorher.st_mode)
            or stat.S_ISLNK(nachher_pfad.st_mode)
            or _ist_link_oder_reparse(pfad)
            or (vorher.st_dev, vorher.st_ino) != (nachher_pfad.st_dev, nachher_pfad.st_ino)
        ):
            raise SicherungsFehler(
                f"Backup-Quelle wurde ausgetauscht oder ist nicht regulaer: {pfad}"
            )
        if vorher.st_size > _MAX_DATEI_UNKOMPRIMIERT:
            raise SicherungsFehler(
                f"{pfad}: {vorher.st_size} Bytes; erlaubt sind hoechstens "
                f"{_MAX_DATEI_UNKOMPRIMIERT}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            daten = handle.read(_MAX_DATEI_UNKOMPRIMIERT + 1)
        nachher = os.fstat(descriptor)
        if (
            len(daten) > _MAX_DATEI_UNKOMPRIMIERT
            or (vorher.st_dev, vorher.st_ino, vorher.st_size, vorher.st_mtime_ns)
            != (nachher.st_dev, nachher.st_ino, nachher.st_size, nachher.st_mtime_ns)
        ):
            raise SicherungsFehler(f"Backup-Quelle wurde beim Lesen veraendert: {pfad}")
        return daten
    finally:
        os.close(descriptor)


def _lese_quelle(verzeichnis: Path) -> dict[str, bytes]:
    dateien: dict[str, bytes] = {}
    gesamt = 0
    for name in FESTWISSEN_DATEIEN:
        pfad = verzeichnis / name
        if _ist_link_oder_reparse(pfad):
            raise SicherungsFehler(
                f"Unsichere Backup-Quelle: {pfad} ist ein Link/Reparse-Point"
            )
        if not pfad.exists():
            continue
        daten = _lese_regulaere_datei(pfad)
        gesamt += len(daten)
        if gesamt > _MAX_ARCHIV_UNKOMPRIMIERT:
            raise SicherungsFehler(
                "Festwissen ist fuer ein Sicherungsarchiv insgesamt zu gross"
            )
        dateien[name] = daten
    return dateien


def _stabiler_snapshot(verzeichnis: Path) -> dict[str, bytes]:
    """Liest bis zu einem ueber zwei Vollscans stabilen Store-Stand.

    Die gemeinsame Sperre verhindert Aenderungen kooperierender Writer. Der
    Doppel-Scan erkennt zusaetzlich alte, noch nicht lock-faehige Prozesse.
    """
    for _ in range(_SNAPSHOT_VERSUCHE):
        erster = _lese_quelle(verzeichnis)
        zweiter = _lese_quelle(verzeichnis)
        if erster == zweiter:
            return zweiter
    raise SicherungsFehler(
        f"Kein konsistenter Snapshot von {verzeichnis}: Festwissen wurde "
        "waehrend der Sicherung fortlaufend geaendert"
    )


def _schreibe_datei_exklusiv(pfad: Path, daten: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(pfad, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(daten)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            os.unlink(pfad)
        except OSError:
            pass
        raise
    _restriktive_rechte(pfad)


def _yaml_recordzahlen(dateien: dict[str, bytes], *, kontext: str) -> dict[str, int]:
    zahlen: dict[str, int] = {}
    for name, daten in dateien.items():
        try:
            text = daten.decode("utf-8")
            wurzel = yaml.safe_load(text)
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise SicherungsFehler(
                f"{kontext}: {name} ist kein gueltiges UTF-8/YAML: {exc}"
            ) from exc
        if wurzel is None:
            wurzel = []
        if not isinstance(wurzel, list):
            raise SicherungsFehler(
                f"{kontext}: {name}: YAML-Root muss eine Liste sein, nicht "
                f"{type(wurzel).__name__}"
            )
        zahlen[name] = len(wurzel)
    return zahlen


def _validiere_semantik(
    dateien: dict[str, bytes], *, kontext: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Validiert rohe YAML-Struktur und den vollstaendigen BeraterStore isoliert."""
    recordzahlen = _yaml_recordzahlen(dateien, kontext=kontext)
    with tempfile.TemporaryDirectory(prefix="risk-layer-festwissen-pruefung-") as tmp:
        temp_root = Path(tmp)
        pruefdir = temp_root / "store"
        pruefdir.mkdir()
        _restriktive_rechte(temp_root, verzeichnis=True)
        _restriktive_rechte(pruefdir, verzeichnis=True)
        for name in FESTWISSEN_DATEIEN:
            if name in dateien:
                _schreibe_datei_exklusiv(pruefdir / name, dateien[name])
        try:
            store = BeraterStore(pruefdir)
        except (BeraterFehler, OSError, UnicodeError, yaml.YAMLError) as exc:
            raise SicherungsFehler(
                f"{kontext}: Festwissen ist als BeraterStore semantisch "
                f"ungueltig: {exc}"
            ) from exc

        summen = {
            "definitionen": len(store.alle()),
            "norm_kuratierungen": recordzahlen.get("norm_kuratierung.yaml", 0),
        }
    return recordzahlen, summen


def _release_metadaten(
    release_id: str | None,
    git_commit: str | None,
    release_manifest_sha256: str | None,
) -> dict[str, str]:
    roh_id = (
        release_id
        if release_id is not None
        else os.environ.get("RISK_LAYER_RELEASE_ID")
        or f"risk-layer-{__version__}"
    )
    if not isinstance(roh_id, str):
        raise SicherungsFehler("release_id muss ein String sein")
    identitaet = roh_id.strip()
    if not identitaet:
        raise SicherungsFehler("release_id darf nicht leer sein")
    metadaten = {"id": identitaet, "package_version": __version__}
    if git_commit is not None:
        if not isinstance(git_commit, str):
            raise SicherungsFehler("git_commit muss ein String sein")
        git_commit = git_commit.strip().lower()
        if not _GIT_COMMIT_RE.fullmatch(git_commit):
            raise SicherungsFehler(
                "git_commit muss ein vollstaendiger 40- oder 64-stelliger "
                "hexadezimaler Git-Hash sein"
            )
        metadaten["git_commit"] = git_commit
    if release_manifest_sha256 is not None:
        if not isinstance(release_manifest_sha256, str):
            raise SicherungsFehler("release_manifest_sha256 muss ein String sein")
        release_manifest_sha256 = release_manifest_sha256.strip().lower()
        if not _SHA256_RE.fullmatch(release_manifest_sha256):
            raise SicherungsFehler(
                "release_manifest_sha256 muss ein 64-stelliger SHA-256 sein"
            )
        metadaten["release_manifest_sha256"] = release_manifest_sha256
    return metadaten


def _erstelle_manifest(
    dateien: dict[str, bytes],
    *,
    zeitstempel: str,
    release_id: str | None,
    git_commit: str | None,
    release_manifest_sha256: str | None,
) -> dict[str, Any]:
    recordzahlen, summen = _validiere_semantik(
        dateien, kontext="Backup-Quelle"
    )
    eintraege = []
    for name in FESTWISSEN_DATEIEN:
        if name in dateien:
            daten = dateien[name]
            eintraege.append(
                {
                    "name": name,
                    "size": len(daten),
                    "sha256": _sha256(daten),
                    "records": recordzahlen[name],
                }
            )
    return {
        "format": ARCHIV_FORMAT,
        "format_version": ARCHIV_FORMAT_VERSION,
        "schema_version": FESTWISSEN_SCHEMA_VERSION,
        "created_at": str(zeitstempel),
        "release": _release_metadaten(
            release_id, git_commit, release_manifest_sha256
        ),
        "files": eintraege,
        "counts": summen,
    }


def _zip_schreibe(zf: zipfile.ZipFile, name: str, daten: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    zf.writestr(info, daten)


def _sicherung_publizieren(
    dateien: dict[str, bytes],
    ziel_dir: str | Path,
    zeitstempel: str,
    *,
    release_id: str | None = None,
    git_commit: str | None = None,
    release_manifest_sha256: str | None = None,
) -> Path:
    """Publiziert einen bereits unter Festwissen-Lock gelesenen Snapshot."""
    ziel = Path(ziel_dir)
    _pruefe_verzeichniskette(ziel)
    ziel.mkdir(parents=True, exist_ok=True)
    _pruefe_verzeichniskette(ziel)
    pfad = ziel / f"{ARCHIV_PREFIX}{zeitstempel_dateiname(zeitstempel)}.zip"
    if pfad.exists() or pfad.is_symlink():
        raise SicherungsFehler(
            f"Sicherungsarchiv existiert bereits und wird nicht ueberschrieben: {pfad}"
        )

    manifest = _erstelle_manifest(
        dateien,
        zeitstempel=zeitstempel,
        release_id=release_id,
        git_commit=git_commit,
        release_manifest_sha256=release_manifest_sha256,
    )
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        dir=str(ziel), prefix=f".{pfad.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    per_rename_publiziert = False
    try:
        with os.fdopen(fd, "wb") as fh:
            with zipfile.ZipFile(fh, "w") as zf:
                _zip_schreibe(zf, MANIFEST_DATEI, manifest_bytes)
                for name in FESTWISSEN_DATEIEN:
                    if name in dateien:
                        _zip_schreibe(zf, name, dateien[name])
                zf.comment = (
                    f"Risk-Layer Festwissen v{ARCHIV_FORMAT_VERSION}; "
                    f"{len(dateien)} Datendatei(en)"
                ).encode("utf-8")
            fh.flush()
            os.fsync(fh.fileno())
        _restriktive_rechte(tmp)
        try:
            # Temp und Ziel liegen im selben Verzeichnis/Filesystem. Der
            # Hardlink publiziert nur die bereits vollstaendig geschriebene und
            # fsync-te Datei und scheitert atomar bei einer Namenskollision.
            os.link(tmp, pfad, follow_symlinks=False)
        except FileExistsError as exc:
            raise SicherungsFehler(
                f"Sicherungsarchiv existiert bereits und wird nicht "
                f"ueberschrieben: {pfad}"
            ) from exc
        except OSError as exc:
            if os.name != "nt":
                raise SicherungsFehler(
                    f"Backup-Ziel unterstuetzt die benoetigte atomare "
                    f"Hardlink-Publikation nicht: {ziel} ({exc})"
                ) from exc
            try:
                # Windows MoveFile/rename ersetzt ein bestehendes Ziel nicht.
                # Das bietet insbesondere auf SMB-Shares eine atomare
                # No-clobber-Alternative, wenn Hardlinks nicht verfuegbar sind.
                os.rename(tmp, pfad)
                per_rename_publiziert = True
            except FileExistsError as rename_exc:
                raise SicherungsFehler(
                    f"Sicherungsarchiv existiert bereits und wird nicht "
                    f"ueberschrieben: {pfad}"
                ) from rename_exc
            except OSError as rename_exc:
                raise SicherungsFehler(
                    f"Backup-Ziel unterstuetzt weder Hardlink noch atomare "
                    f"No-clobber-Publikation: {ziel} ({rename_exc})"
                ) from rename_exc
        if not per_rename_publiziert:
            tmp.unlink()
        _fsync_verzeichnis(ziel)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return pfad


def sichere_festwissen(
    berater_dir: str | Path,
    ziel_dir: str | Path,
    zeitstempel: str,
    *,
    release_id: str | None = None,
    git_commit: str | None = None,
    release_manifest_sha256: str | None = None,
) -> Path:
    """Erzeugt atomar ein manifestiertes, semantisch validiertes v2-Backup."""
    quelle = Path(berater_dir)
    _pruefe_verzeichniskette(quelle)
    if _ist_link_oder_reparse(quelle) or not quelle.is_dir():
        raise SicherungsFehler(
            f"Berater-Verzeichnis existiert nicht oder ist unsicher: {quelle}"
        )
    try:
        with festwissen_sperre(quelle, timeout=_LOCK_TIMEOUT):
            dateien = _stabiler_snapshot(quelle)
            return _sicherung_publizieren(
                dateien,
                ziel_dir,
                zeitstempel=zeitstempel,
                release_id=release_id,
                git_commit=git_commit,
                release_manifest_sha256=release_manifest_sha256,
            )
    except DateisperrenFehler as exc:
        raise SicherungsFehler(str(exc)) from exc


def _pruefe_archiv_limits(
    zf: zipfile.ZipFile, archiv: Path
) -> list[zipfile.ZipInfo]:
    infos = zf.infolist()
    if len(infos) > _MAX_ARCHIV_DATEIEN:
        raise SicherungsFehler(
            f"{archiv}: Archiv enthaelt {len(infos)} Dateien/Eintraege; "
            f"erlaubt sind hoechstens {_MAX_ARCHIV_DATEIEN}"
        )
    if len({i.filename for i in infos}) != len(infos):
        raise SicherungsFehler(f"{archiv}: doppelte Archiv-Eintraege sind unzulaessig")

    gesamt = 0
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x1:
            raise SicherungsFehler(
                f"{archiv}: Verzeichnis oder verschluesselter ZIP-Eintrag "
                f"nicht erlaubt: {info.filename!r}"
            )
        if info.file_size < 0 or info.compress_size < 0:
            raise SicherungsFehler(
                f"{archiv}: ungueltige Groessenangabe fuer {info.filename!r}"
            )
        if info.file_size > _MAX_DATEI_UNKOMPRIMIERT:
            raise SicherungsFehler(
                f"{archiv}: {info.filename!r} ist unkomprimiert "
                f"{info.file_size} Bytes gross; erlaubt sind hoechstens "
                f"{_MAX_DATEI_UNKOMPRIMIERT} Bytes"
            )
        if info.file_size and (
            info.compress_size == 0
            or info.file_size
            > info.compress_size * _MAX_KOMPRESSIONSVERHAELTNIS
        ):
            verhaeltnis = info.file_size / max(info.compress_size, 1)
            raise SicherungsFehler(
                f"{archiv}: extremes Kompressionsverhaeltnis bei "
                f"{info.filename!r} ({verhaeltnis:.1f}:1; erlaubt sind "
                f"hoechstens {_MAX_KOMPRESSIONSVERHAELTNIS}:1)"
            )
        gesamt += info.file_size
    if gesamt > _MAX_ARCHIV_UNKOMPRIMIERT:
        raise SicherungsFehler(
            f"{archiv}: Archiv ist unkomprimiert insgesamt {gesamt} Bytes "
            f"gross; erlaubt sind hoechstens {_MAX_ARCHIV_UNKOMPRIMIERT} Bytes"
        )
    return infos


def _json_ohne_doppelte_keys(daten: bytes, archiv: Path) -> dict[str, Any]:
    def hook(paare: list[tuple[str, Any]]) -> dict[str, Any]:
        ergebnis: dict[str, Any] = {}
        for key, wert in paare:
            if key in ergebnis:
                raise SicherungsFehler(
                    f"{archiv}: doppelter Manifest-Schluessel {key!r}"
                )
            ergebnis[key] = wert
        return ergebnis

    def unendliche_konstante(wert: str) -> Any:
        raise SicherungsFehler(
            f"{archiv}: nicht-endliche JSON-Zahl im Manifest: {wert}"
        )

    try:
        wert = json.loads(
            daten.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=unendliche_konstante,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SicherungsFehler(f"{archiv}: Manifest ist ungueltig: {exc}") from exc
    if not isinstance(wert, dict):
        raise SicherungsFehler(f"{archiv}: Manifest-Root muss ein Objekt sein")
    return wert


def _validiere_manifest(
    manifest: dict[str, Any], archiv: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    top_keys = {
        "format",
        "format_version",
        "schema_version",
        "created_at",
        "release",
        "files",
        "counts",
    }
    if set(manifest) != top_keys:
        raise SicherungsFehler(
            f"{archiv}: Manifest-Schluessel weichen vom v2-Schema ab: "
            f"{sorted(set(manifest) ^ top_keys)!r}"
        )
    if manifest.get("format") != ARCHIV_FORMAT:
        raise SicherungsFehler(f"{archiv}: unbekanntes Sicherungsformat")
    if manifest.get("format_version") != ARCHIV_FORMAT_VERSION:
        raise SicherungsFehler(
            f"{archiv}: Manifest-Version {manifest.get('format_version')!r} "
            f"wird nicht unterstuetzt (erwartet {ARCHIV_FORMAT_VERSION})"
        )
    if manifest.get("schema_version") != FESTWISSEN_SCHEMA_VERSION:
        raise SicherungsFehler(
            f"{archiv}: Festwissen-Schema {manifest.get('schema_version')!r} "
            f"wird nicht unterstuetzt"
        )
    if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"].strip():
        raise SicherungsFehler(f"{archiv}: created_at fehlt im Manifest")
    release = manifest.get("release")
    if (
        not isinstance(release, dict)
        or not isinstance(release.get("id"), str)
        or not release["id"].strip()
        or not isinstance(release.get("package_version"), str)
        or not release["package_version"].strip()
    ):
        raise SicherungsFehler(f"{archiv}: Release-Metadaten fehlen oder sind ungueltig")
    release_pflicht = {"id", "package_version"}
    release_erlaubt = release_pflicht | {"git_commit", "release_manifest_sha256"}
    if not release_pflicht <= set(release) or not set(release) <= release_erlaubt:
        raise SicherungsFehler(
            f"{archiv}: Release-Schluessel weichen vom v2-Schema ab"
        )
    git_commit = release.get("git_commit")
    if git_commit is not None and (
        not isinstance(git_commit, str) or not _GIT_COMMIT_RE.fullmatch(git_commit)
    ):
        raise SicherungsFehler(f"{archiv}: git_commit ist ungueltig")
    release_hash = release.get("release_manifest_sha256")
    if release_hash is not None and (
        not isinstance(release_hash, str) or not _SHA256_RE.fullmatch(release_hash)
    ):
        raise SicherungsFehler(
            f"{archiv}: release_manifest_sha256 ist ungueltig"
        )

    eintraege = manifest.get("files")
    if not isinstance(eintraege, list):
        raise SicherungsFehler(f"{archiv}: files muss eine Liste sein")
    gesehen: set[str] = set()
    for eintrag in eintraege:
        if not isinstance(eintrag, dict):
            raise SicherungsFehler(f"{archiv}: ungueltiger files-Eintrag")
        if set(eintrag) != {"name", "size", "sha256", "records"}:
            raise SicherungsFehler(
                f"{archiv}: files-Eintrag weicht vom v2-Schema ab"
            )
        name = eintrag.get("name")
        if name not in FESTWISSEN_DATEIEN or name in gesehen:
            raise SicherungsFehler(
                f"{archiv}: unzulaessiger/doppelter Manifest-Dateiname {name!r}"
            )
        gesehen.add(name)
        if type(eintrag.get("size")) is not int or eintrag["size"] < 0:
            raise SicherungsFehler(f"{archiv}: ungueltige Groesse fuer {name}")
        if type(eintrag.get("records")) is not int or eintrag["records"] < 0:
            raise SicherungsFehler(f"{archiv}: ungueltige Recordzahl fuer {name}")
        if not isinstance(eintrag.get("sha256"), str) or not _SHA256_RE.fullmatch(
            eintrag["sha256"]
        ):
            raise SicherungsFehler(f"{archiv}: ungueltiger SHA-256 fuer {name}")

    counts = manifest.get("counts")
    if isinstance(counts, dict) and set(counts) != {
        "definitionen",
        "norm_kuratierungen",
    }:
        raise SicherungsFehler(
            f"{archiv}: counts-Schluessel weichen vom v2-Schema ab"
        )
    if (
        not isinstance(counts, dict)
        or type(counts.get("definitionen")) is not int
        or counts["definitionen"] < 0
        or type(counts.get("norm_kuratierungen")) is not int
        or counts["norm_kuratierungen"] < 0
    ):
        raise SicherungsFehler(f"{archiv}: ungueltige Gesamt-Recordzahlen")
    return eintraege, counts


def _lade_archiv(archiv: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    try:
        zf = zipfile.ZipFile(archiv)
    except (zipfile.BadZipFile, OSError) as exc:
        raise SicherungsFehler(
            f"{archiv}: kein gueltiges Zip-Archiv ({exc})"
        ) from exc

    with zf:
        infos = _pruefe_archiv_limits(zf, archiv)
        namen = {i.filename for i in infos}
        erlaubt = set(FESTWISSEN_DATEIEN) | {MANIFEST_DATEI}
        fremd = namen - erlaubt
        if fremd:
            raise SicherungsFehler(
                f"{archiv}: unerwarteter Archiv-Inhalt abgewehrt: "
                f"{sorted(fremd)!r}"
            )
        try:
            inhalte = {info.filename: zf.read(info) for info in infos}
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise SicherungsFehler(
                f"{archiv}: Archiv-Inhalt nicht integer lesbar: {exc}"
            ) from exc

    if MANIFEST_DATEI not in inhalte:
        if not inhalte:
            raise SicherungsFehler(
                f"{archiv}: leeres manifestloses Legacy-Archiv wird abgelehnt"
            )
        legacy_dateien = {
            name: inhalte[name] for name in FESTWISSEN_DATEIEN if name in inhalte
        }
        recordzahlen, summen = _validiere_semantik(
            legacy_dateien, kontext=f"Legacy-Archiv {archiv}"
        )
        info: dict[str, Any] = {
            "format": "legacy-manifestlos",
            "format_version": 1,
            "schema_version": None,
            "legacy": True,
            "created_at": None,
            "release": None,
            "files": [
                {
                    "name": name,
                    "size": len(legacy_dateien[name]),
                    "sha256": _sha256(legacy_dateien[name]),
                    "records": recordzahlen[name],
                }
                for name in FESTWISSEN_DATEIEN
                if name in legacy_dateien
            ],
            "counts": summen,
        }
        return legacy_dateien, info

    manifest = _json_ohne_doppelte_keys(inhalte[MANIFEST_DATEI], archiv)
    eintraege, erwartete_counts = _validiere_manifest(manifest, archiv)
    erwartete_namen = {e["name"] for e in eintraege}
    echte_namen = set(inhalte) - {MANIFEST_DATEI}
    if echte_namen != erwartete_namen:
        raise SicherungsFehler(
            f"{archiv}: Manifest und Archiv-Dateien stimmen nicht ueberein"
        )

    dateien: dict[str, bytes] = {}
    for eintrag in eintraege:
        name = eintrag["name"]
        daten = inhalte[name]
        if len(daten) != eintrag["size"]:
            raise SicherungsFehler(f"{archiv}: Groessenpruefung fehlgeschlagen: {name}")
        if _sha256(daten) != eintrag["sha256"]:
            raise SicherungsFehler(f"{archiv}: SHA-256-Pruefung fehlgeschlagen: {name}")
        dateien[name] = daten

    recordzahlen, echte_counts = _validiere_semantik(
        dateien, kontext=f"Archiv {archiv}"
    )
    for eintrag in eintraege:
        if recordzahlen[eintrag["name"]] != eintrag["records"]:
            raise SicherungsFehler(
                f"{archiv}: Recordzahl stimmt nicht: {eintrag['name']}"
            )
    if echte_counts != erwartete_counts:
        raise SicherungsFehler(f"{archiv}: Gesamt-Recordzahlen stimmen nicht")
    return dateien, {**manifest, "legacy": False}


def pruefe_sicherung(archiv: str | Path) -> dict[str, Any]:
    """Prueft ein Archiv vollstaendig, ohne ein Restore-Ziel zu veraendern.

    Die Rueckgabe ist direkt JSON-serialisierbar und eignet sich fuer
    ``risk-layerctl backup verify``. Bei manifestlosen Altarchiven ist
    ``legacy=True`` gesetzt; die berechneten Hashes sind dort nur Diagnose und
    nicht durch ein mitgeliefertes Manifest gebunden.
    """
    pfad = Path(archiv)
    if not pfad.is_file():
        raise SicherungsFehler(f"Archiv nicht gefunden: {pfad}")
    _dateien, info = _lade_archiv(pfad)
    return info


def bereinige_sicherungen(
    ziel_dir: str | Path,
    *,
    keep_last: int = 30,
    keep_days: int = 90,
    apply: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Plant oder vollzieht eine fail-closed Retention verifizierter v2-Backups.

    Mindestens ``keep_last`` neueste Archive sowie alle Archive innerhalb von
    ``keep_days`` bleiben erhalten. Fremde Dateien werden nie angefasst;
    unsichere oder korrupte passend benannte Archive brechen den Lauf vor der
    ersten Loeschung ab.
    """
    if (
        isinstance(keep_last, bool)
        or not isinstance(keep_last, int)
        or keep_last < 1
    ):
        raise SicherungsFehler("keep_last muss eine Ganzzahl >= 1 sein")
    if (
        isinstance(keep_days, bool)
        or not isinstance(keep_days, int)
        or keep_days < 1
    ):
        raise SicherungsFehler("keep_days muss eine Ganzzahl >= 1 sein")
    ziel = Path(ziel_dir)
    _pruefe_verzeichniskette(ziel)
    if _ist_link_oder_reparse(ziel) or not ziel.is_dir():
        raise SicherungsFehler(f"Unsicheres oder fehlendes Backup-Ziel: {ziel}")

    archive: list[tuple[Path, os.stat_result]] = []
    for pfad in ziel.iterdir():
        if not (pfad.name.startswith(ARCHIV_PREFIX) and pfad.name.endswith(".zip")):
            continue
        try:
            info = os.lstat(pfad)
        except OSError as exc:
            raise SicherungsFehler(f"Backup nicht sicher lesbar: {pfad}: {exc}") from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or _ist_link_oder_reparse(pfad)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise SicherungsFehler(f"Unsicheres passend benanntes Backup: {pfad}")
        manifest = pruefe_sicherung(pfad)
        if manifest.get("legacy") is not False:
            raise SicherungsFehler(
                f"Retention loescht kein manifestloses Legacy-Archiv: {pfad}"
            )
        archive.append((pfad, info))

    archive.sort(key=lambda eintrag: (eintrag[1].st_mtime_ns, eintrag[0].name), reverse=True)
    cutoff = (time.time() if now is None else now) - keep_days * 86400
    behalten = {
        pfad
        for index, (pfad, info) in enumerate(archive)
        if index < keep_last or info.st_mtime >= cutoff
    }
    kandidaten = [(pfad, info) for pfad, info in archive if pfad not in behalten]

    geloescht: list[str] = []
    if apply:
        for pfad, vorher in kandidaten:
            try:
                aktuell = os.lstat(pfad)
            except OSError as exc:
                raise SicherungsFehler(
                    f"Backup hat sich vor Retention veraendert: {pfad}: {exc}"
                ) from exc
            identitaet_vorher = (
                vorher.st_dev,
                vorher.st_ino,
                vorher.st_size,
                vorher.st_mtime_ns,
                vorher.st_nlink,
            )
            identitaet_aktuell = (
                aktuell.st_dev,
                aktuell.st_ino,
                aktuell.st_size,
                aktuell.st_mtime_ns,
                aktuell.st_nlink,
            )
            if identitaet_aktuell != identitaet_vorher or not stat.S_ISREG(
                aktuell.st_mode
            ):
                raise SicherungsFehler(
                    f"Backup wurde vor Retention ausgetauscht: {pfad}"
                )
            pfad.unlink()
            geloescht.append(str(pfad))
        if geloescht:
            _fsync_verzeichnis(ziel)

    return {
        "ok": True,
        "target": str(ziel),
        "apply": apply,
        "keep_last": keep_last,
        "keep_days": keep_days,
        "total": len(archive),
        "kept": len(archive) - len(kandidaten),
        "candidates": [str(pfad) for pfad, _ in kandidaten],
        "deleted": geloescht,
    }


def _staging_erstellen(ziel: Path, dateien: dict[str, bytes]) -> Path:
    _pruefe_verzeichniskette(ziel.parent)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    _pruefe_verzeichniskette(ziel.parent)
    staging = Path(
        tempfile.mkdtemp(dir=str(ziel.parent), prefix=f".{ziel.name}.restore-")
    )
    try:
        _restriktive_rechte(staging, verzeichnis=True)
        for name in FESTWISSEN_DATEIEN:
            if name in dateien:
                _schreibe_datei_exklusiv(staging / name, dateien[name])
        _fsync_verzeichnis(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def _rollback_pfad(ziel: Path) -> Path:
    for _ in range(10):
        kandidat = ziel.parent / f".{ziel.name}.vor-restore-{uuid.uuid4().hex}"
        if not kandidat.exists():
            return kandidat
    raise SicherungsFehler("Kein freier Rollback-Pfad fuer Restore gefunden")


def _promote_transaktional(staging: Path, ziel: Path) -> None:
    """Schaltet den kompletten verwalteten Store um und rollt Fehler zurueck.

    Unverwaltete Eintraege (insbesondere ``sicherungen/``) werden aus dem alten
    Verzeichnis in den neuen Stand uebernommen. Verwaltete, im Snapshot fehlende
    YAMLs bleiben dagegen bewusst entfernt.
    """
    rollback: Path | None = _rollback_pfad(ziel) if ziel.exists() else None
    ziel_aktiv = False
    verschoben: list[str] = []
    try:
        restore_transaktion_beginnen(ziel, staging, rollback)
    except DateisperrenFehler as exc:
        raise SicherungsFehler(f"Restore-Journal konnte nicht angelegt werden: {exc}") from exc
    try:
        if rollback is not None:
            os.replace(ziel, rollback)
        os.replace(staging, ziel)
        ziel_aktiv = True

        if rollback is not None:
            for kind in list(rollback.iterdir()):
                if kind.name in FESTWISSEN_DATEIEN:
                    continue
                os.replace(kind, ziel / kind.name)
                verschoben.append(kind.name)
        _fsync_verzeichnis(ziel)
        _fsync_verzeichnis(ziel.parent)
    except BaseException as exc:
        # Bereits uebernommene unverwaltete Eintraege zurueckbewegen, dann den
        # kompletten alten Store wieder unter seinem Originalnamen aktivieren.
        try:
            if rollback is not None:
                for name in reversed(verschoben):
                    if (ziel / name).exists() or (ziel / name).is_symlink():
                        os.replace(ziel / name, rollback / name)
                if ziel_aktiv and ziel.exists():
                    fehlerstand = ziel.parent / f".{ziel.name}.failed-{uuid.uuid4().hex}"
                    os.replace(ziel, fehlerstand)
                    os.replace(rollback, ziel)
                    shutil.rmtree(fehlerstand, ignore_errors=True)
                elif not ziel.exists():
                    os.replace(rollback, ziel)
            elif ziel_aktiv and ziel.exists():
                shutil.rmtree(ziel, ignore_errors=True)
        except BaseException as rollback_exc:
            raise SicherungsFehler(
                f"Restore-Promotion fehlgeschlagen ({exc}); automatischer "
                f"Rollback ebenfalls fehlgeschlagen ({rollback_exc})"
            ) from rollback_exc
        try:
            restore_transaktion_beenden(ziel)
        except DateisperrenFehler as journal_exc:
            raise SicherungsFehler(
                f"Restore-Promotion fehlgeschlagen ({exc}); alter Stand wurde "
                f"wiederhergestellt, Journal-Cleanup aber nicht ({journal_exc})"
            ) from journal_exc
        raise SicherungsFehler(
            f"Restore-Promotion fehlgeschlagen; alter Stand wurde "
            f"wiederhergestellt: {exc}"
        ) from exc
    else:
        if rollback is not None:
            # Cleanup ist nach erfolgreicher Promotion kein Restore-Fehler.
            # Ein verbliebener, restriktiver Rollback-Baum kann beim naechsten
            # Wartungsfenster entfernt werden; der aktive Stand bleibt korrekt.
            shutil.rmtree(rollback, ignore_errors=True)
        try:
            restore_transaktion_beenden(ziel)
        except DateisperrenFehler as exc:
            raise SicherungsFehler(
                f"Restore ist aktiv, aber das Crash-Recovery-Journal konnte "
                f"nicht abgeschlossen werden: {exc}"
            ) from exc


def stelle_wieder_her(
    archiv: str | Path, ziel_dir: str | Path, force: bool = False
) -> list[str]:
    """Validiert vollstaendig und promotet dann einen exakten Store-Snapshot.

    Ein v2-Leersnapshot entfernt mit ``force=True`` alle drei verwalteten YAMLs.
    Manifestlose, nichtleere Legacy-Archive werden aus Kompatibilitaetsgruenden
    akzeptiert, aber ebenfalls vor der ersten Live-Aenderung semantisch geprueft.
    """
    archiv_pfad = Path(archiv)
    if not archiv_pfad.is_file():
        raise SicherungsFehler(f"Archiv nicht gefunden: {archiv_pfad}")
    ziel = Path(ziel_dir)
    _pruefe_verzeichniskette(ziel)
    if _ist_link_oder_reparse(ziel):
        raise SicherungsFehler(f"Unsicheres Restore-Ziel (Link/Reparse-Point): {ziel}")

    dateien, _info = _lade_archiv(archiv_pfad)
    staging = _staging_erstellen(ziel, dateien)
    try:
        try:
            with festwissen_sperre(ziel, timeout=_LOCK_TIMEOUT):
                if ziel.exists() and not ziel.is_dir():
                    raise SicherungsFehler(
                        f"Restore-Ziel ist kein Verzeichnis: {ziel}"
                    )
                if ziel.exists() and any(ziel.iterdir()) and not force:
                    raise SicherungsFehler(
                        f"Ziel {ziel} ist nicht leer - Wiederherstellen wuerde "
                        "bestehendes Festwissen ueberschreiben. Mit force=True "
                        "(CLI: --force) ausdruecklich bestaetigen."
                    )
                _promote_transaktional(staging, ziel)
        except DateisperrenFehler as exc:
            raise SicherungsFehler(str(exc)) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return [name for name in FESTWISSEN_DATEIEN if name in dateien]


def stelle_wieder_her_mit_sicherung(
    archiv: str | Path,
    ziel_dir: str | Path,
    sicherungs_ziel: str | Path,
    zeitstempel: str,
    *,
    force: bool = False,
    release_id: str | None = None,
    git_commit: str | None = None,
    release_manifest_sha256: str | None = None,
) -> tuple[list[str], Path | None]:
    """Sichert den Live-Stand und restauriert unter genau demselben Lock.

    Damit kann zwischen Pflicht-Vorher-Sicherung und Promotion keine
    BeraterStore-Mutation mehr ungesichert ueberschrieben werden. Ein leeres
    oder noch nicht vorhandenes Ziel braucht keine Vorher-Sicherung.
    """
    archiv_pfad = Path(archiv)
    if not archiv_pfad.is_file():
        raise SicherungsFehler(f"Archiv nicht gefunden: {archiv_pfad}")
    ziel = Path(ziel_dir)
    _pruefe_verzeichniskette(ziel)
    if _ist_link_oder_reparse(ziel):
        raise SicherungsFehler(f"Unsicheres Restore-Ziel (Link/Reparse-Point): {ziel}")

    dateien, _info = _lade_archiv(archiv_pfad)
    staging = _staging_erstellen(ziel, dateien)
    vorher: Path | None = None
    try:
        try:
            with festwissen_sperre(ziel, timeout=_LOCK_TIMEOUT):
                if ziel.exists() and not ziel.is_dir():
                    raise SicherungsFehler(
                        f"Restore-Ziel ist kein Verzeichnis: {ziel}"
                    )
                nichtleer = ziel.exists() and any(ziel.iterdir())
                if nichtleer and not force:
                    raise SicherungsFehler(
                        f"Ziel {ziel} ist nicht leer - Wiederherstellen wuerde "
                        "bestehendes Festwissen ueberschreiben. Mit force=True "
                        "(CLI: --force) ausdruecklich bestaetigen."
                    )
                if nichtleer:
                    live_dateien = _stabiler_snapshot(ziel)
                    vorher = _sicherung_publizieren(
                        live_dateien,
                        sicherungs_ziel,
                        zeitstempel,
                        release_id=release_id,
                        git_commit=git_commit,
                        release_manifest_sha256=release_manifest_sha256,
                    )
                    # Die Pflichtsicherung muss noch unter demselben Lock wieder
                    # vollstaendig lesbar und semantisch gueltig sein.
                    pruefe_sicherung(vorher)
                _promote_transaktional(staging, ziel)
        except DateisperrenFehler as exc:
            raise SicherungsFehler(str(exc)) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return [name for name in FESTWISSEN_DATEIEN if name in dateien], vorher
