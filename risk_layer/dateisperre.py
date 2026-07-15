"""Kleine, plattformuebergreifende Interprozess-Sperre fuer Festwissen.

Die Sperrdatei liegt *neben* dem BeraterStore-Verzeichnis. Dadurch bleibt ihr
Pfad stabil, waehrend ein Restore das Store-Verzeichnis transaktional umbenennt.
Alle Prozesse, die Festwissen mutieren, muessen dieselbe ``festwissen_sperre``
verwenden; Backup und Restore tun dies ebenfalls.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator


class DateisperrenFehler(TimeoutError):
    """Eine Interprozess-Sperre konnte nicht rechtzeitig erworben werden."""


_RESTORE_JOURNAL_VERSION = 1
_MAX_RESTORE_JOURNAL_BYTES = 4096
MAX_OPERATION_TIMEOUT_SECONDS = 3600.0
MAX_POLL_INTERVAL_SECONDS = 60.0


def _bounded_float(
    value: float,
    name: str,
    maximum: float,
    *,
    allow_zero: bool = False,
) -> float:
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        normalized = math.nan
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(normalized)
        or normalized < 0
        or (normalized == 0 and not allow_zero)
        or normalized > maximum
    ):
        operator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} muss endlich und {operator} bis {maximum:g}s sein")
    return normalized


def festwissen_sperrpfad(verzeichnis: str | Path) -> Path:
    """Liefert den kanonischen, auch bei Verzeichnis-Swaps stabilen Lockpfad."""
    store = Path(verzeichnis)
    return store.parent / f".{store.name}.festwissen.lock"


def _restore_journalpfad(store: Path) -> Path:
    return store.parent / f".{store.name}.restore-journal.json"


def _ist_reparse(st: os.stat_result) -> bool:
    attribut = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attribut and getattr(st, "st_file_attributes", 0) & attribut)


def _pruefe_elternkette(pfad: Path) -> None:
    """Blockt umgelenkte Elternverzeichnisse (Symlink/Windows-Reparse-Point)."""
    absolut = Path(os.path.abspath(pfad))
    teile = [absolut, *absolut.parents]
    for verzeichnis in reversed(teile):
        try:
            st = os.lstat(verzeichnis)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(st.st_mode) or _ist_reparse(st):
            raise DateisperrenFehler(
                f"Unsichere Symlink-/Reparse-Komponente im Lockpfad: {verzeichnis}"
            )
        if not stat.S_ISDIR(st.st_mode):
            raise DateisperrenFehler(
                f"Elternkomponente des Lockpfads ist kein Verzeichnis: {verzeichnis}"
            )


def _fsync_verzeichnis(pfad: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(pfad, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _journal_name_pruefen(store: Path, name: object, art: str) -> str:
    if not isinstance(name, str) or Path(name).name != name:
        raise DateisperrenFehler(f"Ungueltiger {art}-Pfad im Restore-Journal")
    prefix = (
        f".{store.name}.restore-" if art == "Staging" else f".{store.name}.vor-restore-"
    )
    if not name.startswith(prefix) or len(name) > 255:
        raise DateisperrenFehler(f"Ungueltiger {art}-Name im Restore-Journal")
    return name


def _sicheres_restore_verzeichnis(pfad: Path, beschreibung: str) -> bool:
    try:
        info = os.lstat(pfad)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DateisperrenFehler(
            f"{beschreibung} kann nicht sicher geprueft werden: {pfad}: {exc}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or _ist_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise DateisperrenFehler(f"Unsicheres {beschreibung}: {pfad}")
    return True


def restore_transaktion_beginnen(
    verzeichnis: str | Path,
    staging: str | Path,
    rollback: str | Path | None,
) -> None:
    """Persistiert vor dem ersten Rename ein kleines Crash-Recovery-Journal."""
    store = Path(os.path.abspath(verzeichnis))
    stage = Path(os.path.abspath(staging))
    rueck = Path(os.path.abspath(rollback)) if rollback is not None else None
    if stage.parent != store.parent or (rueck is not None and rueck.parent != store.parent):
        raise DateisperrenFehler("Restore-Pfade muessen direkte Geschwister des Stores sein")
    _journal_name_pruefen(store, stage.name, "Staging")
    if rueck is not None:
        _journal_name_pruefen(store, rueck.name, "Rollback")
    if not _sicheres_restore_verzeichnis(stage, "Restore-Staging"):
        raise DateisperrenFehler(f"Restore-Staging fehlt: {stage}")

    journal = _restore_journalpfad(store)
    if journal.exists() or journal.is_symlink():
        raise DateisperrenFehler(
            f"Unbereinigtes Restore-Journal trotz Lock-Recovery: {journal}"
        )
    payload = (
        json.dumps(
            {
                "format_version": _RESTORE_JOURNAL_VERSION,
                "staging": stage.name,
                "rollback": rueck.name if rueck is not None else None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        dir=store.parent, prefix=f".{journal.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, journal)
        _fsync_verzeichnis(store.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def restore_transaktion_beenden(verzeichnis: str | Path) -> None:
    store = Path(os.path.abspath(verzeichnis))
    journal = _restore_journalpfad(store)
    try:
        info = os.lstat(journal)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or _ist_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise DateisperrenFehler(f"Unsicheres Restore-Journal: {journal}")
    journal.unlink()
    _fsync_verzeichnis(store.parent)


def _json_eindeutig(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DateisperrenFehler(f"Doppelter Restore-Journal-Schluessel: {key}")
        result[key] = value
    return result


def _restore_nach_crash_reparieren(store: Path) -> None:
    """Rollt eine durch Prozess-/Hostabbruch unterbrochene Promotion sicher zurueck."""
    journal = _restore_journalpfad(store)
    try:
        info = os.lstat(journal)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DateisperrenFehler(f"Restore-Journal nicht lesbar: {journal}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or _ist_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise DateisperrenFehler(f"Unsicheres Restore-Journal: {journal}")
    if info.st_size > _MAX_RESTORE_JOURNAL_BYTES:
        raise DateisperrenFehler("Restore-Journal ueberschreitet Groessenlimit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(journal, flags)
    try:
        opened = os.fstat(fd)
        named = os.lstat(journal)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or _ist_reparse(named)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or opened.st_nlink != 1
        ):
            raise DateisperrenFehler(
                f"Restore-Journal wurde ausgetauscht oder hat Hardlinks: {journal}"
            )
        if os.name != "nt" and stat.S_IMODE(opened.st_mode) & 0o077:
            raise DateisperrenFehler(
                f"Restore-Journal hat zu weite Rechte: {journal}"
            )
        raw = os.read(fd, _MAX_RESTORE_JOURNAL_BYTES + 1)
    finally:
        os.close(fd)
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_eindeutig)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DateisperrenFehler(f"Restore-Journal ist kein gueltiges JSON: {journal}") from exc
    if not isinstance(document, dict) or set(document) != {
        "format_version", "staging", "rollback"
    }:
        raise DateisperrenFehler("Restore-Journal hat falsches Schema")
    if document["format_version"] != _RESTORE_JOURNAL_VERSION:
        raise DateisperrenFehler("Unbekannte Restore-Journal-Version")
    staging = store.parent / _journal_name_pruefen(store, document["staging"], "Staging")
    rollback_name = document["rollback"]
    rollback = (
        store.parent / _journal_name_pruefen(store, rollback_name, "Rollback")
        if rollback_name is not None
        else None
    )
    store_da = _sicheres_restore_verzeichnis(store, "Restore-Ziel")
    staging_da = _sicheres_restore_verzeichnis(staging, "Restore-Staging")
    rollback_da = (
        _sicheres_restore_verzeichnis(rollback, "Restore-Rollback")
        if rollback is not None
        else False
    )

    if rollback_da and rollback is not None:
        if store_da:
            fehlstand = store.parent / f".{store.name}.failed-recovery-{os.getpid()}"
            if fehlstand.exists() or fehlstand.is_symlink():
                raise DateisperrenFehler(f"Crash-Recovery-Ziel existiert bereits: {fehlstand}")
            os.replace(store, fehlstand)
            os.replace(rollback, store)
            shutil.rmtree(fehlstand, ignore_errors=True)
        else:
            os.replace(rollback, store)
        if staging_da:
            shutil.rmtree(staging, ignore_errors=True)
    elif rollback is not None:
        if not store_da:
            raise DateisperrenFehler(
                "Restore-Crash nicht sicher reparierbar: Ziel und Rollback fehlen"
            )
        if staging_da:
            shutil.rmtree(staging, ignore_errors=True)
    elif store_da:
        if staging_da:
            shutil.rmtree(staging, ignore_errors=True)
    elif staging_da:
        os.replace(staging, store)
    else:
        raise DateisperrenFehler(
            "Restore-Crash nicht sicher reparierbar: Ziel und Staging fehlen"
        )
    restore_transaktion_beenden(store)
    _fsync_verzeichnis(store.parent)


def _oeffne_lockdatei(pfad: Path) -> BinaryIO:
    """Oeffnet/erzeugt die Lockdatei ohne Symlink-Folgen und prueft die Identitaet."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    basis_flags = os.O_RDWR | getattr(os, "O_NOINHERIT", 0) | nofollow
    erstellt = False
    try:
        fd = os.open(pfad, basis_flags | os.O_CREAT | os.O_EXCL, 0o600)
        erstellt = True
    except FileExistsError:
        try:
            vor = os.lstat(pfad)
        except OSError as exc:
            raise DateisperrenFehler(
                f"Lockdatei vor dem Oeffnen nicht sicher pruefbar: {pfad}: {exc}"
            ) from exc
        if stat.S_ISLNK(vor.st_mode) or _ist_reparse(vor):
            raise DateisperrenFehler(
                f"Lockdatei ist ein symbolischer Link/Reparse-Point: {pfad}"
            )
        if not stat.S_ISREG(vor.st_mode):
            raise DateisperrenFehler(f"Lockpfad ist keine regulaere Datei: {pfad}")
        try:
            fd = os.open(pfad, basis_flags)
        except OSError as exc:
            raise DateisperrenFehler(
                f"Lockdatei nicht sicher zu oeffnen: {pfad}: {exc}"
            ) from exc
    except OSError as exc:
        # Windows meldet fuer ein bereits vorhandenes Verzeichnis mit O_EXCL
        # teils EACCES statt EEXIST. lstat liefert dennoch die praezise,
        # sicherheitsrelevante Diagnose.
        try:
            vor = os.lstat(pfad)
        except OSError:
            raise DateisperrenFehler(
                f"Lockdatei nicht sicher anzulegen: {pfad}: {exc}"
            ) from exc
        if stat.S_ISLNK(vor.st_mode) or _ist_reparse(vor):
            raise DateisperrenFehler(
                f"Lockdatei ist ein symbolischer Link/Reparse-Point: {pfad}"
            ) from exc
        if not stat.S_ISREG(vor.st_mode):
            raise DateisperrenFehler(
                f"Lockpfad ist keine regulaere Datei: {pfad}"
            ) from exc
        raise DateisperrenFehler(
            f"Lockdatei nicht sicher anzulegen: {pfad}: {exc}"
        ) from exc

    try:
        handle_st = os.fstat(fd)
        nach = os.lstat(pfad)
        if (
            not stat.S_ISREG(handle_st.st_mode)
            or stat.S_ISLNK(nach.st_mode)
            or _ist_reparse(nach)
            or not stat.S_ISREG(nach.st_mode)
            or (handle_st.st_dev, handle_st.st_ino) != (nach.st_dev, nach.st_ino)
        ):
            raise DateisperrenFehler(
                f"Lockdatei wurde waehrend des Oeffnens ausgetauscht: {pfad}"
            )
        # Ein Hardlink koennte eine fremde regulaere Datei unter dem Locknamen
        # einschleusen. Wir schreiben zwar keine Nutzdaten, sperren sie aber auch
        # nicht als Seiteneffekt.
        if handle_st.st_nlink != 1:
            raise DateisperrenFehler(
                f"Lockdatei hat unerwartete Hardlinks ({handle_st.st_nlink}): {pfad}"
            )
        if os.name != "nt":
            geteuid = getattr(os, "geteuid")
            if handle_st.st_uid != geteuid():
                raise DateisperrenFehler(
                    f"Lockdatei gehoert einem anderen Benutzer: {pfad}"
                )
            if erstellt:
                getattr(os, "fchmod")(fd, 0o600)
            elif stat.S_IMODE(handle_st.st_mode) & 0o077:
                raise DateisperrenFehler(
                    f"Lockdatei hat zu weite Rechte (erwartet 0600): {pfad}"
                )
        else:
            # Python kann unter Windows ohne pywin32 keine ACL neu schreiben;
            # O_CREAT erbt die ACL des zuvor geprueften Elternverzeichnisses.
            # chmod entfernt zumindest ein eventuell gesetztes Read-only-Bit.
            os.chmod(pfad, stat.S_IREAD | stat.S_IWRITE)
    except BaseException:
        os.close(fd)
        if erstellt:
            try:
                os.unlink(pfad)
            except OSError:
                pass
        raise
    return os.fdopen(fd, "r+b", buffering=0)


def _versuche_sperren(datei: BinaryIO) -> bool:
    datei.seek(0)
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(datei.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        getattr(fcntl, "flock")(
            datei.fileno(),
            getattr(fcntl, "LOCK_EX") | getattr(fcntl, "LOCK_NB"),
        )
    except (BlockingIOError, PermissionError):
        return False
    return True


def _entsperren(datei: BinaryIO) -> None:
    datei.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(datei.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        getattr(fcntl, "flock")(datei.fileno(), getattr(fcntl, "LOCK_UN"))


@contextmanager
def festwissen_sperre(
    verzeichnis: str | Path,
    *,
    timeout: float = 10.0,
    poll_intervall: float = 0.05,
) -> Iterator[None]:
    """Haelt exklusiv die gemeinsame Festwissen-Sperre.

    ``flock`` (Unix) beziehungsweise ``msvcrt.locking`` (Windows) bindet die
    Sperre an das offene Handle. Nach Prozessabsturz gibt das Betriebssystem sie
    automatisch frei; eine verwaiste Lockdatei ist daher harmlos und muss nicht
    anhand unsicherer PID-/Zeitstempel-Heuristiken geloescht werden.
    """
    timeout = _bounded_float(
        timeout,
        "timeout",
        MAX_OPERATION_TIMEOUT_SECONDS,
        allow_zero=True,
    )
    poll_intervall = _bounded_float(
        poll_intervall,
        "poll_intervall",
        MAX_POLL_INTERVAL_SECONDS,
    )

    store = Path(os.path.abspath(verzeichnis))
    # Der reale Store selbst gehoert zur Sicherheitsidentitaet. Ein Alias ueber
    # Symlink/Junction koennte sonst fuer dieselben Daten einen zweiten,
    # unabhaengigen Lockpfad erzeugen und Read-Modify-Write-Zyklen verlieren.
    _pruefe_elternkette(store)
    pfad = Path(os.path.abspath(festwissen_sperrpfad(store)))
    # Schon vor mkdir die bis zum ersten fehlenden Segment existierende Kette
    # pruefen. Sonst koennte mkdir unter einer vorbereiteten Symlink-Komponente
    # Verzeichnisse an einem fremden Ort erzeugen. Die zweite Pruefung schliesst
    # das Race-Fenster waehrend mkdir soweit ohne openat/Win32-Handle moeglich.
    _pruefe_elternkette(pfad.parent)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    _pruefe_elternkette(pfad.parent)
    _pruefe_elternkette(store)
    datei = _oeffne_lockdatei(pfad)
    erworben = False
    try:
        # msvcrt.locking darf auch einen Bereich hinter EOF sperren. Deshalb
        # bleibt die Lockdatei absichtlich inhaltslos und kann niemals fremde
        # Nutzdaten veraendern.
        ende = time.monotonic() + timeout
        while not (erworben := _versuche_sperren(datei)):
            rest = ende - time.monotonic()
            if rest <= 0:
                raise DateisperrenFehler(
                    f"Festwissen-Sperre nicht innerhalb von {timeout:.3f}s "
                    f"erhalten: {pfad}"
                )
            time.sleep(min(poll_intervall, rest))
        # Ein abgestuerzter Restore wird vor jedem weiteren Store-Zugriff
        # deterministisch abgeschlossen oder auf den alten Stand gerollt.
        _restore_nach_crash_reparieren(store)
        yield
    finally:
        if erworben:
            _entsperren(datei)
        datei.close()
