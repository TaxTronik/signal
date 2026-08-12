"""Abgesicherter Laufzeit-Operator fuer den semantischen Index.

Graph, Katalog und Modell bleiben unveraenderliche Build-Inputs. Neue Indizes
landen als vollstaendige Generation in einem beschreibbaren Overlay; erst ein
atomar ersetzter Zeiger veroeffentlicht die Generation. Die laufende Anwendung
tauscht danach ein einziges (Index, Encoder)-Snapshot aus, sodass parallele
Analysen niemals alte und neue Komponenten mischen.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
import re
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .embedding import (
    EmbeddingBuildCancelled,
    EmbeddingIndex,
    encoder_oeffnen,
    pruefe_aktualitaet,
)
from .graph import graph_oeffnen

_log = logging.getLogger(__name__)

JOB_STATES = frozenset(
    {
        "idle",
        "queued",
        "running",
        "cancelling",
        "cancelled",
        "succeeded",
        "failed",
        "skipped",
    }
)
_ACTIVE_JOB_STATES = frozenset({"queued", "running", "cancelling"})
_GENERATION_RE = re.compile(r"^[a-f0-9]{32}$")
_TEMP_GENERATION_RE = re.compile(r"^\.tmp-[a-f0-9]{32}$")
_STATE_MAX_BYTES = 64 * 1024
_DEFAULT_INTERVAL_DAYS = 7
_STATE_SCHEMA = "risk-layer/embedding-operator-state/v2"


class EmbeddingOperatorUnavailable(RuntimeError):
    """Der Build-Pfad ist in diesem Deployment nicht sicher nutzbar."""


class EmbeddingOperatorConflict(RuntimeError):
    """Der angeforderte Job ist nicht mehr der aktive Single-flight-Job."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(zeitpunkt: datetime) -> str:
    return zeitpunkt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(wert: Any) -> datetime | None:
    if not isinstance(wert, str) or len(wert) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(wert.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _fsync_directory(pfad: Path) -> None:
    """Persistiert Directory-Entries soweit das Zielsystem dies unterstuetzt."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(pfad, flags)
    except OSError:
        # Windows bietet fuer Verzeichnisse nicht auf jedem Dateisystem einen
        # fsync-faehigen Descriptor. Die atomaren Renames bleiben dort erhalten.
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_json(pfad: Path, obj: dict[str, Any]) -> None:
    """Schreibt JSON per fsync + os.replace; niemals einen Teilzustand."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(f".{pfad.name}.{uuid.uuid4().hex}.tmp")
    daten = (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(daten)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, pfad)
        _fsync_directory(pfad.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _default_job() -> dict[str, Any]:
    return {
        "state": "idle",
        "id": None,
        "requested": None,
        "started": None,
        "completed": None,
        "last_success": None,
        "error": None,
    }


def _default_schedule(_configured: bool, _now: datetime) -> dict[str, Any]:
    # Ein Operator-Token gibt nur die Steuerung frei. Installation/Start darf
    # niemals implizit einen rechenintensiven Build planen; der Wochenplan wird
    # erst durch einen ausdruecklichen, bestaetigten Operator-POST aktiviert.
    enabled = False
    return {
        "enabled": enabled,
        "interval_days": _DEFAULT_INTERVAL_DAYS,
        "last_check_at": None,
        "next_run_at": None,
    }


def _safe_job(raw: Any) -> dict[str, Any]:
    out = _default_job()
    if not isinstance(raw, dict):
        return out
    state = raw.get("state")
    if state in JOB_STATES:
        out["state"] = state
    job_id = raw.get("id")
    if isinstance(job_id, str) and _GENERATION_RE.fullmatch(job_id):
        out["id"] = job_id
    for name in ("requested", "started", "completed", "last_success"):
        parsed = _parse_iso(raw.get(name))
        out[name] = _iso(parsed) if parsed is not None else None
    error = raw.get("error")
    if isinstance(error, str):
        out["error"] = error[:240]
    return out


def _safe_schedule(raw: Any, configured: bool, now: datetime) -> dict[str, Any]:
    out = _default_schedule(configured, now)
    if not isinstance(raw, dict):
        return out
    if isinstance(raw.get("enabled"), bool):
        out["enabled"] = raw["enabled"]
    interval = raw.get("interval_days")
    if (
        isinstance(interval, int)
        and not isinstance(interval, bool)
        and 1 <= interval <= 365
    ):
        out["interval_days"] = interval
    for name in ("last_check_at", "next_run_at"):
        parsed = _parse_iso(raw.get(name))
        out[name] = _iso(parsed) if parsed is not None else None
    if not out["enabled"]:
        out["next_run_at"] = None
    elif out["next_run_at"] is None:
        out["next_run_at"] = _iso(now + timedelta(days=out["interval_days"]))
    return out


class EmbeddingOperator:
    """Persistenter Single-flight-Job und globaler Wochenplan."""

    def __init__(
        self,
        *,
        graph_path: str | Path,
        katalog: Any,
        state_dir: str | Path,
        configured: bool,
        device: str,
        snapshot: Callable[[], tuple[Any, Any]],
        activate: Callable[[EmbeddingIndex, Any], None],
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.katalog = katalog
        self.root = Path(state_dir)
        self.generations = self.root / "generations"
        self.state_path = self.root / "operator-state.json"
        self.pointer_path = self.root / "current.json"
        self.configured = configured
        self.device = device
        self._snapshot = snapshot
        self._activate = activate
        self._now = now
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._schedule_wakeup = threading.Event()
        self._cancel_event = threading.Event()
        self._cancel_job_id: str | None = None
        self._scheduler_thread: threading.Thread | None = None
        self._job_thread: threading.Thread | None = None
        self._closing = False
        self._available = False
        self._unavailable_reason: str | None = None

        self._probe_availability()
        self._state = self._load_state()
        if self._available:
            self._cleanup_abandoned_temps()
        pointer = self._pointer_generations()
        recovered_generation = self._recover_generation(pointer)
        self._recover_interrupted_job(recovered_generation)
        # Nur ein strikt validierter Pointer autorisiert Retention. Damit werden
        # auch Final-Verzeichnisse eines harten Crashes zwischen Generation-
        # Rename und Pointer-Publish entfernt, ohne bei defektem State Daten zu
        # erraten oder fremden Pfaden zu folgen.
        pointer = self._pointer_generations()
        if self._available and pointer is not None:
            self._prune_generations({item for item in pointer if item is not None})

    # -- Initialisierung / Recovery -------------------------------------
    def _probe_availability(self) -> None:
        if not self.configured:
            self._unavailable_reason = "Operator-Token ist nicht konfiguriert"
            return
        if not self.graph_path.is_file():
            self._unavailable_reason = "Normgraph ist nicht verfuegbar"
            return
        try:
            self.generations.mkdir(parents=True, exist_ok=True)
            # Builds schreiben und listen genau dieses Unterverzeichnis. Eine
            # Probe nur im Parent wuerde falsch geerbte Volume-ACLs uebersehen.
            list(self.generations.iterdir())
            probe = self.generations / f".write-probe-{uuid.uuid4().hex}"
            fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            probe.unlink()
        except OSError:
            self._unavailable_reason = "Embedding-Overlay ist nicht beschreibbar"
            return

        index, _encoder = self._snapshot()
        encoder_name = getattr(index, "encoder_name", None) or "bge-m3"
        if encoder_name == "bge-m3":
            modell = os.environ.get("RISK_LAYER_EMBEDDING_MODEL")
            offline = os.environ.get("RISK_LAYER_EMBEDDING_OFFLINE") == "1"
            if offline and not modell:
                self._unavailable_reason = "Lokales Embedding-Modell fehlt"
                return
            if modell and not Path(modell).is_dir():
                self._unavailable_reason = (
                    "Lokales Embedding-Modell ist nicht verfuegbar"
                )
                return
            if importlib.util.find_spec("sentence_transformers") is None:
                self._unavailable_reason = "Embedding-Runtime ist nicht installiert"
                return
        self._available = True

    def _load_state(self) -> dict[str, Any]:
        now = self._now()
        default = {
            "schema": _STATE_SCHEMA,
            "job": _default_job(),
            "schedule": _default_schedule(self.configured and self._available, now),
        }
        if not self.state_path.is_file():
            if self._available:
                try:
                    self._persist(default)
                except EmbeddingOperatorUnavailable:
                    pass
            return default
        try:
            if self.state_path.stat().st_size > _STATE_MAX_BYTES:
                raise ValueError("state too large")
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("state is not an object")
            # v1 aktivierte den Wochenplan bereits durch das Vorhandensein des
            # Operator-Tokens. Das war keine belastbare Zustimmung zu einem
            # ressourcenintensiven Build. Beim einmaligen v2-Upgrade wird dieser
            # implizite Plan daher fail-safe ausgeschaltet; danach bleibt jede
            # explizite v2-Entscheidung erhalten.
            legacy_schedule = raw.get("schema") != _STATE_SCHEMA
            state = {
                "schema": _STATE_SCHEMA,
                "job": _safe_job(raw.get("job")),
                "schedule": _safe_schedule(
                    None if legacy_schedule else raw.get("schedule"),
                    self.configured,
                    now,
                ),
            }
            if legacy_schedule and self._available:
                try:
                    self._persist(state)
                except EmbeddingOperatorUnavailable:
                    pass
            return state
        except (OSError, ValueError, json.JSONDecodeError):
            _log.warning(
                "Embedding-Operatorstate ist unlesbar; sichere Defaults werden verwendet"
            )
            if self._available:
                try:
                    self._persist(default)
                except EmbeddingOperatorUnavailable:
                    pass
            return default

    def _persist(self, state: dict[str, Any] | None = None) -> None:
        if not self._available:
            return
        try:
            _atomic_json(self.state_path, state if state is not None else self._state)
        except OSError as exc:
            self._available = False
            self._unavailable_reason = "Embedding-Operatorstate ist nicht beschreibbar"
            raise EmbeddingOperatorUnavailable(self._unavailable_reason) from exc

    def _recover_interrupted_job(self, recovered_generation: str | None) -> None:
        job = self._state["job"]
        if job["state"] not in _ACTIVE_JOB_STATES:
            return
        cancellation_was_requested = job["state"] == "cancelling"
        completed = _iso(self._now())
        job["completed"] = completed
        if recovered_generation is not None and recovered_generation == job.get("id"):
            # Crashfenster nach atomarem Pointer-Swap, aber vor dem letzten
            # Jobstate-Write: die verifizierte Generation ist der Erfolgsbeleg.
            job["state"] = "succeeded"
            job["last_success"] = completed
            job["error"] = None
            schedule = self._state["schedule"]
            if schedule["enabled"]:
                schedule["next_run_at"] = _iso(
                    self._now() + timedelta(days=schedule["interval_days"])
                )
        else:
            job["state"] = "cancelled" if cancellation_was_requested else "failed"
            job["error"] = (
                None
                if cancellation_was_requested
                else "Vorheriger Engine-Prozess wurde waehrend des Builds beendet"
            )
            schedule = self._state["schedule"]
            if schedule["enabled"] and not cancellation_was_requested:
                # Der faellige Check hatte seinen Folgetermin bereits vor dem
                # Build reserviert. Nach einem Prozessabbruch zeitnah erneut
                # pruefen statt den veralteten/fehlenden Index eine Woche liegen
                # zu lassen. Der Scheduler setzt vor dem Versuch wieder den
                # regulaeren Intervalltermin, sodass kein enger Fehlerloop entsteht.
                schedule["next_run_at"] = completed
        try:
            self._persist()
        except EmbeddingOperatorUnavailable:
            pass

    def _pointer_generations(self) -> tuple[str, str | None] | None:
        if not self.pointer_path.is_file():
            return None
        try:
            if self.pointer_path.stat().st_size > 4096:
                raise ValueError("pointer too large")
            raw = json.loads(self.pointer_path.read_text(encoding="utf-8"))
            generation = raw.get("generation") if isinstance(raw, dict) else None
            if not isinstance(generation, str) or not _GENERATION_RE.fullmatch(
                generation
            ):
                raise ValueError("invalid generation")
            previous = raw.get("previous")
            if previous is not None and (
                not isinstance(previous, str)
                or not _GENERATION_RE.fullmatch(previous)
                or previous == generation
            ):
                raise ValueError("invalid previous generation")
            return generation, previous
        except (OSError, ValueError, json.JSONDecodeError):
            _log.warning(
                "Embedding-Generationszeiger ist ungueltig; Legacy-Index bleibt aktiv"
            )
            return None

    def _pointer_generation(self) -> str | None:
        pointer = self._pointer_generations()
        return pointer[0] if pointer is not None else None

    def _activate_recovered_generation(
        self,
        *,
        generation: str,
        index: EmbeddingIndex,
        current: Any,
        current_encoder: Any,
        repoint: bool,
    ) -> str:
        if (
            current_encoder is not None
            and getattr(current, "encoder_name", None) == index.encoder_name
            and getattr(current, "encoder_revision", None) == index.encoder_revision
        ):
            encoder = current_encoder
        else:
            encoder = encoder_oeffnen(index.encoder_name, device=self.device)
        self._activate(index, encoder)
        if repoint:
            # Der Current-Kandidat war unbrauchbar oder fuer die aktiven Inputs
            # veraltet. Den verifizierten Previous-Fallback atomar publizieren,
            # damit ein spaeterer Build/Retention-Lauf ihn nicht verwirft.
            try:
                _atomic_json(self.pointer_path, {"generation": generation})
            except OSError:
                _log.warning(
                    "Embedding-Fallback konnte nicht als Current publiziert werden"
                )
        return generation

    def _recover_generation(
        self, pointer: tuple[str, str | None] | None
    ) -> str | None:
        if pointer is None:
            return None
        current, current_encoder = self._snapshot()
        try:
            graph = graph_oeffnen(self.graph_path)
        except Exception:  # noqa: BLE001 - Overlay darf Engine-Start nie blockieren
            _log.exception("Normgraph konnte fuer Embedding-Recovery nicht geoeffnet werden")
            graph = None
        try:
            release_current = bool(
                graph is not None
                and current is not None
                and pruefe_aktualitaet(current, graph, self.katalog)
            )
        except Exception:  # noqa: BLE001 - Status bleibt degradierend
            _log.exception("Release-/Legacy-Index konnte nicht verifiziert werden")
            release_current = False
        candidates: list[tuple[int, str, EmbeddingIndex, bool]] = []
        for position, generation in enumerate(item for item in pointer if item is not None):
            verzeichnis = self.generations / generation
            try:
                index = EmbeddingIndex.load(verzeichnis)
                if index is None:
                    raise ValueError("incomplete generation")
                candidates.append(
                    (
                        position,
                        generation,
                        index,
                        bool(
                            graph is not None
                            and pruefe_aktualitaet(index, graph, self.katalog)
                        ),
                    )
                )
            except Exception:  # noqa: BLE001 - Pointer darf Engine-Start nie blockieren
                _log.exception(
                    "%s Embedding-Generation ist nicht lesbar",
                    "Aktuelle" if position == 0 else "Vorige",
                )
        if graph is not None:
            graph.close()

        # Zuerst alle nachweislich zu den aktiven Inputs passenden Kandidaten:
        # Current hat Vorrang, Previous rettet Rollbacks mit anderem Graphstand.
        for position, generation, index, is_current in candidates:
            if not is_current:
                continue
            try:
                return self._activate_recovered_generation(
                    generation=generation,
                    index=index,
                    current=current,
                    current_encoder=current_encoder,
                    repoint=position == 1,
                )
            except Exception:  # noqa: BLE001 - Recovery bleibt degradierend
                _log.exception(
                    "%s Embedding-Generation ist nicht aktivierbar",
                    "Aktuelle" if position == 0 else "Vorige",
                )

        if current is not None:
            _log.warning(
                "Persistierte Embedding-Generationen sind veraltet; der %s "
                "Release-/Legacy-Index bleibt aktiv",
                "aktuelle" if release_current else "vorhandene",
            )
            return None

        # Ohne ausgelieferten/Legacy-Index ist ein lesbarer stale Kandidat ein
        # ehrlicher degradierter Fallback (Status current=false), Current vor
        # Previous. Einen passenden Previous haette der Block oben bevorzugt.
        for position, generation, index, _is_current in candidates:
            try:
                return self._activate_recovered_generation(
                    generation=generation,
                    index=index,
                    current=current,
                    current_encoder=current_encoder,
                    repoint=position == 1,
                )
            except Exception:  # noqa: BLE001 - Recovery bleibt degradierend
                _log.exception(
                    "%s stale Embedding-Generation ist nicht aktivierbar",
                    "Aktuelle" if position == 0 else "Vorige",
                )
        return None

    def _cleanup_abandoned_temps(self) -> None:
        """Entfernt nur eindeutig benannte, nie publizierte Crash-Temporaries."""
        try:
            candidates = list(self.generations.iterdir())
        except OSError:
            _log.warning(
                "Embedding-Generationenverzeichnis konnte nicht bereinigt werden"
            )
            return
        for candidate in candidates:
            if not _TEMP_GENERATION_RE.fullmatch(candidate.name):
                continue
            try:
                info = os.lstat(candidate)
                reparse = bool(
                    getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                    and getattr(info, "st_file_attributes", 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT
                )
                if stat.S_ISLNK(info.st_mode) or reparse:
                    _log.warning(
                        "Unerwarteter Link als Embedding-Tempverzeichnis wird "
                        "nicht automatisch entfernt"
                    )
                    continue
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(candidate)
            except OSError:
                _log.warning(
                    "Verwaistes Embedding-Tempverzeichnis konnte nicht "
                    "bereinigt werden"
                )

    # -- Status / API ----------------------------------------------------
    def _index_status(self) -> dict[str, Any]:
        index, encoder = self._snapshot()
        active = index is not None and encoder is not None
        current = (
            None
            if index is None
            else bool(pruefe_aktualitaet(index, self._graph_for_status(), self.katalog))
        )
        return {
            "active": active,
            "current": current,
            "fingerprint": getattr(index, "graph_fingerprint", None),
            "input_fingerprint": getattr(index, "input_fingerprint", None),
            "encoder": getattr(index, "encoder_name", None),
            "revision": (
                getattr(index, "encoder_revision", None)
                or getattr(encoder, "revision", None)
            ),
        }

    def _graph_for_status(self) -> Any:
        """Nutzt den App-Graph aus dem Aktivitaetscheck ohne ihn neu zu oeffnen.

        ``pruefe_aktualitaet`` braucht nur fingerprint(). Der beim App-Start
        geladene Graph ist ueber den Index-Buildinput stabil; ein kleiner Proxy
        vermeidet einen SQLite-Open pro GET. Der Fingerprint wird beim App-Start
        ohnehin auf dem Graphobjekt memoisiert und dem Operator uebergeben.
        """
        return getattr(self, "_status_graph", None)

    def bind_status_graph(self, graph: Any) -> None:
        self._status_graph = graph

    def status(self) -> dict[str, Any]:
        with self._lock:
            job = copy.deepcopy(self._state["job"])
            schedule = copy.deepcopy(self._state["schedule"])
            available = self._available and not self._closing
            reason = self._unavailable_reason
        return {
            "index": self._index_status(),
            "job": job,
            "schedule": schedule,
            "refresh_available": available,
            "unavailable_reason": None if available else reason,
        }

    def trigger(self, *, force: bool) -> dict[str, str]:
        with self._lock:
            if not self._available or self._closing:
                raise EmbeddingOperatorUnavailable(
                    self._unavailable_reason
                    or "Embedding-Operator ist nicht verfuegbar"
                )
            job = self._state["job"]
            if job["state"] in _ACTIVE_JOB_STATES and job["id"]:
                return {"job_id": job["id"], "state": job["state"]}
            job_id = uuid.uuid4().hex
            self._cancel_job_id = None
            self._cancel_event.clear()
            self._state["job"] = {
                "state": "queued",
                "id": job_id,
                "requested": _iso(self._now()),
                "started": None,
                "completed": None,
                "last_success": job.get("last_success"),
                "error": None,
            }
            self._persist()
            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, force),
                name=f"embedding-build-{job_id[:8]}",
                daemon=True,
            )
            self._job_thread = thread
            try:
                thread.start()
            except Exception as exc:  # noqa: BLE001 - OS-/Threadlimit fail-closed
                self._job_thread = None
                failed = self._state["job"]
                if failed.get("id") == job_id:
                    failed["state"] = "failed"
                    failed["completed"] = _iso(self._now())
                    failed["error"] = "Embedding-Worker konnte nicht gestartet werden"
                    self._persist()
                raise EmbeddingOperatorUnavailable(
                    "Embedding-Worker konnte nicht gestartet werden"
                ) from exc
            return {"job_id": job_id, "state": "queued"}

    def cancel(self, *, job_id: str) -> dict[str, str]:
        with self._lock:
            if not self._available or self._closing:
                raise EmbeddingOperatorUnavailable(
                    self._unavailable_reason
                    or "Embedding-Operator ist nicht verfuegbar"
                )
            job = self._state["job"]
            if job.get("id") != job_id:
                raise EmbeddingOperatorConflict(
                    "Der Embedding-Job ist nicht mehr der aktive Job"
                )
            if job["state"] not in _ACTIVE_JOB_STATES:
                return {"job_id": job_id, "state": job["state"]}
            job["state"] = "cancelling"
            self._persist()
            self._cancel_job_id = job_id
            self._cancel_event.set()
            return {"job_id": job_id, "state": "cancelling"}

    def set_schedule(self, *, enabled: bool, interval_days: int) -> dict[str, Any]:
        with self._lock:
            if not self._available or self._closing:
                raise EmbeddingOperatorUnavailable(
                    self._unavailable_reason
                    or "Embedding-Operator ist nicht verfuegbar"
                )
            alt = self._state["schedule"]
            unveraendert = (
                alt["enabled"] == enabled and alt["interval_days"] == interval_days
            )
            next_run = alt["next_run_at"] if unveraendert else None
            if enabled and next_run is None:
                next_run = _iso(self._now() + timedelta(days=interval_days))
            self._state["schedule"] = {
                "enabled": enabled,
                "interval_days": interval_days,
                "last_check_at": alt["last_check_at"],
                "next_run_at": next_run if enabled else None,
            }
            self._persist()
            result = copy.deepcopy(self._state["schedule"])
        self._schedule_wakeup.set()
        return result

    # -- Build / Aktivierung --------------------------------------------
    def _public_build_error(self, exc: Exception) -> str:
        if isinstance(exc, EmbeddingOperatorUnavailable):
            return str(exc)[:240]
        return f"Embedding-Build fehlgeschlagen ({type(exc).__name__})"

    def _run_job(self, job_id: str, force: bool) -> None:
        try:
            with self._lock:
                if self._state["job"].get("id") != job_id:
                    return
                if self._state["job"]["state"] == "cancelling":
                    self._finish_job(job_id, "cancelled")
                    return
                self._state["job"]["state"] = "running"
                self._state["job"]["started"] = _iso(self._now())
                self._persist()

            if not force and self._index_status()["current"]:
                self._finish_job(job_id, "skipped")
                return
            self._build_generation(job_id)
            self._finish_job(job_id, "succeeded")
        except EmbeddingBuildCancelled:
            self._finish_job(job_id, "cancelled")
        except Exception as exc:  # noqa: BLE001 - Jobfehler werden persistiert
            _log.exception("Embedding-Operatorbuild fehlgeschlagen")
            self._finish_job(job_id, "failed", error=self._public_build_error(exc))

    def _build_generation(self, generation: str) -> None:
        def cancelled() -> bool:
            return (
                self._cancel_event.is_set()
                and self._cancel_job_id == generation
            )

        if cancelled():
            raise EmbeddingBuildCancelled("Embedding-Build wurde abgebrochen")
        graph = graph_oeffnen(self.graph_path)
        if graph is None:
            raise EmbeddingOperatorUnavailable("Normgraph ist nicht verfuegbar")
        tmp = self.generations / f".tmp-{generation}"
        final = self.generations / generation
        published = False
        try:
            index_alt, _ = self._snapshot()
            encoder_name = getattr(index_alt, "encoder_name", None) or "bge-m3"
            encoder = encoder_oeffnen(encoder_name, device=self.device)
            if cancelled():
                raise EmbeddingBuildCancelled("Embedding-Build wurde abgebrochen")
            gebaut = EmbeddingIndex.build(
                graph,
                encoder,
                katalog=self.katalog,
                cancelled=cancelled,
            )
            if cancelled():
                raise EmbeddingBuildCancelled("Embedding-Build wurde abgebrochen")

            tmp.mkdir(parents=False, exist_ok=False)
            gebaut.save(tmp)
            for datei in (tmp / "vectors.npy", tmp / "meta.json"):
                # Windows verlangt fuer FlushFileBuffers einen schreibbaren
                # Descriptor; r+b mutiert den Inhalt nicht, erlaubt aber fsync.
                with datei.open("r+b") as stream:
                    os.fsync(stream.fileno())
            # Erst die Dateieintraege innerhalb der privaten Generation, dann
            # nach dem Rename deren Namen im generations/-Parent persistieren.
            # Der Pointer darf erst danach durable sichtbar werden.
            _fsync_directory(tmp)
            verifiziert = EmbeddingIndex.load(tmp)
            if verifiziert is None or not pruefe_aktualitaet(
                verifiziert, graph, self.katalog
            ):
                raise RuntimeError("generated index verification failed")
            if cancelled():
                raise EmbeddingBuildCancelled("Embedding-Build wurde abgebrochen")
            os.replace(tmp, final)
            _fsync_directory(self.generations)
            previous = self._pointer_generation()
            pointer: dict[str, Any] = {"generation": generation}
            if previous is not None and previous != generation:
                pointer["previous"] = previous
            _atomic_json(self.pointer_path, pointer)
            published = True
            self._activate(verifiziert, encoder)
            self._prune_generations({generation, previous})
        finally:
            graph.close()
            if tmp.exists():
                # Exakt diese unveroeffentlichte, zufaellig benannte Generation
                # gehoert dem laufenden Job. Ein published Verzeichnis wird nie geloescht.
                shutil.rmtree(tmp, ignore_errors=True)
            if not published:
                # Nach Rename, aber vor Pointer-Publish darf ein I/O-Fehler kein
                # modellgrosses, fuer immer unerreichbares Verzeichnis hinterlassen.
                # Ein unerwartet ausgetauschter Link/Reparse-Punkt wird nie verfolgt.
                try:
                    info = os.lstat(final)
                    reparse = bool(
                        getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                        and getattr(info, "st_file_attributes", 0)
                        & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    )
                    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(
                        info.st_mode
                    ) and not reparse:
                        shutil.rmtree(final)
                except FileNotFoundError:
                    pass
                except OSError:
                    _log.warning(
                        "Unpublizierte Embedding-Generation konnte nicht bereinigt werden"
                    )

    def _prune_generations(self, keep: set[str | None]) -> None:
        """Behaelt aktuelle + vorige Generation; folgt niemals Reparse-Punkten."""
        try:
            candidates = list(self.generations.iterdir())
        except OSError:
            _log.warning(
                "Embedding-Generationenverzeichnis konnte nicht gelesen werden"
            )
            return
        for candidate in candidates:
            if not _GENERATION_RE.fullmatch(candidate.name) or candidate.name in keep:
                continue
            try:
                info = os.lstat(candidate)
                reparse = bool(
                    getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                    and getattr(info, "st_file_attributes", 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT
                )
                if stat.S_ISLNK(info.st_mode) or reparse:
                    _log.warning(
                        "Unerwarteter Link im Embedding-Generationenverzeichnis "
                        "wird nicht automatisch entfernt"
                    )
                    continue
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(candidate)
            except OSError:
                # Retention darf eine bereits atomar aktivierte Generation nicht
                # nachtraeglich als Buildfehler deklarieren.
                _log.warning("Alte Embedding-Generation konnte nicht bereinigt werden")

    def _finish_job(self, job_id: str, state: str, *, error: str | None = None) -> None:
        completed = self._now()
        with self._lock:
            job = self._state["job"]
            if job.get("id") != job_id:
                return
            job["state"] = state
            job["completed"] = _iso(completed)
            job["error"] = error
            if state == "succeeded":
                job["last_success"] = job["completed"]
                schedule = self._state["schedule"]
                if schedule["enabled"]:
                    schedule["next_run_at"] = _iso(
                        completed + timedelta(days=schedule["interval_days"])
                    )
            self._persist()
        self._schedule_wakeup.set()

    # -- Scheduler / Shutdown ------------------------------------------
    def start(self) -> None:
        with self._lock:
            if (
                not self._available
                or self._closing
                or self._scheduler_thread is not None
            ):
                return
            thread = threading.Thread(
                target=self._scheduler_loop,
                name="embedding-scheduler",
                daemon=True,
            )
            self._scheduler_thread = thread
            try:
                thread.start()
            except Exception as exc:  # noqa: BLE001 - OS-/Threadlimit fail-closed
                self._scheduler_thread = None
                raise EmbeddingOperatorUnavailable(
                    "Embedding-Scheduler konnte nicht gestartet werden"
                ) from exc

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                schedule = copy.deepcopy(self._state["schedule"])
            if not schedule["enabled"] or not schedule["next_run_at"]:
                timeout = None
            else:
                due = _parse_iso(schedule["next_run_at"])
                timeout = max(0.0, (due - self._now()).total_seconds()) if due else 0.0
                # Windows begrenzt Event.wait auf threading.TIMEOUT_MAX
                # (typisch knapp 50 Tage), waehrend der API-Vertrag bis zu
                # 365 Tage erlaubt. In sicheren Etappen warten; _scheduled_check
                # prueft den persistierten Termin unter Lock erneut und kehrt bei
                # einem noch nicht faelligen oder parallel verschobenen Plan um.
                timeout = min(timeout, threading.TIMEOUT_MAX)
            woke = self._schedule_wakeup.wait(timeout)
            self._schedule_wakeup.clear()
            if self._stop.is_set():
                return
            if woke:
                continue
            try:
                self._scheduled_check()
            except EmbeddingOperatorUnavailable:
                _log.warning(
                    "Embedding-Scheduler wurde wegen unbeschreibbarem State deaktiviert"
                )

    def _scheduled_check(self) -> None:
        now = self._now()
        with self._lock:
            schedule = self._state["schedule"]
            if not schedule["enabled"]:
                return
            due = _parse_iso(schedule["next_run_at"])
            if due is not None and due > now:
                return
            schedule["last_check_at"] = _iso(now)
            schedule["next_run_at"] = _iso(
                now + timedelta(days=schedule["interval_days"])
            )
            observed_job = self._state["job"]
            observed_job_identity = (
                observed_job.get("id"), observed_job.get("state")
            )
            active = observed_job["state"] in _ACTIVE_JOB_STATES
            self._persist()
        if active:
            return
        if self._index_status()["current"]:
            with self._lock:
                current_job = self._state["job"]
                # Zwischen Statuspruefung und diesem Lock kann ein manueller
                # Force-Job angenommen (und sogar beendet) worden sein. Dessen
                # ID/State niemals mit dem Scheduler-Skip ueberschreiben.
                if (
                    (current_job.get("id"), current_job.get("state"))
                    != observed_job_identity
                    or current_job["state"] in _ACTIVE_JOB_STATES
                ):
                    return
                vorher = self._state["job"]
                self._state["job"] = {
                    "state": "skipped",
                    "id": uuid.uuid4().hex,
                    "requested": _iso(now),
                    "started": None,
                    "completed": _iso(now),
                    "last_success": vorher.get("last_success"),
                    "error": None,
                }
                self._persist()
            return
        try:
            self.trigger(force=False)
        except EmbeddingOperatorUnavailable:
            _log.warning("Geplanter Embedding-Check konnte keinen Build starten")

    def close(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            self._closing = True
        self._stop.set()
        self._schedule_wakeup.set()
        scheduler = self._scheduler_thread
        if scheduler is not None and scheduler is not threading.current_thread():
            scheduler.join(timeout=max(0.0, deadline - time.monotonic()))
        worker = self._job_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                _log.warning(
                    "Embedding-Build laeuft beim Shutdown noch; unveroeffentlichte "
                    "Generation bleibt crash-sicher isoliert"
                )

    def wait_for_job(self, timeout: float = 10.0) -> None:
        """Nur fuer kontrollierte Tests/Operator-Integrationen."""
        worker = self._job_thread
        if worker is not None:
            worker.join(timeout=timeout)
