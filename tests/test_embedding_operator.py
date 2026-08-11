import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

import risk_layer.embedding_operator as embedding_operator_mod

from risk_layer.catalog import Katalog
from risk_layer.embedding import EmbeddingIndex, HashEncoder
from risk_layer.embedding_operator import EmbeddingOperator, EmbeddingOperatorUnavailable
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.models import Katalogeintrag


def _ressourcen(tmp_path):
    graph = NormGraph.build(
        [
            NormKnoten(
                id="norm:AO:146",
                law="AO",
                para="146",
                absatz=None,
                nummer=None,
                titel="Ordnungsvorschriften fuer Aufzeichnungen",
                text="Aufzeichnungen sind einzeln, vollstaendig und richtig zu fuehren.",
            ),
        ],
        tmp_path / "graph.sqlite",
    )
    katalog = Katalog(
        "2026.1",
        [
            Katalogeintrag(
                id="kasse",
                begriff_kanonisch="Kassenaufzeichnung",
                norm_anker=("§ 146 AO",),
                paraphrasen=("Bareinnahmen werden nicht einzeln festgehalten.",),
            ),
        ],
    )
    encoder = HashEncoder(dim=32)
    index = EmbeddingIndex.build(graph, encoder, katalog=katalog)
    return graph, katalog, index, encoder


def _operator(tmp_path, *, current=True):
    graph, katalog, index, encoder = _ressourcen(tmp_path)
    runtime = [index if current else None, encoder if current else None]
    aktivierungen = []

    def snapshot():
        return runtime[0], runtime[1]

    def activate(neuer_index, neuer_encoder):
        runtime[:] = [neuer_index, neuer_encoder]
        aktivierungen.append(neuer_index)

    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=tmp_path / "overlay",
        configured=True,
        device="cpu",
        snapshot=snapshot,
        activate=activate,
    )
    operator.bind_status_graph(graph)
    return operator, graph, runtime, aktivierungen


def test_default_status_ist_woechentlich_und_index_aktuell(tmp_path):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    try:
        status = operator.status()
        assert status["index"] == {
            "active": True,
            "current": True,
            "fingerprint": graph.fingerprint(),
            "input_fingerprint": _runtime[0].input_fingerprint,
            "encoder": "hash32",
            "revision": None,
        }
        assert (
            status["index"]["input_fingerprint"]
            != _runtime[0].embedding_graph_fingerprint
        )
        assert status["job"]["state"] == "idle"
        assert status["schedule"]["enabled"] is True
        assert status["schedule"]["interval_days"] == 7
        assert status["schedule"]["next_run_at"] is not None
    finally:
        operator.close()
        graph.close()


def test_force_build_publiziert_generation_und_verschiebt_plan(tmp_path):
    operator, graph, runtime, aktivierungen = _operator(tmp_path)
    try:
        vorher = runtime[0]
        accepted = operator.trigger(force=True)
        assert accepted["state"] == "queued"
        operator.wait_for_job()

        status = operator.status()
        assert status["job"]["state"] == "succeeded"
        assert status["job"]["id"] == accepted["job_id"]
        assert status["job"]["last_success"] == status["job"]["completed"]
        assert runtime[0] is not vorher and aktivierungen == [runtime[0]]
        pointer = json.loads(operator.pointer_path.read_text(encoding="utf-8"))
        assert pointer == {"generation": accepted["job_id"]}
        assert (operator.generations / accepted["job_id"] / "meta.json").is_file()
        completed = datetime.fromisoformat(
            status["job"]["completed"].replace("Z", "+00:00")
        )
        next_run = datetime.fromisoformat(
            status["schedule"]["next_run_at"].replace("Z", "+00:00")
        )
        assert (next_run - completed).days == 7
    finally:
        operator.close()
        graph.close()


def test_publish_fsynct_private_und_generations_verzeichnis(
    tmp_path, monkeypatch
):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    fsync_dirs = []
    monkeypatch.setattr(
        embedding_operator_mod,
        "_fsync_directory",
        lambda pfad: fsync_dirs.append(pfad),
    )
    try:
        accepted = operator.trigger(force=True)
        operator.wait_for_job()
        assert operator.status()["job"]["state"] == "succeeded"
        assert operator.generations in fsync_dirs
        assert operator.generations / f".tmp-{accepted['job_id']}" in fsync_dirs
    finally:
        operator.close()
        graph.close()


def test_retention_behaelt_nur_aktuelle_und_vorige_generation(tmp_path):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    try:
        ids = []
        for _ in range(3):
            accepted = operator.trigger(force=True)
            ids.append(accepted["job_id"])
            operator.wait_for_job()
            assert operator.status()["job"]["state"] == "succeeded"
        vorhanden = sorted(
            path.name
            for path in operator.generations.iterdir()
            if path.is_dir() and len(path.name) == 32
        )
        assert vorhanden == sorted(ids[-2:])
        pointer = json.loads(operator.pointer_path.read_text(encoding="utf-8"))
        assert pointer == {"generation": ids[-1], "previous": ids[-2]}
    finally:
        operator.close()
        graph.close()


def test_non_force_ueberspringt_nachweislich_aktuellen_index(tmp_path):
    operator, graph, _runtime, aktivierungen = _operator(tmp_path)
    try:
        accepted = operator.trigger(force=False)
        operator.wait_for_job()
        status = operator.status()
        assert status["job"]["id"] == accepted["job_id"]
        assert status["job"]["state"] == "skipped"
        assert status["job"]["last_success"] is None
        assert aktivierungen == []
        assert not list(operator.generations.iterdir())
    finally:
        operator.close()
        graph.close()


def test_single_flight_gibt_dieselbe_job_id_zurueck(tmp_path, monkeypatch):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    gestartet = threading.Event()
    freigabe = threading.Event()

    def blockierender_build(_job_id):
        gestartet.set()
        assert freigabe.wait(5)

    monkeypatch.setattr(operator, "_build_generation", blockierender_build)
    try:
        first = operator.trigger(force=True)
        assert gestartet.wait(5)
        second = operator.trigger(force=True)
        assert second["job_id"] == first["job_id"]
        assert second["state"] == "running"
        freigabe.set()
        operator.wait_for_job()
        assert operator.status()["job"]["state"] == "succeeded"
    finally:
        freigabe.set()
        operator.close()
        graph.close()


def test_buildfehler_persistiert_keine_exception_details(tmp_path, monkeypatch):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)

    def boom(_job_id):
        raise RuntimeError("SECRET=/srv/private/model und Traceback")

    monkeypatch.setattr(operator, "_build_generation", boom)
    try:
        operator.trigger(force=True)
        operator.wait_for_job()
        job = operator.status()["job"]
        assert job["state"] == "failed"
        assert job["error"] == "Embedding-Build fehlgeschlagen (RuntimeError)"
        persisted = operator.state_path.read_text(encoding="utf-8")
        assert "SECRET" not in persisted and "Traceback" not in persisted
    finally:
        operator.close()
        graph.close()


def test_interrupted_state_wird_beim_start_als_failed_markiert(tmp_path):
    graph, katalog, index, encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    root.mkdir()
    job_id = "a" * 32
    (root / "operator-state.json").write_text(
        json.dumps(
            {
                "job": {
                    "state": "running",
                    "id": job_id,
                    "requested": "2026-08-01T10:00:00Z",
                    "started": "2026-08-01T10:00:01Z",
                    "completed": None,
                    "last_success": "2026-07-01T10:00:00Z",
                    "error": None,
                },
                "schedule": {
                    "enabled": True,
                    "interval_days": 7,
                    "next_run_at": "2026-08-08T10:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    recovered_at = datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)
    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: (index, encoder),
        activate=lambda *_args: None,
        now=lambda: recovered_at,
    )
    operator.bind_status_graph(graph)
    try:
        job = operator.status()["job"]
        assert job["state"] == "failed"
        assert job["id"] == job_id and job["completed"] is not None
        assert "Vorheriger Engine-Prozess" in job["error"]
        assert job["last_success"] == "2026-07-01T10:00:00Z"
        # Ein unterbrochener faelliger Build wird beim Schedulerstart sofort
        # erneut geprueft statt erst am bereits reservierten Wochentermin.
        assert operator.status()["schedule"]["next_run_at"] == job["completed"]
    finally:
        operator.close()
        graph.close()


def test_crash_nach_pointer_swap_wird_als_erfolg_rekonstruiert(tmp_path):
    graph, katalog, legacy, encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    generation = "c" * 32
    (root / "generations" / generation).mkdir(parents=True)
    neu = EmbeddingIndex.build(graph, HashEncoder(dim=64), katalog=katalog)
    neu.save(root / "generations" / generation)
    (root / "current.json").write_text(
        json.dumps({"generation": generation}), encoding="utf-8"
    )
    (root / "operator-state.json").write_text(
        json.dumps(
            {
                "job": {
                    "state": "running",
                    "id": generation,
                    "requested": "2026-08-01T10:00:00Z",
                    "started": "2026-08-01T10:00:01Z",
                },
                "schedule": {"enabled": True, "interval_days": 7},
            }
        ),
        encoding="utf-8",
    )
    runtime = [legacy, encoder]
    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: tuple(runtime),
        activate=lambda idx, enc: runtime.__setitem__(slice(None), [idx, enc]),
    )
    operator.bind_status_graph(graph)
    try:
        job = operator.status()["job"]
        assert job["state"] == "succeeded"
        assert job["last_success"] == job["completed"]
        assert job["error"] is None
        assert runtime[0].dim == 64
    finally:
        operator.close()
        graph.close()


def test_gueltiger_pointer_wird_aktiviert_und_korrupter_faellt_auf_legacy_zurueck(
    tmp_path,
):
    graph, katalog, legacy, encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    generation = "b" * 32
    (root / "generations" / generation).mkdir(parents=True)
    neu = EmbeddingIndex.build(graph, HashEncoder(dim=64), katalog=katalog)
    neu.save(root / "generations" / generation)
    root.mkdir(exist_ok=True)
    (root / "current.json").write_text(
        json.dumps({"generation": generation}), encoding="utf-8"
    )
    runtime = [legacy, encoder]
    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: tuple(runtime),
        activate=lambda idx, enc: runtime.__setitem__(slice(None), [idx, enc]),
    )
    operator.bind_status_graph(graph)
    try:
        assert runtime[0] is not legacy and runtime[0].dim == 64
    finally:
        operator.close()

    # Traversal/Teilzeiger wird ignoriert; der Legacy-Snapshot bleibt aktiv.
    (root / "current.json").write_text(
        json.dumps({"generation": "../outside"}), encoding="utf-8"
    )
    runtime[:] = [legacy, encoder]
    fallback = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: tuple(runtime),
        activate=lambda idx, enc: runtime.__setitem__(slice(None), [idx, enc]),
    )
    fallback.bind_status_graph(graph)
    try:
        assert runtime[0] is legacy
    finally:
        fallback.close()
        graph.close()


def test_stales_overlay_ueberschreibt_release_index_nicht(tmp_path):
    graph, katalog, release_index, encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    generation = "d" * 32
    (root / "generations" / generation).mkdir(parents=True)
    stale = EmbeddingIndex.build(graph, HashEncoder(dim=64), katalog=katalog)
    stale.graph_fingerprint = "0" * 64
    stale.save(root / "generations" / generation)
    (root / "current.json").write_text(
        json.dumps({"generation": generation}), encoding="utf-8"
    )
    runtime = [release_index, encoder]
    aktivierungen = []
    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: tuple(runtime),
        activate=lambda idx, enc: aktivierungen.append((idx, enc)),
    )
    operator.bind_status_graph(graph)
    try:
        assert runtime == [release_index, encoder]
        assert aktivierungen == []
        assert operator.status()["index"]["current"] is True
    finally:
        operator.close()
        graph.close()


def test_start_bereinigt_unreferenzierte_final_generation(tmp_path):
    graph, katalog, release_index, encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    current = "1" * 32
    previous = "2" * 32
    orphan = "3" * 32
    for generation in (current, previous, orphan):
        release_index.save(root / "generations" / generation)
    (root / "current.json").write_text(
        json.dumps({"generation": current, "previous": previous}), encoding="utf-8"
    )
    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: (release_index, encoder),
        activate=lambda *_args: None,
    )
    operator.bind_status_graph(graph)
    try:
        assert (root / "generations" / current).is_dir()
        assert (root / "generations" / previous).is_dir()
        assert not (root / "generations" / orphan).exists()
    finally:
        operator.close()
        graph.close()


def test_korrupter_current_pointer_faellt_atomar_auf_previous_zurueck(tmp_path):
    graph, katalog, release_index, encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    current = "4" * 32
    previous = "5" * 32
    (root / "generations" / current).mkdir(parents=True)
    release_index.save(root / "generations" / previous)
    (root / "current.json").write_text(
        json.dumps({"generation": current, "previous": previous}), encoding="utf-8"
    )
    runtime = [release_index, encoder]
    aktivierungen = []

    def activate(index, runtime_encoder):
        runtime[:] = [index, runtime_encoder]
        aktivierungen.append(index)

    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: tuple(runtime),
        activate=activate,
    )
    operator.bind_status_graph(graph)
    try:
        assert len(aktivierungen) == 1
        assert json.loads((root / "current.json").read_text(encoding="utf-8")) == {
            "generation": previous
        }
        assert not (root / "generations" / current).exists()
        assert (root / "generations" / previous).is_dir()
    finally:
        operator.close()
        graph.close()


def test_staler_current_faellt_ohne_release_index_auf_aktuellen_previous_zurueck(
    tmp_path, monkeypatch
):
    graph, katalog, previous_index, _encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    current = "6" * 32
    previous = "7" * 32
    stale_current = EmbeddingIndex.build(graph, HashEncoder(dim=32), katalog=katalog)
    stale_current.graph_fingerprint = "0" * 64
    stale_current.save(root / "generations" / current)
    previous_index.save(root / "generations" / previous)
    (root / "current.json").write_text(
        json.dumps({"generation": current, "previous": previous}), encoding="utf-8"
    )
    runtime = [None, None]
    aktivierungen = []
    monkeypatch.setattr(
        embedding_operator_mod.importlib.util, "find_spec", lambda _name: object()
    )

    def activate(index, encoder):
        runtime[:] = [index, encoder]
        aktivierungen.append(index)

    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: tuple(runtime),
        activate=activate,
    )
    operator.bind_status_graph(graph)
    try:
        assert aktivierungen == [runtime[0]]
        assert runtime[0].graph_fingerprint == graph.fingerprint()
        assert json.loads((root / "current.json").read_text(encoding="utf-8")) == {
            "generation": previous
        }
        assert not (root / "generations" / current).exists()
    finally:
        operator.close()
        graph.close()


def test_unlesbares_generationen_volume_degradiert_statt_start_zu_blockieren(
    tmp_path, monkeypatch
):
    graph, katalog, release_index, encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    generations = root / "generations"
    original_iterdir = embedding_operator_mod.Path.iterdir

    def acl_fehler(path):
        if path == generations:
            raise PermissionError("acl")
        return original_iterdir(path)

    monkeypatch.setattr(embedding_operator_mod.Path, "iterdir", acl_fehler)
    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: (release_index, encoder),
        activate=lambda *_args: None,
    )
    operator.bind_status_graph(graph)
    try:
        status = operator.status()
        assert status["refresh_available"] is False
        assert status["unavailable_reason"] == (
            "Embedding-Overlay ist nicht beschreibbar"
        )
    finally:
        operator.close()
        graph.close()


def test_start_bereinigt_nur_eindeutige_crash_temporaries(tmp_path):
    graph, katalog, index, encoder = _ressourcen(tmp_path)
    root = tmp_path / "overlay"
    generations = root / "generations"
    abandoned = generations / (".tmp-" + "e" * 32)
    lookalike = generations / ".tmp-nicht-eindeutig"
    abandoned.mkdir(parents=True)
    lookalike.mkdir()
    (abandoned / "teil.npy").write_bytes(b"teilbuild")
    operator = EmbeddingOperator(
        graph_path=graph.path,
        katalog=katalog,
        state_dir=root,
        configured=True,
        device="cpu",
        snapshot=lambda: (index, encoder),
        activate=lambda *_args: None,
    )
    operator.bind_status_graph(graph)
    try:
        assert not abandoned.exists()
        assert lookalike.is_dir()
    finally:
        operator.close()
        graph.close()


def test_geplanter_skip_pflegt_check_aber_nicht_last_success(tmp_path):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    try:
        operator.trigger(force=True)
        operator.wait_for_job()
        last_success = operator.status()["job"]["last_success"]
        with operator._lock:
            operator._state["schedule"]["next_run_at"] = "2020-01-01T00:00:00Z"
        operator._scheduled_check()
        status = operator.status()
        assert status["job"]["state"] == "skipped"
        assert status["job"]["last_success"] == last_success
        assert status["schedule"]["last_check_at"] is not None
        assert status["schedule"]["next_run_at"] is not None
    finally:
        operator.close()
        graph.close()


def test_geplanter_skip_ueberschreibt_keinen_parallelen_force_job(
    tmp_path, monkeypatch
):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    status_pruefung = threading.Event()
    status_freigabe = threading.Event()
    build_gestartet = threading.Event()
    build_freigabe = threading.Event()
    original_index_status = operator._index_status

    def blockierender_status():
        status_pruefung.set()
        assert status_freigabe.wait(5)
        return original_index_status()

    def blockierender_build(_job_id):
        build_gestartet.set()
        assert build_freigabe.wait(5)

    monkeypatch.setattr(operator, "_index_status", blockierender_status)
    monkeypatch.setattr(operator, "_build_generation", blockierender_build)
    with operator._lock:
        operator._state["schedule"]["next_run_at"] = "2020-01-01T00:00:00Z"
    scheduler = threading.Thread(target=operator._scheduled_check)
    try:
        scheduler.start()
        assert status_pruefung.wait(5)
        manual = operator.trigger(force=True)
        assert build_gestartet.wait(5)
        status_freigabe.set()
        scheduler.join(5)
        assert not scheduler.is_alive()
        laufend = operator.status()["job"]
        assert laufend["id"] == manual["job_id"]
        assert laufend["state"] == "running"

        build_freigabe.set()
        operator.wait_for_job()
        fertig = operator.status()["job"]
        assert fertig["id"] == manual["job_id"]
        assert fertig["state"] == "succeeded"
    finally:
        status_freigabe.set()
        build_freigabe.set()
        scheduler.join(5)
        operator.close()
        graph.close()


def test_close_beendet_scheduler_thread(tmp_path):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    operator.start()
    thread = operator._scheduler_thread
    assert thread is not None and thread.is_alive()
    operator.close()
    assert not thread.is_alive()
    graph.close()


def test_scheduler_kappt_lange_intervalle_am_plattform_timeout(tmp_path, monkeypatch):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    beobachtet = []

    class RecordingEvent:
        def wait(self, timeout):
            beobachtet.append(timeout)
            if len(beobachtet) == 2:
                operator._stop.set()
            return False

        def clear(self):
            pass

        def set(self):
            pass

    try:
        with operator._lock:
            operator._state["schedule"].update(
                {
                    "enabled": True,
                    "interval_days": 365,
                    "next_run_at": (
                        datetime.now(timezone.utc) + timedelta(days=365)
                    ).isoformat().replace("+00:00", "Z"),
                }
            )
        monkeypatch.setattr(embedding_operator_mod.threading, "TIMEOUT_MAX", 60.0)
        vorher = operator.status()
        operator._schedule_wakeup = RecordingEvent()
        operator._scheduler_loop()
        assert beobachtet == [60.0, 60.0]
        nachher = operator.status()
        assert nachher["job"] == vorher["job"]
        assert nachher["schedule"] == vorher["schedule"]
    finally:
        operator.close()
        graph.close()


def test_thread_start_fehler_hinterlaesst_keinen_queued_job(tmp_path, monkeypatch):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)

    def start_fehler(_thread):
        raise RuntimeError("thread limit")

    monkeypatch.setattr(embedding_operator_mod.threading.Thread, "start", start_fehler)
    try:
        with pytest.raises(
            EmbeddingOperatorUnavailable,
            match="Embedding-Worker konnte nicht gestartet werden",
        ):
            operator.trigger(force=True)
        job = operator.status()["job"]
        assert job["state"] == "failed"
        assert job["completed"] is not None
        assert job["error"] == "Embedding-Worker konnte nicht gestartet werden"
        assert operator._job_thread is None
    finally:
        operator.close()
        graph.close()


def test_scheduler_start_fehler_hinterlaesst_keinen_join_wedge(tmp_path, monkeypatch):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)

    def start_fehler(_thread):
        raise RuntimeError("thread limit")

    monkeypatch.setattr(embedding_operator_mod.threading.Thread, "start", start_fehler)
    try:
        with pytest.raises(
            EmbeddingOperatorUnavailable,
            match="Embedding-Scheduler konnte nicht gestartet werden",
        ):
            operator.start()
        assert operator._scheduler_thread is None
    finally:
        operator.close()
        graph.close()


def test_pointer_fehler_bereinigt_unpublizierte_generation(tmp_path, monkeypatch):
    operator, graph, _runtime, _aktivierungen = _operator(tmp_path)
    original_atomic_json = embedding_operator_mod._atomic_json

    def pointer_fehler(path, obj):
        if path.name == "current.json":
            raise OSError("disk full")
        return original_atomic_json(path, obj)

    monkeypatch.setattr(embedding_operator_mod, "_atomic_json", pointer_fehler)
    try:
        operator.trigger(force=True)
        operator.wait_for_job()
        assert operator.status()["job"]["state"] == "failed"
        assert list(operator.generations.iterdir()) == []
        assert not operator.pointer_path.exists()
    finally:
        operator.close()
        graph.close()
