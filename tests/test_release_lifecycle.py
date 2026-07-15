from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import risk_layer.release_lifecycle as lifecycle
from risk_layer.release_lifecycle import (
    ActivationError,
    IntegrityError,
    ReleaseCheckError,
    ReleaseLockError,
    ReleaseManager,
    ReleaseState,
    ReleaseStateError,
    SchemaMismatchError,
    UpdateLock,
    verify_bundle,
)
from risk_layer.release_manifest import ArtifactDigest, ReleaseManifest


GIT_A = "a" * 40
GIT_B = "b" * 40


def _bundle(
    root: Path,
    release_id: str,
    *,
    git_sha: str = GIT_A,
    schema_version: str = "1",
    api_version: str = "v1",
    files: dict[str, bytes] | None = None,
    executable: set[str] | None = None,
) -> tuple[Path, ReleaseManifest]:
    files = files or {
        "app/risk_layer.whl": f"wheel-{release_id}".encode(),
        "data/festwissen.bin": f"wissen-{release_id}".encode(),
    }
    executable = executable or set()
    artifacts = []
    root.mkdir(parents=True)
    for relative, content in files.items():
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append(ArtifactDigest(
            path=relative,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
            executable=relative in executable,
        ))
    manifest = ReleaseManifest(
        release_id=release_id,
        git_sha=git_sha,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.path)),
        schema_version=schema_version,
        api_version=api_version,
        features=("festwissen",),
    )
    manifest.write(root / "release.json")
    return root, manifest


def test_verify_bundle_prueft_strikt_ohne_installation(tmp_path):
    bundle, expected = _bundle(tmp_path / "bundle", "r1")
    assert verify_bundle(bundle) == expected
    assert sorted(path.relative_to(tmp_path).as_posix()
                  for path in tmp_path.rglob("*")) == [
        "bundle", "bundle/app", "bundle/app/risk_layer.whl",
        "bundle/data", "bundle/data/festwissen.bin", "bundle/release.json",
    ]


def test_verify_bundle_blockt_hashfehler_extra_datei_und_extra_verzeichnis(tmp_path):
    bundle, _ = _bundle(tmp_path / "hash", "r1")
    (bundle / "app" / "risk_layer.whl").write_bytes(b"manipuliert")
    with pytest.raises(IntegrityError, match="Groesse|SHA-256"):
        verify_bundle(bundle)

    bundle, _ = _bundle(tmp_path / "extra-file", "r2")
    (bundle / "nicht-gelistet.txt").write_text("nein", encoding="utf-8")
    with pytest.raises(IntegrityError, match="Dateiinventar"):
        verify_bundle(bundle)

    bundle, _ = _bundle(tmp_path / "extra-dir", "r3")
    (bundle / "leer").mkdir()
    with pytest.raises(IntegrityError, match="Verzeichnisinventar"):
        verify_bundle(bundle)


def test_verify_bundle_begrenzt_bauminventar_vor_hashpruefung(tmp_path, monkeypatch):
    bundle, _ = _bundle(tmp_path / "bundle", "r1")
    monkeypatch.setattr(lifecycle, "MAX_TREE_ENTRIES", 2)
    with pytest.raises(IntegrityError, match="Eintraege"):
        verify_bundle(bundle)


def test_verify_bundle_blockt_symlink_artefakt(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"ausserhalb")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    link = bundle / "payload.bin"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks stehen auf diesem Testsystem nicht zur Verfuegung")
    manifest = ReleaseManifest(
        release_id="r1",
        git_sha=GIT_A,
        artifacts=(ArtifactDigest(
            "payload.bin", hashlib.sha256(b"ausserhalb").hexdigest(), 9
        ),),
        schema_version="1",
        api_version="v1",
    )
    manifest.write(bundle / "release.json")
    with pytest.raises(IntegrityError, match="Symlink|Reparse"):
        verify_bundle(bundle)


def test_manager_blockt_release_state_und_lock_symlinks(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    outside_bundle, _ = _bundle(tmp_path / "outside-release", "r1")
    release_link = manager.releases_dir / "r1"
    try:
        release_link.symlink_to(outside_bundle, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks stehen auf diesem Testsystem nicht zur Verfuegung")
    with pytest.raises(IntegrityError, match="unsicher"):
        manager.verify("r1")
    release_link.unlink()

    outside_state = tmp_path / "outside-state.json"
    outside_state.write_text("{}", encoding="utf-8")
    manager.state_path.symlink_to(outside_state)
    with pytest.raises(ReleaseStateError, match="Symlink|Reparse"):
        manager.read_state()
    manager.state_path.unlink()

    outside_lock = tmp_path / "outside.lock"
    outside_lock.write_bytes(b"x")
    manager.lock_path.symlink_to(outside_lock)
    with pytest.raises(ReleaseLockError, match="Symlink"):
        manager.stage(outside_bundle)
    assert outside_lock.read_bytes() == b"x"


def test_manager_blockt_symlink_in_elternkette(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks stehen auf diesem Testsystem nicht zur Verfuegung")
    with pytest.raises(IntegrityError, match="Symlink|Reparse"):
        ReleaseManager(linked_parent / "install")


def test_update_lock_blockt_hardlink_und_zu_weite_rechte(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"")
    os.chmod(outside, 0o600)
    try:
        os.link(outside, manager.lock_path)
    except OSError:
        pytest.skip("Hardlinks stehen auf diesem Testsystem nicht zur Verfuegung")
    with pytest.raises(ReleaseLockError, match="Hardlinks"):
        with UpdateLock(manager.lock_path):
            pass
    manager.lock_path.unlink()

    if os.name != "nt":
        manager.lock_path.write_bytes(b"")
        os.chmod(manager.lock_path, 0o666)
        with pytest.raises(ReleaseLockError, match="Rechte"):
            with UpdateLock(manager.lock_path):
                pass


def test_lock_identitaetspruefung_erkennt_austausch_nach_open(tmp_path):
    if os.name == "nt":
        pytest.skip("Windows blockiert den Austausch einer offen gehaltenen Datei")
    lock_path = tmp_path / "update.lock"
    lock_path.write_bytes(b"")
    os.chmod(lock_path, 0o600)
    descriptor = os.open(lock_path, os.O_RDWR)
    replacement = tmp_path / "replacement.lock"
    replacement.write_bytes(b"")
    os.chmod(replacement, 0o600)
    os.replace(replacement, lock_path)
    try:
        with pytest.raises(ReleaseLockError, match="ausgetauscht"):
            lifecycle._validate_lock_identity(descriptor, lock_path)
    finally:
        os.close(descriptor)


def test_stage_kopiert_verifiziert_und_ist_idempotent(tmp_path):
    bundle, manifest = _bundle(tmp_path / "bundle", "r1")
    manager = ReleaseManager(tmp_path / "install")

    record = manager.stage(bundle)
    assert record.manifest == manifest
    assert record.path == manager.releases_dir / "r1"
    assert record.path != bundle
    assert manager.verify("r1") == record
    assert manager.stage(bundle) == record
    assert manager.read_state() == ReleaseState()


def test_stage_setzt_executable_modus_aus_manifest(tmp_path):
    bundle, _ = _bundle(
        tmp_path / "bundle", "r1",
        files={"bin/engine": b"binary"}, executable={"bin/engine"},
    )
    record = ReleaseManager(tmp_path / "install").stage(bundle)
    if os.name != "nt":
        assert record.path.joinpath("bin", "engine").stat().st_mode & 0o111


def test_stage_verweigert_gleiche_id_mit_anderem_manifest(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    first, _ = _bundle(tmp_path / "first", "r1", files={"app.bin": b"eins"})
    second, _ = _bundle(
        tmp_path / "second", "r1", git_sha=GIT_B, files={"app.bin": b"zwei"}
    )
    manager.stage(first)
    with pytest.raises(IntegrityError, match="anderem Manifest"):
        manager.stage(second)
    assert manager.verify("r1").manifest.git_sha == GIT_A


def test_stage_verweigert_release_id_casefold_kollision(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    manager.stage(_bundle(tmp_path / "first", "release-a")[0])
    second, _ = _bundle(tmp_path / "second", "RELEASE-A", git_sha=GIT_B)
    with pytest.raises(IntegrityError, match="Grossschreibung"):
        manager.stage(second)


def test_stage_fehler_hinterlaesst_weder_release_noch_stagingrest(tmp_path):
    bundle, _ = _bundle(tmp_path / "bundle", "r1")
    (bundle / "data" / "festwissen.bin").write_bytes(b"kaputt")
    manager = ReleaseManager(tmp_path / "install")
    with pytest.raises(IntegrityError):
        manager.stage(bundle)
    assert not (manager.releases_dir / "r1").exists()
    assert not list(manager.releases_dir.glob(".staging-*"))


def test_verify_erkennt_nachtraegliche_manipulation_und_extras(tmp_path):
    bundle, _ = _bundle(tmp_path / "bundle", "r1")
    manager = ReleaseManager(tmp_path / "install")
    record = manager.stage(bundle)
    os.chmod(record.path / "app" / "risk_layer.whl", 0o644)
    (record.path / "app" / "risk_layer.whl").write_bytes(b"tamper")
    with pytest.raises(IntegrityError, match="Groesse|SHA-256"):
        manager.verify("r1")

    bundle2, _ = _bundle(tmp_path / "bundle2", "r2")
    record2 = manager.stage(bundle2)
    os.chmod(record2.path, 0o755)
    (record2.path / "extra").write_text("tamper", encoding="utf-8")
    with pytest.raises(IntegrityError, match="Dateiinventar"):
        manager.verify("r2")


def test_current_bindet_beim_switch_vertrauten_manifest_fingerprint(tmp_path):
    bundle, _ = _bundle(tmp_path / "bundle", "r1")
    manager = ReleaseManager(tmp_path / "install")
    record = manager.stage(bundle)
    state = manager.promote("r1")
    assert state.current_manifest_sha256 == record.manifest.fingerprint()

    wheel = record.path / "app" / "risk_layer.whl"
    os.chmod(record.path, 0o755)
    os.chmod(wheel, 0o644)
    os.chmod(record.path / "release.json", 0o644)
    wheel.write_bytes(b"anderes-selbstkonsistentes-wheel")
    artifacts = tuple(
        ArtifactDigest(
            path=item.path,
            sha256=(hashlib.sha256(wheel.read_bytes()).hexdigest()
                    if item.path == "app/risk_layer.whl" else item.sha256),
            size=(wheel.stat().st_size
                  if item.path == "app/risk_layer.whl" else item.size),
            executable=item.executable,
        )
        for item in record.manifest.artifacts
    )
    ReleaseManifest(
        release_id=record.manifest.release_id,
        git_sha=record.manifest.git_sha,
        schema_version=record.manifest.schema_version,
        api_version=record.manifest.api_version,
        features=record.manifest.features,
        artifacts=artifacts,
    ).write(record.path / "release.json")

    with pytest.raises(IntegrityError, match="Vertrauens-State"):
        manager.current()


def test_apply_bindet_vorher_vertrauten_bundle_hash_gegen_toctou(tmp_path):
    first, first_manifest = _bundle(tmp_path / "first", "r1")
    second, _ = _bundle(tmp_path / "second", "r2", git_sha=GIT_B)
    manager = ReleaseManager(tmp_path / "install")
    with pytest.raises(IntegrityError, match="Trust-Pruefung"):
        manager.apply(
            second,
            expected_manifest_sha256=first_manifest.fingerprint(),
        )
    assert manager.read_state() == ReleaseState()
    assert not (manager.releases_dir / "r2").exists()


def test_promote_backup_health_state_und_rollback(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    first = manager.stage(_bundle(tmp_path / "first", "r1")[0])
    second = manager.stage(_bundle(tmp_path / "second", "r2", git_sha=GIT_B)[0])
    events = []

    state1 = manager.promote("r1", activation=lambda record: events.append(
        ("activate", record.manifest.release_id if record else None)
    ))
    assert state1.generation == 1
    assert (state1.current, state1.previous) == ("r1", None)

    def candidate(record):
        events.append(("candidate", record.manifest.release_id))
        return True

    def backup(state, record):
        events.append(("backup", state.current, record.manifest.release_id))
        return None

    def activate(record):
        assert record is not None
        events.append(("activate", record.manifest.release_id))
        assert manager.read_state().current == record.manifest.release_id
        return True

    state2 = manager.promote(
        "r2", backup=backup, candidate_check=candidate, activation=activate
    )
    assert state2.generation == 2
    assert (state2.current, state2.previous) == ("r2", "r1")
    assert manager.current() == second
    assert manager.previous() == first
    assert events[-3:] == [
        ("candidate", "r2"), ("backup", "r1", "r2"), ("activate", "r2")
    ]

    state3 = manager.rollback(activation=activate)
    assert state3.generation == 3
    assert (state3.current, state3.previous) == ("r1", "r2")
    assert events[-1] == ("activate", "r1")


@pytest.mark.parametrize("hook_name", ["candidate", "backup"])
def test_pre_switch_fehler_laesst_state_unveraendert(tmp_path, hook_name):
    manager = ReleaseManager(tmp_path / "install")
    manager.stage(_bundle(tmp_path / "first", "r1")[0])
    manager.stage(_bundle(tmp_path / "second", "r2", git_sha=GIT_B)[0])
    before = manager.promote("r1")

    kwargs = ({"candidate_check": lambda record: False}
              if hook_name == "candidate"
              else {"backup": lambda state, record: False})
    with pytest.raises(ReleaseCheckError):
        manager.promote("r2", **kwargs)
    assert manager.read_state() == before


def test_schemawechsel_wird_ohne_migration_fail_closed_abgelehnt(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    manager.stage(_bundle(tmp_path / "first", "r1", schema_version="1")[0])
    manager.stage(_bundle(tmp_path / "second", "r2", schema_version="2")[0])
    before = manager.promote("r1")
    called = []
    with pytest.raises(SchemaMismatchError, match="ohne Migration"):
        manager.promote("r2", backup=lambda *_: called.append("backup"))
    assert not called
    assert manager.read_state() == before


def test_aktivierungsfehler_kompensiert_state_und_reaktiviert_alt(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    manager.stage(_bundle(tmp_path / "first", "r1")[0])
    manager.stage(_bundle(tmp_path / "second", "r2", git_sha=GIT_B)[0])
    manager.promote("r1")
    calls = []

    def activation(record):
        release_id = record.manifest.release_id if record else None
        calls.append(release_id)
        return release_id != "r2"

    with pytest.raises(ActivationError) as caught:
        manager.promote("r2", activation=activation)
    assert caught.value.rolled_back is True
    state = manager.read_state()
    assert (state.generation, state.current, state.previous) == (3, "r1", None)
    assert calls == ["r2", "r1"]


def test_aktivierungsfehler_meldet_fehlgeschlagene_externe_reaktivierung(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    manager.stage(_bundle(tmp_path / "first", "r1")[0])
    manager.stage(_bundle(tmp_path / "second", "r2", git_sha=GIT_B)[0])
    manager.promote("r1")

    with pytest.raises(ActivationError) as caught:
        manager.promote("r2", activation=lambda record: False)
    assert caught.value.rolled_back is False
    assert caught.value.recovery_error is not None
    assert manager.read_state().current == "r1"


def test_apply_haelt_stage_und_promote_unter_einem_lock(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    bundle, _ = _bundle(tmp_path / "bundle", "r1")
    events = []
    state = manager.apply(
        bundle,
        backup=lambda old, record: events.append((old.current, record.manifest.release_id)),
        candidate_check=lambda record: events.append(record.manifest.release_id),
        activation=lambda record: events.append(record.manifest.release_id if record else None),
    )
    assert state.current == "r1"
    assert events == ["r1", (None, "r1"), "r1"]


def test_erstaktivierungsfehler_ruft_deaktivierung_mit_none_auf(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    bundle, _ = _bundle(tmp_path / "bundle", "r1")
    calls = []

    def activation(record):
        calls.append(record.manifest.release_id if record else None)
        return record is None

    with pytest.raises(ActivationError) as caught:
        manager.apply(bundle, activation=activation)
    assert caught.value.rolled_back is True
    assert calls == ["r1", None]
    assert manager.read_state().current is None


def test_update_lock_blockt_konkurrierende_mutation(tmp_path):
    manager = ReleaseManager(tmp_path / "install", lock_timeout=0.02)
    bundle, _ = _bundle(tmp_path / "bundle", "r1")
    with UpdateLock(manager.lock_path):
        with pytest.raises(ReleaseLockError, match="belegt"):
            manager.stage(bundle)
    assert manager.stage(bundle).manifest.release_id == "r1"


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
def test_update_lock_timeout_ist_endlich_nichtnegativ_und_begrenzt(
    tmp_path: Path,
    value: float,
) -> None:
    path = tmp_path / "update.lock"
    with pytest.raises(ValueError, match="timeout"):
        UpdateLock(path, timeout=value)
    assert not path.exists()


def test_update_lock_timeout_null_bleibt_nonblocking(tmp_path: Path) -> None:
    path = tmp_path / "update.lock"
    with UpdateLock(path, timeout=0):
        with pytest.raises(ReleaseLockError, match="belegt"):
            with UpdateLock(path, timeout=0):
                pass


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf"), 61])
def test_update_lock_pollintervall_ist_endlich_positiv_und_begrenzt(
    tmp_path: Path,
    value: float,
) -> None:
    path = tmp_path / "update.lock"
    with pytest.raises(ValueError, match="poll_interval"):
        UpdateLock(path, poll_interval=value)
    assert not path.exists()


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
def test_release_manager_lock_timeout_wird_vor_mutation_validiert(
    tmp_path: Path,
    value: float,
) -> None:
    home = tmp_path / "install"
    with pytest.raises(ValueError, match="lock_timeout"):
        ReleaseManager(home, lock_timeout=value)
    assert not home.exists()


def test_release_manager_erlaubt_nonblocking_lock_timeout_null(tmp_path: Path) -> None:
    manager = ReleaseManager(tmp_path / "install", lock_timeout=0)
    assert manager.lock_timeout == 0


def test_state_blockt_korruptes_json_doppelte_keys_und_traversal(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    manager.state_path.write_text(
        '{"format_version":2,"generation":1,"generation":2,'
        '"current":null,"current_manifest_sha256":null,'
        '"previous":null,"previous_manifest_sha256":null,"updated_at":null}',
        encoding="utf-8",
    )
    with pytest.raises(ReleaseStateError, match="Doppelter"):
        manager.read_state()

    manager.state_path.write_text(json.dumps({
        "format_version": 2, "generation": 1, "current": "../escape",
        "current_manifest_sha256": "a" * 64,
        "previous": None, "previous_manifest_sha256": None, "updated_at": None,
    }), encoding="utf-8")
    with pytest.raises(ReleaseStateError, match="current"):
        manager.read_state()

    manager.state_path.write_text(json.dumps({
        "format_version": 2, "generation": 2, "current": "Release-A",
        "current_manifest_sha256": "a" * 64,
        "previous": "release-a", "previous_manifest_sha256": "b" * 64,
        "updated_at": None,
    }), encoding="utf-8")
    with pytest.raises(ReleaseStateError, match="identisch"):
        manager.read_state()


def test_state_json_recursionerror_wird_normalisiert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ReleaseManager(tmp_path / "install")
    manager.state_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        lifecycle.json,
        "loads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RecursionError("zu tief")),
    )
    with pytest.raises(ReleaseStateError, match="gueltiges UTF-8-JSON"):
        manager.read_state()


def test_atomarer_state_write_laesst_bei_replace_fehler_alten_state_intakt(
        tmp_path, monkeypatch):
    manager = ReleaseManager(tmp_path / "install")
    manager.stage(_bundle(tmp_path / "bundle", "r1")[0])
    old = manager.promote("r1")
    old_bytes = manager.state_path.read_bytes()

    def fail_replace(source, target):
        raise OSError("simulierter Abbruch vor atomarem Switch")

    monkeypatch.setattr(lifecycle.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulierter Abbruch"):
        lifecycle._write_state_file(
            manager.state_path,
            ReleaseState(generation=old.generation + 1, current=None),
        )
    assert manager.state_path.read_bytes() == old_bytes
    assert not list(manager.home.glob(".state.json.*.tmp"))


def test_rollback_ohne_previous_und_ungueltige_id_fallen_laut(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    with pytest.raises(ReleaseStateError, match="previous"):
        manager.rollback()
    with pytest.raises(ReleaseStateError, match="Ungueltige Release-ID"):
        manager.verify("../escape")


def test_release_prune_ist_dry_run_und_schuetzt_current_previous(tmp_path):
    manager = ReleaseManager(tmp_path / "install")
    for index in range(1, 6):
        record = manager.stage(
            _bundle(
                tmp_path / f"bundle-{index}",
                f"r{index}",
                git_sha=f"{index}" * 40,
            )[0]
        )
        os.utime(record.path, (1_000_000 + index, 1_000_000 + index))
    manager.promote("r4")
    manager.promote("r5")

    plan = manager.prune(keep=2, apply=False)
    assert plan["candidates"] == ["r3", "r2", "r1"]
    assert all((manager.releases_dir / f"r{i}").exists() for i in range(1, 6))

    result = manager.prune(keep=2, apply=True)
    assert result["deleted"] == ["r3", "r2", "r1"]
    assert manager.current().manifest.release_id == "r5"
    assert manager.previous().manifest.release_id == "r4"
