"""Operator-CLI: Pflichtbackup, Manifest-Trust, Health und Release-Switch."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from risk_layer import ops_cli as O
from risk_layer.berater import BeraterStore
from risk_layer.release_lifecycle import ReleaseManager, ReleaseRecord
from risk_layer.release_manifest import ArtifactDigest, ReleaseManifest
from risk_layer.sicherung import pruefe_sicherung


def _artefakt(bundle: Path, relativ: Path) -> ArtifactDigest:
    pfad = bundle / relativ
    daten = pfad.read_bytes()
    return ArtifactDigest(
        path=relativ.as_posix(),
        sha256=hashlib.sha256(daten).hexdigest(),
        size=len(daten),
    )


def _mini_wheel(
    pfad: Path,
    *,
    distribution: str,
    version: str,
    dateien: dict[str, str],
) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    dist = distribution.replace("-", "_")
    metadata = f"{dist}-{version}.dist-info"
    inhalte = {
        **dateien,
        f"{metadata}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n"
        ),
        f"{metadata}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    inhalte[f"{metadata}/RECORD"] = "".join(f"{name},,\n" for name in inhalte)
    with zipfile.ZipFile(pfad, "w") as archiv:
        for name, inhalt in inhalte.items():
            archiv.writestr(name, inhalt)


def _bundle(
    tmp_path: Path,
    release_id: str,
    git: str = "a" * 40,
    *,
    runtime: bool = True,
    host_tag: str | None = None,
) -> Path:
    bundle = tmp_path / release_id
    wheel_rel = Path("app") / "risk_layer-9.9.9-py3-none-any.whl"
    wheel = bundle / wheel_rel
    _mini_wheel(
        wheel,
        distribution="risk-layer",
        version="9.9.9",
        dateien={
            "risk_layer/__init__.py": "__version__ = '9.9.9'\n",
            "risk_layer/web.py": "# candidate import smoke\n",
        },
    )
    artifacts = [_artefakt(bundle, wheel_rel)]
    if runtime:
        tag = host_tag or O._runtime_platform_tag()
        dep_rel = Path("runtime") / "wheelhouse" / tag / "runtime_dep-1.0-py3-none-any.whl"
        dep = bundle / dep_rel
        _mini_wheel(
            dep,
            distribution="runtime-dep",
            version="1.0",
            dateien={"runtime_dep/__init__.py": "VALUE = 1\n"},
        )
        dep_hash = hashlib.sha256(dep.read_bytes()).hexdigest()
        lock_rel = Path("runtime") / "locks" / f"{O._runtime_platform_tag()}.txt"
        lock = bundle / lock_rel
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(
            f"runtime-dep==1.0 \\\n    --hash=sha256:{dep_hash}\n",
            encoding="utf-8",
        )
        artifacts.extend((_artefakt(bundle, lock_rel), _artefakt(bundle, dep_rel)))
    manifest = ReleaseManifest(
        release_id=release_id,
        git_sha=git,
        schema_version="1",
        api_version="v1",
        features=(),
        artifacts=tuple(artifacts),
    )
    manifest.write(bundle / "release.json")
    return bundle


def test_backup_cli_create_verify_restore_mit_persistenter_vorher_sicherung(
    tmp_path, capsys
):
    quelle = tmp_path / "berater"
    BeraterStore(quelle).definiere("Alt", "alter Stand", autor="a", datum="d")
    backups = tmp_path / "offhost"

    assert O.main([
        "backup", "create", "--berater", str(quelle), "--ziel", str(backups),
        "--zeitstempel", "2026-07-13T12:00:00Z",
    ]) == 0
    archiv = Path(capsys.readouterr().out.strip())
    assert archiv.is_file()
    assert O.main(["backup", "verify", str(archiv)]) == 0
    assert json.loads(capsys.readouterr().out)["legacy"] is False

    BeraterStore(quelle).definiere("Neu", "neuer Stand", autor="a", datum="d")
    assert O.main([
        "backup", "restore", str(archiv), "--berater", str(quelle), "--force",
    ]) == 0
    ausgabe = capsys.readouterr().out
    assert "wiederhergestellt" in ausgabe
    assert {eintrag.id for eintrag in BeraterStore(quelle).alle()} == {"alt"}
    vorher = sorted(backups.glob("*pre-restore.zip"))
    assert len(vorher) == 1
    info = pruefe_sicherung(vorher[0])
    assert info["counts"]["definitionen"] == 2


def test_backup_cli_verlangt_explizites_ziel():
    with pytest.raises(SystemExit):
        O.build_parser().parse_args(["backup", "create", "--berater", "x"])


def test_release_verify_braucht_trustentscheidung(tmp_path, capsys):
    bundle = _bundle(tmp_path, "v1")
    assert O.main(["release", "verify", str(bundle), "--trust-local-manifest"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["ok"] is True and info["release_id"] == "v1"

    assert O.main([
        "release", "verify", str(bundle), "--manifest-sha256", "0" * 64,
    ]) == 2
    assert "stimmt nicht" in capsys.readouterr().err


def test_release_install_pflichtbackup_current_previous_rollback(
    tmp_path, capsys, monkeypatch
):
    home = tmp_path / "runtime"
    berater = tmp_path / "berater"
    berater.mkdir()
    backups = tmp_path / "backups"
    erstes = _bundle(tmp_path, "v1", "1" * 40)
    zweites = _bundle(tmp_path, "v2", "2" * 40)
    fake_runtime = O._ReleaseRuntime(tmp_path / "fake-runtime", Path(sys.executable))
    monkeypatch.setattr(
        O,
        "_runtime_sicherstellen",
        lambda runtime_home, record, timeout: fake_runtime,
    )
    monkeypatch.setattr(
        O,
        "_runtime_pruefen",
        lambda runtime_home, record: fake_runtime,
    )

    basis = [
        "--home", str(home), "--berater", str(berater),
        "--backup-ziel", str(backups), "--trust-local-manifest",
        "--no-activate", "--no-health-check",
    ]
    assert O.main(["release", "install", str(erstes), *basis]) == 0
    erste_info = json.loads(capsys.readouterr().out)
    assert erste_info["state"]["current"]["release_id"] == "v1"
    assert Path(erste_info["backup"]).is_file()

    assert O.main(["release", "install", str(zweites), *basis]) == 0
    zweite_info = json.loads(capsys.readouterr().out)
    assert zweite_info["state"]["current"]["release_id"] == "v2"
    assert zweite_info["state"]["previous"]["release_id"] == "v1"

    assert O.main([
        "release", "rollback", "--home", str(home),
        "--no-activate", "--no-health-check",
    ]) == 0
    rollback = json.loads(capsys.readouterr().out)
    assert rollback["state"]["current"]["release_id"] == "v1"
    assert rollback["state"]["previous"]["release_id"] == "v2"
    assert len(list(backups.glob("festwissen_*.zip"))) == 2


def test_candidate_hook_ist_json_array_ohne_shell():
    assert O._parse_command('["python","-V"]', "x") == ("python", "-V")
    with pytest.raises(O.OpsFehler):
        O._parse_command("python -V", "x")


def test_doctor_prueft_release_und_sollschichten(monkeypatch):
    antworten = {
        "/v1/health": {"ok": True, "engineVersion": "1.3.2"},
        "/v1/status": {
            "ok": True,
            "engine_status": {
                "release": {
                    "release_id": "v2",
                    "git_sha": "a" * 40,
                    "manifest_sha256": "b" * 64,
                },
                "schichten": {"matcher": {"aktiv": True}},
            },
        },
    }
    monkeypatch.setattr(O, "_hole_json", lambda basis, pfad, token, timeout: antworten[pfad])
    result = O._doctor(
        url="http://127.0.0.1:8000",
        token_datei=None,
        timeout=1,
        erwartet_release="v2",
        erwartet_git_sha="a" * 40,
        erwartet_manifest_sha256="b" * 64,
        schichten=["matcher"],
    )
    assert result["ok"] is True


def test_doctor_blockt_abweichenden_manifest_hash(monkeypatch):
    antworten = {
        "/v1/health": {"ok": True},
        "/v1/status": {
            "ok": True,
            "engine_status": {
                "release": {
                    "release_id": "v2",
                    "git_sha": "a" * 40,
                    "manifest_sha256": "c" * 64,
                },
                "schichten": {},
            },
        },
    }
    monkeypatch.setattr(O, "_hole_json", lambda basis, pfad, token, timeout: antworten[pfad])
    with pytest.raises(O.OpsFehler, match="manifest_sha256"):
        O._doctor(
            url="http://127.0.0.1:8000",
            token_datei=None,
            timeout=0.01,
            erwartet_release="v2",
            erwartet_git_sha="a" * 40,
            erwartet_manifest_sha256="b" * 64,
        )


def test_doctor_sendet_token_nicht_unverschluesselt_ins_netz(monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", "x" * 32)
    with pytest.raises(O.OpsFehler, match="HTTPS"):
        O._doctor(url="http://example.com", token_datei=None, timeout=0.1)


def test_release_env_enthaelt_nur_manifestierte_identitaet(tmp_path):
    bundle = _bundle(tmp_path, "v1")
    manifest = ReleaseManifest.load(bundle / "release.json")
    env = O._release_env(O.ReleaseRecord(bundle, manifest), "candidate")
    assert env["RISK_LAYER_RELEASE_ID"] == "v1"
    assert env["RISK_LAYER_RELEASE_GIT_SHA"] == "a" * 40
    assert env["RISK_LAYER_RELEASE_MANIFEST_SHA256"] == manifest.fingerprint()


def test_install_verlangt_manifestierten_lock_und_abi_wheelhouse(tmp_path):
    ohne_runtime = _bundle(tmp_path, "ohne-runtime", runtime=False)
    manifest = ReleaseManifest.load(ohne_runtime / "release.json")
    with pytest.raises(O.OpsFehler, match="Runtime-Lock"):
        O._runtime_quellen(ReleaseRecord(ohne_runtime, manifest))

    falscher_tag = "win_amd64-cp999" if O._runtime_platform_tag() != "win_amd64-cp999" else "x-cp1"
    falsches_bundle = _bundle(tmp_path, "falsche-abi", host_tag=falscher_tag)
    manifest = ReleaseManifest.load(falsches_bundle / "release.json")
    with pytest.raises(O.OpsFehler, match="kein Wheel fuer den Host"):
        O._runtime_quellen(ReleaseRecord(falsches_bundle, manifest))


def test_runtime_lock_blockt_externe_quellen(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "boese-paket @ https://example.invalid/boese.whl \\\n"
        f"    --hash=sha256:{'a' * 64}\n",
        encoding="utf-8",
    )
    with pytest.raises(O.OpsFehler, match="externen Quellen"):
        O._runtime_lock_pruefen(lock)


def test_release_runtime_offline_atomar_und_service_nutzt_exakten_interpreter(
    tmp_path, monkeypatch
):
    bundle = _bundle(tmp_path, "venv-release", "3" * 40)
    manifest = ReleaseManifest.load(bundle / "release.json")
    record = ReleaseRecord(bundle, manifest)
    home = tmp_path / "home"

    runtime = O._runtime_sicherstellen(home, record, 120.0)
    assert runtime.created is True
    assert runtime.path.name == f"venv-release-{manifest.fingerprint()[:16]}"
    assert (runtime.path / O._RUNTIME_MARKER).is_file()
    pruefung = subprocess.run(
        [
            str(runtime.python),
            "-I",
            "-c",
            "import risk_layer, runtime_dep; print(risk_layer.__version__, runtime_dep.VALUE)",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert pruefung.returncode == 0, pruefung.stderr
    assert pruefung.stdout.strip() == "9.9.9 1"
    wiederverwendet = O._runtime_sicherstellen(home, record, 30.0)
    assert wiederverwendet.path == runtime.path
    assert wiederverwendet.created is False

    manager = ReleaseManager(home)
    manager.apply(
        bundle,
        candidate_check=lambda staged: bool(O._runtime_sicherstellen(home, staged, 30.0)),
    )
    aufruf: dict[str, object] = {}

    def fake_execve(path, argv, env):
        aufruf.update(path=path, argv=argv, env=env)
        raise RuntimeError("exec abgefangen")

    monkeypatch.setattr(O.os, "chdir", lambda pfad: None)
    monkeypatch.setattr(O.os, "execve", fake_execve)
    args = O.build_parser().parse_args(
        [
            "service", "run", "--home", str(home), "--berater",
            str(tmp_path / "berater"),
        ]
    )
    with pytest.raises(RuntimeError, match="exec abgefangen"):
        O._cmd_service_run(args)
    assert aufruf["path"] == str(runtime.python)
    assert aufruf["argv"][0] == str(runtime.python)  # type: ignore[index]
    assert aufruf["env"]["RISK_LAYER_RUNTIME_PYTHON"] == str(runtime.python)  # type: ignore[index]


def test_runtime_marker_tampering_wird_fail_closed_erkannt(tmp_path):
    bundle = _bundle(tmp_path, "marker-release", "4" * 40)
    manifest = ReleaseManifest.load(bundle / "release.json")
    record = ReleaseRecord(bundle, manifest)
    home = tmp_path / "home"
    runtime = O._runtime_sicherstellen(home, record, 120.0)
    marker = runtime.path / O._RUNTIME_MARKER
    dokument = json.loads(marker.read_text(encoding="utf-8"))
    dokument["release_id"] = "fremd"
    marker.write_text(json.dumps(dokument), encoding="utf-8")
    with pytest.raises(O.OpsFehler, match="passt nicht"):
        O._runtime_pruefen(home, record)


def test_rollback_behaelt_beide_release_runtimes(tmp_path, capsys):
    home = tmp_path / "home"
    manager = ReleaseManager(home)
    bundle_v1 = _bundle(tmp_path, "runtime-v1", "5" * 40)
    bundle_v2 = _bundle(tmp_path, "runtime-v2", "6" * 40)
    manager.apply(bundle_v1)
    manager.apply(bundle_v2)

    def fake_runtime(record: ReleaseRecord) -> Path:
        fingerprint = record.manifest.fingerprint()
        ziel = O._runtime_root(home, erstellen=True) / O._runtime_name(
            record.manifest.release_id, fingerprint
        )
        python = ziel / O._runtime_python_relativ()
        python.parent.mkdir(parents=True)
        shutil.copyfile(sys.executable, python)
        python_hash, python_size = O._datei_hash_und_groesse(python, "Test-Python")
        (ziel / O._RUNTIME_MARKER).write_text(
            json.dumps(O._runtime_marker_payload(record, python_hash, python_size)),
            encoding="utf-8",
        )
        return ziel

    state = manager.read_state()
    current = manager.verify(
        state.current, expected_manifest_sha256=state.current_manifest_sha256
    )
    previous = manager.verify(
        state.previous, expected_manifest_sha256=state.previous_manifest_sha256
    )
    runtime_v2 = fake_runtime(current)
    runtime_v1 = fake_runtime(previous)

    assert O.main(
        [
            "release", "rollback", "--home", str(home),
            "--no-activate", "--no-health-check",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["state"]["current"]["release_id"] == "runtime-v1"
    assert runtime_v1.is_dir()
    assert runtime_v2.is_dir()


def test_backup_create_wendet_explizite_retention_nach_erfolg_an(tmp_path, capsys):
    berater = tmp_path / "berater"
    BeraterStore(berater).definiere("X", "Y", autor="a", datum="d")
    ziel = tmp_path / "backups"
    for index in range(2):
        assert O.main([
            "backup", "create", "--berater", str(berater), "--ziel", str(ziel),
            "--zeitstempel", f"alt-{index}",
        ]) == 0
        capsys.readouterr()
    for pfad in ziel.glob("festwissen_*.zip"):
        os.utime(pfad, (1, 1))

    assert O.main([
        "backup", "create", "--berater", str(berater), "--ziel", str(ziel),
        "--zeitstempel", "neu", "--retention-keep-last", "1",
        "--retention-keep-days", "1", "--json",
    ]) == 0
    ergebnis = json.loads(capsys.readouterr().out)
    assert len(ergebnis["retention"]["deleted"]) == 2
    assert [pfad.name for pfad in ziel.glob("festwissen_*.zip")] == [
        "festwissen_neu.zip"
    ]


def test_release_prune_cli_entfernt_nur_inaktive_generationen_und_runtimes(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    manager = ReleaseManager(home)
    for index in range(1, 5):
        bundle = _bundle(tmp_path, f"prune-v{index}", f"{index}" * 40)
        record = manager.stage(bundle)
        os.utime(record.path, (100 + index, 100 + index))
    manager.promote("prune-v3")
    manager.promote("prune-v4")
    runtimes: list[str] = []
    monkeypatch.setattr(
        O,
        "_runtime_entfernen",
        lambda home, release_id, manifest_sha256: runtimes.append(release_id) or True,
    )

    assert O.main([
        "release", "prune", "--home", str(home), "--keep", "2", "--apply",
    ]) == 0
    ergebnis = json.loads(capsys.readouterr().out)
    assert ergebnis["deleted"] == ["prune-v2", "prune-v1"]
    assert runtimes == ["prune-v2", "prune-v1"]
    assert manager.current().manifest.release_id == "prune-v4"
    assert manager.previous().manifest.release_id == "prune-v3"
