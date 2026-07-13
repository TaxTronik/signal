"""Transaktions- und Hardening-Tests fuer das Compose-Deployment-Werkzeug."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pytest

from risk_layer.release_manifest import ArtifactDigest, ReleaseManifest
import tools.create_container_release_manifest as C
import tools.deploy_container_release as D
import tools.verify_container_release_manifest as V


ROOT = Path(__file__).resolve().parents[1]


def _release(
    root: Path,
    release_id: str,
    git_sha: str,
    *,
    schema_version: str = "1",
    api_version: str = "1.3.2",
    features: tuple[str, ...] = (),
) -> tuple[Path, ReleaseManifest]:
    release = root / release_id
    artifact = release / "catalog" / "begriffe.yaml"
    artifact.parent.mkdir(parents=True)
    payload = f"release: {release_id}\n".encode()
    artifact.write_bytes(payload)
    manifest = ReleaseManifest(
        release_id=release_id,
        git_sha=git_sha,
        schema_version=schema_version,
        api_version=api_version,
        features=features,
        artifacts=(
            ArtifactDigest(
                path="catalog/begriffe.yaml",
                sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
                executable=False,
            ),
        ),
    )
    manifest.write(release / "release.json")
    return release, manifest


def _runtime_files(root: Path) -> tuple[Path, Path]:
    compose = root / "docker-compose.yml"
    compose.write_text("services:\n  engine:\n    image: example.invalid/engine\n", encoding="utf-8")
    secrets = root / "secrets.env"
    secrets.write_text(f"RISK_LAYER_TOKEN={'s' * 40}\nRISK_LAYER_PORT=8000\n", encoding="utf-8")
    if os.name != "nt":
        secrets.chmod(0o600)
    return compose, secrets


def _container_manifest(
    root: Path,
    release: ReleaseManifest,
    compose: Path,
    *,
    digest_character: str = "1",
) -> tuple[Path, str]:
    document = {
        "schema": D.CONTAINER_SCHEMA,
        "release_id": release.release_id,
        "git_sha": release.git_sha,
        "created_at": "2026-07-13T12:00:00Z",
        "engine": {
            "repository": "registry.example/risk-layer-engine",
            "digest": f"sha256:{digest_character * 64}",
            "image": (
                "registry.example/risk-layer-engine@sha256:"
                f"{digest_character * 64}"
            ),
        },
        "festwissen": {
            "artifact": f"festwissen-{release.release_id}.tar.gz",
            "sha256": "2" * 64,
            "manifest": f"festwissen-{release.release_id}.json",
            "manifest_sha256": "3" * 64,
            "embedding": False,
        },
        "native_bundle": {
            "artifact": f"native-{release.release_id}.tar.gz",
            "sha256": "4" * 64,
            "manifest_sha256": release.fingerprint(),
        },
        "compose": {
            "artifact": compose.name,
            "sha256": hashlib.sha256(compose.read_bytes()).hexdigest(),
        },
        "bootstrap": {
            "release_tools": {"artifact": "risk-layer-release-tools.pyz", "sha256": "5" * 64},
            "verify_container": {"artifact": "verify-container.py", "sha256": "6" * 64},
            "extract_native": {"artifact": "extract-native.py", "sha256": "7" * 64},
            "deploy_container": {"artifact": "deploy-container.py", "sha256": "8" * 64},
            "install_release_service": {
                "artifact": "install-release-service.sh",
                "sha256": "9" * 64,
            },
            "release_launcher": {"artifact": "release-launcher.sh", "sha256": "a" * 64},
            "windows_install_service": {
                "artifact": "install-service.ps1",
                "sha256": "b" * 64,
            },
            "windows_restart_task": {
                "artifact": "restart-risk-layer-task.ps1",
                "sha256": "c" * 64,
            },
            "windows_install_backup_task": {
                "artifact": "install-backup-task.ps1",
                "sha256": "d" * 64,
            },
        },
    }
    payload = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    path = root / f"container-{release.release_id}.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _deployer(
    root: Path,
    compose: Path,
    secrets: Path,
    *,
    hook: list[str] | None = None,
) -> D.ContainerDeployer:
    return D.ContainerDeployer(
        root / "state",
        compose_file=compose,
        secret_env=secrets,
        verify_hook=hook or ["verify-release"],
        backup_hook=["backup-festwissen"],
    )


def test_deploy_schreibt_nur_nicht_geheime_generation_und_nutzt_compose_ohne_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release_path, release = _release(tmp_path / "releases", "v1.4.0", "a" * 40)
    container, fingerprint = _container_manifest(tmp_path, release, compose)
    calls: list[tuple[list[str], dict[str, str], float]] = []

    def run(command, *, environment, timeout):
        calls.append((list(command), dict(environment), timeout))

    monkeypatch.setattr(D, "_run_command", run)
    deployer = _deployer(tmp_path, compose, secrets)
    result = deployer.deploy(container, fingerprint, release_path)

    assert result["ok"] is True and result["changed"] is True
    state = result["state"]
    assert state["current"]["release_id"] == "v1.4.0"
    assert state["current"]["engine_digest"] == f"sha256:{'1' * 64}"
    assert state["current"]["container_manifest_sha256"] == fingerprint
    assert state["current"]["release_manifest_sha256"] == release.fingerprint()
    assert state["previous"] is None

    assert calls[0][0] == ["backup-festwissen"]
    assert calls[0][1]["RISK_LAYER_BACKUP_CANDIDATE_RELEASE_ID"] == "v1.4.0"
    assert calls[1][0][-2:] == ["pull", "engine"]
    assert calls[2][0][-6:] == [
        "up",
        "-d",
        "--wait",
        "--wait-timeout",
        "180",
        "engine",
    ]
    assert calls[3][0] == ["verify-release"]
    assert calls[3][1]["RISK_LAYER_EXPECTED_RELEASE_ID"] == "v1.4.0"
    assert all("RISK_LAYER_TOKEN" not in environment for _, environment, _ in calls)

    state_text = (tmp_path / "state" / "state.json").read_text(encoding="utf-8")
    env_path = tmp_path / "state" / state["current"]["env_file"]
    env_text = env_path.read_text(encoding="utf-8")
    assert "ssssssss" not in state_text
    assert "RISK_LAYER_TOKEN" not in state_text
    assert "RISK_LAYER_TOKEN" not in env_text
    assert f"RISK_LAYER_RELEASE_MANIFEST_SHA256={release.fingerprint()}" in env_text
    assert deployer.status()["env_verified"] is True


def test_fehlgeschlagenes_update_reaktiviert_previous_und_kompensiert_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release1_path, release1 = _release(tmp_path / "releases", "v1", "a" * 40)
    release2_path, release2 = _release(tmp_path / "releases", "v2", "b" * 40)
    container1, fingerprint1 = _container_manifest(tmp_path, release1, compose)
    hook_releases: list[str] = []
    compose_generations: list[str] = []

    def run(command, *, environment, timeout):
        del timeout
        if "-f" in command:
            compose_generations.append(str(command[list(command).index("-f") + 1]))
        if list(command) == ["verify-release"]:
            release_id = environment["RISK_LAYER_EXPECTED_RELEASE_ID"]
            hook_releases.append(release_id)
            if release_id == "v2":
                raise D.DeployError("candidate unhealthy")

    monkeypatch.setattr(D, "_run_command", run)
    deployer = _deployer(tmp_path, compose, secrets)
    deployer.deploy(container1, fingerprint1, release1_path)
    compose.write_text(
        "services:\n  engine:\n    image: example.invalid/engine-v2\n",
        encoding="utf-8",
    )
    container2, fingerprint2 = _container_manifest(
        tmp_path, release2, compose, digest_character="5"
    )

    with pytest.raises(D.DeployError, match="vollstaendig") as caught:
        deployer.deploy(container2, fingerprint2, release2_path)

    assert caught.value.rolled_back is True
    assert hook_releases == ["v1", "v2", "v1"]
    state = deployer.status()["state"]
    assert state["generation"] == 3
    assert state["current"]["release_id"] == "v1"
    assert state["previous"] is None
    assert "00000000000000000002-v2.compose.yml" in compose_generations[-3]
    assert "00000000000000000001-v1.compose.yml" in compose_generations[-1]


def test_explizites_rollback_tauscht_current_previous_und_startet_ziel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release1_path, release1 = _release(tmp_path / "releases", "r1", "a" * 40)
    release2_path, release2 = _release(tmp_path / "releases", "r2", "b" * 40)
    container1, fingerprint1 = _container_manifest(tmp_path, release1, compose)
    container2, fingerprint2 = _container_manifest(
        tmp_path, release2, compose, digest_character="6"
    )
    verified: list[str] = []
    operations: list[str] = []

    def run(command, *, environment, timeout):
        del timeout
        operations.append(list(command)[0])
        if list(command) == ["verify-release"]:
            verified.append(environment["RISK_LAYER_EXPECTED_RELEASE_ID"])

    monkeypatch.setattr(D, "_run_command", run)
    deployer = _deployer(tmp_path, compose, secrets)
    deployer.deploy(container1, fingerprint1, release1_path)
    deployer.deploy(container2, fingerprint2, release2_path)
    result = deployer.rollback()

    assert result["state"]["current"]["release_id"] == "r1"
    assert result["state"]["previous"]["release_id"] == "r2"
    assert verified == ["r1", "r2", "r1"]
    assert operations[-4:] == ["backup-festwissen", "docker", "docker", "verify-release"]


def test_dry_run_prueft_artefakte_aber_schreibt_keinen_state_und_startet_nichts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release_path, release = _release(tmp_path / "releases", "dry", "d" * 40)
    container, fingerprint = _container_manifest(tmp_path, release, compose)
    monkeypatch.setattr(
        D,
        "_run_command",
        lambda *args, **kwargs: pytest.fail("dry-run darf keinen Befehl starten"),
    )

    result = _deployer(tmp_path, compose, secrets).deploy(
        container, fingerprint, release_path, dry_run=True
    )

    assert result["dry_run"] is True
    assert result["commands"][0][-2:] == ["pull", "engine"]
    assert not (tmp_path / "state" / "state.json").exists()
    assert not any((tmp_path / "state" / "generations").iterdir())


def test_falscher_manifest_hash_und_managed_secret_override_werden_fail_closed_abgelehnt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release_path, release = _release(tmp_path / "releases", "safe", "e" * 40)
    container, fingerprint = _container_manifest(tmp_path, release, compose)
    monkeypatch.setattr(
        D,
        "_run_command",
        lambda *args, **kwargs: pytest.fail("abgelehntes Deploy darf Docker nicht starten"),
    )
    deployer = _deployer(tmp_path, compose, secrets)

    with pytest.raises(D.DeployError, match="erwarteten SHA"):
        deployer.deploy(container, "0" * 64, release_path)

    secrets.write_text(
        f"RISK_LAYER_TOKEN={'s' * 40}\nRISK_LAYER_ENGINE_DIGEST=sha256:{'f' * 64}\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        secrets.chmod(0o600)
    with pytest.raises(D.DeployError, match="verwalteten Wert"):
        deployer.deploy(container, fingerprint, release_path)
    assert not (tmp_path / "state" / "state.json").exists()


def test_pflichtbackup_laeuft_vor_switch_und_sein_fehler_laesst_state_unveraendert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release_path, release = _release(tmp_path / "releases", "backup", "b" * 40)
    container, fingerprint = _container_manifest(tmp_path, release, compose)
    calls: list[list[str]] = []

    def fail_backup(command, *, environment, timeout):
        del environment, timeout
        calls.append(list(command))
        if list(command) == ["backup-festwissen"]:
            raise D.DeployError("backup failed")

    monkeypatch.setattr(D, "_run_command", fail_backup)
    deployer = _deployer(tmp_path, compose, secrets)

    with pytest.raises(D.DeployError, match="backup failed"):
        deployer.deploy(container, fingerprint, release_path)

    assert calls == [["backup-festwissen"]]
    assert not (tmp_path / "state" / "state.json").exists()
    assert not any((tmp_path / "state" / "generations").iterdir())

    no_backup = D.ContainerDeployer(
        tmp_path / "other-state",
        compose_file=compose,
        secret_env=secrets,
        verify_hook=["verify-release"],
    )
    with pytest.raises(D.DeployError, match="Pflichtbackup"):
        no_backup.deploy(container, fingerprint, release_path, dry_run=True)


def test_manipulierte_env_generation_wird_vor_status_und_rollback_erkannt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release_path, release = _release(tmp_path / "releases", "v1", "a" * 40)
    container, fingerprint = _container_manifest(tmp_path, release, compose)
    monkeypatch.setattr(D, "_run_command", lambda *args, **kwargs: None)
    deployer = _deployer(tmp_path, compose, secrets)
    result = deployer.deploy(container, fingerprint, release_path)
    env_path = tmp_path / "state" / result["state"]["current"]["env_file"]
    env_path.write_text(
        env_path.read_text(encoding="utf-8") + f"RISK_LAYER_TOKEN={'x' * 40}\n",
        encoding="utf-8",
    )

    with pytest.raises(D.DeployError, match="State-SHA"):
        deployer.status()


class _StatusResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = json.dumps(payload).encode()
        self.headers = {"Content-Length": str(len(self.payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, maximum: int) -> bytes:
        return self.payload[:maximum]


def test_builtin_status_prueft_authentifiziert_die_drei_release_identitaeten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release_path, release = _release(tmp_path / "releases", "v9", "9" * 40)
    container, fingerprint = _container_manifest(tmp_path, release, compose)
    record = _deployer(tmp_path, compose, secrets)._candidate(
        D._load_container_manifest(container, fingerprint), release_path, 1
    )
    seen_authorization: list[str] = []

    def urlopen(status_request, timeout):
        assert timeout == 3
        seen_authorization.append(status_request.get_header("Authorization"))
        return _StatusResponse(
            {
                "engine_status": {
                    "release": {
                        "release_id": "v9",
                        "git_sha": "9" * 40,
                        "manifest_sha256": release.fingerprint(),
                    }
                }
            }
        )

    monkeypatch.setattr(D._STATUS_OPENER, "open", urlopen)
    D._verify_status(
        "http://127.0.0.1:8000/v1/status",
        record,
        token="t" * 40,
        timeout=3,
    )
    assert seen_authorization == [f"Bearer {'t' * 40}"]

    monkeypatch.setattr(
        D._STATUS_OPENER,
        "open",
        lambda *args, **kwargs: _StatusResponse(
            {
                "engine_status": {
                    "release": {
                        "release_id": "other",
                        "git_sha": "9" * 40,
                        "manifest_sha256": release.fingerprint(),
                    }
                }
            }
        ),
    )
    with pytest.raises(D.DeployError, match="erwartete Release-Identitaet"):
        D._verify_status(
            "http://127.0.0.1:8000/v1/status",
            record,
            token="t" * 40,
            timeout=3,
        )


def test_optional_manifestgebundenes_llm_modell_wird_streng_toleriert(
    tmp_path: Path,
) -> None:
    compose, _ = _runtime_files(tmp_path)
    _, release = _release(tmp_path / "releases", "llm", "c" * 40)
    container, _ = _container_manifest(tmp_path, release, compose)
    document = json.loads(container.read_text(encoding="utf-8"))
    document["llm_model"] = {
        "artifact": "granite.gguf",
        "sha256": "8" * 64,
        "revision": "refs/tags/v1",
        "provenance": "https://models.example/granite",
    }
    payload = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    container.write_bytes(payload)

    identity = D._load_container_manifest(container, hashlib.sha256(payload).hexdigest())
    assert identity.release_id == "llm"

    document["llm_model"]["unknown"] = "ambiguous"
    payload = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    container.write_bytes(payload)
    with pytest.raises(D.DeployError, match="LLM-Modell-Block"):
        D._load_container_manifest(container, hashlib.sha256(payload).hexdigest())


def test_unsicheres_remote_http_und_rollback_ohne_previous_werden_abgelehnt(
    tmp_path: Path,
) -> None:
    with pytest.raises(D.DeployError, match="nur auf Loopback"):
        D._validate_status_url("http://example.com/v1/status")

    compose, secrets = _runtime_files(tmp_path)
    deployer = _deployer(tmp_path, compose, secrets)
    with pytest.raises(D.DeployError, match="current/previous"):
        deployer.rollback()


def test_externe_befehle_werden_als_argv_ohne_shell_ausgefuehrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)

    monkeypatch.setattr(D.subprocess, "run", run)
    D._run_command(
        ["docker", "compose", "pull", "engine"],
        environment={"PATH": "safe"},
        timeout=12,
    )

    assert seen["command"] == ["docker", "compose", "pull", "engine"]
    assert seen["shell"] is False
    assert seen["check"] is True
    assert seen["env"] == {"PATH": "safe"}
    assert seen["timeout"] == 12


def test_bootstrap_schema_ist_exakt_mit_dem_container_manifest_vertrag_synchron(
    tmp_path: Path,
) -> None:
    compose, _ = _runtime_files(tmp_path)
    _, release = _release(tmp_path / "releases", "bootstrap", "a" * 40)
    manifest, fingerprint = _container_manifest(tmp_path, release, compose)

    assert D._load_container_manifest(manifest, fingerprint).release_id == "bootstrap"

    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert D.BOOTSTRAP_KEYS == C.BOOTSTRAP_KEYS == V.BOOTSTRAP_KEYS
    assert frozenset(document["bootstrap"]) == D.BOOTSTRAP_KEYS
    del document["bootstrap"]["release_launcher"]
    payload = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    manifest.write_bytes(payload)
    with pytest.raises(D.DeployError, match="Bootstrap-Block"):
        D._load_container_manifest(manifest, hashlib.sha256(payload).hexdigest())


def test_veroeffentlichte_secret_env_enthaelt_keine_manifestwerte() -> None:
    template = (ROOT / ".env.release.example").read_text(encoding="utf-8")
    assert "RISK_LAYER_TOKEN=" in template
    assert "RISK_LAYER_DEFINITIONEN_DIR=" in template
    active_lines = [
        line
        for line in template.splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    for key in D.MANAGED_ENV_KEYS:
        assert not any(line.startswith(f"{key}=") for line in active_lines)


def test_compose_scope_und_secret_snapshot_sind_stabil_und_nicht_in_prozess_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release_path, release = _release(tmp_path / "releases", "stable", "b" * 40)
    container, fingerprint = _container_manifest(tmp_path, release, compose)
    snapshots: list[Path] = []
    docker_environments: list[dict[str, str]] = []

    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "hijacked")
    monkeypatch.setenv("COMPOSE_PARALLEL_LIMIT", "99")

    def run(command, *, environment, timeout):
        del timeout
        command = list(command)
        if command == ["backup-festwissen"]:
            secrets.write_text(
                f"RISK_LAYER_TOKEN={'x' * 40}\n"
                f"RISK_LAYER_ENGINE_DIGEST=sha256:{'f' * 64}\n",
                encoding="utf-8",
            )
            if os.name != "nt":
                secrets.chmod(0o600)
            return
        if command[0] == "docker":
            assert command[1:4] == ["compose", "--project-name", D.COMPOSE_PROJECT]
            indices = [index for index, value in enumerate(command) if value == "--env-file"]
            assert len(indices) == 2
            snapshot = Path(command[indices[1] + 1])
            snapshots.append(snapshot)
            snapshot_text = snapshot.read_text(encoding="utf-8")
            assert "RISK_LAYER_ENGINE_DIGEST" not in snapshot_text
            assert "s" * 40 in snapshot_text
            docker_environments.append(dict(environment))

    monkeypatch.setattr(D, "_run_command", run)
    deployer = _deployer(tmp_path, compose, secrets)
    deployer.deploy(container, fingerprint, release_path)

    assert snapshots and len(set(snapshots)) == 1
    assert not snapshots[0].exists()
    assert all("RISK_LAYER_TOKEN" not in env for env in docker_environments)
    assert all(not any(key.startswith("COMPOSE_") for key in env) for env in docker_environments)
    assert _deployer(tmp_path / "other", compose, secrets).lock_store == deployer.lock_store


def test_compose_steuerwert_im_secret_env_wird_vor_docker_abgelehnt(
    tmp_path: Path,
) -> None:
    _, secrets = _runtime_files(tmp_path)
    secrets.write_text(
        f"RISK_LAYER_TOKEN={'s' * 40}\nCOMPOSE_PROJECT_NAME=hijacked\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        secrets.chmod(0o600)
    with pytest.raises(D.DeployError, match="Compose-Steuervariable"):
        D._parse_secret_env(secrets)


def test_offenes_switch_journal_wird_beim_naechsten_deploy_vorwaerts_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    release1_path, release1 = _release(tmp_path / "releases", "journal-v1", "a" * 40)
    container1, fingerprint1 = _container_manifest(tmp_path, release1, compose)
    monkeypatch.setattr(D, "_run_command", lambda *args, **kwargs: None)
    deployer = _deployer(tmp_path, compose, secrets)
    deployer.deploy(container1, fingerprint1, release1_path)

    compose.write_text(
        "services:\n  engine:\n    image: example.invalid/journal-v2\n",
        encoding="utf-8",
    )
    release2_path, release2 = _release(tmp_path / "releases", "journal-v2", "b" * 40)
    container2, fingerprint2 = _container_manifest(
        tmp_path, release2, compose, digest_character="7"
    )
    identity = D._load_container_manifest(container2, fingerprint2)

    with deployer._lock():
        old_state = deployer._read_state()
        candidate = deployer._candidate(identity, release2_path, old_state.generation + 1)
        D._write_immutable(deployer._env_path(candidate), D._env_payload(candidate))
        D._write_immutable(
            deployer._compose_path(candidate),
            deployer._candidate_compose_payload(candidate),
        )
        switched = D.DeploymentState(
            generation=old_state.generation + 1,
            current=candidate,
            previous=old_state.current,
            updated_at=D._utc_now(),
        )
        journal = D.DeploymentJournal(
            operation="deploy",
            phase="switched",
            old_state=old_state,
            switched_state=switched,
            updated_at=D._utc_now(),
        )
        deployer._write_journal(journal)
        deployer._write_state(switched)

    interrupted = deployer.status()
    assert interrupted["ok"] is False
    assert interrupted["recovery_required"] is True
    assert interrupted["journal_phase"] == "switched"

    calls: list[list[str]] = []

    def run(command, *, environment, timeout):
        del environment, timeout
        calls.append(list(command))

    monkeypatch.setattr(D, "_run_command", run)
    restarted = _deployer(tmp_path, compose, secrets)
    result = restarted.deploy(container2, fingerprint2, release2_path)

    assert result["changed"] is False
    assert not restarted.journal_path.exists()
    assert restarted.status()["state"]["current"]["release_id"] == "journal-v2"
    assert sum(call[-2:] == ["pull", "engine"] for call in calls) == 2


def test_deploy_blockt_schema_api_und_feature_inkompatibilitaeten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose, secrets = _runtime_files(tmp_path)
    current_path, current = _release(
        tmp_path / "releases",
        "compatible-base",
        "a" * 40,
        features=("audit", "embedding"),
    )
    current_container, current_fingerprint = _container_manifest(
        tmp_path, current, compose
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        D,
        "_run_command",
        lambda command, **kwargs: calls.append(list(command)),
    )
    deployer = _deployer(tmp_path, compose, secrets)
    deployer.deploy(current_container, current_fingerprint, current_path)
    calls.clear()

    cases = (
        ("schema-break", {"schema_version": "2", "features": ("audit", "embedding")}, "schema_version"),
        ("api-break", {"api_version": "2.0.0", "features": ("audit", "embedding")}, "API-Familie"),
        ("feature-break", {"features": ("embedding",)}, "entfernt deklarierte Features"),
    )
    for release_id, options, message in cases:
        candidate_path, candidate = _release(
            tmp_path / "releases",
            release_id,
            "b" * 40,
            **options,
        )
        candidate_container, candidate_fingerprint = _container_manifest(
            tmp_path, candidate, compose, digest_character="8"
        )
        with pytest.raises(D.DeployError, match=message):
            deployer.deploy(candidate_container, candidate_fingerprint, candidate_path)
        assert calls == []
        assert deployer.status()["state"]["current"]["release_id"] == "compatible-base"


def test_produktionscompose_vertrag_mit_beiden_env_generationen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker-CLI nicht installiert")
    version = D.subprocess.run(
        [docker, "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if version.returncode != 0:
        pytest.skip("Docker-Compose-Plugin nicht installiert")

    project_root = Path(__file__).resolve().parents[1]
    compose = project_root / "docker-compose.yml"
    release = tmp_path / "release"
    definitions = tmp_path / "definitionen"
    release.mkdir()
    definitions.mkdir()
    generated = tmp_path / "generated.env"
    managed = {
        "RISK_LAYER_ENGINE_REPOSITORY": "registry.example/risk-layer-engine",
        "RISK_LAYER_ENGINE_DIGEST": f"sha256:{'1' * 64}",
        "RISK_LAYER_RELEASE_ID": "v1.4.0",
        "RISK_LAYER_RELEASE_GIT_SHA": "a" * 40,
        "RISK_LAYER_RELEASE_MANIFEST_SHA256": "b" * 64,
        "RISK_LAYER_FESTWISSEN_DIR": str(release),
    }
    generated.write_text(
        "\n".join(f"{key}={D._dotenv_value(value)}" for key, value in managed.items())
        + "\n",
        encoding="utf-8",
    )
    secrets = tmp_path / "compose-secrets.env"
    secrets.write_text(
        f"RISK_LAYER_TOKEN={'s' * 40}\nRISK_LAYER_DEFINITIONEN_DIR={definitions}\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        secrets.chmod(0o600)
    deployer = D.ContainerDeployer(tmp_path / "state", secret_env=secrets)
    deployer._prepare_state_dir()
    monkeypatch.setenv("COMPOSE_PROJECT_NAME", "hijacked")
    with deployer._lock(), deployer._secret_snapshot_context() as snapshot:
        command = [
            *D._compose_base(docker, compose, generated, snapshot.path),
            "config",
            "--format",
            "json",
        ]
        configured = D.subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=D._safe_subprocess_env(),
        )
    document = json.loads(configured.stdout)
    assert document["name"] == D.COMPOSE_PROJECT
    assert document["services"]["engine"]["image"].endswith(f"@sha256:{'1' * 64}")
    assert any(
        volume.get("source") == str(release)
        for volume in document["services"]["engine"]["volumes"]
    )
