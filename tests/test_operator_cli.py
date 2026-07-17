"""Additive Operator-Fassade: Profil, Guards und reine Delegate-Abbildung."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

import pytest

from risk_layer import operator_cli as O
from tools import deploy_container_release as D


ROOT = Path(__file__).resolve().parents[1]
REAL_PYTHON = Path(sys.executable).resolve(strict=True)
TOKEN = "t" * 48


def _patch_delegate_run(
    monkeypatch: pytest.MonkeyPatch,
    replacement,
) -> None:
    real_run = O.subprocess.run

    def dispatch(command, **kwargs):
        executable = str(command[0]).casefold() if command else ""
        if executable.endswith("powershell.exe"):
            return real_run(command, **kwargs)
        return replacement(command, **kwargs)

    monkeypatch.setattr(O.subprocess, "run", dispatch)


def _private(path: Path) -> None:
    if os.name == "nt":
        O._harden_windows_directory_acl(path.parent)
        O._harden_windows_acl(path)
    else:
        path.parent.chmod(0o700)
        path.chmod(0o600)


def _write_backup_target(document: dict[str, object]) -> Path:
    backup_dir = Path(str(document["backup_dir"]))
    backup_dir.mkdir(parents=True, exist_ok=True)
    sentinel = backup_dir / O.BACKUP_TARGET_FILENAME
    sentinel.write_text(
        json.dumps(
            {
                "schema": O.BACKUP_TARGET_SCHEMA,
                "target_id": document["backup_target_id"],
            }
        ),
        encoding="utf-8",
    )
    _private(sentinel)
    return sentinel


def _profile_document(
    tmp_path: Path,
    *,
    mode: str = "container",
    activation: list[str] | None = None,
    bootstrap_python: str | None = None,
) -> dict[str, object]:
    container = mode == "container"
    return {
        "schema": O.PROFILE_SCHEMA,
        "mode": mode,
        "python": str(REAL_PYTHON),
        "bootstrap_python": bootstrap_python,
        "docker_bin": str(REAL_PYTHON) if mode == "container" else None,
        "release_home": str((tmp_path / "release-home").absolute()),
        "state_dir": str((tmp_path / "container-state").absolute()) if container else None,
        "berater_dir": str((tmp_path / "definitionen").absolute()),
        "backup_dir": str((tmp_path / "backups").absolute()),
        "backup_target_id": "test-offhost",
        "launcher": str((tmp_path / "bootstrap" / "release-launcher.py").absolute()),
        "release_tools": (
            str((tmp_path / "bootstrap" / "release-tools.pyz").absolute())
            if container
            else None
        ),
        "compose_file": (
            str((tmp_path / "bootstrap" / "docker-compose.yml").absolute())
            if container
            else None
        ),
        "secrets_env": (
            str((tmp_path / "private" / "container-secrets.env").absolute())
            if container
            else None
        ),
        "token_file": str((tmp_path / "private" / "engine.token").absolute()),
        "health_url": "http://127.0.0.1:8000",
        "status_url": "http://127.0.0.1:8000/v1/status",
        "activation_command": activation,
        "required_layers": ["matcher"],
        "retention": {"keep_last": 30, "keep_days": 90},
    }


def _write_profile(
    tmp_path: Path,
    *,
    mode: str = "container",
    activation: list[str] | None = None,
    bootstrap_python: str | None = None,
    prepare: bool = True,
) -> tuple[Path, dict[str, object]]:
    document = _profile_document(
        tmp_path,
        mode=mode,
        activation=activation,
        bootstrap_python=bootstrap_python,
    )
    profile_path = tmp_path / "ops.json"
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    if prepare:
        directories = ["release_home", "berater_dir", "backup_dir"]
        if mode == "container":
            directories.append("state_dir")
        for name in directories:
            Path(str(document[name])).mkdir(parents=True, exist_ok=True)
        _write_backup_target(document)
        control_files = ["launcher"]
        if mode == "container":
            control_files.extend(("release_tools", "compose_file"))
        for name in control_files:
            path = Path(str(document[name]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("delegate\n", encoding="utf-8")
        token_file = Path(str(document["token_file"]))
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(f"{TOKEN}\n", encoding="utf-8")
        _private(token_file)
        if mode == "container":
            secrets_env = Path(str(document["secrets_env"]))
            secrets_env.write_text(
                f"RISK_LAYER_TOKEN={TOKEN}\n"
                f"RISK_LAYER_DEFINITIONEN_DIR={document['berater_dir']}\n",
                encoding="utf-8",
            )
            _private(secrets_env)
    return profile_path, document


def _state(document: dict[str, object], *, native: bool, present: bool = True) -> Path:
    root = Path(str(document["release_home"] if native else document["state_dir"]))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "state.json"
    if present:
        path.write_text("{}\n", encoding="utf-8")
    return path


def _native_status(
    command,
    *,
    release_id: str = "v1.2.3",
    git_sha: str = "a" * 40,
    manifest_sha256: str = "b" * 64,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        0,
        json.dumps(
            {
                "ok": True,
                "state": {
                    "current": {
                        "release_id": release_id,
                        "git_sha": git_sha,
                        "manifest_sha256": manifest_sha256,
                    }
                },
            }
        ),
        "",
    )


def _args(*values: str) -> object:
    return O.build_parser().parse_args(list(values))


def _write_checkout_contract(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "risk_layer").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "risk_layer" / "operator_cli.py").write_text("# operator\n", encoding="utf-8")
    setup = root / "scripts" / ("setup.ps1" if os.name == "nt" else "setup.sh")
    setup.write_text("# setup\n", encoding="utf-8")
    if os.name != "nt":
        setup.chmod(0o700)
    return root


def _write_release_inbox(
    document: dict[str, object],
    *,
    release_id: str = "v2.0.0",
) -> tuple[Path, Path, str, str]:
    inbox = Path(str(document["release_home"])) / O.RELEASE_INBOX_NAME
    inbox.mkdir(parents=True, exist_ok=True)
    bundle = inbox / f"release-{release_id}"
    bundle.mkdir()
    native_payload = (json.dumps({"release_id": release_id}, sort_keys=True) + "\n").encode()
    (bundle / O.NATIVE_MANIFEST_NAME).write_bytes(native_payload)
    native_digest = hashlib.sha256(native_payload).hexdigest()
    container = inbox / f"taxtronik-container-{release_id}.json"
    container_payload = (
        json.dumps(
            {
                "schema": "taxtronik-release/v1",
                "release_id": release_id,
                "git_sha": "a" * 40,
                "native_bundle": {
                    "artifact": f"taxtronik-native-{release_id}.tar.gz",
                    "sha256": "b" * 64,
                    "manifest_sha256": native_digest,
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    container.write_bytes(container_payload)
    container_digest = hashlib.sha256(container_payload).hexdigest()
    Path(f"{container}.sha256").write_text(
        f"{container_digest}  {container.name}\n", encoding="ascii"
    )
    return container, bundle, container_digest, native_digest


def test_standardprofil_ist_kanonischer_produktionspfad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RISK_LAYER_OPS_PROFILE", raising=False)
    args = O.build_parser().parse_args(["status", "--offline"])
    expected = (
        "C:/ProgramData/TaxTronik/RiskLayer/config/ops.json"
        if os.name == "nt"
        else "/etc/risk-layer/ops.json"
    )
    assert O.DEFAULT_PROFILE == expected
    assert args.config == expected


def test_aufruf_ohne_kommando_zeigt_hilfe_ohne_profilzugriff(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert O.main([]) == 0
    assert "signal" in capsys.readouterr().out


def test_checkout_bootstrap_ist_zero_config_und_laesst_etc_profil_aus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_checkout_contract(tmp_path)
    monkeypatch.setenv(O.CHECKOUT_ROOT_ENV, str(root))
    monkeypatch.delenv("RISK_LAYER_OPS_PROFILE", raising=False)
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        O,
        "_dispatch_checkout",
        lambda checkout, args: calls.append((checkout, args.command)) or 0,
    )
    monkeypatch.setattr(
        O,
        "load_profile",
        lambda _path: pytest.fail("Zero-Config-Checkout darf kein /etc-Profil laden"),
    )

    assert O.main(["bootstrap"]) == 0
    assert calls == [(root, "bootstrap")]


def test_checkout_bootstrap_erzeugt_layout_und_fuehrt_setup_aus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_checkout_contract(tmp_path)
    monkeypatch.setenv(O.CHECKOUT_ROOT_ENV, str(root))
    monkeypatch.delenv("RISK_LAYER_OPS_PROFILE", raising=False)
    monkeypatch.setattr(
        O,
        "load_profile",
        lambda _path: pytest.fail("Zero-Config-Checkout darf kein /etc-Profil laden"),
    )

    assert O.main(["bootstrap"]) == 0

    profile = O._checkout_profile(root)
    assert profile.release_home.is_dir()
    assert profile.berater_dir.is_dir()
    assert profile.backup_dir.is_dir()
    assert profile.token_file.is_file()
    assert O._require_backup_target(profile).is_file()


def test_explizites_profil_schaltet_checkout_automatik_ab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_checkout_contract(tmp_path)
    monkeypatch.setenv(O.CHECKOUT_ROOT_ENV, str(root))

    assert O._checkout_root(["--config", str(tmp_path / "ops.json"), "status"]) is None


def test_checkout_layout_erzeugt_lokale_pfade_token_und_sentinel(tmp_path: Path) -> None:
    profile = O._checkout_profile(tmp_path)

    created = O._ensure_checkout_layout(profile)

    assert profile.release_home.is_dir()
    assert (profile.release_home / O.RELEASE_INBOX_NAME).is_dir()
    assert profile.berater_dir.is_dir()
    assert profile.backup_dir.is_dir()
    assert profile.token_file.is_file()
    assert O._require_backup_target(profile).is_file()
    assert str(profile.token_file) in created


def test_checkout_profil_nutzt_persistierten_service_endpoint(tmp_path: Path) -> None:
    state_dir = tmp_path / O.CHECKOUT_STATE_DIR
    state_dir.mkdir(mode=0o700)
    state = state_dir / O.CHECKOUT_SERVICE_STATE_NAME
    state.write_text(
        json.dumps(
            {
                "schema": O.CHECKOUT_SERVICE_SCHEMA,
                "host": "172.18.0.1",
                "port": 8123,
                "scope": "system",
            }
        ),
        encoding="utf-8",
    )

    profile = O._checkout_profile(tmp_path)

    assert profile.health_url == "http://172.18.0.1:8123"
    assert profile.status_url == "http://172.18.0.1:8123/v1/status"


def test_checkout_service_endpoint_mappt_wildcard_auf_loopback(tmp_path: Path) -> None:
    state_dir = tmp_path / O.CHECKOUT_STATE_DIR
    state_dir.mkdir(mode=0o700)
    (state_dir / O.CHECKOUT_SERVICE_STATE_NAME).write_text(
        json.dumps(
            {
                "schema": O.CHECKOUT_SERVICE_SCHEMA,
                "host": "::",
                "port": 8000,
                "scope": "user",
            }
        ),
        encoding="utf-8",
    )

    assert O._checkout_health_urls(tmp_path) == (
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/v1/status",
    )


def test_checkout_bootstrap_delegiert_parameterlos_an_lokales_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_checkout_contract(tmp_path)
    profile = O._checkout_profile(root)
    calls: list[list[str]] = []
    monkeypatch.setattr(O, "_ensure_checkout_layout", lambda _profile: [])
    monkeypatch.setattr(
        O,
        "_checkout_setup_command",
        lambda _root: ["setup-program"],
    )
    monkeypatch.setattr(
        O,
        "_passthrough",
        lambda command: calls.append(list(command)) or 0,
    )

    assert O._checkout_deployment(root, profile, _args("bootstrap")) == 0
    assert calls == [["setup-program"]]


def test_checkout_setup_provisioniert_service_ohne_benutzerparameter() -> None:
    command = O._checkout_setup_command(ROOT)

    if os.name == "nt":
        assert command[-1] == "-Service"
    else:
        assert command[-1] == "--service"


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_checkout_dienstkommandos_werden_zentral_delegiert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root = tmp_path / "checkout"
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        O,
        "_checkout_service_control",
        lambda checkout, _profile, action: calls.append((checkout, action)) or 0,
    )

    assert O._dispatch_checkout(root, _args(operation)) == 0
    assert calls == [(root, operation)]


@pytest.mark.parametrize("operation", ["start", "stop", "restart"])
def test_checkout_dienstbefehl_nutzt_plattformeigenen_benutzerdienst(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setattr(O, "_require_control_file", lambda *_args, **_kwargs: None)
    if os.name == "nt":
        powershell = Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
        monkeypatch.setattr(O, "_windows_powershell", lambda: powershell)
        command = O._checkout_service_command(tmp_path, operation)
        assert command[0] == str(powershell)
        if operation == "restart":
            assert "restart-risk-layer-task.ps1" in " ".join(command)
        else:
            expected = "Start-ScheduledTask" if operation == "start" else "Stop-ScheduledTask"
            assert expected in command[-1]
    else:
        monkeypatch.setattr(O.shutil, "which", lambda _name: "/usr/bin/systemctl")
        monkeypatch.setattr(O, "_require_executable_control", lambda *_args: None)
        monkeypatch.setattr(
            O,
            "_checkout_systemd_scope",
            lambda: (
                Path("/home/signal/.config/systemd/user/risk-layer.service"),
                ["--user"],
                "User-Unit",
            ),
        )
        assert O._checkout_service_command(tmp_path, operation) == [
            "/usr/bin/systemctl",
            "--user",
            operation,
            O.CHECKOUT_SYSTEMD_UNIT,
        ]


def test_root_steuert_systemweite_unit_ohne_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(O.os, "geteuid", lambda: 0, raising=False)

    unit, scope, description = O._checkout_systemd_scope()

    assert unit == Path("/etc/systemd/system/risk-layer.service")
    assert scope == []
    assert "System-Unit" in description


def test_normaler_benutzer_steuert_user_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(O.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(O.Path, "home", lambda: Path("/home/signal"))

    unit, scope, description = O._checkout_systemd_scope()

    assert unit == Path("/home/signal/.config/systemd/user/risk-layer.service")
    assert scope == ["--user"]
    assert "User-Unit" in description


@pytest.mark.skipif(os.name == "nt", reason="POSIX-Ownership-Vertrag")
def test_root_checkout_akzeptiert_dienstkonto_als_definitionseigentuemer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "definitionen"
    directory.mkdir(mode=0o700)
    identity = (os.getuid(), os.getgid())
    monkeypatch.setattr(O, "_checkout_service_identity", lambda: identity)

    O._secure_checkout_directory(directory, allow_service_owner=True)

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX-Ownership-Vertrag")
def test_restore_normalisiert_definitionen_fuer_dienstkonto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "definitionen"
    nested = directory / "fachlich"
    nested.mkdir(parents=True)
    definition = nested / "persoenlich.yaml"
    definition.write_text("schema: test\n", encoding="utf-8")
    identity = (os.getuid(), os.getgid())
    monkeypatch.setattr(O, "_checkout_service_identity", lambda: identity)

    O._normalize_checkout_berater_ownership(directory)

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(nested.stat().st_mode) == 0o700
    assert stat.S_IMODE(definition.stat().st_mode) == 0o600


def test_checkout_update_sichert_zieht_ff_only_und_setzt_neu_auf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_checkout_contract(tmp_path)
    profile = O._checkout_profile(root)
    operations: list[object] = []
    monkeypatch.setattr(O, "_ensure_checkout_layout", lambda _profile: [])
    monkeypatch.setattr(O, "_checkout_is_bootstrapped", lambda *_args: True)
    monkeypatch.setattr(
        O,
        "_checkout_git",
        lambda _root, arguments, **kwargs: (
            operations.append(("git", list(arguments)))
            or subprocess.CompletedProcess(arguments, 0, "", "")
        ),
    )
    monkeypatch.setattr(
        O,
        "_cmd_backup",
        lambda _profile, _args: operations.append("backup") or 0,
    )
    monkeypatch.setattr(O, "_checkout_setup_command", lambda _root: ["setup"])
    monkeypatch.setattr(
        O,
        "_passthrough",
        lambda command: operations.append(("setup", list(command))) or 0,
    )

    assert O._checkout_deployment(root, profile, _args("update")) == 0
    assert operations == [
        ("git", ["status", "--porcelain"]),
        "backup",
        ("git", ["pull", "--ff-only"]),
        ("setup", ["setup"]),
    ]


def test_checkout_update_benennt_blockierende_git_eintraege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_checkout_contract(tmp_path)
    profile = O._checkout_profile(root)
    monkeypatch.setattr(O, "_ensure_checkout_layout", lambda _profile: [])
    monkeypatch.setattr(O, "_checkout_is_bootstrapped", lambda *_args: True)
    monkeypatch.setattr(
        O,
        "_checkout_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["git", "status"], 0, "?? corpus/unbekannt.json\n M signal\n", ""
        ),
    )
    monkeypatch.setattr(
        O,
        "_cmd_backup",
        lambda *_args: pytest.fail("Bei blockiertem Update darf kein Backup starten"),
    )

    with pytest.raises(O.OperatorError, match="corpus/unbekannt.json") as error:
        O._checkout_deployment(root, profile, _args("update"))

    assert "M signal" in str(error.value)


@pytest.mark.parametrize(
    "generated",
    [
        "corpus/norm_manifest.json",
        "corpus/norm_manifest.vorher.json",
        ".definitionen.festwissen.lock",
        "service-start.ps1",
        "service-start.cmd",
    ],
)
def test_checkout_update_ignoriert_generierte_betriebsartefakte(generated: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", generated],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, generated


def test_checkout_ops_nutzt_lokalen_venv_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    profile = O._checkout_profile(root)
    args = _args("backup")
    args.checkout_root = root
    monkeypatch.setattr(
        O,
        "_checkout_ops_command",
        lambda checkout, arguments: [str(checkout), *arguments],
    )
    monkeypatch.setattr(
        O,
        "_ops_command",
        lambda *_args: pytest.fail("Checkout darf keinen profilierten Interpreter nutzen"),
    )

    assert O._operator_ops_command(profile, args, ["backup"]) == [
        str(root),
        "backup",
    ]


def test_profil_ist_strikt_nicht_mehrdeutig_und_container_hat_keine_bootstrap_venv(
    tmp_path: Path,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    assert O.load_profile(profile_path).mode == "container"

    document["unknown"] = True
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(O.OperatorError, match="fehlende/unbekannte"):
        O.load_profile(profile_path)

    document.pop("unknown")
    document["bootstrap_python"] = str(REAL_PYTHON)
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(O.OperatorError, match="bootstrap_python auf null"):
        O.load_profile(profile_path)


@pytest.mark.parametrize(
    "target_id",
    ["", ".hidden", "trailing-", "bad/target", "bad target", "x" * 65, True],
)
def test_profil_verlangt_kurze_sichere_backup_target_id(
    tmp_path: Path,
    target_id: object,
) -> None:
    profile_path, document = _write_profile(tmp_path, prepare=False)
    document["backup_target_id"] = target_id
    profile_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(O.OperatorError, match="backup_target_id"):
        O.load_profile(profile_path)


def test_profil_bindet_modus_urls_layer_und_absolute_executables(tmp_path: Path) -> None:
    profile_path, document = _write_profile(tmp_path)

    document["activation_command"] = [str(REAL_PYTHON), "-V"]
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(O.OperatorError, match="activation_command auf null"):
        O.load_profile(profile_path)

    document["activation_command"] = None
    document["required_layers"] = ["Matcher", "matcher"]
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(O.OperatorError, match="eindeutige Liste"):
        O.load_profile(profile_path)

    document["required_layers"] = ["matcher"]
    document["status_url"] = "http://127.0.0.1:9000/v1/status"
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(O.OperatorError, match="exakt health_url"):
        O.load_profile(profile_path)

    document["status_url"] = "http://127.0.0.1:8000/v1/status"
    document["docker_bin"] = None
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(O.OperatorError, match="docker_bin"):
        O.load_profile(profile_path)

    document["mode"] = "native"
    for field in O.CONTAINER_PATH_FIELDS:
        document[field] = None
    document["docker_bin"] = str(REAL_PYTHON)
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(O.OperatorError, match="docker_bin auf null"):
        O.load_profile(profile_path)


@pytest.mark.parametrize("field", sorted(O.CONTAINER_PATH_FIELDS))
def test_native_profil_setzt_container_pfade_explizit_auf_null(
    tmp_path: Path,
    field: str,
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    profile = O.load_profile(profile_path)
    assert profile.state_dir is None
    assert profile.release_tools is None
    assert profile.compose_file is None
    assert profile.secrets_env is None

    document[field] = str((tmp_path / field).absolute())
    profile_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(O.OperatorError, match=field):
        O.load_profile(profile_path)


@pytest.mark.parametrize(
    ("field", "value_from", "suffix"),
    [
        ("backup_dir", "berater_dir", ""),
        ("backup_dir", "berater_dir", "nested"),
        ("backup_dir", "release_home", ".."),
        ("state_dir", "release_home", ""),
    ],
)
def test_profil_verweigert_ueberlappende_managed_verzeichnisse(
    tmp_path: Path,
    field: str,
    value_from: str,
    suffix: str,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    base = Path(str(document[value_from]))
    document[field] = str((base / suffix).absolute()) if suffix else str(base)
    profile_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(O.OperatorError, match="ueberlappen"):
        O.load_profile(profile_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_file", "launcher"),
        ("launcher", "release_home"),
        ("secrets_env", "berater_child"),
    ],
)
def test_profil_verweigert_kollidierende_datei_und_private_pfade(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    if value == "berater_child":
        replacement = Path(str(document["berater_dir"])) / "secret.env"
    else:
        replacement = Path(str(document[value]))
    document[field] = str(replacement.absolute())
    profile_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(O.OperatorError, match="kollidier"):
        O.load_profile(profile_path)


def test_doctor_ohne_fix_erzeugt_nichts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profile_path, _ = _write_profile(tmp_path, prepare=False)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert O.main(["--config", str(profile_path), "doctor"]) == 1

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert json.loads(capsys.readouterr().out)["created"] == []


def test_doctor_fix_erzeugt_fehlendes_backup_ziel_nie_lokal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, _ = _write_profile(tmp_path, prepare=False)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert O.main(["--config", str(profile_path), "doctor", "--fix"]) == 2
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert "Backup-Ziel" in capsys.readouterr().err


def test_doctor_fix_erzeugt_auch_in_bestehendem_ziel_keinen_sentinel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, document = _write_profile(tmp_path, prepare=False)
    backup_dir = Path(str(document["backup_dir"]))
    backup_dir.mkdir(parents=True)
    if os.name == "nt":
        O._harden_windows_directory_acl(backup_dir, private_owner=False)
    else:
        backup_dir.chmod(0o700)
    sentinel = backup_dir / O.BACKUP_TARGET_FILENAME

    assert O.main(["--config", str(profile_path), "doctor", "--fix"]) == 2
    assert not sentinel.exists()
    assert "Sentinel" in capsys.readouterr().err


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema":"taxtronik.risk-layer-backup-target/v1","target_id":"wrong"}',
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
def test_backup_ziel_sentinel_ist_strikt_und_identitaetsgebunden(
    tmp_path: Path,
    raw: str,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    sentinel = Path(str(document["backup_dir"])) / O.BACKUP_TARGET_FILENAME
    sentinel.write_text(raw, encoding="utf-8")
    _private(sentinel)

    with pytest.raises(O.OperatorError, match="target_id|Doppelter|Schluessel"):
        O._require_backup_target(O.load_profile(profile_path))


def test_backup_ziel_sentinel_darf_keinen_hardlink_besitzen(tmp_path: Path) -> None:
    profile_path, document = _write_profile(tmp_path)
    sentinel = Path(str(document["backup_dir"])) / O.BACKUP_TARGET_FILENAME
    hardlink = sentinel.with_name("sentinel-hardlink.json")
    try:
        os.link(sentinel, hardlink)
    except OSError:
        pytest.skip("Hardlinks werden auf diesem Test-Dateisystem nicht unterstuetzt")

    with pytest.raises(O.OperatorError, match="Hardlinks"):
        O._require_backup_target(O.load_profile(profile_path))


def test_doctor_fix_meldet_betriebssystemfehler_ohne_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, _ = _write_profile(tmp_path)

    def fail(_profile: O.Profile) -> None:
        raise PermissionError("Zugriff verweigert")

    monkeypatch.setattr(O, "_doctor_fix", fail)

    assert O.main(["--config", str(profile_path), "doctor", "--fix"]) == 2
    error = capsys.readouterr().err
    assert "Betriebssystemzugriff fehlgeschlagen" in error
    assert "Traceback" not in error


def test_doctor_fix_erzeugt_nur_profilierte_verzeichnisse_token_und_secret_env(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, document = _write_profile(tmp_path, prepare=False)
    _write_backup_target(document)
    for name in ("launcher", "release_tools", "compose_file"):
        path = Path(str(document[name]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("delegate\n", encoding="utf-8")
    _state(document, native=True)
    if os.name == "nt":
        O._harden_windows_directory_acl(
            Path(str(document["release_home"])), private_owner=False
        )

    assert O.main(["--config", str(profile_path), "doctor", "--fix"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    token_file = Path(str(document["token_file"]))
    secrets_env = Path(str(document["secrets_env"]))
    release_inbox = Path(str(document["release_home"])) / O.RELEASE_INBOX_NAME
    assert len(token_file.read_text(encoding="utf-8").strip()) >= 32
    assert "RISK_LAYER_TOKEN=" in secrets_env.read_text(encoding="utf-8")
    assert release_inbox.is_dir()
    assert not (tmp_path / "state.json").exists()
    if os.name == "nt":
        for path in (token_file.parent, token_file, secrets_env):
            acl = O._windows_acl_document(path)
            assert acl["protected"] is True
            assert acl["owner"] == acl["current"]
            assert all(rule["inherited"] is False for rule in acl["rules"])


def test_doctor_fix_uebernimmt_bestehendes_secret_statt_zweites_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, document = _write_profile(tmp_path, prepare=False)
    _write_backup_target(document)
    for name in ("launcher", "release_tools", "compose_file"):
        path = Path(str(document[name]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("delegate\n", encoding="utf-8")
    secret = Path(str(document["secrets_env"]))
    secret.parent.mkdir(parents=True)
    secret.write_text(
        f"RISK_LAYER_TOKEN={TOKEN}\n"
        f"RISK_LAYER_DEFINITIONEN_DIR={document['berater_dir']}\n",
        encoding="utf-8",
    )
    _private(secret)
    _state(document, native=True)
    if os.name == "nt":
        O._harden_windows_directory_acl(
            Path(str(document["release_home"])), private_owner=False
        )

    assert O.main(["--config", str(profile_path), "doctor", "--fix"]) == 0
    assert Path(str(document["token_file"])).read_text(encoding="utf-8").strip() == TOKEN
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_native_bootstrap_nutzt_nur_explizite_kurzlebige_control_venv(
    tmp_path: Path,
) -> None:
    bootstrap_python = str(REAL_PYTHON)
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=bootstrap_python,
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    profile = O.load_profile(profile_path)
    args = _args("bootstrap", str(bundle), "--trust-local-manifest")

    command = O._deployment_command(profile, args)

    assert command[:5] == [
        bootstrap_python,
        "-I",
        "-m",
        "risk_layer.ops_cli",
        "release",
    ]
    assert "--trust-local-manifest" in command
    assert command[-2:] == ["--no-activate", "--no-health-check"]
    assert not Path(str(document["release_home"]), "state.json").exists()


def test_native_update_braucht_state_und_nutzt_danach_nur_launcher(tmp_path: Path) -> None:
    missing_bootstrap = str((tmp_path / "already-removed-control-python").absolute())
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        activation=[str(REAL_PYTHON), "-c", "pass"],
        bootstrap_python=missing_bootstrap,
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    profile = O.load_profile(profile_path)
    args = _args("update", str(bundle), "--manifest-sha256", "a" * 64)
    with pytest.raises(O.OperatorError, match="vorhandenen Deployment-State"):
        O._deployment_command(profile, args)

    _state(document, native=True)
    command = O._deployment_command(profile, args)
    assert command[:4] == [
        str(document["python"]),
        "-I",
        str(document["launcher"]),
        "--home",
    ]
    assert missing_bootstrap not in command
    activation = json.loads(command[command.index("--activation-command") + 1])
    assert activation[0:2] == [str(REAL_PYTHON), "-c"]
    assert "--health-url" in command and "--no-health-check" not in command
    assert command[command.index("--retention-keep-last") + 1] == "30"
    assert command[command.index("--retention-keep-days") + 1] == "90"


def test_parameterloses_native_update_nutzt_eindeutigen_gehashten_release_eingang(
    tmp_path: Path,
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    _state(document, native=True)
    _container, bundle, _container_digest, native_digest = _write_release_inbox(
        document
    )

    command = O._deployment_command(O.load_profile(profile_path), _args("update"))

    assert str(bundle) in command
    assert command[command.index("--manifest-sha256") + 1] == native_digest
    assert "--trust-local-manifest" not in command


def test_parameterloses_container_deploy_nutzt_publizierte_sidecar(
    tmp_path: Path,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    container, _bundle, container_digest, _native_digest = _write_release_inbox(
        document
    )

    artifact, digest = O._automatic_deployment_artifact(O.load_profile(profile_path))

    assert artifact == container
    assert digest == container_digest


def test_parameterloses_update_blockt_mehrdeutige_oder_manipulierte_inbox(
    tmp_path: Path,
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    first, _bundle, _container_digest, _native_digest = _write_release_inbox(
        document
    )
    second = first.with_name("taxtronik-container-v3.0.0.json")
    second.write_bytes(first.read_bytes())
    Path(f"{second}.sha256").write_text(
        f"{hashlib.sha256(second.read_bytes()).hexdigest()}  {second.name}\n",
        encoding="ascii",
    )
    with pytest.raises(O.OperatorError, match="genau ein"):
        O._automatic_deployment_artifact(O.load_profile(profile_path))

    second.unlink()
    Path(f"{second}.sha256").unlink()
    Path(f"{first}.sha256").write_text(f"{'0' * 64}  {first.name}\n", encoding="ascii")
    with pytest.raises(O.OperatorError, match="stimmt nicht"):
        O._automatic_deployment_artifact(O.load_profile(profile_path))


def test_bootstrap_blockt_vorhandenen_state(tmp_path: Path) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    _state(document, native=True)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(O.OperatorError, match="ohne vorhandenen"):
        O._deployment_command(
            O.load_profile(profile_path),
            _args("bootstrap", str(bundle), "--trust-local-manifest"),
        )


def test_backup_delegate_braucht_release_home_nur_lesend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    _state(document, native=True)
    checked: list[tuple[Path, str]] = []

    def control(path: Path, description: str) -> None:
        checked.append((path, description))

    def reject_write_check(path: Path, description: str) -> None:
        raise AssertionError(f"unerwartete Schreibpruefung fuer {description}: {path}")

    monkeypatch.setattr(O, "_require_control_directory", control)
    monkeypatch.setattr(O, "_require_managed_directory", reject_write_check)
    monkeypatch.setattr(O, "_require_executable_control", lambda *_args: None)
    monkeypatch.setattr(O, "_require_control_file", lambda *_args: None)

    command = O._ops_command(O.load_profile(profile_path), ["backup", "create"])

    assert command[-2:] == ["backup", "create"]
    assert checked == [(Path(str(document["release_home"])), "Release-Home")]


def test_update_blockt_fremd_schreibbaren_deployment_state(tmp_path: Path) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    state = _state(document, native=True)
    if os.name == "nt":
        result = subprocess.run(
            ["icacls.exe", str(state), "/grant", "*S-1-1-0:(M)"],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    else:
        state.chmod(0o666)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(O.OperatorError, match="Deployment-State.*schreib"):
        O._deployment_command(
            O.load_profile(profile_path),
            _args("update", str(bundle), "--trust-local-manifest"),
        )


def test_status_blockt_fremd_schreibbares_managed_state_verzeichnis(
    tmp_path: Path,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    state_dir = Path(str(document["state_dir"]))
    if os.name == "nt":
        result = subprocess.run(
            ["icacls.exe", str(state_dir), "/grant", "*S-1-1-0:(OI)(CI)(M)"],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    else:
        state_dir.chmod(0o777)
    with pytest.raises(O.OperatorError, match="Container-State.*schreib"):
        O._status_state_command(O.load_profile(profile_path))


def test_container_verlangt_expliziten_hash_und_leitet_release_pfad_selbst_ab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    _state(document, native=True)
    release_path = Path(str(document["release_home"])) / "releases" / "v1.2.3"
    release_path.mkdir(parents=True)
    manifest = tmp_path / "taxtronik-container-v1.2.3.json"
    payload = json.dumps(
        {
            "schema": "taxtronik-release/v1",
            "release_id": "v1.2.3",
            "git_sha": "a" * 40,
            "native_bundle": {
                "artifact": "native-v1.2.3.tar.gz",
                "sha256": "c" * 64,
                "manifest_sha256": "b" * 64,
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    manifest.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    profile = O.load_profile(profile_path)
    _patch_delegate_run(monkeypatch, lambda command, **_kwargs: _native_status(command))

    with pytest.raises(O.OperatorError, match="explizit --manifest-sha256"):
        O._deployment_command(profile, _args("deploy", str(manifest)))
    with pytest.raises(O.OperatorError, match="kein --trust-local-manifest"):
        O._deployment_command(
            profile,
            _args("deploy", str(manifest), "--trust-local-manifest"),
        )

    command = O._deployment_command(
        profile,
        _args("deploy", str(manifest), "--manifest-sha256", digest),
    )
    assert command[:4] == [
        str(document["python"]),
        "-I",
        str(document["release_tools"]),
        "deploy-container",
    ]
    assert command[command.index("--release-path") + 1] == str(release_path)
    assert command[command.index("--docker-bin") + 1] == str(document["docker_bin"])
    hook = json.loads(command[command.index("--backup-hook-json") + 1])
    assert hook[2] == str(document["launcher"])
    assert "--retention-keep-last" in hook and "30" in hook
    assert "--retention-keep-days" in hook and "90" in hook
    assert hook[hook.index("--backup-target-id") + 1] == "test-offhost"


def test_container_manifest_hash_hat_expliziten_invarianten_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path, _ = _write_profile(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(O, "_trust_arguments", lambda _profile, _args: [])

    with pytest.raises(O.OperatorError, match="Interner Container-Deploy-Vertrag"):
        O._deployment_command(
            O.load_profile(profile_path),
            _args("deploy", str(manifest)),
        )


def test_container_deploy_blockt_abweichende_native_current_identitaet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    _state(document, native=True)
    release_path = Path(str(document["release_home"])) / "releases" / "v1.2.3"
    release_path.mkdir(parents=True)
    manifest = tmp_path / "taxtronik-container-v1.2.3.json"
    payload = json.dumps(
        {
            "schema": "taxtronik-release/v1",
            "release_id": "v1.2.3",
            "git_sha": "a" * 40,
            "native_bundle": {
                "artifact": "native-v1.2.3.tar.gz",
                "sha256": "c" * 64,
                "manifest_sha256": "b" * 64,
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    manifest.write_bytes(payload)
    calls: list[list[str]] = []

    def wrong_status(command, **_kwargs):
        calls.append(list(command))
        return _native_status(command, manifest_sha256="d" * 64)

    _patch_delegate_run(monkeypatch, wrong_status)
    with pytest.raises(O.OperatorError, match="stimmt nicht"):
        O._deployment_command(
            O.load_profile(profile_path),
            _args(
                "deploy",
                str(manifest),
                "--manifest-sha256",
                hashlib.sha256(payload).hexdigest(),
            ),
        )

    assert len(calls) == 1
    assert "release" in calls[0] and "status" in calls[0]


def test_container_release_id_kann_nicht_aus_release_home_ausbrechen(tmp_path: Path) -> None:
    profile_path, document = _write_profile(tmp_path)
    _state(document, native=True)
    manifest = tmp_path / "manifest.json"
    payload = b'{"schema":"taxtronik-release/v1","release_id":"../escape"}'
    manifest.write_bytes(payload)
    with pytest.raises(O.OperatorError, match="sichere release_id"):
        O._deployment_command(
            O.load_profile(profile_path),
            _args(
                "deploy",
                str(manifest),
                "--manifest-sha256",
                hashlib.sha256(payload).hexdigest(),
            ),
        )


def test_delegate_wird_immer_ohne_shell_ausgefuehrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def run(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(O.subprocess, "run", run)
    assert O._passthrough(["python", "-V"]) == 0
    assert seen["shell"] is False
    assert seen["check"] is False
    assert seen["command"] == ["python", "-V"]


@pytest.mark.parametrize(
    ("mode", "manifest_field", "release_id"),
    [
        ("container", "release_manifest_sha256", "container-current"),
        ("native", "manifest_sha256", "native-current"),
    ],
)
def test_manuelles_backup_bindet_validierte_current_identitaet_explizit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    manifest_field: str,
    release_id: str,
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode=mode,
        bootstrap_python=(str(REAL_PYTHON) if mode == "native" else None),
    )
    _state(document, native=True)
    if mode == "container":
        _state(document, native=False)
    monkeypatch.setenv("RISK_LAYER_RELEASE_ID", "abweichender-launcher")
    monkeypatch.setenv("RISK_LAYER_RELEASE_GIT_SHA", "1" * 40)
    monkeypatch.setenv("RISK_LAYER_RELEASE_MANIFEST_SHA256", "2" * 64)
    backup_calls: list[list[str]] = []

    def run(command, **kwargs):
        is_status = (
            "deploy-container" in command and "status" in command
        ) or ("release" in command and "status" in command)
        if is_status:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "ok": True,
                        "state": {
                            "current": {
                                "release_id": release_id,
                                "git_sha": "a" * 40,
                                manifest_field: "b" * 64,
                            }
                        },
                    }
                ),
                "",
            )
        backup_calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "{}", "")

    _patch_delegate_run(monkeypatch, run)
    assert O.main(["--config", str(profile_path), "backup"]) == 0

    assert len(backup_calls) == 1
    command = backup_calls[0]
    assert command[command.index("--release-id") + 1] == release_id
    assert command[command.index("--git-sha") + 1] == "a" * 40
    assert command[command.index("--release-manifest-sha256") + 1] == "b" * 64


def test_container_backup_blockt_unvollstaendige_current_identitaet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, document = _write_profile(tmp_path)
    _state(document, native=True)
    _state(document, native=False)
    backup_calls: list[list[str]] = []

    def run(command, **kwargs):
        if "deploy-container" in command and "status" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {"ok": True, "state": {"current": {"release_id": "v1"}}}
                ),
                "",
            )
        backup_calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    _patch_delegate_run(monkeypatch, run)
    assert O.main(["--config", str(profile_path), "backup"]) == 2
    assert backup_calls == []
    assert "vollstaendige validierte current-Identitaet" in capsys.readouterr().err


def test_backup_prune_dry_run_ist_read_only_apply_braucht_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    backup_dir = Path(str(document["backup_dir"]))
    real_managed = O._require_managed_directory

    def managed(path: Path, description: str) -> None:
        if path == backup_dir:
            raise O.OperatorError("Backup-Verzeichnis ist absichtlich read-only")
        real_managed(path, description)

    calls: list[list[str]] = []
    monkeypatch.setattr(O, "_require_managed_directory", managed)
    _patch_delegate_run(
        monkeypatch,
        lambda command, **kwargs: (
            calls.append(list(command))
            or subprocess.CompletedProcess(command, 0, "", "")
        ),
    )

    assert O.main(["--config", str(profile_path), "backup-prune"]) == 0
    assert len(calls) == 1 and "--apply" not in calls[0]
    assert O.main(["--config", str(profile_path), "backup-prune", "--apply"]) == 2
    assert len(calls) == 1
    assert "read-only" in capsys.readouterr().err


def test_parameterloses_backup_verify_prueft_das_neueste_archiv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    backup_dir = Path(str(document["backup_dir"]))
    old = backup_dir / "festwissen_alt.zip"
    latest = backup_dir / "festwissen_neu.zip"
    old.write_bytes(b"alt")
    latest.write_bytes(b"neu")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    os.utime(latest, ns=(2_000_000_000, 2_000_000_000))
    calls: list[list[str]] = []
    _patch_delegate_run(
        monkeypatch,
        lambda command, **kwargs: (
            calls.append(list(command))
            or subprocess.CompletedProcess(command, 0, "{}", "")
        ),
    )

    assert O.main(["--config", str(profile_path), "backup-verify"]) == 0

    assert len(calls) == 1
    assert calls[0][-3:] == ["backup", "verify", str(latest)]


def test_restore_list_ist_rein_lesend_und_braucht_keinen_delegate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, document = _write_profile(tmp_path)
    archive = Path(str(document["backup_dir"])) / "festwissen_2026.zip"
    archive.write_bytes(b"zip")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    _patch_delegate_run(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("restore --list darf keinen Delegate starten"),
    )

    assert O.main(["--config", str(profile_path), "restore"]) == 0

    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert json.loads(capsys.readouterr().out)["backups"][0]["name"] == archive.name


def test_restore_liste_und_leeres_ziel_brauchen_kein_schreibbares_backup_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    backup_dir = Path(str(document["backup_dir"]))
    archive = backup_dir / "festwissen_read_only.zip"
    archive.write_bytes(b"zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    confirmation = (
        f"RESTORE {archive.name} SHA256 {digest} TO {document['berater_dir']}"
    )
    real_managed = O._require_managed_directory

    def managed(path: Path, description: str) -> None:
        if path == backup_dir:
            raise O.OperatorError("Backup-Verzeichnis ist absichtlich read-only")
        real_managed(path, description)

    calls: list[list[str]] = []
    monkeypatch.setattr(O, "_require_managed_directory", managed)
    _patch_delegate_run(
        monkeypatch,
        lambda command, **kwargs: (
            calls.append(list(command))
            or subprocess.CompletedProcess(command, 0, "{}", "")
        ),
    )

    assert O.main(["--config", str(profile_path), "restore", "--list"]) == 0
    capsys.readouterr()
    assert O.main(
        [
            "--config",
            str(profile_path),
            "restore",
            "--file",
            str(archive),
            "--confirm",
            confirmation,
        ]
    ) == 0
    assert len(calls) == 1


def test_restore_verlangt_an_archiv_und_ziel_gebundene_bestaetigung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, document = _write_profile(
        tmp_path,
        mode="native",
        bootstrap_python=str(REAL_PYTHON),
    )
    archive = Path(str(document["backup_dir"])) / "festwissen_neu.zip"
    archive.write_bytes(b"zip")
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    _patch_delegate_run(monkeypatch, run)
    assert O.main(["--config", str(profile_path), "restore", "--latest"]) == 2
    assert calls == []
    assert "Restore-Bestaetigung" in capsys.readouterr().err

    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    confirmation = (
        f"RESTORE {archive.name} SHA256 {archive_hash} TO {document['berater_dir']}"
    )
    assert O.main(
        [
            "--config",
            str(profile_path),
            "restore",
            "--latest",
            "--confirm",
            confirmation,
        ]
    ) == 0
    assert calls and calls[0][-1] == "--json"
    assert calls[0][calls[0].index("--expected-sha256") + 1] == archive_hash
    assert ["--backup-ziel", str(document["backup_dir"])] == calls[0][
        calls[0].index("--backup-ziel") : calls[0].index("--backup-ziel") + 2
    ]


def test_container_restore_bindet_vorher_sicherung_an_container_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path, document = _write_profile(tmp_path)
    _state(document, native=True)
    _state(document, native=False)
    archive = Path(str(document["backup_dir"])) / "festwissen_restore.zip"
    archive.write_bytes(b"zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    confirmation = (
        f"RESTORE {archive.name} SHA256 {digest} TO {document['berater_dir']}"
    )
    restore_calls: list[list[str]] = []

    def run(command, **kwargs):
        if "deploy-container" in command and "status" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "ok": True,
                        "state": {
                            "current": {
                                "release_id": "container-current",
                                "git_sha": "a" * 40,
                                "release_manifest_sha256": "b" * 64,
                            }
                        },
                    }
                ),
                "",
            )
        restore_calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setenv("RISK_LAYER_RELEASE_ID", "abweichender-launcher")
    _patch_delegate_run(monkeypatch, run)
    assert O.main(
        [
            "--config",
            str(profile_path),
            "restore",
            "--file",
            str(archive),
            "--confirm",
            confirmation,
        ]
    ) == 0

    assert len(restore_calls) == 1
    command = restore_calls[0]
    assert command[command.index("--release-id") + 1] == "container-current"
    assert command[command.index("--git-sha") + 1] == "a" * 40
    assert command[command.index("--release-manifest-sha256") + 1] == "b" * 64


def test_status_offline_aggregiert_nur_deployment_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, _ = _write_profile(tmp_path)
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"ok": True, "state": {"current": None}}),
            "",
        )

    _patch_delegate_run(monkeypatch, run)
    assert O.main(["--config", str(profile_path), "status", "--offline"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["health"] == {"ok": None, "skipped": "offline"}
    assert result["ok"] is False
    assert result["identity"]["complete"] is False
    assert len(calls) == 1 and "deploy-container" in calls[0]


def test_container_status_ist_unabhaengig_von_backup_und_write_rechten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, document = _write_profile(tmp_path)
    backup_dir = Path(str(document["backup_dir"]))
    (backup_dir / O.BACKUP_TARGET_FILENAME).unlink()
    backup_dir.rmdir()

    def no_managed(*_args) -> None:
        raise AssertionError("status darf keine Managed-/Write-Pruefung ausfuehren")

    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "ok": True,
                    "state": {
                        "current": {
                            "release_id": "v1",
                            "git_sha": "a" * 40,
                            "release_manifest_sha256": "b" * 64,
                        }
                    },
                }
            ),
            "",
        )

    monkeypatch.setattr(O, "_require_managed_directory", no_managed)
    _patch_delegate_run(monkeypatch, run)
    assert O.main(["--config", str(profile_path), "status", "--offline"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_doctor_bleibt_bei_fehlendem_backup_ziel_rot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, document = _write_profile(tmp_path)
    backup_dir = Path(str(document["backup_dir"]))
    (backup_dir / O.BACKUP_TARGET_FILENAME).unlink()
    backup_dir.rmdir()

    assert O.main(["--config", str(profile_path), "doctor"]) == 1
    result = json.loads(capsys.readouterr().out)
    backup = next(item for item in result["checks"] if item["name"] == "backup_dir")
    assert backup["ok"] is False


@pytest.mark.parametrize(
    "inner_ok,valid_identity,expected",
    [(True, True, 0), (False, True, 1), (True, False, 1)],
)
def test_status_braucht_vollstaendige_und_innerlich_gruene_release_identitaet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    inner_ok: bool,
    valid_identity: bool,
    expected: int,
) -> None:
    profile_path, _ = _write_profile(tmp_path)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "ok": inner_ok,
                    "state": {
                        "current": {
                            "release_id": "v1.2.3" if valid_identity else "../bogus",
                            "git_sha": "a" * 40 if valid_identity else "bogus",
                            "release_manifest_sha256": "b" * 64,
                        }
                    },
                }
            ),
            "",
        )

    _patch_delegate_run(monkeypatch, run)
    assert O.main(["--config", str(profile_path), "status", "--offline"]) == expected
    result = json.loads(capsys.readouterr().out)
    expected_ok = inner_ok and valid_identity
    assert result["ok"] is expected_ok
    assert result["identity"]["complete"] is expected_ok


def test_container_doctor_ist_ohne_nativen_launcher_state_nicht_gruen(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path, _ = _write_profile(tmp_path)
    assert O.main(["--config", str(profile_path), "doctor"]) == 1
    result = json.loads(capsys.readouterr().out)
    check = next(item for item in result["checks"] if item["name"] == "native_release_state")
    assert check["ok"] is False


def test_wrapper_sind_duenn_isoliert_und_paritaetisch() -> None:
    bash = (ROOT / "signal").read_text(encoding="utf-8")
    powershell = (ROOT / "signal.ps1").read_text(encoding="utf-8")

    assert "risk_layer/operator_cli.py" in bash
    assert '"$PYTHON" -I' in bash
    assert '"$@"' in bash
    assert "risk_layer\\operator_cli.py" in powershell
    assert "& $python -I" in powershell
    assert "@OperatorArgs" in powershell
    assert "deploy/risk-layer-ops.json" in bash
    assert "deploy\\risk-layer-ops.json" in powershell
    assert "RISK_LAYER_CHECKOUT_ROOT" in bash
    assert "RISK_LAYER_CHECKOUT_ROOT" in powershell
    assert 'elif [[ -f "$OPERATOR" ]]' in bash
    assert "elseif (Test-Path -LiteralPath $operator -PathType Leaf)" in powershell
    assert '$ROOT/.venv/bin/python' in bash
    assert "python3.12" in bash
    assert "python3" in bash
    assert '.venv\\Scripts\\python.exe' in powershell
    assert "Select-Object -First 1" in powershell


def test_plattform_wrapper_startet_die_echte_operator_hilfe() -> None:
    environment = os.environ.copy()
    environment.pop("RISK_LAYER_OPERATOR_PYTHON", None)
    if os.name == "nt":
        command = [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "signal.ps1"),
            "--help",
        ]
    else:
        command = ["bash", str(ROOT / "signal"), "--help"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "bootstrap" in result.stdout


def test_veraltete_root_einstiegspunkte_sind_entfernt() -> None:
    for name in (
        "risk-layerctl",
        "risk-layerctl.ps1",
        "deploy.sh",
        "deploy.ps1",
        "build-release-artifact.sh",
        "build-release-artifact.ps1",
        "install-service.ps1",
        ".env.release.example",
        "start.cmd",
    ):
        assert not (ROOT / name).exists(), name


def test_beispielprofil_hat_restriktives_schema_und_keine_secrets() -> None:
    example = json.loads(
        (ROOT / "deploy" / "risk-layer-ops.example.json").read_text(encoding="utf-8")
    )
    assert set(example) == O.PROFILE_KEYS
    assert example["schema"] == O.PROFILE_SCHEMA
    assert example["mode"] == "container"
    assert example["bootstrap_python"] is None
    assert all(
        PurePosixPath(str(example[field])).is_absolute()
        for field in O.CONTAINER_PATH_FIELDS
    )
    serialized = json.dumps(example).casefold()
    assert "replace_with_at_least" not in serialized
    assert "risk_layer_token=" not in serialized


def test_interpretervertrag_erkennt_cpython_und_lehnt_dummy_ab(tmp_path: Path) -> None:
    detail = O._interpreter_contract(REAL_PYTHON, "Test-Python")
    assert detail.startswith("CPython 3.")

    dummy = tmp_path / ("dummy-python.exe" if os.name == "nt" else "dummy-python")
    dummy.write_text("keine ausfuehrbare Python-Datei\n", encoding="utf-8")
    if os.name != "nt":
        dummy.chmod(0o700)
    with pytest.raises(O.OperatorError, match="ausfuehrbar|Delegate"):
        O._interpreter_contract(dummy, "Dummy-Python")


@pytest.mark.skipif(os.name == "nt", reason="POSIX-Symlinkvertrag")
def test_interpretervertrag_blockt_posix_control_symlink(tmp_path: Path) -> None:
    linked_python = tmp_path / "python-link"
    linked_python.symlink_to(REAL_PYTHON)

    with pytest.raises(O.OperatorError, match="Link/Reparse"):
        O._interpreter_contract(linked_python, "Test-Python")


def test_control_plane_blockt_fremd_schreibbare_datei_und_elternpfad(
    tmp_path: Path,
) -> None:
    profile_path, _ = _write_profile(tmp_path)
    if os.name == "nt":
        result = subprocess.run(
            ["icacls.exe", str(profile_path), "/grant", "*S-1-1-0:(M)"],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    else:
        profile_path.chmod(0o666)
    with pytest.raises(O.OperatorError, match="schreibbar|schreib|Rechte"):
        O.load_profile(profile_path)

    nested = tmp_path / "parent-check"
    nested.mkdir()
    profile_path, _ = _write_profile(nested)
    parent = profile_path.parent
    if os.name == "nt":
        result = subprocess.run(
            ["icacls.exe", str(parent), "/grant", "*S-1-1-0:(OI)(CI)(M)"],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    else:
        parent.chmod(0o777)
    with pytest.raises(O.OperatorError, match="Elternverzeichnis|schreibbar|schreib"):
        O.load_profile(profile_path)


def test_windows_systemverzeichnis_kommt_nicht_aus_der_umgebung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows-spezifische Systempfad-Pruefung")
    monkeypatch.setenv("SystemRoot", r"C:\angreifer")
    executable = O._windows_powershell()
    assert "angreifer" not in str(executable).casefold()
    assert executable.is_file()


def test_secret_env_roundtrip_haelt_windows_pfade_literal_und_blockt_casefold_duplikate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private" / "secret.env"
    path.parent.mkdir(parents=True)
    windows_path = r"C:\new folder\definitionen"
    path.write_text(
        f"RISK_LAYER_TOKEN={TOKEN}\n"
        f"RISK_LAYER_DEFINITIONEN_DIR={O._dotenv_value(windows_path)}\n",
        encoding="utf-8",
    )
    _private(path)
    assert O._parse_secret_env(path)["RISK_LAYER_DEFINITIONEN_DIR"] == windows_path
    assert D._parse_secret_env(path)["RISK_LAYER_DEFINITIONEN_DIR"] == windows_path

    path.write_text(
        f"RISK_LAYER_TOKEN={TOKEN}\nRISK_LAYER_token={'u' * 48}\n",
        encoding="utf-8",
    )
    _private(path)
    with pytest.raises(O.OperatorError, match="doppelt"):
        O._parse_secret_env(path)
    with pytest.raises(D.DeployError, match="doppelt"):
        D._parse_secret_env(path)


def test_private_dateien_duerfen_unter_posix_nicht_zu_weit_sein(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX-Modusbits sind unter Windows nicht verbindlich")
    secret = tmp_path / "secret"
    secret.write_text(TOKEN, encoding="utf-8")
    secret.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH)
    with pytest.raises(O.OperatorError, match="Modus 0600"):
        O._require_file(secret, "Secret", private=True)


def test_control_plane_blockt_posix_gruppen_und_welt_schreibrechte(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX-Modusbits sind unter Windows nicht verbindlich")
    control_file = tmp_path / "control.py"
    control_file.write_text("pass\n", encoding="utf-8")
    control_file.chmod(0o620)
    with pytest.raises(O.OperatorError, match="gruppen-/welt-schreibbar"):
        O._require_control_file(control_file, "Control-Datei")

    control_dir = tmp_path / "control-dir"
    control_dir.mkdir()
    control_dir.chmod(0o702)
    with pytest.raises(O.OperatorError, match="gruppen-/welt-schreibbar"):
        O._require_control_directory(control_dir, "Control-Verzeichnis")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "3600.1"])
def test_operator_timeout_ist_endlich_positiv_und_begrenzt(
    tmp_path: Path,
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path, _ = _write_profile(tmp_path, prepare=False)
    assert O.main(
        ["--config", str(profile_path), "status", "--offline", f"--timeout={value}"]
    ) == 2
    assert "--timeout" in capsys.readouterr().err


def test_operator_timeout_huge_int_wird_als_domaenenfehler_normalisiert() -> None:
    with pytest.raises(O.OperatorError, match="Timeout"):
        O._positive_timeout(10**10000, "Timeout")
