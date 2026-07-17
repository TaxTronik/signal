"""Statische Sicherheitsinvarianten der nativen Service-Installationen."""

import base64
import ctypes
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _render_backup_task_child(
    *, python: Path, release_tools: Path, profile: Path
) -> str:
    installer = (ROOT / "deploy" / "install-backup-task.ps1").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r'^\$taskCommand = @"\r?\n(?P<body>.*?)\r?\n"@$',
        installer,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    child = match.group("body")
    for placeholder, path in (
        ("pythonB64", python),
        ("toolsB64", release_tools),
        ("profileB64", profile),
    ):
        encoded = base64.b64encode(str(path).encode("utf-8")).decode("ascii")
        child = child.replace(f"${placeholder}", encoded)
    child = child.replace("`$", "$")
    assert "$pythonB64" not in child
    assert "$toolsB64" not in child
    assert "$profileB64" not in child
    return child


def _system_windows_powershell() -> Path:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_system_directory = kernel32.GetSystemDirectoryW
    get_system_directory.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    get_system_directory.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32_768)
    length = get_system_directory(buffer, len(buffer))
    assert 0 < length < len(buffer)
    executable = (
        Path(buffer.value) / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    assert executable.is_file()
    return executable


def _run_backup_task_child(
    *, python: Path, release_tools: Path, profile: Path
) -> subprocess.CompletedProcess[str]:
    child = _render_backup_task_child(
        python=python,
        release_tools=release_tools,
        profile=profile,
    )
    encoded = base64.b64encode(child.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        [
            str(_system_windows_powershell()),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


def test_windows_autostart_ist_ausschliesslich_begrenzter_benutzer_task() -> None:
    script = (ROOT / "deploy" / "install-service.ps1").read_text(encoding="utf-8")

    assert '/RU SYSTEM' not in script
    assert '/RL HIGHEST' not in script
    assert '-LogonType Interactive -RunLevel Limited' in script
    assert "LocalService" not in script
    assert "NSSM" not in script
    assert "$RunAs" not in script
    assert "SERVICE_AUTO_START" not in script
    assert 'AppEnvironmentExtra "RISK_LAYER_TOKEN=' not in script
    assert "Remove-RiskLayerTask" in script
    assert "service-start.ps1" in script
    assert "[System.Text.Encoding]::Unicode" in script


def test_windows_autostart_wrapper_verweigert_ungueltiges_token() -> None:
    script = (ROOT / "deploy" / "install-service.ps1").read_text(encoding="utf-8")

    assert "[System.IO.File]::ReadAllText($tokenPath)" in script
    assert "$token.Length -lt 32" in script
    assert '$token -match "\\s"' in script
    assert '$env:RISK_LAYER_TOKEN = $token' in script
    assert script.index("$token.Length -lt 32") < script.index(
        "$env:RISK_LAYER_TOKEN = $token"
    )
    assert 'exit 20' in script


def test_systemd_unit_verweigert_root_und_begrenzt_rechte() -> None:
    unit = (ROOT / "deploy" / "risk-layer.service.in").read_text(encoding="utf-8")

    assert 'ExecStartPre=/bin/sh -c' in unit
    assert '"$(id -u)" -eq 0' in unit
    for directive in (
        "NoNewPrivileges=true",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "PrivateTmp=true",
        "ProtectSystem=full",
        "ProtectKernelTunables=true",
        "ProtectKernelModules=true",
        "ProtectControlGroups=true",
        "RestrictSUIDSGID=true",
        "UMask=0077",
    ):
        assert directive in unit
    assert "WantedBy=default.target" in unit
    assert "WantedBy=multi-user.target" not in unit


def test_systemd_renderer_maskiert_pfade_und_specifier() -> None:
    script = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
    unit = (ROOT / "deploy" / "risk-layer.service.in").read_text(encoding="utf-8")

    assert "def systemd_escape" in script
    assert '.replace("%", "%%")' in script
    assert '.replace(" ", "\\\\x20")' in script
    assert "Steuerzeichen sind nicht erlaubt" in script
    assert 's|@WORKDIR@' not in script
    assert "WorkingDirectory=@WORKDIR@" in unit
    assert "EnvironmentFile=@ENVFILE@" in unit
    assert 'ExecStart="@PYTHON@" -P -m risk_layer.web' in unit


def test_native_dienste_importieren_nicht_aus_dem_checkout() -> None:
    unit = (ROOT / "deploy" / "risk-layer.service.in").read_text(encoding="utf-8")
    windows = (ROOT / "deploy" / "install-service.ps1").read_text(encoding="utf-8")

    assert 'ExecStart="@PYTHON@" -P -m risk_layer.web' in unit
    assert '& $python -P -m risk_layer.web' in windows


def test_release_systemd_hat_sicheren_renderer_und_readonly_release_home() -> None:
    unit = (ROOT / "deploy" / "risk-layer-release.service.in").read_text(
        encoding="utf-8"
    )
    installer = (ROOT / "deploy" / "install-release-service.py").read_text(
        encoding="utf-8"
    )

    assert 'ReadOnlyPaths="@RELEASE_HOME@" "@LAUNCHER@" "@ENVFILE@"' in unit
    assert '"@LAUNCHER@"' in unit
    assert "-- service run --home" in unit
    assert "systemd-analyze" in installer
    assert '["systemctl", "--user", "daemon-reload"]' in installer
    assert "os.replace(temporary_name, output)" in installer
    assert '.replace("%", "%%")' in installer
    assert "shell=True" not in installer


def test_windows_release_task_startet_verifiziertes_current() -> None:
    installer = (ROOT / "deploy" / "install-service.ps1").read_text(encoding="utf-8")
    restart = (ROOT / "deploy" / "restart-risk-layer-task.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string]$ReleaseHome" in installer
    assert "[string]$BeraterDir" in installer
    assert "[string]$ReleaseLauncher" in installer
    assert "release-launcher.py" in installer
    assert "-- service run" in installer
    assert "ToBase64String" in installer and "FromBase64String" in installer
    assert "Start-ScheduledTask" in restart
    assert "Stop-ScheduledTask" in restart
    assert "RunLevel Highest" not in restart


def test_backup_scheduler_nutzen_current_runtime_und_sichere_argumente() -> None:
    unit = (ROOT / "deploy" / "risk-layer-backup.service").read_text(
        encoding="utf-8"
    )
    windows = (ROOT / "deploy" / "install-backup-task.ps1").read_text(
        encoding="utf-8"
    )

    assert "/usr/bin/python3.12 -I ${RISK_LAYER_RELEASE_TOOLS}" in unit
    assert "ops --config ${RISK_LAYER_OPS_PROFILE} backup" in unit
    assert "[string]$ReleaseTools" in windows
    assert "[string]$OperatorProfile" in windows
    assert "[string]$BeraterDir" not in windows
    assert "ToBase64String" in windows and "FromBase64String" in windows
    assert "& Invoke-Expression" not in windows
    assert "ops --config `$profile backup" in windows
    assert "[Environment]::SystemDirectory" in windows
    assert "'WindowsPowerShell'" in windows
    assert "'v1.0'" in windows
    assert "'powershell.exe'" in windows
    assert "$PSHOME" not in windows
    assert "Test-Path -LiteralPath $powershell -PathType Leaf" in windows
    assert "New-ScheduledTaskAction -Execute $powershell" in windows
    assert "`$ErrorActionPreference = 'Stop'" in windows
    assert "`$LASTEXITCODE = `$null" in windows
    assert "if (`$null -eq `$childExit)" in windows
    assert "exit [int]`$childExit" in windows
    assert "catch {" in windows
    assert "exit 1" in windows
    assert "-LogonType Interactive" in windows
    assert "-RunLevel Limited" in windows


def test_backup_scheduler_child_ist_fail_closed_und_propagiert_native_exitcodes(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows-spezifischer Scheduled-Task-Vertrag")

    release_tools = tmp_path / "release-tools.py"
    profile = tmp_path / "ops.json"
    release_tools.write_text("raise SystemExit(37)\n", encoding="utf-8")
    profile.write_text("{}\n", encoding="utf-8")

    missing = _run_backup_task_child(
        python=tmp_path / "missing-python.exe",
        release_tools=release_tools,
        profile=profile,
    )
    assert missing.returncode == 1, (missing.stdout, missing.stderr)
    assert "Python-Interpreter fehlt zur Laufzeit" in missing.stderr

    unstartable_python = tmp_path / "invalid-python.exe"
    unstartable_python.write_bytes(b"not a Windows executable")
    unstartable = _run_backup_task_child(
        python=unstartable_python,
        release_tools=release_tools,
        profile=profile,
    )
    assert unstartable.returncode == 1, (unstartable.stdout, unstartable.stderr)

    no_native_exit = tmp_path / "no-native-exit.ps1"
    no_native_exit.write_text(
        "param([switch]$I, "
        "[Parameter(ValueFromRemainingArguments=$true)][object[]]$Remaining)\n"
        "return\n",
        encoding="utf-8",
    )
    null_exit = _run_backup_task_child(
        python=no_native_exit,
        release_tools=release_tools,
        profile=profile,
    )
    assert null_exit.returncode == 1, (null_exit.stdout, null_exit.stderr)
    assert "Python-Prozess lieferte keinen externen Exitcode" in null_exit.stderr

    propagated = _run_backup_task_child(
        python=Path(sys.executable),
        release_tools=release_tools,
        profile=profile,
    )
    assert propagated.returncode == 37, (propagated.stdout, propagated.stderr)


def test_force_rotiert_auth_token_nicht_still() -> None:
    """Artefakte neu bauen und Client-Credentials rotieren sind getrennte
    Betriebsereignisse; ein Update darf nicht nebenbei alle Clients aussperren."""
    linux = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
    windows = (ROOT / "scripts" / "setup.ps1").read_text(encoding="utf-8")

    assert "--rotate-token" in linux
    assert '$ROTATE_TOKEN -eq 1' in linux
    assert '! -f "$TOKEN_FILE" || $FORCE -eq 1' not in linux
    assert "[switch]$RotateToken" in windows
    assert "-or $RotateToken" in windows
    assert ('if ((-not (Test-Path -LiteralPath $tokenPath)) -or $Force)'
            not in windows)


def test_container_enthaelt_lizenzdatei_im_paket_build() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY pyproject.toml README.md LICENSE" in dockerfile
    assert "requirements-runtime-lock.txt" in dockerfile
    assert "requirements-ner-lock.txt" in dockerfile


def test_publish_uebergibt_secrets_sicher_und_veroeffentlicht_nie_latest() -> None:
    workflow = (ROOT / ".forgejo" / "workflows" / "build.yml").read_text(
        encoding="utf-8"
    )

    assert "REGISTRY_TOKEN: ${{ secrets.REGISTRY_TOKEN }}" in workflow
    assert "PUBLISH_TAG: ${{" in workflow
    assert 'printf \'%s\' "$REGISTRY_TOKEN"' in workflow
    assert 'echo "${{ secrets.REGISTRY_TOKEN }}"' not in workflow
    assert 'TAG="${{' not in workflow
    assert "PUBLISH_LATEST" not in workflow
    assert 'docker push "$IMAGE:latest"' not in workflow
    assert "kein latest" in workflow
    assert "startsWith(github.ref, 'refs/tags/v')" in workflow
    assert "release-quality:" in workflow
    assert "release-security:" in workflow
    assert "needs: [validate-deploy, engine-image, release-quality, release-security]" in workflow
    assert 'docker push "$IMAGE:$PUBLISH_TAG" 2>&1 | tee' in workflow
