"""Statische Sicherheitsinvarianten der nativen Service-Installationen."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_autostart_ist_ausschliesslich_begrenzter_benutzer_task() -> None:
    script = (ROOT / "install-service.ps1").read_text(encoding="utf-8")

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
    script = (ROOT / "install-service.ps1").read_text(encoding="utf-8")

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
    script = (ROOT / "deploy.sh").read_text(encoding="utf-8")
    unit = (ROOT / "deploy" / "risk-layer.service.in").read_text(encoding="utf-8")

    assert "def systemd_escape" in script
    assert '.replace("%", "%%")' in script
    assert '.replace(" ", "\\\\x20")' in script
    assert "Steuerzeichen sind nicht erlaubt" in script
    assert 's|@WORKDIR@' not in script
    assert "WorkingDirectory=@WORKDIR@" in unit
    assert "EnvironmentFile=@ENVFILE@" in unit
    assert 'ExecStart="@PYTHON@"' in unit


def test_container_enthaelt_lizenzdatei_im_paket_build() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY pyproject.toml README.md LICENSE" in dockerfile
    assert "requirements-runtime-lock.txt" in dockerfile
    assert "requirements-ner-lock.txt" in dockerfile


def test_publish_uebergibt_secrets_und_tag_nur_ueber_umgebung() -> None:
    workflow = (ROOT / ".forgejo" / "workflows" / "build.yml").read_text(
        encoding="utf-8"
    )

    assert "REGISTRY_TOKEN: ${{ secrets.REGISTRY_TOKEN }}" in workflow
    assert "PUBLISH_TAG: ${{" in workflow
    assert "PUBLISH_LATEST: ${{" in workflow
    assert 'printf \'%s\' "$REGISTRY_TOKEN"' in workflow
    assert 'echo "${{ secrets.REGISTRY_TOKEN }}"' not in workflow
    assert 'TAG="${{' not in workflow
    assert 'if [[ "$PUBLISH_LATEST" == "true" ]]' in workflow
    assert workflow.index('if [[ "$PUBLISH_LATEST" == "true" ]]') < workflow.index(
        'docker push "$IMAGE:latest"'
    )
