from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_compose_reicht_operator_secret_und_rw_overlay_durch():
    for name, volume in (
        ("docker-compose.yml", "embedding-state"),
        ("docker-compose.dev.yml", "embedding-state-dev"),
    ):
        compose = _text(name)
        assert 'RISK_LAYER_OPERATOR_TOKEN: "${RISK_LAYER_OPERATOR_TOKEN:-}"' in compose
        assert "RISK_LAYER_EMBEDDING_DIR: /state/embedding" in compose
        assert f"source: {volume}" in compose
        assert "target: /state/embedding" in compose
        assert f"  {volume}:" in compose
        assert "replicas: 1" in compose


def test_container_secret_example_nutzt_getrenntes_secret():
    example = _text("deploy/container-secrets.env.example")
    assert "RISK_LAYER_TOKEN=" in example
    assert "RISK_LAYER_OPERATOR_TOKEN=" in example
    assert "A_DIFFERENT_32_BYTE_SECRET" in example


def test_setup_provisioniert_operator_token_getrennt_und_privat():
    linux = _text("scripts/setup.sh")
    windows = _text("scripts/setup.ps1")
    installer = _text("deploy/install-service.ps1")

    assert 'OPERATOR_TOKEN_FILE=".operator_token"' in linux
    assert 'chmod 600 -- "$OPERATOR_TOKEN_FILE"' in linux
    assert '[[ "$OPERATOR_TOKEN" != "$TOKEN" ]]' in linux
    assert "RISK_LAYER_OPERATOR_TOKEN=%s" in linux
    assert "RISK_LAYER_EMBEDDING_DIR=%s" in linux
    assert '$operatorTokenPath = Join-Path $RepoRoot ".operator_token"' in windows
    assert "Set-EngineTokenAcl -Path $operatorTokenPath" in windows
    assert "$operatorToken -eq $token" in windows
    assert "$env:RISK_LAYER_OPERATOR_TOKEN = $operatorToken" in windows
    assert "$env:RISK_LAYER_EMBEDDING_DIR" in windows
    assert 'Join-Path $PSScriptRoot ".operator_token"' in installer
    assert "$operatorToken -eq $token" in installer
    assert "$env:RISK_LAYER_OPERATOR_TOKEN = $operatorToken" in installer
    assert "$env:RISK_LAYER_EMBEDDING_DIR" in installer


def test_docker_buildkontext_schliesst_beide_engine_secrets_aus():
    dockerignore = _text(".dockerignore")
    assert ".engine_token" in dockerignore.splitlines()
    assert ".operator_token" in dockerignore.splitlines()
    assert ".risk-layer-ops.json" in dockerignore.splitlines()
    assert ".signal/" in dockerignore.splitlines()


def test_lokale_embedding_transaktion_macht_git_arbeitsbaum_nicht_schmutzig():
    gitignore = _text(".gitignore").splitlines()
    for pattern in (
        "/corpus/.embedding.festwissen.lock",
        "/corpus/.embedding.restore-journal.json",
        "/corpus/..embedding.restore-journal.json.*.tmp",
        "/corpus/.embedding.restore-*/",
        "/corpus/.embedding.vor-restore-*/",
        "/corpus/.embedding.failed-*/",
    ):
        assert pattern in gitignore


def test_image_bereitet_volumeziel_fuer_unprivilegierten_prozess_vor():
    dockerfile = _text("Dockerfile")
    assert "/state/embedding" in dockerfile
    assert "USER risklayer:risklayer" in dockerfile


def test_release_runbook_provisioniert_operator_secret_und_overlay():
    runbook = _text("docs/BETRIEB_DEPLOY_UPDATE_BACKUP.md")
    assert "/var/lib/risk-layer/embedding" in runbook
    assert "RISK_LAYER_OPERATOR_TOKEN=%s" in runbook
    assert "RISK_LAYER_EMBEDDING_DIR=%s" in runbook
    assert "~/.config/risk-layer/operator.token" in runbook
    assert 'WriteAllText("$Stable\\.operator_token"' in runbook
    assert "Operator- und Bearer-Token muessen verschieden sein" in runbook
