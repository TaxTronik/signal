import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import risk_layer.web as web
from risk_layer.web import Anwendung, make_handler


BEARER = "b" * 48
OPERATOR = "o" * 48


def test_sigterm_nutzt_den_graceful_keyboardinterrupt_pfad():
    with pytest.raises(KeyboardInterrupt):
        web._sigterm_als_graceful_stop(15, None)


def _starte(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _, port = httpd.server_address
    return httpd, f"http://127.0.0.1:{port}"


def _request(url, *, method="GET", body=None, bearer=None, operator=None):
    headers = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if bearer is not None:
        headers["Authorization"] = f"Bearer {bearer}"
    if operator is not None:
        headers["x-risk-layer-operator-token"] = operator
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_get_embedding_status_braucht_nur_normalen_bearer(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", BEARER)
    monkeypatch.delenv("RISK_LAYER_OPERATOR_TOKEN", raising=False)
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "missing.sqlite"))
    httpd, base = _starte(app)
    try:
        assert _request(f"{base}/v1/embedding/status")[0] == 401
        status, body = _request(f"{base}/v1/embedding/status", bearer=BEARER)
        assert status == 200
        assert body["ok"] is True and body["engineVersion"] == "1.4.0"
        assert body["index"]["active"] is False
        assert body["index"]["current"] is None
        assert body["job"]["state"] == "idle"
        assert body["schedule"]["enabled"] is False
        assert body["refresh_available"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_operator_posts_sind_zweifach_authentisiert(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", BEARER)
    monkeypatch.setenv("RISK_LAYER_OPERATOR_TOKEN", OPERATOR)
    monkeypatch.setenv("RISK_LAYER_EMBEDDING_DIR", str(tmp_path / "overlay"))
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "missing.sqlite"))
    monkeypatch.setattr(
        app.embedding_operator,
        "trigger",
        lambda *, force: {"job_id": "a" * 32, "state": "queued"},
    )
    httpd, base = _starte(app)
    payload = b'{"force":true}'
    try:
        assert (
            _request(
                f"{base}/v1/embedding/refresh",
                method="POST",
                body=payload,
                operator=OPERATOR,
            )[0]
            == 401
        )
        assert (
            _request(
                f"{base}/v1/embedding/refresh",
                method="POST",
                body=payload,
                bearer=BEARER,
            )[0]
            == 403
        )
        assert (
            _request(
                f"{base}/v1/embedding/refresh",
                method="POST",
                body=payload,
                bearer=BEARER,
                operator="falsch",
            )[0]
            == 403
        )
        status, body = _request(
            f"{base}/v1/embedding/refresh",
            method="POST",
            body=payload,
            bearer=BEARER,
            operator=OPERATOR,
        )
        assert status == 202
        assert body["job_id"] == "a" * 32 and body["state"] == "queued"
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_fehlende_operator_konfiguration_bleibt_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", BEARER)
    monkeypatch.delenv("RISK_LAYER_OPERATOR_TOKEN", raising=False)
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "missing.sqlite"))
    httpd, base = _starte(app)
    try:
        status, body = _request(
            f"{base}/v1/embedding/refresh",
            method="POST",
            body=b'{"force":false}',
            bearer=BEARER,
            operator=OPERATOR,
        )
        assert status == 503
        assert body == {"error": "embedding operator unavailable"}
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_unverfuegbarer_operator_aktiviert_keinen_defaultplan(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", BEARER)
    monkeypatch.setenv("RISK_LAYER_OPERATOR_TOKEN", OPERATOR)
    monkeypatch.setenv("RISK_LAYER_EMBEDDING_DIR", str(tmp_path / "overlay"))
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "missing.sqlite"))
    httpd, base = _starte(app)
    try:
        status, body = _request(f"{base}/v1/embedding/status", bearer=BEARER)
        assert status == 200
        assert body["refresh_available"] is False
        assert body["schedule"]["enabled"] is False
        assert body["schedule"]["next_run_at"] is None
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_schedule_schema_und_grenzen(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", BEARER)
    monkeypatch.setenv("RISK_LAYER_OPERATOR_TOKEN", OPERATOR)
    monkeypatch.setenv("RISK_LAYER_EMBEDDING_DIR", str(tmp_path / "overlay"))
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "missing.sqlite"))
    monkeypatch.setattr(
        app.embedding_operator,
        "set_schedule",
        lambda *, enabled, interval_days: {
            "enabled": enabled,
            "interval_days": interval_days,
            "last_check_at": None,
            "next_run_at": None,
        },
    )
    httpd, base = _starte(app)
    try:
        for invalid in (
            b'{"enabled":"yes","interval_days":7}',
            b'{"enabled":true,"interval_days":true}',
            b'{"enabled":true,"interval_days":0}',
            b'{"enabled":true,"interval_days":366}',
        ):
            assert (
                _request(
                    f"{base}/v1/embedding/schedule",
                    method="POST",
                    body=invalid,
                    bearer=BEARER,
                    operator=OPERATOR,
                )[0]
                == 400
            )
        status, body = _request(
            f"{base}/v1/embedding/schedule",
            method="POST",
            body=b'{"enabled":true,"interval_days":7}',
            bearer=BEARER,
            operator=OPERATOR,
        )
        assert status == 200
        assert body["schedule"]["enabled"] is True
        assert body["schedule"]["interval_days"] == 7
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_zu_kurzes_operator_secret_verweigert_serverstart(monkeypatch, capsys):
    monkeypatch.delenv("RISK_LAYER_TOKEN", raising=False)
    monkeypatch.setenv("RISK_LAYER_OPERATOR_TOKEN", "zu-kurz")
    assert web.serve(host="127.0.0.1", port=1) == 2
    assert "RISK_LAYER_OPERATOR_TOKEN" in capsys.readouterr().err


def test_operator_secret_muss_von_beiden_bearern_verschieden_sein(
    tmp_path, monkeypatch, capsys
):
    current = "c" * 48
    previous = "p" * 48
    monkeypatch.setenv("RISK_LAYER_TOKEN", current)
    monkeypatch.setenv("RISK_LAYER_TOKEN_VORGAENGER", previous)
    for duplicate in (current, previous):
        monkeypatch.setenv("RISK_LAYER_OPERATOR_TOKEN", duplicate)
        assert web.serve(host="127.0.0.1", port=1) == 2
        assert "verschieden" in capsys.readouterr().err

    # Auch make_handler/Anwendung bleiben bei umgangener serve()-Validierung
    # fail-closed; der duplizierte Header aktiviert keinen Operatorplan.
    monkeypatch.setenv("RISK_LAYER_OPERATOR_TOKEN", current)
    monkeypatch.setenv("RISK_LAYER_EMBEDDING_DIR", str(tmp_path / "overlay"))
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "missing.sqlite"))
    httpd, base = _starte(app)
    try:
        status, body = _request(
            f"{base}/v1/embedding/refresh",
            method="POST",
            body=b'{"force":true}',
            bearer=current,
            operator=current,
        )
        assert status == 503
        assert body == {"error": "embedding operator unavailable"}
        status, body = _request(f"{base}/v1/embedding/status", bearer=current)
        assert status == 200
        assert body["refresh_available"] is False
        assert body["schedule"]["enabled"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_operator_secret_ohne_normalen_bearer_ist_unzulaessig(monkeypatch, capsys):
    monkeypatch.delenv("RISK_LAYER_TOKEN", raising=False)
    monkeypatch.delenv("RISK_LAYER_TOKEN_VORGAENGER", raising=False)
    monkeypatch.setenv("RISK_LAYER_OPERATOR_TOKEN", OPERATOR)
    assert web.serve(host="127.0.0.1", port=1) == 2
    assert "beide Vertrauensgrenzen" in capsys.readouterr().err
