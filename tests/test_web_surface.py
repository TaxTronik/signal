"""Surface-Trennung + Auth + static-Boundary (Code-Review-Sicherheitsfixes).

Echt-Server-Roundtrip auf ephemerem Loopback-Port. Nagelt fest:
  * --api-only bedient NUR das authentifizierte /v1 (GUI + /api sind 404),
  * /v1 verlangt im Token-Modus einen korrekten Bearer-Token,
  * der statische Handler bricht nicht aus seinem Verzeichnis aus.
"""

import http.client
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from risk_layer.web import _MAX_BODY, Anwendung, _ist_loopback, make_handler


def _starte(app, *, api_only: bool):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, api_only=api_only))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _, port = httpd.server_address
    return httpd, f"http://127.0.0.1:{port}"


def _get(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_ist_loopback():
    assert _ist_loopback("127.0.0.1") and _ist_loopback("localhost") and _ist_loopback("::1")
    assert _ist_loopback("127.0.0.5")        # ganzes 127.0.0.0/8
    assert not _ist_loopback("0.0.0.0") and not _ist_loopback("192.168.1.5")
    # leerer/None-Host = alle Interfaces, NICHT loopback (sonst GUI ungewollt ans Netz)
    assert not _ist_loopback("") and not _ist_loopback(None)


def test_api_only_v1_braucht_korrekten_token(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)), api_only=True)
    try:
        assert _get(f"{base}/v1/health")[0] == 401             # ohne Token
        assert _get(f"{base}/v1/health", token="falsch")[0] == 401
        assert _get(f"{base}/v1/health", token="geheim")[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_auth_token_mit_nicht_ascii_kein_500(tmp_path, monkeypatch):
    # Nicht-ASCII-Token im Env: compare_digest auf str hätte TypeError → 500 geworfen.
    # Der Byte-Vergleich macht daraus ein sauberes 401 (falscher Token), kein Crash.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "gehäim-sécret")
    httpd, base = _starte(Anwendung(None, str(tmp_path)), api_only=True)
    try:
        assert _get(f"{base}/v1/health", token="falsch")[0] == 401   # nicht 500
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_api_only_schaltet_gui_und_api_ab(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)), api_only=True)
    try:
        # Selbst mit gültigem Token sind GUI/-api im api-only-Modus nicht erreichbar:
        # keine §203-Daten-Endpunkte (fall/aufgabe/recherche) ans Netz.
        assert _get(f"{base}/api/health", token="geheim")[0] == 404
        assert _get(f"{base}/", token="geheim")[0] == 404
        assert _get(f"{base}/static/app.js", token="geheim")[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_body_groesse_und_form_werden_abgewiesen(tmp_path, monkeypatch):
    # Body wird vor der Auth gelesen → ein Cap schützt (auch pre-auth) vor Memory-DoS.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)), api_only=True)
    try:
        _, port = httpd.server_address
        # (1) Übergroße Content-Length → 413, OHNE den Riesen-Body zu übertragen
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.putrequest("POST", "/v1/analyse")
        c.putheader("Content-Length", str(_MAX_BODY + 1))
        c.endheaders()
        assert c.getresponse().status == 413
        c.close()
        # (2) Kaputter JSON-Body → 400 (klassifiziert, kein 500)
        c2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c2.request("POST", "/v1/analyse", body=b"{kaputt",
                   headers={"Content-Type": "application/json"})
        assert c2.getresponse().status == 400
        c2.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_static_bleibt_im_verzeichnis(tmp_path, monkeypatch):
    monkeypatch.delenv("RISK_LAYER_TOKEN", raising=False)   # Vollmodus, lokal
    httpd, base = _starte(Anwendung(None, str(tmp_path)), api_only=False)
    try:
        assert _get(f"{base}/static/app.js")[0] == 200       # echte Datei
        assert _get(f"{base}/static/../web.py")[0] == 404     # kein Datei-Leak nach außen
    finally:
        httpd.shutdown()
        httpd.server_close()
