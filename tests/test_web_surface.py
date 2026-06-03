"""Härtung des schlanken /v1-Servers: Auth + Body-Limit + nur /v1 erreichbar.

Echt-Server-Roundtrip auf ephemerem Loopback-Port.
"""

import http.client
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from risk_layer.web import _MAX_BODY, Anwendung, _ist_loopback, make_handler


def _starte(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
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
    # leerer/None-Host = alle Interfaces, NICHT loopback (sonst ungewollt ans Netz)
    assert not _ist_loopback("") and not _ist_loopback(None)


def test_v1_braucht_korrekten_token(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
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
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        assert _get(f"{base}/v1/health", token="falsch")[0] == 401   # nicht 500
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_nur_v1_erreichbar(tmp_path, monkeypatch):
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        # Alles außer /v1 ist 404 – keine GUI, kein /api, kein static.
        assert _get(f"{base}/api/health", token="geheim")[0] == 404
        assert _get(f"{base}/", token="geheim")[0] == 404
        assert _get(f"{base}/static/app.js", token="geheim")[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_body_groesse_und_form_werden_abgewiesen(tmp_path, monkeypatch):
    # Auth läuft VOR dem Body-Read; mit gültigem Token greifen die Body-Checks.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        _, port = httpd.server_address
        # (1) Übergroße Content-Length → 413, OHNE den Riesen-Body zu übertragen
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.putrequest("POST", "/v1/analyse")
        c.putheader("Authorization", "Bearer geheim")
        c.putheader("Content-Length", str(_MAX_BODY + 1))
        c.endheaders()
        assert c.getresponse().status == 413
        c.close()
        # (2) Kaputter JSON-Body → 400 (klassifiziert, kein 500)
        c2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c2.request("POST", "/v1/analyse", body=b"{kaputt",
                   headers={"Authorization": "Bearer geheim",
                            "Content-Type": "application/json"})
        assert c2.getresponse().status == 400
        c2.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_500_leakt_keinen_exception_text(tmp_path, monkeypatch):
    # Interner Fehler darf weder Mandantentext-Fragmente noch interne Pfade an den
    # Client geben (§ 203 / Info-Leak): generische Meldung + Fehlerklasse, kein exc-Text.
    import json as _json
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    app = Anwendung(None, str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("GEHEIM-Mandantentext-xyz")
    monkeypatch.setattr(app, "analyse", boom)
    httpd, base = _starte(app)
    try:
        req = urllib.request.Request(
            f"{base}/v1/analyse", data=b'{"text":"x"}',
            headers={"Authorization": "Bearer geheim", "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                status, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, body = e.code, e.read()
        assert status == 500
        txt = body.decode("utf-8")
        assert "GEHEIM" not in txt and "Mandantentext" not in txt   # kein Leak
        assert _json.loads(txt)["typ"] == "RuntimeError"            # Fehlerklasse ok
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_embedding_encoder_fehler_degradiert(tmp_path, monkeypatch):
    # Encoder nicht ladbar (z.B. torch fehlt) -> Engine startet trotzdem ohne
    # Schicht 1.7, statt komplett zu crashen.
    import risk_layer.embedding as emb

    class FakeIdx:
        encoder_name = "x"
    monkeypatch.setattr(emb, "index_oeffnen", lambda *a, **k: FakeIdx())

    def boom(*a, **k):
        raise RuntimeError("kein torch")
    monkeypatch.setattr(emb, "encoder_oeffnen", boom)
    app = Anwendung(None, str(tmp_path))
    assert app.embedding is None and app.embedding_encoder is None
