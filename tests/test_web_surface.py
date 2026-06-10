"""Härtung des schlanken /v1-Servers: Auth + Body-Limit + nur /v1 erreichbar.

Echt-Server-Roundtrip auf ephemerem Loopback-Port.
"""

import http.client
import logging
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import risk_layer.web as web
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


def test_nicht_objekt_body_400_statt_500(tmp_path, monkeypatch):
    # Gültiges JSON, aber kein OBJEKT ('[1]', '"x"') → klarer 400er Client-Fehler,
    # kein AttributeError/500 in den Handlern.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        _, port = httpd.server_address
        for body in (b"[1]", b'"x"', b"42", b"null"):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("POST", "/v1/analyse", body=body,
                      headers={"Authorization": "Bearer geheim",
                               "Content-Type": "application/json"})
            assert c.getresponse().status == 400, body
            c.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_post_ohne_content_length_411(tmp_path, monkeypatch):
    # Fehlende Content-Length hieße still leerer Body → 411 (Length Required);
    # Transfer-Encoding: chunked wird nicht dekodiert → ebenfalls 411.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        _, port = httpd.server_address
        # (1) POST ganz ohne Content-Length
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.putrequest("POST", "/v1/analyse", skip_accept_encoding=True)
        c.putheader("Authorization", "Bearer geheim")
        c.endheaders()
        assert c.getresponse().status == 411
        c.close()
        # (2) Transfer-Encoding: chunked
        c2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c2.putrequest("POST", "/v1/analyse", skip_accept_encoding=True)
        c2.putheader("Authorization", "Bearer geheim")
        c2.putheader("Transfer-Encoding", "chunked")
        c2.endheaders()
        c2.send(b"0\r\n\r\n")
        assert c2.getresponse().status == 411
        c2.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_falsche_methode_405_mit_allow_header(tmp_path, monkeypatch):
    # Bekannter Pfad, falsche Methode → 405 (nicht 404) + Allow-Header.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        status, _ = _get(f"{base}/v1/analyse", token="geheim")     # GET auf POST-Route
        assert status == 405
        req = urllib.request.Request(f"{base}/v1/analyse",
                                     headers={"Authorization": "Bearer geheim"})
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("405 erwartet")
        except urllib.error.HTTPError as e:
            assert e.code == 405 and e.headers.get("Allow") == "POST"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_fehlgeschlagene_auth_wird_geloggt(tmp_path, monkeypatch, caplog):
    # 401er sind sicherheitsrelevant und MÜSSEN im Betreiber-Log sichtbar sein
    # (Quell-IP + Pfad) – ohne Token-Inhalte. Erfolgreiche Requests bleiben still.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        with caplog.at_level(logging.WARNING, logger="risk_layer.web"):
            assert _get(f"{base}/v1/health", token="falscher-token-xyz")[0] == 401
        meldungen = [r.getMessage() for r in caplog.records if "401" in r.getMessage()]
        assert meldungen, "401 muss geloggt werden"
        assert any("/v1/health" in m and "127.0.0.1" in m for m in meldungen)
        assert all("falscher-token-xyz" not in m for m in meldungen)  # kein Token-Leak
    finally:
        httpd.shutdown()
        httpd.server_close()


# -- Token-Rotation (L8): Übergangsfenster mit Vorgänger-Token ----------------

def test_rotationsfenster_akzeptiert_alten_und_neuen_token(tmp_path, monkeypatch):
    # RISK_LAYER_TOKEN_VORGAENGER gesetzt → alter UND neuer Token gelten
    # (beide konstant-zeitig geprüft); jeder dritte Wert bleibt 401.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "neu")
    monkeypatch.setenv("RISK_LAYER_TOKEN_VORGAENGER", "alt")
    monkeypatch.setattr(web, "_rotations_hinweis_geloggt", False)
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        assert _get(f"{base}/v1/health", token="neu")[0] == 200
        assert _get(f"{base}/v1/health", token="alt")[0] == 200
        assert _get(f"{base}/v1/health", token="weder-noch")[0] == 401
        assert _get(f"{base}/v1/health")[0] == 401          # ganz ohne Token: weiter 401
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ohne_vorgaenger_gilt_nur_der_aktuelle_token(tmp_path, monkeypatch):
    # Fenster geschlossen (VORGAENGER ungesetzt bzw. leer) → der alte Token ist tot.
    monkeypatch.setenv("RISK_LAYER_TOKEN", "neu")
    monkeypatch.delenv("RISK_LAYER_TOKEN_VORGAENGER", raising=False)
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        assert _get(f"{base}/v1/health", token="neu")[0] == 200
        assert _get(f"{base}/v1/health", token="alt")[0] == 401
        # leerer String zählt wie ungesetzt — sonst würde das "Leeren" der Env
        # nach der Rotation das Fenster nicht wirklich schließen.
        monkeypatch.setenv("RISK_LAYER_TOKEN_VORGAENGER", "")
        assert _get(f"{base}/v1/health", token="alt")[0] == 401
        assert _get(f"{base}/v1/health", token="neu")[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_vorgaenger_treffer_loggt_einmalig_ohne_tokenwert(tmp_path, monkeypatch, caplog):
    # Treffer auf den Vorgänger-Token → GENAU EIN Info-Log je Prozess
    # ("Rotationsfenster aktiv …"), egal wie viele Requests; Token-Werte
    # erscheinen nie im Log (Secret-Hygiene).
    monkeypatch.setenv("RISK_LAYER_TOKEN", "neu")
    monkeypatch.setenv("RISK_LAYER_TOKEN_VORGAENGER", "altes-geheimnis-xyz")
    monkeypatch.setattr(web, "_rotations_hinweis_geloggt", False)
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        with caplog.at_level(logging.INFO, logger="risk_layer.web"):
            assert _get(f"{base}/v1/health", token="altes-geheimnis-xyz")[0] == 200
            assert _get(f"{base}/v1/health", token="altes-geheimnis-xyz")[0] == 200
            assert _get(f"{base}/v1/health", token="neu")[0] == 200
        rot = [r.getMessage() for r in caplog.records
               if "Rotationsfenster" in r.getMessage()]
        assert len(rot) == 1, "Hinweis muss genau EINMAL je Prozess kommen"
        assert "RISK_LAYER_TOKEN_VORGAENGER" in rot[0]      # sagt, was zu tun ist
        assert all("altes-geheimnis-xyz" not in r.getMessage()
                   for r in caplog.records)                  # kein Token-Leak
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_neuer_token_loggt_kein_rotationsfenster(tmp_path, monkeypatch, caplog):
    # Nur der NEUE Token in Verwendung → kein Rotations-Hinweis (das Log soll
    # ausschließlich auf umzustellende Alt-Clients zeigen).
    monkeypatch.setenv("RISK_LAYER_TOKEN", "neu")
    monkeypatch.setenv("RISK_LAYER_TOKEN_VORGAENGER", "alt")
    monkeypatch.setattr(web, "_rotations_hinweis_geloggt", False)
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        with caplog.at_level(logging.INFO, logger="risk_layer.web"):
            assert _get(f"{base}/v1/health", token="neu")[0] == 200
        assert not any("Rotationsfenster" in r.getMessage() for r in caplog.records)
    finally:
        httpd.shutdown()
        httpd.server_close()


# -- serve()/main(): Exit-Codes + TLS-Warnung beim Netz-Bind -----------------

def test_serve_netzbind_ohne_token_exit_ungleich_null(monkeypatch, capsys):
    monkeypatch.delenv("RISK_LAYER_TOKEN", raising=False)
    rc = web.serve(host="0.0.0.0", port=1)
    assert rc != 0
    assert "RISK_LAYER_TOKEN" in capsys.readouterr().err


def test_main_gibt_fehlstart_exit_code_durch(monkeypatch, capsys):
    monkeypatch.delenv("RISK_LAYER_TOKEN", raising=False)
    assert web.main(["--host", "0.0.0.0", "--port", "1"]) != 0


def test_serve_belegter_port_exit_ungleich_null_und_tls_warnung(tmp_path, monkeypatch, capsys):
    # Bind schlägt fehl (OSError) → Exit-Code != 0; vorher MUSS beim Netz-Bind
    # die Klartext-/TLS-Warnung auf stderr erscheinen (Reverse-Proxy-Pflicht).
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")

    class _BindFehlschlag:
        def __init__(self, *a, **k):
            raise OSError("Port belegt (Test)")

    monkeypatch.setattr(web, "ThreadingHTTPServer", _BindFehlschlag)
    rc = web.serve(host="0.0.0.0", port=59999, berater_dir=str(tmp_path))
    assert rc != 0
    err = capsys.readouterr().err
    assert "TLS" in err and "Reverse-Proxy" in err


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


def test_veralteter_embedding_index_warnt_beim_anwendung_start(tmp_path, monkeypatch, caplog):
    # Staleness-Check (embedding.pruefe_aktualitaet): Anwendung.__init__ prüft
    # EINMALIG nach dem Laden von Embedding+Graph, ob der Index noch zum
    # Normgraphen passt. Weicht der Fingerprint ab → laute Warnung im Log;
    # der Start degradiert, crasht nie.
    import risk_layer.embedding as emb

    class FakeIdx:
        encoder_name = "hash512"
        graph_fingerprint = "alt"          # ≠ FakeGraph.fingerprint()

    class FakeGraph:
        def fingerprint(self):
            return "neu"

        def stats(self):
            return {"knoten": 0}

    monkeypatch.setattr(emb, "index_oeffnen", lambda *a, **k: FakeIdx())

    def kein_encoder(*a, **k):              # Encoder-Load abkürzen (kein Modell im Test)
        raise RuntimeError("kein Encoder (Test)")
    monkeypatch.setattr(emb, "encoder_oeffnen", kein_encoder)
    monkeypatch.setattr(web, "graph_oeffnen", lambda *a, **k: FakeGraph())
    with caplog.at_level(logging.WARNING, logger="risk_layer.embedding"):
        Anwendung(None, str(tmp_path))
    assert any("VERALTET" in r.getMessage() for r in caplog.records), \
        "Fingerprint-Abweichung muss beim Start als Warnung im Log stehen"


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


# -- Engine-Status (GET /v1/status + Degradationsgrund-Abgriff) ---------------

def test_v1_status_braucht_token(tmp_path, monkeypatch):
    # /v1/status ist auth-pflichtig wie ALLE /v1-Routen (der Engine-Zustand
    # verrät Deployment-Details) – mit Token kommt der Schichten-Block.
    import json as _json
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    httpd, base = _starte(Anwendung(None, str(tmp_path)))
    try:
        assert _get(f"{base}/v1/status")[0] == 401                 # ohne Token
        assert _get(f"{base}/v1/status", token="falsch")[0] == 401
        status, body = _get(f"{base}/v1/status", token="geheim")
        assert status == 200
        obj = _json.loads(body.decode("utf-8"))
        assert obj["ok"] is True and "schichten" in obj["engine_status"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_v1_radar_braucht_token(tmp_path, monkeypatch):
    # /v1/radar ist auth-pflichtig wie ALLE /v1-Routen (der Bericht verrät
    # Festwissens- und Korpus-Details). Expliziter graph_pfad in tmp_path hält
    # den Test umgebungsunabhängig (Dev-Maschinen haben oft corpus/graph.sqlite):
    # ohne Graph antwortet die Route authentifiziert mit 503 (fehlende
    # Betriebsvoraussetzung), nicht 200.
    import json as _json
    monkeypatch.setenv("RISK_LAYER_TOKEN", "geheim")
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "kein-graph.sqlite"))
    httpd, base = _starte(app)
    try:
        assert _get(f"{base}/v1/radar")[0] == 401                  # ohne Token
        assert _get(f"{base}/v1/radar", token="falsch")[0] == 401
        status, body = _get(f"{base}/v1/radar", token="geheim")
        assert status == 503
        obj = _json.loads(body.decode("utf-8"))
        assert obj["ok"] is False and "Normgraph" in obj["fehler"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_engine_status_traegt_encoder_degradationsgrund(tmp_path, monkeypatch):
    # Der GRUND der Degradation wird am bestehenden try/except ABGEGRIFFEN
    # (Muster test_embedding_encoder_fehler_degradiert): Encoder-Ladefehler
    # → embedding.aktiv False + grund im Status, nicht nur im Log.
    import risk_layer.embedding as emb

    class FakeIdx:
        encoder_name = "hash512"
        graph_fingerprint = "fp-test"
    monkeypatch.setattr(emb, "index_oeffnen", lambda *a, **k: FakeIdx())

    def boom(*a, **k):
        raise RuntimeError("kein torch")
    monkeypatch.setattr(emb, "encoder_oeffnen", boom)
    app = Anwendung(None, str(tmp_path))
    st = app.engine_status()["schichten"]
    assert st["embedding"]["aktiv"] is False
    assert "kein torch" in st["embedding"]["grund"]
    # Index-Metadaten bleiben trotz Encoder-Ausfall erhalten: der Status belegt,
    # WELCHER Index benutzt worden wäre (Haftungsakte).
    assert st["embedding"]["index_fingerprint"] == "fp-test"
    assert st["embedding"]["encoder"] == "hash512"
    # Folge-Degradation: ohne Encoder auch kein Reranker – ebenfalls begründet.
    assert st["reranker"]["aktiv"] is False and st["reranker"]["grund"]


def test_engine_status_reranker_grund_ohne_gpu(tmp_path, monkeypatch):
    # Reranker läuft NUR auf GPU: ohne cuda-Device wird er gar nicht erst
    # versucht – der Status nennt das ehrlich (statt still aktiv:false).
    import risk_layer.embedding as emb

    class FakeIdx:
        encoder_name = "hash512"
        graph_fingerprint = "fp-test"

    class FakeEncoder:
        name = "hash512"
    monkeypatch.setenv("RISK_LAYER_EMB_DEVICE", "cpu")
    monkeypatch.setattr(emb, "index_oeffnen", lambda *a, **k: FakeIdx())
    monkeypatch.setattr(emb, "encoder_oeffnen", lambda *a, **k: FakeEncoder())
    app = Anwendung(None, str(tmp_path))
    st = app.engine_status()["schichten"]
    assert st["embedding"]["aktiv"] is True
    assert st["reranker"]["aktiv"] is False
    assert "GPU" in st["reranker"]["grund"]
