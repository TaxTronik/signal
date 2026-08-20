"""Modell-Integration: Registry, HF-Auflösung, Download, Engine-Manager (gemockt)."""

import hashlib
import io
import json
import logging
import urllib.error
from pathlib import Path

import pytest

from risk_layer import modell as M
from risk_layer import llama_cpp_manifest as EM


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_registry_default_ist_granite_4_1_8b():
    assert M.DEFAULT_MODELL == "granite-4.1-8b"
    g = M.MODELLE["granite-4.1-8b"]
    assert g.repo == "ibm-granite/granite-4.1-8b-GGUF" and g.quant == "Q5_K_M"
    # Qwen3-14B bleibt als "Gruendlich-Modus" verfuegbar (nur nicht mehr Default)
    assert M.MODELLE["qwen3-14b"].repo == "Qwen/Qwen3-14B-GGUF"


def test_registry_granite_eintraege():
    # Verifizierte, oeffentliche IBM-Granite-4.1-GGUF (Apache-2.0) fuer den Head-to-Head.
    g8 = M.MODELLE["granite-4.1-8b"]
    assert g8.repo == "ibm-granite/granite-4.1-8b-GGUF" and g8.quant == "Q5_K_M"
    assert "granite-4.1-3b" in M.MODELLE


def test_datei_url_form():
    info = M.MODELLE["qwen3-14b"]
    url = M.datei_url(
        info.repo, "Qwen3-14B-Q5_K_M.gguf", revision=info.revision,
    )
    assert url == (
        "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/"
        "530227a7d994db8eca5ab5ced2fb692b614357fd/"
        "Qwen3-14B-Q5_K_M.gguf?download=true"
    )


def test_passende_dateien_filtert_quant(monkeypatch):
    revision = "a" * 40
    gesehen = {}

    def fake_dateien(repo, *, revision, timeout=25.0):
        gesehen["revision"] = revision
        return [
        "README.md", "Qwen3-14B-Q4_K_M.gguf", "Qwen3-14B-Q5_K_M.gguf", "Qwen3-14B-Q6_K.gguf",
        ]

    monkeypatch.setattr(M, "hf_dateien", fake_dateien)
    assert M.passende_dateien("x", "Q5_K_M", revision=revision) == [
        "Qwen3-14B-Q5_K_M.gguf"
    ]
    assert gesehen["revision"] == revision


def test_modell_pfad_findet_gguf(tmp_path):
    (tmp_path / "Qwen3-14B-Q5_K_M.gguf").write_bytes(b"x")
    info = M.MODELLE["qwen3-14b"]
    assert M.modell_pfad(info, tmp_path).name == "Qwen3-14B-Q5_K_M.gguf"
    assert M.modell_pfad(info, tmp_path / "leer") is None


def test_modell_pfad_disambiguiert_bei_gleichem_quant(tmp_path):
    # Qwen UND Granite mit gleicher Quant -> jeder Eintrag muss SEINE Datei finden,
    # nicht die alphabetisch erste (sonst laedt die Engine trotz Default das falsche Modell).
    (tmp_path / "Qwen3-14B-Q5_K_M.gguf").write_bytes(b"x")
    (tmp_path / "granite-4.1-8b-Q5_K_M.gguf").write_bytes(b"x")
    assert M.modell_pfad(M.MODELLE["granite-4.1-8b"], tmp_path).name == "granite-4.1-8b-Q5_K_M.gguf"
    assert M.modell_pfad(M.MODELLE["qwen3-14b"], tmp_path).name == "Qwen3-14B-Q5_K_M.gguf"


def test_finde_llama_server(monkeypatch, tmp_path):
    monkeypatch.setattr(M.shutil, "which", lambda name: None)
    # leeres engine_dir, damit eine real beschaffte ./engine den Test nicht stört
    leer = tmp_path / "kein_engine"
    assert M.finde_llama_server(engine_dir=leer) is None
    binpath = tmp_path / "llama-server"
    binpath.write_text("#!/bin/sh\n")
    assert M.finde_llama_server(str(binpath)) == str(binpath)
    assert M.finde_llama_server(str(tmp_path / "fehlt")) is None
    # in ./engine abgelegtes Binary wird gefunden
    (tmp_path / "eng").mkdir()
    (tmp_path / "eng" / "llama-server-helper.exe").write_text("nicht das Binary")
    (tmp_path / "eng" / "llama-server.exe").write_text("x")
    assert M.finde_llama_server(engine_dir=tmp_path / "eng").endswith("llama-server.exe")


def test_finde_llama_server_bevorzugt_gepinnten_baum_vor_path(monkeypatch, tmp_path):
    verwaltet = tmp_path / EM.LLAMA_CPP_RELEASE / "llama-server.exe"
    verwaltet.parent.mkdir()
    verwaltet.write_text("x")
    monkeypatch.setattr(M.shutil, "which", lambda name: "C:/alt/llama-server.exe")
    assert M.finde_llama_server(engine_dir=tmp_path) == str(verwaltet)


def test_manager_kommando(tmp_path):
    binpath = tmp_path / "llama-server"
    binpath.write_text("x")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    mgr = M.LlamaServerManager(model, binary=str(binpath), port=8080, n_ctx=8192)
    cmd = mgr.kommando()
    assert cmd[0] == str(binpath)
    assert "-m" in cmd and str(model) in cmd
    assert "--port" in cmd and "8080" in cmd
    assert "-ngl" in cmd          # alle Layer auf die GPU
    # /slots + /metrics, damit /v1/llm/status die Queue liefert
    assert "--slots" in cmd and "--metrics" in cmd


def test_manager_start_ohne_binary_meldet_fehlende_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "finde_llama_server", lambda *a, **k: None)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    mgr = M.LlamaServerManager(model)  # kein Binary auffindbar
    with pytest.raises(FileNotFoundError):
        mgr.start()


def test_manager_leitet_serverausgabe_in_logdatei_und_schliesst_beim_stop(monkeypatch, tmp_path):
    # stdout/stderr gehen in eine Logdatei statt DEVNULL: Startfehler wie
    # "unknown architecture" (zu alte Engine) müssen diagnostizierbar sein.
    binpath = tmp_path / "llama-server"
    binpath.write_text("x")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    aufruf = {}

    class FakeProc:
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): return 0
        def kill(self): pass

    def fake_popen(cmd, stdout=None, stderr=None, **kw):
        aufruf["stdout"], aufruf["stderr"] = stdout, stderr
        return FakeProc()

    monkeypatch.setattr(M.subprocess, "Popen", fake_popen)
    mgr = M.LlamaServerManager(model, binary=str(binpath))
    mgr.start()
    assert aufruf["stdout"] is not M.subprocess.DEVNULL
    assert hasattr(aufruf["stdout"], "write")              # echtes Datei-Handle
    assert aufruf["stderr"] is M.subprocess.STDOUT         # stderr im selben Log
    assert mgr.log_pfad and Path(mgr.log_pfad).exists()
    mgr.stop()
    assert aufruf["stdout"].closed                          # Handle beim Stop zu
    Path(mgr.log_pfad).unlink()  # auf Windows nur möglich, wenn das Handle zu ist


def test_download_datei_schreibt_datei(monkeypatch, tmp_path):
    class FakeResp:
        status = 200
        headers = {"Content-Length": "5"}
        def __init__(self): self._b = io.BytesIO(b"hello")
        def read(self, n): return self._b.read(n)
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda req, timeout=60: FakeResp())
    ziel = tmp_path / "f.gguf"
    M.download_datei("http://x/f", ziel)
    assert ziel.read_bytes() == b"hello"


def _fake_resp(status, headers, body):
    class FakeResp:
        def __init__(self):
            self.status = status
            self.headers = headers
            self._b = io.BytesIO(body)
        def read(self, n): return self._b.read(n)
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return FakeResp()


def test_download_datei_resume_206_setzt_fort(monkeypatch, tmp_path):
    # Teildownload + Server mit Range-Support (206): anhängen, Gesamtgröße
    # aus Content-Range ("bytes a-b/<gesamt>").
    ziel = tmp_path / "f.gguf"
    ziel.write_bytes(b"he")
    def urlopen(req, timeout=60):
        assert req.get_header("Range") == "bytes=2-"
        return _fake_resp(206, {"Content-Range": "bytes 2-4/5", "Content-Length": "3"}, b"llo")
    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    stand = []
    M.download_datei("http://x/f", ziel, fortschritt=lambda g, t: stand.append((g, t)))
    assert ziel.read_bytes() == b"hello"
    assert stand[-1] == (5, 5)


def test_download_datei_resume_416_bedeutet_bereits_vollstaendig(monkeypatch, tmp_path):
    # Datei ist KOMPLETT, Resume fragt "bytes=<size>-" -> Server: 416. Das ist
    # kein Fehler, sondern "fertig": kein Traceback, Datei unangetastet,
    # Fortschritt meldet sauber 100 % (Gesamtgröße aus "bytes */<gesamt>").
    ziel = tmp_path / "f.gguf"
    ziel.write_bytes(b"hello")
    def urlopen(req, timeout=60):
        assert req.get_header("Range") == "bytes=5-"
        raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable",
                                     {"Content-Range": "bytes */5"}, None)
    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    stand = []
    out = M.download_datei("http://x/f", ziel, fortschritt=lambda g, t: stand.append((g, t)))
    assert out == ziel and ziel.read_bytes() == b"hello"
    assert stand == [(5, 5)]


def test_download_datei_416_ohne_content_range_faellt_auf_dateigroesse_zurueck(monkeypatch, tmp_path):
    ziel = tmp_path / "f.gguf"
    ziel.write_bytes(b"hello")
    def urlopen(req, timeout=60):
        raise urllib.error.HTTPError("http://x/f", 416, "Range Not Satisfiable", {}, None)
    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    stand = []
    M.download_datei("http://x/f", ziel, fortschritt=lambda g, t: stand.append((g, t)))
    assert stand == [(5, 5)]


def test_download_datei_andere_http_fehler_bleiben_sichtbar(monkeypatch, tmp_path):
    # NUR 416 wird als "bereits vollständig" gedeutet – ein 404 muss weiter knallen.
    ziel = tmp_path / "f.gguf"
    ziel.write_bytes(b"he")
    def urlopen(req, timeout=60):
        raise urllib.error.HTTPError("http://x/f", 404, "Not Found", {}, None)
    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        M.download_datei("http://x/f", ziel)


def test_download_datei_resume_server_ignoriert_range(monkeypatch, tmp_path):
    # Server OHNE Range-Support antwortet trotz Range-Header mit 200 und der
    # GANZEN Datei: neu schreiben statt anhängen (sonst läge die Datei doppelt
    # vor) und Content-Length NICHT um das Vorhandene aufblähen.
    ziel = tmp_path / "f.gguf"
    ziel.write_bytes(b"he")
    monkeypatch.setattr(M.urllib.request, "urlopen",
                        lambda req, timeout=60: _fake_resp(200, {"Content-Length": "5"}, b"hello"))
    stand = []
    M.download_datei("http://x/f", ziel, fortschritt=lambda g, t: stand.append((g, t)))
    assert ziel.read_bytes() == b"hello"      # NICHT b"hehello"
    assert stand[-1] == (5, 5)                # gesamt = 5, nicht 5 + 2


# --- Integritätsnachweis (L4): Sidecar, HF-Erwartungswert, Verify ----------

def test_download_datei_schreibt_sha256_sidecar_im_sha256sum_format(monkeypatch, tmp_path):
    # Nach erfolgreichem Download liegt <datei>.sha256 im EXAKTEN sha256sum-
    # Format: "<hex>  <dateiname>\n" — zwei Leerzeichen, nur der Dateiname ohne
    # Pfad. Damit prüft auch `sha256sum -c f.gguf.sha256` ohne diese Engine.
    monkeypatch.setattr(M.urllib.request, "urlopen",
                        lambda req, timeout=60: _fake_resp(200, {"Content-Length": "5"}, b"hello"))
    ziel = tmp_path / "f.gguf"
    M.download_datei("http://x/f", ziel)
    sc = tmp_path / "f.gguf.sha256"
    assert sc.read_text(encoding="utf-8") == f"{_sha(b'hello')}  f.gguf\n"
    assert M.sidecar_lesen(ziel) == _sha(b"hello")


def test_download_datei_resume_206_sidecar_hash_ueber_gesamtdatei(monkeypatch, tmp_path):
    # Resume (206): der Sidecar muss den Hash der GESAMTDATEI tragen — der
    # vorhandene Teil wird einmalig nachgehasht, der Netz-Teil streamend.
    ziel = tmp_path / "f.gguf"
    ziel.write_bytes(b"he")
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda req, timeout=60: _fake_resp(
        206, {"Content-Range": "bytes 2-4/5", "Content-Length": "3"}, b"llo"))
    M.download_datei("http://x/f", ziel)
    assert ziel.read_bytes() == b"hello"
    assert M.sidecar_lesen(ziel) == _sha(b"hello")     # NICHT nur _sha(b"llo")


def test_download_datei_digest_mismatch_gegen_hf_erwartung_ist_harter_fehler(monkeypatch, tmp_path):
    # Abweichung vom HF-Erwartungswert = manipuliertes/korruptes Artefakt →
    # harter Fehler mit klarer Meldung; die Datei darf NICHT liegen bleiben
    # (sonst meldete der nächste Resume-Versuch sie per 416 still als "fertig").
    monkeypatch.setattr(M.urllib.request, "urlopen",
                        lambda req, timeout=60: _fake_resp(200, {"Content-Length": "5"}, b"hello"))
    ziel = tmp_path / "f.gguf"
    with pytest.raises(RuntimeError, match="weicht|Integrität"):
        M.download_datei("http://x/f", ziel, erwartet_sha256="ab" * 32)
    assert not ziel.exists()
    assert not (tmp_path / "f.gguf.sha256").exists()   # kein Sidecar für Korruptes


def test_download_datei_digest_match_gegen_hf_erwartung(monkeypatch, tmp_path):
    # Passender Erwartungswert (Groß-/Kleinschreibung egal) → Erfolg + Sidecar.
    monkeypatch.setattr(M.urllib.request, "urlopen",
                        lambda req, timeout=60: _fake_resp(200, {"Content-Length": "5"}, b"hello"))
    ziel = tmp_path / "f.gguf"
    M.download_datei("http://x/f", ziel, erwartet_sha256=_sha(b"hello").upper())
    assert ziel.read_bytes() == b"hello"
    assert M.sidecar_lesen(ziel) == _sha(b"hello")


def test_download_datei_416_holt_fehlenden_sidecar_nach(monkeypatch, tmp_path):
    # Datei war schon vollständig (416, z. B. Download aus einer Version vor L4):
    # der Stream-Hasher lief nie — der Sidecar wird einmalig per Vollhash nachgeholt.
    ziel = tmp_path / "f.gguf"
    ziel.write_bytes(b"hello")
    def urlopen(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable",
                                     {"Content-Range": "bytes */5"}, None)
    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    M.download_datei("http://x/f", ziel)
    assert M.sidecar_lesen(ziel) == _sha(b"hello")


def test_download_datei_416_vertraut_sidecar_nicht_statt_manifest(monkeypatch, tmp_path):
    # Ein Angreifer darf kein fertiges Archiv plus selbst ausgestellten Sidecar
    # platzieren und damit den unabhängigen Engine-Manifest-Digest umgehen.
    ziel = tmp_path / "engine.zip"
    ziel.write_bytes(b"boese")
    M.sidecar_schreiben(ziel, _sha(b"boese"))

    def urlopen(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable",
                                     {"Content-Range": "bytes */5"}, None)

    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="Integrität|Erwartungswert"):
        M.download_datei("http://x/engine.zip", ziel,
                         erwartet_sha256=_sha(b"offiziell"))
    assert not ziel.exists()
    assert not M.sidecar_pfad(ziel).exists()


def test_hf_lfs_digests_liest_tree_endpunkt(monkeypatch):
    # Der HF-Tree-Endpunkt liefert je LFS-Datei lfs.oid = sha256; Nicht-LFS-
    # Dateien (README) haben keinen Digest und fehlen im Ergebnis.
    eintraege = [
        {"type": "file", "path": "README.md", "oid": "gitsha", "size": 5},
        {"type": "file", "path": "a.gguf", "oid": "gitsha",
         "lfs": {"oid": "AB" * 32, "size": 99, "pointerSize": 135}},
        {"type": "directory", "path": "sub"},
    ]
    gefragt = {}

    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(req, timeout=25.0):
        gefragt["url"] = req.full_url
        return FakeResp(json.dumps(eintraege).encode("utf-8"))
    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    revision = "b" * 40
    d = M.hf_lfs_digests("org/repo", revision=revision)
    assert d == {"a.gguf": "ab" * 32}                   # nur LFS, kleingeschrieben
    assert f"/api/models/org/repo/tree/{revision}" in gefragt["url"]


def test_hf_dateien_liest_exakten_commit_endpunkt(monkeypatch):
    revision = "c" * 40
    gefragt = {}

    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(url, timeout=25.0):
        gefragt["url"] = url
        return FakeResp(json.dumps({"siblings": [{"rfilename": "a.gguf"}]}).encode())

    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    assert M.hf_dateien("org/repo", revision=revision) == ["a.gguf"]
    assert gefragt["url"].endswith(f"/api/models/org/repo/revision/{revision}")


def test_download_modell_reicht_hf_erwartungswert_durch(monkeypatch, tmp_path):
    erhalten = {}

    def fake_passende(repo, quant, *, revision):
        erhalten["listen_revision"] = revision
        return ["a.gguf"]

    def fake_digests(repo, *, revision):
        erhalten["digest_revision"] = revision
        return {"a.gguf": "f" * 64}

    monkeypatch.setattr(M, "passende_dateien", fake_passende)
    monkeypatch.setattr(M, "hf_lfs_digests", fake_digests)

    def fake_dl(url, ziel, *, fortschritt=None, erwartet_sha256=None):
        erhalten["erwartet"] = erwartet_sha256
        erhalten["url"] = url
        return ziel
    monkeypatch.setattr(M, "download_datei", fake_dl)
    info = M.MODELLE[M.DEFAULT_MODELL]
    M.download_modell(info, tmp_path)
    assert erhalten["erwartet"] == "f" * 64
    assert erhalten["listen_revision"] == info.revision
    assert erhalten["digest_revision"] == info.revision
    assert f"/resolve/{info.revision}/" in erhalten["url"]


def test_download_modell_degradiert_ohne_hf_digests(monkeypatch, tmp_path):
    # Digest-Abfrage scheitert (offline/HF-API down) → Download läuft trotzdem,
    # nur ohne Erwartungswert (der Sidecar pinnt dann den Erst-Download).
    def kaputt(repo, *, revision):
        raise urllib.error.URLError("offline (Test)")
    monkeypatch.setattr(M, "hf_lfs_digests", kaputt)
    monkeypatch.setattr(
        M, "passende_dateien", lambda repo, quant, *, revision: ["a.gguf"]
    )
    erhalten = {}

    def fake_dl(url, ziel, *, fortschritt=None, erwartet_sha256=None):
        erhalten["erwartet"] = erwartet_sha256
        return ziel
    monkeypatch.setattr(M, "download_datei", fake_dl)
    M.download_modell(M.MODELLE[M.DEFAULT_MODELL], tmp_path)
    assert erhalten["erwartet"] is None


def test_verifiziere_artefakte_vier_faelle(tmp_path):
    # (1) ok: Datei + passender Sidecar
    ok_datei = tmp_path / "a.gguf"
    ok_datei.write_bytes(b"aaa")
    M.sidecar_schreiben(ok_datei, _sha(b"aaa"))
    # (2) kein Sidecar
    (tmp_path / "b.gguf").write_bytes(b"bbb")
    # (3) Sidecar da, Datei fehlt
    M.sidecar_schreiben(tmp_path / "c.gguf", _sha(b"ccc"))
    # (4) Hash-Mismatch
    falsch = tmp_path / "d.gguf"
    falsch.write_bytes(b"echt")
    M.sidecar_schreiben(falsch, _sha(b"anders"))
    res = M.verifiziere_artefakte(tmp_path)
    assert [r["datei"] for r in res] == ["a.gguf", "b.gguf", "c.gguf", "d.gguf"]  # deterministisch
    a, b, c, d = res
    assert a == {"datei": "a.gguf", "erwartet": _sha(b"aaa"), "ist": _sha(b"aaa"),
                 "ok": True, "grund": None}
    assert b["ok"] is False and b["erwartet"] is None and b["ist"] is None
    assert "kein Sidecar" in b["grund"]
    assert c["ok"] is False and c["erwartet"] == _sha(b"ccc") and c["ist"] is None
    assert "fehlt" in c["grund"]
    assert d["ok"] is False and d["grund"] == "Digest weicht ab"
    assert d["erwartet"] == _sha(b"anders") and d["ist"] == _sha(b"echt")


def test_verifiziere_artefakte_leer_und_fehlendes_verzeichnis(tmp_path):
    assert M.verifiziere_artefakte(tmp_path / "gibtsnicht") == []   # kein Crash
    assert M.verifiziere_artefakte(tmp_path) == []


def test_verifiziere_artefakte_vertrag_signatur():
    # FESTER VERTRAG für die CLI ('risk-layer modell verify'): Default-Verzeichnis
    # ist DEFAULT_MODELL_DIR; Aufruf ohne Argumente muss funktionieren.
    import inspect
    sig = inspect.signature(M.verifiziere_artefakte)
    assert list(sig.parameters) == ["verzeichnis"]
    assert sig.parameters["verzeichnis"].default == M.DEFAULT_MODELL_DIR


def test_registry_hat_nur_exakte_hf_commit_pins():
    erwartet = {
        "qwen3-14b": "530227a7d994db8eca5ab5ced2fb692b614357fd",
        "qwen3-8b": "7c41481f57cb95916b40956ab2f0b139b296d974",
        "qwen3.5-4b": "e87f176479d0855a907a41277aca2f8ee7a09523",
        "lfm2.5-8b-a1b": "dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0",
        "granite-4.1-8b": "865b82c2e7970d82e3731278c88c57ae7138359c",
        "granite-4.1-3b": "ab4701481089b58a082ef63cc1cee738887293ff",
    }
    assert {key: info.revision for key, info in M.MODELLE.items()} == erwartet
    assert not hasattr(M, "HF_REVISION")
    for info in M.MODELLE.values():
        assert len(info.revision) == 40
        assert set(info.revision) <= set("0123456789abcdef")


def test_hf_helfer_verweigern_bewegliche_revisionen():
    with pytest.raises(ValueError, match="40-stelliger Commit-Hash"):
        M.datei_url("a/b", "c.gguf", revision="main")
    with pytest.raises(ValueError, match="40-stelliger Commit-Hash"):
        M.hf_dateien("a/b", revision="latest")
    with pytest.raises(ValueError, match="40-stelliger Commit-Hash"):
        M.hf_lfs_digests("a/b", revision="v1")
    with pytest.raises(ValueError, match="40-stelliger Commit-Hash"):
        M.ModellInfo("key", "Name", "a/b", "main", "Q5_K_M", 1.0)


def test_manager_start_loggt_sidecar_digest_oder_ungeprueft(monkeypatch, tmp_path, caplog):
    # Engine-Start: Digest aus dem Sidecar wird GELOGGT (kein Vollhash der
    # GB-Datei beim Start!); ohne Sidecar gibt es den "ungeprüft"-Hinweis
    # mit Verweis auf 'modell verify'.
    binpath = tmp_path / "llama-server"
    binpath.write_text("x")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")

    class FakeProc:
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): return 0
        def kill(self): pass
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: FakeProc())

    mgr = M.LlamaServerManager(model, binary=str(binpath))
    with caplog.at_level(logging.INFO, logger="risk_layer.modell"):
        mgr.start()
    mgr.stop()
    assert any("ungeprüft" in r.getMessage() and "verify" in r.getMessage()
               for r in caplog.records)
    Path(mgr.log_pfad).unlink()

    caplog.clear()
    M.sidecar_schreiben(model, _sha(b"x"))
    mgr2 = M.LlamaServerManager(model, binary=str(binpath))
    with caplog.at_level(logging.INFO, logger="risk_layer.modell"):
        mgr2.start()
    mgr2.stop()
    assert any(_sha(b"x") in r.getMessage() for r in caplog.records)
    Path(mgr2.log_pfad).unlink()


# --- Backend-Erkennung / -Auswahl -----------------------------------------

def test_backend_patterns_explizit(monkeypatch):
    monkeypatch.setattr(M.sys, "platform", "win32")
    assert M.backend_patterns("hip")[0] == "bin-win-hip-radeon"
    assert M.backend_patterns("cuda")[0].startswith("bin-win-cuda")
    assert M.backend_patterns("vulkan") == ["bin-win-vulkan-x64"]
    monkeypatch.setattr(M.sys, "platform", "linux")
    assert M.backend_patterns("hip") == ["bin-ubuntu-rocm"]
    # Linux+NVIDIA: KEIN CUDA-Prebuilt -> Vulkan-Fallback
    assert "bin-ubuntu-vulkan-x64" in M.backend_patterns("nvidia")
    assert "cuda" not in " ".join(M.backend_patterns("nvidia"))


def test_backend_patterns_auto_nutzt_vendor(monkeypatch):
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M, "erkenne_gpu_vendor", lambda: "amd")
    assert M.backend_patterns("auto")[0] == "bin-win-hip-radeon"
    monkeypatch.setattr(M, "erkenne_gpu_vendor", lambda: "nvidia")
    assert M.backend_patterns("auto")[0].startswith("bin-win-cuda")
    monkeypatch.setattr(M, "erkenne_gpu_vendor", lambda: None)
    assert M.backend_patterns("auto") == M._BACKEND_WIN["cpu"]


def test_llm_laufzeit_backend_steuert_gpu_layer(monkeypatch):
    monkeypatch.setenv("RISK_LAYER_LLM_BACKEND", "cpu")
    assert M.llm_laufzeit_backend() == "cpu"
    assert M.llm_gpu_layers() == 0
    monkeypatch.setenv("RISK_LAYER_LLM_BACKEND", "gpu")
    assert M.llm_laufzeit_backend() == "gpu"
    assert M.llm_gpu_layers() == -1
    monkeypatch.setenv("RISK_LAYER_LLM_BACKEND", "ungueltig")
    assert M.llm_laufzeit_backend() == "auto"
    assert M.llm_gpu_layers() == -1


def test_waehle_asset_trennt_cudart_und_formate():
    assets = [
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", "u-cudart"),
        ("llama-b1-bin-win-cuda-12.4-x64.zip", "u-cuda"),
        ("llama-b1-bin-ubuntu-rocm-7.2-x64.tar.gz", "u-rocm"),
    ]
    # 'cuda'-Build matcht NICHT das cudart-Paket (Praefix-Trennung)
    assert M._waehle_asset(assets, "bin-win-cuda-12") == ("llama-b1-bin-win-cuda-12.4-x64.zip", "u-cuda")
    assert M._waehle_asset(assets, "bin-win-cuda-12", cudart=True) == ("cudart-llama-bin-win-cuda-12.4-x64.zip", "u-cudart")
    # .tar.gz (Linux) wird ebenso gefunden
    assert M._waehle_asset(assets, "bin-ubuntu-rocm") == ("llama-b1-bin-ubuntu-rocm-7.2-x64.tar.gz", "u-rocm")
    assert M._waehle_asset(assets, "bin-win-vulkan-x64") is None


def test_engine_manifest_ist_fest_hashgepinnt_und_ohne_latest():
    assert EM.LLAMA_CPP_RELEASE == "b9986"
    assert M.ENGINE_ASSETS
    assert not hasattr(M, "GITHUB_RELEASES")
    assert len({a.name for a in M.ENGINE_ASSETS}) == len(M.ENGINE_ASSETS)
    for asset in M.ENGINE_ASSETS:
        assert len(asset.sha256) == 64
        assert set(asset.sha256) <= set("0123456789abcdef")
        assert asset.groesse > 0
        assert f"/releases/download/{EM.LLAMA_CPP_RELEASE}/" in asset.url
        assert "/latest/" not in asset.url


def test_engine_manifest_deckt_jeden_auto_backend_ab():
    for table in (M._BACKEND_WIN, M._BACKEND_LINUX):
        for patterns in table.values():
            assert any(M._waehle_engine_asset(pattern) for pattern in patterns)


def test_finde_engine_asset_liest_keine_release_metadaten(monkeypatch):
    def kein_netz(*args, **kwargs):
        raise AssertionError("Release-Metadaten dürfen nicht zur Laufzeit abgefragt werden")

    monkeypatch.setattr(M.urllib.request, "urlopen", kein_netz)
    treffer = M.finde_engine_asset("bin-win-vulkan-x64")
    assert treffer is not None
    assert treffer[0] == "llama-b9986-bin-win-vulkan-x64.zip"
    assert "/download/b9986/" in treffer[1]


def test_lade_und_entpacke_prueft_hash_vor_extraktion(monkeypatch, tmp_path):
    asset = M.EngineAsset("manipuliert.zip", "ab" * 32, 5)
    monkeypatch.setattr(
        M.urllib.request,
        "urlopen",
        lambda req, timeout=60: _fake_resp(200, {"Content-Length": "5"}, b"boese"),
    )
    with pytest.raises(RuntimeError, match="Integrität|Erwartungswert"):
        M._lade_und_entpacke(asset, tmp_path, None)
    assert not (tmp_path / asset.name).exists()
    assert not M.sidecar_pfad(tmp_path / asset.name).exists()


def test_lade_und_entpacke_prueft_manifestgroesse(monkeypatch, tmp_path):
    asset = M.EngineAsset("falsch-kurz.zip", "ab" * 32, 99)

    def fake_download(url, ziel, *, fortschritt=None, erwartet_sha256=None):
        assert erwartet_sha256 == asset.sha256
        ziel.write_bytes(b"kurz")
        M.sidecar_schreiben(ziel, asset.sha256)
        return ziel

    monkeypatch.setattr(M, "download_datei", fake_download)
    with pytest.raises(RuntimeError, match="Größe|Manifest"):
        M._lade_und_entpacke(asset, tmp_path, None)
    assert not (tmp_path / asset.name).exists()
    assert not M.sidecar_pfad(tmp_path / asset.name).exists()


def test_download_engine_cuda_zieht_auch_cudart(monkeypatch, tmp_path):
    geladen: list[M.EngineAsset] = []
    ziele: list[Path] = []
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        M,
        "_lade_und_entpacke",
        lambda asset, zd, f: (geladen.append(asset), ziele.append(zd)),
    )
    monkeypatch.setattr(
        M,
        "_finde_llama_server_im_verzeichnis",
        lambda d: str(tmp_path / "llama-server.exe"),
    )
    out = M.download_engine(tmp_path, backend="cuda")
    assert out.endswith("llama-server.exe")
    assert [a.name for a in geladen] == [
        "llama-b9986-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]
    installations_dir = (
        tmp_path / "b9986" / "llama-b9986-bin-win-cuda-12.4-x64"
    )
    assert ziele == [installations_dir, installations_dir]
    # Nicht der Download pinnt seinen eigenen Hash (TOFU), sondern das
    # repository-versionierte Manifest reicht ihn bis vor die Extraktion durch.
    assert all(len(a.sha256) == 64 for a in geladen)


def test_download_engine_amd_nimmt_hip_ohne_cudart(monkeypatch, tmp_path):
    geladen: list[str] = []
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(M, "_lade_und_entpacke", lambda asset, zd, f: geladen.append(asset.name))
    monkeypatch.setattr(M, "_finde_llama_server_im_verzeichnis", lambda d: "x/llama-server")
    M.download_engine(tmp_path, backend="hip")
    assert geladen == ["llama-b9986-bin-win-hip-radeon-x64.zip"]


def test_download_engine_meldet_fehlend(monkeypatch, tmp_path):
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(M, "ENGINE_ASSETS", ())
    with pytest.raises(RuntimeError):
        M.download_engine(tmp_path, backend="cuda")


def test_download_engine_pattern_kann_keine_fremde_plattform_waehlen(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="Plattform"):
        M.download_engine(tmp_path, pattern="bin-ubuntu-x64")
    with pytest.raises(ValueError, match="leer"):
        M.download_engine(tmp_path, pattern=" ")


def test_download_engine_cuda_ohne_gepinntes_cudart_bricht_vor_download_ab(
    monkeypatch, tmp_path
):
    # Ein nur halb registriertes CUDA-Bundle darf nicht installiert werden.
    assets = tuple(a for a in M.ENGINE_ASSETS if not a.name.startswith("cudart"))
    geladen: list[str] = []
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(M, "ENGINE_ASSETS", assets)
    monkeypatch.setattr(M, "_lade_und_entpacke", lambda asset, zd, f: geladen.append(asset.name))
    with pytest.raises(RuntimeError, match="cudart"):
        M.download_engine(tmp_path, backend="cuda")
    assert geladen == []


@pytest.mark.parametrize(
    ("system", "maschine"),
    [("darwin", "x86_64"), ("linux", "aarch64")],
)
def test_download_engine_unterstuetzt_nur_gepinnte_plattformen(
    monkeypatch, tmp_path, system, maschine
):
    monkeypatch.setattr(M.sys, "platform", system)
    monkeypatch.setattr(M.platform, "machine", lambda: maschine)
    with pytest.raises(RuntimeError, match="manuell|--llm-bin"):
        M.download_engine(tmp_path, backend="cpu")
