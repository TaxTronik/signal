"""Modell-Integration: Registry, HF-Auflösung, Download, Engine-Manager (gemockt)."""

import io
from pathlib import Path

import pytest

from risk_layer import modell as M


def test_registry_default_ist_qwen3_14b():
    assert M.DEFAULT_MODELL == "qwen3-14b"
    info = M.MODELLE["qwen3-14b"]
    assert info.repo == "Qwen/Qwen3-14B-GGUF"
    assert info.quant == "Q5_K_M"


def test_datei_url_form():
    url = M.datei_url("Qwen/Qwen3-14B-GGUF", "Qwen3-14B-Q5_K_M.gguf")
    assert url == "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q5_K_M.gguf?download=true"


def test_passende_dateien_filtert_quant(monkeypatch):
    monkeypatch.setattr(M, "hf_dateien", lambda repo, timeout=25.0: [
        "README.md", "Qwen3-14B-Q4_K_M.gguf", "Qwen3-14B-Q5_K_M.gguf", "Qwen3-14B-Q6_K.gguf",
    ])
    assert M.passende_dateien("x", "Q5_K_M") == ["Qwen3-14B-Q5_K_M.gguf"]


def test_modell_pfad_findet_gguf(tmp_path):
    (tmp_path / "Qwen3-14B-Q5_K_M.gguf").write_bytes(b"x")
    info = M.MODELLE["qwen3-14b"]
    assert M.modell_pfad(info, tmp_path).name == "Qwen3-14B-Q5_K_M.gguf"
    assert M.modell_pfad(info, tmp_path / "leer") is None


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
    (tmp_path / "eng" / "llama-server.exe").write_text("x")
    assert M.finde_llama_server(engine_dir=tmp_path / "eng").endswith("llama-server.exe")


def test_manager_kommando(tmp_path):
    binpath = tmp_path / "llama-server"; binpath.write_text("x")
    model = tmp_path / "m.gguf"; model.write_bytes(b"x")
    mgr = M.LlamaServerManager(model, binary=str(binpath), port=8080, n_ctx=8192)
    cmd = mgr.kommando()
    assert cmd[0] == str(binpath)
    assert "-m" in cmd and str(model) in cmd
    assert "--port" in cmd and "8080" in cmd
    assert "-ngl" in cmd  # alle Layer auf die GPU


def test_manager_start_ohne_binary_meldet_fehlende_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "finde_llama_server", lambda *a, **k: None)
    model = tmp_path / "m.gguf"; model.write_bytes(b"x")
    mgr = M.LlamaServerManager(model)  # kein Binary auffindbar
    with pytest.raises(FileNotFoundError):
        mgr.start()


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
