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
    assert "-ngl" in cmd          # alle Layer auf die GPU
    # /slots + /metrics, damit /v1/llm/status die Queue liefert
    assert "--slots" in cmd and "--metrics" in cmd


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


def test_download_engine_cuda_zieht_auch_cudart(monkeypatch, tmp_path):
    assets = [
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", "u-cudart"),
        ("llama-b1-bin-win-cuda-12.4-x64.zip", "u-cuda"),
        ("llama-b1-bin-win-vulkan-x64.zip", "u-vk"),
    ]
    geladen: list[str] = []
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M, "_release_assets", lambda **k: assets)
    monkeypatch.setattr(M, "_lade_und_entpacke", lambda name, url, zd, f: geladen.append(name))
    monkeypatch.setattr(M, "finde_llama_server", lambda **k: str(tmp_path / "llama-server.exe"))
    out = M.download_engine(tmp_path, backend="cuda")
    assert out.endswith("llama-server.exe")
    assert geladen == ["llama-b1-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]


def test_download_engine_amd_nimmt_hip_ohne_cudart(monkeypatch, tmp_path):
    assets = [("llama-b1-bin-win-hip-radeon-x64.zip", "u-hip"),
              ("llama-b1-bin-win-vulkan-x64.zip", "u-vk")]
    geladen: list[str] = []
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M, "_release_assets", lambda **k: assets)
    monkeypatch.setattr(M, "_lade_und_entpacke", lambda name, url, zd, f: geladen.append(name))
    monkeypatch.setattr(M, "finde_llama_server", lambda **k: "x/llama-server")
    M.download_engine(tmp_path, backend="hip")
    assert geladen == ["llama-b1-bin-win-hip-radeon-x64.zip"]


def test_download_engine_meldet_fehlend(monkeypatch, tmp_path):
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M, "_release_assets", lambda **k: [("readme.txt", "u")])
    with pytest.raises(RuntimeError):
        M.download_engine(tmp_path, backend="cuda")


def test_download_engine_cuda_ohne_cudart_crasht_nicht(monkeypatch, tmp_path):
    # CUDA-Build da, aber kein cudart-Paket -> nur warnen, nicht crashen, Build trotzdem da
    assets = [("llama-b1-bin-win-cuda-12.4-x64.zip", "u-cuda")]
    geladen: list[str] = []
    monkeypatch.setattr(M.sys, "platform", "win32")
    monkeypatch.setattr(M, "_release_assets", lambda **k: assets)
    monkeypatch.setattr(M, "_lade_und_entpacke", lambda name, url, zd, f: geladen.append(name))
    monkeypatch.setattr(M, "finde_llama_server", lambda **k: "x/llama-server")
    assert M.download_engine(tmp_path, backend="cuda") == "x/llama-server"
    assert geladen == ["llama-b1-bin-win-cuda-12.4-x64.zip"]
