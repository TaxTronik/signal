"""Modell-Integration (Stufe 1): GGUF-Download + llama-server-Auto-Launch.

Zwei getrennte Belange:
  * Das **Modell** (GGUF) wird hier verwaltet und (resumebar) von Hugging Face
    geladen – ein einmaliger, vom Nutzer ausgelöster Build-Schritt.
  * Die **Engine** (llama.cpp ``llama-server``, RDNA4 → Vulkan-Backend) wird NICHT
    durch dieses Projekt installiert; sie ist Voraussetzung. Liegt sie auf dem
    PATH (oder wird ihr Pfad übergeben), startet ``LlamaServerManager`` sie mit
    dem Modell, wartet auf Bereitschaft und beendet sie wieder.

Zur Laufzeit der Analyse bleibt alles lokal (/NF10/): die App spricht den
lokalen llama-server über HTTP an (siehe ``llm.LlamaServerProvider``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .llm import server_verfuegbar

HF = "https://huggingface.co"
DEFAULT_MODELL_DIR = Path("models")
DEFAULT_ENGINE_DIR = Path("engine")
GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=12"


@dataclass(frozen=True)
class ModellInfo:
    key: str
    name: str
    repo: str            # Hugging-Face-Repo mit GGUF-Dateien
    quant: str           # z. B. "Q5_K_M"
    groesse_gb: float
    notiz: str = ""


# Kuratierte Empfehlungen (16 GB VRAM, RDNA4/Vulkan). Default = Qwen3-14B Q5_K_M.
MODELLE: dict[str, ModellInfo] = {
    "qwen3-14b": ModellInfo("qwen3-14b", "Qwen3-14B", "Qwen/Qwen3-14B-GGUF", "Q5_K_M", 10.1,
                            "Empfehlung: Sweet Spot 16 GB, starke deutsche Rechtssprache"),
    "qwen3-8b": ModellInfo("qwen3-8b", "Qwen3-8B", "Qwen/Qwen3-8B-GGUF", "Q6_K", 6.7,
                           "mehr Kontext-Headroom, schneller"),
}
DEFAULT_MODELL = "qwen3-14b"


# ---------------------------------------------------------------------------
# Hugging-Face-Auflösung (kein Dateinamen-Raten – aktuelle Dateiliste abfragen)
# ---------------------------------------------------------------------------

def hf_dateien(repo: str, *, timeout: float = 25.0) -> list[str]:
    url = f"{HF}/api/models/{repo}"
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        data = json.load(r)
    return [s["rfilename"] for s in data.get("siblings", [])]


def passende_dateien(repo: str, quant: str, *, timeout: float = 25.0) -> list[str]:
    """GGUF-Dateien eines Repos, die zur Quantisierung passen (inkl. Shards)."""
    q = quant.lower()
    ggufs = [f for f in hf_dateien(repo, timeout=timeout) if f.lower().endswith(".gguf")]
    treffer = sorted(f for f in ggufs if q in f.lower())
    return treffer or sorted(ggufs)


def datei_url(repo: str, datei: str) -> str:
    return f"{HF}/{repo}/resolve/main/{urllib.parse.quote(datei)}?download=true"


# ---------------------------------------------------------------------------
# Download (streaming, resumebar)
# ---------------------------------------------------------------------------

def download_datei(
    url: str, ziel: Path, *, fortschritt: Callable[[int, int], None] | None = None,
    timeout: float = 60.0, blocksize: int = 1 << 20,
) -> Path:
    """Lädt ``url`` nach ``ziel`` – setzt einen Teildownload per HTTP-Range fort."""
    ziel.parent.mkdir(parents=True, exist_ok=True)
    vorhanden = ziel.stat().st_size if ziel.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "risk-layer/0.1"})
    if vorhanden:
        req.add_header("Range", f"bytes={vorhanden}-")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        # Gesamtgröße bestimmen (Content-Range bei Fortsetzung, sonst Content-Length)
        cr = resp.headers.get("Content-Range")
        gesamt = int(cr.split("/")[-1]) if cr else (
            int(resp.headers.get("Content-Length", 0)) + vorhanden)
        modus = "ab" if (vorhanden and resp.status == 206) else "wb"
        gelesen = vorhanden if modus == "ab" else 0
        with ziel.open(modus) as fh:
            while True:
                chunk = resp.read(blocksize)
                if not chunk:
                    break
                fh.write(chunk)
                gelesen += len(chunk)
                if fortschritt:
                    fortschritt(gelesen, gesamt)
    return ziel


def download_modell(
    info: ModellInfo, ziel_dir: Path | str = DEFAULT_MODELL_DIR,
    *, fortschritt: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Lädt alle zur Quantisierung passenden GGUF-Dateien des Modells."""
    ziel_dir = Path(ziel_dir)
    pfade: list[Path] = []
    for datei in passende_dateien(info.repo, info.quant):
        pfade.append(download_datei(datei_url(info.repo, datei), ziel_dir / datei,
                                    fortschritt=fortschritt))
    return pfade


def modell_pfad(info: ModellInfo, ziel_dir: Path | str = DEFAULT_MODELL_DIR) -> Path | None:
    """Pfad der (ersten) lokal vorhandenen GGUF-Datei des Modells, falls geladen."""
    ziel_dir = Path(ziel_dir)
    if not ziel_dir.exists():
        return None
    q = info.quant.lower()
    treffer = sorted(p for p in ziel_dir.glob("*.gguf") if q in p.name.lower())
    return treffer[0] if treffer else None


# ---------------------------------------------------------------------------
# Engine-Auto-Launch (llama.cpp llama-server)
# ---------------------------------------------------------------------------

def finde_llama_server(explizit: str | None = None,
                       engine_dir: Path | str = DEFAULT_ENGINE_DIR) -> str | None:
    """Findet das ``llama-server``-Binary (explizit, PATH oder lokale ./engine)."""
    if explizit:
        return explizit if Path(explizit).exists() else None
    auf_pfad = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if auf_pfad:
        return auf_pfad
    # lokal beschaffte Engine (siehe download_engine)
    engine_dir = Path(engine_dir)
    if engine_dir.exists():
        treffer = sorted(p for p in engine_dir.glob("**/llama-server*")
                         if p.is_file() and p.suffix.lower() in ("", ".exe"))
        if treffer:
            return str(treffer[0])
    return None


# ---------------------------------------------------------------------------
# Engine-Beschaffung (llama.cpp Vulkan-Release von GitHub)
# ---------------------------------------------------------------------------

def finde_engine_asset(pattern: str = "bin-win-vulkan-x64", *, timeout: float = 30.0
                       ) -> tuple[str, str] | None:
    """Sucht in den letzten Releases das passende Vulkan-Binary-Archiv."""
    req = urllib.request.Request(GITHUB_RELEASES, headers={
        "User-Agent": "risk-layer/0.1", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        releases = json.load(r)
    p = pattern.lower()
    for rel in releases:
        for a in rel.get("assets", []):
            name = a["name"].lower()
            if p in name and name.endswith(".zip"):
                return a["name"], a["browser_download_url"]
    return None


def download_engine(ziel_dir: Path | str = DEFAULT_ENGINE_DIR, *,
                    pattern: str = "bin-win-vulkan-x64",
                    fortschritt: Callable[[int, int], None] | None = None) -> str | None:
    """Lädt die llama.cpp-Vulkan-Engine und entpackt sie nach ``ziel_dir``.

    NETZZUGRIFF (Build-Schritt). Bindet KEINE GPU, startet nichts. Gibt den Pfad
    zum ``llama-server``-Binary zurück (oder None, falls nicht gefunden).
    """
    treffer = finde_engine_asset(pattern)
    if not treffer:
        raise RuntimeError(f"Kein Engine-Release zu '{pattern}' gefunden.")
    name, url = treffer
    ziel_dir = Path(ziel_dir)
    ziel_dir.mkdir(parents=True, exist_ok=True)
    zip_pfad = ziel_dir / name
    download_datei(url, zip_pfad, fortschritt=fortschritt)
    with zipfile.ZipFile(zip_pfad) as zf:
        zf.extractall(ziel_dir)
    zip_pfad.unlink()
    return finde_llama_server(engine_dir=ziel_dir)


class LlamaServerManager:
    """Startet/stoppt einen lokalen ``llama-server`` mit dem gewählten Modell."""

    def __init__(
        self, model_path: str | Path, *, binary: str | None = None,
        host: str = "127.0.0.1", port: int = 8080, n_ctx: int = 8192,
        n_gpu_layers: int = -1, extra: tuple[str, ...] = (),
    ):
        self.model_path = str(model_path)
        self.binary = finde_llama_server(binary)
        self.host = host
        self.port = port
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.extra = extra
        self._proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def kommando(self) -> list[str]:
        return [
            self.binary or "llama-server",
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),  # alle Layer auf die GPU (Vulkan)
            *self.extra,
        ]

    @property
    def laeuft(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.laeuft:
            return
        if not self.binary:
            raise FileNotFoundError(
                "llama-server nicht gefunden. Installiere die llama.cpp-Vulkan-Release "
                "und lege sie auf den PATH (oder übergib --llm-bin)."
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modell nicht gefunden: {self.model_path}")
        self._proc = subprocess.Popen(
            self.kommando(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def warte_bereit(self, timeout: float = 180.0, intervall: float = 1.5) -> bool:
        import time
        ende = time.time() + timeout
        while time.time() < ende:
            if not self.laeuft:
                return False
            if server_verfuegbar(self.url, timeout=2.0):
                return True
            time.sleep(intervall)
        return server_verfuegbar(self.url, timeout=2.0)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            # Sanftes terminate() zuerst – aber NICHT lange warten: steckt der Server
            # gerade in einer Inferenz, ignoriert er SIGTERM bis zum Ende (Sekunden).
            # Wir werfen den Prozess ohnehin weg, also nach kurzer Frist hart killen,
            # damit "Modell stoppen" im Browser nicht hängt.
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        self._proc = None


def menschlich(bytes_: int) -> str:
    g = bytes_ / (1 << 30)
    return f"{g:.2f} GB" if g >= 1 else f"{bytes_ / (1 << 20):.0f} MB"
