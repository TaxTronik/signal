"""Modell-Integration (Stufe 1): GGUF-Download + llama-server-Auto-Launch.

Zwei getrennte Belange:
  * Das **Modell** (GGUF) wird hier verwaltet und (resumebar) von Hugging Face
    geladen – ein einmaliger, vom Nutzer ausgelöster Build-Schritt.
  * Die **Engine** (llama.cpp ``llama-server``, RDNA4 → Vulkan-Backend) kann
    explizit aus einem festen, hash-gepinnten Release bezogen werden. Alternativ
    nutzt das Projekt ein manuell verwaltetes Binary auf dem PATH oder am
    übergebenen Pfad. ``LlamaServerManager`` startet es mit dem Modell, wartet
    auf Bereitschaft und beendet es wieder.

Zur Laufzeit der Analyse bleibt alles lokal (/NF10/): die App spricht den
lokalen llama-server über HTTP an (siehe ``llm.LlamaServerProvider``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .llama_cpp_manifest import (
    ENGINE_ASSETS,
    LLAMA_CPP_RELEASE,
    EngineAsset,
)
from .llm import server_verfuegbar
from .persistenz import atomar_schreiben

HF = "https://huggingface.co"
# Jede Registry-Zeile pinnt ihren eigenen vollständigen Commit. Branches oder
# Tags werden bewusst nicht akzeptiert: Dateiliste, Tree-Digests und Download
# müssen denselben unveränderlichen Repository-Stand verwenden.
DEFAULT_MODELL_DIR = Path("models")
DEFAULT_ENGINE_DIR = Path("engine")

_log = logging.getLogger(__name__)


def _hf_commit_revision(revision: str) -> str:
    """Validiert einen unveraenderlichen, vollstaendigen HF-Commit-Pin."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(
            "Hugging-Face-Revision muss ein vollstaendiger, kleingeschriebener "
            "40-stelliger Commit-Hash sein (keine Branches oder Tags)"
        )
    return revision


@dataclass(frozen=True)
class ModellInfo:
    key: str
    name: str
    repo: str            # Hugging-Face-Repo mit GGUF-Dateien
    revision: str        # voller, unveraenderlicher HF-Commit-Hash
    quant: str           # z. B. "Q5_K_M"
    groesse_gb: float
    notiz: str = ""

    def __post_init__(self) -> None:
        _hf_commit_revision(self.revision)


# Kuratierte Empfehlungen (16 GB VRAM, RDNA4/Vulkan). Default = Qwen3-14B Q5_K_M.
MODELLE: dict[str, ModellInfo] = {
    "qwen3-14b": ModellInfo(
        "qwen3-14b", "Qwen3-14B", "Qwen/Qwen3-14B-GGUF",
        "530227a7d994db8eca5ab5ced2fb692b614357fd", "Q5_K_M", 10.1,
        "Gruendlich-Modus: breiterer Recall, aber langsam (GEMV-limitiert auf RDNA4)"),
    "qwen3-8b": ModellInfo(
        "qwen3-8b", "Qwen3-8B", "Qwen/Qwen3-8B-GGUF",
        "7c41481f57cb95916b40956ab2f0b139b296d974", "Q6_K", 6.7,
        "mehr Kontext-Headroom, schneller"),
    # Qwen3.5-4B (kein offizielles GGUF -> unsloth-Community-GGUF). LLM-A/B gegen
    # granite: ist das LLM (Level-2-Verwertung) der Engpass? Klein + schnell, lokal.
    "qwen3.5-4b": ModellInfo(
        "qwen3.5-4b", "Qwen3.5-4B", "unsloth/Qwen3.5-4B-GGUF",
        "e87f176479d0855a907a41277aca2f8ee7a09523", "Q4_K_M", 2.5,
        "Qwen3.5 4B (unsloth GGUF); /no_think setzt der Provider selbst"),
    # Liquid LFM2.5-8B-A1B: 8B total / 1B aktiv (extrem sparse MoE) -> sehr schnell.
    # LLM-A/B-Kandidat. ACHTUNG: LFM2.5 ist neu -> braucht eine recente llama.cpp-Engine
    # (sonst 'unknown architecture' beim Laden -> ggf. 'modell engine' neu ziehen).
    "lfm2.5-8b-a1b": ModellInfo(
        "lfm2.5-8b-a1b", "LFM2.5-8B-A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF",
        "dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0", "Q5_K_M", 5.5,
        "Liquid LFM2.5 MoE (1B aktiv), offizielles GGUF"),
    # IBM Granite 4.1 (aktuelle Linie, Apache-2.0; die 4.x-Reihe setzt auf hybride
    # Mamba-2/Transformer-Architektur -> bandbreiten-freundlich, ideal fuer den hier
    # GPU-bandbreiten-limitierten Generierungs-Engpass). Kleinere Gewichte als das
    # 14B -> mehr tok/s. Head-to-Head gegen Qwen3-14B auf den Cases.
    # (Hybrid braucht eine recente llama.cpp; wird beim ersten Laden bestaetigt.)
    "granite-4.1-8b": ModellInfo(
        "granite-4.1-8b", "Granite-4.1-8B", "ibm-granite/granite-4.1-8b-GGUF",
        "865b82c2e7970d82e3731278c88c57ae7138359c", "Q5_K_M", 5.8,
        "Default: IBM 4.1 Hybrid (Mamba-2), Apache-2.0; schnell auf bandbreiten-limitierter GPU"),
    "granite-4.1-3b": ModellInfo(
        "granite-4.1-3b", "Granite-4.1-3B", "ibm-granite/granite-4.1-3b-GGUF",
        "ab4701481089b58a082ef63cc1cee738887293ff", "Q5_K_M", 2.1,
        "IBM 4.1, Apache-2.0; sehr schnell (Speed-Anker)"),
}
DEFAULT_MODELL = "granite-4.1-8b"


# ---------------------------------------------------------------------------
# Hugging-Face-Auflösung (kein Dateinamen-Raten – aktuelle Dateiliste abfragen)
# ---------------------------------------------------------------------------

def hf_dateien(repo: str, *, revision: str, timeout: float = 25.0) -> list[str]:
    # Revision explizit in der API-URL (gleicher Stand wie datei_url): Dateiliste
    # und Download dürfen nie aus zwei verschiedenen Revisionen stammen.
    revision = _hf_commit_revision(revision)
    url = f"{HF}/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}"
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        data = json.load(r)
    return [s["rfilename"] for s in data.get("siblings", [])]


def hf_lfs_digests(repo: str, *, revision: str,
                   timeout: float = 25.0) -> dict[str, str]:
    """sha256-Erwartungswerte je Datei (``pfad → hex``) aus dem HF-Tree-Endpunkt.

    Warum der Tree-Endpunkt: er ist ein simpler GET wie ``hf_dateien`` (kein
    POST-``paths-info``, kein ``huggingface_hub``-Paket — /NF10/-Geist: keine
    schweren Abhängigkeiten) und liefert für LFS-Dateien ``lfs.oid`` = sha256
    des Inhalts. Nicht-LFS-Dateien (README, configs) tragen keinen LFS-Digest
    und fehlen im Ergebnis — für sie greift dann das Sidecar-Pinning beim
    Erst-Download. ``recursive=true`` deckt Repos mit Unterordnern ab.
    """
    revision = _hf_commit_revision(revision)
    url = (f"{HF}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}"
           "?recursive=true")
    req = urllib.request.Request(url, headers={"User-Agent": "risk-layer/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        eintraege = json.load(r)
    digests: dict[str, str] = {}
    for e in eintraege:
        oid = (e.get("lfs") or {}).get("oid")
        if e.get("type") == "file" and oid:
            digests[e["path"]] = str(oid).lower()
    return digests


def passende_dateien(
    repo: str, quant: str, *, revision: str, timeout: float = 25.0,
) -> list[str]:
    """GGUF-Dateien eines Repos, die zur Quantisierung passen (inkl. Shards)."""
    q = quant.lower()
    ggufs = [
        f for f in hf_dateien(repo, revision=revision, timeout=timeout)
        if f.lower().endswith(".gguf")
    ]
    treffer = sorted(f for f in ggufs if q in f.lower())
    return treffer or sorted(ggufs)


def datei_url(repo: str, datei: str, *, revision: str) -> str:
    revision = _hf_commit_revision(revision)
    return (
        f"{HF}/{repo}/resolve/{revision}/{urllib.parse.quote(datei)}?download=true"
    )


# ---------------------------------------------------------------------------
# Integritätsnachweis (IDW PS 861, Mapping-Lücke L4): sha256-Sidecars
# ---------------------------------------------------------------------------

SIDECAR_SUFFIX = ".sha256"


def sidecar_pfad(datei: Path) -> Path:
    """``<datei>.sha256`` neben der Datei (sha256sum-Konvention)."""
    return datei.with_name(datei.name + SIDECAR_SUFFIX)


def sidecar_schreiben(datei: Path, digest: str) -> None:
    """Schreibt den Integritäts-Sidecar im exakten sha256sum-Format.

    Format ``<hex>  <dateiname>`` (ZWEI Leerzeichen, nur der Dateiname ohne
    Pfad): so prüft auch ein ``sha256sum -c <datei>.sha256`` ohne diese Engine
    — ein Wirtschaftsprüfer braucht kein Python für den Nachweis. Geschrieben
    wird atomar (persistenz.atomar_schreiben), damit nie ein halber Sidecar
    liegt; das Format verträgt keine Kommentarzeilen, Zusatzinfos gehören
    deshalb ins Log, nicht in die Datei.
    """
    atomar_schreiben(sidecar_pfad(datei), f"{digest.lower()}  {datei.name}\n")


def sidecar_lesen(datei: Path) -> str | None:
    """Erwarteter sha256 (hex, klein) aus dem Sidecar — ``None`` bei fehlend/unlesbar.

    Degradierend: ein fehlender/kaputter Sidecar wirft hier nie — die Befund-
    Seite ist ``verifiziere_artefakte`` (ok=False mit Grund), der Engine-Start
    sagt schlicht „ungeprüft" (LlamaServerManager.start).
    """
    try:
        erster = sidecar_pfad(datei).read_text(encoding="utf-8").split()[0].lower()
    except (OSError, IndexError):
        return None
    if len(erster) == 64 and all(c in "0123456789abcdef" for c in erster):
        return erster
    return None


def datei_sha256(pfad: str | Path, *, blocksize: int = 1 << 20) -> str:
    """sha256 einer Datei, blockweise gelesen (GB-Dateien, konstanter Speicher)."""
    del blocksize  # file_digest wählt die Blockgröße selbst; Param bleibt API-stabil
    with Path(pfad).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _pruefe_und_pinne(ziel: Path, ist: str, erwartet: str | None) -> None:
    """Digest-Abgleich + Sidecar-Pinning nach vollständigem Download.

    Mit Erwartungswert (z. B. HF-LFS- oder Engine-Manifest-Digest) ist eine
    Abweichung ein HARTER Fehler:
    die Datei wird entfernt — bliebe sie liegen, meldete der nächste Resume-
    Versuch sie per 416 still als „fertig" und das korrupte/manipulierte
    Artefakt ginge in Betrieb. Ohne Erwartungswert pinnt der Sidecar den
    Erst-Download (Trust-on-first-use); das wird GELOGGT statt in den Sidecar
    geschrieben, weil das sha256sum-Format keine Kommentare kennt.
    """
    if erwartet is not None and ist != erwartet.lower():
        ziel.unlink(missing_ok=True)
        sidecar_pfad(ziel).unlink(missing_ok=True)
        raise RuntimeError(
            f"Integritätsfehler bei {ziel.name}: sha256 weicht vom gepinnten "
            f"Erwartungswert ab (erwartet {erwartet.lower()}, ist {ist}) – das "
            "Artefakt ist manipuliert oder korrupt. Die Datei wurde entfernt; "
            "Download wiederholen (und bei erneuter Abweichung Quelle prüfen).")
    if erwartet is None:
        _log.info("kein HF-Erwartungsdigest für %s verfügbar — Sidecar pinnt den "
                  "Erst-Download (sha256 %s); künftige Prüfungen laufen dagegen "
                  "('risk-layer modell verify').", ziel.name, ist)
    sidecar_schreiben(ziel, ist)


def verifiziere_artefakte(verzeichnis: str | Path = DEFAULT_MODELL_DIR) -> list[dict]:
    """Prüft jedes Modellartefakt im Verzeichnis gegen seinen sha256-Sidecar.

    Vertrag (CLI ``risk-layer modell verify``): je Datei ein Dict
    ``{"datei", "erwartet", "ist", "ok", "grund"}`` — deterministisch nach
    Dateiname sortiert. Geprüft wird ALLES, was kein Sidecar ist (GGUF wie
    sonstige Artefakte): eine unbekannte Datei im Modellverzeichnis ist ein
    ehrliches ok=False („kein Sidecar"), kein stilles Auslassen. Sidecars ohne
    zugehörige Datei werden ebenfalls Befund (Datei fehlt).
    """
    verzeichnis = Path(verzeichnis)
    befunde: list[dict] = []
    if not verzeichnis.exists():
        return befunde   # kein Modellverzeichnis = nichts zu prüfen (kein Crash)
    kandidaten: set[str] = set()
    for p in verzeichnis.iterdir():
        # Nur Top-Level: download_modell legt Registry-Modelle flach ab.
        # Versteckte Dateien (atomar_schreiben-Temps ".<name>.tmp") sind keine
        # Artefakte und würden nur Pseudo-Befunde erzeugen.
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name.endswith(SIDECAR_SUFFIX):
            kandidaten.add(p.name[: -len(SIDECAR_SUFFIX)])   # Sidecar ohne Datei → Befund
        else:
            kandidaten.add(p.name)
    for name in sorted(kandidaten):
        datei = verzeichnis / name
        if not sidecar_pfad(datei).exists():
            befunde.append({"datei": name, "erwartet": None, "ist": None, "ok": False,
                            "grund": "kein Sidecar (.sha256) — Erst-Download mit aktueller "
                                     "Version nachholen oder Sidecar manuell erzeugen"})
            continue
        erwartet = sidecar_lesen(datei)
        if erwartet is None:
            # Sidecar liegt da, ist aber kein sha256sum-Format → eigener Grund,
            # NICHT „kein Sidecar": der Operator soll die kaputte Datei finden.
            befunde.append({"datei": name, "erwartet": None, "ist": None, "ok": False,
                            "grund": "Sidecar unlesbar (kein sha256sum-Format)"})
            continue
        if not datei.exists():
            befunde.append({"datei": name, "erwartet": erwartet, "ist": None, "ok": False,
                            "grund": "Datei fehlt (Sidecar vorhanden)"})
            continue
        ist = datei_sha256(datei)
        ok = ist == erwartet
        befunde.append({"datei": name, "erwartet": erwartet, "ist": ist, "ok": ok,
                        "grund": None if ok else "Digest weicht ab"})
    return befunde


# ---------------------------------------------------------------------------
# Download (streaming, resumebar)
# ---------------------------------------------------------------------------

def download_datei(
    url: str, ziel: Path, *, fortschritt: Callable[[int, int], None] | None = None,
    timeout: float = 60.0, blocksize: int = 1 << 20,
    erwartet_sha256: str | None = None,
) -> Path:
    """Lädt ``url`` nach ``ziel`` – setzt einen Teildownload per HTTP-Range fort.

    Integritätsnachweis (Mapping-Lücke L4): der sha256 wird STREAMEND während
    des Downloads mitberechnet — kein zweiter Festplatten-Pass über GB-Dateien.
    Nach Erfolg entsteht der Sidecar ``<datei>.sha256`` (sha256sum-Format); ist
    ``erwartet_sha256`` gesetzt (HF-LFS-Digest), ist eine Abweichung ein harter
    Fehler und die Datei wird entfernt.
    """
    ziel.parent.mkdir(parents=True, exist_ok=True)
    vorhanden = ziel.stat().st_size if ziel.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "risk-layer/0.1"})
    if vorhanden:
        req.add_header("Range", f"bytes={vorhanden}-")
    try:
        antwort = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
    except urllib.error.HTTPError as exc:
        # 416 = Range nicht erfüllbar: die Datei ist BEREITS VOLLSTÄNDIG geladen
        # (Resume ab Dateiende). Kein Fehler – erneutes "modell download" soll
        # sauber "fertig" melden statt mit Traceback abzubrechen. Die Gesamt-
        # größe steht (falls geliefert) im Content-Range ("bytes */<gesamt>").
        if exc.code == 416 and vorhanden:
            cr = exc.headers.get("Content-Range", "") if exc.headers else ""
            try:
                gesamt = int(cr.split("/")[-1])
            except ValueError:
                gesamt = vorhanden
            _log.info("bereits vollständig: %s (%s)", ziel.name, menschlich(vorhanden))
            if fortschritt:
                fortschritt(vorhanden, gesamt)
            # Integritätsnachweis NACHHOLEN: es lief kein Byte durch den Stream-
            # Hasher. Mit einem unabhängigen Erwartungswert MUSS die bereits
            # vorhandene Datei immer voll gehasht werden; ein daneben gelegter,
            # selbst ausgestellter Sidecar darf den Engine-Manifest-Pin nicht
            # umgehen. Nur im TOFU-Fall ohne Erwartungswert genügt ein bereits
            # vorhandener Sidecar. Das ist ein nutzerausgelöster Build-Schritt,
            # daher ist der einmalige Voll-Pass hier vertretbar.
            if erwartet_sha256 is not None or sidecar_lesen(ziel) is None:
                _pruefe_und_pinne(ziel, datei_sha256(ziel), erwartet_sha256)
            return ziel
        raise
    hasher = hashlib.sha256()
    with antwort as resp:
        if vorhanden and resp.status == 206:
            # echte Fortsetzung: Gesamtgröße aus Content-Range ("bytes a-b/<gesamt>")
            cr = resp.headers.get("Content-Range")
            gesamt = int(cr.split("/")[-1]) if cr else (
                int(resp.headers.get("Content-Length", 0)) + vorhanden)
            modus, gelesen = "ab", vorhanden
            # Resume: den schon liegenden Teil EINMALIG nachhashen, dann den
            # Netz-Stream weiterhashen. Warum hier (statt Vollhash nach dem
            # Abschluss): ob der Server 206 oder 200 antwortet, wissen wir erst
            # JETZT — und so liest die Platte jedes vorhandene Byte genau einmal,
            # der neue Teil kommt direkt aus dem Netz-Puffer. Bei GB-Dateien
            # dauert das Nachhashen Sekunden; die offene, noch ungelesene
            # HTTP-Antwort übersteht das problemlos.
            with ziel.open("rb") as fh:
                while True:
                    chunk = fh.read(blocksize)
                    if not chunk:
                        break
                    hasher.update(chunk)
        else:
            # Frischer Download ODER der Server IGNORIERT den Range-Header und
            # antwortet 200 mit der GANZEN Datei: dann neu schreiben statt anhängen
            # (sonst läge die Datei doppelt vor) – und Content-Length ist hier schon
            # die Gesamtgröße, ``vorhanden`` darf NICHT aufaddiert werden. Der
            # Hasher startet leer: alles, was in die Datei geht, läuft durch ihn.
            gesamt = int(resp.headers.get("Content-Length", 0))
            modus, gelesen = "wb", 0
        with ziel.open(modus) as fh:
            while True:
                chunk = resp.read(blocksize)
                if not chunk:
                    break
                fh.write(chunk)
                hasher.update(chunk)
                gelesen += len(chunk)
                if fortschritt:
                    fortschritt(gelesen, gesamt)
    _pruefe_und_pinne(ziel, hasher.hexdigest(), erwartet_sha256)
    return ziel


def download_modell(
    info: ModellInfo, ziel_dir: Path | str = DEFAULT_MODELL_DIR,
    *, fortschritt: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Lädt alle zur Quantisierung passenden GGUF-Dateien des Modells.

    Je Datei wird der sha256-Erwartungswert von Hugging Face (LFS-Digest)
    mitgeprüft, sofern abfragbar — Abweichung = harter Fehler (download_datei).
    """
    ziel_dir = Path(ziel_dir)
    # Erwartungswerte DEGRADIEREND beschaffen: schlägt die Tree-Abfrage fehl
    # (HF-API-Hickser, Proxy), läuft der Download trotzdem — dann pinnt der
    # Sidecar den Erst-Download (Log in _pruefe_und_pinne). Ein verfügbarer
    # Erwartungswert macht Abweichungen dagegen zum harten Fehler.
    try:
        erwartet = hf_lfs_digests(info.repo, revision=info.revision)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        _log.warning("HF-Digest-Abfrage für %s fehlgeschlagen (%s) — Download ohne "
                     "Erwartungswert, Sidecar pinnt den Erst-Download.", info.repo, exc)
        erwartet = {}
    pfade: list[Path] = []
    for datei in passende_dateien(info.repo, info.quant, revision=info.revision):
        pfade.append(download_datei(datei_url(info.repo, datei, revision=info.revision),
                                    ziel_dir / datei,
                                    fortschritt=fortschritt,
                                    erwartet_sha256=erwartet.get(datei)))
    return pfade


def modell_pfad(info: ModellInfo, ziel_dir: Path | str = DEFAULT_MODELL_DIR) -> Path | None:
    """Pfad der lokal vorhandenen GGUF-Datei des Modells, falls geladen.

    Matcht Repo-Stamm UND Quant: liegen mehrere Modelle mit gleicher Quant in
    ./models (z. B. Qwen3-14B-Q5_K_M und granite-4.1-8b-Q5_K_M), waehlt ein reiner
    Quant-Match sonst das alphabetisch erste – also potentiell das FALSCHE Modell.
    """
    ziel_dir = Path(ziel_dir)
    if not ziel_dir.exists():
        return None
    q = info.quant.lower()
    stamm = info.repo.split("/")[-1].lower().replace("-gguf", "")
    treffer = sorted(p for p in ziel_dir.glob("*.gguf")
                     if q in p.name.lower() and stamm in p.name.lower())
    if not treffer:   # Fallback (Einzelmodell-Setup): nur Quant
        treffer = sorted(p for p in ziel_dir.glob("*.gguf") if q in p.name.lower())
    return treffer[0] if treffer else None


# ---------------------------------------------------------------------------
# Engine-Auto-Launch (llama.cpp llama-server)
# ---------------------------------------------------------------------------

def _finde_llama_server_im_verzeichnis(verzeichnis: Path) -> str | None:
    if not verzeichnis.exists():
        return None
    treffer = sorted(p for p in verzeichnis.glob("**/llama-server*")
                     if p.is_file()
                     and p.name.lower() in ("llama-server", "llama-server.exe"))
    return str(treffer[0]) if treffer else None


def finde_llama_server(explizit: str | None = None,
                       engine_dir: Path | str = DEFAULT_ENGINE_DIR) -> str | None:
    """Findet ``llama-server`` (explizit, gepinnt verwaltet, PATH, Legacy-dir)."""
    if explizit:
        return explizit if Path(explizit).is_file() else None
    engine_dir = Path(engine_dir)
    # Ein von download_engine verifizierter aktueller Release hat Vorrang vor
    # einem zufällig/alt auf PATH liegenden Binary. Ein EXPLIZITER Pfad bleibt
    # als bewusst operatorverwaltete Ausnahme höchste Priorität.
    verwaltet = engine_dir / LLAMA_CPP_RELEASE
    verwaltetes_binary = _finde_llama_server_im_verzeichnis(verwaltet)
    if verwaltetes_binary:
        return verwaltetes_binary
    auf_pfad = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if auf_pfad:
        return auf_pfad
    # Legacy-/manuell abgelegte lokale Engine (vor Einführung des Release-dir).
    return _finde_llama_server_im_verzeichnis(engine_dir)


# ---------------------------------------------------------------------------
# Backend-Erkennung (GPU-Vendor → passendes llama.cpp-Release)
# ---------------------------------------------------------------------------

def _gpu_namen() -> list[str]:
    """GPU-Bezeichnungen (best effort, plattformabhängig, ohne Zusatzpakete)."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"],
                capture_output=True, text=True, timeout=15)
        else:
            out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return [z.strip() for z in out.stdout.splitlines() if z.strip()]


def erkenne_gpu_vendor() -> str | None:
    """``'nvidia'`` | ``'amd'`` | ``None`` – Best-Effort-Erkennung des Herstellers."""
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocminfo") or shutil.which("rocm-smi") or Path("/opt/rocm").exists():
        return "amd"
    blob = " ".join(_gpu_namen()).lower()
    if any(s in blob for s in ("nvidia", "geforce", "rtx", "quadro", "tesla")):
        return "nvidia"
    if any(s in blob for s in ("amd", "radeon", "instinct")):
        return "amd"
    return None


# Geordnete Asset-Muster (best → Fallback) je OS + Backend. Sie werden
# AUSSCHLIESSLICH gegen das im Repository gepinnte Manifest ausgewertet;
# ein neuer Upstream-Release wird nie still zur Laufzeit ausgewählt. Compute-
# Backends (CUDA/HIP) VOR Vulkan: schneller UND robust headless/RDP – Vulkan ist
# eine Grafik-API und fällt ohne aktives Display in einen langsamen Pfad. CPU zuletzt.
_BACKEND_WIN: dict[str, list[str]] = {
    "nvidia": ["bin-win-cuda-12", "bin-win-cuda-13", "bin-win-vulkan-x64", "bin-win-cpu-x64"],
    "cuda":   ["bin-win-cuda-12", "bin-win-cuda-13"],
    "amd":    ["bin-win-hip-radeon", "bin-win-vulkan-x64", "bin-win-cpu-x64"],
    "hip":    ["bin-win-hip-radeon"],
    "rocm":   ["bin-win-hip-radeon"],
    "vulkan": ["bin-win-vulkan-x64"],
    "cpu":    ["bin-win-cpu-x64"],
}
_BACKEND_LINUX: dict[str, list[str]] = {
    # llama.cpp liefert KEIN Linux-CUDA-Prebuilt → NVIDIA fällt auf Vulkan zurück
    # (volle CUDA-Speed nur via Source-/Docker-Build, siehe Deploy-Doku).
    "nvidia": ["bin-ubuntu-vulkan-x64", "bin-ubuntu-x64"],
    "cuda":   ["bin-ubuntu-vulkan-x64", "bin-ubuntu-x64"],
    "amd":    ["bin-ubuntu-rocm", "bin-ubuntu-vulkan-x64", "bin-ubuntu-x64"],
    "hip":    ["bin-ubuntu-rocm"],
    "rocm":   ["bin-ubuntu-rocm"],
    "vulkan": ["bin-ubuntu-vulkan-x64"],
    "cpu":    ["bin-ubuntu-x64"],
}


def backend_patterns(backend: str = "auto") -> list[str]:
    """Geordnete Release-Asset-Muster für OS + (erkanntes) Backend."""
    if backend == "auto":
        backend = erkenne_gpu_vendor() or "cpu"
    table = _BACKEND_WIN if sys.platform == "win32" else _BACKEND_LINUX
    if backend not in table:
        raise ValueError(f"unbekanntes llama.cpp-Backend: {backend!r}")
    return list(table[backend])   # Kopie: Tabelle bleibt unveraendert


# ---------------------------------------------------------------------------
# Engine-Beschaffung (fest gepinnter llama.cpp-Release, backend-aware)
# ---------------------------------------------------------------------------

def _waehle_engine_asset(pattern: str, *, cudart: bool = False) -> EngineAsset | None:
    """Wählt nur aus dem fest gepinnten Manifest, nie aus Netz-Metadaten."""
    treffer = _waehle_asset([(a.name, a.url) for a in ENGINE_ASSETS], pattern,
                            cudart=cudart)
    if treffer is None:
        return None
    return next(a for a in ENGINE_ASSETS if a.name == treffer[0])


def _pruefe_engine_plattform() -> None:
    """Auto-Downloads sind nur für die tatsächlich gepinnten Ziele erlaubt."""
    maschine = platform.machine().lower()
    if maschine not in {"amd64", "x86_64"}:
        raise RuntimeError(
            f"Kein gepinntes llama.cpp-Engine-Asset für Architektur {maschine!r}; "
            "llama-server manuell bauen/installieren und --llm-bin verwenden."
        )
    if sys.platform != "win32" and not sys.platform.startswith("linux"):
        raise RuntimeError(
            f"Kein gepinntes llama.cpp-Engine-Asset für Plattform {sys.platform!r}; "
            "llama-server manuell bauen/installieren und --llm-bin verwenden."
        )


def _waehle_asset(assets: list[tuple[str, str]], pattern: str, *, cudart: bool = False
                  ) -> tuple[str, str] | None:
    """Erstes Archiv, dessen Name ``pattern`` enthält. ``cudart`` trennt das
    CUDA-Runtime-Paket (``cudart-…``) vom eigentlichen Engine-Build."""
    p = pattern.lower()
    for name, url in assets:
        low = name.lower()
        if low.startswith("cudart") != cudart:
            continue
        if p in low and (low.endswith(".zip") or low.endswith(".tar.gz")):
            return name, url
    return None


def finde_engine_asset(pattern: str = "bin-win-vulkan-x64", *, timeout: float = 30.0
                       ) -> tuple[str, str] | None:
    """Passendes, hash-gepinntes Engine-Archiv des festen Releases.

    ``timeout`` bleibt aus API-Kompatibilität erhalten; die Funktion greift
    nicht mehr aufs Netz zu.
    """
    del timeout
    asset = _waehle_engine_asset(pattern)
    return (asset.name, asset.url) if asset else None


def _lade_und_entpacke(asset: EngineAsset, ziel_dir: Path,
                       fortschritt: Callable[[int, int], None] | None) -> None:
    name, url = asset.name, asset.url
    arch = ziel_dir / name
    # HARTER Abgleich VOR jeder Extraktion. download_datei entfernt das Archiv
    # bei einer Abweichung, sodass kein Byte daraus ausgeführt werden kann.
    download_datei(url, arch, fortschritt=fortschritt,
                   erwartet_sha256=asset.sha256)
    ist_groesse = arch.stat().st_size
    if ist_groesse != asset.groesse:
        arch.unlink(missing_ok=True)
        sidecar_pfad(arch).unlink(missing_ok=True)
        raise RuntimeError(
            f"Integritätsfehler bei {name}: Größe weicht vom gepinnten "
            f"Manifest ab (erwartet {asset.groesse}, ist {ist_groesse})."
        )
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(arch) as zf:
            # Zip-Slip: ein Member-Pfad darf nicht aus ziel_dir ausbrechen (../ oder
            # absolut). tar deckt das via filter="data" ab; ZIP kennt das nicht, also
            # vor dem Entpacken jeden Pfad gegen ziel_dir normalisieren.
            ziel = ziel_dir.resolve()
            for m in zf.namelist():
                try:
                    (ziel_dir / m).resolve().relative_to(ziel)
                except ValueError:
                    raise RuntimeError(f"unsicherer Pfad im ZIP abgewehrt (Zip-Slip): {m!r}") from None
            zf.extractall(ziel_dir)
    else:  # .tar.gz (Linux-Builds)
        with tarfile.open(arch) as tf:
            # filter="data" (Schutz gegen Tar-Slip/absolute Pfade/Sonderdateien) gibt
            # es erst ab Python 3.11.4 – deshalb requires-python >= 3.11.4. BEWUSST
            # kein Fallback ohne Filter: das wäre der Verlust genau dieses Schutzes.
            tf.extractall(ziel_dir, filter="data")
    arch.unlink()
    # Das Archiv ist transient (entpackt + gelöscht) — sein Download-Sidecar
    # darf nicht als verwaister Integritätsnachweis im Engine-Verzeichnis
    # zurückbleiben (verifiziere_artefakte meldete sonst "Datei fehlt").
    sidecar_pfad(arch).unlink(missing_ok=True)


def download_engine(ziel_dir: Path | str = DEFAULT_ENGINE_DIR, *,
                    backend: str = "auto", pattern: str | None = None,
                    fortschritt: Callable[[int, int], None] | None = None) -> str | None:
    """Lädt die zum ``backend`` passende, hash-gepinnte Engine und entpackt sie.

    ``backend``: ``'auto'|'cuda'|'hip'|'vulkan'|'cpu'``. ``'auto'`` erkennt den
    GPU-Hersteller (NVIDIA→CUDA, AMD→HIP) und nimmt Vulkan/CPU als Fallback.
    ``pattern`` überschreibt die Auto-Wahl, kann aber nur ein bereits im
    Repository-Manifest registriertes Asset auswählen.

    NETZZUGRIFF (Build-Schritt). Bindet KEINE GPU, startet nichts. Gibt den Pfad
    zum ``llama-server``-Binary zurück (oder None, falls nicht auffindbar).
    """
    _pruefe_engine_plattform()
    ziel_dir = Path(ziel_dir)
    if pattern is not None and not pattern.strip():
        raise ValueError("Engine-Asset-Muster darf nicht leer sein")
    muster = [pattern] if pattern else backend_patterns(backend)
    treffer: EngineAsset | None = None
    gewaehlt: str | None = None
    for p in muster:
        treffer = _waehle_engine_asset(p)
        if treffer:
            gewaehlt = p
            break
    if not treffer:
        raise RuntimeError(
            f"Kein Engine-Release für Backend '{backend}' gefunden "
            f"(geprüfte Muster: {', '.join(muster)}). Auf Linux+NVIDIA gibt es kein "
            "Prebuilt – llama.cpp aus Source oder via Docker (server-cuda) bauen.")
    plattform_marker = "bin-win-" if sys.platform == "win32" else "bin-ubuntu-"
    if plattform_marker not in treffer.name:
        raise RuntimeError(
            f"Das gepinnte Asset {treffer.name!r} passt nicht zur aktuellen "
            f"Plattform (erwarteter Namensmarker {plattform_marker!r})."
        )
    cudart: EngineAsset | None = None
    if gewaehlt and "cuda" in gewaehlt and sys.platform == "win32":
        cudart = _waehle_engine_asset(gewaehlt, cudart=True)
        if cudart is None:
            # Vor dem ersten Download abbrechen: ein unvollständiges oder nicht
            # ebenfalls hash-gepinntes CUDA-Bundle darf nicht entstehen.
            raise RuntimeError(
                f"Zum gepinnten CUDA-Asset {treffer.name!r} fehlt das ebenfalls "
                "gepinnt erwartete cudart-Paket im Engine-Manifest."
            )
    # Jeder autorisierte Release erhält einen eigenen Baum. Dadurch kann eine
    # alte, ehemals per latest bezogene Engine nicht versehentlich als Ergebnis
    # dieses Downloads zurückgegeben oder später vorgezogen werden.
    archiv_stamm = (treffer.name[:-7] if treffer.name.endswith(".tar.gz")
                     else Path(treffer.name).stem)
    installations_dir = ziel_dir / LLAMA_CPP_RELEASE / archiv_stamm
    installations_dir.mkdir(parents=True, exist_ok=True)
    _lade_und_entpacke(treffer, installations_dir, fortschritt)
    # CUDA: zugehörige Runtime-DLLs (cudart) ins gleiche Verzeichnis (Windows).
    if cudart is not None:
        _lade_und_entpacke(cudart, installations_dir, fortschritt)
    return _finde_llama_server_im_verzeichnis(installations_dir)


class LlamaServerManager:
    """Startet/stoppt einen lokalen ``llama-server`` mit dem gewählten Modell."""

    def __init__(
        self, model_path: str | Path, *, binary: str | None = None,
        host: str = "127.0.0.1", port: int = 8080, n_ctx: int = 8192,
        n_gpu_layers: int = -1, extra: tuple[str, ...] = (),
        niedrige_prioritaet: bool = False,
    ):
        self.model_path = str(model_path)
        self.binary = finde_llama_server(binary)
        self.host = host
        self.port = port
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.extra = extra
        # Für gedrosselte Läufe (llm_bench): der Server soll die Maschine
        # nicht lahmlegen — OS-Priorität unter normal statt harter CPU-Limits.
        self.niedrige_prioritaet = niedrige_prioritaet
        self._proc: subprocess.Popen | None = None
        self._log_datei: BinaryIO | None = None  # offenes Handle der Server-Logdatei
        self.log_pfad: str | None = None       # für Diagnose-Hinweise sichtbar

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
            # Damit /v1/llm/status die QUEUE liefert: /slots (Sicht je Slot) +
            # /metrics (Prometheus-Fallback). Ohne diese Flags antwortet der
            # llama-server nur auf /health → Queue bliebe null.
            "--slots",
            "--metrics",
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
        # Integritäts-Transparenz beim Start (L4): den Digest des geladenen GGUF
        # aus dem SIDECAR loggen — bewusst KEIN Vollhash an dieser Stelle (eine
        # GB-Datei zu hashen würde jeden Engine-Start um Sekunden bis Minuten
        # verzögern). Die echte Prüfung ist der explizite Schritt
        # 'risk-layer modell verify' (verifiziere_artefakte).
        digest = sidecar_lesen(Path(self.model_path))
        if digest:
            _log.info("GGUF-Digest (sha256 lt. Sidecar): %s — %s",
                      digest, Path(self.model_path).name)
        else:
            _log.info("GGUF ungeprüft — kein Sidecar für %s; Integrität mit "
                      "'risk-layer modell verify' bzw. erneutem 'modell download' "
                      "nachweisen.", Path(self.model_path).name)
        # stdout/stderr NICHT nach DEVNULL: Startfehler ("unknown architecture" bei
        # zu alter Engine, s. Hinweis bei lfm2.5 oben) wären sonst undiagnostizierbar.
        # Logdatei statt PIPE, damit der Server nie an einem vollen, ungelesenen
        # Pipe-Puffer blockiert; das Handle wird in stop() geschlossen.
        fd, self.log_pfad = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
        log_datei = os.fdopen(fd, "wb")
        self._log_datei = log_datei
        _log.info("llama-server-Log: %s", self.log_pfad)
        popen_kwargs: dict[str, Any] = {}
        if self.niedrige_prioritaet:
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
            else:
                popen_kwargs["preexec_fn"] = lambda: os.nice(10)
        try:
            self._proc = subprocess.Popen(
                self.kommando(),
                stdout=log_datei, stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
        except Exception:
            log_datei.close()           # Handle nicht verwaisen lassen
            self._log_datei = None
            raise

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

    def wait(self) -> int | None:
        """Blockiert, bis der Server-Prozess endet (z.B. via Strg+C im Vordergrund).
        Rückgabe: Exit-Code, oder None wenn kein Prozess läuft."""
        return self._proc.wait() if self._proc else None

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
        if self._log_datei is not None:
            # Log-Handle sauber freigeben (Windows hält die Datei sonst gesperrt);
            # die Datei selbst bleibt zur Diagnose liegen (Pfad: self.log_pfad).
            self._log_datei.close()
            self._log_datei = None


def menschlich(bytes_: int) -> str:
    g = bytes_ / (1 << 30)
    return f"{g:.2f} GB" if g >= 1 else f"{bytes_ / (1 << 20):.0f} MB"
