"""Gedrosselter Durchsatz-Benchmark des lokalen llama-server → Timeout-Kalibrierung.

Problem: Der feste Schicht-2-Timeout (``RISK_LAYER_LLM_TIMEOUT``, Default 300 s)
passt nicht auf jede Hardware. Auf einer billigen CPU oder langsamen GPU dauert
eine valide Antwort länger als der Default — jede Generierung wird abgerissen
und Schicht 2 fällt de facto komplett aus (nur als Log-Warnung sichtbar).

Lösung: EINMAL messen statt raten. Der Benchmark erhebt die reale Prompt-
(``pp``) und Generierungs-Geschwindigkeit (``tg``) des Systems und leitet
daraus einen Timeout ab, der das Worst-Case-Budget der Pipeline abdeckt
(voller Prompt + ``n_predict``, mit Sicherheitsfaktor). Das Ergebnis landet in
``.signal/llm_bench.json``; ``llm._env_timeout`` liest es mit der Auflösung
**Env-Variable > Kalibrierung > Default** — ein manuell gesetzter
``RISK_LAYER_LLM_TIMEOUT`` gewinnt also immer.

Drosselung (die Maschine muss während des Benchmarks bedienbar bleiben):
  * eigener llama-server mit HALBIERTEN Threads (``--threads``/``--threads-batch``)
  * Prozess-Priorität unter normal (Windows ``BELOW_NORMAL``, POSIX ``nice +10``)
  * ein einzelner, kurzer Messlauf (kleines ``n_predict``) statt Dauerlast

Die Drosselung unterschätzt die Produktions-Geschwindigkeit systematisch —
der berechnete Timeout fällt dadurch GRÖSSER aus. Das ist gewollt: ein zu
großer Timeout kostet nichts, ein zu kleiner verwirft valide Kandidaten.
Läuft bereits ein Server unter ``base_url``, wird GEGEN DIESEN gemessen
(keine Thread-Drossel möglich, aber auch kein zweites Modell im RAM — auf
knappen Maschinen wichtiger als die Drossel).
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import modell as M
from .llm import BENCH_DATEI, BENCH_SCHEMA, server_verfuegbar

# Worst-Case-Budget eines Schicht-2-Requests: Systemprompt + Segment + Kontext
# füllen bis ~halben n_ctx (8192); die Antwort ist auf n_predict=768 begrenzt
# (llm.LlamaServerProvider). Der Timeout muss GENAU dieses Budget überleben.
PROMPT_BUDGET_TOKENS = 4096
ANTWORT_BUDGET_TOKENS = 768
SICHERHEIT = 3.0
# Nie unter den bisherigen Default kalibrieren: schnelle Maschinen brauchen
# keinen knapperen Timeout (er kostet nichts), und Messrauschen nach unten
# darf keine funktionierende Installation verschlechtern.
TIMEOUT_MINIMUM = 300.0
MESS_N_PREDICT = 128
# Harte Obergrenze des Messlaufs selbst: schafft die Hardware nicht einmal
# den Mini-Lauf in dieser Zeit, ist sie für die 768-Token-Produktion
# hoffnungslos — dann ist "mitLLM aus lassen" die ehrliche Empfehlung.
MESS_TIMEOUT = 900.0


def _drossel_threads() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def _freier_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mess_prompt(ziel_zeichen: int = 4000) -> str:
    """Deterministischer, fachnaher Fließtext in Produktions-Prompt-Größe.

    Die Prompt-Verarbeitung skaliert (besonders auf CPU) nahezu linear mit der
    Länge — ein kurzer Kunstprompt würde ``pp`` drastisch überschätzen und der
    kalibrierte Timeout wäre zu knapp.
    """
    absatz = (
        "Die Gesellschaft veräußerte im Wirtschaftsjahr Anteile an einer "
        "Kapitalgesellschaft und bildete für den Veräußerungsgewinn eine "
        "Rücklage. Strittig ist, ob die Voraussetzungen der Übertragung "
        "stiller Reserven vorliegen und in welchem Umfang die Rücklage "
        "gewinnerhöhend aufzulösen ist. Zudem wurden Aufwendungen für die "
        "Überlassung von Wirtschaftsgütern zwischen Schwestergesellschaften "
        "geltend gemacht, deren betriebliche Veranlassung zu prüfen bleibt. "
    )
    wiederholungen = max(1, math.ceil(ziel_zeichen / len(absatz)))
    return ("Fasse die steuerlich relevanten Vorgänge zusammen:\n\n"
            + absatz * wiederholungen)


def empfehlung(
    pp_tok_s: float, tg_tok_s: float, *,
    prompt_budget: int = PROMPT_BUDGET_TOKENS,
    antwort_budget: int = ANTWORT_BUDGET_TOKENS,
    sicherheit: float = SICHERHEIT,
    minimum: float = TIMEOUT_MINIMUM,
) -> float:
    """Empfohlener Timeout in Sekunden (auf volle Minuten aufgerundet)."""
    if pp_tok_s <= 0 or tg_tok_s <= 0:
        raise ValueError("Token-Raten müssen > 0 sein")
    dauer = prompt_budget / pp_tok_s + antwort_budget / tg_tok_s
    return max(minimum, math.ceil(dauer * sicherheit / 60.0) * 60.0)


def _auswerten(antwort: dict[str, Any], dauer_s: float) -> dict[str, Any]:
    """Raten aus der Server-Antwort ziehen — ``timings`` (llama.cpp-Erweiterung)
    bevorzugt, sonst ``usage`` + Wanduhr.

    Der Wanduhr-Fallback rechnet die GESAMTE Dauer in beide Raten ein und
    unterschätzt sie damit systematisch — konservativ, der Timeout wird nur
    größer, nie zu knapp.
    """
    timings = antwort.get("timings")
    if isinstance(timings, dict):
        pp = timings.get("prompt_per_second")
        tg = timings.get("predicted_per_second")
        if (isinstance(pp, (int, float)) and pp > 0
                and isinstance(tg, (int, float)) and tg > 0):
            return {
                "quelle": "timings",
                "pp_tok_s": round(float(pp), 2),
                "tg_tok_s": round(float(tg), 2),
                "prompt_tokens": timings.get("prompt_n"),
                "antwort_tokens": timings.get("predicted_n"),
                "dauer_s": round(dauer_s, 1),
            }
    usage = antwort.get("usage") or {}
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(pt, int) and pt > 0 and isinstance(ct, int) and ct > 0 and dauer_s > 0:
        return {
            "quelle": "usage",
            "pp_tok_s": round(pt / dauer_s, 2),
            "tg_tok_s": round(ct / dauer_s, 2),
            "prompt_tokens": pt,
            "antwort_tokens": ct,
            "dauer_s": round(dauer_s, 1),
        }
    raise ValueError("Server-Antwort ohne verwertbare timings/usage")


def messe(base_url: str, *, n_predict: int = MESS_N_PREDICT,
          timeout: float = MESS_TIMEOUT) -> dict[str, Any]:
    """Ein einzelner Mess-Request gegen einen laufenden llama-server."""
    body = {
        "messages": [{"role": "user", "content": _mess_prompt()}],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": n_predict,
        # Kein Prompt-Cache: ein warmer Cache würde pp massiv überschätzen und
        # der kalibrierte Timeout wäre für den kalten Produktionsfall zu knapp.
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (lokal)
        antwort = json.load(resp)
    return _auswerten(antwort, time.monotonic() - start)


def bench(
    *,
    base_url: str = "http://127.0.0.1:8080",
    modell_key: str | None = None,
    binary: str | None = None,
    melde: Callable[[str], None] | None = None,
    zeitstempel: str | None = None,
) -> dict[str, Any]:
    """Kompletter Benchmark-Lauf: Server besorgen, messen, Empfehlung rechnen.

    Gibt das persistierbare Ergebnis-Dokument zurück (``speichere``); wirft bei
    Bedienfehlern (Modell/Binary fehlt, Server startet nicht) eine Exception.
    """
    sag = melde or (lambda _s: None)
    mgr: M.LlamaServerManager | None = None
    drossel: dict[str, Any]
    if server_verfuegbar(base_url, timeout=2.0):
        # Kein zweites Modell in den RAM laden — auf knappen Maschinen wiegt
        # das schwerer als die fehlende Thread-Drossel des Fremd-Servers.
        sag(f"Nutze bereits laufenden llama-server: {base_url} "
            "(misst den IST-Zustand, keine Thread-Drossel möglich).")
        url = base_url
        drossel = {"eigener_server": False}
    else:
        info = M.MODELLE.get(modell_key or M.DEFAULT_MODELL)
        if info is None:
            raise ValueError(f"unbekanntes Modell: {modell_key}")
        pfad = M.modell_pfad(info)
        if pfad is None:
            raise FileNotFoundError(
                "Modell nicht geladen. Erst: risk-layer modell download")
        threads = _drossel_threads()
        port = _freier_port()
        mgr = M.LlamaServerManager(
            pfad, binary=binary, port=port, niedrige_prioritaet=True,
            extra=("--threads", str(threads), "--threads-batch", str(threads)),
        )
        sag(f"Starte gedrosselten llama-server (Port {port}, "
            f"{threads}/{os.cpu_count() or '?'} Threads, niedrige Priorität) …")
        mgr.start()
        # Großzügig warten: das Modell-Laden selbst ist auf langsamen Platten/
        # CPUs Teil dessen, was diese Maschine eben braucht.
        if not mgr.warte_bereit(timeout=600.0):
            mgr.stop()
            raise RuntimeError(
                f"llama-server wurde nicht bereit (Log: {mgr.log_pfad})")
        url = mgr.url
        drossel = {"eigener_server": True, "threads": threads,
                   "cpu_kerne": os.cpu_count(), "niedrige_prioritaet": True}
    try:
        sag(f"Messlauf läuft (1 Request, n_predict={MESS_N_PREDICT}, "
            f"Abbruch nach {MESS_TIMEOUT:.0f} s) …")
        try:
            messung: dict[str, Any] | None = messe(url)
        except TimeoutError:
            messung = None
    finally:
        if mgr is not None:
            mgr.stop()
    ergebnis: dict[str, Any] = {
        "schema": BENCH_SCHEMA,
        "gemessen_am": zeitstempel or datetime.now(timezone.utc)
        .isoformat(timespec="seconds"),
        "drossel": drossel,
        "messung": messung,
        "budget": {"prompt_tokens": PROMPT_BUDGET_TOKENS,
                   "antwort_tokens": ANTWORT_BUDGET_TOKENS,
                   "sicherheit": SICHERHEIT},
    }
    if messung is None:
        ergebnis["empfohlener_timeout_s"] = None
        ergebnis["hinweis"] = (
            f"Messlauf überschritt {MESS_TIMEOUT:.0f} s für {MESS_N_PREDICT} "
            "Tokens — diese Hardware ist für Schicht 2 praktisch unbrauchbar. "
            "mitLLM aus lassen oder RISK_LAYER_LLM_TIMEOUT bewusst manuell setzen.")
    else:
        ergebnis["empfohlener_timeout_s"] = empfehlung(
            messung["pp_tok_s"], messung["tg_tok_s"])
    return ergebnis


def speichere(ergebnis: dict[str, Any], ziel: Path = BENCH_DATEI) -> Path:
    """Ergebnis atomar nach ``.signal/llm_bench.json`` schreiben (0600).

    Gleiche Symlink-Härtung wie beim Service-State (scripts/setup.sh):
    der Zustandsordner darf nicht auf fremde Ziele umgebogen werden.
    """
    if ziel.parent.is_symlink():
        raise RuntimeError("Zustands-Verzeichnis darf kein Symlink sein")
    ziel.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if ziel.is_symlink():
        raise RuntimeError("Kalibrierungs-Datei darf kein Symlink sein")
    payload = json.dumps(ergebnis, ensure_ascii=False, sort_keys=True) + "\n"
    tmp = ziel.parent / f".{ziel.name}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, ziel)
    finally:
        tmp.unlink(missing_ok=True)
    return ziel
