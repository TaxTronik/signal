"""Schicht-2-LLM-Provider (Stufe 1) /LF110/–/LF140/.

Lokale, offline-fähige Anbindung an llama.cpp. Das Modell darf ausschließlich
*zeigen* – die Sicherheit liegt im Code: Constrained Decoding (JSON-Schema) +
nachgelagerte Span-Verifikation (``schicht2.verifiziere_kandidaten``).

Zwei Pfade:
  * ``LlamaServerProvider`` – HTTP zu einem laufenden ``llama-server`` (empfohlen
    auf RDNA4, Vulkan-Backend). Nur Standardbibliothek.
  * ``LlamaCppProvider``    – in-process über das optionale ``llama-cpp-python``.

Beide liefern ROHE Kandidaten; das Gate verwirft alles, was nicht wörtlich im
Text steht. Modell-/Quantisierungs-/Seed-/Temperatur-Angaben gehen ins Audit
(/LF140/). Temperatur=0, fixierter Seed (Reproduzierbarkeit, /NF20/).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import PureWindowsPath
from typing import Any

from .schicht2 import KANDIDATEN_JSON_SCHEMA, SYSTEMPROMPT

_log = logging.getLogger("risk_layer.llm")


def _env_timeout(default: float = 300.0) -> float:
    """Schicht-2-Generierung kann bei großem Text / langsamer GPU minutenlang
    dauern. Default großzügig (sonst werden valide, nur späte Antworten
    verworfen); per ``RISK_LAYER_LLM_TIMEOUT`` (Sekunden) durch das Host-System
    tunebar."""
    roh = os.environ.get("RISK_LAYER_LLM_TIMEOUT")
    if roh:
        try:
            return float(roh)
        except ValueError:
            _log.warning("RISK_LAYER_LLM_TIMEOUT=%r ungültig — nutze %.0fs", roh, default)
    return default


def server_verfuegbar(base_url: str = "http://127.0.0.1:8080", *, timeout: float = 2.0) -> bool:
    """Schneller Erreichbarkeits-Check eines llama-server (GET /health).

    ``urlopen`` wirft fuer Status >= 400 einen ``HTTPError`` (Subklasse von
    ``URLError``) – ein ladender Server (503) landet also im ``except`` und gilt
    korrekt als NICHT verfuegbar. Der Statusvergleich deckt nur 2xx/3xx ab.
    """
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as r:  # noqa: S310
            return 200 <= r.status < 400
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def _hole_bytes(url: str, *, timeout: float, cap: int = 1_048_576) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
            return r.read(cap) if r.status == 200 else None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return None


def _hole_json(url: str, *, timeout: float) -> Any:
    roh = _hole_bytes(url, timeout=timeout)
    if roh is None:
        return None
    try:
        return json.loads(roh.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def _metrik(text: str, name: str) -> float | None:
    """Liest einen Prometheus-Zähler. Matcht den Metrik-Namen als GANZEN Token
    (auch mit ``{labels}``) – kein bloßes Präfix, sonst faengt z. B. die Suche nach
    ``foo`` faelschlich ``foo_total``."""
    for zeile in text.splitlines():
        zeile = zeile.strip()
        if not zeile or zeile.startswith("#"):
            continue
        teile = zeile.split()
        if len(teile) < 2:
            continue
        key = teile[0]
        if key == name or key.startswith(name + "{"):
            try:
                return float(teile[-1])
            except ValueError:
                return None
    return None


def _queue_info(base: str, *, timeout: float) -> dict[str, Any] | None:
    """Slot-/Queue-Zustand des llama-server. Primär ``/slots`` (direkte Sicht je
    Slot), Fallback ``/metrics`` (Prometheus). ``None``, wenn der Server beides
    nicht exponiert (llama-server braucht ``--slots`` bzw. ``--metrics``)."""
    slots = _hole_json(f"{base}/slots", timeout=timeout)
    if isinstance(slots, list):
        kompakt: list[dict[str, Any]] = []
        aktiv = 0
        for s in slots:
            if not isinstance(s, dict):
                continue
            # is_processing (neuere llama.cpp) ist maßgeblich; sonst state als INT
            # (0 = idle). Bewusst KEIN bool() auf state, damit ein String-state
            # ('idle') keinen idle-Slot faelschlich als aktiv zaehlt.
            ip = s.get("is_processing")
            if ip is not None:
                busy = bool(ip)
            else:
                st = s.get("state", 0)
                busy = isinstance(st, (int, float)) and st != 0
            aktiv += int(busy)
            kompakt.append({"id": s.get("id"), "aktiv": bool(busy)})
        return {"quelle": "slots", "slots_gesamt": len(kompakt),
                "aktiv": aktiv, "frei": len(kompakt) - aktiv, "slots": kompakt}
    roh = _hole_bytes(f"{base}/metrics", timeout=timeout)
    if roh is not None:
        txt = roh.decode("utf-8", "replace")
        proc = _metrik(txt, "llamacpp:requests_processing")
        wartend = _metrik(txt, "llamacpp:requests_deferred")
        if proc is not None or wartend is not None:
            return {"quelle": "metrics", "aktiv": proc, "wartend": wartend}
    return None


def server_status(base_url: str = "http://127.0.0.1:8080", *, timeout: float = 2.0) -> dict[str, Any]:
    """Erreichbarkeit + Queue/Slot-Zustand eines llama-server.

    Returns ``{erreichbar: bool, queue: {...}|None}``. ``queue`` zeigt die
    Auslastung (aktive/freie Slots bzw. processing/deferred), damit das
    Host-System entscheiden kann, ob es ``mitLLM:true`` jetzt sendet oder wartet.
    """
    base = base_url.rstrip("/")
    if not server_verfuegbar(base, timeout=timeout):
        return {"erreichbar": False, "queue": None}
    return {"erreichbar": True, "queue": _queue_info(base, timeout=timeout)}


def _user_prompt(segment_text: str, bekannte_begriffe: list[str],
                 kontext_normen: list[str] | None = None, abweichung: bool = False,
                 denk_suffix: str = "") -> str:
    bekannt = ", ".join(bekannte_begriffe) if bekannte_begriffe else "(keine)"
    kontext = ""
    if kontext_normen:
        kontext = ("Im Text/Korpus reale Normen (BEVORZUGT für norm_vorschlag, falls passend):\n- "
                   + "\n- ".join(kontext_normen) + "\n\n")
    if abweichung:
        bekannt_zeile = (
            "ABWEICHUNGS-MODUS: Ordne AUCH die folgenden, bereits erkannten Begriffe ein "
            "und nenne ihren norm_vorschlag (mit wörtlicher Stelle als span_text), damit "
            f"Abweichungen sichtbar werden: {bekannt}")
    else:
        bekannt_zeile = f"Bereits durch Schicht 1 erkannte Begriffe (NICHT erneut ausgeben): {bekannt}"
    return (
        f"{kontext}{bekannt_zeile}\n\n"
        f"--- TEXT ---\n{segment_text}\n--- ENDE ---\n\n"
        # denk_suffix MODELL-BEWUSST: Qwen3 braucht "/no_think", sonst geht es TROTZ
        # JSON-Schema in den Thinking-Modus und liefert nur hastigen Minimal-Output
        # (gemessen: 8 -> 1 Kandidat). Granite & Co. brauchen es NICHT (schadet Recall).
        f"Gib NUR das JSON nach Schema zurück.{denk_suffix}"
    )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(s: str) -> str:
    return _THINK_RE.sub("", s).strip()


def _kandidaten_aus_text(s: str) -> list[dict[str, Any]]:
    """Robuste Kandidaten-Extraktion.

    Erst striktes JSON. Schlägt das fehl (kleine Modelle brechen das JSON gern bei
    der Token-Grenze ab), werden die VOLLSTÄNDIGEN ``{...}``-Objekte des
    ``kandidaten``-Arrays einzeln geborgen – ein abgeschnittenes letztes Objekt
    wird verworfen. So liefert auch ein truncierter Lauf brauchbare Kandidaten.
    """
    s = _strip_think(s or "")
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            roh = d.get("kandidaten")
            # Typ-Validierung: kaputte Backends liefern ``kandidaten`` auch mal als
            # String/Objekt oder Listen mit Nicht-Dicts – das darf nicht erst
            # ausserhalb des try/except in _nachziehen crashen.
            if isinstance(roh, list):
                return [k for k in roh if isinstance(k, dict)]
            return []
    except json.JSONDecodeError:
        pass
    i = s.find('"kandidaten"')
    start_arr = s.find("[", i) if i >= 0 else s.find("[")
    if start_arr < 0:
        return []
    out: list[dict[str, Any]] = []
    tiefe = 0
    start = None
    in_string = False
    escape = False
    for j in range(start_arr, len(s)):
        c = s[j]
        # String-bewusster Scanner: ``{``/``}`` INNERHALB von JSON-Strings (z. B.
        # eine begruendung_kurz mit "{...}") duerfen die Klammertiefe nicht
        # verschieben, sonst zerschneidet die Bergung valide Objekte.
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            if tiefe == 0:
                start = j
            tiefe += 1
        elif c == "}" and tiefe > 0:
            tiefe -= 1
            if tiefe == 0 and start is not None:
                try:
                    obj = json.loads(s[start:j + 1])
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(obj, dict):
                        out.append(obj)
                start = None
    return out


def _finde_span(text: str, span: str) -> tuple[int, int]:
    """Lokalisiert ``span`` im Text. Returns (offset, länge) oder (-1, 0).

    Stufen: exaktes Vorkommen → whitespace-tolerantes Vorkommen (der Quelltext ist
    oft zeilenumbrochen, das Modell gibt Umbrüche als Leerzeichen zurück). Die
    zurückgegebene Spanne verweist immer auf den ECHTEN Quelltext – die Span-Treue
    (/AB30/) bleibt gewahrt, nur Whitespace gilt als äquivalent.
    """
    span = span.strip()
    if not span:
        return -1, 0
    i = text.find(span)
    if i >= 0:
        return i, len(span)
    teile = [re.escape(t) for t in span.split()]
    if not teile:
        return -1, 0
    m = re.compile(r"\s+".join(teile)).search(text)
    return (m.start(), m.end() - m.start()) if m else (-1, 0)


def _offset(wert: Any) -> int:
    """Defensives Parsen von ``char_offset``.

    0 ist ein GÜLTIGER Offset – ein ``or``-Fallback wuerde ihn verwerfen.
    None/Strings/Müll vom Modell ergeben -1 (⇒ Suche via ``_finde_span``)
    statt eines ValueError ausserhalb jedes try/except.
    """
    if wert is None or isinstance(wert, bool):
        return -1
    try:
        return int(wert)
    except (TypeError, ValueError):
        return -1


def _nachziehen(segment_text: str, roh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verifiziert + verortet jede Spanne im Quelltext (whitespace-tolerant).

    Nicht auffindbare Spannen werden verworfen – eine Halluzination kann so
    höchstens einen Fehlalarm erzeugen, nie einen erfundenen Rechtssatz.
    """
    if not isinstance(roh, list):
        return []
    out: list[dict[str, Any]] = []
    for k in roh:
        if not isinstance(k, dict):
            continue
        span = str(k.get("span_text") or "").strip()
        if not span:
            continue
        off = _offset(k.get("char_offset"))
        if 0 <= off and segment_text[off:off + len(span)] == span:
            laenge = len(span)
        else:
            off, laenge = _finde_span(segment_text, span)
            if off < 0:
                continue
        out.append({
            "span_text": segment_text[off:off + laenge],   # echte Quell-Spanne
            "char_offset": off,
            "begruendung_kurz": str(k.get("begruendung_kurz") or ""),
            "konfidenz": str(k.get("konfidenz") or "niedrig"),
            "norm_vorschlag": str(k.get("norm_vorschlag") or "").strip(),
        })
    return out


class LlamaServerProvider:
    """Kandidaten von einem lokalen llama-server (llama.cpp) über HTTP."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        modell: str = "unbekannt",
        quantisierung: str = "unbekannt",
        seed: int = 0,
        temperatur: float = 0.0,
        timeout: float | None = None,
        n_predict: int = 768,
        pruefe_verfuegbar: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.modell = modell
        self.quantisierung = quantisierung
        self.seed = seed
        self.temperatur = temperatur
        self.timeout = _env_timeout() if timeout is None else timeout
        self.n_predict = n_predict
        self.pruefe_verfuegbar = pruefe_verfuegbar
        # Von kandidaten() je Aufruf gesetzt, von _infer in den Prompt gezogen
        # (Template-Method – statt externer Injektion durch die Pipeline).
        # threading.local statt einfacher Attribute: ein GETEILTER Provider in
        # mehreren Threads (z. B. Web-Server) wuerde sich Kontext/Abweichung
        # sonst gegenseitig ueberschreiben. _infer behaelt bewusst die
        # 2-Argument-Signatur – Tests/Mocks ueberschreiben genau diese.
        self._je_aufruf = threading.local()
        # Modellname-Cache: None = noch nicht erfolgreich ermittelt (Finding:
        # ein Fehlschlag darf den Fallback nicht dauerhaft einfrieren).
        self._modell_cache: str | None = None

    def audit_info(self) -> dict[str, Any]:
        return {
            "engine": "llama-server",
            "modell": self.modell,
            "quantisierung": self.quantisierung,
            "seed": self.seed,
            "temperatur": self.temperatur,
        }

    def _modell_name(self) -> str:
        """Am Server geladener Modellname (via /v1/models) – fuer die modell-
        bewusste Prompt-Steuerung (z. B. Thinking-Suffix).

        Gecacht wird NUR eine erfolgreiche Antwort: schlaegt /v1/models fehl
        (Server laedt noch / kurzzeitig weg), bleibt der Cache leer und der
        naechste Aufruf fragt erneut. Sonst friert der Fallback-Name dauerhaft
        ein und die Qwen-Schalter (/no_think, enable_thinking=False) greifen
        fuer die gesamte Provider-Lebensdauer nie.
        """
        if self._modell_cache is not None:
            return self._modell_cache
        name = self.modell or ""
        d = _hole_json(f"{self.base_url}/v1/models", timeout=3.0)
        if isinstance(d, dict):
            daten = d.get("data") or d.get("models") or []
            if daten and isinstance(daten[0], dict):
                name = str(daten[0].get("id") or daten[0].get("name") or name)
                self._modell_cache = name          # nur Erfolg cachen
        return name

    def _denk_suffix(self) -> str:
        """Qwen3 braucht '/no_think' (sonst Thinking-Modus trotz JSON-Schema); andere
        Modelle (Granite, …) brauchen es nicht und es daempft deren Recall."""
        return " /no_think" if "qwen" in self._modell_name().lower() else ""

    def _infer(self, segment_text: str, bekannte_begriffe: list[str]) -> str:
        """Chat-Completion mit dem MODELL-EIGENEN Chat-Template (instruct!) +
        erzwungenem JSON-Schema (/LF110/). Gibt den Antworttext zurück."""
        # Je-Aufruf-Zustand aus dem threadlokalen Speicher (Defaults, falls
        # _infer direkt – etwa in Tests – ohne kandidaten() aufgerufen wird).
        kontext = getattr(self._je_aufruf, "kontext", None)
        abweichung = getattr(self._je_aufruf, "abweichung", False)
        body = {
            "messages": [
                {"role": "system", "content": SYSTEMPROMPT},
                {"role": "user", "content": _user_prompt(
                    segment_text, bekannte_begriffe, kontext,
                    abweichung, denk_suffix=self._denk_suffix())},
            ],
            "temperature": self.temperatur,
            "seed": self.seed,
            "max_tokens": self.n_predict,
            "cache_prompt": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "kandidaten", "schema": KANDIDATEN_JSON_SCHEMA, "strict": True},
            },
        }
        # Reasoning-Modelle (Qwen3.5+) denken sonst bis max_tokens und emittieren KEIN
        # content (alles in reasoning_content) -> 0 Kandidaten. enable_thinking=False
        # schaltet das ab (getestet: finish=stop, JSON in content). qwen-gated, da der
        # Kwarg nicht jedes Chat-Template kennt; /no_think bleibt als Fallback.
        if "qwen" in self._modell_name().lower():
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (lokal)
            data = json.load(resp)
        return data["choices"][0]["message"]["content"]

    def kandidaten(self, segment_text: str, bekannte_begriffe: list[str],
                   kontext_normen: list[str] | None = None,
                   abweichung: bool = False) -> list[dict[str, Any]]:
        self._je_aufruf.kontext = kontext_normen    # vom _infer in den Prompt gezogen
        self._je_aufruf.abweichung = abweichung
        # Readiness-Gate: lädt/down ⇒ sofort leer (kein 2-Minuten-Hänger).
        if self.pruefe_verfuegbar and not server_verfuegbar(self.base_url, timeout=2.0):
            return []
        try:
            roh_text = self._infer(segment_text, bekannte_begriffe)
        except TimeoutError:
            # Server erreichbar, aber Generierung > timeout: valide Kandidaten
            # gehen verloren. NICHT still – sonst unsichtbar (Diagnose-Falle).
            _log.warning(
                "Schicht-2 LLM-Timeout nach %.0fs (%s) — Kandidaten verworfen. "
                "RISK_LAYER_LLM_TIMEOUT erhöhen oder schnelleres Backend/Modell nutzen.",
                self.timeout, self.base_url)
            return []
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            _log.warning("Schicht-2 LLM nicht erreichbar (%s): %s — Kandidaten verworfen.",
                         self.base_url, exc)
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            # leere choices (IndexError), Felder fehlen (KeyError), Body kein JSON
            # (JSONDecodeError) — alles "Antwort unbrauchbar", nicht crashen.
            _log.warning("Schicht-2 LLM: unerwartetes Antwortformat (%s) — Kandidaten verworfen.", exc)
            return []
        return _nachziehen(segment_text, _kandidaten_aus_text(roh_text))


def _modell_dateiname(model_path: str) -> str:
    """Dateiname eines Modellpfads fuers Audit – robust fuer ``/`` UND ``\\``.

    ``PureWindowsPath`` akzeptiert beide Trenner; ein ``rsplit("/")`` liesse
    unter Windows den kompletten Pfad als Modellnamen stehen.
    """
    return PureWindowsPath(model_path).name


class LlamaCppProvider:
    """In-process-Variante über das optionale ``llama-cpp-python``."""

    def __init__(self, model_path: str, *, quantisierung: str = "unbekannt",
                 seed: int = 0, temperatur: float = 0.0, n_ctx: int = 4096, **llama_kwargs: Any):
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - optionaler Pfad
            raise ImportError("llama-cpp-python fehlt: pip install \".[llm]\"") from exc
        self.model_path = model_path
        self.modell = _modell_dateiname(model_path)
        self.quantisierung = quantisierung
        self.seed = seed
        self.temperatur = temperatur
        self._llm = Llama(model_path=model_path, seed=seed, n_ctx=n_ctx, verbose=False, **llama_kwargs)

    def audit_info(self) -> dict[str, Any]:
        return {"engine": "llama-cpp-python", "modell": self.modell,
                "quantisierung": self.quantisierung, "seed": self.seed, "temperatur": self.temperatur}

    def kandidaten(self, segment_text: str, bekannte_begriffe: list[str],
                   kontext_normen: list[str] | None = None,
                   abweichung: bool = False) -> list[dict[str, Any]]:  # pragma: no cover
        res = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEMPROMPT},
                {"role": "user", "content": _user_prompt(segment_text, bekannte_begriffe, kontext_normen, abweichung)},
            ],
            temperature=self.temperatur, seed=self.seed, max_tokens=256,
            response_format={"type": "json_object", "schema": KANDIDATEN_JSON_SCHEMA},
        )
        try:
            data = json.loads(_strip_think(res["choices"][0]["message"]["content"]))
        except (KeyError, json.JSONDecodeError):
            return []
        return _nachziehen(segment_text, data.get("kandidaten", []) or [])
