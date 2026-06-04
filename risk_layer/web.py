"""Headless /v1-Engine (HTTP-Server, nur Standardbibliothek).

Reiner Maschinen-Vertrag unter ``/v1/*`` (siehe ``api_v1.py`` und
INTEGRATION_TAXTRONIK.md) – **zustandslos** bzgl. Mandantendaten (§ 203 StGB):
Text rein, Analyse raus, kein Persist. Persistenz, Oberfläche und Audit übernimmt
das Host-System (TaxTronik).

    RISK_LAYER_TOKEN=<secret> python -m risk_layer.web --host 0.0.0.0

Auth über Bearer-Token (``RISK_LAYER_TOKEN``). Ohne gesetztes Token ist nur ein
Loopback-Bind erlaubt (lokale Entwicklung); ein Netz-Bind ohne Token wird
abgelehnt. Bewusst NUR stdlib – kein FastAPI/Node/Build-Schritt.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import sys
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .api_v1 import dispatch
from .berater import BeraterFehler, BeraterStore
from .catalog import lade_katalog
from .graph import graph_oeffnen
from .models import Begriffskarte, Risikomarkierung, RunReport, Status
from .pipeline import Pipeline
from .triggerindex import index_oeffnen
from .zitate import kanonisiere_fundstelle, parse_normzitat

_log = logging.getLogger(__name__)

# Loopback-Adressen: OHNE Token darf nur an diese gebunden werden (lokale Dev).
# Alles andere (0.0.0.0, ::, echte IPs, "" = alle Interfaces) ist ein Netz-Bind
# und erfordert ein RISK_LAYER_TOKEN.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _ist_loopback(host: str) -> bool:
    # Bewusst KEIN "": ein leerer Host bindet (wie 0.0.0.0) alle Interfaces.
    h = (host or "").strip().lower()
    return h in _LOOPBACK_HOSTS or h.startswith("127.")


# ---------------------------------------------------------------------------
# Serverzustand
# ---------------------------------------------------------------------------

class Anwendung:
    """Hält die vorkompilierte Pipeline + Graph/Trigger/Embedding für /v1."""

    def __init__(self, katalog_pfad: str | None, berater_dir: str, graph_pfad: str | None = None,
                 *, llm_url: str = "http://127.0.0.1:8080",
                 llm_model: str | None = None, llm_bin: str | None = None):
        self.katalog = lade_katalog(katalog_pfad)
        self.pipeline = Pipeline(self.katalog)
        self.berater_dir = Path(berater_dir)
        self.graph = graph_oeffnen(graph_pfad) if graph_pfad else graph_oeffnen()
        self.trigger = index_oeffnen()
        # Semantischer Embedding-Index (Schicht 1.7) – nur wenn vorab gebaut.
        # Encoder lazy, kein Startup-Download.
        from .embedding import index_oeffnen as _emb_open, encoder_oeffnen as _enc_open
        self.embedding = _emb_open()
        # Query-Encoder: GPU bevorzugt, fällt intern auf CPU zurück. Per
        # RISK_LAYER_EMB_DEVICE setzbar.
        _emb_dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
        try:
            self.embedding_encoder = (
                _enc_open(self.embedding.encoder_name, device=_emb_dev) if self.embedding else None)
        except Exception as exc:  # noqa: BLE001 – Schicht 1.7 ist optional, nie Show-Stopper
            _log.warning("Embedding-Encoder nicht ladbar (%s) — Engine laeuft ohne Schicht 1.7.", exc)
            self.embedding = self.embedding_encoder = None
        # Schicht-1.7b-Reranker (Cross-Encoder): NUR auf GPU – fp16 macht ihn ~25x
        # schneller (~1 s/Dok), auf CPU waere fp32 die 12-s-Falle. Lazy (Modell laedt
        # beim ersten Reranking). Hebt held-out-Recall ~2x; optional/degradierend.
        self.reranker = None
        if self.embedding_encoder is not None and "cuda" in (_emb_dev or ""):
            try:
                from .reranker import reranker_oeffnen
                self.reranker = reranker_oeffnen(device=_emb_dev)
            except Exception as exc:  # noqa: BLE001 – Rerank ist optional, nie Show-Stopper
                _log.warning("Reranker nicht ladbar (%s) — Schicht 1.7 ohne Rerank.", exc)
        self.llm_url = llm_url        # Schicht 2 (mitLLM:true) verbindet sich hierher
        self.llm_model = llm_model    # expliziter GGUF-Pfad (sonst Registry-Default)
        self.llm_bin = llm_bin        # llama-server-Binary
        self.manager = None           # LlamaServerManager, falls VON UNS gestartet
        self._llm_lock = threading.Lock()  # serialisiert llm_start/llm_stop (Concurrency)
        self._berater_lock = threading.Lock()  # serialisiert define-on-the-fly-Writes

    def norm_suche(self, q: str, limit: int = 8) -> dict[str, Any]:
        """Norm-Auswahl: erst Zitat parsen + im Graph bestätigen, sonst Titel-Suche."""
        q = (q or "").strip()
        if not q or self.graph is None:
            return {"treffer": []}
        treffer: list[dict[str, Any]] = []
        for a in parse_normzitat(q):
            if a.norm_id:
                r = self.graph.aufloesen(a.norm_id.to_id())
                if r.get("gefunden"):
                    treffer.append({"id": r["verwendet"], "titel": r.get("titel", ""), "zitat": q})
        if not treffer:
            treffer = self.graph.suche_titel(q, limit)
        return {"treffer": treffer}

    def _berater(self) -> BeraterStore | None:
        # frisch laden, damit gerade definierte Begriffe sofort wirken
        try:
            return BeraterStore(self.berater_dir)
        except BeraterFehler:
            return None

    def _provider(self, schicht2: bool, llm_url: str | None):
        if not schicht2:
            return None
        from .llm import LlamaServerProvider
        return LlamaServerProvider(llm_url or self.llm_url)

    def analyse(self, text: str, *, unknown: bool, berater: bool, nutzer: str | None,
                schicht2: bool = False, llm_url: str | None = None,
                abweichung: bool = False) -> dict[str, Any]:
        store = self._berater() if berater else None
        report = self.pipeline.run(text, unknown=unknown, berater=store, nutzer=nutzer,
                                   graph=self.graph, provider=self._provider(schicht2, llm_url),
                                   abweichung=abweichung, trigger=self.trigger,
                                   detektor=True, streit=True,
                                   embedding=self.embedding,
                                   embedding_encoder=self.embedding_encoder,
                                   reranker=self.reranker)
        return antwort_json(report, text)

    def definiere(self, payload: dict[str, Any]) -> dict[str, Any]:
        # load–modify–write serialisieren: parallele Defines auf dieselbe Scope-Datei
        # wuerden sich sonst ueberschreiben (Lost-Update der Festwissens-YAML).
        with self._berater_lock:
            store = BeraterStore(self.berater_dir)
            d = store.definiere(
                payload["begriff"],
                payload.get("definition_text", ""),
                autor=payload.get("autor") or "berater",
                datum=payload.get("datum") or date.today().isoformat(),
                scope=payload.get("scope", "personal"),
                governance_typ=payload.get("governance_typ", "IN"),
                schadensintensitaet=payload.get("schadensintensitaet", "mittel"),
                norm_anker=tuple(payload.get("norm_anker") or ()),
                konfidenz=payload.get("konfidenz", "mittel"),
                katalog_id=payload.get("katalog_id"),
            )
        return {"ok": True, "id": d.id, "scope": d.scope, "geankert_an": list(d.geankert_an),
                "katalog_id": d.katalog_id}

    def kuratiere_norm(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Katalog-weite Norm-Kuratierung (verwerfen/ergaenzen/zuruecksetzen) an einem
        Katalogbegriff. Katalog-State, fallzustandslos, KEIN Audit (der Host
        protokolliert die Handlung in seiner Chain)."""
        katalog_id = (payload.get("katalog_id") or "").strip()
        if not katalog_id or not any(e.id == katalog_id for e in self.katalog):
            return {"ok": False, "fehler": f"unbekannte katalog_id: {katalog_id!r}"}
        with self._berater_lock:
            store = BeraterStore(self.berater_dir)
            try:
                store.kuratiere_norm(
                    katalog_id, payload.get("norm", ""), payload.get("aktion", ""),
                    scope=payload.get("scope", "personal"),
                    autor=payload.get("autor") or "berater",
                    datum=payload.get("datum") or date.today().isoformat())
            except BeraterFehler as exc:
                return {"ok": False, "fehler": str(exc)}
        return {"ok": True, "katalog_id": katalog_id, "norm": payload.get("norm", ""),
                "aktion": payload.get("aktion", ""), "scope": payload.get("scope", "personal")}

    def kuratierung_lesen(self, *, katalog_id: str | None = None,
                          nutzer: str | None = None) -> dict[str, Any]:
        """Geltende Norm-Kuratierung (geteilt + persönliche des Nutzers) zum Überlagern
        im Host. Mit ``katalog_id`` die aufbereitete Sicht eines Begriffs
        (``verworfen``/``ergaenzt``), sonst alle Roh-Records. /v1/analyse bleibt
        davon unberührt (katalog-rein); der Host wendet die Kuratierung beim Rendern an."""
        store = self._berater()
        if store is None:
            return {"kuratierungen": []}
        if katalog_id:
            kur = store.kuratierung_fuer(katalog_id, nutzer=nutzer)
            return {"katalog_id": katalog_id, "verworfen": sorted(kur["verworfen"]),
                    "ergaenzt": kur["ergaenzt"]}
        return {"kuratierungen": [k.as_dict()
                                  for k in store.kuratierungen(nutzer=nutzer)]}

    # -- Schicht-2-LLM-Steuerung (über /v1/llm/*) ---------------------------
    def _modell_pfad(self) -> str | None:
        from . import modell as M
        if self.llm_model:
            return self.llm_model
        p = M.modell_pfad(M.MODELLE[M.DEFAULT_MODELL])
        return str(p) if p else None

    def llm_status(self) -> dict[str, Any]:
        """Erreichbarkeit + Modell/Engine + **Queue** (Slot-/Auslastungszustand)."""
        from . import modell as M
        from .llm import server_status
        st = server_status(self.llm_url)
        return {
            "url": self.llm_url,
            "verfuegbar": st["erreichbar"],
            "queue": st["queue"],
            "modell_geladen": bool(self._modell_pfad()),
            "binary_vorhanden": bool(M.finde_llama_server(self.llm_bin)),
            "von_uns_gestartet": bool(self.manager and self.manager.laeuft),
        }

    def llm_start(self) -> dict[str, Any]:
        """Startet llama-server mit dem SERVER-konfigurierten Modell (kein Request-
        Input → kein Injection-Vektor). Idempotent + thread-sicher; blockiert NICHT
        bis fertig – Bereitschaft über ``/v1/llm/status`` (``verfuegbar``) pollen."""
        from urllib.parse import urlparse
        from . import modell as M
        from .llm import server_verfuegbar
        with self._llm_lock:
            # Schon von UNS gestartet (auch noch im Modell-Lade-Fenster, in dem
            # /health NICHT antwortet) ODER extern erreichbar? → kein Doppel-Spawn.
            if (self.manager and self.manager.laeuft) or server_verfuegbar(self.llm_url):
                return {"ok": True, "hinweis": "läuft bereits", "url": self.llm_url}
            pfad = self._modell_pfad()
            if not pfad:
                return {"ok": False, "fehler": "Modell nicht geladen – 'risk-layer modell download'"}
            u = urlparse(self.llm_url)
            mgr = M.LlamaServerManager(pfad, binary=self.llm_bin,
                                       host=u.hostname or "127.0.0.1", port=u.port or 8080)
            try:
                mgr.start()   # Popen kehrt sofort zurück; das Laden läuft im Subprozess
            except FileNotFoundError as exc:
                return {"ok": False, "fehler": str(exc)}
            self.manager = mgr
            return {"ok": True, "startet": True, "modell": pfad, "url": mgr.url}

    def llm_stop(self) -> dict[str, Any]:
        with self._llm_lock:
            if self.manager:
                self.manager.stop()
                self.manager = None
        return {"ok": True}


# ---------------------------------------------------------------------------
# JSON-Aufbereitung (der /v1-Antwort-Vertrag)
# ---------------------------------------------------------------------------

def _norm_anker_json(norm_anker: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    """Katalog-Norm-Vorschläge einer Karte (pur). Die Berater-Kuratierung
    (verworfen/ergänzt) liefert separat GET /v1/katalog/kuratierung; der Host
    überlagert sie beim Rendern — /v1/analyse bleibt katalog-rein/reproduzierbar."""
    out: list[dict[str, Any]] = []
    for zitat in norm_anker:
        ids = [a.norm_id.to_id() for a in parse_normzitat(zitat) if a.norm_id]
        out.append({"zitat": zitat, "ids": ids})
    return out


def _quellen_json(quellen) -> list[dict[str, Any]]:
    return [
        {"typ": q.typ, "fundstelle": q.fundstelle, "id": kanonisiere_fundstelle(q.typ, q.fundstelle)}
        for q in quellen
    ]


def _karte_json(k: Begriffskarte) -> dict[str, Any]:
    e = k.eintrag
    return {
        "begriff_id": e.id,
        "begriff": e.begriff_kanonisch,
        "status": k.status.value,
        "herkunft": k.herkunft(),
        "matched_text": k.treffer.matched_text,
        "start": k.treffer.start,
        "end": k.treffer.end,
        "norm_anker": _norm_anker_json(e.norm_anker),
        "normketten": [kette.as_dict() for kette in e.normketten],
        "quellen": _quellen_json(k.quellen),
        "governance_typ": e.governance_typ.value,
        "wertungsoffen": k.wertungsoffen,
        "schadensintensitaet": e.schadensintensitaet,
        "kaskadenreichweite": e.kaskadenreichweite,
        "gefaehrdungsnotiz": e.gefaehrdungsnotiz.strip(),
        "rechercheauftrag": k.rechercheauftrag,
        "surface": k.treffer.surface,
        "gesetzestext": k.gesetzestext,
        "berater_notizen": k.berater_notizen,
        "abweichung": k.abweichung,
        # Provenienz für die Integration: "wörtlich" (Surface-Treffer) vs. "muster"
        # (Beschreibungs-Ko-Okkurrenz). via/ist_streitig/streit_signal sind Pflicht-
        # felder der Begriffskarte (Defaults im Dataclass) – direkter Zugriff, damit
        # ein Rename laut bricht statt still eine falsche Provenienz zu emittieren.
        "via": k.via,
        "ist_streitig": k.ist_streitig,
        "streit_signal": k.streit_signal,
    }


def _risiko_json(r: Risikomarkierung) -> dict[str, Any]:
    d = r.as_dict()
    d["norm_anker"] = _norm_anker_json(r.norm_anker)
    return d


def _spans_json(karten: list[Begriffskarte], risiken: list[Risikomarkierung]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for i, k in enumerate(karten):
        cls = "luecke" if k.status is Status.LUECKE else f"gov-{k.eintrag.governance_typ.value.lower()}"
        # Schicht 1b (Beschreibungsmuster) als eigener Layer ein-/ausblendbar,
        # bleibt aber Katalog-Vertrauenstier (deterministisch belegt).
        layer = "muster" if k.via == "muster" else "schicht1"
        spans.append({
            "start": k.treffer.start, "end": k.treffer.end, "status": k.status.value,
            "cls": cls, "titel": k.eintrag.begriff_kanonisch, "layer": layer,
            "via": k.via, "streitig": k.ist_streitig, "wertungsoffen": k.wertungsoffen,
            "ref": {"type": "karte", "index": i},
        })
    for j, r in enumerate(risiken):
        if r.status is Status.BERATER_DEFINIERT:
            cls, layer = "berater", "berater"
        elif r.quelle == "streit":
            cls, layer = "streit", "streit"
        elif r.quelle == "trigger":
            cls, layer = "trigger", "trigger"
        # Bewusst KEIN "embedding"-Layer: das Embedding erzeugt nie ein Risiko/Span,
        # sondern ausschließlich semantik_vorschlaege (Vorschlags-Modus).
        elif r.quelle == "llm":
            cls, layer = "unknown", "llm"
        else:
            cls, layer = "unknown", "heuristik"
        spans.append({
            "start": r.start, "end": r.end, "status": r.status.value,
            "cls": cls, "titel": r.titel, "layer": layer,
            "streitig": r.ist_streitig,
            "ref": {"type": "risiko", "index": j},
        })
    return spans


def antwort_json(report: RunReport, text: str) -> dict[str, Any]:
    return {
        "text": text,
        "katalog_version": report.katalog_version,
        "erzeugt_am": report.erzeugt_am,
        "text_hash": report.text_hash,
        "summary": report.zusammenfassung(),
        "spans": _spans_json(report.karten, report.risiken),
        "semantik": report.semantik_vorschlaege,
        "karten": [_karte_json(k) for k in report.karten],
        "risiken": [_risiko_json(r) for r in report.risiken],
    }


# ---------------------------------------------------------------------------
# HTTP-Handler (/v1)
# ---------------------------------------------------------------------------

# Obergrenze für Request-Bodies (Defense-in-depth gegen Memory-DoS). 8 MiB ist
# großzügig für Sachverhaltstexte. Auth läuft VOR dem Body-Read, daher kein
# Pre-Auth-Read mehr; der Cap deckelt zusätzlich authentifizierte Clients.
_MAX_BODY = 8 * 1024 * 1024


class _RequestFehler(Exception):
    """Client-Fehler im Request-Body (Größe/Form) → definierter HTTP-Status statt 500."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code
        self.msg = msg


def make_handler(app: Anwendung):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RiskLayer/0.1"
        # Socket-Timeout: eine hängende/langsame Verbindung darf nicht dauerhaft
        # einen Worker-Thread binden (slow-read/slowloris im Netzbetrieb).
        timeout = 30

        def log_message(self, *args):  # ruhiger Konsolen-Output
            pass

        # -- Helpers ----------------------------------------------------
        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            try:
                laenge = int(self.headers.get("Content-Length", 0) or 0)
            except (TypeError, ValueError):
                raise _RequestFehler(400, "ungültige Content-Length")
            if laenge < 0 or laenge > _MAX_BODY:
                raise _RequestFehler(413, f"Body zu groß (max {_MAX_BODY} Bytes)")
            roh = self.rfile.read(laenge) if laenge else b"{}"
            try:
                return json.loads(roh.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise _RequestFehler(400, "ungültiger JSON-Body")

        def _auth_ok(self) -> bool:
            """Bearer-Token-Prüfung (Integrationsvertrag), KONSTANT-ZEITIG.

            Token aus RISK_LAYER_TOKEN. Ist KEINS gesetzt, wird nicht geprüft –
            aber das ist nur bei Loopback-Bind erreichbar: ``serve()`` lehnt einen
            Netz-Bind ohne Token ab. Der Vergleich nutzt ``hmac.compare_digest``
            auf utf-8-Bytes (kein Timing-Orakel, robust gegen Nicht-ASCII-Token).
            """
            erwartet = os.environ.get("RISK_LAYER_TOKEN", "")
            if not erwartet:
                return True
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return False
            return hmac.compare_digest(auth[7:].encode("utf-8"), erwartet.encode("utf-8"))

        def _fehler_500(self, exc: Exception) -> None:
            # § 203 / Info-Leak: roher Exception-Text kann Mandantentext-Fragmente
            # oder interne Pfade enthalten → NICHT an den Client. Server-seitig voll
            # loggen (Betreiber-Log), dem Client nur die Fehlerklasse melden.
            _log.exception("Unbehandelter /v1-Fehler bei %s %s", self.command, urlparse(self.path).path)
            self._json(500, {"error": "interner Fehler", "typ": type(exc).__name__})

        # -- Routen (ausschließlich /v1) --------------------------------
        def do_GET(self):
            try:
                _p = urlparse(self.path)
                if not _p.path.startswith("/v1/"):
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                if not self._auth_ok():
                    self._json(401, {"error": "unauthorized"})
                    return
                code, obj = dispatch(app, "GET", _p.path, {}, parse_qs(_p.query))
                self._json(code, obj)
            except Exception as exc:  # noqa: BLE001 – einheitliche Fehlerantwort
                self._fehler_500(exc)

        def do_POST(self):
            try:
                pfad = self.path.split("?")[0]
                if not pfad.startswith("/v1/"):
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                # Auth VOR dem Body-Read: unauthentifizierte Requests werden
                # abgewiesen, bevor irgendein Body gelesen wird.
                if not self._auth_ok():
                    self._json(401, {"error": "unauthorized"})
                    return
                payload = self._read_json()
                code, obj = dispatch(app, "POST", pfad, payload, {})
                self._json(code, obj)
            except _RequestFehler as exc:
                self._json(exc.code, {"error": exc.msg})
            except BeraterFehler as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 – einheitliche Fehlerantwort
                self._fehler_500(exc)

    return Handler


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

def serve(*, host: str = "127.0.0.1", port: int = 8000, katalog: str | None = None,
          berater_dir: str = "definitionen", graph_pfad: str | None = None,
          llm_url: str = "http://127.0.0.1:8080",
          llm_model: str | None = None, llm_bin: str | None = None) -> None:
    # Netz-Bind ohne Token verweigern (§ 203 StGB): die /v1-Engine ist auth-pflichtig,
    # sobald sie das Loopback verlässt. Ohne Token nur lokal (Entwicklung).
    if not _ist_loopback(host) and not os.environ.get("RISK_LAYER_TOKEN", ""):
        print(f"Host {host!r} ist kein Loopback, aber RISK_LAYER_TOKEN ist nicht gesetzt. "
              "Setze ein Token (Bearer-Auth, § 203 StGB) oder binde an 127.0.0.1.",
              file=sys.stderr)
        return

    app = Anwendung(katalog, berater_dir, graph_pfad, llm_url=llm_url,
                    llm_model=llm_model, llm_bin=llm_bin)

    class _Server(ThreadingHTTPServer):
        # KEIN SO_REUSEADDR: ein zweiter Start auf belegtem Port soll laut scheitern,
        # nicht (unter Windows) still denselben Port doppelt binden.
        allow_reuse_address = False
        daemon_threads = True

    try:
        httpd = _Server((host, port), make_handler(app))
    except OSError as exc:
        print(f"Port {port} ist belegt – läuft die Engine schon? ({exc})", file=sys.stderr)
        return

    gstat = app.graph.stats() if app.graph else None
    auth = "Bearer-Token (gesetzt)" if os.environ.get("RISK_LAYER_TOKEN") else "KEIN Token (nur Loopback)"
    print(f"Risk-Layer /v1-Engine läuft auf http://{host}:{port}/v1/")
    print(f"  Katalog v{app.katalog.version} · {len(app.katalog)} Begriffe · Auth: {auth}")
    print(f"  Normgraph: {('%d Knoten' % gstat['knoten']) if gstat else 'nicht gebaut (nur Schicht 1)'} · "
          f"Embedding: {'geladen' if app.embedding else 'nicht gebaut'}")
    print("  (Strg+C zum Beenden – es verlässt nichts diese Maschine.)")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbeendet.")
    finally:
        app.llm_stop()        # einen von UNS gestarteten llama-server mit beenden
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(prog="risk-layer-web",
                                description="Headless /v1-Engine des Risk-Layer (zustandslos, Bearer-Auth).")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind-Host. Nicht-Loopback erfordert RISK_LAYER_TOKEN.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--katalog", default=None)
    p.add_argument("--berater", default="definitionen", help="Verzeichnis der Berater-Definitionen")
    p.add_argument("--graph", default=None, help="Pfad zur Normgraph-SQLite (Default: corpus/graph.sqlite)")
    p.add_argument("--llm-url", default="http://127.0.0.1:8080",
                   help="llama-server URL für Schicht 2 (mitLLM:true, /v1/llm/*)")
    p.add_argument("--llm-model", default=None, help="expliziter GGUF-Pfad (sonst aus Registry)")
    p.add_argument("--llm-bin", default=None, help="Pfad zum llama-server-Binary")
    args = p.parse_args(argv)
    serve(host=args.host, port=args.port, katalog=args.katalog,
          berater_dir=args.berater, graph_pfad=args.graph, llm_url=args.llm_url,
          llm_model=args.llm_model, llm_bin=args.llm_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
