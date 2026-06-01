"""Lokale Web-GUI (Stufe 3, benutzerfreundliche Oberfläche).

Bewusst NUR Standardbibliothek – kein FastAPI/Node/Build-Schritt. Damit läuft die
App offline auf dem Kanzlei-Rechner:

    python -m risk_layer.web         # öffnet den Browser auf 127.0.0.1:8000

Strikt lokal gebunden (127.0.0.1); kein externer Netzaufruf zur Laufzeit
(/NF10/, /NF60/). Sämtliche Verarbeitung passiert in-process über dieselbe
deterministische Pipeline wie die CLI.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .api_v1 import dispatch
from .audit import baue_audit
from .berater import BeraterFehler, BeraterStore
from .catalog import lade_katalog
from .fall import FallStore
from .graph import graph_oeffnen
from .triggerindex import index_oeffnen
from .models import Begriffskarte, Risikomarkierung, RunReport, Status
from .pipeline import Pipeline
from .report import report_html
from .zitate import kanonisiere_fundstelle, parse_entscheidung, parse_normzitat

STATIC = Path(__file__).parent / "static"
STATIC_RESOLVED = STATIC.resolve()
_EX = Path(__file__).parent.parent / "examples"

# Loopback-Adressen: nur an diese darf die volle GUI-/api-Oberfläche gebunden
# werden. Alles andere (0.0.0.0, ::, echte IPs, "" = alle Interfaces) ist ein
# Netz-Bind und erfordert den abgeschotteten --api-only-Modus mit Pflicht-Token.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _ist_loopback(host: str) -> bool:
    # Bewusst KEIN "": ein leerer Host bindet (wie 0.0.0.0) alle Interfaces, ist
    # also gerade NICHT loopback und muss in den --api-only-Modus gezwungen werden.
    h = (host or "").strip().lower()
    return h in _LOOPBACK_HOSTS or h.startswith("127.")
BEISPIELE = {
    "organschaft": _EX / "beispiel_organschaft.txt",
    "beschreibung": _EX / "beispiel_beschreibung.txt",
    "ao": _EX / "beispiel_ao.txt",
    "ao_edge": _EX / "beispiel_ao_edge.txt",
}


# ---------------------------------------------------------------------------
# Serverzustand
# ---------------------------------------------------------------------------

class Anwendung:
    """Hält die vorkompilierte Pipeline und den Berater-Store-Pfad."""

    def __init__(self, katalog_pfad: str | None, berater_dir: str, graph_pfad: str | None = None,
                 *, llm_url: str = "http://127.0.0.1:8080", llm_default: bool = True,
                 llm_model: str | None = None, llm_bin: str | None = None,
                 faelle_dir: str = "faelle"):
        self.katalog = lade_katalog(katalog_pfad)
        self.pipeline = Pipeline(self.katalog)
        self.berater_dir = Path(berater_dir)
        self.faelle = FallStore(faelle_dir)
        from .aufgabe import AufgabenStore
        self.aufgaben = AufgabenStore(Path(faelle_dir).parent / "aufgaben"
                                      if Path(faelle_dir).name == "faelle" else "aufgaben")
        self.graph = graph_oeffnen(graph_pfad) if graph_pfad else graph_oeffnen()
        self.trigger = index_oeffnen()
        # Semantischer Embedding-Index (Schicht 1.7) – nur wenn vorab gebaut.
        # Encoder lazy, kein Startup-Download.
        from .embedding import index_oeffnen as _emb_open, encoder_oeffnen as _enc_open
        self.embedding = _emb_open()
        # Query-Encoder: GPU bevorzugt (schnelleres Satz-Encoding); fällt intern auf
        # CPU zurück, falls keine GPU verfügbar ist. Per RISK_LAYER_EMB_DEVICE setzbar.
        _emb_dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
        self.embedding_encoder = (
            _enc_open(self.embedding.encoder_name, device=_emb_dev) if self.embedding else None)
        self.llm_url = llm_url
        self.llm_default = llm_default        # Schicht 2 läuft standardmäßig mit
        self.llm_model = llm_model            # expliziter GGUF-Pfad (sonst aus Registry)
        self.llm_bin = llm_bin                # llama-server-Binary
        self.manager = None                   # LlamaServerManager, falls von uns gestartet

    def _modell_pfad(self) -> str | None:
        from . import modell as M
        if self.llm_model:
            return self.llm_model
        p = M.modell_pfad(M.MODELLE[M.DEFAULT_MODELL])
        return str(p) if p else None

    def llm_status(self) -> dict[str, Any]:
        from .llm import server_verfuegbar
        from . import modell as M
        return {
            "url": self.llm_url,
            "standardmaessig_an": self.llm_default,
            "verfuegbar": server_verfuegbar(self.llm_url),
            "modell_geladen": bool(self._modell_pfad()),
            "binary_vorhanden": bool(M.finde_llama_server(self.llm_bin)),
            "von_uns_gestartet": bool(self.manager and self.manager.laeuft),
        }

    def llm_start(self) -> dict[str, Any]:
        """Startet llama-server mit dem Modell (Request blockiert NICHT bis fertig)."""
        from urllib.parse import urlparse
        from . import modell as M
        from .llm import server_verfuegbar
        if server_verfuegbar(self.llm_url):
            return {"ok": True, "hinweis": "läuft bereits"}
        pfad = self._modell_pfad()
        if not pfad:
            return {"ok": False, "fehler": "Modell nicht geladen – 'risk-layer modell download'"}
        u = urlparse(self.llm_url)
        self.manager = M.LlamaServerManager(pfad, binary=self.llm_bin,
                                            host=u.hostname or "127.0.0.1", port=u.port or 8080)
        try:
            self.manager.start()
        except FileNotFoundError as exc:
            self.manager = None
            return {"ok": False, "fehler": str(exc)}
        return {"ok": True, "startet": True, "modell": pfad, "url": self.manager.url}

    def llm_stop(self) -> dict[str, Any]:
        if self.manager:
            self.manager.stop()
            self.manager = None
        return {"ok": True}

    def abgleich(self, *, fetch: bool = True, alle: bool = False) -> dict[str, Any]:
        from . import abgleich as A
        treffer = A.laufe(self.katalog, berater=self._berater(), fetch=fetch, nur_neue=not alle)
        return {"treffer": [t.as_dict() for t in treffer]}

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
                                   embedding_encoder=self.embedding_encoder)
        return antwort_json(report, text)

    def report_html(self, text: str, *, unknown: bool, berater: bool, nutzer: str | None,
                    schicht2: bool = False, llm_url: str | None = None,
                    abweichung: bool = False) -> str:
        store = self._berater() if berater else None
        report = self.pipeline.run(text, unknown=unknown, berater=store, nutzer=nutzer,
                                   graph=self.graph, provider=self._provider(schicht2, llm_url),
                                   abweichung=abweichung, trigger=self.trigger,
                                   detektor=True, streit=True,
                                   embedding=self.embedding,
                                   embedding_encoder=self.embedding_encoder)
        return report_html(report, text)

    def definiere(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    def aufgabe_abschliessen(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Schließt eine Rechercheaufgabe ab und lässt ihr Ergebnis zurückfließen.

        Der Wachstums-Mechanismus: ist ein Begriff verankert und ein Ergebnis da,
        wird daraus eine Berater-Definition (define-on-the-fly) – an den Begriff
        geankert. So baut erledigte Delegation den gemeinsamen Katalog mit auf.
        """
        aufg = self.aufgaben.laden(payload.get("id", ""))
        if aufg is None:
            return {"ok": False, "error": "Aufgabe nicht gefunden"}
        aufg["ergebnis"] = str(payload.get("ergebnis", aufg.get("ergebnis", "")))
        aufg["ergebnis_quellen"] = list(payload.get("ergebnis_quellen") or aufg.get("ergebnis_quellen") or [])
        begriff = (aufg.get("anker") or {}).get("begriff") or ""
        def_id = None
        # Nur in den Katalog spiegeln, wenn ausdrücklich gewünscht UND ein Begriff
        # + Ergebnis vorliegt – sonst bleibt die Aufgabe nur erledigt (kein Zwang).
        if payload.get("in_katalog") and begriff and aufg["ergebnis"].strip():
            norm = (aufg.get("anker") or {}).get("norm") or ""
            res = self.definiere({
                "begriff": begriff, "definition_text": aufg["ergebnis"],
                "autor": aufg.get("an") or "berater",
                "scope": payload.get("scope", "personal"),
                "norm_anker": [norm] if norm else [],
            })
            def_id = res.get("id")
            aufg["berater_def_id"] = def_id
        aufg["status"] = "erledigt"
        gespeichert = self.aufgaben.speichern(aufg)
        return {"ok": True, "aufgabe": gespeichert, "berater_def_id": def_id}


# ---------------------------------------------------------------------------
# JSON-Aufbereitung
# ---------------------------------------------------------------------------

def _norm_anker_json(norm_anker: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
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
        "normketten": [kette.as_dict() for kette in getattr(e, "normketten", ())],
        "quellen": _quellen_json(k.quellen),
        "governance_typ": e.governance_typ.value,
        "schadensintensitaet": e.schadensintensitaet,
        "kaskadenreichweite": e.kaskadenreichweite,
        "gefaehrdungsnotiz": e.gefaehrdungsnotiz.strip(),
        "rechercheauftrag": k.rechercheauftrag,
        "surface": k.treffer.surface,
        "gesetzestext": k.gesetzestext,
        "berater_notizen": k.berater_notizen,
        "abweichung": k.abweichung,
        # Provenienz für die Integration: "wörtlich" (Surface-Treffer) vs. "muster"
        # (Beschreibungs-Ko-Okkurrenz). Bisher nur in _spans_json, nicht an der Karte
        # selbst – die TS-Seite mappt aber karten[], also hier explizit mitliefern.
        # via/ist_streitig/streit_signal sind Pflichtfelder der Begriffskarte (Defaults
        # im Dataclass) – direkter Zugriff, damit ein Rename laut bricht statt still
        # eine falsche Provenienz zu emittieren.
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
            "via": k.via, "streitig": k.ist_streitig,
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
        # sondern ausschließlich semantik_vorschlaege (Vorschlags-Modus). Ein
        # früherer Auto-Markier-Branch wurde verworfen (kein robuster Schwellwert).
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
        "audit": baue_audit(report),
    }


# ---------------------------------------------------------------------------
# HTTP-Handler
# ---------------------------------------------------------------------------

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}

# Obergrenze für Request-Bodies: schützt vor (im --api-only-Netzbetrieb sogar
# PRE-AUTH, da do_POST den Body vor der Token-Prüfung liest) Memory-DoS über eine
# riesige Content-Length. 8 MiB ist großzügig für Sachverhaltstexte.
_MAX_BODY = 8 * 1024 * 1024


class _RequestFehler(Exception):
    """Client-Fehler im Request-Body (Größe/Form) → definierter HTTP-Status statt 500."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code
        self.msg = msg


def make_handler(app: Anwendung, *, api_only: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RiskLayer/0.1"
        # Socket-Timeout: eine hängende/langsame Verbindung darf nicht dauerhaft
        # einen Worker-Thread binden (slow-read/slowloris im Netzbetrieb).
        timeout = 30
        # --api-only: nur das authentifizierte /v1 wird bedient. Die GUI-Oberfläche
        # (/static + /api/*) ist in diesem Modus vollständig abgeschaltet, damit
        # eine Netz-Exposition keine §203-Daten-Endpunkte ohne Auth erreichbar macht.
        _api_only = api_only

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

        def _v1_auth_ok(self) -> bool:
            """Bearer-Token-Prüfung (Integrationsvertrag), KONSTANT-ZEITIG.

            Token aus RISK_LAYER_TOKEN. Ist KEINS gesetzt, wird nicht geprüft –
            aber das ist nur im lokalen GUI-Modus erreichbar: ``serve()`` erzwingt,
            dass ohne Token ausschließlich an Loopback gebunden wird und dass
            ``--api-only`` ein Token voraussetzt. Damit ist der fail-open-Pfad
            nie netz-erreichbar. Der Vergleich nutzt ``hmac.compare_digest``
            (kein Timing-Orakel über ``==``).
            """
            erwartet = os.environ.get("RISK_LAYER_TOKEN", "")
            if not erwartet:
                return True
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return False
            # Byte-Vergleich: compare_digest auf str wirft bei Nicht-ASCII-Token
            # TypeError (→ 500 statt Auth). utf-8-Bytes sind robust UND konstant-zeitig.
            return hmac.compare_digest(auth[7:].encode("utf-8"), erwartet.encode("utf-8"))

        def _v1(self, method: str, pfad: str, payload: dict[str, Any] | None,
                query: dict) -> bool:
            """Behandelt /v1/* zentral. Gibt True zurück, wenn der Pfad ein
            /v1-Endpoint war (Antwort wurde gesendet), sonst False."""
            if not pfad.startswith("/v1/"):
                return False
            if not self._v1_auth_ok():
                self._json(401, {"error": "unauthorized"})
                return True
            code, obj = dispatch(app, method, pfad, payload or {}, query)
            self._json(code, obj)
            return True

        def _static(self, name: str):
            if self._api_only:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            pfad = (STATIC / name).resolve()
            # Pfad-GRENZE statt String-Präfix: STATIC_RESOLVED muss ein echter
            # Vorfahr sein. '..'-Anteile lösen nach außerhalb auf und fliegen damit
            # raus; ein Geschwisterordner wie 'static_secret' teilt zwar das
            # String-Präfix, ist aber kein parent → korrekt abgewiesen.
            if STATIC_RESOLVED not in pfad.parents or not pfad.is_file():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            ct = _CONTENT_TYPES.get(pfad.suffix, "application/octet-stream")
            self._send(200, pfad.read_bytes(), ct)

        # -- Routen -----------------------------------------------------
        def do_GET(self):
            try:
                _p = urlparse(self.path)
                if self._v1("GET", _p.path, None, parse_qs(_p.query)):
                    return
                if self._api_only:
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                if self.path in ("/", "/index.html"):
                    self._static("index.html")
                elif self.path == "/api/health":
                    graph_stats = app.graph.stats() if app.graph else None
                    self._json(200, {"status": "ok", "katalog_version": app.katalog.version,
                                     "begriffe": len(app.katalog), "graph": graph_stats,
                                     "llm": app.llm_status()})
                elif self.path.startswith("/api/beispiel"):
                    name = self.path.rsplit("/", 1)[-1]
                    pfad = BEISPIELE.get(name, BEISPIELE["organschaft"])
                    self._json(200, {"text": pfad.read_text(encoding="utf-8") if pfad.exists() else ""})
                elif self.path.startswith("/api/norm/suche"):
                    q = parse_qs(urlparse(self.path).query).get("q", [""])[0]
                    self._json(200, app.norm_suche(q))
                elif self.path == "/api/mandanten":
                    self._json(200, {"mandanten": app.faelle.mandanten()})
                elif self.path.split("?")[0] == "/api/aufgaben":
                    qs = parse_qs(urlparse(self.path).query)
                    self._json(200, {"aufgaben": app.aufgaben.liste(
                        mandant=qs.get("mandant", [None])[0],
                        fall_id=qs.get("fall_id", [None])[0],
                        an=qs.get("an", [None])[0],
                        nur_aktiv=qs.get("nur_aktiv", ["0"])[0] in ("1", "true"))})
                elif self.path.split("?")[0] == "/api/faelle":
                    mandant = parse_qs(urlparse(self.path).query).get("mandant", [None])[0]
                    self._json(200, {"faelle": app.faelle.liste(mandant)})
                elif self.path.startswith("/api/fall/"):
                    fid = unquote(self.path[len("/api/fall/"):])
                    fall = app.faelle.laden(fid)
                    self._json(200 if fall else 404, fall or {"error": "nicht gefunden"})
                elif self.path.startswith("/static/"):
                    self._static(self.path[len("/static/"):])
                else:
                    self._static(self.path.lstrip("/"))
            except Exception as exc:  # noqa: BLE001 – einheitliche Fehlerantwort
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            try:
                payload = self._read_json()
                if self._v1("POST", self.path.split("?")[0], payload, {}):
                    return
                if self._api_only:
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                if self.path == "/api/analyse":
                    self._json(200, app.analyse(
                        payload.get("text", ""),
                        unknown=bool(payload.get("unknown")),
                        berater=bool(payload.get("berater")),
                        nutzer=payload.get("nutzer"),
                        schicht2=bool(payload.get("schicht2", app.llm_default)),
                        llm_url=payload.get("llm_url"),
                        abweichung=bool(payload.get("abweichung")),
                    ))
                elif self.path == "/api/zitat":
                    self._json(200, self._zitat(payload))
                elif self.path == "/api/definiere":
                    self._json(200, app.definiere(payload))
                elif self.path == "/api/llm/start":
                    self._json(200, app.llm_start())
                elif self.path == "/api/llm/stop":
                    self._json(200, app.llm_stop())
                elif self.path == "/api/recherche/entwurf":
                    from .recherche import baue_entwurf
                    self._json(200, baue_entwurf(
                        payload.get("text", ""), payload.get("begriffe", []),
                        payload.get("frage")))
                elif self.path == "/api/abgleich":
                    self._json(200, app.abgleich(fetch=payload.get("fetch", True),
                                                 alle=bool(payload.get("alle"))))
                elif self.path == "/api/mandant/speichern":
                    self._json(200, app.faelle.mandant_speichern(
                        payload.get("slug", ""), payload.get("name", ""),
                        payload.get("mandantennummer", "")))
                elif self.path == "/api/aufgabe/speichern":
                    self._json(200, app.aufgaben.speichern(payload))
                elif self.path == "/api/aufgabe/status":
                    r = app.aufgaben.status_setzen(payload.get("id", ""), payload.get("status", ""))
                    self._json(200 if r else 404, r or {"error": "nicht gefunden"})
                elif self.path == "/api/aufgabe/loeschen":
                    self._json(200, {"ok": app.aufgaben.loeschen(payload.get("id", ""))})
                elif self.path == "/api/aufgabe/abschliessen":
                    # Ergebnis fließt zurück: als Berater-Definition (define-on-the-fly),
                    # an den Begriff der Aufgabe verankert. So wächst der Katalog durch
                    # erledigte Delegation – kollaborativer Zusammenbau.
                    self._json(200, app.aufgabe_abschliessen(payload))
                elif self.path == "/api/fall/speichern":
                    self._json(200, app.faelle.speichern(payload))
                elif self.path == "/api/fall/loeschen":
                    self._json(200, {"ok": app.faelle.loeschen(payload.get("id", ""))})
                elif self.path == "/api/recherche/export":
                    from .recherche import exportiere
                    self._json(200, exportiere(
                        payload.get("sachverhalt", ""), payload.get("begriffe", []),
                        payload.get("frage", ""),
                        webhook_url=payload.get("webhook_url") or None,
                        force=bool(payload.get("force"))))
                elif self.path == "/api/report":
                    html = app.report_html(
                        payload.get("text", ""),
                        unknown=bool(payload.get("unknown")),
                        berater=bool(payload.get("berater")),
                        nutzer=payload.get("nutzer"),
                        schicht2=bool(payload.get("schicht2", app.llm_default)),
                        llm_url=payload.get("llm_url"),
                        abweichung=bool(payload.get("abweichung")),
                    )
                    self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain; charset=utf-8")
            except _RequestFehler as exc:
                self._json(exc.code, {"error": exc.msg})
            except BeraterFehler as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 – einheitliche Fehlerantwort
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        @staticmethod
        def _zitat(payload: dict[str, Any]) -> dict[str, Any]:
            zitat = payload.get("zitat", "").strip()
            if payload.get("urteil"):
                uid = parse_entscheidung(zitat)
                if uid is None:
                    return {"typ": "urteil", "ids": [], "fehler": "kein Entscheidungszitat erkannt"}
                return {"typ": "urteil", "ids": [uid.to_id()], "aliase": list(uid.aliase)}
            anker = parse_normzitat(zitat)
            return {"typ": "norm", "anker": [a.as_dict() for a in anker]}

    return Handler


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

def serve(*, host: str = "127.0.0.1", port: int = 8000, katalog: str | None = None,
          berater_dir: str = "definitionen", graph_pfad: str | None = None,
          llm_url: str = "http://127.0.0.1:8080", llm_default: bool = True,
          llm_model: str | None = None, llm_bin: str | None = None, llm_autostart: bool = False,
          faelle_dir: str = "faelle", open_browser: bool = True, api_only: bool = False) -> None:
    # --- Surface-Guards (vor allem anderen) ----------------------------------
    # Zwei Oberflächen, strikt getrennt:
    #   • GUI/-api: bequem, browser-getrieben, NICHT auth-bar → nur an Loopback.
    #   • /v1: maschinen-API, Bearer-Token → einziges, das ans Netz darf (--api-only).
    token_gesetzt = bool(os.environ.get("RISK_LAYER_TOKEN", ""))
    if api_only and not token_gesetzt:
        print("--api-only erfordert ein RISK_LAYER_TOKEN (Pflicht-Auth, § 203 StGB). "
              "Setze die Variable und starte erneut.", file=sys.stderr)
        return
    if not api_only and not _ist_loopback(host):
        print(f"Host {host!r} ist kein Loopback: die GUI-/api-Oberfläche darf nicht "
              "ans Netz (keine Auth, persistiert Mandantendaten). Für eine "
              "Netz-Exposition '--api-only' nutzen (nur authentifiziertes /v1).",
              file=sys.stderr)
        return

    app = Anwendung(katalog, berater_dir, graph_pfad, llm_url=llm_url, llm_default=llm_default,
                    llm_model=llm_model, llm_bin=llm_bin, faelle_dir=faelle_dir)

    class _Server(ThreadingHTTPServer):
        # KEIN SO_REUSEADDR: ein zweiter Start auf belegtem Port soll laut scheitern,
        # nicht (unter Windows) still denselben Port doppelt binden.
        allow_reuse_address = False
        daemon_threads = True

    try:
        httpd = _Server((host, port), make_handler(app, api_only=api_only))
    except OSError as exc:
        print(f"Port {port} ist belegt – läuft die GUI schon? ({exc})", file=sys.stderr)
        return
    url = f"http://{host}:{port}/"
    gstat = app.graph.stats() if app.graph else None
    if api_only:
        print(f"Risk-Layer Engine (API-only) läuft auf http://{host}:{port}/v1/")
        print(f"  Katalog v{app.katalog.version} · {len(app.katalog)} Begriffe · "
              "Auth: Bearer-Token (Pflicht) · GUI + /api: ABGESCHALTET")
        print(f"  Normgraph: {('%d Knoten' % gstat['knoten']) if gstat else 'nicht gebaut (nur Schicht 1)'}")
    else:
        llm = app.llm_status()
        print(f"Risk-Layer GUI läuft lokal auf {url}")
        print(f"  Katalog v{app.katalog.version} · {len(app.katalog)} Begriffe · "
              f"Berater-Definitionen: {app.berater_dir}")
        print(f"  Normgraph: {('%d Knoten' % gstat['knoten']) if gstat else 'nicht gebaut (nur Schicht 1)'}")
        print(f"  Schicht 2 (LLM): standardmäßig an · {llm['url']} · "
              f"Modell {'geladen' if llm['modell_geladen'] else 'fehlt'} · "
              f"Engine {'gefunden' if llm['binary_vorhanden'] else 'fehlt'} · "
              f"{'erreichbar ✓' if llm['verfuegbar'] else 'nicht erreichbar'}")
    if llm_autostart:
        print("  Auto-Start llama-server:", app.llm_start())
    print("  (Strg+C zum Beenden – es verlässt nichts diese Maschine.)")
    sys.stdout.flush()
    if open_browser and not api_only:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbeendet.")
    finally:
        app.llm_stop()
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(prog="risk-layer-web", description="Lokale Web-GUI des Risk-Layer.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--katalog", default=None)
    p.add_argument("--berater", default="definitionen", help="Verzeichnis der Berater-Definitionen")
    p.add_argument("--faelle", default="faelle", help="Verzeichnis der Fall-Datenbank")
    p.add_argument("--graph", default=None, help="Pfad zur Normgraph-SQLite (Default: corpus/graph.sqlite)")
    p.add_argument("--llm-url", default="http://127.0.0.1:8080", help="llama-server URL (Schicht 2)")
    p.add_argument("--no-llm", action="store_true", help="Schicht 2 standardmäßig AUS")
    p.add_argument("--llm-model", default=None, help="expliziter GGUF-Pfad (sonst aus Registry)")
    p.add_argument("--llm-bin", default=None, help="Pfad zum llama-server-Binary")
    p.add_argument("--llm-autostart", action="store_true", help="llama-server beim Start hochfahren")
    p.add_argument("--no-browser", action="store_true", help="Browser nicht automatisch öffnen")
    p.add_argument("--api-only", action="store_true",
                   help="Nur authentifiziertes /v1 bedienen (GUI + /api abgeschaltet); "
                        "erfordert RISK_LAYER_TOKEN. Form für Netz-Exposition/Container.")
    args = p.parse_args(argv)
    serve(host=args.host, port=args.port, katalog=args.katalog,
          berater_dir=args.berater, graph_pfad=args.graph, faelle_dir=args.faelle,
          llm_url=args.llm_url, llm_default=not args.no_llm,
          llm_model=args.llm_model, llm_bin=args.llm_bin, llm_autostart=args.llm_autostart,
          open_browser=not args.no_browser, api_only=args.api_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
