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
import copy
import hmac
import json
import logging
import os
import sys
import threading
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .anonymize import ner_status
from .api_v1 import dispatch
from .berater import BeraterFehler, BeraterStore
from .catalog import lade_katalog
from .graph import DEFAULT_DB, graph_oeffnen
from .models import Begriffskarte, Risikomarkierung, RunReport, Status
from .pipeline import AnalyseOptionen, Pipeline, fusion_aktiviert
from .reranker import sim_rettung
from .radar import RadarFehler, lade_manifest, manifest_vorher_pfad, radar_bericht
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


# --- Token-Rotation (IDW PS 861, Mapping-Lücke L8) ---------------------------
# Merker, ob der Rotationshinweis in diesem Prozess schon geloggt wurde: ein
# Rotationsfenster kann tausende Requests umfassen — EIN Hinweis je Prozess
# reicht dem Betreiber als Erinnerung, alles weitere wäre Log-Rauschen. Das
# Lock verhindert Doppel-Logs aus parallelen Handler-Threads
# (ThreadingHTTPServer); Tests setzen den Merker per monkeypatch zurück.
_rotations_hinweis_geloggt = False
_rotations_hinweis_lock = threading.Lock()


def _logge_rotationsfenster_einmalig() -> None:
    global _rotations_hinweis_geloggt
    with _rotations_hinweis_lock:
        if _rotations_hinweis_geloggt:
            return
        _rotations_hinweis_geloggt = True
    # Bewusst OHNE Token-Werte (Secret-Hygiene wie bei _401/_llm_url_anzeige):
    # die Meldung sagt nur, DASS das Fenster genutzt wird, nie WOMIT.
    _log.info("Rotationsfenster aktiv — Vorgänger-Token in Verwendung; nach "
              "Abschluss RISK_LAYER_TOKEN_VORGAENGER leeren")


def _llm_url_anzeige(url: str | None) -> str | None:
    """Nur ``host:port`` für den Engine-Status (Secret-Hygiene).

    Warum gekürzt: der Status wandert in /v1-Antworten, die der Host in seiner
    Audit-Chain ARCHIVIERT. Eine konfigurierte LLM-URL kann aber Userinfo-,
    Pfad- oder Query-Bestandteile mit Tokens tragen – die gehören nicht in
    dauerhaft abgelegte Dokumente. host:port reicht als Deployment-Nachweis."""
    if not url:
        return None
    u = urlparse(url)
    if not u.hostname:
        return None
    return f"{u.hostname}:{u.port}" if u.port else u.hostname


# Fehlertext für /v1/los/* mit backend=qpu ohne IBM-Zugang (503): nennt beide
# Abhilfen (Credentials hinterlegen ODER bewusst anderes Backend) UND den
# Install-Hinweis — bewusst KEINE stille Degradation auf lokalen Zufall
# (Attestierungs-Anspruch des Quantenlos, s. quantenlos.py-Docstring).
_IBM_CREDENTIALS_FEHLEN = (
    "backend=qpu braucht IBM-Quantum-Credentials (~/.qiskit/qiskit-ibm.json oder "
    "QISKIT_IBM_TOKEN) — Zugang hinterlegen (QiskitRuntimeService.save_account) "
    "oder das Backend bewusst anders wählen (simulator|csprng); stille Degradation "
    "auf lokalen Zufall gibt es nicht. Quanten-Extras: "
    'pip install "risk-layer[quanten]".')


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
        # Graph-DB-PFAD festhalten, nicht nur das geöffnete Objekt: das Radar
        # (radar()) leitet daraus die Manifest-Pfade ab — die norm_manifest*.json
        # liegen per Konvention NEBEN der Graph-Datei (radar.manifest_vorher_pfad),
        # und im kein-Graph-Fall gäbe es sonst keinen Pfad mehr zum Benennen.
        # Default-Konvention wie CLI/graph_oeffnen: corpus/graph.sqlite.
        self.graph_pfad = Path(graph_pfad) if graph_pfad else DEFAULT_DB
        self.graph = graph_oeffnen(self.graph_pfad)
        self.trigger = index_oeffnen()
        # Semantischer Embedding-Index (Schicht 1.7) – nur wenn vorab gebaut.
        # Encoder lazy, kein Startup-Download.
        from .embedding import (index_oeffnen as _emb_open, encoder_oeffnen as _enc_open,
                                pruefe_aktualitaet as _emb_aktuell)
        self.embedding = _emb_open()
        # Graph-Fingerprint EINMAL berechnen und am Objekt memoisieren: der
        # direkt folgende Staleness-Check (pruefe_aktualitaet) ruft
        # graph.fingerprint() intern erneut auf – ohne Memo liefe der Korpus-
        # Voll-Scan doppelt. Der Graph ist während der Server-Laufzeit
        # UNVERÄNDERLICH (Rebuilds laufen als separate CLI-Prozesse), der
        # eingefrorene Wert ist also gefahrlos; er wandert zusätzlich in den
        # Engine-Status (Degradations-Nachweis, s. unten).
        graph_fp: str | None = None
        if self.graph is not None:
            try:
                graph_fp = self.graph.fingerprint()
                self.graph.fingerprint = lambda _fp=graph_fp: _fp  # type: ignore[method-assign]
            except Exception:  # noqa: BLE001 – Status nie Show-Stopper; bewusst
                pass           # KEIN Log: pruefe_aktualitaet meldet den Fall gleich selbst.
        # Staleness-Check EINMALIG beim Start (nie pro Request): warnt laut im
        # Log, wenn der Embedding-Index nicht mehr zum aktuellen Normgraphen
        # passt (Fingerprint-Abweichung). Degradierend – wirft nie. Das Ergebnis
        # wird für den Engine-Status abgegriffen (index_aktuell; None = kein
        # Index geladen, also nichts zu prüfen).
        index_aktuell: bool | None = (
            _emb_aktuell(self.embedding, self.graph) if self.embedding is not None else None)
        # Index-Metadaten VOR dem Encoder-Versuch sichern: schlägt der Encoder
        # fehl, wird self.embedding auf None degradiert – der Status soll aber
        # weiter belegen, WELCHER Index benutzt worden wäre.
        emb_index_fp = getattr(self.embedding, "graph_fingerprint", None)
        emb_encoder_name = getattr(self.embedding, "encoder_name", None)
        # Degradations-GRÜNDE zusätzlich zum Log in den Engine-Status schreiben
        # (Log-Verhalten unverändert): der Host legt den Status mit jedem
        # Analyseergebnis in seiner Audit-Chain ab (Haftungsakte).
        emb_grund: str | None = None if self.embedding is not None else "kein Embedding-Index gebaut"
        # Query-Encoder: GPU bevorzugt, fällt intern auf CPU zurück. Per
        # RISK_LAYER_EMB_DEVICE setzbar.
        _emb_dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
        try:
            self.embedding_encoder = (
                _enc_open(self.embedding.encoder_name, device=_emb_dev) if self.embedding else None)
        except Exception as exc:  # noqa: BLE001 – Schicht 1.7 ist optional, nie Show-Stopper
            _log.warning("Embedding-Encoder nicht ladbar (%s) — Engine laeuft ohne Schicht 1.7.", exc)
            emb_grund = f"Embedding-Encoder nicht ladbar: {exc}"
            self.embedding = self.embedding_encoder = None
        # Schicht-1.7b-Reranker (Cross-Encoder): NUR auf GPU – fp16 macht ihn ~25x
        # schneller (~1 s/Dok), auf CPU waere fp32 die 12-s-Falle. Lazy (Modell laedt
        # beim ersten Reranking). Hebt held-out-Recall ~2x; optional/degradierend.
        self.reranker = None
        reranker_grund: str | None = None
        if self.embedding_encoder is not None and "cuda" in (_emb_dev or ""):
            try:
                from .reranker import reranker_oeffnen
                self.reranker = reranker_oeffnen(device=_emb_dev)
                if self.reranker is None:
                    # reranker_oeffnen("auto") degradiert ohne sentence-transformers
                    # STILL zu None – für die Haftungsakte trotzdem benennen.
                    reranker_grund = "Cross-Encoder nicht verfügbar (sentence-transformers fehlt)"
            except Exception as exc:  # noqa: BLE001 – Rerank ist optional, nie Show-Stopper
                _log.warning("Reranker nicht ladbar (%s) — Schicht 1.7 ohne Rerank.", exc)
                reranker_grund = f"Reranker nicht ladbar: {exc}"
        elif self.embedding_encoder is None:
            reranker_grund = "ohne Embedding-Encoder kein Reranking (Schicht 1.7 inaktiv)"
        else:
            reranker_grund = f"nur auf GPU (RISK_LAYER_EMB_DEVICE={_emb_dev!r} ist kein cuda-Device)"
        self.llm_url = llm_url        # Schicht 2 (mitLLM:true) verbindet sich hierher
        self.llm_model = llm_model    # expliziter GGUF-Pfad (sonst Registry-Default)
        self.llm_bin = llm_bin        # llama-server-Binary
        self.manager = None           # LlamaServerManager, falls VON UNS gestartet
        self._llm_lock = threading.Lock()  # serialisiert llm_start/llm_stop (Concurrency)
        self._berater_lock = threading.Lock()  # serialisiert define-on-the-fly-Writes
        # --- Engine-Status: Degradations-Transparenz für die Host-Audit-Chain ---
        # EINMALIG beim Start erhoben, danach UNVERÄNDERLICH (die geladenen
        # Ressourcen ändern sich zur Laufzeit nicht). Antwortet auf die
        # Haftungsakte-Frage „mit welchem Engine-Zustand wurde diese Analyse
        # erstellt?" – bisher stand das nur im Betreiber-Log. Bewusst KEINE
        # Zeitstempel/Zufallswerte (Fingerprints sind deterministisch): der
        # Determinismus-Vertrag von /v1/analyse bleibt unangetastet.
        self._engine_status: dict[str, Any] = {
            "katalog_version": self.katalog.version,
            "schichten": {
                # Matcher (Schicht 1) und Beschreibungsmuster (1b) sind der
                # nicht-degradierende Kern: ohne Katalog startet __init__ gar
                # nicht erst – läuft der Server, laufen sie.
                "matcher": {"aktiv": True},
                "muster": {"aktiv": True},
                "trigger": {"aktiv": self.trigger is not None,
                            "grund": None if self.trigger is not None
                            else "kein Trigger-Index gebaut"},
                "graph": {"aktiv": self.graph is not None, "fingerprint": graph_fp},
                "embedding": {
                    "aktiv": self.embedding is not None and self.embedding_encoder is not None,
                    "encoder": emb_encoder_name,
                    "index_fingerprint": emb_index_fp,
                    "index_aktuell": index_aktuell,
                    # Betriebspunkt der Schicht 1.7 (additiv, s. pipeline.py):
                    # RRF-Fusion läuft NUR mit Reranker UND Trigger-Index (ohne
                    # Reranker wäre sie gemessen schädlich); die sim-Rettung ist
                    # nur mit Reranker sinnvoll — sonst null. Env-Werte werden
                    # beim Start EINMAL gelesen (Status bleibt deterministisch).
                    "fusion": bool(self.reranker is not None
                                   and self.trigger is not None
                                   and fusion_aktiviert()),
                    "sim_rettung": (sim_rettung() if self.reranker is not None
                                    else None),
                    "grund": emb_grund,
                },
                "reranker": {"aktiv": self.reranker is not None, "grund": reranker_grund},
                "ner": ner_status(),
                # Bewusst KEINE Live-Erreichbarkeitsprüfung beim Start (das LLM
                # ist lazy; Erreichbarkeit liefert GET /v1/llm/status). url nur
                # host:port – kein Token o. Ä. in archivierte Antworten.
                "llm": {"konfiguriert": bool(self.llm_url),
                        "url": _llm_url_anzeige(self.llm_url)},
            },
        }

    def engine_status(self) -> dict[str, Any]:
        """Beim Start erhobener Engine-Zustand (Degradations-Transparenz).

        Liefert eine KOPIE (deepcopy): Aufrufer reichern den Block an (z. B.
        mischt ``analyse`` die Lauf-Optionen dazu) – der eingefrorene
        Startup-Status darf dadurch nie mutiert werden.
        """
        return copy.deepcopy(self._engine_status)

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
                abweichung: bool = False, lemma: bool = False) -> dict[str, Any]:
        store = self._berater() if berater else None
        # Lauf-Optionen gebündelt über AnalyseOptionen (Single Source of Truth der
        # Defaults in pipeline.py); Ressourcen-Objekte bleiben Einzel-Parameter.
        # Trigger-Detektor + Streit-Verstärker sind in /v1 immer aktiv.
        report = self.pipeline.run(
            text,
            optionen=AnalyseOptionen(unknown=unknown, nutzer=nutzer,
                                     abweichung=abweichung,
                                     detektor=True, streit=True,
                                     lemma=lemma),
            berater=store, graph=self.graph,
            provider=self._provider(schicht2, llm_url), trigger=self.trigger,
            embedding=self.embedding,
            embedding_encoder=self.embedding_encoder,
            reranker=self.reranker)
        res = antwort_json(report, text)
        # Degradations-Nachweis JE Analyse (Haftungsakte): der Startup-Status
        # (welche Schichten KONNTEN laufen) plus die effektiven Lauf-Optionen
        # dieses Aufrufs (was WAR angefordert) machen jedes vom Host
        # archivierte Ergebnis selbsterklärend (INTEGRATION_TAXTRONIK.md §4).
        # Additiv und ohne Zeitstempel/Zufall – Determinismus-Vertrag bleibt.
        res["engine_status"] = {
            **self.engine_status(),
            "optionen": {"unknown": unknown, "berater": berater,
                         "abweichung": abweichung, "mit_llm": schicht2},
        }
        return res

    def definiere(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Form-Validierung: ein String statt Liste ("§ 15 EStG") würde von
        # tuple() zeichenweise zersplittert und STILL in die Festwissens-YAML
        # geschrieben. (api_v1 prüft vorab → 422; hier Defense-in-depth.)
        norm_anker = payload.get("norm_anker") or ()
        if isinstance(norm_anker, str) or not all(isinstance(x, str) for x in norm_anker):
            raise BeraterFehler("norm_anker muss eine Liste von Strings sein")
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
                norm_anker=tuple(norm_anker),
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

    def review_setzen(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Review-Lebenszyklus-Übergang (entwurf → geprüft → freigegeben, nur
        vorwärts) für einen GETEILTEN Berater-Eintrag. Festwissens-State,
        fallzustandslos, KEIN Audit — der Host protokolliert die Handlung in
        seiner Chain; deshalb liefert die Antwort den vollständigen Übergang
        (``alter_status`` → ``neuer_status``) zurück."""
        eintrag_id = (payload.get("id") or "").strip()
        neuer_status = (payload.get("status") or "").strip()
        pruefer = payload.get("pruefer") or "berater"
        with self._berater_lock:
            store = BeraterStore(self.berater_dir)
            # Bestehenden Eintrag VOR der Mutation lesen: erst damit wird der
            # Übergang (alter_status) für die Host-Chain auditierbar. Nur der
            # geteilt-Scope trägt einen Review-Status (berater.setze_review_status).
            vorher = next((e for e in store.alle()
                           if e.id == eintrag_id and e.scope == "geteilt"), None)
            if vorher is None:
                return {"ok": False, "fehler": f"kein geteilter Eintrag {eintrag_id!r}"}
            alter_status = vorher.review_status
            try:
                store.setze_review_status(eintrag_id, neuer_status, reviewer=pruefer)
            except BeraterFehler as exc:
                return {"ok": False, "fehler": str(exc)}
        return {"ok": True, "id": eintrag_id, "alter_status": alter_status,
                "neuer_status": neuer_status, "pruefer": pruefer}

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

    def radar(self) -> dict[str, Any]:
        """Norm-Änderungs-Radar: Welche Festwissens-Anker trifft der letzte
        Korpus-Rebuild? Vergleicht das beim ``graph build`` rotierte Vorher-
        Manifest mit dem aktuellen Graphen (radar.radar_bericht).

        Die Engine BERICHTET nur — der Host (TaxTronik) erzeugt aus den
        ``betroffen``-Einträgen seine Review-Aufgaben und protokolliert die
        Handlung in SEINER Audit-Chain. Bewusst KEINE Status-Rücksetzung
        (freigegeben → entwurf): das bräche die Nur-vorwärts-Invariante des
        Review-Lebenszyklus (review_setzen) und nähme dem Berufsträger die
        fachliche Entscheidung ab, ob die Gesetzesänderung die Definition
        überhaupt berührt."""
        if self.graph is None:
            # Muster der Graph-Guards (norm_suche, /v1/normgraph/*): ohne
            # gebauten Graph gibt es keinen aktuellen Stand zum Vergleichen.
            return {"ok": False,
                    "fehler": f"kein Normgraph gebaut ({self.graph_pfad}) – erst "
                              "'risk-layer korpus fetch' + 'risk-layer graph build'"}
        try:
            alt = lade_manifest(manifest_vorher_pfad(self.graph_pfad))
        except RadarFehler as exc:
            # Korruptes Manifest: laut scheitern statt still einen leeren Diff
            # zu melden. Der Pfad DARF in den Fehlertext (kein Mandantenbezug;
            # er sagt dem Operator, welche Datei zu reparieren/löschen ist).
            return {"ok": False, "fehler": str(exc)}
        if alt is None:
            # KEIN Fehler — frische Installation ist kein Defekt: vor dem
            # zweiten 'graph build' existiert schlicht kein Vorher-Stand, erst
            # die Manifest-Rotation des Builds erzeugt ihn (cli._cmd_graph).
            return {"ok": True, "vergleichsstand": False,
                    "hinweis": "kein Vergleichsstand – das Radar ist erst nach dem "
                               "naechsten 'graph build' verfuegbar (die Rotation legt "
                               "das Vorher-Manifest an)",
                    "betroffen": [],
                    "anzahl": {"geaendert": 0, "neu": 0, "entfallen": 0,
                               "betroffen": 0, "freigegeben": 0}}
        # Frischen Store unter Lock öffnen (Muster review_setzen/kuratiere_norm):
        # hier nur read-only, aber so liest das Radar nie eine Scope-YAML,
        # während ein paralleler Define-Write sie gerade atomar ersetzt. Der
        # Bericht selbst läuft AUSSERHALB des Locks (Graph-Voll-Scan über
        # knoten_hashes – er soll die Schreib-Handler nicht blockieren; der
        # Store ist nach dem Konstruktor vollständig im Speicher).
        with self._berater_lock:
            store = BeraterStore(self.berater_dir)
        # Zeitstempel hier GEWOLLT (Bericht über einen Zeitpunkt-Vergleich,
        # KEIN Determinismus-Vertrag wie bei /v1/analyse): erstellt_am
        # dokumentiert für die Host-Chain, WANN der Vergleich lief.
        bericht = radar_bericht(alt, self.graph, store, self.katalog,
                                erstellt_am=datetime.now().isoformat(timespec="seconds"))
        return {**bericht, "vergleichsstand": True}

    # -- Quantenlos (über /v1/los/*): beweisbar blinde Stichprobe -----------
    def los_ziehen(self, rahmen: list[str], k: int, backend: str) -> dict[str, Any]:
        """Quantenlos-Ziehung (quantenlos.ziehe): zustandslos — der Nachweis
        geht an den Host (der ihn in SEINE Audit-Chain legt), die Engine merkt
        sich NICHTS (kein Nachweis-Archiv, kein Job-Registry).

        ``backend`` kommt validiert aus api_v1 und ist dort PFLICHT: der Host
        wählt die Entropie-Quelle bewusst (Attestierungs-Anspruch, s.
        quantenlos-Modul-Docstring — stille Degradation qpu→csprng gibt es
        nicht). qpu reicht den Job nur EIN (→ status:wartet + job_id; Abholen
        über ``los_abholen``); simulator/csprng liefern sofort status:fertig.

        ``roh_bits_hex`` liegt der fertig-Antwort additiv bei: der Host
        archiviert die Roh-Bits NEBEN dem Nachweis (wie die CLI ihre
        ``.rohbits.hex``) und kann die volle Ableitungskette später offline
        nachrechnen (``los_pruefen`` mit roh_bits) — die Engine hebt nichts auf.

        Fehlerform ``{"ok": False, "fehler": …}``: fehlende Quanten-Extras
        bzw. backend=qpu ohne IBM-Credentials — api_v1 mappt das auf 503
        (fehlende Betriebsvoraussetzung, Muster /v1/radar ohne Graph)."""
        from . import quantenlos as Q
        try:
            if backend == "qpu":
                if not Q.ibm_credentials_vorhanden():
                    return {"ok": False, "fehler": _IBM_CREDENTIALS_FEHLEN}
                start = Q.qpu_los_starten(rahmen, k)
                return {"ok": True, "status": "wartet", **start}
            quelle: Q.EntropieQuelle = (Q.SimulatorQuelle() if backend == "simulator"
                                        else Q.CsprngQuelle())
            nachweis = Q.ziehe(rahmen, k, quelle)
        except Q.QuantenlosFehler as exc:
            # Nach der api_v1-Form-Validierung (rahmen/k/backend) bleibt hier
            # nur eine Fehlerquelle: Quanten-Extras nicht installiert → 503.
            return {"ok": False, "fehler": str(exc)}
        return {"ok": True, "status": "fertig", "nachweis": nachweis,
                "roh_bits_hex": (quelle.letzte_roh_bits or b"").hex()}

    def los_abholen(self, job_id: str, rahmen: list[str], k: int) -> dict[str, Any]:
        """Holt eine qpu-Ziehung ab — REKONSTRUIERT statt nachgeschlagen.

        Kein Job-Registry (Zustandslosigkeit): quantenlos.qpu_los_abholen baut
        Commitment + Stichprobe komplett neu aus job_id + rahmen + k; die
        Bindung Job↔Rahmen sichern die Commitment-Tags des IBM-Jobs. Läuft der
        Job noch → status:wartet (Host pollt erneut), fertig → status:fertig
        mit Nachweis (bit-identisch zum synchronen Pfad) + roh_bits_hex.

        Fehlerformen: ``bindung_verletzt: true`` = der Job gehört nicht zu
        diesem Rahmen (Client-Fehler → api_v1: 400, strukturell statt
        Fehlertext-Parsing); sonst fehlende Betriebsvoraussetzung → 503."""
        from . import quantenlos as Q
        if not Q.ibm_credentials_vorhanden():
            return {"ok": False, "fehler": _IBM_CREDENTIALS_FEHLEN}
        try:
            res = Q.qpu_los_abholen(job_id, rahmen, k)
        except Q.QuantenlosBindungsFehler as exc:
            return {"ok": False, "fehler": str(exc), "bindung_verletzt": True}
        except Q.QuantenlosFehler as exc:
            return {"ok": False, "fehler": str(exc)}
        return {"ok": True, **res}

    def los_pruefen(self, nachweis: dict[str, Any], rahmen: list[str], *,
                    roh_bits: bytes | None = None,
                    online: bool = False) -> dict[str, Any]:
        """Nachweis gegen die Rahmenliste nachrechnen (quantenlos.pruefe).

        Wirft nie für widersprüchliche Nachweise — ein Widerspruch ist
        ``gueltig: false`` mit benannten Befunden, kein HTTP-Fehler (die
        Prüfung HAT stattgefunden). ``online=True`` holt die Job-Counts über
        die IBM-Job-ID erneut; eine NICHT durchführbare Online-Prüfung (kein
        Token/Netz) ist per Protokoll-Semantik ein Hinweis, kein Fehlschlag."""
        from . import quantenlos as Q
        return Q.pruefe(nachweis, rahmen, roh_bits, online=online)

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

        def log_message(self, *args):  # ruhiger Konsolen-Output (bewusst);
            pass                       # fehlgeschlagene Auth loggt _401 separat

        # -- Helpers ----------------------------------------------------
        def _send(self, code: int, body: bytes, content_type: str,
                  extra_headers: dict[str, str] | None = None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any,
                  extra_headers: dict[str, str] | None = None):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8", extra_headers)

        def _dispatch_json(self, code: int, obj: Any):
            """Antwort aus dispatch(); bei 405 wird das ``allow``-Feld als
            sauberer Allow-Header gespiegelt (RFC 9110)."""
            extra = None
            if code == 405 and isinstance(obj, dict) and obj.get("allow"):
                extra = {"Allow": str(obj["allow"])}
            self._json(code, obj, extra)

        def _401(self):
            # Fehlgeschlagene Auth SICHTBAR machen (Betreiber-Log): Quell-IP +
            # Pfad – NIE Token-Inhalte. Erfolgreiche Requests bleiben bewusst
            # still (log_message oben).
            _log.warning("401 unauthorized: %s %s von %s", self.command,
                         urlparse(self.path).path, self.client_address[0])
            self._json(401, {"error": "unauthorized"})

        def _read_json(self) -> dict[str, Any]:
            # Chunked wird vom stdlib-Handler nicht dekodiert – ohne harte
            # Ablehnung würde der Body still falsch (leer) gelesen.
            te = (self.headers.get("Transfer-Encoding") or "").lower()
            if "chunked" in te:
                raise _RequestFehler(411, "Transfer-Encoding: chunked wird nicht "
                                          "unterstützt – Content-Length erforderlich")
            cl = self.headers.get("Content-Length")
            if cl is None:
                # Fehlende Content-Length hieße still leerer Body → laut: 411.
                raise _RequestFehler(411, "Content-Length erforderlich")
            try:
                laenge = int(cl)
            except (TypeError, ValueError):
                raise _RequestFehler(400, "ungültige Content-Length")
            if laenge < 0 or laenge > _MAX_BODY:
                raise _RequestFehler(413, f"Body zu groß (max {_MAX_BODY} Bytes)")
            roh = self.rfile.read(laenge) if laenge else b"{}"
            try:
                obj = json.loads(roh.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise _RequestFehler(400, "ungültiger JSON-Body")
            # Top-Level MUSS ein Objekt sein: '[1]' oder '"x"' liefe sonst als
            # AttributeError in die Handler (500 statt klarem Client-Fehler).
            if not isinstance(obj, dict):
                raise _RequestFehler(400, "JSON-Body muss ein Objekt sein")
            return obj

        def _auth_ok(self) -> bool:
            """Bearer-Token-Prüfung (Integrationsvertrag), KONSTANT-ZEITIG.

            Token aus RISK_LAYER_TOKEN. Ist KEINS gesetzt, wird nicht geprüft –
            aber das ist nur bei Loopback-Bind erreichbar: ``serve()`` lehnt einen
            Netz-Bind ohne Token ab. Der Vergleich nutzt ``hmac.compare_digest``
            auf utf-8-Bytes (kein Timing-Orakel, robust gegen Nicht-ASCII-Token).

            Rotationsfenster (L8): ist zusätzlich RISK_LAYER_TOKEN_VORGAENGER
            gesetzt, gilt ÜBERGANGSWEISE auch der alte Token — so lassen sich
            Engine und Clients nacheinander statt im Gleichschritt umstellen
            (Schrittfolge: INTEGRATION_TAXTRONIK.md § 3 F). Beide Vergleiche
            laufen IMMER vollständig, erst danach wird ODER-verknüpft: ein
            Kurzschluss (``or`` direkt zwischen den compare_digest-Aufrufen)
            wäre ein Timing-Leck — die Antwortzeit verriete, ob bereits der
            erste Vergleich getroffen hat.
            """
            erwartet = os.environ.get("RISK_LAYER_TOKEN", "")
            if not erwartet:
                # Kein aktueller Token → keine Auth (nur Loopback erreichbar,
                # Logik unverändert). Ein NUR gesetzter Vorgänger aktiviert
                # bewusst keine Auth: das Rotationsfenster setzt einen gültigen
                # NEUEN Token voraus, sonst wäre die Rotation keine.
                return True
            auth = self.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return False
            geboten = auth[7:].encode("utf-8")
            aktuell_ok = hmac.compare_digest(geboten, erwartet.encode("utf-8"))
            vorgaenger = os.environ.get("RISK_LAYER_TOKEN_VORGAENGER", "")
            vorgaenger_ok = (hmac.compare_digest(geboten, vorgaenger.encode("utf-8"))
                             if vorgaenger else False)
            if vorgaenger_ok and not aktuell_ok:
                # ECHTER Vorgänger-Treffer (nicht bloß alter == neuer Wert):
                # der Betreiber soll einmalig sehen, dass noch Clients auf dem
                # alten Token stehen und das Fenster zu schließen ist.
                _logge_rotationsfenster_einmalig()
            return aktuell_ok or vorgaenger_ok

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
                    self._401()
                    return
                code, obj = dispatch(app, "GET", _p.path, {}, parse_qs(_p.query))
                self._dispatch_json(code, obj)
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
                    self._401()
                    return
                payload = self._read_json()
                code, obj = dispatch(app, "POST", pfad, payload, {})
                self._dispatch_json(code, obj)
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
          llm_model: str | None = None, llm_bin: str | None = None,
          llm_autostart: bool = False) -> int:
    """Startet die /v1-Engine. Rückgabe: Exit-Code (0 = sauberer Lauf/Stop,
    2 = Startverweigerung) – main() reicht ihn durch, damit Service-Manager
    (systemd/NSSM) und deploy-Skripte einen Fehlstart als Fehler sehen."""
    # Netz-Bind ohne Token verweigern (§ 203 StGB): die /v1-Engine ist auth-pflichtig,
    # sobald sie das Loopback verlässt. Ohne Token nur lokal (Entwicklung).
    if not _ist_loopback(host) and not os.environ.get("RISK_LAYER_TOKEN", ""):
        print(f"Host {host!r} ist kein Loopback, aber RISK_LAYER_TOKEN ist nicht gesetzt. "
              "Setze ein Token (Bearer-Auth, § 203 StGB) oder binde an 127.0.0.1.",
              file=sys.stderr)
        return 2

    if not _ist_loopback(host):
        # Kein TLS in der Engine selbst (bewusst stdlib-pur): beim Netz-Bind gehen
        # Bearer-Token UND Mandantentexte sonst im KLARTEXT übers Netz. Betrieb
        # nur hinter einem TLS-Reverse-Proxy (siehe INTEGRATION_TAXTRONIK.md).
        warnung = (f"WARNUNG: Netz-Bind auf {host!r} ohne TLS – die Engine spricht "
                   "reines HTTP. Token und Mandantentexte (§ 203 StGB) NUR hinter "
                   "einem TLS-Reverse-Proxy betreiben (INTEGRATION_TAXTRONIK.md).")
        print(warnung, file=sys.stderr)
        _log.warning("%s", warnung)

    app = Anwendung(katalog, berater_dir, graph_pfad, llm_url=llm_url,
                    llm_model=llm_model, llm_bin=llm_bin)

    if llm_autostart:                 # Schicht 2 (llama-server) mit hochfahren – ein-Befehl-Start.
        r = app.llm_start()           # idempotent + nicht-blockierend; Bereitschaft via /v1/llm/status
        print("  LLM-Autostart: " + (
            f"startet {r.get('modell')}" if r.get("startet") else
            r.get("hinweis") or r.get("fehler") or "?"), file=sys.stderr)

    class _Server(ThreadingHTTPServer):
        # KEIN SO_REUSEADDR: ein zweiter Start auf belegtem Port soll laut scheitern,
        # nicht (unter Windows) still denselben Port doppelt binden.
        allow_reuse_address = False
        daemon_threads = True

    try:
        httpd = _Server((host, port), make_handler(app))
    except OSError as exc:
        print(f"Port {port} ist belegt – läuft die Engine schon? ({exc})", file=sys.stderr)
        return 2

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
    return 0


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
    p.add_argument("--llm-autostart", action="store_true",
                   help="Schicht 2 (llama-server) beim Start mit hochfahren – ein-Befehl-Start "
                        "des gesamten Stacks (Modell aus der Registry/--llm-model).")
    args = p.parse_args(argv)
    # Exit-Code durchreichen: Startverweigerung (Netz-Bind ohne Token, Port belegt)
    # MUSS als Fehler enden – sonst meldet jeder Service-Manager „erfolgreich".
    return serve(host=args.host, port=args.port, katalog=args.katalog,
                 berater_dir=args.berater, graph_pfad=args.graph, llm_url=args.llm_url,
                 llm_model=args.llm_model, llm_bin=args.llm_bin,
                 llm_autostart=args.llm_autostart)


if __name__ == "__main__":
    raise SystemExit(main())
