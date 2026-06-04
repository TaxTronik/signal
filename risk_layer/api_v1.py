"""Engine-API /v1/* – der stabile, ZUSTANDSLOSE Vertrag für TaxTronik.

Diese Schicht ist der Integrationsvertrag (siehe INTEGRATION_TAXTRONIK.md §4).
Grundsätze, die sie vom GUI-`/api/*` unterscheiden:

  * ZUSTANDSLOS bzgl. Mandantendaten (§ 203): kein Fall-/Aufgaben-/Audit-Persist.
    Text rein, Analyse raus. Mandantendaten leben in TaxTronik (Postgres/SeaweedFS).
    Persistiert wird AUSSCHLIESSLICH Festwissen: der Katalog (define-on-the-fly,
    Berater-Begriffe) – das ist kein Mandantengeheimnis.
  * STABILES Schema: additive Änderungen erlaubt, keine Brüche. Die TS-Seite
    (`packages/risk-layer`) typisiert gegen genau diese Antworten.
  * AUTH: Bearer-Token (shared secret) wird vom HTTP-Layer geprüft, bevor hierher
    delegiert wird (siehe web.py). Dieses Modul selbst macht keine Netz-/IO-Magie.

Aufruf-Kontrakt: ``dispatch(app, method, pfad, payload, query) -> (status, dict)``.
``app`` ist die bestehende ``Anwendung`` (liefert pipeline/katalog/graph/embedding).
"""

from __future__ import annotations

from typing import Any

V1_VERSION = "1.0.0"


def _analyse(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/analyse – zustandsloser Lauf über einen Text.

    Body:  {text, mitLLM?: bool, optionen?: {unknown?, berater?, abweichung?}}
    mitLLM steuert Schicht 2 (LLM). false ⇒ schnell/deterministisch (empfohlen für
    den ersten, synchronen Aufruf; den LLM-Lauf optional asynchron nachholen).

    Antwort = das volle Analyse-JSON (spans, karten, risiken, semantik mit
    Normketten, summary, textHash, katalogVersion) PLUS engineVersion. KEIN
    Persistieren – TaxTronik speichert das Ergebnis in RiskAnalysis/RiskMarking.
    """
    text = str(payload.get("text", ""))
    opt = payload.get("optionen") or {}
    mit_llm = bool(payload.get("mitLLM", False))
    res = app.analyse(
        text,
        unknown=bool(opt.get("unknown")),
        berater=False,                  # zustandslos: keine Berater-Store-Defs einmischen
        nutzer=None,
        schicht2=mit_llm,
        # Bewusst KEIN client-geliefertes llm_url: auf der netz-exponierten
        # Maschinen-API wäre das eine (blinde) SSRF – die Engine würde ihre
        # LLM-Calls auf ein vom Aufrufer gewähltes Ziel lenken. Es gilt strikt
        # der server-konfigurierte llama-server (self.llm_url).
        llm_url=None,
        abweichung=bool(opt.get("abweichung")),
    )
    res["engineVersion"] = V1_VERSION
    return 200, res


def _katalog_lesen(app) -> tuple[int, dict[str, Any]]:
    """GET /v1/katalog – Version + alle Begriffe (für Anzeige/Sync in TaxTronik)."""
    k = app.katalog
    begriffe = [{
        "id": e.id,
        "begriff": e.begriff_kanonisch,
        "aliase": list(e.aliase),
        "governance_typ": e.governance_typ.value,
        "schadensintensitaet": e.schadensintensitaet,
        "kaskadenreichweite": e.kaskadenreichweite,
        "norm_anker": list(e.norm_anker),
        "normketten": [kette.as_dict() for kette in e.normketten],
        "hat_definitionsquelle": e.hat_definitionsquelle,
    } for e in k]
    return 200, {"version": k.version, "begriffe": begriffe, "engineVersion": V1_VERSION}


def _katalog_definiere(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/katalog/definiere – Berater-Begriff anlegen (Delegations-Rückfluss).

    Das ist die EINZIGE schreibende Operation der Engine: erledigte Recherche
    fließt als Berater-Definition in den Katalog zurück (kollaborativer Zusammenbau).
    Festwissen, kein Mandantengeheimnis. Body wie GUI-/api/definiere:
      {begriff, definition_text, autor?, scope?, norm_anker?[], governance_typ?, ...}
    """
    begriff = str(payload.get("begriff", "")).strip()
    if not begriff:
        return 400, {"error": "Feld 'begriff' fehlt"}
    # § 203: Dieser Store ist Festwissen und wird an JEDEN Client ausgeliefert
    # (GET /v1/katalog). Ein verbatim-Write darf daher kein Mandantengeheimnis
    # tragen – derselbe Detektor wie beim Recherche-Export prüft begriff + Text.
    from .recherche import pruefe_rest
    rest = pruefe_rest(f"{begriff}\n{payload.get('definition_text', '')}")
    if rest:
        return 422, {"error": "definition_text enthält identifizierende Angaben "
                              "(§ 203 StGB) – bitte abstrahieren.", "rest": rest}
    return 200, app.definiere(payload)


def _katalog_norm_kuratieren(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/katalog/norm_kuratieren – katalog-weite Norm-Kuratierung an EINEM
    Begriff: {katalog_id, norm, aktion: verwerfen|ergaenzen|zuruecksetzen, scope, autor}.
    Rein strukturell (Begriff-ID + Norm-Zitat + aktion + Autor-Tag) – KEIN Fall-/
    Mandantentext, daher § 203-unkritisch und fallzustandslos. Die Engine hält nur den
    Katalog-State; die Handlung (wer/wann) auditiert der Host in seiner Chain."""
    res = app.kuratiere_norm(payload)
    return (200 if res.get("ok") else 400), res


def _katalog_kuratierung(app, query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """GET /v1/katalog/kuratierung?katalog_id=&nutzer= – geltende Norm-Kuratierung
    zum Überlagern im Host. Ohne ``katalog_id`` alle Roh-Records (geteilt + die
    persönlichen des Nutzers), mit ``katalog_id`` die aufbereitete Sicht eines Begriffs
    ({verworfen, ergaenzt}). Reiner Katalog-State, kein Fallbezug; /v1/analyse bleibt
    davon unberührt (katalog-rein) – die Anwendung der Kuratierung macht der Host."""
    katalog_id = (query.get("katalog_id") or [None])[0]
    nutzer = (query.get("nutzer") or [None])[0]
    return 200, {**app.kuratierung_lesen(katalog_id=katalog_id, nutzer=nutzer),
                 "engineVersion": V1_VERSION}


def _normgraph_aufloesen(app, query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """GET /v1/normgraph/aufloesen?id=norm:AO:162 – einen Knoten auflösen."""
    nid = (query.get("id") or [""])[0]
    if app.graph is None:
        return 200, {"gefunden": False, "grund": "kein Normgraph geladen"}
    if not nid:
        return 400, {"error": "Parameter 'id' fehlt"}
    return 200, app.graph.aufloesen(nid)


def _normgraph_suche(app, query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """GET /v1/normgraph/suche?q=... – Norm-Auswahl (Zitat ODER Titel)."""
    q = (query.get("q") or [""])[0]
    return 200, app.norm_suche(q)


def _embedding_suche(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/embedding/suche – semantische Norm-VORSCHLÄGE zu freiem Text.

    Body: {text, k?: int}. Liefert Bedeutungs-Nachbarn (mit sim-Wert) – ausdrücklich
    Vorschläge, kein Mapping. Leer, wenn kein Embedding-Index gebaut ist.
    """
    text = str(payload.get("text", ""))
    if app.embedding is None or app.embedding_encoder is None:
        return 200, {"treffer": [], "grund": "kein Embedding-Index"}
    # k aus dem Request-Body robust parsen + clampen: unbegrenztes/negatives k
    # ist ein DoS-Vektor (riesiger Sort+Serialize auf einem Worker-Thread).
    try:
        k = int(payload.get("k", 5))
    except (TypeError, ValueError):
        k = 5
    k = max(1, min(k, 50))
    hits = app.embedding.suche(text, app.embedding_encoder, k=k, min_sim=0.0)
    return 200, {"treffer": hits, "engineVersion": V1_VERSION}


def _llm_status(app) -> tuple[int, dict[str, Any]]:
    """GET /v1/llm/status – Erreichbarkeit + Modell/Engine + **Queue** (Slot-/
    Auslastungszustand) des llama-server. Damit kann das Host-System entscheiden,
    ob es ``mitLLM:true`` jetzt sendet oder wartet."""
    return 200, {**app.llm_status(), "engineVersion": V1_VERSION}


def _llm_start(app) -> tuple[int, dict[str, Any]]:
    """POST /v1/llm/start – startet den SERVER-konfigurierten llama-server.

    Kein Request-Input (Modell/Binary kommen aus der Server-Config) → kein
    Injection-Vektor. Idempotent; blockiert nicht bis fertig – Bereitschaft über
    GET /v1/llm/status (Feld ``verfuegbar``) pollen.
    """
    return 200, {**app.llm_start(), "engineVersion": V1_VERSION}


def dispatch(app, method: str, pfad: str, payload: dict[str, Any],
             query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """Zentrale /v1-Weiche. Gibt (HTTP-Status, JSON-dict) zurück.

    Unbekannte Pfade → (404, …). Auth/HTTP-Plumbing macht der Aufrufer (web.py).
    """
    if method == "GET":
        if pfad == "/v1/health":
            return 200, {"ok": True, "engineVersion": V1_VERSION,
                         "katalogVersion": app.katalog.version,
                         "graph": app.graph is not None,
                         "embedding": app.embedding is not None}
        if pfad == "/v1/katalog":
            return _katalog_lesen(app)
        if pfad == "/v1/normgraph/aufloesen":
            return _normgraph_aufloesen(app, query)
        if pfad == "/v1/normgraph/suche":
            return _normgraph_suche(app, query)
        if pfad == "/v1/katalog/kuratierung":
            return _katalog_kuratierung(app, query)
        if pfad == "/v1/llm/status":
            return _llm_status(app)
    elif method == "POST":
        if pfad == "/v1/analyse":
            return _analyse(app, payload)
        if pfad == "/v1/katalog/definiere":
            return _katalog_definiere(app, payload)
        if pfad == "/v1/katalog/norm_kuratieren":
            return _katalog_norm_kuratieren(app, payload)
        if pfad == "/v1/embedding/suche":
            return _embedding_suche(app, payload)
        if pfad == "/v1/llm/start":
            return _llm_start(app)
    return 404, {"error": f"unbekannter /v1-Endpoint: {method} {pfad}"}
