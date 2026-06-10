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

from .berater import REVIEW_REIHENFOLGE

# 1.1.0: additiv – neue Route POST /v1/katalog/review (Review-Lebenszyklus des
# geteilten Festwissens läuft jetzt durch /v1, damit der Host die Freigabe in
# seiner Audit-Chain protokollieren kann). Keine bestehende Antwort geändert.
# 1.2.0: additiv – neue Route GET /v1/status + engine_status-Block in der
# /v1/analyse-Antwort (Degradations-Transparenz: mit welchem Engine-Zustand
# wurde diese Analyse erstellt? Der Host legt den Block mit dem Ergebnis in
# seiner Audit-Chain ab). Ein paralleles Feature ergänzt in DERSELBEN Version
# zusätzlich GET /v1/radar. Keine bestehende Antwort geändert.
V1_VERSION = "1.2.0"


def _analyse(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/analyse – zustandsloser Lauf über einen Text.

    Body:  {text, mitLLM?: bool, optionen?: {unknown?, berater?, abweichung?, lemma?}}
    mitLLM steuert Schicht 2 (LLM). false ⇒ schnell/deterministisch (empfohlen für
    den ersten, synchronen Aufruf; den LLM-Lauf optional asynchron nachholen).

    Antwort = das volle Analyse-JSON (spans, karten, risiken, semantik mit
    Normketten, summary, text_hash, katalog_version) PLUS engineVersion. KEIN
    Persistieren – TaxTronik speichert das Ergebnis in RiskAnalysis/RiskMarking.

    HINWEIS Schreibweise (bewusst EINGEFROREN, Vertrag): /v1/analyse liefert
    ``text_hash``/``katalog_version`` in snake_case (siehe web.antwort_json),
    /v1/health dagegen ``katalogVersion`` in camelCase. Inkonsequent, aber die
    TS-Seite typisiert gegen genau diese Felder – NICHT vereinheitlichen.
    """
    # BEWUSST kein separates Zeichenlimit auf `text`: der Pfad ist auth-pflichtig
    # (vertrauenswuerdige Integration), Sachverhalte/Klausuren sind legitim lang, und
    # der Body ist bereits auf _MAX_BODY (8 MiB, web.py) gedeckelt. Bei mitLLM=true
    # laeuft der ganze Text durch den Provider (pipeline.run smart_windowing=False) –
    # konsistent zur CLI. Wird LLM-Kosten auf grossen Dokumenten je zum Problem, ist
    # der Hebel `smart_windowing` (implementiert + getestet, dann CLI+Web konsistent
    # setzen UND gegen Gold messen), KEIN blankes Textlimit. Der `text`-Echo in der
    # Antwort ist tragend: die spans referenzieren Offsets in genau diesen Text.
    text = str(payload.get("text", ""))
    opt = payload.get("optionen") or {}
    # Robustheit: ein truthy Nicht-Objekt ({"optionen": "x"}) würde unten bei
    # opt.get(...) als AttributeError → 500 enden. Klar klassifizieren: 400.
    if not isinstance(opt, dict):
        return 400, {"error": "Feld 'optionen' muss ein JSON-Objekt sein"}
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
        # /LF50/: optionaler Lemma-Robustheitspfad (flektierte Katalogformen).
        # Default False – Bestands-Clients erhalten byte-identische Antworten.
        lemma=bool(opt.get("lemma")),
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
    # Form-Validierung: ein String statt Liste würde sonst zeichenweise
    # zersplittert STILL in die Festwissens-YAML geschrieben.
    norm_anker = payload.get("norm_anker")
    if norm_anker is not None and (
            isinstance(norm_anker, str) or not isinstance(norm_anker, (list, tuple))
            or not all(isinstance(x, str) for x in norm_anker)):
        return 422, {"error": "Feld 'norm_anker' muss eine Liste von Strings sein"}
    # § 203: Dieser Store ist Festwissen und wird an JEDEN Client ausgeliefert
    # (GET /v1/katalog). Ein verbatim-Write darf daher kein Mandantengeheimnis
    # tragen – derselbe Detektor wie beim Recherche-Export prüft begriff + Text.
    from .recherche import pruefe_rest
    rest = pruefe_rest(f"{begriff}\n{payload.get('definition_text', '')}")
    if rest:
        return 422, {"error": "definition_text enthält identifizierende Angaben "
                              "(§ 203 StGB) – bitte abstrahieren.", "rest": rest}
    return 200, app.definiere(payload)


def _katalog_review(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/katalog/review – Review-Lebenszyklus eines GETEILTEN Berater-
    Eintrags: {id, status: entwurf|geprüft|freigegeben, pruefer?}. Nur Vorwärts-
    Übergänge (entwurf → geprüft → freigegeben, berater.setze_review_status).

    Der Lebenszyklus ist die Kanonizitäts-Garantie des geteilten Festwissens –
    er MUSS durch /v1 laufen, damit der Host die Freigabe in seiner Audit-Chain
    protokollieren kann (die Engine auditiert nicht, s. web.review_setzen).
    Die Antwort nennt deshalb den VOLLSTÄNDIGEN Übergang (alter_status →
    neuer_status). Reiner Festwissens-State, kein Mandantenbezug (§ 203-unkritisch).
    """
    eintrag_id = str(payload.get("id", "")).strip()
    if not eintrag_id:
        return 400, {"error": "Feld 'id' fehlt"}
    status = str(payload.get("status", "")).strip()
    if not status:
        return 400, {"error": "Feld 'status' fehlt"}
    # Form-Validierung wie norm_anker in _katalog_definiere: ein unbekannter
    # Statuswert ist ein 422 (Form), kein Lebenszyklus-Fehler (400).
    if status not in REVIEW_REIHENFOLGE:
        return 422, {"error": "Feld 'status' muss eines von "
                              f"{'|'.join(REVIEW_REIHENFOLGE)} sein, nicht {status!r}"}
    res = app.review_setzen(payload)
    # BeraterFehler (Rückwärts-Übergang, unbekannte id) → 400 mit Fehlertext
    # (Muster _katalog_norm_kuratieren: ok:false trägt 'fehler').
    return (200 if res.get("ok") else 400), res


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
    # Parameter-Validierung ZUERST: ein fehlendes 'id' ist auch ohne geladenen
    # Graph ein Client-Fehler (400), kein 200-„nicht gefunden".
    nid = (query.get("id") or [""])[0]
    if not nid:
        return 400, {"error": "Parameter 'id' fehlt"}
    if app.graph is None:
        return 200, {"gefunden": False, "grund": "kein Normgraph geladen",
                     "engineVersion": V1_VERSION}
    return 200, {**app.graph.aufloesen(nid), "engineVersion": V1_VERSION}


def _normgraph_suche(app, query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """GET /v1/normgraph/suche?q=... – Norm-Auswahl (Zitat ODER Titel)."""
    q = (query.get("q") or [""])[0]
    return 200, {**app.norm_suche(q), "engineVersion": V1_VERSION}


def _embedding_suche(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/embedding/suche – semantische Norm-VORSCHLÄGE zu freiem Text.

    Body: {text, k?: int}. Liefert Bedeutungs-Nachbarn (mit sim-Wert) – ausdrücklich
    Vorschläge, kein Mapping. Leer, wenn kein Embedding-Index gebaut ist.
    """
    text = str(payload.get("text", ""))
    if app.embedding is None or app.embedding_encoder is None:
        return 200, {"treffer": [], "grund": "kein Embedding-Index",
                     "engineVersion": V1_VERSION}
    # k aus dem Request-Body robust parsen + clampen: unbegrenztes/negatives k
    # ist ein DoS-Vektor (riesiger Sort+Serialize auf einem Worker-Thread).
    try:
        k = int(payload.get("k", 5))
    except (TypeError, ValueError):
        k = 5
    k = max(1, min(k, 50))
    hits = app.embedding.suche(text, app.embedding_encoder, k=k, min_sim=0.0)
    return 200, {"treffer": hits, "engineVersion": V1_VERSION}


def _status(app) -> tuple[int, dict[str, Any]]:
    """GET /v1/status – Engine-Zustand (Degradations-Transparenz, seit 1.2.0).

    Welche Schichten KONNTEN beim Start geladen werden (Trigger/Graph/
    Embedding/Reranker/NER), inkl. Degradations-GRUND und deterministischer
    Fingerprints – die Engine degradiert bewusst an vielen Stellen, und für
    die Haftungsakte des Hosts muss dieser Zustand abfragbar sein, nicht nur
    im Betreiber-Log stehen. Bewusst KEINE Live-Prüfung (das LLM bleibt lazy;
    Erreichbarkeit liefert GET /v1/llm/status) und KEINE Zeitstempel.
    Derselbe Block hängt als ``engine_status`` an jeder /v1/analyse-Antwort
    (dort zusätzlich mit den effektiven Lauf-``optionen``).
    """
    return 200, {"ok": True, "engineVersion": V1_VERSION,
                 "engine_status": app.engine_status()}


def _radar(app) -> tuple[int, dict[str, Any]]:
    """GET /v1/radar – Norm-Änderungs-Radar (Re-Review-Trigger, seit 1.2.0).

    Nach einem Korpus-Rebuild (Operator: 'korpus fetch' + 'graph build', der
    Build rotiert das Norm-Manifest) benennt der Bericht, WELCHE Normen sich
    geändert haben / entfallen sind und welche Festwissens-Anker (Berater-
    Definitionen, Norm-Kuratierungen, Katalog) betroffen sind – dringlichkeits-
    sortiert (freigegebene geteilte Definitionen zuerst: ihre stille
    Veralterung ist der haftungsrelevanteste Fall). Die Engine MUTIERT nichts
    und auditiert nicht: der Host erzeugt je betroffen[]-Eintrag seine
    Review-Aufgaben und protokolliert die Handlung in SEINER Chain; die
    anschließenden Freigaben laufen wieder über POST /v1/katalog/review.

    Statuscodes: frische Installation (noch kein Vorher-Manifest) ist KEIN
    Fehler → 200 mit vergleichsstand:false. Kein Graph gebaut → 503: fehlende
    Betriebsvoraussetzung, Operator-Aktion nötig – anders als die degradierenden
    200er von /v1/normgraph/* bzw. /v1/embedding/suche liefert das Radar ohne
    Graph keinen brauchbaren Teil-Bericht. Korruptes Vorher-Manifest → 500;
    der Fehlertext nennt den Datei-Pfad (kein Mandantenbezug – er sagt dem
    Operator, welche Datei zu reparieren ist).
    """
    res = app.radar()
    if res.get("ok"):
        return 200, {**res, "engineVersion": V1_VERSION}
    # ok:false hat genau zwei Quellen (web.Anwendung.radar): kein Graph (503)
    # oder RadarFehler/korruptes Manifest (500) – robust am App-Zustand
    # unterschieden statt brüchig am Fehlertext geparst.
    return (503 if app.graph is None else 500), {**res, "engineVersion": V1_VERSION}


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


def _llm_stop(app) -> tuple[int, dict[str, Any]]:
    """POST /v1/llm/stop – stoppt einen VON UNS gestarteten llama-server.

    Gegenstück zu /v1/llm/start (sonst könnte der Host VRAM belegen, aber nie
    freigeben). Kein Request-Input; idempotent – ein extern gestarteter oder
    gar nicht laufender Server bleibt unberührt (``ok: true``).
    """
    return 200, {**app.llm_stop(), "engineVersion": V1_VERSION}


# Bekannte Routen je Methode – für 405 (Methode falsch) statt 404 (Pfad unbekannt).
_GET_PFADE = frozenset({"/v1/health", "/v1/status", "/v1/radar", "/v1/katalog",
                        "/v1/normgraph/aufloesen", "/v1/normgraph/suche",
                        "/v1/katalog/kuratierung", "/v1/llm/status"})
_POST_PFADE = frozenset({"/v1/analyse", "/v1/katalog/definiere", "/v1/katalog/norm_kuratieren",
                         "/v1/katalog/review", "/v1/embedding/suche",
                         "/v1/llm/start", "/v1/llm/stop"})


def dispatch(app, method: str, pfad: str, payload: dict[str, Any],
             query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """Zentrale /v1-Weiche. Gibt (HTTP-Status, JSON-dict) zurück.

    Unbekannte Pfade → (404, …); bekannter Pfad mit falscher Methode → (405, …)
    mit ``allow``-Feld (web.py setzt daraus den Allow-Header). Auth/HTTP-Plumbing
    macht der Aufrufer (web.py).
    """
    if method == "GET":
        if pfad == "/v1/health":
            # ``katalogVersion`` hier in camelCase, /v1/analyse liefert dagegen
            # ``katalog_version`` (snake_case) – Inkonsistenz bewusst EINGEFROREN,
            # die TS-Seite typisiert gegen genau diese Felder (kein Bruch).
            return 200, {"ok": True, "engineVersion": V1_VERSION,
                         "katalogVersion": app.katalog.version,
                         "graph": app.graph is not None,
                         "embedding": app.embedding is not None}
        if pfad == "/v1/status":
            return _status(app)
        if pfad == "/v1/radar":
            return _radar(app)
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
        if pfad == "/v1/katalog/review":
            return _katalog_review(app, payload)
        if pfad == "/v1/embedding/suche":
            return _embedding_suche(app, payload)
        if pfad == "/v1/llm/start":
            return _llm_start(app)
        if pfad == "/v1/llm/stop":
            return _llm_stop(app)
    # Bekannter Pfad, falsche Methode → 405 (nicht 404); ``allow`` nennt die
    # richtige Methode (web.py spiegelt das als Allow-Header).
    if method == "GET" and pfad in _POST_PFADE:
        return 405, {"error": f"Methode GET nicht erlaubt für {pfad}", "allow": "POST"}
    if method == "POST" and pfad in _GET_PFADE:
        return 405, {"error": f"Methode POST nicht erlaubt für {pfad}", "allow": "GET"}
    return 404, {"error": f"unbekannter /v1-Endpoint: {method} {pfad}"}
