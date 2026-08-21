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

import re
from typing import Any

from .berater import REVIEW_REIHENFOLGE
from .models import GUELTIGE_GOVERNANCE, GUELTIGE_SCHADEN

# 1.1.0: additiv – neue Route POST /v1/katalog/review (Review-Lebenszyklus des
# geteilten Festwissens läuft jetzt durch /v1, damit der Host die Freigabe in
# seiner Audit-Chain protokollieren kann). Keine bestehende Antwort geändert.
# 1.2.0: additiv – neue Route GET /v1/status + engine_status-Block in der
# /v1/analyse-Antwort (Degradations-Transparenz: mit welchem Engine-Zustand
# wurde diese Analyse erstellt? Der Host legt den Block mit dem Ergebnis in
# seiner Audit-Chain ab). Ein paralleles Feature ergänzt in DERSELBEN Version
# zusätzlich GET /v1/radar. Keine bestehende Antwort geändert.
# 1.3.0: additiv – neue Routen POST /v1/los/{ziehen,abholen,pruefen}
# (Quantenlos: beweisbar blinde Stichprobe als Engine-Dienst; der Host baut
# den Rahmen aus seiner Chain, legt den Nachweis IN die Chain und erzeugt die
# Prüf-Aufgaben aus der Stichprobe). Auch hier zustandslos: KEIN Job-Registry —
# /v1/los/abholen rekonstruiert die Ziehung aus job_id+rahmen+k, die Bindung
# Job↔Rahmen sichern die Commitment-Tags des IBM-Jobs. Keine bestehende
# Antwort geändert.
# 1.3.2: Sicherheits-Hardening – strikte Bool-/String-Schemas, Ressourcenlimits
# und fail-closed Prüfung manipulierter bzw. nur teilweise belegter Nachweise.
# 1.4.0: additive, getrennt operator-authentisierte Embedding-Status-, Refresh-
# und Schedule-Routen. Analyse-/Suchvertraege bleiben unveraendert.
V1_VERSION = "1.4.0"

_MAX_FESTWISSEN_BEGRIFF = 512
_MAX_FESTWISSEN_TEXT = 100_000
_MAX_NORM_ANKER = 100
_MAX_NORM_ANKER_CHARS = 1_024
_MAX_AKTEUR_CHARS = 256
_MAX_STRUKTURFELD_CHARS = 512
_MAX_DATUM_CHARS = 64
_MAX_LOS_JOB_ID_CHARS = 512
_MAX_IBM_TOKEN_CHARS = 16_384
_MAX_LOS_ROH_HEX_CHARS = 2_000_000


def _pii_rest_fuer_felder(felder: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Alle persistierten Freitexte gemeinsam durch das lokale PII-Gate schicken.

    Harte Trenner verhindern, dass Inhalte beim Zusammenfuehren ineinanderlaufen.
    Die internen Feldnamen duerfen selbst NICHT in den Prueftext: statistische
    NER-Modelle koennen Bezeichner wie ``Feld pruefer`` als Person erkennen und
    damit jeden noch so sauberen Actor-Tag faelschlich ablehnen. Ein gemeinsamer
    Lauf ist zudem deutlich billiger als bis zu hundert einzelne NER-Aufrufe
    fuer ``norm_anker``.
    """
    from .recherche import pruefe_rest

    text = "\n\n---\n\n".join(wert for _name, wert in felder if wert)
    return pruefe_rest(text) if text else []


def _bool_feld(obj: dict[str, Any], feld: str) -> tuple[bool, str | None]:
    """Optionales JSON-Bool strikt lesen; Strings wie ``"false"`` nie aktivieren."""
    if feld not in obj:
        return False, None
    wert = obj[feld]
    if not isinstance(wert, bool):
        return False, f"Feld '{feld}' muss true oder false sein"
    return wert, None


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
    text = payload.get("text", "")
    if not isinstance(text, str):
        return 400, {"error": "Feld 'text' muss ein String sein"}
    opt = payload["optionen"] if "optionen" in payload else {}
    # Robustheit: ein truthy Nicht-Objekt ({"optionen": "x"}) würde unten bei
    # opt.get(...) als AttributeError → 500 enden. Klar klassifizieren: 400.
    if not isinstance(opt, dict):
        return 400, {"error": "Feld 'optionen' muss ein JSON-Objekt sein"}
    mit_llm, fehler = _bool_feld(payload, "mitLLM")
    if fehler:
        return 400, {"error": fehler}
    optionen: dict[str, bool] = {}
    for feld in ("unknown", "abweichung", "lemma"):
        wert, fehler = _bool_feld(opt, feld)
        if fehler:
            return 400, {"error": f"Feld 'optionen.{feld}' muss true oder false sein"}
        optionen[feld] = wert
    res = app.analyse(
        text,
        unknown=optionen["unknown"],
        berater=False,                  # zustandslos: keine Berater-Store-Defs einmischen
        nutzer=None,
        schicht2=mit_llm,
        # Bewusst KEIN client-geliefertes llm_url: auf der netz-exponierten
        # Maschinen-API wäre das eine (blinde) SSRF – die Engine würde ihre
        # LLM-Calls auf ein vom Aufrufer gewähltes Ziel lenken. Es gilt strikt
        # der server-konfigurierte llama-server (self.llm_url).
        llm_url=None,
        abweichung=optionen["abweichung"],
        # /LF50/: optionaler Lemma-Robustheitspfad (flektierte Katalogformen).
        # Default False – Bestands-Clients erhalten byte-identische Antworten.
        lemma=optionen["lemma"],
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
    begriff_roh = payload.get("begriff", "")
    if not isinstance(begriff_roh, str):
        return 400, {"error": "Feld 'begriff' muss ein String sein"}
    begriff = begriff_roh.strip()
    if not begriff:
        return 400, {"error": "Feld 'begriff' fehlt"}
    definition_text = payload.get("definition_text", "")
    if not isinstance(definition_text, str):
        return 400, {"error": "Feld 'definition_text' muss ein String sein"}
    if len(begriff) > _MAX_FESTWISSEN_BEGRIFF or len(definition_text) > _MAX_FESTWISSEN_TEXT:
        return 413, {"error": "Festwissen-Text zu lang (begriff maximal "
                              f"{_MAX_FESTWISSEN_BEGRIFF}, definition_text maximal "
                              f"{_MAX_FESTWISSEN_TEXT} Zeichen)"}
    for feld in ("autor", "datum", "scope", "governance_typ",
                 "schadensintensitaet", "konfidenz"):
        if feld in payload and not isinstance(payload[feld], str):
            return 400, {"error": f"Feld '{feld}' muss ein String sein"}
    if ("katalog_id" in payload and payload["katalog_id"] is not None
            and not isinstance(payload["katalog_id"], str)):
        return 400, {"error": "Feld 'katalog_id' muss ein String oder null sein"}
    for feld, limit in (("autor", _MAX_AKTEUR_CHARS),
                        ("datum", _MAX_DATUM_CHARS),
                        ("scope", _MAX_STRUKTURFELD_CHARS),
                        ("governance_typ", _MAX_STRUKTURFELD_CHARS),
                        ("schadensintensitaet", _MAX_STRUKTURFELD_CHARS),
                        ("konfidenz", _MAX_STRUKTURFELD_CHARS),
                        ("katalog_id", _MAX_STRUKTURFELD_CHARS)):
        wert = payload.get(feld)
        if isinstance(wert, str) and len(wert) > limit:
            return 413, {"error": f"Feld '{feld}' ist zu lang (maximal {limit} Zeichen)"}
    if payload.get("scope", "personal") not in ("personal", "geteilt"):
        return 422, {"error": "Feld 'scope' muss personal|geteilt sein"}
    if payload.get("governance_typ", "IN") not in GUELTIGE_GOVERNANCE:
        return 422, {"error": "Feld 'governance_typ' muss FP|FF|IN sein"}
    if payload.get("schadensintensitaet", "mittel") not in GUELTIGE_SCHADEN:
        return 422, {"error": "Feld 'schadensintensitaet' muss niedrig|mittel|hoch sein"}
    if payload.get("konfidenz", "mittel") not in ("niedrig", "mittel", "hoch"):
        return 422, {"error": "Feld 'konfidenz' muss niedrig|mittel|hoch sein"}
    # Form-Validierung: ein String statt Liste würde sonst zeichenweise
    # zersplittert STILL in die Festwissens-YAML geschrieben.
    norm_anker = payload.get("norm_anker")
    if norm_anker is not None and (
            not isinstance(norm_anker, list)
            or not all(isinstance(x, str) for x in norm_anker)):
        return 422, {"error": "Feld 'norm_anker' muss eine Liste von Strings sein"}
    if isinstance(norm_anker, list) and (
            len(norm_anker) > _MAX_NORM_ANKER
            or any(len(x) > _MAX_NORM_ANKER_CHARS for x in norm_anker)):
        return 413, {"error": "Feld 'norm_anker' ueberschreitet das Ressourcenlimit"}
    # § 203: Dieser Store ist Festwissen und wird an JEDEN Client ausgeliefert.
    # Ein verbatim-Write darf daher in KEINEM persistierten Freitextfeld ein
    # Mandantengeheimnis tragen – Begriff, Definition, Actor-Tags, Katalog-ID
    # und Normanker laufen gemeinsam durch denselben Detektor wie der Export.
    pii_felder = [("begriff", begriff), ("definition_text", definition_text)]
    pii_felder.extend((f"norm_anker[{i}]", wert)
                      for i, wert in enumerate(norm_anker or []))
    for feld in ("autor", "datum", "katalog_id"):
        wert = payload.get(feld)
        if isinstance(wert, str):
            pii_felder.append((feld, wert))
    rest = _pii_rest_fuer_felder(pii_felder)
    if rest:
        return 422, {"error": "Festwissensfelder enthalten identifizierende Angaben "
                              "(§ 203 StGB) – bitte abstrahieren.", "rest": rest}
    sauber = dict(payload)
    sauber["begriff"] = begriff
    sauber["definition_text"] = definition_text
    return 200, app.definiere(sauber)


def _katalog_review(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/katalog/review – Review-Lebenszyklus eines GETEILTEN Berater-
    Eintrags: {id, status: entwurf|geprüft|freigegeben, pruefer}. Nur Vorwärts-
    Übergänge (entwurf → geprüft → freigegeben, berater.setze_review_status).

    Der Lebenszyklus ist die Kanonizitäts-Garantie des geteilten Festwissens –
    er MUSS durch /v1 laufen, damit der Host die Freigabe in seiner Audit-Chain
    protokollieren kann (die Engine auditiert nicht, s. web.review_setzen).
    Die Antwort nennt deshalb den VOLLSTÄNDIGEN Übergang (alter_status →
    neuer_status). Reiner Festwissens-State, kein Mandantenbezug (§ 203-unkritisch).
    """
    eintrag_id_roh = payload.get("id", "")
    if not isinstance(eintrag_id_roh, str):
        return 400, {"error": "Feld 'id' muss ein String sein"}
    eintrag_id = eintrag_id_roh.strip()
    if not eintrag_id:
        return 400, {"error": "Feld 'id' fehlt"}
    if len(eintrag_id) > _MAX_STRUKTURFELD_CHARS:
        return 413, {"error": f"Feld 'id' ist zu lang "
                              f"(maximal {_MAX_STRUKTURFELD_CHARS} Zeichen)"}
    status_roh = payload.get("status", "")
    if not isinstance(status_roh, str):
        return 400, {"error": "Feld 'status' muss ein String sein"}
    status = status_roh.strip()
    if not status:
        return 400, {"error": "Feld 'status' fehlt"}
    # Form-Validierung wie norm_anker in _katalog_definiere: ein unbekannter
    # Statuswert ist ein 422 (Form), kein Lebenszyklus-Fehler (400).
    if status not in REVIEW_REIHENFOLGE:
        return 422, {"error": "Feld 'status' muss eines von "
                              f"{'|'.join(REVIEW_REIHENFOLGE)} sein, nicht {status!r}"}
    pruefer_roh = payload.get("pruefer")
    if not isinstance(pruefer_roh, str) or not pruefer_roh.strip():
        return 400, {"error": "Feld 'pruefer' muss ein nicht-leerer String sein"}
    pruefer = pruefer_roh.strip()
    if len(pruefer) > _MAX_AKTEUR_CHARS:
        return 413, {"error": f"Feld 'pruefer' ist zu lang "
                              f"(maximal {_MAX_AKTEUR_CHARS} Zeichen)"}
    rest = _pii_rest_fuer_felder([("pruefer", pruefer)])
    if rest:
        return 422, {"error": "Feld 'pruefer' enthält identifizierende Angaben "
                              "(§ 203 StGB) – bitte ein internes Actor-Tag verwenden.",
                     "rest": rest}
    # Architekturgrenze: der gemeinsame Bearer-Token authentisiert den Client,
    # nicht die behauptete Person. Bis ein JWT-/mTLS-/Proxy-Principal existiert,
    # bleibt ``pruefer`` ein auditierbares Actor-Tag und darf deshalb weder
    # fehlen noch still durch einen generischen Default ersetzt werden.
    sauber = dict(payload)
    sauber.update({"id": eintrag_id, "status": status, "pruefer": pruefer})
    res = app.review_setzen(sauber)
    # BeraterFehler (Rückwärts-Übergang, unbekannte id) → 400 mit Fehlertext
    # (Muster _katalog_norm_kuratieren: ok:false trägt 'fehler').
    return (200 if res.get("ok") else 400), res


def _katalog_norm_kuratieren(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/katalog/norm_kuratieren – katalog-weite Norm-Kuratierung an EINEM
    Begriff: {katalog_id, norm, aktion: verwerfen|ergaenzen|zuruecksetzen, scope, autor}.
    Rein strukturell (Begriff-ID + Norm-Zitat + aktion + Autor-Tag) – KEIN Fall-/
    Mandantentext, daher § 203-unkritisch und fallzustandslos. Die Engine hält nur den
    Katalog-State; die Handlung (wer/wann) auditiert der Host in seiner Chain."""
    for feld in ("katalog_id", "norm", "aktion"):
        if not isinstance(payload.get(feld), str):
            return 400, {"error": f"Feld '{feld}' muss ein String sein"}
    for feld in ("scope", "autor", "datum"):
        if feld in payload and not isinstance(payload[feld], str):
            return 400, {"error": f"Feld '{feld}' muss ein String sein"}
    for feld, limit in (("katalog_id", _MAX_STRUKTURFELD_CHARS),
                        ("norm", _MAX_NORM_ANKER_CHARS),
                        ("aktion", _MAX_STRUKTURFELD_CHARS),
                        ("scope", _MAX_STRUKTURFELD_CHARS),
                        ("autor", _MAX_AKTEUR_CHARS),
                        ("datum", _MAX_DATUM_CHARS)):
        wert = payload.get(feld)
        if isinstance(wert, str) and len(wert) > limit:
            return 413, {"error": f"Feld '{feld}' ist zu lang (maximal {limit} Zeichen)"}
    if payload.get("scope", "personal") not in ("personal", "geteilt"):
        return 422, {"error": "Feld 'scope' muss personal|geteilt sein"}
    if payload["aktion"] not in ("verwerfen", "ergaenzen", "zuruecksetzen"):
        return 422, {"error": "Feld 'aktion' muss verwerfen|ergaenzen|zuruecksetzen sein"}
    pii_felder = [("norm", payload["norm"])]
    for feld in ("autor", "datum", "katalog_id"):
        wert = payload.get(feld)
        if isinstance(wert, str):
            pii_felder.append((feld, wert))
    rest = _pii_rest_fuer_felder(pii_felder)
    if rest:
        return 422, {"error": "Kuratierungsfelder enthalten identifizierende Angaben "
                              "(§ 203 StGB) – bitte abstrahieren.", "rest": rest}
    res = app.kuratiere_norm(payload)
    return (200 if res.get("ok") else 400), res


def _katalog_kuratierung(app, query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    """GET /v1/katalog/kuratierung?katalog_id=&nutzer= – geltende Norm-Kuratierung
    zum Überlagern im Host. Ohne ``katalog_id`` alle Roh-Records (geteilt + die
    persönlichen des Nutzers), mit ``katalog_id`` die aufbereitete Sicht eines Begriffs
    ({verworfen, ergaenzt}). Reiner Katalog-State, kein Fallbezug; /v1/analyse bleibt
    davon unberührt (katalog-rein) – die Anwendung der Kuratierung macht der Host."""
    katalog_ids = query.get("katalog_id")
    nutzer_ids = query.get("nutzer")
    katalog_id = katalog_ids[0] if katalog_ids else None
    nutzer = nutzer_ids[0] if nutzer_ids else None
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
    text = payload.get("text", "")
    if not isinstance(text, str):
        return 400, {"error": "Feld 'text' muss ein String sein"}
    # k aus dem Request-Body robust parsen + clampen: unbegrenztes/negatives k
    # ist ein DoS-Vektor (riesiger Sort+Serialize auf einem Worker-Thread).
    k_roh = payload.get("k", 5)
    if isinstance(k_roh, bool) or not isinstance(k_roh, int):
        return 400, {"error": "Feld 'k' muss eine Ganzzahl sein"}
    k = k_roh
    k = max(1, min(k, 50))
    embedding, embedding_encoder = app.embedding_snapshot()
    if embedding is None or embedding_encoder is None:
        return 200, {"treffer": [], "grund": "kein Embedding-Index",
                     "engineVersion": V1_VERSION}
    hits = embedding.suche(text, embedding_encoder, k=k, min_sim=0.0)
    return 200, {"treffer": hits, "engineVersion": V1_VERSION}


def _embedding_status(app) -> tuple[int, dict[str, Any]]:
    """GET /v1/embedding/status – Live-Index, Job und globaler Zeitplan."""
    return 200, {
        "ok": True,
        "engineVersion": V1_VERSION,
        **app.embedding_operator_status(),
    }


def _embedding_refresh(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/embedding/refresh – asynchron und pro Prozess single-flight."""
    force = payload.get("force", False)
    if not isinstance(force, bool):
        return 400, {"error": "Feld 'force' muss ein Boolean sein"}
    from .embedding_operator import EmbeddingOperatorUnavailable
    try:
        job = app.embedding_refresh(force=force)
    except EmbeddingOperatorUnavailable as exc:
        return 503, {"error": str(exc)}
    return 202, {"ok": True, "engineVersion": V1_VERSION, **job}


def _embedding_cancel(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/embedding/cancel – bricht genau den beobachteten Job ab."""
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
        return 400, {"error": "Feld 'job_id' muss eine gueltige Job-ID sein"}
    from .embedding_operator import (
        EmbeddingOperatorConflict,
        EmbeddingOperatorUnavailable,
    )
    try:
        job = app.embedding_cancel(job_id=job_id)
    except EmbeddingOperatorConflict as exc:
        return 409, {"error": str(exc)}
    except EmbeddingOperatorUnavailable as exc:
        return 503, {"error": str(exc)}
    return 202, {"ok": True, "engineVersion": V1_VERSION, **job}


def _embedding_schedule(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/embedding/schedule – globaler Check-Rhythmus (1..365 Tage)."""
    enabled = payload.get("enabled")
    interval_days = payload.get("interval_days")
    if not isinstance(enabled, bool):
        return 400, {"error": "Feld 'enabled' muss ein Boolean sein"}
    if (isinstance(interval_days, bool) or not isinstance(interval_days, int)
            or not 1 <= interval_days <= 365):
        return 400, {"error": "Feld 'interval_days' muss eine Ganzzahl 1..365 sein"}
    from .embedding_operator import EmbeddingOperatorUnavailable
    try:
        schedule = app.embedding_schedule(
            enabled=enabled, interval_days=interval_days
        )
    except EmbeddingOperatorUnavailable as exc:
        return 503, {"error": str(exc)}
    return 200, {
        "ok": True,
        "engineVersion": V1_VERSION,
        "schedule": schedule,
    }


def _status(app) -> tuple[int, dict[str, Any]]:
    """GET /v1/status – Engine-Zustand (Degradations-Transparenz, seit 1.2.0).

    Welche Schichten sind im konsistenten Runtime-Snapshot geladen (Trigger/Graph/
    Embedding/Reranker/NER), inkl. Degradations-GRUND und deterministischer
    Fingerprints – die Engine degradiert bewusst an vielen Stellen, und für
    die Haftungsakte des Hosts muss dieser Zustand abfragbar sein, nicht nur
    im Betreiber-Log stehen. Bewusst KEINE Live-Prüfung (das LLM bleibt lazy;
    Erreichbarkeit liefert GET /v1/llm/status) und KEINE Zeitstempel.
    Derselbe Snapshot hängt als ``engine_status`` an jeder /v1/analyse-Antwort
    (dort zusätzlich mit den effektiven Lauf-``optionen``).
    """
    return 200, {
        "ok": True,
        "engineVersion": V1_VERSION,
        "engine_status": app.engine_status(),
        # Live und additiv: Legacy-Records, die beim Upgrade auf das harte
        # Vier-Augen-Gate sicher herabgestuft wurden, dürfen weder verschwinden
        # noch still als kanonisch weiterlaufen. Der Host kann pro ID eine neue
        # Review-Aufgabe anlegen; die Engine gibt sie niemals automatisch frei.
        "festwissen_status": app.berater_status(),
    }


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


# --- /v1/los/* (Quantenlos — beweisbar blinde Stichprobe, seit 1.3.0) --------

_LOS_BACKENDS = ("qpu", "simulator", "csprng")
_MAX_LOS_RAHMEN_ITEMS = 100_000
_MAX_LOS_ITEM_CHARS = 1_024


def _los_rahmen(payload: dict[str, Any]) -> tuple[list[str] | None,
                                                  tuple[int, dict[str, Any]] | None]:
    """Form-Validierung des Prüf-Rahmens (alle drei /v1/los/*-Routen).

    Der Rahmen ist eine nicht-leere Liste von ID-/Hash-Strings (NIE Inhalte,
    § 203 — der Host baut sie aus seiner Chain). Alles andere ist ein 400,
    BEVOR Quantenlos-Code läuft (Muster _katalog_definiere/norm_anker)."""
    rahmen = payload.get("rahmen")
    if not isinstance(rahmen, list) or not rahmen:
        return None, (400, {"error": "Feld 'rahmen' muss ein nicht-leeres "
                                     "Array von ID-Strings sein"})
    if len(rahmen) > _MAX_LOS_RAHMEN_ITEMS:
        return None, (413, {"error": "Feld 'rahmen' enthält zu viele Items "
                                     f"(maximal {_MAX_LOS_RAHMEN_ITEMS})"})
    if not all(isinstance(x, str) and x.strip() for x in rahmen):
        return None, (400, {"error": "Feld 'rahmen' muss ein nicht-leeres "
                                     "Array von ID-Strings sein"})
    if any(len(x) > _MAX_LOS_ITEM_CHARS for x in rahmen):
        return None, (413, {"error": "Ein Rahmen-Item ist zu lang "
                                     f"(maximal {_MAX_LOS_ITEM_CHARS} Zeichen)"})
    return rahmen, None


def _los_k(payload: dict[str, Any],
           rahmen: list[str]) -> tuple[int | None, tuple[int, dict[str, Any]] | None]:
    """k validieren: Ganzzahl >= 1 (400), k <= n (422 — Form ok, aber mehr
    ziehen als der Rahmen hergibt ist semantisch unerfüllbar). n zählt
    EINDEUTIGE Items, exakt wie quantenlos.rahmen_normalisieren dedupliziert."""
    k = payload.get("k")
    # bool ist int-Subtyp: {"k": true} wäre sonst still k=1.
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        return None, (400, {"error": "Feld 'k' muss eine Ganzzahl >= 1 sein"})
    n = len({x.strip() for x in rahmen})
    if k > n:
        return None, (422, {"error": f"k ({k}) übersteigt den Rahmen "
                                     f"(n={n} eindeutige Items)"})
    return k, None


def _los_ibm_token(payload: dict[str, Any]) -> tuple[str | None,
                                                     tuple[int, dict[str, Any]] | None]:
    """Optionaler IBM-Token pro Request (``ibm_token``, seit 1.3.1).

    Der Host verwaltet den Token zentral (z. B. verschlüsselt in seiner
    Konsolen-Config) und reicht ihn mit — die Engine bleibt zustandslos:
    nie gespeichert (kein ``save_account``), nie geloggt, nie in der Antwort.
    Fehlt das Feld (oder ist leer), gilt die Maschinen-Auflösung
    (``QISKIT_IBM_TOKEN`` / gespeichertes Konto). Falscher Typ → 400."""
    token = payload.get("ibm_token")
    if token is None:
        return None, None
    if not isinstance(token, str):
        return None, (400, {"error": "Feld 'ibm_token' muss ein String sein "
                                     "(optional — fehlt er, gilt der "
                                     "Maschinen-Zugang der Engine)"})
    if len(token) > _MAX_IBM_TOKEN_CHARS:
        return None, (413, {"error": "Feld 'ibm_token' ist zu lang"})
    return token.strip() or None, None


def _los_ziehen(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/los/ziehen – beweisbar blinde Stichprobe (Quantenlos, Protokoll v1).

    Body: {rahmen: [str,…], k: int, backend: "qpu"|"simulator"|"csprng",
    ibm_token?: str}. ``backend`` ist PFLICHT — der Host entscheidet die
    Entropie-Quelle bewusst (kein stiller Default: wer Attestierung will, sagt
    qpu; wer lokal würfelt, sagt es laut). ``ibm_token`` ist der optionale,
    zentral vom Host verwaltete IBM-Zugang (s. ``_los_ibm_token``). Antworten:
      * simulator/csprng → 200 {ok, status:"fertig", nachweis, roh_bits_hex}
      * qpu → 200 {ok, status:"wartet", job_id, backend, commitment, k}
        (Job läuft in der IBM-Queue — Abholen über POST /v1/los/abholen)
      * 400 (rahmen/k/backend-Form), 422 (k>n), 503 {ok:false, fehler}
        (Quanten-Extras fehlen bzw. qpu ohne IBM-Credentials).
    Zustandslos: die Engine archiviert weder Nachweis noch Roh-Bits — beides
    legt der Host in seine Chain (roh_bits_hex macht die volle Ableitungskette
    später offline nachrechenbar, s. /v1/los/pruefen)."""
    rahmen, fehler = _los_rahmen(payload)
    if fehler:
        return fehler
    assert rahmen is not None
    k, fehler = _los_k(payload, rahmen)
    if fehler:
        return fehler
    assert k is not None
    backend = payload.get("backend")
    if backend not in _LOS_BACKENDS:
        return 400, {"error": "Feld 'backend' muss qpu|simulator|csprng sein "
                              "(Pflicht — der Host wählt die Entropie-Quelle "
                              "bewusst, kein stiller Default)"}
    ibm_token, fehler = _los_ibm_token(payload)
    if fehler:
        return fehler
    res = app.los_ziehen(rahmen, k, backend, ibm_token=ibm_token)
    # ok:false hat hier genau eine Klasse: fehlende Betriebsvoraussetzung
    # (Quanten-Extras/IBM-Credentials) → 503 (Muster /v1/radar ohne Graph).
    return (200 if res.get("ok") else 503), {**res, "engineVersion": V1_VERSION}


def _los_abholen(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/los/abholen – eingereichten qpu-Job abholen (zustandslos).

    Body: {job_id: str, rahmen: [str,…], k: int, ibm_token?: str} — exakt die
    Werte, die der Host seit /v1/los/ziehen selbst trägt (``ibm_token``
    optional, s. ``_los_ibm_token``); die Engine hält kein Job-Registry,
    sondern REKONSTRUIERT die Ziehung (Commitment aus dem Rahmen, Counts über
    die IBM-Job-API) und prüft die Tag-Bindung Job↔Rahmen hart. Antworten:
    fertig-/wartet-Form wie /v1/los/ziehen; 400 (Form bzw. Job gehört nicht zu
    diesem Rahmen — ``bindung_verletzt:true``), 422 (k>n), 503 (Extras/
    Credentials fehlen)."""
    job_id_roh = payload.get("job_id")
    if not isinstance(job_id_roh, str):
        return 400, {"error": "Feld 'job_id' muss ein String sein"}
    job_id = job_id_roh.strip()
    if not job_id:
        return 400, {"error": "Feld 'job_id' fehlt"}
    if len(job_id) > _MAX_LOS_JOB_ID_CHARS:
        return 413, {"error": f"Feld 'job_id' ist zu lang "
                              f"(maximal {_MAX_LOS_JOB_ID_CHARS} Zeichen)"}
    rahmen, fehler = _los_rahmen(payload)
    if fehler:
        return fehler
    assert rahmen is not None
    k, fehler = _los_k(payload, rahmen)
    if fehler:
        return fehler
    assert k is not None
    ibm_token, fehler = _los_ibm_token(payload)
    if fehler:
        return fehler
    res = app.los_abholen(job_id, rahmen, k, ibm_token=ibm_token)
    if res.get("ok"):
        return 200, {**res, "engineVersion": V1_VERSION}
    # bindung_verletzt = Client-Fehler (falscher job_id↔rahmen-Mix) → 400;
    # sonst fehlende Betriebsvoraussetzung → 503. Strukturelles Feld statt
    # Fehlertext-Parsing (Muster-Begründung wie bei _radar).
    return (400 if res.get("bindung_verletzt") else 503), {**res, "engineVersion": V1_VERSION}


def _los_pruefen(app, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /v1/los/pruefen – Nachweis gegen die Rahmenliste nachrechnen.

    Body: {nachweis: {…}, rahmen: [str,…], online?: bool, roh_bits_hex?: str,
    ibm_token?: str}. Offline immer: Commitment, n, Stichprobe⊆Rahmen. Mit
    roh_bits_hex (vom Host neben dem Nachweis archiviert) prüft zusätzlich
    die volle Ableitungskette. QPU-Nachweise benötigen außerdem online:true
    (Counts via IBM-Job-ID erneut abrufen; ibm_token optional für den Refetch),
    damit ihre Dritt-Attestierung belegt ist.
    → 200 {ok:true, gueltig: bool, geprueft: […], hinweise: […]} — ein
    WIDERSPRUCH oder eine unvollständige/nicht durchführbare Vollprüfung ist
    gueltig:false (die Prüfung fand statt), kein HTTP-Fehler. 400 nur für
    Form-Fehler des Requests."""
    nachweis = payload.get("nachweis")
    if not isinstance(nachweis, dict) or not nachweis:
        return 400, {"error": "Feld 'nachweis' muss ein JSON-Objekt sein "
                              "(die Antwort von /v1/los/ziehen|abholen)"}
    rahmen, fehler = _los_rahmen(payload)
    if fehler:
        return fehler
    roh_bits = None
    if "roh_bits_hex" in payload:
        roh_hex = payload["roh_bits_hex"]
        if not isinstance(roh_hex, str) or not roh_hex:
            return 400, {"error": "Feld 'roh_bits_hex' muss ein Hex-String sein"}
        if (len(roh_hex) > _MAX_LOS_ROH_HEX_CHARS
                or len(roh_hex) % 2
                or any(c not in "0123456789abcdefABCDEF" for c in roh_hex)):
            return 400, {"error": "Feld 'roh_bits_hex' ist kein kanonischer Hex-String"}
        try:
            roh_bits = bytes.fromhex(roh_hex)
        except ValueError:
            return 400, {"error": "Feld 'roh_bits_hex' ist kein gültiger Hex-String"}
    ibm_token, fehler = _los_ibm_token(payload)
    if fehler:
        return fehler
    online, bool_fehler = _bool_feld(payload, "online")
    if bool_fehler:
        return 400, {"error": bool_fehler}
    erg = app.los_pruefen(nachweis, rahmen, roh_bits=roh_bits,
                          online=online,
                          ibm_token=ibm_token)
    return 200, {"ok": True, **erg, "engineVersion": V1_VERSION}


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
                        "/v1/katalog/kuratierung", "/v1/llm/status",
                        "/v1/embedding/status"})
_POST_PFADE = frozenset({"/v1/analyse", "/v1/katalog/definiere", "/v1/katalog/norm_kuratieren",
                         "/v1/katalog/review", "/v1/embedding/suche",
                         "/v1/embedding/refresh", "/v1/embedding/cancel",
                         "/v1/embedding/schedule",
                         "/v1/los/ziehen", "/v1/los/abholen", "/v1/los/pruefen",
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
                         "embedding": app.embedding_snapshot()[0] is not None}
        if pfad == "/v1/status":
            return _status(app)
        if pfad == "/v1/embedding/status":
            return _embedding_status(app)
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
        if pfad == "/v1/embedding/refresh":
            return _embedding_refresh(app, payload)
        if pfad == "/v1/embedding/cancel":
            return _embedding_cancel(app, payload)
        if pfad == "/v1/embedding/schedule":
            return _embedding_schedule(app, payload)
        if pfad == "/v1/los/ziehen":
            return _los_ziehen(app, payload)
        if pfad == "/v1/los/abholen":
            return _los_abholen(app, payload)
        if pfad == "/v1/los/pruefen":
            return _los_pruefen(app, payload)
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
