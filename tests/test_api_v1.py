"""Engine-API /v1/* – der Integrationsvertrag mit TaxTronik.

Diese Tests frieren den Vertrag ein: Form der Antworten + Zustandslosigkeit.
Sie laufen gegen ``dispatch`` direkt (ohne HTTP-Server), mit einer echten
``Anwendung`` auf temporärem, mandantenfreiem Verzeichnis.
"""

import hashlib
import json
import sys

import pytest

from risk_layer import quantenlos as QM
from risk_layer import radar
from risk_layer.api_v1 import dispatch, V1_VERSION
from risk_layer.berater import BeraterStore
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.web import Anwendung


@pytest.fixture()
def app(tmp_path):
    # graph_pfad=None ⇒ Default-Graph (corpus/graph.sqlite), falls vorhanden
    return Anwendung(None, str(tmp_path))


def test_health_vertrag(app):
    code, obj = dispatch(app, "GET", "/v1/health", {}, {})
    assert code == 200
    assert obj["ok"] is True
    assert obj["engineVersion"] == V1_VERSION
    assert "katalogVersion" in obj


def test_katalog_lesen_traegt_normketten(app):
    code, obj = dispatch(app, "GET", "/v1/katalog", {}, {})
    assert code == 200 and obj["begriffe"]
    e = next(b for b in obj["begriffe"] if b["id"] == "kst_finanzielle_eingliederung")
    assert e["governance_typ"] in ("FP", "FF", "IN")
    # die Normkette (Kaskade) ist Teil des Vertrags
    assert e["normketten"] and e["normketten"][0]["glieder"][0].startswith("§ 14 KStG")


def test_analyse_ist_zustandslos_und_typisiert(app, tmp_path):
    text = "Bareinnahmen tauchen im Kassenbuch gar nicht auf, die Belege fehlen."
    code, obj = dispatch(app, "POST", "/v1/analyse", {"text": text, "mitLLM": False}, {})
    assert code == 200
    for key in ("spans", "karten", "risiken", "semantik", "summary", "text_hash",
                "katalog_version", "engineVersion"):
        assert key in obj, key
    # ZUSTANDSLOS: kein Fall/keine Aufgabe wurde nebenbei persistiert
    assert list(tmp_path.rglob("*.json")) == []


def test_analyse_mitLLM_default_false_ist_schnell(app):
    # ohne mitLLM darf kein LLM-Kandidat entstehen (deterministischer Pfad)
    code, obj = dispatch(app, "POST", "/v1/analyse", {"text": "Eine Schätzung droht."}, {})
    assert code == 200
    assert all(r.get("quelle") != "llm" for r in obj["risiken"])


def test_definiere_ist_einzige_schreiboperation(app):
    code, obj = dispatch(app, "POST", "/v1/katalog/definiere", {
        "begriff": "Testbegriff Integration",
        "definition_text": "Arbeitsdefinition aus erledigter Recherche.",
        "scope": "personal", "autor": "junior",
    }, {})
    assert code == 200 and obj["ok"] is True and obj["id"]


def test_definiere_ohne_begriff_400(app):
    code, obj = dispatch(app, "POST", "/v1/katalog/definiere", {"definition_text": "x"}, {})
    assert code == 400


def test_definiere_blockt_mandantengeheimnis_203(app):
    """§ 203: der verbatim in den geteilten Store geschriebene Text darf keine
    identifizierenden Angaben tragen (er wird an JEDEN Client ausgeliefert)."""
    code, obj = dispatch(app, "POST", "/v1/katalog/definiere", {
        "begriff": "Spezialfall",
        "definition_text": "Gilt für die Muster GmbH, vereinbart am 1. Januar 2020.",
        "scope": "geteilt",
    }, {})
    assert code == 422
    assert obj["rest"]                  # nennt die erkannten identifizierenden Angaben


def test_v1_analyse_ignoriert_client_llm_url_ssrf(app, monkeypatch):
    """Ein client-geliefertes llm_url darf die LLM-Calls der Engine NICHT umlenken
    (blinde SSRF auf der netz-exponierten Maschinen-API)."""
    erfasst = {}

    def fake_analyse(text, **kw):
        erfasst.update(kw)
        return {"spans": [], "karten": [], "risiken": [], "semantik": [],
                "summary": {}, "text_hash": "x", "katalog_version": "v"}

    monkeypatch.setattr(app, "analyse", fake_analyse)
    dispatch(app, "POST", "/v1/analyse",
             {"text": "x", "mitLLM": True,
              "optionen": {"llm_url": "http://169.254.169.254/latest/"}}, {})
    assert erfasst["llm_url"] is None


def test_embedding_suche_robust_gegen_unsinniges_k(app):
    """k aus dem Body wird robust geparst/geclamped – kein 500, kein DoS."""
    for k in ("abc", -1, 10 ** 9):
        code, _ = dispatch(app, "POST", "/v1/embedding/suche", {"text": "x", "k": k}, {})
        assert code == 200


def test_embedding_suche_dedupliziert(app):
    code, obj = dispatch(app, "POST", "/v1/embedding/suche",
                         {"text": "Lohnsteuer nicht abgeführt.", "k": 5}, {})
    assert code == 200
    ids = [t["norm_id"] for t in obj["treffer"]]
    assert len(ids) == len(set(ids)), "jede Norm darf nur einmal vorkommen"


def test_unbekannter_v1_pfad_404(app):
    code, obj = dispatch(app, "GET", "/v1/gibtsnicht", {}, {})
    assert code == 404


def test_bekannter_pfad_falsche_methode_405(app):
    """Bekannter Pfad mit falscher Methode → 405 (nicht 404), mit allow-Feld."""
    code, obj = dispatch(app, "GET", "/v1/analyse", {}, {})
    assert code == 405 and obj["allow"] == "POST"
    code, obj = dispatch(app, "POST", "/v1/health", {}, {})
    assert code == 405 and obj["allow"] == "GET"
    # /v1/katalog/review ist eine bekannte POST-Route → GET liefert 405, nicht 404
    code, obj = dispatch(app, "GET", "/v1/katalog/review", {}, {})
    assert code == 405 and obj["allow"] == "POST"


# --- POST /v1/katalog/review (Review-Lebenszyklus des geteilten Festwissens) ---
# Der Lebenszyklus (entwurf → geprüft → freigegeben) ist die Kanonizitäts-
# Garantie des geteilten Festwissens; er läuft über /v1, damit der Host die
# Freigabe in seiner Audit-Chain protokollieren kann (die Engine auditiert nicht).

def _geteilter_entwurf(app, begriff="Testbegriff Review"):
    """Legt einen geteilten Eintrag (review_status=entwurf) an, liefert seine id."""
    code, obj = dispatch(app, "POST", "/v1/katalog/definiere", {
        "begriff": begriff,
        "definition_text": "Arbeitsdefinition aus erledigter Recherche.",
        "scope": "geteilt", "autor": "junior",
    }, {})
    assert code == 200 and obj["ok"] is True
    return obj["id"]


def test_review_antwort_traegt_vollstaendigen_uebergang(app):
    """Erfolgsfall: die Antwort macht den Übergang AUDITIERBAR — alter_status
    wird VOR der Mutation gelesen, der Host schreibt den vollen Übergang in
    seine Chain."""
    eid = _geteilter_entwurf(app)
    code, obj = dispatch(app, "POST", "/v1/katalog/review",
                         {"id": eid, "status": "geprüft", "pruefer": "senior"}, {})
    assert code == 200 and obj["ok"] is True
    assert obj["id"] == eid
    assert obj["alter_status"] == "entwurf"
    assert obj["neuer_status"] == "geprüft"
    assert obj["pruefer"] == "senior"


def test_review_rueckwaerts_uebergang_400(app):
    """Nur Vorwärts-Schritte (berater.setze_review_status): freigegeben →
    geprüft ist unzulässig → 400 mit Fehlertext."""
    eid = _geteilter_entwurf(app)
    code, _ = dispatch(app, "POST", "/v1/katalog/review",
                       {"id": eid, "status": "freigegeben", "pruefer": "senior"}, {})
    assert code == 200
    code, obj = dispatch(app, "POST", "/v1/katalog/review",
                         {"id": eid, "status": "geprüft", "pruefer": "senior"}, {})
    assert code == 400 and obj["ok"] is False
    assert "vorwärts" in obj["fehler"]


def test_review_unbekannte_id_400(app):
    code, obj = dispatch(app, "POST", "/v1/katalog/review",
                         {"id": "gibtsnicht", "status": "geprüft"}, {})
    assert code == 400 and obj["ok"] is False
    assert "gibtsnicht" in obj["fehler"]


def test_review_fehlender_oder_ungueltiger_status(app):
    """Validierung konsistent zu _katalog_definiere: fehlendes Pflichtfeld → 400,
    Form-Fehler (unbekannter Statuswert) → 422. Nichts wird mutiert."""
    # fehlender status → 400
    code, obj = dispatch(app, "POST", "/v1/katalog/review", {"id": "x"}, {})
    assert code == 400 and "status" in obj["error"]
    # unbekannter Statuswert → 422
    code, obj = dispatch(app, "POST", "/v1/katalog/review",
                         {"id": "x", "status": "genehmigt"}, {})
    assert code == 422 and "status" in obj["error"]
    # fehlende id → 400
    code, obj = dispatch(app, "POST", "/v1/katalog/review", {"status": "geprüft"}, {})
    assert code == 400 and "id" in obj["error"]


def test_analyse_optionen_nicht_objekt_400(app):
    """{"optionen": "x"} (truthy Nicht-Dict) → klarer 400, kein AttributeError/500."""
    code, obj = dispatch(app, "POST", "/v1/analyse",
                         {"text": "x", "optionen": "x"}, {})
    assert code == 400 and "optionen" in obj["error"]


def test_definiere_norm_anker_string_422(app, tmp_path):
    """norm_anker als String würde zeichenweise zersplittert → 422 statt stillem
    Datenmüll in der Festwissens-YAML."""
    code, obj = dispatch(app, "POST", "/v1/katalog/definiere", {
        "begriff": "Testbegriff", "definition_text": "Arbeitsdefinition.",
        "norm_anker": "§ 15 EStG",
    }, {})
    assert code == 422 and "norm_anker" in obj["error"]
    assert list(tmp_path.rglob("*.yaml")) == []      # nichts persistiert


def test_normgraph_aufloesen_ohne_id_400(app):
    """Parameter-Validierung VOR der Graph-Prüfung: fehlendes id → 400,
    auch wenn (gerade weil) kein Graph geladen ist."""
    code, obj = dispatch(app, "GET", "/v1/normgraph/aufloesen", {}, {})
    assert code == 400 and "id" in obj["error"]


def test_normgraph_und_embedding_tragen_engineVersion(app):
    """engineVersion ist in allen 200er-Antworten vorhanden (additiv ergänzt)."""
    code, obj = dispatch(app, "GET", "/v1/normgraph/suche", {}, {"q": ["§ 42 AO"]})
    assert code == 200 and obj["engineVersion"] == V1_VERSION
    code, obj = dispatch(app, "GET", "/v1/normgraph/aufloesen", {}, {"id": ["norm:AO:42"]})
    assert code == 200 and obj["engineVersion"] == V1_VERSION
    # kein-Index-Zweig von /v1/embedding/suche (Anwendung ohne gebauten Index)
    if app.embedding is None:
        code, obj = dispatch(app, "POST", "/v1/embedding/suche", {"text": "x"}, {})
        assert code == 200 and obj["engineVersion"] == V1_VERSION


def test_v1_llm_status_traegt_queue(app, monkeypatch):
    """GET /v1/llm/status liefert Status UND Queue (Slot-/Auslastungszustand)."""
    monkeypatch.setattr(app, "llm_status", lambda: {
        "url": "http://127.0.0.1:8080", "verfuegbar": True,
        "queue": {"quelle": "slots", "slots_gesamt": 4, "aktiv": 1, "frei": 3,
                  "slots": [{"id": 0, "aktiv": True}]},
        "modell_geladen": True, "binary_vorhanden": True, "von_uns_gestartet": False,
    })
    code, obj = dispatch(app, "GET", "/v1/llm/status", {}, {})
    assert code == 200
    for key in ("url", "verfuegbar", "queue", "modell_geladen", "binary_vorhanden",
                "von_uns_gestartet", "engineVersion"):
        assert key in obj, key
    assert obj["queue"]["slots_gesamt"] == 4 and obj["queue"]["aktiv"] == 1


def test_v1_llm_start_route(app, monkeypatch):
    """POST /v1/llm/start reicht das Ergebnis durch (+ engineVersion); kein Subprozess."""
    monkeypatch.setattr(app, "llm_start", lambda: {"ok": False, "fehler": "kein Modell (Test)"})
    code, obj = dispatch(app, "POST", "/v1/llm/start", {}, {})
    assert code == 200 and obj["ok"] is False and obj["engineVersion"]


def test_v1_llm_stop_route(app, monkeypatch):
    """POST /v1/llm/stop existiert als Gegenstück zu /v1/llm/start (der Host kann
    den llama-server nicht nur starten, sondern auch wieder stoppen = VRAM frei)."""
    aufgerufen = []
    monkeypatch.setattr(app, "llm_stop", lambda: aufgerufen.append(1) or {"ok": True})
    code, obj = dispatch(app, "POST", "/v1/llm/stop", {}, {})
    assert code == 200 and obj["ok"] is True and obj["engineVersion"] == V1_VERSION
    assert aufgerufen == [1]


def test_v1_llm_stop_idempotent_ohne_manager(app):
    """Ohne von uns gestarteten Server ist stop ein No-Op mit ok:true (idempotent)."""
    code, obj = dispatch(app, "POST", "/v1/llm/stop", {}, {})
    assert code == 200 and obj["ok"] is True


def test_analyse_karten_tragen_via_provenienz(app):
    """Jede Karte MUSS via tragen (wörtlich|muster) – die TS-Seite mappt darauf."""
    text = ("Bareinnahmen erscheinen nicht in der Kasse, Belege fehlen. "
            "Es droht die Schätzung der Besteuerungsgrundlagen.")
    code, obj = dispatch(app, "POST", "/v1/analyse", {"text": text, "mitLLM": False}, {})
    assert code == 200 and obj["karten"]
    for k in obj["karten"]:
        assert "via" in k and k["via"] in ("wörtlich", "muster")
    # mindestens ein wörtlicher und ein Muster-Treffer in diesem Text
    vias = {k["via"] for k in obj["karten"]}
    assert "wörtlich" in vias and "muster" in vias


# --- Load-bearing-Garantien fürs Highlight-Overlay (INTEGRATION_TAXTRONIK.md §4d) ---
# Diese zwei Tests nageln den Vertrag fest, den TaxTronik zum Rendern braucht.
# Bricht hier etwas, verschiebt sich live JEDES Highlight – also hart absichern.

# Text MIT echten PDF-/DOCX-Extraktions-Artefakten: Soft-Hyphen (U+00AD),
# zero-width-space (U+200B), no-break-space (U+00A0). Genau was unpdf/mammoth liefert.
_ARTEFAKT_TEXT = (
    "Die feh­lende Buchführung und nicht nachvollziehbare Kassenführung​ "
    "führen zur Schätzung der Besteuerungsgrundlagen. Es droht ein "
    "Gestaltungsmissbrauch nach § 42 AO."
)


def test_v1_spans_offsets_treffen_originaltext_exakt(app):
    """spans[].start/end MÜSSEN in den EXAKT übergebenen Text indizieren –
    auch bei PDF-Artefakten. text[start:end] == matched_text der Referenz."""
    code, obj = dispatch(app, "POST", "/v1/analyse",
                         {"text": _ARTEFAKT_TEXT, "mitLLM": False}, {})
    assert code == 200 and obj["spans"], "es sollte markiert werden"
    for s in obj["spans"]:
        ref = s["ref"]
        ziel = (obj["karten"][ref["index"]] if ref["type"] == "karte"
                else obj["risiken"][ref["index"]])
        scheibe = _ARTEFAKT_TEXT[s["start"]:s["end"]]
        assert scheibe == ziel["matched_text"], (
            f"Offset-Drift: text[{s['start']}:{s['end']}]={scheibe!r} "
            f"!= matched_text={ziel['matched_text']!r} (layer={s['layer']})")


def test_v1_texthash_ist_sha256_des_exakten_texts(app):
    """text_hash == sha256(text.encode('utf-8')) – byte-genau, für RiskAnalysis.textHash."""
    text = "Eine Schätzung der Besteuerungsgrundlagen droht wegen § 42 AO."
    code, obj = dispatch(app, "POST", "/v1/analyse", {"text": text, "mitLLM": False}, {})
    assert code == 200
    assert obj["text_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_v1_schnelle_phase_ist_deterministisch(app):
    """mitLLM:false: gleicher Text → identische Spans + identischer text_hash."""
    text = _ARTEFAKT_TEXT
    a = dispatch(app, "POST", "/v1/analyse", {"text": text, "mitLLM": False}, {})[1]
    b = dispatch(app, "POST", "/v1/analyse", {"text": text, "mitLLM": False}, {})[1]
    assert a["text_hash"] == b["text_hash"]
    spans_a = [(s["start"], s["end"], s["layer"]) for s in a["spans"]]
    spans_b = [(s["start"], s["end"], s["layer"]) for s in b["spans"]]
    assert spans_a == spans_b


# --- GET /v1/status + engine_status-Block (Degradations-Transparenz, 1.2.0) --
# Die Engine degradiert bewusst (Trigger/Graph/Embedding/Reranker/NER/LLM
# optional). Für die Haftungsakte muss der Engine-Zustand IN der Antwort
# stehen – der Host legt ihn mit dem Analyseergebnis in seiner Audit-Chain ab.

_SCHICHTEN = ("matcher", "muster", "trigger", "graph", "embedding",
              "reranker", "ner", "llm")


def test_v1_status_vertrag(app):
    code, obj = dispatch(app, "GET", "/v1/status", {}, {})
    assert code == 200 and obj["ok"] is True
    assert obj["engineVersion"] == V1_VERSION
    st = obj["engine_status"]
    assert st["katalog_version"] == app.katalog.version
    for schicht in _SCHICHTEN:
        assert schicht in st["schichten"], schicht
    # Kern-Schichten sind nie degradiert; conftest patcht index_oeffnen→None,
    # die Embedding-Schicht ist hier also nachweislich inaktiv UND begründet.
    assert st["schichten"]["matcher"]["aktiv"] is True
    assert st["schichten"]["muster"]["aktiv"] is True
    emb = st["schichten"]["embedding"]
    assert emb["aktiv"] is False
    assert emb["index_aktuell"] is None        # kein Index ⇒ nichts zu prüfen
    assert emb["grund"]                        # Degradation IMMER begründet
    # NER hat die ner_status()-Form ("verfuegbar" statt "aktiv" – Könnte-Aussage)
    assert set(st["schichten"]["ner"]) == {"verfuegbar", "modell", "grund"}
    # LLM: nur Konfiguration (KEINE Live-Prüfung beim Status); url nur host:port
    llm = st["schichten"]["llm"]
    assert llm["konfiguriert"] is True and llm["url"] == "127.0.0.1:8080"


def test_v1_status_post_405(app):
    """POST auf die neue GET-Route → 405 mit allow-Feld (Muster aller Routen)."""
    code, obj = dispatch(app, "POST", "/v1/status", {}, {})
    assert code == 405 and obj["allow"] == "GET"


def test_v1_status_liefert_kopie(app):
    """Mutationen am Antwort-Objekt dürfen den eingefrorenen Startup-Status
    nicht verändern (engine_status() liefert eine Kopie)."""
    a = dispatch(app, "GET", "/v1/status", {}, {})[1]["engine_status"]
    a["schichten"]["matcher"]["aktiv"] = False
    b = dispatch(app, "GET", "/v1/status", {}, {})[1]["engine_status"]
    assert b["schichten"]["matcher"]["aktiv"] is True


def test_v1_analyse_traegt_engine_status_mit_optionen(app):
    """/v1/analyse = Startup-Status (welche Schichten KONNTEN laufen) PLUS die
    effektiven Lauf-Optionen dieses Aufrufs (was WAR angefordert) – jedes
    archivierte Ergebnis ist damit selbsterklärend."""
    code, obj = dispatch(app, "POST", "/v1/analyse",
                         {"text": "Eine Schätzung droht.", "mitLLM": False,
                          "optionen": {"unknown": True}}, {})
    assert code == 200
    st = obj["engine_status"]
    assert st["katalog_version"] == obj["katalog_version"]
    for schicht in _SCHICHTEN:
        assert schicht in st["schichten"], schicht
    # berater ist in /v1 immer False (zustandslos), mit_llm spiegelt mitLLM
    assert st["optionen"] == {"unknown": True, "berater": False,
                              "abweichung": False, "mit_llm": False}


def test_v1_analyse_engine_status_bleibt_deterministisch(app):
    """Der Block enthält keine Zeitstempel/Zufallswerte: zwei Läufe liefern
    BYTE-identische engine_status-Blöcke (der Host darf sie hashen/chainen)."""
    import json
    a = dispatch(app, "POST", "/v1/analyse", {"text": _ARTEFAKT_TEXT}, {})[1]
    b = dispatch(app, "POST", "/v1/analyse", {"text": _ARTEFAKT_TEXT}, {})[1]
    assert json.dumps(a["engine_status"], ensure_ascii=False, sort_keys=True) == \
        json.dumps(b["engine_status"], ensure_ascii=False, sort_keys=True)


def test_v1_mitLLM_false_erzeugt_keine_llm_oder_embedding_spans(app):
    """Die schnelle Phase liefert nur deterministische Layer (kein llm/embedding-Span)."""
    code, obj = dispatch(app, "POST", "/v1/analyse",
                         {"text": _ARTEFAKT_TEXT, "mitLLM": False}, {})
    assert code == 200
    layer = {s["layer"] for s in obj["spans"]}
    assert "llm" not in layer and "embedding" not in layer
    # 'embedding' darf grundsätzlich nie ein Span-Layer sein (nur semantik-Strom)
    assert all(s["layer"] != "embedding" for s in obj["spans"])


# --- GET /v1/radar (Norm-Änderungs-Radar — Re-Review-Trigger, seit 1.2.0) ----
# Nach einem Korpus-Rebuild benennt das Radar betroffene Festwissens-Anker;
# der Host erzeugt daraus Review-Aufgaben und protokolliert in SEINER Chain
# (die Engine mutiert nichts, keine Status-Rücksetzung). Die Tests bauen echte
# Mini-Graphen in tmp_path (Muster tests/test_radar.py) und reichen graph_pfad
# IMMER explizit — der Default (corpus/graph.sqlite) wäre auf Dev-Maschinen
# umgebungsabhängig vorhanden.

def _knoten(nid: str, text: str) -> NormKnoten:
    teile = nid.split(":")
    return NormKnoten(id=nid, law=teile[1], para=teile[2], titel=teile[2], text=text)


def _graph_mit_vorher_manifest(tmp_path):
    """Mini-Korpus MIT Vergleichsstand: Stand ALT als Vorher-Manifest sichern
    (wie die Build-Rotation in cli._cmd_graph), dann den Graph mit geändertem
    Text neu bauen — norm:AO:162 ist danach 'geaendert', norm:KStG:8c
    'entfallen'. Liefert den Graph-DB-Pfad."""
    db = tmp_path / "graph.sqlite"
    g_alt = NormGraph.build(
        [_knoten("norm:AO:162", "Die Finanzbehörde schätzt."),
         _knoten("norm:KStG:8c", "Verlustabzug bei Körperschaften.")], db)
    radar.speichere_manifest(
        radar.manifest_vorher_pfad(db),
        radar.erstelle_manifest(g_alt, erstellt_am="2026-01-01T00:00:00"))
    # Windows: offenes Handle blockiert sonst das unlink() des Rebuilds.
    g_alt.close()
    NormGraph.build([_knoten("norm:AO:162", "GEÄNDERTER Wortlaut nach Rebuild.")],
                    db).close()
    return db


def test_v1_radar_ohne_graph_503(tmp_path):
    """Kein Graph gebaut → 503 (fehlende Betriebsvoraussetzung; Operator-Aktion
    'graph build' — anders als die degradierenden 200er von /v1/normgraph/*
    gibt es hier keinen brauchbaren Teil-Bericht)."""
    app = Anwendung(None, str(tmp_path), graph_pfad=str(tmp_path / "kein-graph.sqlite"))
    code, obj = dispatch(app, "GET", "/v1/radar", {}, {})
    assert code == 503 and obj["ok"] is False
    assert "Normgraph" in obj["fehler"]
    assert obj["engineVersion"] == V1_VERSION


def test_v1_radar_frische_installation_vergleichsstand_false(tmp_path):
    """Vor dem zweiten 'graph build' existiert KEIN Vorher-Manifest — das ist
    der Normalzustand einer frischen Installation, kein Defekt: 200 mit
    vergleichsstand:false und leerem, wohlgeformtem Bericht."""
    db = tmp_path / "graph.sqlite"
    NormGraph.build([_knoten("norm:AO:162", "Die Finanzbehörde schätzt.")], db).close()
    app = Anwendung(None, str(tmp_path / "berater"), graph_pfad=str(db))
    code, obj = dispatch(app, "GET", "/v1/radar", {}, {})
    assert code == 200 and obj["ok"] is True
    assert obj["vergleichsstand"] is False
    assert "graph build" in obj["hinweis"]
    assert obj["betroffen"] == []
    assert obj["anzahl"] == {"geaendert": 0, "neu": 0, "entfallen": 0,
                             "betroffen": 0, "freigegeben": 0}
    assert obj["engineVersion"] == V1_VERSION


def test_v1_radar_voller_durchstich(tmp_path):
    """Voller Durchstich: rotiertes Vorher-Manifest + geankerte FREIGEGEBENE
    geteilte Definition → der Bericht nennt sie als betroffenen Anker, vorn
    (Dringlichkeit), mit konsistenter anzahl. Genau daraus erzeugt der Host
    seine Review-Aufgaben."""
    import json
    db = _graph_mit_vorher_manifest(tmp_path)
    berater_dir = tmp_path / "berater"
    store = BeraterStore(berater_dir)
    d = store.definiere("Schätzungsrisiko", "Arbeitsdefinition.", autor="rey",
                        datum="2026-01-02", scope="geteilt", norm_anker=("§ 162 AO",))
    store.setze_review_status(d.id, "geprüft", reviewer="rv")
    store.setze_review_status(d.id, "freigegeben", reviewer="rv")

    app = Anwendung(None, str(berater_dir), graph_pfad=str(db))
    code, obj = dispatch(app, "GET", "/v1/radar", {}, {})
    assert code == 200 and obj["ok"] is True
    assert obj["vergleichsstand"] is True
    assert obj["engineVersion"] == V1_VERSION
    assert obj["stand_alt"]["erstellt_am"] == "2026-01-01T00:00:00"
    assert obj["stand_neu"]["erstellt_am"]            # Zeitstempel gewollt (Bericht)
    assert obj["normen"]["geaendert"] == ["norm:AO:162"]
    assert obj["normen"]["entfallen"] == ["norm:KStG:8c"]
    # die freigegebene Definition ist betroffen (norm:AO:162 geändert) …
    assert any(t["typ"] == "berater" and t["id"] == d.id
               and t["review_status"] == "freigegeben"
               and t["aenderung"] == "geaendert" for t in obj["betroffen"])
    # … und steht VORN (dringlichkeitssortiert: freigegeben/geteilt zuerst)
    assert obj["betroffen"][0]["review_status"] == "freigegeben"
    assert obj["anzahl"]["betroffen"] == len(obj["betroffen"]) >= 1
    assert obj["anzahl"]["freigegeben"] >= 1
    json.dumps(obj)                                   # /v1-tauglich serialisierbar


def test_v1_radar_korruptes_manifest_500(tmp_path):
    """Korruptes Vorher-Manifest → laut scheitern (500) statt still leerem
    Diff; der Fehlertext nennt die Datei (kein Mandantenbezug — er sagt dem
    Operator, was zu reparieren ist)."""
    db = tmp_path / "graph.sqlite"
    NormGraph.build([_knoten("norm:AO:162", "Text.")], db).close()
    radar.manifest_vorher_pfad(db).write_text("{kein json", encoding="utf-8")
    app = Anwendung(None, str(tmp_path / "berater"), graph_pfad=str(db))
    code, obj = dispatch(app, "GET", "/v1/radar", {}, {})
    assert code == 500 and obj["ok"] is False
    assert radar.MANIFEST_VORHER_NAME in obj["fehler"]
    assert obj["engineVersion"] == V1_VERSION


def test_v1_radar_post_405(app):
    """POST auf die GET-Route → 405 mit allow-Feld (Muster aller Routen)."""
    code, obj = dispatch(app, "POST", "/v1/radar", {}, {})
    assert code == 405 and obj["allow"] == "GET"


# --- POST /v1/los/* (Quantenlos — beweisbar blinde Stichprobe, seit 1.3.0) ----
# Vertragstests: exakte Antwortformen (fertig/wartet/Fehler), backend ist
# PFLICHT (kein stiller Default), 422 bei k>n, 503 bei fehlender Betriebs-
# voraussetzung (Quanten-Extras/IBM-Credentials). Der QPU-Pfad läuft komplett
# über die quantenlos-Seams (_qpu_job_einreichen/_qpu_job_holen) gemockt —
# Kanonisierung, Nachweisbau und Bindungs-Check bleiben der ECHTE Code;
# simulator/csprng laufen real (qiskit-aer ist in der Test-Umgebung da).

_LOS_RAHMEN = [f"text_hash:{i:04d}" for i in range(12)]

_NACHWEIS_KEYS = {"protokoll_version", "gezogen_am", "rahmen", "k",
                  "stichprobe", "entropie", "ableitung"}


def test_los_ziehen_csprng_fertig_form(app, tmp_path):
    """fertig-Form des Vertrags + Zustandslosigkeit (kein Nachweis-Archiv)."""
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": _LOS_RAHMEN, "k": 5, "backend": "csprng"}, {})
    assert code == 200 and obj["ok"] is True and obj["status"] == "fertig"
    assert obj["engineVersion"] == V1_VERSION
    n = obj["nachweis"]
    assert set(n) == _NACHWEIS_KEYS
    assert n["protokoll_version"] == QM.PROTOKOLL_VERSION
    assert n["rahmen"] == {"commitment": QM.rahmen_commitment(_LOS_RAHMEN), "n": 12}
    assert n["k"] == 5 and len(n["stichprobe"]) == 5
    assert set(n["stichprobe"]) <= set(_LOS_RAHMEN)
    assert n["entropie"]["quelle_klasse"] == "csprng_fallback"
    # roh_bits_hex (additiv): der Host archiviert die Roh-Bits NEBEN dem
    # Nachweis — die Engine hebt nichts auf.
    assert n["entropie"]["roh_counts_sha256"] == hashlib.sha256(
        bytes.fromhex(obj["roh_bits_hex"])).hexdigest()
    # ZUSTANDSLOS: weder Nachweis noch Job-Registry auf der Platte
    assert list(tmp_path.rglob("*")) == []


def test_los_ziehen_simulator_laeuft_echt(app):
    """Der Simulator-Pfad läuft ECHT (qiskit-aer, gleicher Schaltkreis wie die
    QPU) — und der Nachweis sagt selbst laut, dass er keine Attestierung hat."""
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": _LOS_RAHMEN, "k": 3, "backend": "simulator"}, {})
    assert code == 200 and obj["status"] == "fertig"
    e = obj["nachweis"]["entropie"]
    assert e["quelle_klasse"] == "simulator" and e["backend"] == "aer_simulator"
    assert "SIMULATOR" in e["hinweis"]


def test_los_ziehen_form_validierung_400(app):
    """rahmen (fehlend/kein Array/leer/leeres Item) und k (fehlend/kein int/
    bool/<1) sind 400er — BEVOR Quantenlos-Code läuft."""
    for body in ({"k": 1, "backend": "csprng"},
                 {"rahmen": "x", "k": 1, "backend": "csprng"},
                 {"rahmen": [], "k": 1, "backend": "csprng"},
                 {"rahmen": ["a", "  "], "k": 1, "backend": "csprng"},
                 {"rahmen": ["a", 7], "k": 1, "backend": "csprng"}):
        code, obj = dispatch(app, "POST", "/v1/los/ziehen", body, {})
        assert code == 400 and "rahmen" in obj["error"], body
    for k in (None, "3", 1.5, 0, -2, True):
        code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                             {"rahmen": ["a", "b"], "k": k, "backend": "csprng"}, {})
        assert code == 400 and "'k'" in obj["error"], k


def test_los_ziehen_backend_ist_pflicht_400(app):
    """backend ist PFLICHT — der Host wählt die Entropie-Quelle bewusst, es
    gibt keinen stillen Default (Attestierungs-Anspruch des Quantenlos)."""
    for body in ({"rahmen": ["a", "b"], "k": 1},
                 {"rahmen": ["a", "b"], "k": 1, "backend": ""},
                 {"rahmen": ["a", "b"], "k": 1, "backend": "drand"}):
        code, obj = dispatch(app, "POST", "/v1/los/ziehen", body, {})
        assert code == 400 and "backend" in obj["error"], body


def test_los_ziehen_k_groesser_n_422(app):
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": ["a", "b", "c"], "k": 4, "backend": "csprng"}, {})
    assert code == 422 and "k" in obj["error"]
    # n zählt EINDEUTIGE Items (rahmen_normalisieren dedupliziert): drei
    # Einträge, aber n=2 → k=3 ist schon unerfüllbar.
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": ["a", "b", "a"], "k": 3, "backend": "csprng"}, {})
    assert code == 422


def test_los_ziehen_qpu_ohne_credentials_503(app, monkeypatch):
    """qpu ohne IBM-Zugang: lauter 503 mit Anleitung — bewusst KEINE stille
    Degradation auf lokalen Zufall (Muster der CLI, Exit 2)."""
    monkeypatch.setattr(QM, "ibm_credentials_vorhanden", lambda token=None: False)
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": _LOS_RAHMEN, "k": 3, "backend": "qpu"}, {})
    assert code == 503 and obj["ok"] is False
    assert "IBM" in obj["fehler"] and "Credentials" in obj["fehler"]
    assert "pip install" in obj["fehler"] and "[quanten]" in obj["fehler"]
    assert obj["engineVersion"] == V1_VERSION


def test_los_ziehen_ohne_quanten_extras_503(app, monkeypatch):
    """Fehlendes qiskit = fehlende Betriebsvoraussetzung (503) mit Install-
    Hinweis (sys.modules-Block simuliert die schlanke Installation)."""
    for modul in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime"):
        monkeypatch.setitem(sys.modules, modul, None)
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": _LOS_RAHMEN, "k": 3, "backend": "simulator"}, {})
    assert code == 503 and obj["ok"] is False
    assert "pip install" in obj["fehler"] and "[quanten]" in obj["fehler"]


class _FakeQpuSeite:
    """Fake NUR der IBM-Seite über die quantenlos-Seams (einreichen/holen) —
    Kanonisierung, Nachweisbau und Bindungs-Check bleiben der echte Code."""

    def __init__(self):
        self.fertig = True
        self.commitments: dict[str, str] = {}
        self.tokens: list[str | None] = []      # pro Seam-Aufruf gesehener ibm_token

    def counts(self, commitment):
        basis = hashlib.sha256(commitment.encode()).hexdigest()
        return {bin(i)[2:].zfill(8): int(basis[i], 16) + 1 for i in range(8)}

    def einreichen(self, *, commitment, n_qubits, shots, backend_name=None,
                   service=None, token=None):
        assert shots * n_qubits >= 512          # MIN_ROH_BITS-Untergrenze gilt auch hier
        self.tokens.append(token)
        job_id = f"fake-job-{len(self.commitments)}"
        self.commitments[job_id] = commitment
        return {"job_id": job_id, "backend": "fake_torino",
                "tags": ["quantenlos", commitment[:32]]}

    def holen(self, job_id, *, service=None, token=None):
        self.tokens.append(token)
        if not self.fertig:
            return {"fertig": False, "status": "QUEUED", "backend": "fake_torino"}
        c = self.commitments[job_id]
        return {"fertig": True, "status": "DONE", "counts": self.counts(c),
                "tags": ["quantenlos", c[:32]], "backend": "fake_torino",
                "creation_date": "2026-06-11T12:00:00+00:00"}


@pytest.fixture()
def fake_qpu(monkeypatch):
    fake = _FakeQpuSeite()
    monkeypatch.setattr(QM, "ibm_credentials_vorhanden", lambda token=None: True)
    monkeypatch.setattr(QM, "_qpu_job_einreichen", fake.einreichen)
    monkeypatch.setattr(QM, "_qpu_job_holen", fake.holen)
    return fake


def test_los_ziehen_qpu_wartet_form_exakt(app, fake_qpu):
    """qpu reicht nur ein → wartet-Form EXAKT wie im Vertrag (der TS-Adapter
    typisiert dagegen): {ok, status, job_id, backend, commitment, k}."""
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": _LOS_RAHMEN, "k": 5, "backend": "qpu"}, {})
    assert code == 200
    assert obj == {"ok": True, "status": "wartet", "job_id": "fake-job-0",
                   "backend": "fake_torino",
                   "commitment": QM.rahmen_commitment(_LOS_RAHMEN), "k": 5,
                   "engineVersion": V1_VERSION}


def test_los_abholen_wartet_solange_job_in_queue(app, fake_qpu):
    fake_qpu.fertig = False
    _, start = dispatch(app, "POST", "/v1/los/ziehen",
                        {"rahmen": _LOS_RAHMEN, "k": 5, "backend": "qpu"}, {})
    code, obj = dispatch(app, "POST", "/v1/los/abholen",
                         {"job_id": start["job_id"], "rahmen": _LOS_RAHMEN, "k": 5}, {})
    assert code == 200 and obj["ok"] is True and obj["status"] == "wartet"
    assert obj["job_id"] == start["job_id"] and obj["k"] == 5
    assert obj["commitment"] == start["commitment"]
    assert obj["engineVersion"] == V1_VERSION


def test_los_abholen_fertig_rekonstruiert_und_prueft_roundtrip(app, fake_qpu):
    """Voller qpu-Durchstich: ziehen (wartet) → abholen (fertig — Nachweis
    REKONSTRUIERT aus job_id+rahmen+k, kein Job-Registry) → pruefen gültig
    inkl. voller Ableitungskette über roh_bits_hex."""
    _, start = dispatch(app, "POST", "/v1/los/ziehen",
                        {"rahmen": _LOS_RAHMEN, "k": 4, "backend": "qpu"}, {})
    code, obj = dispatch(app, "POST", "/v1/los/abholen",
                         {"job_id": start["job_id"], "rahmen": _LOS_RAHMEN, "k": 4}, {})
    assert code == 200 and obj["ok"] is True and obj["status"] == "fertig"
    n = obj["nachweis"]
    assert set(n) == _NACHWEIS_KEYS
    assert n["rahmen"]["commitment"] == start["commitment"] and n["k"] == 4
    e = n["entropie"]
    assert e["quelle_klasse"] == "qpu" and e["job_id"] == start["job_id"]
    assert e["job_tags"] == ["quantenlos", start["commitment"][:32]]
    # Roundtrip über die API: gleicher Rahmen → gültig, volle Kette
    code, erg = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": n, "rahmen": _LOS_RAHMEN,
                          "roh_bits_hex": obj["roh_bits_hex"]}, {})
    assert code == 200 and erg["ok"] is True and erg["gueltig"] is True
    assert {"rahmen_commitment", "roh_bits_hash", "ableitungskette"} <= set(erg["geprueft"])


def test_los_abholen_fremder_job_bindung_400(app, fake_qpu):
    """Tag-Bindung HART: ein Job, dessen Tags nicht das Commitment DIESES
    Rahmens tragen, wird nie zum Nachweis — 400 (Client-Fehler), kein
    nachträglich 'passender' Fremd-Job."""
    _, start = dispatch(app, "POST", "/v1/los/ziehen",
                        {"rahmen": _LOS_RAHMEN, "k": 3, "backend": "qpu"}, {})
    code, obj = dispatch(app, "POST", "/v1/los/abholen",
                         {"job_id": start["job_id"],
                          "rahmen": ["x:1", "x:2", "x:3", "x:4"], "k": 3}, {})
    assert code == 400 and obj["ok"] is False
    assert obj["bindung_verletzt"] is True and "Commitment" in obj["fehler"]


def test_los_abholen_form_und_credentials(app, monkeypatch):
    # job_id fehlt → 400 (vor allem anderen)
    code, obj = dispatch(app, "POST", "/v1/los/abholen",
                         {"rahmen": ["a", "b"], "k": 1}, {})
    assert code == 400 and "job_id" in obj["error"]
    # ohne IBM-Credentials kann nichts abgeholt werden → 503
    monkeypatch.setattr(QM, "ibm_credentials_vorhanden", lambda token=None: False)
    code, obj = dispatch(app, "POST", "/v1/los/abholen",
                         {"job_id": "j1", "rahmen": ["a", "b"], "k": 1}, {})
    assert code == 503 and obj["ok"] is False and "Credentials" in obj["fehler"]


def test_los_pruefen_roundtrip_und_manipulierter_rahmen(app):
    """Kern-Vertrag: ziehen → pruefen gültig (Commitment ist ordnungs-
    invariant); EIN getauschtes Rahmen-Item → gueltig:false, benannt."""
    _, zug = dispatch(app, "POST", "/v1/los/ziehen",
                      {"rahmen": _LOS_RAHMEN, "k": 5, "backend": "csprng"}, {})
    code, obj = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": zug["nachweis"],
                          "rahmen": list(reversed(_LOS_RAHMEN))}, {})
    assert code == 200 and obj["ok"] is True and obj["gueltig"] is True
    assert "rahmen_commitment" in obj["geprueft"]
    assert obj["engineVersion"] == V1_VERSION

    boese = _LOS_RAHMEN[:-1] + ["text_hash:9999"]
    code, obj = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": zug["nachweis"], "rahmen": boese}, {})
    assert code == 200 and obj["ok"] is True and obj["gueltig"] is False
    assert any("rahmen_commitment" in h for h in obj["hinweise"])


def test_los_pruefen_volle_kette_mit_roh_bits_hex(app):
    """roh_bits_hex (vom Host neben dem Nachweis archiviert) macht die VOLLE
    Ableitungskette offline nachrechenbar — eine untergeschobene, im Rahmen
    liegende Stichprobe fällt erst HIER auf."""
    import copy
    _, zug = dispatch(app, "POST", "/v1/los/ziehen",
                      {"rahmen": _LOS_RAHMEN, "k": 5, "backend": "csprng"}, {})
    code, obj = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": zug["nachweis"], "rahmen": _LOS_RAHMEN,
                          "roh_bits_hex": zug["roh_bits_hex"]}, {})
    assert code == 200 and obj["gueltig"] is True
    assert "ableitungskette" in obj["geprueft"]

    manipuliert = copy.deepcopy(zug["nachweis"])
    ersatz = next(i for i in _LOS_RAHMEN if i not in manipuliert["stichprobe"])
    manipuliert["stichprobe"][0] = ersatz
    code, obj = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": manipuliert, "rahmen": _LOS_RAHMEN,
                          "roh_bits_hex": zug["roh_bits_hex"]}, {})
    assert code == 200 and obj["gueltig"] is False
    assert any("ableitungskette" in h for h in obj["hinweise"])


def test_los_pruefen_online_mit_gemocktem_refetch(app, fake_qpu, monkeypatch):
    """online:true nutzt den Refetch-Seam (_qpu_job_abrufen): frisch geholte
    Counts + Tag-Check fließen in die Prüfung ein (volle Kette ohne Archiv)."""
    _, start = dispatch(app, "POST", "/v1/los/ziehen",
                        {"rahmen": _LOS_RAHMEN, "k": 3, "backend": "qpu"}, {})
    _, abh = dispatch(app, "POST", "/v1/los/abholen",
                      {"job_id": start["job_id"], "rahmen": _LOS_RAHMEN, "k": 3}, {})
    monkeypatch.setattr(QM, "_qpu_job_abrufen", lambda job_id, **kw: {
        "counts": fake_qpu.counts(start["commitment"]),
        "tags": ["quantenlos", start["commitment"][:32]],
        "backend": "fake_torino", "creation_date": "2026-06-11T12:00:00+00:00"})
    code, obj = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": abh["nachweis"], "rahmen": _LOS_RAHMEN,
                          "online": True}, {})
    assert code == 200 and obj["gueltig"] is True
    assert {"online_counts_hash", "online_job_tags",
            "ableitungskette"} <= set(obj["geprueft"])


def test_los_pruefen_form_validierung_400(app):
    code, obj = dispatch(app, "POST", "/v1/los/pruefen", {"rahmen": ["a"]}, {})
    assert code == 400 and "nachweis" in obj["error"]
    code, obj = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": {"x": 1}, "rahmen": []}, {})
    assert code == 400 and "rahmen" in obj["error"]
    code, obj = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": {"x": 1}, "rahmen": ["a"],
                          "roh_bits_hex": "zz"}, {})
    assert code == 400 and "roh_bits_hex" in obj["error"]


def test_los_ibm_token_pro_request_wird_durchgereicht(app, fake_qpu, monkeypatch):
    """ibm_token im Body (zentral vom Host verwaltet, seit 1.3.1): zählt als
    Credentials AUCH ohne Maschinen-Zugang, erreicht die IBM-Seams — und
    taucht in keiner Antwort auf (die Engine speichert/echot ihn nie)."""
    # Maschine ohne Zugang: nur der mitgereichte Token zählt (echte Semantik
    # von ibm_credentials_vorhanden, nicht das pauschale True der Fixture).
    monkeypatch.setattr(QM, "ibm_credentials_vorhanden", lambda token=None: bool(token))
    code, start = dispatch(app, "POST", "/v1/los/ziehen",
                           {"rahmen": _LOS_RAHMEN, "k": 3, "backend": "qpu",
                            "ibm_token": "ibm-tok-geheim"}, {})
    assert code == 200 and start["status"] == "wartet"
    assert fake_qpu.tokens == ["ibm-tok-geheim"]
    assert "ibm-tok-geheim" not in json.dumps(start)

    code, obj = dispatch(app, "POST", "/v1/los/abholen",
                         {"job_id": start["job_id"], "rahmen": _LOS_RAHMEN,
                          "k": 3, "ibm_token": "ibm-tok-geheim"}, {})
    assert code == 200 and obj["status"] == "fertig"
    assert fake_qpu.tokens == ["ibm-tok-geheim"] * 2
    assert "ibm-tok-geheim" not in json.dumps(obj)

    # pruefen online: Token erreicht den Refetch-Seam (_qpu_job_abrufen)
    gesehen = []

    def abrufen(job_id, token=None):
        gesehen.append(token)
        return {"counts": fake_qpu.counts(start["commitment"]),
                "tags": ["quantenlos", start["commitment"][:32]],
                "backend": "fake_torino", "creation_date": None}
    monkeypatch.setattr(QM, "_qpu_job_abrufen", abrufen)
    code, erg = dispatch(app, "POST", "/v1/los/pruefen",
                         {"nachweis": obj["nachweis"], "rahmen": _LOS_RAHMEN,
                          "online": True, "ibm_token": "ibm-tok-geheim"}, {})
    assert code == 200 and erg["gueltig"] is True
    assert gesehen == ["ibm-tok-geheim"]
    assert "ibm-tok-geheim" not in json.dumps(erg)


def test_los_ibm_token_form(app, monkeypatch):
    """Falscher Typ → 400; leerer String zählt als 'nicht gesetzt' (dann
    entscheidet wie bisher der Maschinen-Zugang → ohne ihn 503)."""
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": ["a", "b"], "k": 1, "backend": "qpu",
                          "ibm_token": 123}, {})
    assert code == 400 and "ibm_token" in obj["error"]

    monkeypatch.setattr(QM, "ibm_credentials_vorhanden", lambda token=None: bool(token))
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": ["a", "b"], "k": 1, "backend": "qpu",
                          "ibm_token": "  "}, {})
    assert code == 503 and obj["ok"] is False and "Credentials" in obj["fehler"]
    # und der 503-Text nennt die neue Abhilfe (Token pro Request)
    assert "ibm_token" in obj["fehler"]


def test_los_ziehen_ibm_fehler_503_ohne_token_im_text(app, monkeypatch):
    """Die IBM-Seite scheitert (z. B. ungültiger Token): saubere 503-Form
    statt nackter 500 — und der mitgereichte Token steht NIE im Fehlertext
    (er landet sonst in Logs und in der Audit-Chain des Hosts)."""
    monkeypatch.setattr(QM, "ibm_credentials_vorhanden", lambda token=None: True)

    def kaputt(**kw):
        raise RuntimeError(f"401 Unauthorized: token {kw.get('token')} invalid")
    monkeypatch.setattr(QM, "_qpu_job_einreichen", kaputt)
    code, obj = dispatch(app, "POST", "/v1/los/ziehen",
                         {"rahmen": _LOS_RAHMEN, "k": 2, "backend": "qpu",
                          "ibm_token": "ibm-tok-geheim"}, {})
    assert code == 503 and obj["ok"] is False
    assert "IBM-Quantum-Zugriff fehlgeschlagen" in obj["fehler"]
    assert "ibm-tok-geheim" not in json.dumps(obj)
    assert "[ibm_token]" in obj["fehler"]


def test_los_routen_falsche_methode_405(app):
    """GET auf die drei POST-Routen → 405 mit allow-Feld (Muster aller Routen)."""
    for pfad in ("/v1/los/ziehen", "/v1/los/abholen", "/v1/los/pruefen"):
        code, obj = dispatch(app, "GET", pfad, {}, {})
        assert code == 405 and obj["allow"] == "POST", pfad
