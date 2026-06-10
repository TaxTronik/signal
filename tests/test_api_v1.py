"""Engine-API /v1/* – der Integrationsvertrag mit TaxTronik.

Diese Tests frieren den Vertrag ein: Form der Antworten + Zustandslosigkeit.
Sie laufen gegen ``dispatch`` direkt (ohne HTTP-Server), mit einer echten
``Anwendung`` auf temporärem, mandantenfreiem Verzeichnis.
"""

import hashlib

import pytest

from risk_layer.api_v1 import dispatch, V1_VERSION
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


def test_v1_mitLLM_false_erzeugt_keine_llm_oder_embedding_spans(app):
    """Die schnelle Phase liefert nur deterministische Layer (kein llm/embedding-Span)."""
    code, obj = dispatch(app, "POST", "/v1/analyse",
                         {"text": _ARTEFAKT_TEXT, "mitLLM": False}, {})
    assert code == 200
    layer = {s["layer"] for s in obj["spans"]}
    assert "llm" not in layer and "embedding" not in layer
    # 'embedding' darf grundsätzlich nie ein Span-Layer sein (nur semantik-Strom)
    assert all(s["layer"] != "embedding" for s in obj["spans"])
