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
    assert not (tmp_path).glob("*.json").__class__ is None  # tmp existiert
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
