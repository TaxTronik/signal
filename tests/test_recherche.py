"""Rechercheauftrag-Export mit Geheimnis-Re-Check (§ 203 StGB)."""

from risk_layer.recherche import (
    _webhook_ziel_ok, baue_entwurf, export_payload, exportiere)

BEGRIFFE = [{"begriff": "organisatorische Eingliederung", "herkunft": "① wörtlich",
             "norm_anker": ["§ 2 Abs. 2 Nr. 2 UStG"]}]


def test_entwurf_abstrahiert_und_traegt_begriffe():
    e = baue_entwurf("Die M-GmbH ist organisatorisch eingegliedert.", BEGRIFFE)
    assert "M-GmbH" not in e["abstrahiert"]
    assert e["begriffe"] == BEGRIFFE
    assert e["frage"]
    # Rückübersetzungstabelle ist im Entwurf (lokal), trägt die Firma
    assert any(x["typ"] == "firma" for x in e["ersetzungen"])


def test_export_payload_enthaelt_keine_rueckuebersetzung():
    p = export_payload("Die [Gesellschaft A] ist eingegliedert.", BEGRIFFE, "Frage?")
    assert "ersetzungen" not in p
    assert p["sachverhalt"].startswith("Die [Gesellschaft A]")
    assert p["schema"].startswith("risk-layer/rechercheauftrag")


def test_export_verweigert_bei_restgeheimnis():
    r = exportiere("Die M-GmbH ist eingegliedert.", BEGRIFFE, "Frage?")
    assert r["ok"] is False
    assert r["grund"] == "geheimnis_verdacht"
    assert r["rest"]                     # nennt die noch enthaltene Firma


def test_export_ok_bei_abstrahiertem_text_ohne_webhook():
    r = exportiere("Die [Gesellschaft A] ist eingegliedert.", BEGRIFFE, "Frage?")
    assert r["ok"] is True and r["versendet"] is False
    assert r["payload"]["sachverhalt"].startswith("Die [Gesellschaft A]")


def test_export_force_ueberschreibt_warnung():
    r = exportiere("Die M-GmbH ...", BEGRIFFE, "Frage?", force=True)
    assert r["ok"] is True and r["versendet"] is False


# --- SSRF-Härtung + § 203-force-Sperre (Code-Review-Fixes) ------------------

def test_webhook_default_deny_ohne_allowlist(monkeypatch):
    monkeypatch.delenv("RISK_LAYER_WEBHOOK_ALLOW", raising=False)
    ok, grund = _webhook_ziel_ok("http://irgendwo.intern:8080/x")
    assert ok is False and "RISK_LAYER_WEBHOOK_ALLOW" in grund


def test_webhook_allowlist_und_schema(monkeypatch):
    monkeypatch.setenv("RISK_LAYER_WEBHOOK_ALLOW", "n8n.kanzlei.local,127.0.0.1:5678")
    assert _webhook_ziel_ok("http://n8n.kanzlei.local/hook")[0] is True
    assert _webhook_ziel_ok("http://127.0.0.1:5678/hook")[0] is True
    # nicht freigegebene Ziele (SSRF-Klassiker Cloud-Metadaten) bleiben gesperrt
    assert _webhook_ziel_ok("http://169.254.169.254/latest/")[0] is False
    assert _webhook_ziel_ok("http://127.0.0.1:9999/hook")[0] is False   # falscher Port
    assert _webhook_ziel_ok("file:///etc/passwd")[0] is False           # nur http/https


def test_export_webhook_ohne_allowlist_verweigert(monkeypatch):
    monkeypatch.delenv("RISK_LAYER_WEBHOOK_ALLOW", raising=False)
    r = exportiere("Die [Gesellschaft A] ist eingegliedert.", BEGRIFFE, "Frage?",
                   webhook_url="http://irgendwo.intern/x")
    assert r["ok"] is False and r["grund"] == "webhook_nicht_erlaubt"
    assert "payload" in r                      # lokaler Download bleibt möglich


def test_export_force_plus_webhook_blockiert_netzversand():
    # § 203: force erlaubt den lokalen Download, aber ein identifizierender
    # Restbefund darf NIE automatisch über den Webhook hinaus – die Sperre greift
    # noch vor der Allowlist-/Netz-Stufe.
    r = exportiere("Die M-GmbH ...", BEGRIFFE, "Frage?", force=True,
                   webhook_url="http://n8n.kanzlei.local/hook")
    assert r["ok"] is False and r["grund"] == "force_kein_webhook"
    assert r["rest"]
