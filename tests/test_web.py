"""Web-Schicht: API-Aufbereitung & Define-on-the-fly (ohne Sockets testbar)."""

import pytest

from risk_layer.berater import BeraterFehler
from risk_layer.web import Anwendung

# Hinweis: die frühere autouse-Fixture ``_kein_embedding`` (deaktiviert den
# Embedding-Index, damit kein Test das schwere BGE-M3-Modell lädt) liegt jetzt
# projektweit in tests/conftest.py.


def _text():
    return open("examples/beispiel_organschaft.txt", encoding="utf-8").read()


def test_analyse_json_vertrag(tmp_path):
    app = Anwendung(None, str(tmp_path))
    res = app.analyse(_text(), unknown=False, berater=False, nutzer=None)
    for schluessel in ("text", "summary", "spans", "karten", "risiken", "katalog_version"):
        assert schluessel in res
    # Spans referenzieren auf Karten/Risiken
    for s in res["spans"]:
        assert s["ref"]["type"] in ("karte", "risiko")
        assert 0 <= s["start"] < s["end"] <= len(res["text"])
    # ohne unknown/berater stammen Risiken NUR aus deterministischen Zusatz-Ebenen
    # (Trigger/Streit) – nie aus dem LLM (schicht2 ist hier aus).
    for r in res["risiken"]:
        assert r["quelle"] in ("trigger", "streit")


def test_spans_tragen_layer_kennzeichnung(tmp_path):
    app = Anwendung(None, str(tmp_path))
    res = app.analyse(_text(), unknown=False, berater=False, nutzer=None)
    layer_werte = {s["layer"] for s in res["spans"]}
    assert layer_werte <= {"schicht1", "muster", "trigger", "llm", "heuristik",
                           "berater", "streit", "manual"}


def test_karte_json_traegt_kanonische_ids(tmp_path):
    app = Anwendung(None, str(tmp_path))
    res = app.analyse(_text(), unknown=False, berater=False, nutzer=None)
    org = next(k for k in res["karten"] if k["begriff_id"] == "ust_organisatorische_eingliederung")
    assert org["norm_anker"][0]["ids"] == ["norm:UStG:2:abs2:nr2"]


def test_define_on_the_fly_zyklus(tmp_path):
    app = Anwendung(None, str(tmp_path))
    # vorher: Betriebsaufspaltung ist unknown_risiko
    vor = app.analyse(_text(), unknown=True, berater=True, nutzer="rey")
    assert any(r["status"] == "unknown_risiko" and "Betriebsaufspaltung" in r["titel"]
               for r in vor["risiken"])
    # definieren
    app.definiere({"begriff": "Betriebsaufspaltung",
                   "definition_text": "Aufteilung in Besitz- und Betriebsunternehmen.",
                   "autor": "rey", "scope": "personal", "norm_anker": ["§ 15 EStG"]})
    # nachher: dieselbe Stelle ist berater_definiert
    nach = app.analyse(_text(), unknown=True, berater=True, nutzer="rey")
    assert any(r["status"] == "berater_definiert" and "Betriebsaufspaltung" in r["titel"]
               for r in nach["risiken"])
    assert not any(r["status"] == "unknown_risiko" and "Betriebsaufspaltung" in r["titel"]
                   for r in nach["risiken"])


def test_definiere_norm_anker_string_wird_nicht_zersplittert(tmp_path):
    """Ein String statt Liste ('§ 15 EStG') darf NICHT zeichenweise zersplittert
    in die Festwissens-YAML geschrieben werden – klarer Fehler statt Datenmüll."""
    app = Anwendung(None, str(tmp_path))
    with pytest.raises(BeraterFehler):
        app.definiere({"begriff": "Betriebsaufspaltung",
                       "definition_text": "Aufteilung in Besitz- und Betriebsunternehmen.",
                       "autor": "rey", "scope": "personal",
                       "norm_anker": "§ 15 EStG"})
    # nichts wurde persistiert
    assert list(tmp_path.rglob("*.yaml")) == []
