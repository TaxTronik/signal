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


def test_analyse_antwort_traegt_engine_status(tmp_path):
    """Jede Analyse-Antwort trägt den Degradations-Nachweis: Startup-Status
    + effektive Lauf-Optionen (Haftungsakte, INTEGRATION_TAXTRONIK.md §4)."""
    app = Anwendung(None, str(tmp_path))
    res = app.analyse(_text(), unknown=True, berater=True, nutzer="rey")
    st = res["engine_status"]
    assert st["katalog_version"] == res["katalog_version"]
    # conftest patcht embedding.index_oeffnen→None: Schicht 1.7 ist hier
    # nachweislich degradiert UND begründet (kein stilles Fehlen im Status).
    emb = st["schichten"]["embedding"]
    assert emb["aktiv"] is False
    assert emb["grund"] == "kein Embedding-Index gebaut"
    assert emb["index_aktuell"] is None and emb["index_fingerprint"] is None
    # effektive Lauf-Optionen genau DIESES Aufrufs
    assert st["optionen"] == {"unknown": True, "berater": True,
                              "abweichung": False, "mit_llm": False}


def test_engine_status_liefert_unveraenderliche_kopie(tmp_path):
    """engine_status() gibt eine Kopie heraus – der beim Start eingefrorene
    Status darf von Aufrufern (api_v1 mischt optionen dazu) nie mutierbar sein."""
    app = Anwendung(None, str(tmp_path))
    st = app.engine_status()
    st["schichten"]["ner"]["verfuegbar"] = "manipuliert"
    st["katalog_version"] = "manipuliert"
    frisch = app.engine_status()
    assert frisch["schichten"]["ner"]["verfuegbar"] != "manipuliert"
    assert frisch["katalog_version"] != "manipuliert"


def test_engine_status_ner_aus_dem_modul_cache(tmp_path, monkeypatch):
    """Der NER-Status kommt aus anonymize.ner_status(): ein bereits geladenes
    Modell (Modul-Cache) wird mit dem TATSÄCHLICHEN Modellnamen gemeldet –
    ohne dass die Anwendung das Modell selbst lädt."""
    import risk_layer.anonymize as anon

    class _Nlp:
        meta = {"lang": "de", "name": "core_news_sm"}
    monkeypatch.setattr(anon, "_NLP", _Nlp())
    app = Anwendung(None, str(tmp_path))
    ner = app.engine_status()["schichten"]["ner"]
    assert ner == {"verfuegbar": True, "modell": "de_core_news_sm", "grund": None}


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
