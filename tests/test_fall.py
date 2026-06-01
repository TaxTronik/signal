"""Fall-Datenbank: speichern, laden, liste, löschen."""

from risk_layer.fall import FallStore


def test_speichern_und_laden(tmp_path):
    store = FallStore(tmp_path)
    f = store.speichern({
        "name": "Müller Organschaft 2026",
        "text": "Die M-GmbH ...",
        "optionen": {"unknown": True, "schicht2": True},
        "markierungen": [{"start": 4, "end": 10, "label": "prüfen", "farbe": "#f00", "notiz": "x"}],
    })
    # ohne Mandant landet der Fall im virtuellen Mandanten "allgemein"
    assert f["id"] == "allgemein/mueller-organschaft-2026"
    assert f["mandant"] == "allgemein"
    assert f["erstellt_am"] and f["geaendert_am"]
    geladen = store.laden(f["id"])
    assert geladen["text"] == "Die M-GmbH ..."
    assert geladen["markierungen"][0]["label"] == "prüfen"
    assert geladen["optionen"]["schicht2"] is True


def test_mandantenakte_gruppiert_faelle(tmp_path):
    store = FallStore(tmp_path)
    store.mandant_speichern("beta-gmbh", "Beta GmbH", "M-1024")
    store.speichern({"mandant": "beta-gmbh", "name": "Außenprüfung 2024", "text": "..."})
    store.speichern({"mandant": "beta-gmbh", "name": "USt-Sonderprüfung", "text": "..."})
    store.speichern({"mandant": "alpha-ag", "name": "Organschaft", "text": "..."})
    regal = {m["slug"]: m for m in store.mandanten()}
    assert regal["beta-gmbh"]["faelle"] == 2
    assert regal["beta-gmbh"]["name"] == "Beta GmbH"
    assert regal["beta-gmbh"]["mandantennummer"] == "M-1024"
    assert {f["name"] for f in store.liste("beta-gmbh")} == {"Außenprüfung 2024", "USt-Sonderprüfung"}


def test_risiko_governance_persistiert(tmp_path):
    store = FallStore(tmp_path)
    f = store.speichern({"mandant": "beta", "name": "F", "text": "abcdef",
        "markierungen": [{"start": 0, "end": 3, "label": "vGA?",
            "norm": {"id": "norm:KStG:8", "zitat": "§ 8 Abs. 3 KStG", "titel": "vGA"},
            "risiko": {"governance_typ": "IN", "schadensintensitaet": "hoch",
                       "wahrscheinlichkeit": "möglich", "kaskadenreichweite": "3",
                       "kontrolle": "Fremdvergleich", "verantwortlich": "StB", "status": "in_pruefung"}}]})
    r = store.laden(f["id"])["markierungen"][0]
    assert r["norm"]["zitat"] == "§ 8 Abs. 3 KStG"
    assert r["risiko"]["governance_typ"] == "IN"
    assert r["risiko"]["kaskadenreichweite"] == 3          # zu int normalisiert
    assert r["risiko"]["status"] == "in_pruefung"


def test_markierung_ohne_risiko_bleibt_schlank(tmp_path):
    store = FallStore(tmp_path)
    f = store.speichern({"name": "X", "text": "abc",
                         "markierungen": [{"start": 0, "end": 1, "label": "nur Notiz"}]})
    m = store.laden(f["id"])["markierungen"][0]
    assert m["risiko"] is None and m["norm"] is None


def test_update_behaelt_erstellt_am(tmp_path):
    store = FallStore(tmp_path)
    a = store.speichern({"name": "Fall A", "text": "v1"})
    b = store.speichern({"id": a["id"], "name": "Fall A", "text": "v2"})
    assert b["erstellt_am"] == a["erstellt_am"]
    assert store.laden(a["id"])["text"] == "v2"


def test_liste_und_loeschen(tmp_path):
    store = FallStore(tmp_path)
    store.speichern({"name": "Fall A", "text": "..."})
    store.speichern({"name": "Fall B", "text": "..."})
    assert {e["name"] for e in store.liste()} == {"Fall A", "Fall B"}
    fid = store.liste()[0]["id"]
    assert store.loeschen(fid) is True
    assert store.laden(fid) is None
    assert len(store.liste()) == 1


def test_markierungen_werden_normalisiert(tmp_path):
    store = FallStore(tmp_path)
    f = store.speichern({"name": "X", "text": "abc", "markierungen": [{"start": "1", "end": "2"}]})
    m = f["markierungen"][0]
    assert m["start"] == 1 and m["end"] == 2 and m["farbe"]   # Default-Farbe gesetzt


def test_farbe_wird_auf_hex_validiert_gegen_xss(tmp_path):
    """Eine gespeicherte Farbe landet im style-Attribut des Overlays – nur Hex
    darf durch, sonst bräche `red" onmouseover=…` aus (XSS)."""
    store = FallStore(tmp_path)
    f = store.speichern({"name": "X", "text": "abcdef", "markierungen": [
        {"start": 0, "end": 3, "farbe": 'red" onmouseover="alert(1)'},
        {"start": 0, "end": 3, "farbe": "#abc"},
    ]})
    assert f["markierungen"][0]["farbe"] == "#fde68a"   # bösartig → Default
    assert f["markierungen"][1]["farbe"] == "#abc"      # gültiges Hex bleibt


def test_span_offsets_werden_geclamped(tmp_path):
    store = FallStore(tmp_path)
    f = store.speichern({"name": "X", "text": "abc", "markierungen": [
        {"start": -5, "end": 999, "label": "out of range"},
        {"start": 2, "end": 1, "label": "rückwärts"},
    ]})
    m0, m1 = f["markierungen"]
    assert (m0["start"], m0["end"]) == (0, 3)   # auf [0, len(text)] geklammert
    assert m1["start"] <= m1["end"]             # start <= end erzwungen
