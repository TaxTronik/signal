"""Rechercheaufgaben – Delegation als erstklassiges, verankertes Objekt."""

from risk_layer.aufgabe import AufgabenStore


def test_anlegen_normalisiert_und_verankert(tmp_path):
    s = AufgabenStore(tmp_path)
    a = s.speichern({
        "mandant": "beta-gmbh", "fall_id": "beta-gmbh/ap",
        "von": "StB Müller", "an": "Junior",
        "titel": "Betriebsaufspaltung", "frage": "Definition?", "prioritaet": "hoch",
        "anker": {"begriff": "Betriebsaufspaltung", "norm": "§ 15 EStG", "start": "13", "end": "43"},
    })
    assert a["id"].startswith("a-")
    assert a["status"] == "offen"
    assert a["anker"]["start"] == 13 and a["anker"]["begriff"] == "Betriebsaufspaltung"
    assert s.laden(a["id"])["an"] == "Junior"


def test_titel_default_und_status_validierung(tmp_path):
    s = AufgabenStore(tmp_path)
    a = s.speichern({"status": "quatsch"})           # ungültiger Status → offen
    assert a["titel"] == "Rechercheaufgabe" and a["status"] == "offen"


def test_filter_mandant_bearbeiter_aktiv(tmp_path):
    s = AufgabenStore(tmp_path)
    a = s.speichern({"mandant": "beta", "an": "Junior", "titel": "A"})
    s.speichern({"mandant": "alpha", "an": "Senior", "titel": "B"})
    assert {x["titel"] for x in s.liste(mandant="beta")} == {"A"}
    assert {x["titel"] for x in s.liste(an="Senior")} == {"B"}
    assert len(s.liste(nur_aktiv=True)) == 2          # beide offen = aktiv
    s.status_setzen(a["id"], "erledigt")
    assert {x["titel"] for x in s.liste(nur_aktiv=True)} == {"B"}


def test_status_setzen_und_loeschen(tmp_path):
    s = AufgabenStore(tmp_path)
    a = s.speichern({"titel": "X"})
    assert s.status_setzen(a["id"], "in_bearbeitung")["status"] == "in_bearbeitung"
    assert s.status_setzen(a["id"], "ungueltig")["status"] == "in_bearbeitung"   # bleibt
    assert s.loeschen(a["id"]) is True
    assert s.laden(a["id"]) is None


def test_update_bewahrt_erstellt_am(tmp_path):
    s = AufgabenStore(tmp_path)
    a = s.speichern({"titel": "X"})
    b = s.speichern({"id": a["id"], "titel": "X", "ergebnis": "fertig"})
    assert b["erstellt_am"] == a["erstellt_am"]
    assert s.laden(a["id"])["ergebnis"] == "fertig"
