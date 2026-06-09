"""Robustheit der Abgleich-Persistenz: korrupte State-Datei blockiert nicht.

Ergänzt test_rss_abgleich.py (Fachlogik) um die Fehlerpfade von
``lade_gesehen``/``laufe``: eine kaputte gesehen.json (abgebrochener Schreib-
vorgang, Handbearbeitung) darf weitere Läufe nicht unmöglich machen.
"""

import json
import logging

from risk_layer import abgleich
from risk_layer.catalog import lade_katalog


def test_lade_gesehen_fehlende_datei(tmp_path):
    assert abgleich.lade_gesehen(tmp_path / "nix.json") == set()


def test_lade_gesehen_liest_gueltige_datei(tmp_path):
    p = tmp_path / "gesehen.json"
    p.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert abgleich.lade_gesehen(p) == {"a", "b"}


def test_lade_gesehen_korrupte_datei_blockiert_nicht(tmp_path, caplog):
    p = tmp_path / "gesehen.json"
    p.write_text("{kaputt", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="risk_layer.abgleich"):
        assert abgleich.lade_gesehen(p) == set()
    # Original wird zur Diagnose gesichert, der State-Pfad ist wieder frei
    assert not p.exists()
    assert (tmp_path / "gesehen.json.corrupt").read_text(encoding="utf-8") == "{kaputt"
    assert any("Korrupte State-Datei" in r.getMessage() for r in caplog.records)


def test_lade_gesehen_falscher_typ_gilt_als_korrupt(tmp_path):
    # gültiges JSON, aber keine Liste (z. B. Objekt nach Handbearbeitung)
    p = tmp_path / "gesehen.json"
    p.write_text(json.dumps({"keine": "liste"}), encoding="utf-8")
    assert abgleich.lade_gesehen(p) == set()
    assert (tmp_path / "gesehen.json.corrupt").exists()


def test_laufe_nach_korrupter_datei_wieder_moeglich(tmp_path):
    state = tmp_path / "gesehen.json"
    state.write_text("###", encoding="utf-8")
    treffer = abgleich.laufe(lade_katalog(), eintraege=[], state_pfad=state, fetch=False)
    assert treffer == []
    # Status wurde frisch geschrieben und ist wieder lesbar
    assert json.loads(state.read_text(encoding="utf-8")) == []
    assert abgleich.lade_gesehen(state) == set()
