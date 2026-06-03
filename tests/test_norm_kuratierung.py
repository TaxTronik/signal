"""Katalog-weite Norm-Kuratierung (Berater-Aufsatz): verwerfen/ergaenzen/zuruecksetzen,
Scope-Filter, Idempotenz, Persistenz + die quelle/verworfen-Provenienz im Karte-JSON.

Fallzustandslos + ohne Engine-Audit (die Handlung auditiert der Host) — hier nur der
Katalog-State und seine Auswirkung auf die Norm-Vorschläge.
"""

import pytest

from risk_layer.berater import BeraterFehler, BeraterStore
from risk_layer.web import _norm_anker_json


def test_verwerfen_persistiert_und_zuruecksetzen(tmp_path):
    s = BeraterStore(tmp_path)
    s.kuratiere_norm("ao_schaetzung", "§ 162 AO", "verwerfen",
                     scope="geteilt", autor="a", datum="2026-06-04")
    assert "norm:AO:162" in s.kuratierung_fuer("ao_schaetzung")["verworfen"]
    # persistiert (frischer Store liest die YAML)
    assert "norm:AO:162" in BeraterStore(tmp_path).kuratierung_fuer("ao_schaetzung")["verworfen"]
    # zuruecksetzen = Undo
    s.kuratiere_norm("ao_schaetzung", "§ 162 AO", "zuruecksetzen",
                     scope="geteilt", autor="a", datum="2026-06-04")
    assert "norm:AO:162" not in s.kuratierung_fuer("ao_schaetzung")["verworfen"]


def test_ergaenzen(tmp_path):
    s = BeraterStore(tmp_path)
    s.kuratiere_norm("ao_schaetzung", "§ 90 AO", "ergaenzen",
                     scope="geteilt", autor="a", datum="d")
    erg = s.kuratierung_fuer("ao_schaetzung")["ergaenzt"]
    assert erg and erg[0]["id"] == "norm:AO:90"


def test_scope_personal_nur_eigener_autor_geteilt_fuer_alle(tmp_path):
    s = BeraterStore(tmp_path)
    s.kuratiere_norm("ao_schaetzung", "§ 162 AO", "verwerfen",
                     scope="personal", autor="alice", datum="d")
    assert "norm:AO:162" in s.kuratierung_fuer("ao_schaetzung", nutzer="alice")["verworfen"]
    assert "norm:AO:162" not in s.kuratierung_fuer("ao_schaetzung", nutzer="bob")["verworfen"]
    s.kuratiere_norm("ao_schaetzung", "§ 158 AO", "verwerfen",
                     scope="geteilt", autor="alice", datum="d")
    assert "norm:AO:158" in s.kuratierung_fuer("ao_schaetzung", nutzer="bob")["verworfen"]


def test_idempotent_und_validierung(tmp_path):
    s = BeraterStore(tmp_path)
    s.kuratiere_norm("x", "§ 162 AO", "verwerfen", scope="geteilt", autor="a", datum="d")
    s.kuratiere_norm("x", "§ 162 AO", "verwerfen", scope="geteilt", autor="a", datum="d")
    assert sum(1 for k in s._kuratierungen if k.katalog_id == "x") == 1   # kein Duplikat
    with pytest.raises(BeraterFehler):
        s.kuratiere_norm("x", "§ 162 AO", "quatsch", scope="geteilt", autor="a", datum="d")
    with pytest.raises(BeraterFehler):
        s.kuratiere_norm("x", "kein zitat", "verwerfen", scope="geteilt", autor="a", datum="d")
    with pytest.raises(BeraterFehler):
        s.kuratiere_norm("x", "§ 162 AO", "verwerfen", scope="kanzlei", autor="a", datum="d")


def test_norm_anker_json_traegt_quelle_und_verworfen():
    kur = {"verworfen": {"norm:AO:162"}, "ergaenzt": [{"zitat": "§ 90 AO", "id": "norm:AO:90"}]}
    out = {o["zitat"]: o for o in _norm_anker_json(["§ 162 AO", "§ 158 AO"], kur)}
    assert out["§ 162 AO"]["quelle"] == "katalog" and out["§ 162 AO"]["verworfen"] is True
    assert out["§ 158 AO"]["verworfen"] is False
    assert out["§ 90 AO"]["quelle"] == "berater" and out["§ 90 AO"]["verworfen"] is False
    # ohne Kuratierung rückwärtskompatibel: quelle katalog, verworfen False
    plain = _norm_anker_json(["§ 162 AO"])
    assert plain[0]["quelle"] == "katalog" and plain[0]["verworfen"] is False
