"""Katalog-weite Norm-Kuratierung (Berater-Aufsatz): verwerfen/ergaenzen/zuruecksetzen,
Scope-Filter, Idempotenz, Persistenz + die Bulk-Lesesicht fürs Host-Overlay.

Fallzustandslos + ohne Engine-Audit (die Handlung auditiert der Host) — hier nur der
Katalog-State. /v1/analyse bleibt katalog-rein; der Host überlagert die Kuratierung
selbst (GET /v1/katalog/kuratierung), darum trägt das Analyse-JSON KEINE quelle/verworfen.
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


def test_norm_anker_json_bleibt_pur():
    # /v1/analyse ist katalog-rein: nur {zitat, ids}, KEINE Kuratierungs-Felder
    # (verworfen/quelle). Sonst wäre die Analyse abhängig vom Berater-State.
    out = _norm_anker_json(["§ 162 AO"])
    assert out == [{"zitat": "§ 162 AO", "ids": ["norm:AO:162"]}]
    assert "verworfen" not in out[0] and "quelle" not in out[0]


def test_kuratierungen_bulk_scope_filter(tmp_path):
    s = BeraterStore(tmp_path)
    s.kuratiere_norm("x", "§ 162 AO", "verwerfen", scope="geteilt", autor="alice", datum="d")
    s.kuratiere_norm("x", "§ 90 AO", "ergaenzen", scope="personal", autor="alice", datum="d")
    s.kuratiere_norm("y", "§ 158 AO", "verwerfen", scope="personal", autor="bob", datum="d")
    # bob sieht das geteilte (x/162) + sein eigenes (y/158), NICHT alices personal (x/90)
    bob = {(k.katalog_id, k.norm_id) for k in s.kuratierungen(nutzer="bob")}
    assert bob == {("x", "norm:AO:162"), ("y", "norm:AO:158")}
    # auf einen Begriff gefiltert
    nur_x = {k.norm_id for k in s.kuratierungen(katalog_id="x", nutzer="alice")}
    assert nur_x == {"norm:AO:162", "norm:AO:90"}
