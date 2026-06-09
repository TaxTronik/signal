"""Defensive Struktur-Validierung des Katalog-Loaders.

Kaputte YAML-Pflege darf nicht als roher TypeError/KeyError oder – schlimmer –
still falsch (zeichenweise zerlegte ausloeser-Gruppen) durchrutschen, sondern
muss als KatalogFehler mit Begriffs-Kontext scheitern (/NF50/).
"""

import pytest

from risk_layer.catalog import Katalog, KatalogFehler, lade_katalog
from risk_layer.models import Katalogeintrag


def _schreibe(tmp_path, body: str):
    p = tmp_path / "begriffe.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_ausloeser_gruppen_als_strings_wird_abgelehnt(tmp_path):
    # Strings statt Listen würden zeichenweise zerlegt → Muster matcht jeden Satz
    p = _schreibe(tmp_path, """
version: "1"
begriffe:
  - id: kst_organschaft
    begriff_kanonisch: Organschaft
    ausloeser:
      - gruppen: ["Organschaft", "Eingliederung"]
""")
    with pytest.raises(KatalogFehler) as exc:
        lade_katalog(p)
    assert "kst_organschaft" in str(exc.value)


def test_ausloeser_listen_von_listen_bleiben_gueltig(tmp_path):
    p = _schreibe(tmp_path, """
version: "1"
begriffe:
  - id: kst_organschaft
    begriff_kanonisch: Organschaft
    ausloeser:
      - gruppen: [["Organschaft"], ["Eingliederung", "eingegliedert"]]
""")
    e = lade_katalog(p).get("kst_organschaft")
    assert e.ausloeser[0]["gruppen"] == [["Organschaft"], ["Eingliederung", "eingegliedert"]]


def test_begriffe_null_wird_abgelehnt(tmp_path):
    # `begriffe:` ohne Wert → None: KatalogFehler statt TypeError beim Iterieren
    p = _schreibe(tmp_path, "version: '1'\nbegriffe:\n")
    with pytest.raises(KatalogFehler):
        lade_katalog(p)


def test_string_eintrag_wird_abgelehnt(tmp_path):
    # String-Eintrag: '"id" not in raw' wäre nur ein Substring-Check
    p = _schreibe(tmp_path, "version: '1'\nbegriffe:\n  - nur eine id im text\n")
    with pytest.raises(KatalogFehler):
        lade_katalog(p)


def test_gueltig_ab_bis_datum_wird_string(tmp_path):
    # unquotiertes YAML-Datum käme sonst als datetime.date statt Optional[str]
    p = _schreibe(tmp_path, """
version: "1"
begriffe:
  - id: x
    begriff_kanonisch: X
    gueltig_ab: 2024-01-01
    gueltig_bis: 2026-12-31
""")
    e = lade_katalog(p).get("x")
    assert e.gueltig_ab == "2024-01-01" and isinstance(e.gueltig_ab, str)
    assert e.gueltig_bis == "2026-12-31" and isinstance(e.gueltig_bis, str)


def test_katalog_init_prueft_id_duplikate():
    # auch bei DIREKTER Konstruktion (nicht nur lade_katalog) dürfen Duplikate
    # nicht still in _by_id verschwinden
    e1 = Katalogeintrag(id="x", begriff_kanonisch="X")
    e2 = Katalogeintrag(id="x", begriff_kanonisch="X-Variante")
    with pytest.raises(KatalogFehler):
        Katalog(version="t", eintraege=[e1, e2])
