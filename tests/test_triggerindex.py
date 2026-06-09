"""Maschineller Trigger-Index: Build, Lookup, Provenienz, Kuratierung."""

import itertools
import sqlite3

import pytest

from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.triggerindex import TriggerIndex, index_oeffnen


@pytest.fixture()
def index(tmp_path):
    knoten = [
        NormKnoten(id="norm:AO:146", law="AO", para="146",
                   titel="Ordnungsvorschriften für die Führung von Büchern und Aufzeichnungen",
                   text="Kasseneinnahmen und Kasseneinnahmen sind täglich festzuhalten. "
                        "Bareinnahmen, Belege und Aufzeichnungen müssen vollständig sein."),
        NormKnoten(id="norm:AO:162", law="AO", para="162", titel="Schätzung von Besteuerungsgrundlagen",
                   text="Soweit die Finanzbehörde die Besteuerungsgrundlagen nicht ermitteln kann, schätzt sie."),
        NormKnoten(id="norm:HGB:257", law="HGB", para="257", titel="Aufbewahrung von Unterlagen",
                   text="Jeder Kaufmann hat Handelsbücher, Belege und Unterlagen aufzubewahren."),
    ]
    g = NormGraph.build(knoten, tmp_path / "g.sqlite")
    return TriggerIndex.build(g, tmp_path / "trigger.sqlite")


def test_build_stats(index):
    s = index.stats()
    assert s["normen"] == 3 and s["trigger"] > 0 and s["geprueft"] == 0


def test_kassen_belege_treffen_146_vor_257(index):
    res = index.kandidaten("Die Bareinnahmen erscheinen nicht in der Kasse; die Aufzeichnungen fehlen.")
    ids = [c["norm_id"] for c in res]
    assert "norm:AO:146" in ids
    # § 146 (Aufzeichnungen/Kasse/Bareinnahmen) rankt vor § 257 HGB (nur „Belege")
    if "norm:HGB:257" in ids:
        assert ids.index("norm:AO:146") < ids.index("norm:HGB:257")


def test_provenienz_und_ungeprueft(index):
    res = index.kandidaten("Aufzeichnungen und Bareinnahmen")
    c = next(c for c in res if c["norm_id"] == "norm:AO:146")
    assert c["quelle"] == "trigger_index"
    assert c["geprueft"] is False
    assert c["trigger"]            # die auslösenden Stichwörter werden mitgeliefert


def test_kuratierung_setzt_geprueft(index):
    res = index.kandidaten("Aufzeichnungen")
    c = next(c for c in res if c["norm_id"] == "norm:AO:146")
    wort = c["trigger"][0]
    index.bestaetige("norm:AO:146", wort, True)
    res2 = index.kandidaten("Aufzeichnungen", nur_geprueft=True)
    assert any(x["norm_id"] == "norm:AO:146" for x in res2)


# --- Öffnungs-/Lebenszyklus-Robustheit ----------------------------------------

def test_konstruktor_lehnt_fehlenden_pfad_ab(tmp_path):
    # Tippfehler-Pfad darf NICHT still eine leere DB anlegen
    with pytest.raises(FileNotFoundError, match="Trigger-Index"):
        TriggerIndex(tmp_path / "tippfehler.sqlite")
    assert not (tmp_path / "tippfehler.sqlite").exists()


def test_konstruktor_lehnt_fremde_db_ab(tmp_path):
    fremd = tmp_path / "fremd.sqlite"
    con = sqlite3.connect(str(fremd))
    con.execute("CREATE TABLE x (a TEXT)")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="trigger"):
        TriggerIndex(fremd)


def test_index_oeffnen_none_bei_fehlender_datei(tmp_path):
    assert index_oeffnen(tmp_path / "nix.sqlite") is None


def test_build_legt_weiterhin_neue_datei_an(index, tmp_path):
    # der Build-Pfad darf von der strengen Öffnungs-Prüfung nicht betroffen sein
    assert index.path.exists()
    assert index.stats()["normen"] == 3


def test_close_schliesst_verbindung(index):
    index.close()
    with pytest.raises(sqlite3.ProgrammingError):
        index.stats()


# --- Chunking (SQLite-Parameterlimit bei sehr großen Dokumenten) --------------

def test_kandidaten_chunking_grosse_wortmenge(index):
    # > 1000 distinkte Wörter ⇒ mehrere IN-Blöcke; die Treffer-Wörter liegen
    # (alphabetisch) in verschiedenen Blöcken und müssen zusammen aggregieren
    fueller = ["b" + "".join(t) for t in itertools.product("abcdefghij", repeat=3)]
    text = "Aufzeichnungen " + " ".join(fueller) + " Bareinnahmen"
    res = index.kandidaten(text)
    c = next(c for c in res if c["norm_id"] == "norm:AO:146")
    assert "aufzeichnungen" in c["trigger"]


def test_detektiere_chunking_grosse_wortmenge(index):
    fueller = ["c" + "".join(t) for t in itertools.product("abcdefghij", repeat=3)]
    text = "Die Bareinnahmen erscheinen nicht in der Kasse " + " ".join(fueller)
    marken = index.detektiere(text, idf_min=0.5)
    assert any(m["wort"] == "Bareinnahmen" for m in marken)
