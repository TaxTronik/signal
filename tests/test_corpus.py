"""Korpus-Ableitung & Slug-Mapping (ohne Netzaufruf)."""

from risk_layer import corpus
from risk_layer.catalog import lade_katalog


def test_ableiten_aus_katalog_liefert_gesetze():
    gesetze = corpus.ableiten_aus_katalog(lade_katalog())
    laws = {g.law for g in gesetze}
    # die im Katalog per norm_anker referenzierten Gesetze
    assert {"UStG", "KStG", "EStG"} <= laws


def test_slug_mapping_jahressuffixe():
    assert corpus.slug_fuer("UStG") == "ustg_1980"
    assert corpus.slug_fuer("KStG") == "kstg_1977"
    assert corpus.slug_fuer("EStG") == "estg"
    assert corpus.slug_fuer("AO") == "ao_1977"
    # EU-/Verfassungsrecht: kein gii-Download
    assert corpus.slug_fuer("MwStSystRL") is None


def test_url_form():
    g = next(g for g in corpus.ableiten_aus_katalog(lade_katalog()) if g.law == "UStG")
    assert g.url == "https://www.gesetze-im-internet.de/ustg_1980/xml.zip"


def test_config_roundtrip(tmp_path):
    gesetze = corpus.ableiten_aus_katalog(lade_katalog())
    corpus.schreibe_config(gesetze, tmp_path)
    geladen = corpus.lade_config(tmp_path)
    assert {g.law for g in geladen} == {g.law for g in gesetze}


def test_hole_gesetz_lehnt_pfad_traversal_ab(tmp_path):
    # g.law (aus editierbarer gesetze.yaml) bildet den Dateinamen -> ein Traversal-
    # law muss VOR jedem Netz-/FS-Zugriff scheitern, nichts darf angelegt werden.
    import pytest
    g = corpus.Gesetz(law="../../evil", slug="x", url="http://invalid.invalid/x.zip")
    with pytest.raises(ValueError):
        corpus.hole_gesetz(g, korpus=tmp_path)
    assert not (tmp_path / "gii").exists()
