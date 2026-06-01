"""RSS-Parser + Abgleich-Schleife (Erweiterung §5) – offline über Fixtures."""

from pathlib import Path

from risk_layer import abgleich, rss
from risk_layer.catalog import lade_katalog

FIX = Path(__file__).parent / "fixtures"


def _eintraege():
    return (rss.parse_rss((FIX / "feed_bfh.xml").read_bytes(), "BFH")
            + rss.parse_rss((FIX / "feed_bmf.xml").read_bytes(), "BMF"))


def test_rss_parse_liest_items():
    bfh = rss.parse_rss((FIX / "feed_bfh.xml").read_bytes(), "BFH")
    assert len(bfh) == 2
    assert bfh[0].titel == "XI R 29/22"
    assert bfh[0].guid == "STRE111"


def test_index_trackt_katalognormen():
    idx = abgleich.baue_index(lade_katalog())
    assert "norm:KStG:8c" in idx.norm
    assert "norm:UStG:2" in idx.norm
    # Urteil aus definitionsquelle wird als az getrackt
    assert "urteil:BFH:XI-R-29/22" in idx.az


def test_bmf_sanierungsklausel_trifft_schaedlichen_beteiligungserwerb():
    idx = abgleich.baue_index(lade_katalog())
    treffer = abgleich.abgleich(_eintraege(), idx)
    refs = {b["ref"] for t in treffer for b in t.betroffene}
    assert "kst_schaedlicher_beteiligungserwerb" in refs


def test_bfh_trifft_ueber_norm_und_az():
    idx = abgleich.baue_index(lade_katalog())
    treffer = abgleich.abgleich(_eintraege(), idx)
    # XI R 29/22 dockt an § 2 UStG UND an das im Katalog zitierte Az an
    xi = next(t for t in treffer if t.eintrag.titel == "XI R 29/22")
    refs = {b["ref"] for b in xi.betroffene}
    assert "ust_organisatorische_eingliederung" in refs
    assert "urteil:BFH:XI-R-29/22" in xi.az


def test_persistenz_meldet_nur_neue(tmp_path):
    state = tmp_path / "gesehen.json"
    kat = lade_katalog()
    ein = _eintraege()
    erste = abgleich.laufe(kat, eintraege=ein, state_pfad=state, fetch=False)
    assert erste                      # beim ersten Lauf Treffer
    zweite = abgleich.laufe(kat, eintraege=ein, state_pfad=state, fetch=False)
    assert zweite == []               # alles bereits gesehen
