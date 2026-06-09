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


def test_rss_lehnt_dtd_entity_ab():
    # Entity-Expansion-Schutz (billion laughs): Feed mit DTD/<!ENTITY> wird
    # abgelehnt statt von expat expandiert.
    import pytest
    bombe = (b'<?xml version="1.0"?>\n'
             b'<!DOCTYPE rss [<!ENTITY a "AAAAAAAA">]>\n'
             b'<rss><channel><item><title>&a;</title></item></channel></rss>')
    with pytest.raises(ValueError):
        rss.parse_rss(bombe, "BFH")


def test_text_doppelt_escaptes_html_ohne_literale_tags():
    # "&amp;lt;p&amp;gt;" → (XML-Parser) "&lt;p&gt;" → (unescape) wäre "<p>"
    # als literaler Tag im Klartext gelandet; der zweite Strip räumt das ab.
    feed = (b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>T</title>'
            b'<description>&amp;lt;p&amp;gt;Hallo&amp;lt;/p&amp;gt;</description>'
            b'<guid>g1</guid></item></channel></rss>')
    eintrag = rss.parse_rss(feed, "BFH")[0]
    assert "<" not in eintrag.beschreibung and ">" not in eintrag.beschreibung
    assert "Hallo" in eintrag.beschreibung


def test_hole_alle_mit_fehlern_meldet_ausgefallenen_feed(monkeypatch, caplog):
    # Fehlerkanal wie corpus.hole_alle: ein toter Feed darf nicht mehr wie
    # „keine neuen Entscheidungen" aussehen.
    import logging

    def fake_hole_feed(url, quelle, *, timeout=25.0):
        if quelle == "BFH":
            raise ValueError("Feed kaputt")
        return [rss.RssEintrag(quelle, "Titel", "link", "guid-1", "datum", "Text")]

    monkeypatch.setattr(rss, "hole_feed", fake_hole_feed)
    with caplog.at_level(logging.WARNING, logger="risk_layer.rss"):
        erg = rss.hole_alle_mit_fehlern()
    assert list(erg.fehler) == ["BFH"]
    assert "ValueError" in erg.fehler["BFH"]
    assert [e.quelle for e in erg.eintraege] == ["BMF"]
    assert any("BFH" in r.message for r in caplog.records)
    # Abwärtskompatible Fassade liefert weiterhin die nackte Liste (abgleich.laufe)
    assert [e.quelle for e in rss.hole_alle()] == ["BMF"]


def test_hole_feed_lehnt_zu_grossen_feed_explizit_ab(monkeypatch):
    # Stilles Kappen bei _MAX_FEED endete als kryptischer ParseError (stiller
    # Ausfall) – jetzt gibt es einen sprechenden Fehler für den Fehlerkanal.
    import pytest

    class _FakeResp:
        def __init__(self, daten):
            self.daten = daten
            self.pos = 0

        def read(self, amt=None):
            if amt is None:
                amt = len(self.daten) - self.pos
            out = self.daten[self.pos:self.pos + amt]
            self.pos += len(out)
            return out

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    daten = b"<rss>" + b"x" * 100
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=25.0: _FakeResp(daten))
    monkeypatch.setattr(rss, "_MAX_FEED", 50)     # Cap testbar klein
    with pytest.raises(ValueError, match=r"größer als"):
        rss.hole_feed("http://x/feed.rss", "BFH")
