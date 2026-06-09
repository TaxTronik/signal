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


# --- Sentinel-Law "?" wird nicht in die Config übernommen --------------------

def test_ableiten_filtert_unaufloesbare_gesetze():
    from risk_layer.catalog import Katalog
    from risk_layer.models import Definitionsquelle, GovernanceTyp, Katalogeintrag

    eintrag = Katalogeintrag(
        id="x", begriff_kanonisch="Testbegriff",
        norm_anker=("§ 5 XYZG", "§ 2 UStG"),     # XYZG ist keiner LawMap bekannt → law='?'
        definitionsquelle=(Definitionsquelle("Gesetz", "§ 2 UStG"),),
        governance_typ=GovernanceTyp.FF,
    )
    gesetze = corpus.ableiten_aus_katalog(Katalog("t", [eintrag]))
    laws = {g.law for g in gesetze}
    assert laws == {"UStG"}                      # kein Phantom-Gesetz '?'
    assert all("/?/" not in g.url for g in gesetze)


# --- defensives lade_config (handeditierte Datei) ----------------------------

def test_lade_config_meldet_kaputten_eintrag(tmp_path):
    import pytest
    pfad = corpus.config_pfad(tmp_path)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text("- law: AO\n", encoding="utf-8")          # slug fehlt
    with pytest.raises(ValueError, match=r"Eintrag 1"):
        corpus.lade_config(tmp_path)
    pfad.write_text("law: AO\n", encoding="utf-8")            # gar keine Liste
    with pytest.raises(ValueError, match=r"Liste"):
        corpus.lade_config(tmp_path)
    pfad.write_text("- law: EStDV\n  slug: estdv_1955\n  art: verordung\n",
                    encoding="utf-8")                          # Tippfehler in art
    with pytest.raises(ValueError, match=r"art"):
        corpus.lade_config(tmp_path)


# --- ZIP-Download: Größen-Cap + deterministische XML-Wahl --------------------

class _FakeResp:
    """Minimale urlopen-Response: liefert ``daten`` häppchenweise."""

    def __init__(self, daten: bytes):
        self.daten = daten
        self.pos = 0

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            amt = len(self.daten) - self.pos
        out = self.daten[self.pos:self.pos + amt]
        self.pos += len(out)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _zip_mit(eintraege: dict[str, str]) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, inhalt in eintraege.items():
            zf.writestr(name, inhalt)
    return buf.getvalue()


def test_hole_gesetz_mehrere_xml_deterministisch(tmp_path, monkeypatch, caplog):
    import logging
    # Archiv-Reihenfolge b vor a → gewählt wird trotzdem die SORTIERT erste
    daten = _zip_mit({"b.xml": "<b/>", "a.xml": "<a/>"})
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=30.0: _FakeResp(daten))
    g = corpus.Gesetz(law="TestG", slug="testg", url="http://x/testg/xml.zip")
    with caplog.at_level(logging.WARNING, logger="risk_layer.corpus"):
        pfad = corpus.hole_gesetz(g, tmp_path)
    assert pfad.read_bytes() == b"<a/>"
    assert any("XML-Dateien" in r.message for r in caplog.records)


def test_hole_gesetz_cap_lehnt_riesendownload_ab(tmp_path, monkeypatch):
    import pytest
    daten = _zip_mit({"a.xml": "<a/>"})
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=30.0: _FakeResp(daten + b"EXTRA"))
    monkeypatch.setattr(corpus, "_MAX_ZIP", len(daten))   # Cap testbar klein
    g = corpus.Gesetz(law="TestG", slug="testg", url="http://x/testg/xml.zip")
    with pytest.raises(ValueError, match=r"größer als"):
        corpus.hole_gesetz(g, tmp_path)
