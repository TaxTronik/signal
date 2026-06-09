"""gii-Parser (Parser-Spez. Kap. 4) – auf einer committeten Mini-Fixture."""

import logging
from pathlib import Path

from risk_layer import gii

FIXTURE = Path(__file__).parent / "fixtures" / "mini_gii.xml"


def test_knoten_ids_und_granularitaet():
    knoten = {k.id: k for k in gii.parse_datei(FIXTURE)}
    # §, Absatz und Nummer werden als eigene Knoten erzeugt
    for kid in ["norm:TestG:2", "norm:TestG:2:abs1", "norm:TestG:2:abs2",
                "norm:TestG:2:abs2:nr1", "norm:TestG:2:abs2:nr2", "norm:TestG:3"]:
        assert kid in knoten, kid


def test_law_aus_amtabk_ohne_jahressuffix():
    knoten = gii.parse_datei(FIXTURE)
    assert all(k.law == "TestG" for k in knoten)  # nicht „TestG 1990"


def test_metadaten_und_text():
    knoten = {k.id: k for k in gii.parse_datei(FIXTURE)}
    n = knoten["norm:TestG:2:abs2:nr2"]
    assert n.titel == "Beispielnorm"
    assert n.gueltig_ab == "1990-01-01"
    assert "eingegliedert" in n.text


def test_rahmennorm_uebersprungen():
    # Die erste <norm> (nur Gliederung/Inhaltsübersicht) erzeugt keinen §-Knoten
    ids = {k.id for k in gii.parse_datei(FIXTURE)}
    assert not any(":Inhalt" in i for i in ids)


# --- verschachtelte Listen (Buchst. unter Nummern, realer Fall § 4 UStG) ------

_NESTED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<dokumente>
  <norm>
    <metadaten>
      <amtabk>TestG</amtabk>
      <enbez>§ 4</enbez>
      <titel>Verschachtelte Norm</titel>
    </metadaten>
    <textdaten>
      <text format="XML">
        <Content>
          <P>(1) Steuerfrei sind:
            <DL Type="arabic">
              <DT>1.</DT>
              <DD><LA>erste Nummer mit Buchstaben:
                <DL Type="alpha">
                  <DT>a)</DT><DD><LA>Buchstabe a unter Nummer 1.</LA></DD>
                  <DT>b)</DT><DD><LA>Buchstabe b unter Nummer 1.</LA></DD>
                </DL>
              </LA></DD>
              <DT>2.</DT>
              <DD><LA>zweite Nummer mit Buchstaben:
                <DL Type="alpha">
                  <DT>a)</DT><DD><LA>Buchstabe a unter Nummer 2.</LA></DD>
                </DL>
              </LA></DD>
            </DL>
          </P>
        </Content>
      </text>
    </textdaten>
  </norm>
</dokumente>
"""


def test_verschachtelte_dl_bekommt_nr_kontext(tmp_path):
    pfad = tmp_path / "nested.xml"
    pfad.write_text(_NESTED_XML, encoding="utf-8")
    knoten = {k.id: k for k in gii.parse_datei(pfad)}
    # Buchst. a unter Nr. 1 und Nr. 2 kollidieren NICHT mehr – distinkte IDs
    assert "norm:TestG:4:abs1:nr1:lita" in knoten
    assert "norm:TestG:4:abs1:nr1:litb" in knoten
    assert "norm:TestG:4:abs1:nr2:lita" in knoten
    assert "Nummer 1" in knoten["norm:TestG:4:abs1:nr1:lita"].text
    assert "Nummer 2" in knoten["norm:TestG:4:abs1:nr2:lita"].text
    # nr-Kontext steht auch im Metadatum
    assert knoten["norm:TestG:4:abs1:nr2:lita"].nummer == "2"
    # kein kontextloser (kollidierender) Sammel-Knoten mehr
    assert "norm:TestG:4:abs1:lita" not in knoten


def test_dt_ohne_dd_verschiebt_paarung_nicht(tmp_path):
    # zip(dts, dds) hätte bei fehlendem DD alle Paarungen still verschoben –
    # Nr. 1 bekäme den Text von Nr. 2. Die Kind-Reihenfolge paart robust.
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<dokumente>
  <norm>
    <metadaten><amtabk>TestG</amtabk><enbez>§ 5</enbez><titel>Lückenliste</titel></metadaten>
    <textdaten><text format="XML"><Content>
      <P>(1) Liste:
        <DL Type="arabic">
          <DT>1.</DT>
          <DT>2.</DT><DD><LA>zweite Nummer.</LA></DD>
        </DL>
      </P>
    </Content></text></textdaten>
  </norm>
</dokumente>
"""
    pfad = tmp_path / "lueckig.xml"
    pfad.write_text(xml, encoding="utf-8")
    knoten = {k.id: k for k in gii.parse_datei(pfad)}
    assert "norm:TestG:5:abs1:nr2" in knoten
    assert "zweite Nummer" in knoten["norm:TestG:5:abs1:nr2"].text
    # die DT-Marke ohne DD fällt einzeln weg statt die Texte zu verschieben
    assert "norm:TestG:5:abs1:nr1" not in knoten


def test_paragraphen_bereich_wird_geloggt(tmp_path, caplog):
    # „§§ x bis y"-enbez wird weiterhin übersprungen, aber sichtbar geloggt
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<dokumente>
  <norm>
    <metadaten><amtabk>TestG</amtabk><enbez>§§ 4 bis 6</enbez><titel>Bereich</titel></metadaten>
  </norm>
</dokumente>
"""
    pfad = tmp_path / "bereich.xml"
    pfad.write_text(xml, encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="risk_layer.gii"):
        knoten = gii.parse_datei(pfad)
    assert knoten == []
    assert any("ohne Einzel-§" in r.getMessage() for r in caplog.records)


def test_parse_verzeichnis_ueberspringt_korrupte_datei(tmp_path, caplog):
    (tmp_path / "a_kaputt.xml").write_text("<dokumente><norm>", encoding="utf-8")
    (tmp_path / "b_ok.xml").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="risk_layer.gii"):
        knoten = gii.parse_verzeichnis(tmp_path)
    # die intakte Datei wurde trotz der korrupten verarbeitet
    assert any(k.id == "norm:TestG:2" for k in knoten)
    assert any("übersprungen" in r.getMessage() for r in caplog.records)
