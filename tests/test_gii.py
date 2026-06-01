"""gii-Parser (Parser-Spez. Kap. 4) – auf einer committeten Mini-Fixture."""

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
