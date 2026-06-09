"""NormID-Validierung (__post_init__): die ID ist der universelle Join über alle
Korpora – widersprüchliche oder degenerierte IDs ("norm::15", para+artikel
gleichzeitig, unbekannte Tiefenlabels) müssen am Konstruktor scheitern."""

import pytest

from risk_layer.idschema import NormID


def test_para_und_artikel_schliessen_sich_aus():
    # to_id() würde para still ignorieren – kein stiller Informationsverlust
    with pytest.raises(ValueError):
        NormID(law="EStG", para="15", artikel="11")


def test_law_darf_nicht_leer_sein():
    # law="" ergäbe die degenerierte ID "norm::15"
    with pytest.raises(ValueError):
        NormID(law="", para="15")


def test_tiefe_labels_gegen_tiefe_labels_validiert():
    with pytest.raises(ValueError):
        NormID(law="AO", para="162", tiefe=(("absatz", "1"),))


def test_gueltige_normids_unveraendert():
    nid = NormID(law="AO", para="162", tiefe=(("abs", "1"), ("nr", "2")))
    assert nid.to_id() == "norm:AO:162:abs1:nr2"
    assert NormID(law="GG", artikel="11").to_id() == "norm:GG:art11"
    # unbekanntes Gesetz wird vom Parser als law="?" geflaggt – bleibt erlaubt
    assert NormID(law="?", para="5").to_id() == "norm:?:5"
