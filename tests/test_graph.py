"""Normgraph: Bau, Kanten, Dangling, Auflösung, Graph-Anreicherung (/LF170/, /T50/)."""

from pathlib import Path

import pytest

from risk_layer import gii
from risk_layer.catalog import Katalog
from risk_layer.graph import NormGraph
from risk_layer.models import (
    Begriffskarte, Definitionsquelle, GovernanceTyp, Katalogeintrag, Status, Treffer,
)
from risk_layer.resolver import anreichern_mit_graph

FIXTURE = Path(__file__).parent / "fixtures" / "mini_gii.xml"


@pytest.fixture()
def graph(tmp_path):
    knoten = gii.parse_datei(FIXTURE)
    return NormGraph.build(knoten, tmp_path / "g.sqlite")


def test_intra_law_kante(graph):
    # „§ 2 Abs. 1" im Text von nr2 → Kante auf vorhandenen Knoten
    r = graph.aufloesen("norm:TestG:2:abs2:nr2")
    assert r["gefunden"]
    assert "norm:TestG:2:abs1" in r["verweist_auf"]


def test_dangling_cross_law(graph):
    # „§ 5 EStG" verweist auf einen nicht im Korpus vorhandenen Knoten (/T50/)
    ziele = {d["dst"] for d in graph.dangling()}
    assert "norm:EStG:5" in ziele


def test_aufloesen_fallback_auf_vorfahr(graph):
    # nicht existente Satz-ID fällt auf den vorhandenen Absatz zurück
    r = graph.aufloesen("norm:TestG:2:abs1:s1")
    assert r["gefunden"]
    assert r["verwendet"] == "norm:TestG:2:abs1"


def test_stats(graph):
    s = graph.stats()
    assert s["knoten"] >= 6 and s["kanten"] >= 1
    assert "TestG" in s["gesetze"]


def test_suche_titel(tmp_path):
    from risk_layer.gii import NormKnoten
    knoten = [
        NormKnoten(id="norm:AO:42", law="AO", para="42", text="…",
                   titel="Missbrauch von rechtlichen Gestaltungsmöglichkeiten"),
        NormKnoten(id="norm:AO:162", law="AO", para="162", text="…",
                   titel="Schätzung von Besteuerungsgrundlagen"),
    ]
    g = NormGraph.build(knoten, tmp_path / "s.sqlite")
    res = g.suche_titel("Missbrauch Gestaltung")
    assert res and res[0]["id"] == "norm:AO:42"
    assert g.suche_titel("Schätzung")[0]["id"] == "norm:AO:162"


def test_suche_text_findet_relevante_paragraphen(tmp_path):
    from risk_layer.gii import NormKnoten
    knoten = [
        NormKnoten(id="norm:AO:162", law="AO", para="162", titel="Schätzung von Besteuerungsgrundlagen",
                   text="Soweit die Finanzbehörde die Besteuerungsgrundlagen nicht ermitteln kann, schätzt sie."),
        NormKnoten(id="norm:AO:146", law="AO", para="146",
                   titel="Ordnungsvorschriften für die Buchführung und für Aufzeichnungen",
                   text="Buchungen und Aufzeichnungen sind vollständig, richtig und geordnet vorzunehmen."),
        NormKnoten(id="norm:AO:1", law="AO", para="1", titel="Anwendungsbereich", text="Dieses Gesetz gilt."),
    ]
    g = NormGraph.build(knoten, tmp_path / "vt.sqlite")
    ids = {r["id"] for r in g.suche_text("Die Aufzeichnungen und die Buchführung sind mangelhaft.")}
    assert "norm:AO:146" in ids


def test_graph_query_aus_anderem_thread(graph):
    # Regression: der Web-Server bedient Requests in Worker-Threads; die
    # SQLite-Verbindung muss thread-übergreifend lesbar sein (check_same_thread).
    import threading
    ergebnis: dict = {}

    def arbeiter():
        ergebnis["stats"] = graph.stats()
        ergebnis["auf"] = graph.aufloesen("norm:TestG:2:abs2:nr2")["gefunden"]

    t = threading.Thread(target=arbeiter)
    t.start()
    t.join()
    assert ergebnis["stats"]["knoten"] >= 6
    assert ergebnis["auf"] is True


def test_graph_anreicherung_haengt_gesetzestext_an(tmp_path):
    # Knoten mit der ID eines Katalog-Ankers bauen
    from risk_layer.gii import NormKnoten
    knoten = [NormKnoten(id="norm:UStG:2:abs2:nr2", law="UStG", para="2",
                         absatz="2", nummer="2", text="… organisatorisch eingegliedert …",
                         titel="Unternehmer")]
    g = NormGraph.build(knoten, tmp_path / "u.sqlite")

    eintrag = Katalogeintrag(
        id="x", begriff_kanonisch="organisatorische Eingliederung",
        norm_anker=("§ 2 Abs. 2 Nr. 2 UStG",),
        definitionsquelle=(Definitionsquelle("Verwaltung", "Abschn. 2.8 UStAE"),),
        governance_typ=GovernanceTyp.FF,
    )
    kat = Katalog("t", [eintrag])
    treffer = Treffer("x", 0, 5, "xxxxx", "organisatorische Eingliederung", 0)
    karte = Begriffskarte(treffer=treffer, eintrag=eintrag, status=Status.TREFFER,
                          quellen=list(eintrag.definitionsquelle))
    anreichern_mit_graph([karte], g)
    assert karte.gesetzestext
    assert karte.gesetzestext[0]["verwendet"] == "norm:UStG:2:abs2:nr2"
    assert "eingegliedert" in karte.gesetzestext[0]["text"]
