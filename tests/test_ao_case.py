"""End-to-End: AO-Betriebsprüfungs-Sachverhalt (mehrere Gesetze, Graph aktiv)."""

import pytest

from risk_layer.catalog import lade_katalog
from risk_layer.graph import NormGraph
from risk_layer.pipeline import Pipeline
from risk_layer.gii import NormKnoten


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    p = tmp_path_factory.mktemp("g") / "g.sqlite"
    knoten = [
        NormKnoten(id="norm:AO:146", law="AO", para="146", absatz=None, nummer=None,
                   titel="Ordnungsvorschriften", text="Buchführung und Aufzeichnungen."),
        NormKnoten(id="norm:AO:162", law="AO", para="162", absatz=None, nummer=None,
                   titel="Schätzung", text="Schätzung der Besteuerungsgrundlagen."),
        NormKnoten(id="norm:AO:173", law="AO", para="173", absatz=None, nummer=None,
                   titel="Aufhebung", text="Aufhebung oder Änderung von Steuerbescheiden."),
        NormKnoten(id="norm:AO:42", law="AO", para="42", absatz=None, nummer=None,
                   titel="Gestaltungsmissbrauch", text="Missbrauch von rechtlichen Gestaltungsmöglichkeiten."),
        NormKnoten(id="norm:AO:69", law="AO", para="69", absatz=None, nummer=None,
                   titel="Haftung", text="Haftung der Vertreter."),
    ]
    return NormGraph.build(knoten, p)


def test_ao_case_schicht1_und_graph(graph):
    text = (
        "Wegen nicht ordnungsmäßiger Buchführung droht die Schätzung der "
        "Besteuerungsgrundlagen. Zudem prüft das Finanzamt einen Missbrauch von "
        "rechtlichen Gestaltungsmöglichkeiten (§ 42 AO) sowie eine Änderung wegen "
        "neuer Tatsachen. Schließlich steht die Haftung des Geschäftsführers im Raum."
    )
    rep = Pipeline(lade_katalog()).run(text, graph=graph)
    begriffe = {k.eintrag.id for k in rep.karten}
    assert "ao_schaetzung" in begriffe
    assert "ao_gestaltungsmissbrauch" in begriffe
    # § 42 AO wird als Norm-Anker aufgelöst (Graph aktiv)
    schaetzung = next(k for k in rep.karten if k.eintrag.id == "ao_schaetzung")
    assert any("162" in g["verwendet"] for g in schaetzung.gesetzestext)


def test_ao_edge_case_wird_durch_schicht1b_erkannt(graph):
    """Der Edge-Case nennt KEINE Begriffe wörtlich, sondern UMSCHREIBT sie.

    Früher blieb Schicht 1 hier leer (nur das LLM erkannte etwas). Mit Schicht 1b
    (deterministische Beschreibungs-/Negativbefund-Muster) werden die umschriebenen
    Konzepte jetzt OHNE Modell erkannt – reproduzierbar und korpus-fundiert.
    """
    edge = (
        "In mehreren Jahren erscheinen Bareinnahmen nicht in der Kasse und die "
        "Belege fehlen. Weil sich die wirklichen Umsätze nicht feststellen lassen, "
        "setzt das Amt die Bemessungsgrundlagen nach Erfahrungssätzen an. Die "
        "bestandskräftigen Bescheide werden geändert, weil nachträglich Umstände "
        "bekannt werden, die zuvor nicht vorlagen."
    )
    rep = Pipeline(lade_katalog()).run(edge, graph=graph)
    muster_ids = {k.eintrag.id for k in rep.karten if k.via == "muster"}
    # Buchführungs-/Kassenmangel, Schätzung und neue Tatsachen – alle umschrieben.
    assert "ao_buchfuehrungs_kassenmangel" in muster_ids
    assert "ao_schaetzung" in muster_ids
    assert "ao_neue_tatsachen" in muster_ids
    # Es bleibt deterministisch: jede Muster-Karte trägt herkunft schicht 1b.
    for k in rep.karten:
        if k.via == "muster":
            assert k.herkunft()["interpretation"] is False
