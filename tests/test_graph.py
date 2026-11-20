"""Normgraph: Bau, Kanten, Dangling, Auflösung, Graph-Anreicherung (/LF170/, /T50/)."""

from pathlib import Path

import pytest

from risk_layer import gii
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


@pytest.mark.parametrize("vorhanden", [False, True])
def test_abgebrochener_rebuild_veroeffentlicht_keine_teildatenbank(tmp_path, vorhanden):
    ziel = tmp_path / "graph.sqlite"
    knoten = gii.parse_datei(FIXTURE)
    if vorhanden:
        NormGraph.build(knoten, ziel).close()
    vorher = ziel.read_bytes() if vorhanden else None

    def unterbrochen():
        yield knoten[0]
        raise ValueError("unvollständiger Import")

    with pytest.raises(ValueError, match="unvollständiger Import"):
        NormGraph.build(unterbrochen(), ziel)
    assert (ziel.read_bytes() if ziel.exists() else None) == vorher
    assert list(tmp_path.iterdir()) == ([ziel] if vorhanden else [])


def test_dangling_cross_law(graph):
    # „§ 5 EStG" verweist auf einen nicht im Korpus vorhandenen Knoten (/T50/)
    ziele = {d["dst"] for d in graph.dangling()}
    assert "norm:EStG:5" in ziele


def test_aufloesen_fallback_auf_vorfahr(graph):
    # nicht existente Satz-ID fällt auf den vorhandenen Absatz zurück
    r = graph.aufloesen("norm:TestG:2:abs1:s1")
    assert r["gefunden"]
    assert r["verwendet"] == "norm:TestG:2:abs1"


def test_aufloesen_misserfolg_hat_konsistente_schluessel(graph):
    # Misserfolgs-Dict trägt DIESELBEN Schlüssel wie der Erfolgsfall
    # ("verweist_auf"/"verwendet"); "verweise" bleibt als Alt-Alias erhalten.
    r = graph.aufloesen("norm:NixG:1")
    assert r["gefunden"] is False
    assert r["verweist_auf"] == [] and r["dangling"] == []
    assert r["verwendet"] is None
    assert r["verweise"] == []                  # additiver Alt-Schlüssel


def test_mehrfachzitat_behaelt_hoechste_konfidenz(tmp_path):
    # Zwei Zitate aufs selbe Ziel: explizit (hoch) ZUERST, implizit (mittel)
    # danach – Last-Write-Wins hätte „mittel" stehen lassen, gewinnen muss
    # die HÖCHSTE Konfidenz.
    from risk_layer.gii import NormKnoten
    k = NormKnoten(
        id="norm:AO:1", law="AO", para="1", titel="Anwendungsbereich",
        text="Siehe § 2 Abs. 1 AO im Detail. Später gilt § 2 Abs. 1 entsprechend.")
    g = NormGraph.build([k], tmp_path / "konf.sqlite")
    rows = g._query(
        "SELECT konfidenz FROM kanten WHERE src='norm:AO:1' AND dst='norm:AO:2:abs1'")
    assert [r["konfidenz"] for r in rows] == ["hoch"]


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
    # suche_titel teilt sich den Knoten-Cache mit suche_text – Mutationen am
    # Ergebnis dürfen NICHT in den Cache durchschlagen (Kopien).
    res[0]["titel"] = "MUTIERT"
    assert g.suche_titel("Missbrauch Gestaltung")[0]["titel"] != "MUTIERT"


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


def test_suche_text_bricht_score_gleichstand_stabil_nach_id(tmp_path):
    """Gleiche Scores duerfen nicht von Set-/Hash-Reihenfolgen abhaengen."""
    knoten = [
        gii.NormKnoten(id="norm:Z:2", law="Z", para="2", titel="Alpha",
                       text="Alpha Begriff gemeinsam"),
        gii.NormKnoten(id="norm:A:1", law="A", para="1", titel="Alpha",
                       text="Alpha Begriff gemeinsam"),
    ]
    graph = NormGraph.build(knoten, tmp_path / "tie.sqlite")

    assert [treffer["id"] for treffer in graph.suche_text(
        "Alpha Begriff gemeinsam", min_score=1.0
    )] == ["norm:A:1", "norm:Z:2"]


def test_suche_text_erreicht_schwelle_mit_zehn_haeufigen_woertern(tmp_path):
    text = "Punkte Texte Suche Daten Worte Testen Index Graph Quelle Grenze"
    knoten = [gii.NormKnoten(id=f"norm:A:{i}", law="A", para=str(i), titel="Titel",
                             text=text) for i in range(26)]
    graph = NormGraph.build(knoten, tmp_path / "schwelle.sqlite")
    try:
        treffer, = graph.suche_text(text, k=1, min_score=2.0)
        treffer["titel"] = "geändert"
        assert graph.suche_text(text, k=1)[0]["titel"] == "Titel"
        assert not graph.suche_text(text, min_score=2.01)
    finally:
        graph.close()


def test_fingerprint_stabil_ueber_zwei_builds(tmp_path):
    # Staleness-Erkennung steht und fällt mit der Stabilität: ZWEI Builds aus
    # demselben Fixture-Korpus müssen denselben Fingerprint liefern – sonst
    # würde jeder Graph-Rebuild den Embedding-Index fälschlich als veraltet melden.
    g1 = NormGraph.build(gii.parse_datei(FIXTURE), tmp_path / "f1.sqlite")
    g2 = NormGraph.build(gii.parse_datei(FIXTURE), tmp_path / "f2.sqlite")
    fp = g1.fingerprint()
    assert fp == g2.fingerprint()
    assert fp == g1.fingerprint()                  # idempotent (kein Zustand)
    assert len(fp) == 64 and int(fp, 16) >= 0      # sha256-Hex


def test_fingerprint_unabhaengig_von_insert_reihenfolge(tmp_path):
    from risk_layer.gii import NormKnoten
    knoten = [
        NormKnoten(id="norm:AO:146", law="AO", para="146", titel="Ordnungsvorschriften",
                   text="Bücher sind vollständig zu führen."),
        NormKnoten(id="norm:AO:162", law="AO", para="162", titel="Schätzung",
                   text="Die Finanzbehörde schätzt."),
    ]
    g1 = NormGraph.build(knoten, tmp_path / "a.sqlite")
    g2 = NormGraph.build(list(reversed(knoten)), tmp_path / "b.sqlite")
    # Sortiert wird im Fingerprint (ORDER BY id) – die Reihenfolge beim Bau
    # darf das Ergebnis nicht beeinflussen (deterministisch).
    assert g1.fingerprint() == g2.fingerprint()


def test_fingerprint_aendert_sich_bei_textaenderung(tmp_path):
    from risk_layer.gii import NormKnoten
    basis = NormKnoten(id="norm:AO:162", law="AO", para="162", titel="Schätzung",
                       text="Die Finanzbehörde schätzt.")
    geaendert = NormKnoten(id="norm:AO:162", law="AO", para="162", titel="Schätzung",
                           text="GEÄNDERTER Wortlaut nach Korpus-Rebuild.")
    g1 = NormGraph.build([basis], tmp_path / "v1.sqlite")
    g2 = NormGraph.build([geaendert], tmp_path / "v2.sqlite")
    # Genau der Fall, den die Staleness-Erkennung aufdecken soll: gleicher
    # Knoten, neuer Normtext ⇒ anderer Fingerprint.
    assert g1.fingerprint() != g2.fingerprint()


def test_knoten_hashes_form_und_inhalt(tmp_path):
    import hashlib
    from risk_layer.gii import NormKnoten
    knoten = [
        NormKnoten(id="norm:AO:162", law="AO", para="162", titel="Schätzung",
                   text="Die Finanzbehörde schätzt."),
        NormKnoten(id="norm:AO:146", law="AO", para="146", titel="Ordnung", text=""),
    ]
    g = NormGraph.build(knoten, tmp_path / "kh.sqlite")
    hashes = g.knoten_hashes()
    assert set(hashes) == {"norm:AO:146", "norm:AO:162"}
    # Hash = sha256 über den UTF-8-Normtext; leerer Text hasht den Leerstring
    assert hashes["norm:AO:162"] == hashlib.sha256(
        "Die Finanzbehörde schätzt.".encode("utf-8")).hexdigest()
    assert hashes["norm:AO:146"] == hashlib.sha256(b"").hexdigest()
    # deterministische Reihenfolge (ORDER BY id) – das Radar-Manifest und der
    # Fingerprint hängen daran
    assert list(hashes) == sorted(hashes)


def test_fingerprint_byte_identisch_zum_alt_algorithmus(tmp_path):
    # KONTRAKT: fingerprint() baut jetzt auf knoten_hashes() auf, muss aber
    # BYTE-IDENTISCH zum historischen Algorithmus bleiben – der Wert ist in
    # gebauten Embedding-Indizes (meta.json) eingebrannt; eine Abweichung
    # meldete jeden bestehenden Index fälschlich als veraltet.
    import hashlib
    g = NormGraph.build(gii.parse_datei(FIXTURE), tmp_path / "fp.sqlite")
    h = hashlib.sha256()
    for r in g._query("SELECT id, text FROM knoten ORDER BY id"):
        text_hash = hashlib.sha256((r["text"] or "").encode("utf-8")).hexdigest()
        h.update(r["id"].encode("utf-8") + b"\x00" + text_hash.encode("ascii") + b"\x00")
    assert g.fingerprint() == h.hexdigest()


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
    treffer = Treffer("x", 0, 5, "xxxxx", "organisatorische Eingliederung", 0)
    karte = Begriffskarte(treffer=treffer, eintrag=eintrag, status=Status.TREFFER,
                          quellen=list(eintrag.definitionsquelle))
    anreichern_mit_graph([karte], g)
    assert karte.gesetzestext
    assert karte.gesetzestext[0]["verwendet"] == "norm:UStG:2:abs2:nr2"
    assert "eingegliedert" in karte.gesetzestext[0]["text"]
