"""Norm-Änderungs-Radar (radar.py): Manifest, Diff, Anker-Kreuzung, Bericht.

Das Radar ist der Re-Review-Trigger des Festwissens: Nach einem Korpus-/
Graph-Rebuild muss benennbar sein, WELCHE Normen sich geändert haben und
welche Anker (Berater-Definitionen, Kuratierungen, Katalog) betroffen sind –
sonst veralten freigegebene Definitionen still. Die Engine mutiert nichts.
"""

import json

import pytest

from risk_layer import radar
from risk_layer.berater import BeraterStore
from risk_layer.catalog import Katalog
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.models import GovernanceTyp, Katalogeintrag


def _knoten(nid: str, text: str) -> NormKnoten:
    teile = nid.split(":")
    return NormKnoten(id=nid, law=teile[1], para=teile[2], titel=teile[2], text=text)


def _graph(tmp_path, name: str, knoten: list[NormKnoten]) -> NormGraph:
    return NormGraph.build(knoten, tmp_path / f"{name}.sqlite")


# Zwei Korpus-Stände (Muster test_graph: direkte Knoten-Inserts, Textänderung):
#   norm:AO:162   → Text GEÄNDERT
#   norm:AO:140   → Text GEÄNDERT (Präfix-Falle: darf Anker norm:AO:14 NICHT treffen)
#   norm:KStG:8c  → ENTFALLEN
#   norm:AO:14    → unverändert
#   norm:UStG:2   → NEU
_ALT = [
    _knoten("norm:AO:162", "Die Finanzbehörde schätzt."),
    _knoten("norm:AO:140", "Wer Bücher führt."),
    _knoten("norm:AO:14", "Wirtschaftlicher Geschäftsbetrieb."),
    _knoten("norm:KStG:8c", "Verlustabzug bei Körperschaften."),
]
_NEU = [
    _knoten("norm:AO:162", "GEÄNDERTER Wortlaut nach Korpus-Rebuild."),
    _knoten("norm:AO:140", "GEÄNDERTE Buchführungspflichten."),
    _knoten("norm:AO:14", "Wirtschaftlicher Geschäftsbetrieb."),
    _knoten("norm:UStG:2", "Unternehmer ist, wer …"),
]


@pytest.fixture()
def manifeste(tmp_path):
    g_alt = _graph(tmp_path, "alt", _ALT)
    g_neu = _graph(tmp_path, "neu", _NEU)
    return (radar.erstelle_manifest(g_alt, erstellt_am="2026-01-01T00:00:00"),
            radar.erstelle_manifest(g_neu, erstellt_am="2026-06-01T00:00:00"),
            g_neu)


# -- Manifest -------------------------------------------------------------------

def test_manifest_form_und_roundtrip(tmp_path):
    g = _graph(tmp_path, "g", _ALT)
    m = radar.erstelle_manifest(g, erstellt_am="2026-01-01T00:00:00")
    assert m["version"] == radar.MANIFEST_VERSION
    assert m["erstellt_am"] == "2026-01-01T00:00:00"
    assert m["fingerprint"] == g.fingerprint()
    assert m["knoten"] == g.knoten_hashes()
    # Roundtrip speichern → laden: identischer Inhalt
    pfad = tmp_path / "m.json"
    radar.speichere_manifest(pfad, m)
    assert radar.lade_manifest(pfad) == m


def test_manifest_speichern_deterministisch(tmp_path):
    # Gleicher Stand → byte-gleiche Datei (sortierte Keys, diff-freundlich)
    g = _graph(tmp_path, "g", _ALT)
    m = radar.erstelle_manifest(g, erstellt_am="2026-01-01T00:00:00")
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    radar.speichere_manifest(p1, m)
    radar.speichere_manifest(p2, m)
    assert p1.read_bytes() == p2.read_bytes()


def test_lade_manifest_fehlende_datei_ist_none(tmp_path):
    # Kein Vergleichsstand ist der NORMALZUSTAND vor dem zweiten Build – kein Fehler
    assert radar.lade_manifest(tmp_path / "gibtsnicht.json") is None


def test_lade_manifest_korrupt_meldet_pfad(tmp_path):
    pfad = tmp_path / "kaputt.json"
    pfad.write_text("{kein json", encoding="utf-8")
    with pytest.raises(radar.RadarFehler) as exc:
        radar.lade_manifest(pfad)
    assert "kaputt.json" in str(exc.value)


def test_lade_manifest_falsche_struktur_und_version(tmp_path):
    pfad = tmp_path / "m.json"
    pfad.write_text("[]", encoding="utf-8")                    # Liste statt Mapping
    with pytest.raises(radar.RadarFehler):
        radar.lade_manifest(pfad)
    pfad.write_text(json.dumps({"version": 99, "fingerprint": "x", "knoten": {}}),
                    encoding="utf-8")
    with pytest.raises(radar.RadarFehler) as exc:
        radar.lade_manifest(pfad)
    assert "99" in str(exc.value) and "m.json" in str(exc.value)


# -- Diff -----------------------------------------------------------------------

def test_diff_manifeste_faelle_und_sortierung():
    alt = {"knoten": {"b": "1", "a": "2", "z": "3", "y": "4"}}
    neu = {"knoten": {"b": "X", "a": "Y", "n2": "5", "n1": "6"}}
    d = radar.diff_manifeste(alt, neu)
    # sortiert + deterministisch, unabhängig von Dict-Reihenfolgen
    assert d == {"geaendert": ["a", "b"], "neu": ["n1", "n2"], "entfallen": ["y", "z"]}


def test_diff_manifeste_aus_echten_graphen(manifeste):
    alt, neu, _g = manifeste
    d = radar.diff_manifeste(alt, neu)
    assert d["geaendert"] == ["norm:AO:140", "norm:AO:162"]
    assert d["neu"] == ["norm:UStG:2"]
    assert d["entfallen"] == ["norm:KStG:8c"]


# -- Anker-Matching ---------------------------------------------------------------

def test_anker_trifft_exakt_und_beide_richtungen():
    assert radar.anker_trifft("norm:KStG:8c", "norm:KStG:8c")            # exakt
    # Knoten UNTERHALB des Ankers (Anker = §, geänderter Satz darunter)
    assert radar.anker_trifft("norm:KStG:8c", "norm:KStG:8c:abs1:s2")
    # Anker UNTERHALB des Knotens (Anker auf Satz-Ebene, Text hängt am §-Knoten)
    assert radar.anker_trifft("norm:KStG:8c:abs1:s2", "norm:KStG:8c")


def test_anker_trifft_praefix_falle():
    # KEIN Substring-Match: norm:AO:14 darf norm:AO:140 nicht treffen – in
    # BEIDEN Richtungen (Segment-Vergleich auf ':').
    assert not radar.anker_trifft("norm:AO:14", "norm:AO:140")
    assert not radar.anker_trifft("norm:AO:140", "norm:AO:14")
    assert not radar.anker_trifft("norm:AO:14", "norm:EStG:14")          # anderes Gesetz
    assert radar.anker_trifft("norm:AO:14", "norm:AO:14:abs1")           # echtes Kind


# -- betroffene_anker -------------------------------------------------------------

def _katalog() -> Katalog:
    return Katalog("t", [
        Katalogeintrag(id="ao_schaetzung", begriff_kanonisch="Schätzung",
                       norm_anker=("§ 162 AO",), governance_typ=GovernanceTyp.IN),
        Katalogeintrag(id="ohne_anker", begriff_kanonisch="Gesamtwürdigung"),
    ])


def test_betroffene_anker_kreuzt_alle_quellen(tmp_path, manifeste):
    alt, neu, _g = manifeste
    store = BeraterStore(tmp_path / "berater")
    # (a) Berater-Definitionen über geankert_an (Knoten-IDs aus Zitat-Parser)
    store.definiere("Schätzungsrisiko", "…", autor="rey", datum="d",
                    norm_anker=("§ 162 AO",))                       # geaendert
    d2 = store.definiere("Verlustfalle", "…", autor="rey", datum="d",
                         scope="geteilt", norm_anker=("§ 8c KStG",))  # entfallen
    store.setze_review_status(d2.id, "geprüft", reviewer="rv")
    store.setze_review_status(d2.id, "freigegeben", reviewer="rv")
    # Präfix-Falle mit echtem Store: Anker norm:AO:14, geändert ist norm:AO:140
    store.definiere("Geschäftsbetrieb", "…", autor="rey", datum="d",
                    norm_anker=("§ 14 AO",))
    # (b) Norm-Kuratierungs-Record
    store.kuratiere_norm("ao_schaetzung", "§ 162 AO", "ergaenzen",
                         scope="geteilt", autor="rey", datum="d")

    erg = radar.betroffene_anker(radar.diff_manifeste(alt, neu), store, _katalog())
    betroffen = erg["betroffen"]

    b = [t for t in betroffen if t["typ"] == "berater"]
    assert {"typ": "berater", "id": "schaetzungsrisiko", "begriff": "Schätzungsrisiko",
            "scope": "personal", "review_status": "entwurf", "aktion": None,
            "norm_id": "norm:AO:162", "ausloeser": "norm:AO:162",
            "aenderung": "geaendert"} in b
    # entfallener Knoten trifft die freigegebene geteilte Definition
    assert any(t["id"] == d2.id and t["aenderung"] == "entfallen"
               and t["review_status"] == "freigegeben" for t in b)
    # Präfix-Falle: norm:AO:14 wird durch die Änderung an norm:AO:140 NICHT getroffen
    assert not any(t["norm_id"] == "norm:AO:14" for t in betroffen)

    k = [t for t in betroffen if t["typ"] == "kuratierung"]
    assert k and k[0]["id"] == "ao_schaetzung" and k[0]["aktion"] == "ergaenzen" \
        and k[0]["scope"] == "geteilt" and k[0]["ausloeser"] == "norm:AO:162"

    kat = [t for t in betroffen if t["typ"] == "katalog"]
    assert kat == [{"typ": "katalog", "id": "ao_schaetzung", "begriff": "Schätzung",
                    "scope": None, "review_status": None, "aktion": None,
                    "norm_id": "norm:AO:162", "ausloeser": "norm:AO:162",
                    "aenderung": "geaendert"}]

    assert erg["anzahl"]["betroffen"] == len(betroffen)
    assert erg["anzahl"]["freigegeben"] == 1


def test_betroffene_anker_neu_loest_keinen_befund_aus(tmp_path, manifeste):
    # Ein NEUER Knoten macht bestehende Anker nicht falsch – kein Befund.
    alt, neu, _g = manifeste
    store = BeraterStore(tmp_path / "b")
    store.definiere("Unternehmer", "…", autor="rey", datum="d",
                    norm_anker=("§ 2 UStG",))                  # norm:UStG:2 ist NEU
    erg = radar.betroffene_anker(radar.diff_manifeste(alt, neu), store, None)
    assert erg["betroffen"] == []


def test_betroffene_anker_ohne_quellen_nennt_hinweise():
    diff = {"geaendert": ["norm:AO:162"], "neu": [], "entfallen": []}
    erg = radar.betroffene_anker(diff, None, None)
    assert erg["betroffen"] == [] and len(erg["hinweise"]) == 2
    assert any("Berater" in h for h in erg["hinweise"])
    assert any("Katalog" in h for h in erg["hinweise"])
    # Katalog ohne auflösbare Anker → ebenfalls laut benennen, kein stilles Okay
    erg2 = radar.betroffene_anker(
        diff, None, Katalog("t", [Katalogeintrag(id="x", begriff_kanonisch="X")]))
    assert any("Norm-Anker" in h for h in erg2["hinweise"])


def test_dringlichkeits_sortierung(tmp_path):
    # freigegebene geteilte Definition zuerst, dann geprüft/entwurf/personal,
    # danach Kuratierung und Katalog – deterministisch.
    store = BeraterStore(tmp_path / "b")
    store.definiere("P", "…", autor="rey", datum="d", norm_anker=("§ 162 AO",))
    store.definiere("E", "…", autor="rey", datum="d", scope="geteilt",
                    norm_anker=("§ 162 AO",))
    g = store.definiere("G", "…", autor="rey", datum="d", scope="geteilt",
                        norm_anker=("§ 162 AO",))
    store.setze_review_status(g.id, "geprüft", reviewer="rv")
    f = store.definiere("F", "…", autor="rey", datum="d", scope="geteilt",
                        norm_anker=("§ 162 AO",))
    store.setze_review_status(f.id, "geprüft", reviewer="rv")
    store.setze_review_status(f.id, "freigegeben", reviewer="rv")
    store.kuratiere_norm("ao_schaetzung", "§ 162 AO", "verwerfen",
                         scope="personal", autor="rey", datum="d")

    diff = {"geaendert": ["norm:AO:162"], "neu": [], "entfallen": []}
    erg = radar.betroffene_anker(diff, store, _katalog())
    reihen = [(t["typ"], t["review_status"], t["scope"]) for t in erg["betroffen"]]
    assert reihen == [
        ("berater", "freigegeben", "geteilt"),
        ("berater", "geprüft", "geteilt"),
        ("berater", "entwurf", "geteilt"),
        ("berater", "entwurf", "personal"),
        ("kuratierung", None, "personal"),
        ("katalog", None, None),
    ]


def test_betroffene_anker_dedupliziert_doppelte_anker(tmp_path):
    # derselbe Anker doppelt am Eintrag (zwei Zitate, eine Knoten-ID) → EIN Treffer
    store = BeraterStore(tmp_path / "b")
    store.definiere("Schätzung doppelt", "…", autor="rey", datum="d",
                    norm_anker=("§ 162 AO", "162 AO"))
    diff = {"geaendert": ["norm:AO:162"], "neu": [], "entfallen": []}
    erg = radar.betroffene_anker(diff, store, None)
    assert len(erg["betroffen"]) == 1


# -- radar_bericht ----------------------------------------------------------------

def test_radar_bericht_gesamtform(tmp_path, manifeste):
    alt, _neu, g_neu = manifeste
    store = BeraterStore(tmp_path / "b")
    store.definiere("Schätzungsrisiko", "…", autor="rey", datum="d",
                    norm_anker=("§ 162 AO",))
    bericht = radar.radar_bericht(alt, g_neu, store, _katalog(),
                                  erstellt_am="2026-06-10T12:00:00")
    assert bericht["ok"] is True
    assert bericht["stand_alt"] == {"erstellt_am": "2026-01-01T00:00:00",
                                    "fingerprint": alt["fingerprint"]}
    assert bericht["stand_neu"] == {"erstellt_am": "2026-06-10T12:00:00",
                                    "fingerprint": g_neu.fingerprint()}
    assert bericht["normen"]["geaendert"] == ["norm:AO:140", "norm:AO:162"]
    assert bericht["normen"]["entfallen"] == ["norm:KStG:8c"]
    assert bericht["anzahl"]["geaendert"] == 2 and bericht["anzahl"]["neu"] == 1
    assert bericht["anzahl"]["betroffen"] == len(bericht["betroffen"]) >= 2
    assert bericht["anzahl"]["freigegeben"] == 0
    # /v1-tauglich: vollständig JSON-serialisierbar (keine Objekte/Sets im Bericht)
    json.dumps(bericht)


def test_radar_bericht_mutiert_keine_stores(tmp_path, manifeste):
    # Die Engine berichtet NUR – kein Review-Status wird zurückgesetzt
    # (Nur-vorwärts-Invariante des Lebenszyklus bleibt unangetastet).
    alt, _neu, g_neu = manifeste
    store = BeraterStore(tmp_path / "b")
    f = store.definiere("Verlustfalle", "…", autor="rey", datum="d",
                        scope="geteilt", norm_anker=("§ 8c KStG",))
    store.setze_review_status(f.id, "geprüft", reviewer="rv")
    store.setze_review_status(f.id, "freigegeben", reviewer="rv")
    vorher = (tmp_path / "b" / "geteilt.yaml").read_bytes()
    bericht = radar.radar_bericht(alt, g_neu, store, None, erstellt_am="t")
    assert any(t["id"] == f.id for t in bericht["betroffen"])  # Befund da …
    assert (tmp_path / "b" / "geteilt.yaml").read_bytes() == vorher  # … Datei unberührt
    assert BeraterStore(tmp_path / "b").finde("Verlustfalle").review_status == "freigegeben"
