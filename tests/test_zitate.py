"""Parser-Validierung (Parser-Spez. Kap. 5): /T10/ /T20/ /T40/.

Korpus-unabhängig: prüft den Zitat-Parser und die kanonischen IDs gegen die in
der Spezifikation handannotierten Beispiele.
"""

import pytest

from risk_layer.idschema import NormID
from risk_layer.zitate import (
    finde_normzitate,
    norm_zu_zitat,
    parse_entscheidung,
    parse_normzitat,
)


def _ids(zitat):
    return [a.norm_id.to_id() for a in parse_normzitat(zitat) if a.norm_id]


# /T20/ Gold-Set Normzitate ---------------------------------------------------

@pytest.mark.parametrize("zitat,erwartet", [
    ("§ 2 Abs. 2 Nr. 2 UStG", ["norm:UStG:2:abs2:nr2"]),
    ("§ 14 Abs. 1 Nr. 3 Satz 4 KStG", ["norm:KStG:14:abs1:nr3:s4"]),
    ("§ 9 Nr. 1 Satz 2 GewStG", ["norm:GewStG:9:nr1:s2"]),
    ("Art. 11 MwStSystRL", ["norm:MwStSystRL:art11"]),
    ("§ 2 Abs. 2 S. 2 GewStG", ["norm:GewStG:2:abs2:s2"]),
    ("§ 14 Abs. 1 Nr. 3 KStG", ["norm:KStG:14:abs1:nr3"]),
    ("§ 8c Abs. 1 KStG", ["norm:KStG:8c:abs1"]),
])
def test_t20_einzelzitate(zitat, erwartet):
    assert _ids(zitat) == erwartet


def test_t20_liste():
    assert _ids("§§ 14, 17 KStG") == ["norm:KStG:14", "norm:KStG:17"]


def test_t20_bereich():
    assert _ids("§§ 28 bis 31 GewStG") == [
        "norm:GewStG:28", "norm:GewStG:29", "norm:GewStG:30", "norm:GewStG:31",
    ]


def test_t20_verschachtelter_bereich():
    assert _ids("§ 14 Abs. 1 Nr. 1 bis 3 KStG") == [
        "norm:KStG:14:abs1:nr1",
        "norm:KStG:14:abs1:nr2",
        "norm:KStG:14:abs1:nr3",
    ]


def test_ff_offener_bereich_geflaggt():
    anker = parse_normzitat("§ 2 Abs. 2 ff. UStG")
    assert anker and anker[0].norm_id.offener_bereich is True
    assert anker[0].konfidenz == "mittel"


@pytest.mark.parametrize("zitat,erwartet", [
    ("162 AO", ["norm:AO:162"]),
    ("162 Abs. 1 AO", ["norm:AO:162:abs1"]),
    ("8c Abs. 1 KStG", ["norm:KStG:8c:abs1"]),
])
def test_ohne_paragraphzeichen(zitat, erwartet):
    assert _ids(zitat) == erwartet


def test_longest_match_gewstg_vor_stg():
    # 'GewStG' darf nicht als '...StG' fehlinterpretiert werden
    assert _ids("§ 7 GewStG") == ["norm:GewStG:7"]


# /T30/ (Teil) relative Verweise werden geflaggt ------------------------------

def test_relativer_verweis_wird_geflaggt():
    anker = parse_normzitat("Satz 1")
    assert len(anker) == 1
    assert anker[0].relativ is True
    assert anker[0].norm_id is None
    assert anker[0].konfidenz == "niedrig"


# /T10/ Round-Trip: Zitat → ID → Zitat → ID (idempotent) ----------------------

@pytest.mark.parametrize("zitat", [
    "§ 2 Abs. 2 Nr. 2 UStG",
    "§ 14 Abs. 1 Nr. 3 Satz 4 KStG",
    "§ 8c Abs. 1 KStG",
    "Art. 11 MwStSystRL",
])
def test_t10_round_trip(zitat):
    nid = parse_normzitat(zitat)[0].norm_id
    rekonstruiert = norm_zu_zitat(nid)
    nid2 = parse_normzitat(rekonstruiert)[0].norm_id
    assert nid == nid2
    assert nid.to_id() == nid2.to_id()


# /T40/ Gold-Set Entscheidungszitate -----------------------------------------

@pytest.mark.parametrize("zitat,gericht,az", [
    ("BFH v. 18.01.2023 – XI R 29/22", "BFH", "XI-R-29/22"),
    ("BFH v. 11.7.2024 – III R 41/22", "BFH", "III-R-41/22"),
    ("EuGH v. 1.12.2022 – C-141/20", "EuGH", "C-141/20"),
    ("BGH v. 10.7.2014 – IX ZR 285/14", "BGH", "IX-ZR-285/14"),
    ("BVerfG – 2 BvL 19/17", "BVerfG", "2-BvL-19/17"),
])
def test_t40_entscheidungszitate(zitat, gericht, az):
    uid = parse_entscheidung(zitat)
    assert uid is not None
    assert uid.gericht == gericht
    assert uid.aktenzeichen == az
    assert uid.to_id() == f"urteil:{gericht}:{az}"


def test_t40_fundstelle_als_alias():
    uid = parse_entscheidung("BFH v. 18.1.2023 – XI R 29/22, BStBl. II 2023, 409")
    assert "BStBl. II 2023, 409" in uid.aliase


def test_t40_paralleles_aktenzeichen_als_alias():
    uid = parse_entscheidung("BFH v. 18.1.2023 – XI R 29/22 (XI R 16/18)")
    assert any("XI R 16/18" in a for a in uid.aliase)


def test_fg_mit_ort():
    uid = parse_entscheidung("FG Hamburg v. 1.2.2018 – 2 K 245/17")
    assert uid.gericht == "FGHamburg"
    assert uid.aktenzeichen == "2-K-245/17"


# Freitext-Scan: weiterhin korrekt UND gegen polynomialen Lazy-Scan begrenzt ----

def test_finde_normzitate_findet_normale_zitate():
    ids = {a.norm_id.to_id() for a in finde_normzitate(
        "Siehe § 162 Abs. 1 Satz 2 AO und § 42 AO.") if a.norm_id}
    assert any(i.startswith("norm:AO:162") for i in ids)
    assert "norm:AO:42" in ids


def test_finde_normzitate_bounded_gegen_redos():
    import time
    # "§ 1 " + langer Nicht-Gesetz-Schwanz: die {0,80}-Begrenzung verhindert den
    # polynomialen Lazy-Scan (frueher *?). Muss schnell zurueckkehren.
    boese = "§ 1 " + ("a " * 1000000) + "AO"
    t = time.time()
    finde_normzitate(boese)
    assert time.time() - t < 2.0


# LawMap MUSS den Korpus abdecken --------------------------------------------

def test_seed_deckt_korpus():
    """Jedes Gesetz in corpus/gesetze.yaml muss über die LawMap auflösen, sonst
    parsen Zitate darauf zu LAW='?' (genau der *DV-Bug: Gesetz im Korpus, aber
    nicht im Parser). Liest die kanonische Liste, deckt damit künftige Gesetze
    automatisch ab."""
    from risk_layer.corpus import lade_config
    from risk_layer.lawmap import DEFAULT_LAWMAP

    gesetze = lade_config()
    if not gesetze:
        pytest.skip("keine corpus/gesetze.yaml vorhanden")
    fehlend = [g.law for g in gesetze if DEFAULT_LAWMAP.resolve(g.law) != g.law]
    assert not fehlend, f"LawMap deckt diese Korpus-Gesetze nicht ab: {fehlend}"
    # End-to-end: '§ 1 {law}' parst sauber zur §-ID (kein ':st'-Artefakt, kein '?')
    for g in gesetze:
        assert _ids(f"§ 1 {g.law}") == [f"norm:{g.law}:1"], g.law
