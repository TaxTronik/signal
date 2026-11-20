"""Parser-Validierung (Parser-Spez. Kap. 5): /T10/ /T20/ /T40/.

Korpus-unabhängig: prüft den Zitat-Parser und die kanonischen IDs gegen die in
der Spezifikation handannotierten Beispiele.
"""

import pytest

from risk_layer.zitate import (
    extrahiere_anker,
    finde_normzitate,
    kanonisiere_fundstelle,
    norm_zu_zitat,
    parse_entscheidung,
    parse_normzitat,
)


def _ids(zitat):
    return [a.norm_id.to_id() for a in parse_normzitat(zitat) if a.norm_id]


@pytest.mark.parametrize(("fundstelle", "erwartet"), [
    ("§ 39 Abs. 2 Nr. 1 Satz 1 AO", "norm:AO:39:abs2:nr1:s1"),
    ("§ 15 Abs. 2 Satz 1 EStG (Legaldefinition Gewerbebetrieb)", "norm:EStG:15:abs2:s1"),
    ("Art. 1 GG", "norm:GG:art1"),
    ("§§ 7, 8 AStG", None),
    ("§ 146 AO (Ordnungsvorschriften); § 158 AO (Beweiskraft)", None),
    ("§ 2 des Wertpapierinstitutsgesetzes", None),
])
def test_fundstellen_id_fuer_eindeutige_normquelle(fundstelle, erwartet):
    assert kanonisiere_fundstelle("Gesetz", fundstelle) == erwartet


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


@pytest.mark.parametrize(("zitat", "nummern"), [
    ("§§ 1, 4 bis 5 AO", [1, 4, 5]),
    ("§§ 14–17 und 25 AO", [14, 15, 16, 17, 25]),
])
def test_gemischte_liste_und_bereich_auch_im_normgraph(zitat, nummern):
    erwartet = [f"norm:AO:{n}" for n in nummern]
    assert _ids(zitat) == erwartet
    assert [a.norm_id.to_id() for a in extrahiere_anker(zitat, default_law="EStG")] == erwartet


def test_expansionsgrenze_gilt_fuer_das_gesamte_zitat(monkeypatch):
    from risk_layer import zitate
    monkeypatch.setattr(zitate, "_MAX_ANKER_PRO_ZITAT", 4)
    assert len(_ids("§§ 1 bis 2 Abs. 1 bis 2 AO")) == 4
    for text in ("§§ 1 bis 5 AO", "§§ 1 bis 3 Abs. 1 bis 2 AO"):
        anker, = parse_normzitat(text)
        assert anker.norm_id is None and anker.konfidenz == "niedrig"


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


def test_buchst_wert_mit_punkt_ist_kein_offener_bereich():
    # "Buchst. f." trägt den lit-WERT f – die ff./f.-Erkennung darf ihn nicht
    # fressen und das Zitat nicht fälschlich als offenen Bereich flaggen.
    anker = parse_normzitat("§ 33 Abs. 1 Buchst. f. EStG")
    assert [a.norm_id.to_id() for a in anker] == ["norm:EStG:33:abs1:litf"]
    assert anker[0].norm_id.offener_bereich is False


def test_ff_nach_buchst_wert_bleibt_offener_bereich():
    # "Buchst. a ff." – das ff. gehört NICHT zum lit-Wert, sondern flaggt den Bereich
    anker = parse_normzitat("§ 4 Nr. 21 Buchst. a ff. UStG")
    assert anker[0].norm_id.to_id() == "norm:UStG:4:nr21:lita"
    assert anker[0].norm_id.offener_bereich is True


# Tiefenmarker mit reinen Buchstabenwerten („Buchst. a") – gemeinsame Grammatik
# von Parser und Span-Scanner (vorher divergent, Span endete vor „Buchst.") ----

def test_buchstaben_tiefenwert_parst():
    assert _ids("§ 4 Nr. 21 Buchst. a UStG") == ["norm:UStG:4:nr21:lita"]


def test_extrahiere_anker_buchstabenwert_nicht_fehlattribuiert():
    # Repro: der Span muss „Buchst. a" einschließen; vorher endete er davor,
    # „Buchst" wurde als (unbekanntes) Gesetz probiert und das Zitat landete
    # per default_law beim FALSCHEN Gesetz (norm:EStG:4:nr21).
    anker = extrahiere_anker("Nach § 4 Nr. 21 Buchst. a UStG ist befreit.",
                             default_law="EStG")
    assert [a.norm_id.to_id() for a in anker if a.norm_id] == ["norm:UStG:4:nr21:lita"]


# §-Bereiche mit einfachem Bindestrich ----------------------------------------

def test_para_bereich_mit_bindestrich_expandiert():
    assert _ids("§§ 14-17 AO") == [
        "norm:AO:14", "norm:AO:15", "norm:AO:16", "norm:AO:17",
    ]


def test_para_suffix_und_liste_kippen_nicht():
    # Buchstabensuffix bleibt ein einzelner §, Aufzählung bleibt Liste
    assert _ids("§ 64a AO") == ["norm:AO:64a"]
    assert _ids("§§ 14, 15 AO") == ["norm:AO:14", "norm:AO:15"]


# „und"-Aufzählung von Tiefenwerten (Listen-Expansion, Spez. 2.2) -------------

def test_tiefen_liste_mit_und_expandiert():
    anker = parse_normzitat("§ 1 Satz 1 und 2 AO")
    assert [a.norm_id.to_id() for a in anker] == ["norm:AO:1:s1", "norm:AO:1:s2"]
    assert anker[0].konfidenz == "mittel"      # expandiert ⇒ nicht „hoch"


def test_tiefen_liste_mit_komma_und_und():
    assert _ids("§ 2 Abs. 2 Nr. 2, 3 und 5 UStG") == [
        "norm:UStG:2:abs2:nr2", "norm:UStG:2:abs2:nr3", "norm:UStG:2:abs2:nr5",
    ]


def test_gemischte_tiefenliste_in_graph_und_freitext():
    text = "Nach § 1 Nr. 1, 3 bis 5 AO und Art. 11 GG."
    erwartet = ["norm:AO:1:nr1", "norm:AO:1:nr3", "norm:AO:1:nr4",
                "norm:AO:1:nr5", "norm:GG:art11"]
    for anker in (finde_normzitate(text), extrahiere_anker(text, default_law="EStG")):
        assert [a.norm_id.to_id() for a in anker] == erwartet


def test_gesetz_im_spaeteren_freitext_belegt_keinen_relativen_verweis():
    assert not finde_normzitate("Zu § 17 fehlt ein Verweis auf die AO.")


@pytest.mark.parametrize(("zitat", "ziel"), [
    ("§ 139b der Abgabenordnung", "norm:AO:139b"),
    ("§ 43 Absatz 1 Satz 1 des Einkommensteuergesetzes", "norm:EStG:43:abs1:s1"),
    ("§ 257 des Handelsgesetzbuchs", "norm:HGB:257"),
    ("Art. 1 Abs. 1 des Grundgesetzes", "norm:GG:art1:abs1"),
    ("§ 1 der Umsatzsteuer-Durchführungsverordnung", "norm:UStDV:1"),
])
def test_langname_bestimmt_zielgesetz_auch_im_freitext(zitat, ziel):
    assert _ids(zitat) == [ziel]
    for anker in (finde_normzitate(zitat), extrahiere_anker(zitat, "BewG")):
        assert [a.norm_id.to_id() for a in anker if a.norm_id] == [ziel]


@pytest.mark.parametrize("fremd", [
    "§ 2 Absatz 1 des Wertpapierinstitutsgesetzes",
    "§§ 2, 4 Absatz 1 Satz 1 und 2 des Bürgerlichen Gesetzbuches",
    "Art. 5 der Verordnung (EU) 2016/679",
    "§ 2 WpIG",
    "§ 17 der Abgabenordnungsverordnung",
])
def test_unbekannte_explizite_quelle_ergibt_keinen_eigenverweis(fremd):
    text = fremd + "; davon unabhängig gilt § 9 dieses Gesetzes."
    assert not finde_normzitate(text)
    assert [a.norm_id.to_id() for a in extrahiere_anker(text, "HGB") if a.norm_id] == [
        "norm:HGB:9"
    ]


def test_singularverweis_ohne_gesetz_frisst_keine_fremde_aufzaehlung():
    text = "Die Unterlagen nach § 1 und 2 weitere Belege sind einzureichen."
    assert [a.norm_id.to_id() for a in extrahiere_anker(text, "AO")] == ["norm:AO:1"]
    assert [a.norm_id.to_id() for a in finde_normzitate("§ 2 Abs. 1 der AO")] == ["norm:AO:2:abs1"]


@pytest.mark.parametrize(("text", "erwartet"), [
    ("§ 72a Absatz 4 und § 93c Absatz 4 der Abgabenordnung finden keine Anwendung.",
     ["norm:AO:72a:abs4", "norm:AO:93c:abs4"]),
    ("§ 76 Absatz 4 und § 111 Absatz 5 des Aktiengesetzes",
     ["norm:AktG:76:abs4", "norm:AktG:111:abs5"]),
    ("Art. 1 Abs. 1, Artikel 2 und Art. 3 des Grundgesetzes",
     ["norm:GG:art1:abs1", "norm:GG:art2", "norm:GG:art3"]),
    ("§ 1 und § 2 AO sowie § 3 oder § 4 EStG",
     ["norm:AO:1", "norm:AO:2", "norm:EStG:3", "norm:EStG:4"]),
    ("§ 1 oder § 2 WpIG, § 3 AO", ["norm:AO:3"]),
    ("Artikel 5 Nummer 20 und 21 des Zollkodex der Union", []),
    ("Artikel 4 und Artikel 5 des Zollkodexes der Union, § 3 AO", ["norm:AO:3"]),
])
def test_gemeinsamer_gesetzesname_in_zitierketten(text, erwartet):
    for anker in (finde_normzitate(text), extrahiere_anker(text, "BewG")):
        assert [a.norm_id.to_id() for a in anker if a.norm_id] == erwartet
        assert all(a.roh in text for a in anker)


@pytest.mark.parametrize("trenner", [". ", "; ", ", im Übrigen gilt ", " und nach "])
def test_gemeinsamer_gesetzesname_endet_an_bindungsgrenzen(trenner):
    text = "§ 1a" + trenner + "§ 2 AO"
    assert [a.norm_id.to_id() for a in finde_normzitate(text)] == ["norm:AO:2"]
    assert [a.norm_id.to_id() for a in extrahiere_anker(text, "HGB")] == [
        "norm:HGB:1a", "norm:AO:2",
    ]


def test_gesetzesname_bindet_weder_rueckwaerts_an_fremde_normart_noch_vorwaerts():
    text = "§ 1 und Art. 2 GG sowie Art. 3"
    assert [a.norm_id.to_id() for a in finde_normzitate(text)] == ["norm:GG:art2"]
    assert [a.norm_id.to_id() for a in extrahiere_anker(text, "HGB")] == [
        "norm:HGB:1", "norm:GG:art2", "norm:HGB:art3",
    ]


def test_und_frisst_kein_folgewort():
    # „und folgende" ist KEIN Listenwert – der Einzelbuchstabe darf das
    # Folgewort nicht anschneiden (s. (?!\w) in der Grammatik)
    assert _ids("§ 1 Satz 1 und folgende AO") == ["norm:AO:1:s1"]


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


def test_entscheidung_ohne_datum_az_verankert():
    # Ohne Datum darf der Az-Scanner führende Wörter („Urteil") nicht mitfressen –
    # sonst joint urteil:BFH:Urteil-XI-R-29/22 nie mit urteil:BFH:XI-R-29/22.
    uid = parse_entscheidung("BFH, Urteil XI R 29/22")
    assert uid is not None
    assert uid.to_id() == "urteil:BFH:XI-R-29/22"


def test_entscheidung_beschluss_und_sonderregister():
    uid = parse_entscheidung("BFH, Beschluss v. 28.11.2016 – GrS 1/15")
    assert uid is not None
    assert uid.to_id() == "urteil:BFH:GrS-1/15"      # Sonderregister bleibt parsebar


# Freitext-Scan: weiterhin korrekt UND gegen polynomialen Lazy-Scan begrenzt ----

def test_finde_normzitate_findet_normale_zitate():
    ids = {a.norm_id.to_id() for a in finde_normzitate(
        "Siehe § 162 Abs. 1 Satz 2 AO und § 42 AO.") if a.norm_id}
    assert any(i.startswith("norm:AO:162") for i in ids)
    assert "norm:AO:42" in ids


def test_finde_normzitate_ueberlaeuft_keine_satzgrenze():
    # „AO" gehört zum FOLGESATZ – vorher entstand norm:AO:17:abs2 mit
    # Konfidenz hoch (falsch-positiv), weil die Zeichenklasse Punkt+Wörter
    # zwischen §-Zahl und Gesetz zuließ.
    anker = finde_normzitate("Es gilt § 17 Abs. 2. Die AO regelt das Verfahren.")
    assert not [a.norm_id.to_id() for a in anker if a.norm_id]


def test_finde_normzitate_inline_abkuerzungen_bleiben():
    # legitime Abkürzungspunkte INNERHALB des Zitats (Abs./S.) laufen weiter durch
    ids = {a.norm_id.to_id() for a in finde_normzitate(
        "Nach § 17 Abs. 2 S. 1 AO ist zu schätzen.") if a.norm_id}
    assert "norm:AO:17:abs2:s1" in ids


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
