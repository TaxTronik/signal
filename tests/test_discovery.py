"""Discovery-Harness (Pass 1): Verarbeitung der LLM-Roh-Kandidaten.

Ohne LLM/Graph/GPU — verarbeite_kandidaten + Gates werden mit Stubs geprüft:
Anti-Halluzination (Textgrundlage muss im § stehen), abgeleitete Authority,
Relations-Erdung, Paraphrase-Gate, Deckel, Review-/Training-Gates.
"""

from risk_layer.discovery import (
    Kandidat, Karte, besteht_review_gate, besteht_training_gate, verarbeite_kandidaten,
    verarbeite_karte,
)

TEXT = ("Wird das Einkommen geschätzt, sind die Besteuerungsgrundlagen zu ermitteln. "
        "Mehr als 50 Prozent der Anteile gelten als schädlich. "
        "Die Verluste sind nicht abziehbar.")
TITEL = "Schätzung"
QUELL = "norm:EStDV:8"          # Verordnungs-§ (für Default-Authority)
LAW = "EStDV"
IM_GRAPH = {"norm:EStDV:8", "norm:EStG:51", "norm:HGB:22"}


def _verarbeite(roh, verweist=(), enable_normrelation=False, enable_begriff=True):
    return verarbeite_kandidaten(
        roh, quell_norm=QUELL, quell_law=LAW, node_text=TEXT, node_titel=TITEL,
        verweist_ids=verweist, ist_im_graph=lambda nid: nid in IM_GRAPH,
        enable_begriff=enable_begriff, enable_normrelation=enable_normrelation)


def test_begriff_default_aus():
    # Pass-1-Default: begriff aus (Substantiv-Rauschen); per Flag wieder an.
    roh = [{"typ": "begriff", "inhalt": "schädlich",
            "textgrundlage": "gelten als schädlich", "konfidenz": "hoch"}]
    assert _verarbeite(roh, enable_begriff=False) == []
    assert len(_verarbeite(roh, enable_begriff=True)) == 1


def test_textgrundlage_muss_im_paragraph_stehen():
    roh = [
        {"typ": "begriff", "inhalt": "Schätzung", "textgrundlage": "Besteuerungsgrundlagen", "konfidenz": "hoch"},
        {"typ": "begriff", "inhalt": "Halluzination", "textgrundlage": "Umsatzsteuervoranmeldung", "konfidenz": "hoch"},
    ]
    out = _verarbeite(roh)
    assert [k.inhalt for k in out] == ["Schätzung"]


def test_authority_wird_abgeleitet_nicht_geglaubt():
    roh = [
        {"typ": "begriff", "inhalt": "a", "textgrundlage": "schädlich", "ziel_norm": "§ 8 EStDV", "konfidenz": "hoch"},
        {"typ": "begriff", "inhalt": "b", "textgrundlage": "geschätzt", "ziel_norm": "§ 51 EStG", "konfidenz": "hoch"},
    ]
    out = {k.inhalt: k for k in _verarbeite(roh)}
    assert out["a"].quelle_typ == "verordnung" and out["a"].source_rank == "statutory_regulation"
    assert out["a"].ziel_norm_id == "norm:EStDV:8" and out["a"].ziel_im_graph is True
    assert out["b"].quelle_typ == "gesetz" and out["b"].source_rank == "statute"


def test_default_zielnorm_ist_der_quell_paragraph():
    roh = [{"typ": "begriff", "inhalt": "x", "textgrundlage": "schädlich", "konfidenz": "hoch"}]
    out = _verarbeite(roh)
    assert out[0].ziel_norm_id == QUELL and out[0].quelle_typ == "verordnung"


def test_paraphrase_keine_wortlautkopie():
    roh = [
        {"typ": "paraphrase", "inhalt": "nicht abziehbar", "textgrundlage": "nicht abziehbar", "konfidenz": "hoch"},
        {"typ": "paraphrase", "inhalt": "die Anteile sind fürs Finanzamt futsch", "textgrundlage": "Mehr als 50 Prozent", "konfidenz": "hoch"},
    ]
    out = _verarbeite(roh)
    assert [k.inhalt for k in out] == ["die Anteile sind fürs Finanzamt futsch"]


def test_paraphrase_umstellung_raus():
    # Reine Wortumstellung der Grundlage (Token-Jaccard ~1) -> kein Register-Bridge -> raus.
    roh = [{"typ": "paraphrase", "inhalt": "Nicht abziehbar sind die Verluste",
            "textgrundlage": "Die Verluste sind nicht abziehbar", "konfidenz": "hoch"}]
    assert _verarbeite(roh) == []


def test_faithfulness_norm_im_inhalt_blockt_gate():
    # inhalt behauptet § 162 AO, Textgrundlage belegt das nicht -> kritischer Befund.
    roh = [{"typ": "paraphrase", "inhalt": "die Schätzung richtet sich nach § 162 AO",
            "textgrundlage": "Besteuerungsgrundlagen", "konfidenz": "hoch"}]
    out = _verarbeite(roh)
    assert len(out) == 1 and any(b.startswith("norm_unbelegt") for b in out[0].befunde)
    assert not besteht_review_gate(out[0]) and not besteht_training_gate(out[0])


def test_faithfulness_zahl_nur_beratend():
    # erfundene Sachzahl (80 statt 50) -> Befund, aber Gate bleibt offen (advisory).
    roh = [{"typ": "paraphrase", "inhalt": "mindestens 80 Prozent sind nötig",
            "textgrundlage": "Mehr als 50 Prozent", "konfidenz": "hoch"}]
    out = _verarbeite(roh)
    assert len(out) == 1 and any(b.startswith("zahl_unbelegt") for b in out[0].befunde)
    assert besteht_review_gate(out[0])


def test_cross_ref_braucht_fremdnorm_im_graph():
    roh = [
        # Fremdnorm in bezugsnorm + im Graph -> gehalten, relation verweist_auf
        {"typ": "cross_ref", "inhalt": "verweist auf HGB", "textgrundlage": "schädlich",
         "bezugsnorm": "§ 22 HGB", "konfidenz": "hoch"},
        # Fremdnorm nicht im Graph -> raus
        {"typ": "cross_ref", "inhalt": "x", "textgrundlage": "geschätzt",
         "bezugsnorm": "§ 999 HGB", "konfidenz": "hoch"},
        # keine parsebare Fremdnorm (nur Prosa in inhalt) -> raus
        {"typ": "cross_ref", "inhalt": "§ 22 des Handelsgesetzbuchs", "textgrundlage": "Verluste",
         "konfidenz": "hoch"},
    ]
    out = _verarbeite(roh)
    assert len(out) == 1
    k = out[0]
    assert k.typ == "cross_ref" and k.relation == "verweist_auf"
    assert k.bezugsnorm_id == "norm:HGB:22" and k.ziel_norm_id == QUELL


def test_cross_ref_kein_selbstbezug():
    roh = [{"typ": "cross_ref", "inhalt": "self", "textgrundlage": "schädlich",
            "bezugsnorm": "§ 8 EStDV", "konfidenz": "hoch"}]   # == Quell-§
    assert _verarbeite(roh) == []


def test_normrelation_default_aus_und_an():
    roh = [{"typ": "normrelation", "inhalt": "konkretisiert HGB", "textgrundlage": "schädlich",
            "bezugsnorm": "§ 22 HGB", "relation": "konkretisiert", "konfidenz": "hoch"}]
    assert _verarbeite(roh) == []                                  # Flag aus -> verworfen
    out = _verarbeite(roh, verweist={"norm:HGB:22"}, enable_normrelation=True)
    assert len(out) == 1 and out[0].relation == "konkretisiert"
    assert out[0].bezugsnorm_id == "norm:HGB:22"


def test_normrelation_an_aber_ungestuetzt_raus():
    # Flag an, aber kein Verweis/Wortlaut/Titel -> raus (anders als cross_ref: streng)
    roh = [{"typ": "normrelation", "inhalt": "x", "textgrundlage": "geschätzt",
            "bezugsnorm": "§ 22 HGB", "relation": "konkretisiert", "konfidenz": "hoch"}]
    assert _verarbeite(roh, enable_normrelation=True) == []


def test_begriff_traegt_keine_relation():
    roh = [{"typ": "begriff", "inhalt": "Schätzung", "textgrundlage": "geschätzt",
            "bezugsnorm": "§ 22 HGB", "relation": "verweist_auf", "konfidenz": "hoch"}]
    out = _verarbeite(roh)
    assert len(out) == 1 and out[0].relation == "keine" and out[0].bezugsnorm == ""


def test_begriff_bloesses_wort_raus():
    # textgrundlage == inhalt (nur das Wort) -> kein Beleg für Unbestimmtheit -> raus;
    # mit umgebendem Satz als textgrundlage bleibt der begriff.
    roh = [
        {"typ": "begriff", "inhalt": "schädlich", "textgrundlage": "schädlich", "konfidenz": "hoch"},
        {"typ": "begriff", "inhalt": "schädlich",
         "textgrundlage": "Mehr als 50 Prozent der Anteile gelten als schädlich", "konfidenz": "hoch"},
    ]
    out = _verarbeite(roh)
    assert len(out) == 1 and out[0].textgrundlage.startswith("Mehr als 50")


def test_deckel_sechs_pro_paragraph():
    roh = [{"typ": "begriff", "inhalt": f"b{i}", "textgrundlage": "Besteuerungsgrundlagen",
            "konfidenz": "hoch"} for i in range(8)]
    assert len(_verarbeite(roh)) == 6


def test_deckel_drei_paraphrasen():
    roh = [{"typ": "paraphrase", "inhalt": f"laienformulierung nummer {i}",
            "textgrundlage": "Mehr als 50 Prozent", "konfidenz": "hoch"} for i in range(5)]
    out = _verarbeite(roh)
    assert sum(1 for k in out if k.typ == "paraphrase") == 3


def test_dedup_gleicher_kandidat():
    roh = [
        {"typ": "begriff", "inhalt": "Schätzung", "textgrundlage": "geschätzt", "konfidenz": "hoch"},
        {"typ": "begriff", "inhalt": "Schätzung", "textgrundlage": "Besteuerungsgrundlagen", "konfidenz": "hoch"},
    ]
    assert len(_verarbeite(roh)) == 1


def _k(**kw):
    base = dict(quell_norm=QUELL, typ="paraphrase", funktion="unklar", inhalt="x",
               textgrundlage="schädlich", quelle_typ="verordnung", source_rank="statutory_regulation",
               ziel_norm="", ziel_norm_id="norm:EStDV:8", ziel_im_graph=True, bezugsnorm="",
               bezugsnorm_id="", relation="keine", nachbarnormen=[], generisch=False, konfidenz="hoch")
    base.update(kw)
    return Kandidat(**base)


def test_review_gate():
    assert besteht_review_gate(_k(konfidenz="mittel"))
    assert besteht_review_gate(_k(konfidenz="hoch"))
    assert not besteht_review_gate(_k(konfidenz="niedrig"))
    assert not besteht_review_gate(_k(generisch=True))
    assert not besteht_review_gate(_k(ziel_norm_id=""))
    assert not besteht_review_gate(_k(befunde=["norm_unbelegt:norm:HGB:99"]))   # kritisch
    assert besteht_review_gate(_k(befunde=["zahl_unbelegt:80"]))                # advisory


def test_training_gate():
    assert besteht_training_gate(_k(typ="paraphrase", konfidenz="hoch"))
    assert not besteht_training_gate(_k(konfidenz="mittel"))           # nur hoch
    assert not besteht_training_gate(_k(ziel_im_graph=False))          # muss im Graph sein
    assert not besteht_training_gate(_k(typ="cross_ref"))              # nur paraphrase|begriff
    assert not besteht_training_gate(_k(typ="begriff", funktion="unklar"))
    assert besteht_training_gate(_k(typ="begriff", funktion="tatbestandsmerkmal"))
    assert not besteht_training_gate(_k(befunde=["norm_unbelegt:norm:HGB:99"]))   # kritisch


# --- Pass 2: Card-Discovery (verarbeite_karte teilt die Pass-1-Gates) ---

NODE = {"id": QUELL, "law": LAW, "zitat": "§ 8 EStDV", "titel": TITEL, "text": TEXT}


def _karte(card, node=NODE, verweist=()):
    return verarbeite_karte(card, node=node, verweist_ids=verweist,
                            ist_im_graph=lambda nid: nid in IM_GRAPH)


def test_karte_guter_anker_akzeptiert():
    card = {"anchors": [{"text": "die Anteile sind fürs Finanzamt futsch",
                         "textgrundlage": "Mehr als 50 Prozent"}],
            "nachbarnormen": [], "matcher_terms": [],
            "advisory": {"hinweis": "trägt", "risikotyp": "haftung"}}
    k = _karte(card)
    assert isinstance(k, Karte) and k.anker_vorgeschlagen == 1 and k.anker_akzeptiert == 1
    assert [c.typ for c in k.kandidaten] == ["paraphrase"]
    assert k.kandidaten[0].inhalt == "die Anteile sind fürs Finanzamt futsch"
    assert k.advisory["hinweis"] == "trägt" and k.advisory["risikotyp"] == "haftung"
    assert k.review_status == "neu"


def test_karte_umstellung_raus():
    card = {"anchors": [{"text": "Nicht abziehbar sind die Verluste",
                         "textgrundlage": "Die Verluste sind nicht abziehbar"}],
            "nachbarnormen": [], "hinweis": ""}
    k = _karte(card)
    assert k.anker_vorgeschlagen == 1 and k.anker_akzeptiert == 0 and k.kandidaten == []


def test_karte_unbelegte_norm_wird_geflaggt():
    card = {"anchors": [{"text": "richtet sich nach § 162 AO",
                         "textgrundlage": "Besteuerungsgrundlagen"}],
            "nachbarnormen": [], "hinweis": ""}
    k = _karte(card)
    assert len(k.kandidaten) == 1
    assert any(b.startswith("norm_unbelegt") for b in k.kandidaten[0].befunde)
    assert not besteht_review_gate(k.kandidaten[0])      # krit. Befund -> nicht reviewfähig


def test_karte_nachbarnorm_nur_wenn_im_wortlaut_und_graph():
    node = {"id": QUELL, "law": LAW, "zitat": "§ 8 EStDV", "titel": TITEL,
            "text": "Die Besteuerungsgrundlagen sind nach § 22 HGB zu ermitteln."}
    card = {"anchors": [], "nachbarnormen": ["§ 22 HGB", "§ 999 HGB"], "hinweis": ""}
    refs = [c for c in _karte(card, node=node).kandidaten if c.typ == "cross_ref"]
    assert len(refs) == 1 and refs[0].bezugsnorm_id == "norm:HGB:22"


def test_karte_leer_erlaubt():
    k = _karte({"anchors": [], "nachbarnormen": [], "advisory": {"hinweis": "kein Anker trägt"}})
    assert k.kandidaten == [] and k.anker_akzeptiert == 0 and k.matcher_terms == []
    assert k.advisory["hinweis"] == "kein Anker trägt"


def test_karte_matcher_terms_nur_aus_wortlaut():
    card = {"anchors": [], "nachbarnormen": [],
            "matcher_terms": ["Besteuerungsgrundlagen", "erfundener Begriff"],
            "advisory": {"risikotyp": "schaetzung", "pruefpunkte": ["x", "  "]}}
    k = _karte(card)
    assert k.matcher_terms == ["Besteuerungsgrundlagen"]   # erfundener Begriff nicht im §
    assert k.advisory["risikotyp"] == "schaetzung" and k.advisory["pruefpunkte"] == ["x"]
