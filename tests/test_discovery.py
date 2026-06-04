"""Discovery-Harness (Pass 1): Verarbeitung der LLM-Roh-Kandidaten.

Ohne LLM/Graph/GPU — verarbeite_kandidaten + Gates werden mit Stubs geprüft:
Anti-Halluzination (Textgrundlage muss im § stehen), abgeleitete Authority,
Relations-Erdung, Paraphrase-Gate, Deckel, Review-/Training-Gates.
"""

from risk_layer.discovery import (
    Kandidat, besteht_review_gate, besteht_training_gate, verarbeite_kandidaten,
)

TEXT = ("Wird das Einkommen geschätzt, sind die Besteuerungsgrundlagen zu ermitteln. "
        "Mehr als 50 Prozent der Anteile gelten als schädlich. "
        "Die Verluste sind nicht abziehbar.")
TITEL = "Schätzung"
QUELL = "norm:EStDV:8"          # Verordnungs-§ (für Default-Authority)
LAW = "EStDV"
IM_GRAPH = {"norm:EStDV:8", "norm:EStG:51", "norm:HGB:22"}


def _verarbeite(roh, verweist=(), enable_normrelation=False):
    return verarbeite_kandidaten(
        roh, quell_norm=QUELL, quell_law=LAW, node_text=TEXT, node_titel=TITEL,
        verweist_ids=verweist, ist_im_graph=lambda nid: nid in IM_GRAPH,
        enable_normrelation=enable_normrelation)


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


def test_training_gate():
    assert besteht_training_gate(_k(typ="paraphrase", konfidenz="hoch"))
    assert not besteht_training_gate(_k(konfidenz="mittel"))           # nur hoch
    assert not besteht_training_gate(_k(ziel_im_graph=False))          # muss im Graph sein
    assert not besteht_training_gate(_k(typ="cross_ref"))              # nur paraphrase|begriff
    assert not besteht_training_gate(_k(typ="begriff", funktion="unklar"))
    assert besteht_training_gate(_k(typ="begriff", funktion="tatbestandsmerkmal"))
