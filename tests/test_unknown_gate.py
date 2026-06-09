"""Unknown-Gate & Proposer (Erweiterung §3, /LF270/–/LF310/)."""

from risk_layer.models import Status
from risk_layer.unknown_gate import DefinitorischProposer, UnknownGate


def test_negativfilter_veto_named_entity():
    gate = UnknownGate()
    erg = gate.bewerte("Die M-GmbH handelt.", "M-GmbH", 4)
    assert erg.status is Status.VERWORFEN
    assert erg.signale.get("veto") is True


def test_negativfilter_veto_stoppwort():
    gate = UnknownGate()
    erg = gate.bewerte("Das Unternehmen wächst.", "Unternehmen", 4)
    assert erg.status is Status.VERWORFEN


def test_norm_naehe_erhoeht_score():
    gate = UnknownGate()
    text = "Nach § 8c Abs. 1 KStG ist die Konzernklausel zu prüfen."
    off = text.index("Konzernklausel")
    erg = gate.bewerte(text, "Konzernklausel", off)
    assert erg.signale["norm_naehe"] is True
    assert erg.angenommen  # Norm-Nähe + Nominalkompositum über Schwelle


def test_norm_naehe_allein_eskaliert_nicht():
    # Norm-Nähe ist stark gewichtet, liegt aber bewusst KNAPP UNTER der
    # Schwelle: die Signale werden KOMBINIERT – ein §-Zitat im Kontext macht
    # aus einem Alltagsnomen ohne weiteres Signal keinen Rechtsbegriff.
    gate = UnknownGate()
    text = "Nach § 8c KStG steht der Tisch neben dem Fenster im Raum."
    off = text.index("Tisch")
    erg = gate.bewerte(text, "Tisch", off)
    assert erg.signale["norm_naehe"] is True
    assert not erg.angenommen
    assert erg.status is Status.VERWORFEN


def test_proposer_schlaegt_alltagsnomen_neben_normzitat_nicht_vor():
    proposer = DefinitorischProposer()
    text = "Nach § 8c KStG steht der Tisch neben dem Fenster im Raum."
    titel = {r.titel for r in proposer.vorschlagen(text)}
    assert not titel & {"Tisch", "Fenster", "Raum"}, titel


def test_definitorische_form_erhoeht_score():
    gate = UnknownGate()
    text = "Eine Betriebsaufspaltung setzt voraus, dass eine Verflechtung besteht."
    off = text.index("Betriebsaufspaltung")
    erg = gate.bewerte(text, "Betriebsaufspaltung", off)
    assert erg.signale["definitorische_form"] is True
    assert erg.angenommen


def test_belangloses_wort_unter_schwelle():
    gate = UnknownGate()
    text = "Der Tisch steht im Raum."
    erg = gate.bewerte(text, "Tisch", 4)
    assert not erg.angenommen
    assert erg.status is Status.VERWORFEN


def test_llm_signal_ist_additiv():
    gate = UnknownGate(schwelle=0.4)
    text = "Hier steht der Begriff Quotennießbrauch ohne weiteren Kontext."
    off = text.index("Quotennießbrauch")
    ohne = gate.bewerte(text, "Quotennießbrauch", off)
    mit = gate.bewerte(text, "Quotennießbrauch", off, llm_signal=1.0)
    assert mit.score > ohne.score


def test_proposer_findet_nicht_katalogisierten_begriff():
    proposer = DefinitorischProposer()
    text = ("Fraglich ist, ob eine Betriebsaufspaltung vorliegt. Eine "
            "Betriebsaufspaltung setzt voraus, dass eine Verflechtung besteht.")
    risiken = proposer.vorschlagen(text)
    titel = {r.titel for r in risiken}
    assert any("Betriebsaufspaltung" in t for t in titel)
    assert all(r.status is Status.UNKNOWN_RISIKO for r in risiken)


def test_proposer_ueberspringt_belegte_spans():
    proposer = DefinitorischProposer()
    text = "Eine Betriebsaufspaltung setzt voraus, dass etwas gilt."
    s = text.index("Betriebsaufspaltung")
    e = s + len("Betriebsaufspaltung")
    risiken = proposer.vorschlagen(text, belegte_spans=[(s, e)])
    assert all("Betriebsaufspaltung" not in r.matched_text for r in risiken)


def test_proposer_determinismus():
    proposer = DefinitorischProposer()
    text = "Eine Betriebsaufspaltung setzt voraus, dass eine Verflechtung besteht."
    a = [r.as_dict() for r in proposer.vorschlagen(text)]
    b = [r.as_dict() for r in proposer.vorschlagen(text)]
    assert a == b
