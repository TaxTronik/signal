"""Abnahmekriterien /AB10/–/AB60/ aus dem Lastenheft (Kapitel 9)."""

import json

import pytest

from risk_layer import ingest as ing
from risk_layer.audit import baue_audit
from risk_layer.catalog import Katalog, lade_katalog
from risk_layer.matcher import Schicht1Matcher
from risk_layer.models import Katalogeintrag, GovernanceTyp, Status, Treffer
from risk_layer.pipeline import Pipeline
from risk_layer.resolver import resolve


@pytest.fixture(scope="module")
def katalog():
    return lade_katalog()


@pytest.fixture(scope="module")
def pipeline(katalog):
    return Pipeline(katalog)


# /AB10/ Schicht-1-Recall = 100 % auf dem Katalog ----------------------------

def test_ab10_recall_100_prozent_auf_katalog(katalog):
    """Jeder Katalogbegriff wird in einem Text mit seiner kanonischen Form erkannt."""
    matcher = Schicht1Matcher(katalog)
    fehlend = []
    for eintrag in katalog:
        text = f"Im Sachverhalt ist die {eintrag.begriff_kanonisch} zu prüfen."
        res = ing.ingest(text)
        ids = {t.begriff_id for t in matcher.match(res)}
        if eintrag.id not in ids:
            fehlend.append(eintrag.id)
    assert fehlend == [], f"nicht erkannt: {fehlend}"


def test_ab10_recall_auch_auf_aliasen(katalog):
    """Auch jede hinterlegte Oberflächenvariante (Alias) wird erkannt."""
    matcher = Schicht1Matcher(katalog)
    fehlend = []
    for eintrag in katalog:
        for alias in eintrag.aliase:
            text = f"Stichwort: {alias}."
            res = ing.ingest(text)
            ids = {t.begriff_id for t in matcher.match(res)}
            if eintrag.id not in ids:
                fehlend.append((eintrag.id, alias))
    assert fehlend == [], f"Alias nicht erkannt: {fehlend}"


# /AB20/ Keine erfundenen Schicht-1-Treffer ----------------------------------

def test_ab20_keine_erfundenen_treffer(pipeline, katalog):
    text = (
        "Die organisatorische Eingliederung und der schädliche Beteiligungserwerb "
        "sowie die tatsächliche Durchführung sind zu prüfen."
    )
    report = pipeline.run(text)
    for k in report.karten:
        # jeder Treffer ist über eine existierende Katalog-ID belegt
        assert k.eintrag.id in katalog
        # und steht wörtlich an der angegebenen Stelle
        assert text[k.treffer.start:k.treffer.end] == k.treffer.matched_text


def test_ab20_leerer_text_keine_treffer(pipeline):
    assert pipeline.run("").karten == []


# /AB40/ Definition niemals aus dem Modell -----------------------------------

def test_ab40_fehlende_quelle_erzeugt_luecke_keine_definition(katalog):
    """Bewusste Lücke im Katalog ⇒ Status LÜCKE, KEINE Definition."""
    luecken_eintrag = next(e for e in katalog if not e.hat_definitionsquelle)
    treffer = Treffer(
        begriff_id=luecken_eintrag.id, start=0, end=5,
        matched_text="xxxxx", surface=luecken_eintrag.begriff_kanonisch, segment_index=0,
    )
    karte = resolve(treffer, katalog)
    assert karte.status is Status.LUECKE
    assert karte.quellen == []
    assert karte.rechercheauftrag


def test_ab40_leerer_katalog_erzeugt_nur_luecken():
    """Mit Eintrag ohne Quelle entsteht nie eine Definition – nur Lücken."""
    leer_eintrag = Katalogeintrag(
        id="x", begriff_kanonisch="Testbegriff", aliase=(),
        definitionsquelle=(), governance_typ=GovernanceTyp.IN,
    )
    kat = Katalog(version="t", eintraege=[leer_eintrag])
    treffer = Treffer(
        begriff_id="x", start=0, end=11, matched_text="Testbegriff",
        surface="Testbegriff", segment_index=0,
    )
    karte = resolve(treffer, kat)
    assert karte.status is Status.LUECKE
    assert karte.quellen == []


def test_ab40_quellen_sind_immer_teilmenge_des_katalogs(pipeline):
    """Resolver erfindet keine Quellen – sie stammen ausschließlich aus dem Katalog."""
    report = pipeline.run(open(
        "examples/beispiel_organschaft.txt", encoding="utf-8"
    ).read())
    for k in report.karten:
        erlaubt = set(k.eintrag.definitionsquelle)
        assert set(k.quellen) <= erlaubt


# /AB50/ Reproduzierbarkeit Schicht 1 ----------------------------------------

def test_ab50_byte_identische_reproduzierbarkeit(pipeline):
    text = open("examples/beispiel_organschaft.txt", encoding="utf-8").read()
    ts = "2026-05-31T00:00:00+00:00"
    a = baue_audit(pipeline.run(text, erzeugt_am=ts))
    b = baue_audit(pipeline.run(text, erzeugt_am=ts))
    assert json.dumps(a, ensure_ascii=False, sort_keys=True) == \
        json.dumps(b, ensure_ascii=False, sort_keys=True)


# /AB60/ Audit-Vollständigkeit -----------------------------------------------

def test_ab60_audit_enthaelt_alle_versionsangaben(pipeline):
    report = pipeline.run("Die organisatorische Eingliederung ist zu prüfen.")
    audit = baue_audit(report)
    assert audit["eingabe_hash"]
    assert audit["katalog_version"]
    assert audit["erzeugt_am"]
    assert audit["schicht1"]["engine"]
    assert audit["schicht1"]["deterministisch"] is True
    # Modellangaben für reine Schicht 1 explizit null (belegt: kein Modell beteiligt)
    assert audit["schicht2"]["modell"] is None
    assert "treffer" in audit and len(audit["treffer"]) >= 1
