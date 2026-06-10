"""Lemma-Robustheitspfad /LF50/ – Offset-Treue, Dedupe, Degradation, Provenienz.

Alle Tests (bis auf den Smoke-Test) injizieren ein FAKE-NLP über den Lade-Seam
``risk_layer.lemma._lade_nlp`` (tests/conftest.py patcht ihn autouse auf None –
kein Test lädt implizit das echte Modell). Der Fake liefert Token-Objekte mit
text/idx/lemma_ – exakt der Seam-Vertrag von ``lemma_sicht``.
"""

from __future__ import annotations

import json
import re
from importlib.util import find_spec

import pytest

from risk_layer import ingest as ing
from risk_layer import lemma as lemma_mod
from risk_layer.catalog import Katalog, lade_katalog
from risk_layer.matcher import Schicht1Matcher, entferne_lemma_dubletten
from risk_layer.models import Definitionsquelle, Katalogeintrag, Treffer
from risk_layer.pipeline import AnalyseOptionen, Pipeline
from risk_layer.web import antwort_json

# VOR jedem Fixture-Patch gegriffen: das echte Loader-Objekt für den Smoke-Test.
_ECHTER_LOADER = lemma_mod._lade_nlp

TS = "2026-01-01T00:00:00+00:00"


# --- Fake-NLP über den Seam ---------------------------------------------------

class _Tok:
    __slots__ = ("text", "idx", "lemma_")

    def __init__(self, text: str, idx: int, lemma: str):
        self.text = text
        self.idx = idx
        self.lemma_ = lemma


class _FakeNLP:
    """Whitespace-/Interpunktions-Tokenizer + Wörterbuch-Lemmatisierer."""

    def __init__(self, lemmata: dict[str, str] | None = None,
                 punkt_lemma: str = "--"):
        self.lemmata = {k.lower(): v for k, v in (lemmata or {}).items()}
        self.punkt_lemma = punkt_lemma   # Artefakt des echten Lemmatizers

    def __call__(self, text: str):
        toks = []
        for m in re.finditer(r"[0-9A-Za-zÄÖÜäöüß\-]+|[^\s]", text):
            wort = m.group(0)
            if re.match(r"[0-9A-Za-zÄÖÜäöüß\-]", wort):
                lemma = self.lemmata.get(wort.lower(), wort)
            else:
                lemma = self.punkt_lemma   # wie spaCy-de: "." → "--"
            toks.append(_Tok(wort, m.start(), lemma))
        return toks


@pytest.fixture
def fake_nlp(monkeypatch):
    """Injiziert ein Fake-NLP über den Seam; Test konfiguriert das Wörterbuch."""
    def _setze(lemmata: dict[str, str]):
        nlp = _FakeNLP(lemmata)
        monkeypatch.setattr(lemma_mod, "_lade_nlp", lambda: nlp)
        return nlp
    return _setze


def _mini_katalog() -> Katalog:
    eintrag = Katalogeintrag(
        id="kst_schaedlicher_beteiligungserwerb",
        begriff_kanonisch="schädlicher Beteiligungserwerb",
        definitionsquelle=(Definitionsquelle(typ="Gesetz", fundstelle="§ 8c KStG"),),
    )
    return Katalog(version="test", eintraege=[eintrag])


# --- LemmaSicht: Offset-Treue -------------------------------------------------

def test_lemma_sicht_offsets_exakt_inkl_umlaute_und_textende(fake_nlp):
    fake_nlp({"Gewinnausschüttungen": "Gewinnausschüttung", "Die": "der",
              "prüften": "prüfen"})
    text = "Die Gewinnausschüttungen prüften"
    sicht = lemma_mod.lemma_sicht(text)
    assert sicht is not None
    assert sicht.lemma_text == "der Gewinnausschüttung prüfen"
    # Span von "Gewinnausschüttung" im lemma_text → Originalspan der flektierten Form
    ls = sicht.lemma_text.index("Gewinnausschüttung")
    os_, oe = sicht.rueckabbildung(ls, ls + len("Gewinnausschüttung"))
    assert text[os_:oe] == "Gewinnausschüttungen"
    # Token am Textende: Originalspan endet exakt bei len(text)
    le = sicht.lemma_text.index("prüfen")
    os2, oe2 = sicht.rueckabbildung(le, le + len("prüfen"))
    assert (text[os2:oe2], oe2) == ("prüften", len(text))


def test_rueckabbildung_mehrwort_umschliesst_alle_tokens(fake_nlp):
    fake_nlp({"verdeckten": "verdeckt", "Gewinnausschüttungen": "Gewinnausschüttung"})
    text = "Wegen  der verdeckten   Gewinnausschüttungen!"
    sicht = lemma_mod.lemma_sicht(text)
    ls = sicht.lemma_text.index("verdeckt Gewinnausschüttung")
    os_, oe = sicht.rueckabbildung(ls, ls + len("verdeckt Gewinnausschüttung"))
    # Mehrfach-Leerzeichen im Original: umschließende Span vom ersten bis letzten Token
    assert text[os_:oe] == "verdeckten   Gewinnausschüttungen"


def test_punktuations_artefakt_wird_nicht_uebernommen(fake_nlp):
    # der echte de-Lemmatizer liefert für Interpunktion "--" – das Artefakt darf
    # den lemma_text nicht verfälschen (Token behält seinen Originaltext).
    fake_nlp({})
    sicht = lemma_mod.lemma_sicht("Kasse.")
    assert sicht.lemma_text == "Kasse ."


def test_rueckabbildung_ausserhalb_wirft(fake_nlp):
    fake_nlp({})
    sicht = lemma_mod.lemma_sicht("Wort")
    with pytest.raises(ValueError):
        sicht.rueckabbildung(10, 12)


# --- Matcher: Lemma-Pfad ------------------------------------------------------

def test_match_lemma_findet_flektierte_form_offset_treu(fake_nlp):
    fake_nlp({"schädlichsten": "schädlich", "schädlicher": "schädlich",
              "Beteiligungserwerbe": "Beteiligungserwerb"})
    matcher = Schicht1Matcher(_mini_katalog())
    text = "Die schädlichsten Beteiligungserwerbe erfolgten unterjährig."
    res = ing.ingest(text)
    assert matcher.match(res) == []          # heute: kein exakter Treffer
    treffer = matcher.match_lemma(res)
    assert [t.begriff_id for t in treffer] == ["kst_schaedlicher_beteiligungserwerb"]
    t = treffer[0]
    assert t.via_lemma is True
    assert t.surface == "schädlicher Beteiligungserwerb"   # Regelbeleg = Katalogform
    # Offset-Treue ist heilig: die Marke zeigt wörtlich auf den ORIGINALTEXT
    assert text[t.start:t.end] == t.matched_text == "schädlichsten Beteiligungserwerbe"


def test_match_lemma_ohne_nlp_ist_noop():
    # conftest._kein_lemma patcht den Loader auf None → Degradation: leere Liste
    matcher = Schicht1Matcher(_mini_katalog())
    res = ing.ingest("Die schädlichsten Beteiligungserwerbe erfolgten unterjährig.")
    assert matcher.match_lemma(res) == []


def test_match_lemma_respektiert_env_schalter(fake_nlp, monkeypatch):
    fake_nlp({"Beteiligungserwerbe": "Beteiligungserwerb", "schädlichsten": "schädlich"})
    monkeypatch.setenv("RISK_LAYER_LEMMA", "aus")
    matcher = Schicht1Matcher(_mini_katalog())
    res = ing.ingest("Die schädlichsten Beteiligungserwerbe erfolgten unterjährig.")
    assert matcher.match_lemma(res) == []


def test_match_bestand_bleibt_unveraendert_mit_fake_nlp(fake_nlp):
    # match() ruft den Lemma-Pfad NIE: gleiches Ergebnis mit/ohne injiziertem NLP
    matcher = Schicht1Matcher(_mini_katalog())
    res = ing.ingest("Ein schädlicher Beteiligungserwerb liegt vor.")
    vorher = [t.as_dict() for t in matcher.match(res)]
    fake_nlp({"egal": "egal"})
    nachher = [t.as_dict() for t in matcher.match(res)]
    assert vorher == nachher and vorher
    assert all(d["via_lemma"] is False for d in vorher)


# --- Dedupe: exakt vor Lemma --------------------------------------------------

def _t(bid: str, start: int, end: int, **kw) -> Treffer:
    return Treffer(begriff_id=bid, start=start, end=end,
                   matched_text="x" * (end - start), surface="s",
                   segment_index=0, **kw)


def test_dedupe_exakter_treffer_gewinnt_bei_ueberlappung():
    exakt = [_t("a", 0, 10)]
    lemma_treffer = [_t("a", 5, 15, via_lemma=True),     # überlappt gleiche ID → weg
                     _t("a", 20, 30, via_lemma=True),    # disjunkt → bleibt
                     _t("b", 5, 15, via_lemma=True)]     # andere ID → bleibt
    rest = entferne_lemma_dubletten(exakt, lemma_treffer)
    assert [(t.begriff_id, t.start, t.end) for t in rest] == [("a", 20, 30), ("b", 5, 15)]


# --- Pipeline: Verträge -------------------------------------------------------

def test_pipeline_lemma_false_ist_byte_identisch_und_laedt_nie(monkeypatch):
    # Der Seam darf bei lemma=False NICHT einmal berührt werden (Determinismus).
    def _explodiert():
        raise AssertionError("Lemma-Loader darf bei lemma=False nie laufen")
    monkeypatch.setattr(lemma_mod, "_lade_nlp", _explodiert)
    pipeline = Pipeline(lade_katalog())
    text = "Die organisatorische Eingliederung ist zu prüfen."
    alt = pipeline.run(text, optionen=AnalyseOptionen(erzeugt_am=TS))
    neu = pipeline.run(text, optionen=AnalyseOptionen(erzeugt_am=TS, lemma=False))
    assert json.dumps(antwort_json(alt, text), ensure_ascii=False, sort_keys=True) == \
        json.dumps(antwort_json(neu, text), ensure_ascii=False, sort_keys=True)
    assert alt.karten


def test_pipeline_degradation_ohne_spacy_wie_heute():
    # Loader → None (conftest): lemma=True degradiert zum byte-identischen Lauf
    pipeline = Pipeline(lade_katalog())
    text = "Die Tochtergesellschaft ist finanziell einzugliedern."
    ohne = pipeline.run(text, optionen=AnalyseOptionen(erzeugt_am=TS))
    mit = pipeline.run(text, optionen=AnalyseOptionen(erzeugt_am=TS, lemma=True))
    assert json.dumps(antwort_json(ohne, text), ensure_ascii=False, sort_keys=True) == \
        json.dumps(antwort_json(mit, text), ensure_ascii=False, sort_keys=True)


def test_pipeline_lemma_treffer_mit_ehrlicher_provenienz(fake_nlp):
    fake_nlp({"Gestaltungsmissbräuche": "Gestaltungsmissbrauch"})
    pipeline = Pipeline(lade_katalog())
    text = "Solche Gestaltungsmissbräuche erkennt das Finanzamt nicht an."
    rep = pipeline.run(text, optionen=AnalyseOptionen(erzeugt_am=TS, lemma=True))
    karte = next(k for k in rep.karten if k.eintrag.id == "ao_gestaltungsmissbrauch")
    assert karte.via == "lemma"
    assert karte.treffer.via_lemma is True
    assert text[karte.treffer.start:karte.treffer.end] == karte.treffer.matched_text \
        == "Gestaltungsmissbräuche"
    h = karte.herkunft()
    # Provenienz ehrlich: der Lemma-Pfad ist in methode/kurz sichtbar (/LF50/)
    assert h["methode"] == "katalog_lemma"
    assert "Lemma" in h["kurz"]
    assert h["schicht"] == "1" and h["interpretation"] is False
    # auch im JSON-Vertrag sichtbar (karten[].via / herkunft)
    res = antwort_json(rep, text)
    k_json = next(k for k in res["karten"] if k["begriff_id"] == "ao_gestaltungsmissbrauch")
    assert k_json["via"] == "lemma" and k_json["herkunft"]["methode"] == "katalog_lemma"


def test_pipeline_dedupe_exakt_vor_lemma(fake_nlp):
    # Identitäts-Lemmata: der Lemma-Pfad würde den exakten Treffer ERNEUT finden –
    # der Dedupe lässt den wörtlichen Beleg gewinnen (keine Doppel-Karte).
    fake_nlp({})
    pipeline = Pipeline(lade_katalog())
    text = "Ein schädlicher Beteiligungserwerb liegt vor."
    rep = pipeline.run(text, optionen=AnalyseOptionen(erzeugt_am=TS, lemma=True))
    karten = [k for k in rep.karten if k.eintrag.id == "kst_schaedlicher_beteiligungserwerb"]
    assert len(karten) == 1
    assert karten[0].via == "wörtlich" and karten[0].treffer.via_lemma is False


# --- Smoke: echtes spaCy (nur wenn installiert) -------------------------------

@pytest.mark.skipif(find_spec("de_core_news_lg") is None,
                    reason="de_core_news_lg nicht installiert")
def test_smoke_echtes_spacy_findet_flektierte_form(monkeypatch):
    monkeypatch.setattr(lemma_mod, "_lade_nlp", _ECHTER_LOADER)
    pipeline = Pipeline(lade_katalog())
    # zu-Infinitiv: heute exakt unauffindbar („durchgeführt" ≠ „durchzuführen")
    text = "Der Gewinnabführungsvertrag ist tatsächlich durchzuführen."
    ohne = pipeline.run(text, optionen=AnalyseOptionen(erzeugt_am=TS))
    assert all(k.eintrag.id != "kst_tatsaechliche_durchfuehrung" for k in ohne.karten)
    mit = pipeline.run(text, optionen=AnalyseOptionen(erzeugt_am=TS, lemma=True))
    karte = next(k for k in mit.karten if k.eintrag.id == "kst_tatsaechliche_durchfuehrung")
    assert karte.via == "lemma"
    assert text[karte.treffer.start:karte.treffer.end] == "tatsächlich durchzuführen"
