"""Eval-Harness (tools/eval_harness.py) — Kernlogik LLM-/GPU-frei.

Getestet wird die PURE Mess-Logik mit dem abhängigkeitsfreien HashEncoder
(Mini-Index wie tests/test_embedding.py) und Inline-Mini-Gold: Treffer-Logik
inkl. Präfix-Falle, handverifizierte Recall/MRR-Berechnung, die Äquivalenz des
Floor-Nachfilters zur Live-Suche und das Berichts-Schema. Dazu die
Experiment-Erweiterungen: RRF-Fusion (handnachgerechnete Mathematik),
Rerank-Abstention-Policies P0–P3 und die Negativ-Gold-Präzisionsmetrik
(Fehlvorschlagsquote). KEIN Test lädt echte Modelle — der Harness lädt den
Index direkt über ``EmbeddingIndex.load`` (der conftest-Patch auf
``index_oeffnen`` greift hier gar nicht erst).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_layer.embedding import EmbeddingIndex, HashEncoder
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph

from tools.eval_harness import (baue_bericht, floor_filter, fusioniere_treffer,
                                ist_treffer, konfig_kandidaten, lade_gold,
                                lade_negativ_gold, metriken, negativ_metriken,
                                rrf_scores, sammle_rohdaten, sweep,
                                treffer_rang)


def _graph(tmp_path):
    """Mini-Normgraph; enthält BEWUSST §14 UND §140 (Präfix-Falle auf ID-Ebene)."""
    knoten = [
        NormKnoten(id="norm:AO:146", law="AO", para="146", absatz=None, nummer=None,
                   titel="Ordnungsvorschriften für die Buchführung und für Aufzeichnungen",
                   text="Bücher und Aufzeichnungen sind vollständig und richtig zu führen."),
        NormKnoten(id="norm:AO:162", law="AO", para="162", absatz=None, nummer=None,
                   titel="Schätzung von Besteuerungsgrundlagen",
                   text="Soweit die Finanzbehörde die Besteuerungsgrundlagen nicht ermitteln kann, schätzt sie."),
        NormKnoten(id="norm:AO:42", law="AO", para="42", absatz=None, nummer=None,
                   titel="Missbrauch von rechtlichen Gestaltungsmöglichkeiten",
                   text="Durch Missbrauch von Gestaltungsmöglichkeiten kann das Gesetz nicht umgangen werden."),
        NormKnoten(id="norm:AO:14", law="AO", para="14", absatz=None, nummer=None,
                   titel="Wirtschaftlicher Geschäftsbetrieb",
                   text="Ein wirtschaftlicher Geschäftsbetrieb ist eine selbständige nachhaltige Tätigkeit."),
        NormKnoten(id="norm:AO:140", law="AO", para="140", absatz=None, nummer=None,
                   titel="Buchführungs- und Aufzeichnungspflichten nach anderen Gesetzen",
                   text="Wer nach anderen Gesetzen Bücher zu führen hat, hat diese Pflichten auch steuerlich zu erfüllen."),
    ]
    return NormGraph.build(knoten, tmp_path / "g.sqlite")


def _index(tmp_path, dim=512):
    enc = HashEncoder(dim=dim)
    return EmbeddingIndex.build(_graph(tmp_path), enc), enc


# --- Treffer-Logik: Pfadsegment-genauer Präfix ---------------------------------

def test_ist_treffer_exakt_und_tiefer():
    assert ist_treffer("norm:AO:42", "norm:AO:42")
    assert ist_treffer("norm:AO:42:abs1", "norm:AO:42")          # Ziel als Präfix
    assert ist_treffer("norm:AO:42:abs1:nr2", "norm:AO:42")
    assert not ist_treffer("norm:AO:69", "norm:AO:42")


def test_ist_treffer_praefix_falle_14_vs_140():
    # DIE Falle: naives startswith ließe norm:AO:14 auf norm:AO:140 feuern.
    assert ist_treffer("norm:AO:14", "norm:AO:14")
    assert ist_treffer("norm:AO:14:abs1", "norm:AO:14")
    assert not ist_treffer("norm:AO:140", "norm:AO:14")           # Segmentgrenze!
    assert not ist_treffer("norm:AO:140:abs1", "norm:AO:14")
    # Umgekehrt trifft das tiefe Ziel den flachen Kandidaten NICHT (Top-r-Logik:
    # nur Ziel-als-Präfix ist spezifiziert).
    assert not ist_treffer("norm:AO:14", "norm:AO:14:abs1")


def test_treffer_rang_erste_zielnorm_zaehlt():
    kandidaten = ["norm:AO:140", "norm:EStG:4h", "norm:AO:14:abs1", "norm:AO:14"]
    assert treffer_rang(kandidaten, {"norm:AO:14"}) == 3          # 140 zählt NICHT
    assert treffer_rang(kandidaten, {"norm:AO:140"}) == 1
    assert treffer_rang(kandidaten, {"norm:KStG:14"}) is None
    assert treffer_rang([], {"norm:AO:14"}) is None


# --- Metriken: handverifiziert ---------------------------------------------------

def test_metriken_handverifiziert():
    # 5 Fälle: Ränge 1, 3, 7, verfehlt-mit-Kandidaten, Abstention (= verfehlt).
    ergebnisse = [
        {"rang": 1, "abstention": False, "latenz_ms": 10.0},
        {"rang": 3, "abstention": False, "latenz_ms": 20.0},
        {"rang": 7, "abstention": False, "latenz_ms": 30.0},
        {"rang": None, "abstention": False, "latenz_ms": 40.0},
        {"rang": None, "abstention": True, "latenz_ms": 50.0},
    ]
    m = metriken(ergebnisse)
    assert m["n"] == 5
    assert m["recall@1"] == pytest.approx(1 / 5)
    assert m["recall@5"] == pytest.approx(2 / 5)
    assert m["recall@10"] == pytest.approx(3 / 5)
    assert m["recall@50"] == pytest.approx(3 / 5)
    # MRR von Hand: (1/1 + 1/3 + 1/7 + 0 + 0) / 5 — gerundet auf 4 Stellen.
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 7) / 5, abs=5e-5)
    assert m["abstention"] == pytest.approx(1 / 5)
    assert m["median_latenz_ms"] == 30.0
    assert metriken([]) == {"n": 0}


# --- Floor-Nachfilter: Äquivalenz zur Live-Suche ---------------------------------

def test_floor_nachfilter_aequivalent_zur_live_suche(tmp_path):
    idx, enc = _index(tmp_path)
    queries = ["Schätzung der Besteuerungsgrundlagen durch die Finanzbehörde",
               "Bücher und Aufzeichnungen vollständig führen",
               "Missbrauch von Gestaltungsmöglichkeiten"]
    for q in queries:
        roh = idx.suche(q, enc, k=10, min_sim=0.0)
        for floor in (0.05, 0.2, 0.4, 0.9):
            live = idx.suche(q, enc, k=10, min_sim=floor)
            assert floor_filter(roh, floor) == live, (q, floor)


def test_floor_filter_schneidet_nur_den_schwanz():
    roh = [{"norm_id": "a", "sim": 0.9}, {"norm_id": "b", "sim": 0.5},
           {"norm_id": "c", "sim": 0.5}, {"norm_id": "d", "sim": 0.1}]
    assert [t["norm_id"] for t in floor_filter(roh, 0.5)] == ["a", "b", "c"]
    assert floor_filter(roh, 0.95) == []
    assert floor_filter(roh, 0.0) == roh


# --- Konfigurations-Pipeline (Floor → Rerank → Schwelle → Top-k) ------------------

def test_konfig_kandidaten_ohne_rerank_topk_und_floor():
    roh = [{"norm_id": f"norm:AO:{i}", "sim": 1.0 - i / 10} for i in range(1, 8)]
    assert konfig_kandidaten(roh, floor=0.0, topk=3) == \
        ["norm:AO:1", "norm:AO:2", "norm:AO:3"]
    assert konfig_kandidaten(roh, floor=0.65, topk=10) == \
        ["norm:AO:1", "norm:AO:2", "norm:AO:3"]                    # Floor schneidet


def test_konfig_kandidaten_rerank_sortiert_schwellt_und_kappt():
    q = "frage"
    roh = [{"norm_id": "norm:AO:1", "sim": 0.9},
           {"norm_id": "norm:AO:2", "sim": 0.8},
           {"norm_id": "norm:AO:3", "sim": 0.7},
           {"norm_id": "norm:AO:4", "sim": 0.6}]
    scores = {(q, "norm:AO:1"): 0.02, (q, "norm:AO:2"): 0.9,
              (q, "norm:AO:3"): 0.4, (q, "norm:AO:4"): 0.4}
    # Umsortierung nach Score; Gleichstand (3 vs. 4) stabil in Embedding-Reihenfolge;
    # Schwelle 0.05 wirft norm:AO:1 raus (Abstention-Mechanik).
    assert konfig_kandidaten(roh, floor=0.0, topk=10, query=q,
                             rerank_scores=scores, schwelle=0.05) == \
        ["norm:AO:2", "norm:AO:3", "norm:AO:4"]
    # rerank_topn kappt VOR dem Reranking (norm:AO:4 nie gescort/gesehen).
    assert konfig_kandidaten(roh, floor=0.0, topk=10, query=q,
                             rerank_scores=scores, schwelle=0.05, rerank_topn=2) == \
        ["norm:AO:2"]
    # Alle unter Schwelle ⇒ leer (= Abstention des Falls).
    assert konfig_kandidaten(roh, floor=0.0, topk=10, query=q,
                             rerank_scores=scores, schwelle=0.95) == []
    with pytest.raises(ValueError):
        konfig_kandidaten(roh, floor=0.0, topk=5, query=q,
                          rerank_scores=scores, schwelle=None)


# --- Rerank-Policies P1/P2/P3 (Experiment 2) ---------------------------------------

_ROH4 = [{"norm_id": "norm:AO:1", "sim": 0.9},
         {"norm_id": "norm:AO:2", "sim": 0.8},
         {"norm_id": "norm:AO:3", "sim": 0.56},
         {"norm_id": "norm:AO:4", "sim": 0.4}]
_SCORES4 = {("q", "norm:AO:1"): 0.02,   # sim-stärkster, aber rerank UNTER Schwelle
            ("q", "norm:AO:2"): 0.9,
            ("q", "norm:AO:3"): 0.04,   # unter Schwelle, sim über hoch=0.55
            ("q", "norm:AO:4"): 0.4}


def test_policy_p1_ordering_only_keine_abstention():
    # P1: rerank sortiert NUR; niemand fliegt (Abstention rein über den Floor).
    assert konfig_kandidaten(_ROH4, floor=0.0, topk=10, query="q",
                             rerank_scores=_SCORES4, policy="P1") == \
        ["norm:AO:2", "norm:AO:4", "norm:AO:3", "norm:AO:1"]
    # Floor wirkt weiterhin davor (0.5 wirft norm:AO:4 raus).
    assert konfig_kandidaten(_ROH4, floor=0.5, topk=10, query="q",
                             rerank_scores=_SCORES4, policy="P1") == \
        ["norm:AO:2", "norm:AO:3", "norm:AO:1"]


def test_policy_p2_oder_abstention_sim_rettet():
    # P2: bleibt, wenn rerank>=0.10 ODER sim>=0.55 — norm:AO:1 (sim 0.9) und
    # norm:AO:3 (sim 0.56) überleben trotz Mini-Rerank; norm:AO:4 (0.4/0.4) bleibt
    # über die Schwelle; bei hoch=0.95 fallen die sim-Geretteten wieder.
    assert konfig_kandidaten(_ROH4, floor=0.0, topk=10, query="q",
                             rerank_scores=_SCORES4, schwelle=0.10,
                             policy="P2", hoch=0.55) == \
        ["norm:AO:2", "norm:AO:4", "norm:AO:3", "norm:AO:1"]
    assert konfig_kandidaten(_ROH4, floor=0.0, topk=10, query="q",
                             rerank_scores=_SCORES4, schwelle=0.10,
                             policy="P2", hoch=0.95) == \
        ["norm:AO:2", "norm:AO:4"]
    with pytest.raises(ValueError):                  # P2 braucht hoch
        konfig_kandidaten(_ROH4, floor=0.0, topk=5, query="q",
                          rerank_scores=_SCORES4, schwelle=0.10, policy="P2")


def test_policy_p3_top1_rescue_nur_der_sim_staerkste():
    # P3: wie P0, aber der sim-STÄRKSTE (norm:AO:1, sim 0.9>=hoch) bleibt immer —
    # norm:AO:3 (auch unter Schwelle, sim 0.56) wird NICHT gerettet (kein Top-1).
    assert konfig_kandidaten(_ROH4, floor=0.0, topk=10, query="q",
                             rerank_scores=_SCORES4, schwelle=0.10,
                             policy="P3", hoch=0.55) == \
        ["norm:AO:2", "norm:AO:4", "norm:AO:1"]
    # hoch über dem Top-sim ⇒ exakt P0-Verhalten.
    assert konfig_kandidaten(_ROH4, floor=0.0, topk=10, query="q",
                             rerank_scores=_SCORES4, schwelle=0.10,
                             policy="P3", hoch=0.95) == \
        konfig_kandidaten(_ROH4, floor=0.0, topk=10, query="q",
                          rerank_scores=_SCORES4, schwelle=0.10)
    with pytest.raises(ValueError):                  # unbekannte Policy
        konfig_kandidaten(_ROH4, floor=0.0, topk=5, query="q",
                          rerank_scores=_SCORES4, schwelle=0.10, policy="P9")


# --- RRF-Fusion (Experiment 3) ------------------------------------------------------

def test_rrf_scores_handverifiziert():
    # Listen ["a","b","c"] und ["b","d"], k=60:
    #   a=1/61, b=1/62+1/61, c=1/63, d=1/62 — b > a > d > c.
    s = rrf_scores([["a", "b", "c"], ["b", "d"]], k_rrf=60)
    assert s["a"] == pytest.approx(1 / 61)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["c"] == pytest.approx(1 / 63)
    assert s["d"] == pytest.approx(1 / 62)
    # k=20 gewichtet Spitzenränge stärker (1/21 statt 1/61) — gleiche Mathematik.
    s20 = rrf_scores([["a", "b", "c"], ["b", "d"]], k_rrf=20)
    assert s20["b"] == pytest.approx(1 / 22 + 1 / 21)
    assert rrf_scores([], k_rrf=60) == {}


def test_fusioniere_treffer_ordnung_sim_und_tiebreak():
    dense = [{"norm_id": "a", "sim": 0.8}, {"norm_id": "b", "sim": 0.7},
             {"norm_id": "c", "sim": 0.6}]
    fused = fusioniere_treffer(dense, ["b", "d"], k_rrf=60)
    assert [t["norm_id"] for t in fused] == ["b", "a", "d", "c"]
    # Dense-sim bleibt erhalten; Trigger-only-Kandidat trägt sim=0.0.
    by_id = {t["norm_id"]: t for t in fused}
    assert by_id["a"]["sim"] == 0.8 and by_id["d"]["sim"] == 0.0
    assert by_id["b"]["rrf"] == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    # Gleichstand (beide Rang 1 in je einer Liste) ⇒ deterministisch per norm_id.
    fused2 = fusioniere_treffer([{"norm_id": "y", "sim": 0.9}], ["x"], k_rrf=60)
    assert [t["norm_id"] for t in fused2] == ["x", "y"]


def test_konfig_kandidaten_fusion_mit_und_ohne_rerank():
    dense = [{"norm_id": "norm:AO:1", "sim": 0.9},
             {"norm_id": "norm:AO:2", "sim": 0.8}]
    # fusion-ohne-rerank: RRF-Topliste direkt (Trigger-Treffer rückt rein).
    assert konfig_kandidaten(dense, floor=0.0, topk=3, trigger_ids=["norm:AO:9"],
                             k_rrf=60) == \
        ["norm:AO:1", "norm:AO:9", "norm:AO:2"]
    # fusion-vor-rerank: die FUSIONIERTE Topliste geht in den Reranker; der
    # Trigger-only-Kandidat (sim=0.0) kann per Score gewinnen oder rausfallen.
    scores = {("q", "norm:AO:1"): 0.2, ("q", "norm:AO:2"): 0.05,
              ("q", "norm:AO:9"): 0.8}
    assert konfig_kandidaten(dense, floor=0.0, topk=3, query="q",
                             rerank_scores=scores, schwelle=0.10,
                             trigger_ids=["norm:AO:9"], k_rrf=60) == \
        ["norm:AO:9", "norm:AO:1"]
    # P2 rettet den Trigger-only-Kandidaten NICHT über sim (sim=0.0 < hoch).
    scores2 = {**scores, ("q", "norm:AO:9"): 0.01}
    assert konfig_kandidaten(dense, floor=0.0, topk=3, query="q",
                             rerank_scores=scores2, schwelle=0.10,
                             policy="P2", hoch=0.55,
                             trigger_ids=["norm:AO:9"], k_rrf=60) == \
        ["norm:AO:1", "norm:AO:2"]


# --- Negativ-Gold + Fehlvorschlagsquote (Experiment 1) ------------------------------

def test_negativ_metriken_handverifiziert():
    erg = [{"kategorie": "alltag", "n_kandidaten": 0},
           {"kategorie": "alltag", "n_kandidaten": 3},
           {"kategorie": "steuer_neutral", "n_kandidaten": 5},
           {"kategorie": "steuer_neutral", "n_kandidaten": 0}]
    m = negativ_metriken(erg)
    assert m["n"] == 4
    assert m["fehlvorschlagsquote"] == pytest.approx(2 / 4)
    assert m["mittlere_kandidaten"] == pytest.approx(8 / 4)
    assert m["pro_kategorie"] == {"alltag": 0.5, "steuer_neutral": 0.5}
    assert negativ_metriken([]) == {"n": 0}
    # Volle Abstention = perfekte Präzision.
    perfekt = negativ_metriken([{"kategorie": "alltag", "n_kandidaten": 0}])
    assert perfekt["fehlvorschlagsquote"] == 0.0


def test_lade_negativ_gold_format(tmp_path):
    p = tmp_path / "neg.yaml"
    p.write_text(
        "gold_version: v1\nstand: '2026-06-10'\nfaelle:\n"
        "- query: Vielen Dank für das Gespräch.\n  kategorie: alltag\n"
        "- query: Die Voranmeldung wurde übermittelt.\n  kategorie: steuer_neutral\n"
        "- query: ''\n  kategorie: alltag\n"          # leer → fliegt raus
        "- query: Ohne Kategorie geht auch.\n",       # → 'unkategorisiert'
        encoding="utf-8")
    g = lade_negativ_gold(p)
    assert g["name"] == "neg.yaml" and g["version"] == "v1"
    assert g["faelle"] == [
        ("Vielen Dank für das Gespräch.", "alltag"),
        ("Die Voranmeldung wurde übermittelt.", "steuer_neutral"),
        ("Ohne Kategorie geht auch.", "unkategorisiert")]
    assert len(lade_negativ_gold(p, max_faelle=1)["faelle"]) == 1
    leer = tmp_path / "leer.yaml"
    leer.write_text("faelle: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lade_negativ_gold(leer)


def test_negativ_gold_fixture_eingefroren():
    """Das echte Negativ-Gold v1: 25–35 Sätze, keine Duplikate, nur die drei
    definierten Kategorien — Schutz gegen versehentliches Zerlegen des Golds."""
    pfad = Path(__file__).resolve().parent.parent / "fixtures/gold/negativ_v1.yaml"
    g = lade_negativ_gold(pfad)
    assert g["version"] == "v1"
    queries = [q for q, _ in g["faelle"]]
    assert 25 <= len(queries) <= 35
    assert len(set(queries)) == len(queries)
    assert {k for _, k in g["faelle"]} == \
        {"alltag", "juristisch_neutral", "steuer_neutral"}


# --- Gold-Lader --------------------------------------------------------------------

def test_lade_gold_format_und_max_faelle(tmp_path):
    p = tmp_path / "mini.yaml"
    p.write_text(
        "gold_version: v9\nstand: '2026-01-01'\nfaelle:\n"
        "- query: Frage eins\n  ziele: [norm:AO:42]\n"
        "- query: Frage zwei\n  ziele: [norm:AO:14, norm:AO:162]\n"
        "- query: ''\n  ziele: [norm:AO:1]\n"      # leer → fliegt raus
        "- query: ohne Ziele\n  ziele: []\n",      # ohne Ziele → fliegt raus
        encoding="utf-8")
    g = lade_gold(p)
    assert g["name"] == "mini.yaml" and g["version"] == "v9"
    assert [(q, sorted(z)) for q, z in g["faelle"]] == [
        ("Frage eins", ["norm:AO:42"]),
        ("Frage zwei", ["norm:AO:14", "norm:AO:162"])]
    assert len(lade_gold(p, max_faelle=1)["faelle"]) == 1
    leer = tmp_path / "leer.yaml"
    leer.write_text("faelle: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lade_gold(leer)


# --- Sweep + Berichts-Schema (Ende-zu-Ende über den HashEncoder-Mini-Index) ---------

def _mini_lauf(tmp_path):
    idx, enc = _index(tmp_path)
    # Inline-Mini-Gold: Wortlaut-nahe Queries (HashEncoder ist lexisch) + ein
    # bewusst nicht auffindbarer Fall (Abstention-/Verfehlt-Pfad).
    faelle = [
        ("mini_a.yaml", "Schätzung von Besteuerungsgrundlagen", {"norm:AO:162"}),
        ("mini_a.yaml", "Missbrauch von Gestaltungsmöglichkeiten", {"norm:AO:42"}),
        ("mini_b.yaml", "Quantenphysik im Weltall", {"norm:KStG:14"}),
    ]
    roh = sammle_rohdaten(idx, enc, [q for _, q, _ in faelle], k_roh=10)
    ergebnisse = sweep(faelle, roh, topk_liste=[5], floor_liste=[0.0, 0.3, 0.99],
                       rerank_modi=["aus"], schwellen_liste=[0.1])
    return idx, enc, faelle, roh, ergebnisse


def test_sweep_sortiert_beste_konfiguration_zuerst(tmp_path):
    _, _, _, _, ergebnisse = _mini_lauf(tmp_path)
    assert len(ergebnisse) == 3                       # 1 topk × 3 floors × 1 modus
    r5 = [e["aggregat"]["recall@5"] for e in ergebnisse]
    assert r5 == sorted(r5, reverse=True)             # beste zuerst (R@5, dann MRR)
    # Floor 0.99 lässt beim HashEncoder nichts durch ⇒ volle Abstention, 0 Recall.
    schlechteste = ergebnisse[-1]
    assert schlechteste["params"]["floor"] == 0.99
    assert schlechteste["aggregat"]["abstention"] == 1.0
    assert schlechteste["aggregat"]["recall@5"] == 0.0
    # Bei Floor 0.0 treffen beide Wortlaut-Fälle auf Rang 1 (handverifiziert):
    beste = ergebnisse[0]
    assert beste["aggregat"]["recall@1"] == pytest.approx(2 / 3, abs=5e-5)
    assert beste["aggregat"]["mrr"] == pytest.approx(2 / 3, abs=5e-5)


def test_sweep_rerank_braucht_scores(tmp_path):
    _, _, faelle, roh, _ = _mini_lauf(tmp_path)
    with pytest.raises(ValueError):
        sweep(faelle, roh, topk_liste=[5], floor_liste=[0.0],
              rerank_modi=["an"], schwellen_liste=[0.1], rerank_scores=None)


def test_bericht_schema_vollstaendig_und_serialisierbar(tmp_path):
    idx, enc, faelle, roh, ergebnisse = _mini_lauf(tmp_path)
    gold_dateien = [
        {"name": "mini_a.yaml", "version": "v1", "stand": "2026-01-01",
         "faelle": [(q, z) for n, q, z in faelle if n == "mini_a.yaml"]},
        {"name": "mini_b.yaml", "version": "v2", "stand": "2026-01-02",
         "faelle": [(q, z) for n, q, z in faelle if n == "mini_b.yaml"]},
    ]
    bericht = baue_bericht(
        konfiguration={"floor_liste": [0.0, 0.3, 0.99], "rerank": "aus"},
        gold_dateien=gold_dateien, encoder_name=enc.name, index=idx,
        reranker_name=None, ergebnisse=ergebnisse,
        zeitstempel="2026-06-10T12:00:00")
    # Pflicht-Felder des Berichts-Vertrags (Konfiguration + Gold-Versionen +
    # Encoder + Index-Fingerprint + Zeitstempel):
    assert bericht["harness"] == "eval_harness/2"
    assert bericht["erstellt_am"] == "2026-06-10T12:00:00"
    assert bericht["encoder"] == enc.name and bericht["reranker"] is None
    assert bericht["gold"]["mini_a.yaml"] == \
        {"version": "v1", "stand": "2026-01-01", "n": 2}
    assert bericht["negativ_gold"] is None            # ohne --negativ-gold
    assert bericht["index"]["n_vektoren"] == len(idx.ids)
    assert bericht["index"]["graph_fingerprint"] == idx.graph_fingerprint
    assert bericht["n_faelle"] == 3
    # Ist-Default-Auszug: Liste der markierten Zeilen (hier keine — 0,35 fehlt im Sweep).
    assert isinstance(bericht["ist_default"], list)
    assert bericht["ist_default"] == [e for e in bericht["ergebnisse"] if e["ist_default"]]
    # Jedes Sweep-Ergebnis trägt params + aggregat + pro_gold mit allen Metriken.
    for e in bericht["ergebnisse"]:
        assert set(e) == {"params", "ist_default", "aggregat", "pro_gold", "negativ"}
        assert set(e["params"]) == {"fusion", "k_rrf", "rerank", "policy",
                                    "floor", "schwelle", "hoch", "topk",
                                    "rerank_topn"}
        assert e["negativ"] is None                   # ohne Negativ-Gold
        for m in [e["aggregat"], *e["pro_gold"].values()]:
            assert set(m) == {"n", "recall@1", "recall@5", "recall@10", "recall@50",
                              "mrr", "abstention", "median_latenz_ms"}
        assert set(e["pro_gold"]) == {"mini_a.yaml", "mini_b.yaml"}
    # Bericht muss verlustfrei als JSON persistierbar sein (atomar_schreiben-Pfad).
    rund = json.loads(json.dumps(bericht, ensure_ascii=False))
    assert rund["ergebnisse"][0]["aggregat"] == bericht["ergebnisse"][0]["aggregat"]


def test_ist_default_markierung_im_sweep(tmp_path):
    idx, enc = _index(tmp_path)
    faelle = [("g.yaml", "Schätzung von Besteuerungsgrundlagen", {"norm:AO:162"})]
    roh = sammle_rohdaten(idx, enc, ["Schätzung von Besteuerungsgrundlagen"], k_roh=10)
    ergebnisse = sweep(faelle, roh, topk_liste=[5], floor_liste=[0.30, 0.35],
                       rerank_modi=["aus"], schwellen_liste=[0.1])
    defaults = [e for e in ergebnisse if e["ist_default"]]
    assert len(defaults) == 1
    assert defaults[0]["params"]["floor"] == 0.35   # Ist-Default der Produktion


# --- Sweep mit Negativ-Gold, Policies und RRF-Fusion (Experimente 1–3) --------------

def test_sweep_negativ_und_fusion_ende_zu_ende(tmp_path):
    idx, enc = _index(tmp_path)
    faelle = [("g.yaml", "Schätzung von Besteuerungsgrundlagen", {"norm:AO:162"}),
              ("g.yaml", "Quantenphysik im Weltall", {"norm:KStG:14"})]
    negativ = [("Vielen Dank für das Gespräch.", "alltag"),
               ("Quantenphysik im Weltall", "steuer_neutral")]
    queries = [q for _, q, _ in faelle] + [q for q, _ in negativ]
    roh = sammle_rohdaten(idx, enc, queries, k_roh=10)
    # Lexikalische Stütze: Trigger findet die Schätzungs-Norm überall — auf den
    # Negativen genau der Fehlalarm-Druck, den die Fusion sichtbar machen soll.
    trigger_listen = {q: ["norm:AO:162"] for q in queries}
    ergebnisse = sweep(
        faelle, roh, topk_liste=[5], floor_liste=[0.0, 0.99],
        rerank_modi=["aus"], schwellen_liste=[0.1],
        fusion_modi=["aus", "rrf"], k_rrf_liste=[20, 60],
        trigger_listen=trigger_listen, trigger_latenz={q: 1.0 for q in queries},
        negativ_faelle=negativ)
    # 2 Floors × (aus + rrf20 + rrf60) = 6 Zeilen, jede mit Negativ-Block.
    assert len(ergebnisse) == 6
    for e in ergebnisse:
        assert set(e["negativ"]) == {"n", "fehlvorschlagsquote",
                                     "mittlere_kandidaten", "pro_kategorie"}
        assert e["negativ"]["n"] == 2
    nach_params = {(e["params"]["fusion"], e["params"]["k_rrf"],
                    e["params"]["floor"]): e for e in ergebnisse}
    # Floor 0.99: Embedding leer — OHNE Fusion volle Abstention (Fehlvorschlag 0),
    # MIT Fusion liefert der Trigger trotzdem (Fehlvorschlag 1.0, R@5 gerettet).
    ohne = nach_params[("aus", None, 0.99)]
    mit = nach_params[("rrf", 60, 0.99)]
    assert ohne["aggregat"]["recall@5"] == 0.0
    assert ohne["negativ"]["fehlvorschlagsquote"] == 0.0
    assert mit["aggregat"]["recall@5"] == pytest.approx(1 / 2)
    assert mit["negativ"]["fehlvorschlagsquote"] == 1.0
    assert mit["negativ"]["pro_kategorie"] == {"alltag": 1.0, "steuer_neutral": 1.0}
    # Fusion-Zeilen tragen die Trigger-Latenz oben drauf.
    assert (mit["aggregat"]["median_latenz_ms"]
            > ohne["aggregat"]["median_latenz_ms"])
    # Keine Default-Zeile (floor 0.35 fehlt; Fusion ist nie Default).
    assert not any(e["ist_default"] for e in ergebnisse)


def test_sweep_fusion_braucht_trigger_listen(tmp_path):
    _, _, faelle, roh, _ = _mini_lauf(tmp_path)
    with pytest.raises(ValueError):
        sweep(faelle, roh, topk_liste=[5], floor_liste=[0.0],
              rerank_modi=["aus"], schwellen_liste=[0.1], fusion_modi=["rrf"])


def test_sweep_policy_kombinationen_und_schluessel(tmp_path):
    _, _, faelle, roh, _ = _mini_lauf(tmp_path)
    # Scores für alle (query, kandidat)-Paare der rohen Top-10 (Pool des Sweeps).
    scores = {(q, t["norm_id"]): 0.5
              for q in roh for t in roh[q]["treffer"][:10]}
    ergebnisse = sweep(
        faelle, roh, topk_liste=[5], floor_liste=[0.0],
        rerank_modi=["an"], schwellen_liste=[0.05, 0.10],
        rerank_scores=scores, policies=["P0", "P1", "P2", "P3"],
        hoch_liste=[0.55, 0.60])
    # P0×2 Schwellen + P1×1 + (P2+P3)×2 Schwellen×2 hoch = 2+1+8 = 11 Zeilen.
    assert len(ergebnisse) == 11
    kombis = {(e["params"]["policy"], e["params"]["schwelle"], e["params"]["hoch"])
              for e in ergebnisse}
    assert ("P1", None, None) in kombis
    assert ("P0", 0.1, None) in kombis and ("P0", 0.05, None) in kombis
    assert ("P2", 0.05, 0.6) in kombis and ("P3", 0.1, 0.55) in kombis
