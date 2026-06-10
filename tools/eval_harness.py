"""Zentraler Eval-Harness für das Embedding-/Reranker-Retrieval (Schicht 1.7).

AB JETZT die EINE Mess-Stelle, mit der Qualitäts-Optimierungen der Engine
bewiesen werden: gleiche Gold-Sets, gleiche Metriken, gleicher Bericht — jede
Änderung (Encoder, Floor, Reranker, Schwelle) wird hier gegen die eingefrorenen
Gold-Dateien gemessen statt anekdotisch behauptet.

Methodik:
  * Gold = fixtures/gold/*.yaml (Retrieval-Gold: query + ziele). Ein Fall gilt
    bei Rang r als getroffen, wenn EINE seiner Ziel-Norm-IDs in den Top-r liegt —
    auch als PRÄFIX, aber Pfadsegment-genau: norm:AO:42 trifft norm:AO:42:abs1,
    norm:AO:14 trifft NICHT norm:AO:140.
  * Encoder/Index werden EINMAL geladen; je Query EINE rohe Suche (min_sim=0)
    mit k = max(topk, rerank_topn). Floor und Rerank-Schwelle sind NACHFILTER
    auf den rohen (norm_id, sim)-Listen — identisch zur Live-Suche (die Suche
    schneidet eine absteigend sortierte Liste am Floor ab; Nachfiltern der
    Roh-Liste liefert dieselbe Menge). Reranker-Scores werden je (query,
    kandidat) EINMAL berechnet und über alle Schwellen wiederverwendet.
  * Sweep = kartesisches Produkt topk × floor × rerank-Modus × schwelle.
    Metriken je Konfiguration, je Gold-Datei und aggregiert: Recall@1/5/10/50,
    MRR, Abstention-Quote (Queries ohne Kandidat über Schwelle/Floor),
    Median-Latenz/Query. Latenz = rohe Embedding-Suche (+ Reranker-Pass bei
    rerank=an); die Nachfilter selbst kosten nichts.
  * Hinweis Rundung: die Live-Suche vergleicht den UNGERUNDETEN sim mit
    min_sim, liefert aber 3 Nachkommastellen; der Nachfilter arbeitet auf den
    gelieferten (gerundeten) Werten. Abweichung nur im ±0,0005-Band um den
    Floor — für den Sweep irrelevant.

Ist-Default der Produktion (pipeline.py): Floor 0,35 + Rerank-Schwelle 0,10
(mit Reranker, Top-N 20). Die Default-Listen enthalten beide Werte, damit der
Vergleich beste Konfiguration vs. Ist-Default immer in der Tabelle steht.

Erweiterungen (Experimente Semantik-Schicht):
  * NEGATIV-GOLD (--negativ-gold): Sätze OHNE steuerlichen Schlüsselbegriff-
    Hinweis, für die das System abstinieren soll (fixtures/gold/negativ_v1.yaml).
    Je Konfiguration zusätzlich: Fehlvorschlagsquote (Anteil Negativ-Sätze mit
    ≥1 finalem Kandidaten) + mittlere Kandidatenzahl + Quote je Kategorie —
    erst damit ist der Präzisions-Zweck des Rerankers (Hub-Fehlalarm-Abwehr)
    überhaupt messbar.
  * RERANK-POLICIES (--policies): P0 = heute (rerank sortiert, Abstention rein
    über rerank≥schwelle). P1 = ordering-only (rerank sortiert nur; Abstention
    allein über den Embedding-Floor). P2 = oder-Abstention (Kandidat bleibt,
    wenn rerank≥schwelle ODER sim≥hoch). P3 = top1-rescue (wie P0, aber der
    sim-stärkste Kandidat des Rerank-Pools bleibt immer, wenn sim≥hoch).
    ``hoch`` wird über --hoch-liste gesweept; alles NACHFILTER auf den
    gecachten Scores (kein neuer Modell-Pass je Konfiguration).
  * RRF-FUSION (--fusion rrf|beides): je Query zusätzlich lexikalische
    Kandidaten aus dem Trigger-Index (TriggerIndex.kandidaten — dieselbe
    deterministische Tokenisierung wie die Pipeline, die den ganzen Satz
    übergibt). Reciprocal Rank Fusion über (gefloorte) Embedding-Liste und
    Trigger-Liste: score(d) = Σ 1/(k_rrf + rang_i(d)); --k-rrf-liste sweept
    k_rrf. Varianten: fusion-ohne-rerank (RRF-Topliste direkt) und
    fusion-vor-rerank (RRF-Topliste in den Reranker, dann Policy-Filter).
    Trigger-only-Kandidaten tragen sim=0,0 (kein Embedding-Treffer über Floor).
    Rerank-Latenz je Query bleibt der Top-N-Pass; zusätzliche Fusion-Kandidaten
    werden in einem UNGEMESSENEN Cache-Pass gescort (Produktion würde exakt
    Top-N Paare scoren — die gemessene Latenz bleibt damit repräsentativ).

Aufruf (Beispiele):
  python tools/eval_harness.py --corpus C:/dev/taxtronikexctract/corpus \\
      --rerank aus --bericht out/berichte
  python tools/eval_harness.py --corpus C:/dev/taxtronikexctract/corpus \\
      --rerank beides --benchmark-out benchmarks/harness-baseline.json
  python tools/eval_harness.py --corpus C:/dev/taxtronikexctract/corpus \\
      --rerank beides --fusion beides --policies P0 P1 P2 P3 \\
      --negativ-gold fixtures/gold/negativ_v1.yaml --floor-liste 0.35 \\
      --benchmark-out benchmarks/harness-fusion-policy.json

Exit-Codes: 0 = ok, 2 = Fehler (Index/Gold/Reranker/Trigger-Index fehlt o. Ä.).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole                            # noqa: E402

from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.graph import graph_oeffnen                        # noqa: E402
from risk_layer.persistenz import atomar_schreiben                # noqa: E402
from risk_layer.reranker import (RERANK_TOPN, baue_norm_doc_fn,   # noqa: E402
                                 reranker_oeffnen)
from risk_layer.sicherung import zeitstempel_dateiname            # noqa: E402
from risk_layer.triggerindex import TriggerIndex                  # noqa: E402

# Recall-Tiefen des Berichts (fest, damit Berichte über Läufe vergleichbar bleiben).
ZIEL_KS = (1, 5, 10, 50)

# Ist-Default der Produktion (pipeline.py / rerank_schwelle) — wird in der
# Tabelle markiert und im Bericht separat ausgewiesen.
IST_DEFAULT_FLOOR = 0.35
IST_DEFAULT_SCHWELLE = 0.10

# Rerank-Abstention-Policies (s. Modul-Docstring): P0 = Ist-Verhalten.
POLICIES = ("P0", "P1", "P2", "P3")

DEFAULT_GOLD = ("fixtures/gold/ao_v1.yaml", "fixtures/gold/organschaft_v1.yaml")


# --- Treffer-Logik (pur, testbar) --------------------------------------------

def ist_treffer(kandidat_id: str, ziel_id: str) -> bool:
    """Pfadsegment-genauer Präfix-Treffer: das Gold-Ziel trifft den Kandidaten,
    wenn er identisch ist ODER ein tieferer Knoten desselben Ziels
    (norm:AO:42 trifft norm:AO:42:abs1). KEIN String-Präfix: norm:AO:14
    trifft norm:AO:140 NICHT — verglichen wird an der ':'-Segmentgrenze."""
    return kandidat_id == ziel_id or kandidat_id.startswith(ziel_id + ":")


def treffer_rang(kandidaten: list[str], ziele: set[str]) -> int | None:
    """Rang (1-basiert) des ersten Kandidaten, der EIN Gold-Ziel trifft; None = verfehlt."""
    for i, kid in enumerate(kandidaten, start=1):
        if any(ist_treffer(kid, z) for z in ziele):
            return i
    return None


# --- Nachfilter (pur, testbar) ------------------------------------------------

def floor_filter(treffer: list[dict[str, Any]], floor: float) -> list[dict[str, Any]]:
    """Floor als NACHFILTER auf der rohen, absteigend sortierten Trefferliste.

    Äquivalent zu ``idx.suche(..., min_sim=floor)``: die Suche bricht beim
    ersten sim < floor ab, also schneidet der Filter nur den Listen-Schwanz.
    """
    return [t for t in treffer if t["sim"] >= floor]


def rrf_scores(listen: list[list[str]], k_rrf: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k_rrf + rang_i(d)) über alle
    Listen, in denen d vorkommt (Ränge 1-basiert). Pur und handnachrechenbar."""
    scores: dict[str, float] = {}
    for liste in listen:
        for rang, nid in enumerate(liste, start=1):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k_rrf + rang)
    return scores


def fusioniere_treffer(dense: list[dict[str, Any]], trigger_ids: list[str],
                       k_rrf: int) -> list[dict[str, Any]]:
    """RRF-Fusion der (bereits gefloorten) Embedding-Treffer mit der
    lexikalischen Trigger-Kandidatenliste — Ergebnis treffer-förmig
    (norm_id/sim/rrf), absteigend nach RRF-Score.

    Trigger-only-Kandidaten tragen sim=0,0 (kein Embedding-Treffer über Floor) —
    sim-basierte Policy-Bedingungen (P2/P3 ``hoch``) können sie also nie retten.
    Deterministischer Tie-Breaker: bester Einzel-Rang, dann norm_id.
    """
    dense_ids = [t["norm_id"] for t in dense]
    scores = rrf_scores([dense_ids, list(trigger_ids)], k_rrf)
    sim_map = {t["norm_id"]: t["sim"] for t in dense}
    bester_rang: dict[str, int] = {}
    for liste in (dense_ids, list(trigger_ids)):
        for rang, nid in enumerate(liste, start=1):
            bester_rang[nid] = min(bester_rang.get(nid, rang), rang)
    geordnet = sorted(scores, key=lambda n: (-scores[n], bester_rang[n], n))
    return [{"norm_id": nid, "sim": sim_map.get(nid, 0.0),
             "rrf": round(scores[nid], 6)} for nid in geordnet]


def konfig_kandidaten(treffer: list[dict[str, Any]], *, floor: float, topk: int,
                      query: str | None = None,
                      rerank_scores: dict[tuple[str, str], float] | None = None,
                      schwelle: float | None = None,
                      rerank_topn: int = RERANK_TOPN,
                      policy: str = "P0", hoch: float | None = None,
                      trigger_ids: list[str] | None = None,
                      k_rrf: int | None = None) -> list[str]:
    """Finale Kandidaten-IDs EINER Query unter EINER Konfiguration.

    Pipeline: Floor-Nachfilter → (optional RRF-Fusion mit Trigger-Liste) →
    rerank=aus: Top-k | rerank=an: Top-N → nach Cross-Encoder-Score absteigend
    (stabil: Gleichstand behält Listen-Reihenfolge, wie ``rerank_normen``) →
    Policy-Abstention → Top-k.

    Policies (Abstention NACH der Rerank-Sortierung):
      P0  Kandidat bleibt, wenn rerank ≥ schwelle (Ist-Verhalten).
      P1  ordering-only: alle bleiben (Abstention rein über den Floor davor).
      P2  oder-Abstention: bleibt, wenn rerank ≥ schwelle ODER sim ≥ hoch.
      P3  top1-rescue: wie P0, zusätzlich bleibt der sim-stärkste Kandidat
          des Rerank-Pools immer, wenn sein sim ≥ hoch.
    """
    kand = floor_filter(treffer, floor)
    if k_rrf is not None:
        kand = fusioniere_treffer(kand, trigger_ids or [], k_rrf)
    if rerank_scores is None:
        return [t["norm_id"] for t in kand[:topk]]
    if policy not in POLICIES:
        raise ValueError(f"Unbekannte Policy {policy!r} (gültig: {POLICIES})")
    if policy in ("P0", "P2", "P3") and schwelle is None:
        raise ValueError(f"Policy {policy} braucht eine Schwelle")
    if policy in ("P2", "P3") and hoch is None:
        raise ValueError(f"Policy {policy} braucht hoch (sim-Schwelle)")
    pool = kand[:rerank_topn]
    geordnet = sorted(pool, key=lambda t: rerank_scores[(query, t["norm_id"])],
                      reverse=True)
    if policy == "P1":
        behalten = geordnet
    elif policy == "P2":
        behalten = [t for t in geordnet
                    if rerank_scores[(query, t["norm_id"])] >= schwelle
                    or t["sim"] >= hoch]
    elif policy == "P3":
        # sim-stärkster Kandidat des POOLS (max statt pool[0]: nach RRF-Fusion
        # ist die Liste nicht mehr sim-sortiert); Gleichstand: erster gewinnt.
        rescue = max(pool, key=lambda t: t["sim"], default=None)
        rescue_id = (rescue["norm_id"]
                     if rescue is not None and rescue["sim"] >= hoch else None)
        behalten = [t for t in geordnet
                    if rerank_scores[(query, t["norm_id"])] >= schwelle
                    or t["norm_id"] == rescue_id]
    else:                                  # P0 — Ist-Verhalten
        behalten = [t for t in geordnet
                    if rerank_scores[(query, t["norm_id"])] >= schwelle]
    return [t["norm_id"] for t in behalten][:topk]


# --- Metriken (pur, handverifizierbar) ----------------------------------------

def metriken(ergebnisse: list[dict[str, Any]]) -> dict[str, Any]:
    """Recall@k/MRR/Abstention/Median-Latenz über Fall-Ergebnisse.

    Jedes Ergebnis: ``rang`` (1-basiert oder None = verfehlt), ``abstention``
    (kein Kandidat über Floor/Schwelle ⇒ zählt zugleich als verfehlt) und
    ``latenz_ms``. Verfehlte Fälle tragen MRR-Beitrag 0.
    """
    n = len(ergebnisse)
    if n == 0:
        return {"n": 0}
    raenge = [e["rang"] for e in ergebnisse]
    out: dict[str, Any] = {"n": n}
    for k in ZIEL_KS:
        out[f"recall@{k}"] = round(sum(1 for r in raenge if r and r <= k) / n, 4)
    out["mrr"] = round(sum(1.0 / r for r in raenge if r) / n, 4)
    out["abstention"] = round(sum(1 for e in ergebnisse if e["abstention"]) / n, 4)
    out["median_latenz_ms"] = round(
        statistics.median(e["latenz_ms"] for e in ergebnisse), 1)
    return out


def negativ_metriken(ergebnisse: list[dict[str, Any]]) -> dict[str, Any]:
    """Präzisionsmetrik auf dem Negativ-Gold (pur, handverifizierbar).

    Jedes Ergebnis: ``kategorie`` + ``n_kandidaten`` (finale Kandidaten der
    Konfiguration für diesen Negativ-Satz). Fehlvorschlagsquote = Anteil Sätze
    mit ≥1 Kandidat (für die ABSTINIEREN korrekt wäre); dazu die mittlere
    Kandidatenzahl und die Quote je Kategorie (alltag/juristisch/steuer-hart).
    """
    n = len(ergebnisse)
    if n == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": n,
        "fehlvorschlagsquote": round(
            sum(1 for e in ergebnisse if e["n_kandidaten"] > 0) / n, 4),
        "mittlere_kandidaten": round(
            sum(e["n_kandidaten"] for e in ergebnisse) / n, 2),
    }
    pro_kat: dict[str, list[dict[str, Any]]] = {}
    for e in ergebnisse:
        pro_kat.setdefault(e["kategorie"], []).append(e)
    out["pro_kategorie"] = {
        kat: round(sum(1 for e in lst if e["n_kandidaten"] > 0) / len(lst), 4)
        for kat, lst in sorted(pro_kat.items())}
    return out


# --- Gold ----------------------------------------------------------------------

def lade_gold(pfad: Path, max_faelle: int | None = None) -> dict[str, Any]:
    """Lädt eine Retrieval-Gold-Datei (query/ziele); wirft bei leerem/kaputtem Gold."""
    daten = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
    faelle: list[tuple[str, set[str]]] = []
    for f in daten.get("faelle") or []:
        q = str(f.get("query") or "").strip()
        ziele = {str(z).strip() for z in (f.get("ziele") or []) if str(z).strip()}
        if q and ziele:
            faelle.append((q, ziele))
    if not faelle:
        raise ValueError(f"{pfad}: kein Retrieval-Gold (faelle mit query+ziele) gefunden.")
    if max_faelle is not None:
        faelle = faelle[:max_faelle]
    return {"name": pfad.name, "version": str(daten.get("gold_version", "?")),
            "stand": str(daten.get("stand", "?")), "faelle": faelle}


def lade_negativ_gold(pfad: Path, max_faelle: int | None = None) -> dict[str, Any]:
    """Lädt Negativ-Gold (Sätze OHNE Ziele — korrekte Antwort: Abstention).

    Format wie Retrieval-Gold, aber ``faelle`` tragen nur query (+ optionale
    ``kategorie`` für die Auswertung je Schwierigkeitsgrad). Wirft bei
    leerem/kaputtem Gold — ein leeres Negativ-Set wäre eine stille Nicht-Messung.
    """
    daten = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
    faelle: list[tuple[str, str]] = []
    for f in daten.get("faelle") or []:
        q = str(f.get("query") or "").strip()
        kat = str(f.get("kategorie") or "unkategorisiert").strip()
        if q:
            faelle.append((q, kat))
    if not faelle:
        raise ValueError(f"{pfad}: kein Negativ-Gold (faelle mit query) gefunden.")
    if max_faelle is not None:
        faelle = faelle[:max_faelle]
    return {"name": pfad.name, "version": str(daten.get("gold_version", "?")),
            "stand": str(daten.get("stand", "?")), "faelle": faelle}


# --- Rohdaten-Sammlung (Modelle EINMAL, dann nur Nachfilter) --------------------

def sammle_rohdaten(idx: EmbeddingIndex, enc, queries: list[str],
                    k_roh: int) -> dict[str, dict[str, Any]]:
    """Je Query EINE rohe Suche (min_sim=0, k=k_roh) + Latenzmessung."""
    roh: dict[str, dict[str, Any]] = {}
    for q in queries:
        if q in roh:                       # identische Query in mehreren Gold-Dateien
            continue
        t0 = time.perf_counter()
        treffer = idx.suche(q, enc, k=k_roh, min_sim=0.0)
        roh[q] = {"treffer": treffer,
                  "latenz_ms": (time.perf_counter() - t0) * 1000.0}
    return roh


def sammle_trigger_kandidaten(trigger_idx: TriggerIndex, queries: list[str],
                              k: int) -> tuple[dict[str, list[str]],
                                               dict[str, float]]:
    """Je Query EINMAL die lexikalische Trigger-Kandidatenliste (gerankte
    norm_ids, §-Ebene — derselbe ID-Raum wie der Embedding-Index) + Latenz.
    Tokenisierung = ``TriggerIndex.kandidaten`` selbst (deterministisch, wie
    die Pipeline, die den ganzen Satz übergibt)."""
    listen: dict[str, list[str]] = {}
    latenz: dict[str, float] = {}
    for q in sorted(set(queries)):         # deterministische Abarbeitung
        t0 = time.perf_counter()
        kand = trigger_idx.kandidaten(q, k=k)
        latenz[q] = (time.perf_counter() - t0) * 1000.0
        listen[q] = [c["norm_id"] for c in kand]
    return listen, latenz


def sammle_rerank_scores(reranker, doc_fn, roh: dict[str, dict[str, Any]],
                         topn: int,
                         extra_ids: dict[str, list[str]] | None = None,
                         ) -> tuple[dict[tuple[str, str], float],
                                    dict[str, float]]:
    """Cross-Encoder-Scores je (query, kandidat) — EIN Pass über die rohen
    Top-N, gecacht für alle Floor/Schwellen-Kombinationen (jede gefilterte
    Kandidatenmenge ist Teilmenge der rohen Top-N).

    ``extra_ids`` (Fusion): zusätzliche Kandidaten je Query (tiefere Embedding-
    Ränge + Trigger-only), die in einem ZWEITEN, UNGEMESSENEN Pass gescort
    werden — die gemessene Latenz bleibt der Top-N-Pass, denn die Produktion
    würde je Query exakt Top-N Paare scoren (egal woher die Kandidaten kommen).
    """
    scores: dict[tuple[str, str], float] = {}
    latenz: dict[str, float] = {}
    for q in sorted(roh):                  # deterministische Abarbeitung
        kand = roh[q]["treffer"][:topn]
        t0 = time.perf_counter()
        sc = reranker.score([(q, doc_fn(t["norm_id"])) for t in kand]) if kand else []
        latenz[q] = (time.perf_counter() - t0) * 1000.0
        for t, s in zip(kand, sc):
            scores[(q, t["norm_id"])] = float(s)
        rest = list(dict.fromkeys(                     # dedupe, Reihenfolge stabil
            nid for nid in (extra_ids or {}).get(q, ())
            if (q, nid) not in scores))
        if rest:                                       # Cache-Pass (ungemessen)
            for nid, s in zip(rest, reranker.score([(q, doc_fn(n)) for n in rest])):
                scores[(q, nid)] = float(s)
    return scores, latenz


# --- Sweep (pur über Rohdaten — testbar ohne Modelle) ---------------------------

def _params_schluessel(p: dict[str, Any]) -> str:
    teile = [p["rerank"], f"floor={p['floor']}", f"schwelle={p['schwelle']}",
             f"topk={p['topk']}"]
    if p.get("policy") is not None:
        teile.insert(1, f"policy={p['policy']}")
    if p.get("hoch") is not None:
        teile.append(f"hoch={p['hoch']}")
    if p.get("fusion", "aus") != "aus":
        teile.insert(0, f"rrf{p.get('k_rrf')}")
    return "|".join(teile)


def ist_default_konfig(p: dict[str, Any]) -> bool:
    """Markiert die Ist-Default-Zeile der Produktion (0,35 / 0,10, keine
    Fusion, Policy P0 = heutige Abstention-Mechanik)."""
    if p.get("fusion", "aus") != "aus":
        return False
    if abs(p["floor"] - IST_DEFAULT_FLOOR) > 1e-9:
        return False
    if p["rerank"] == "an":
        return ((p.get("policy") or "P0") == "P0"
                and abs((p["schwelle"] or -1) - IST_DEFAULT_SCHWELLE) < 1e-9)
    return True


def _policy_kombis(policies: list[str], schwellen: list[float],
                   hoch_liste: list[float]) -> list[tuple[str, float | None,
                                                          float | None]]:
    """(policy, schwelle, hoch)-Kombinationen je Policy-Bedarf: P1 braucht
    weder Schwelle noch hoch (eine Zeile), P0 nur Schwellen, P2/P3 das
    kartesische Produkt Schwellen × hoch."""
    kombis: list[tuple[str, float | None, float | None]] = []
    for pol in sorted(policies):
        if pol == "P1":
            kombis.append((pol, None, None))
        elif pol == "P0":
            kombis.extend((pol, s, None) for s in sorted(schwellen))
        else:
            kombis.extend((pol, s, h) for s in sorted(schwellen)
                          for h in sorted(hoch_liste))
    return kombis


def sweep(faelle: list[tuple[str, str, set[str]]],
          roh: dict[str, dict[str, Any]], *,
          topk_liste: list[int], floor_liste: list[float],
          rerank_modi: list[str], schwellen_liste: list[float],
          rerank_scores: dict[tuple[str, str], float] | None = None,
          rerank_latenz: dict[str, float] | None = None,
          rerank_topn: int = RERANK_TOPN,
          policies: list[str] | tuple[str, ...] = ("P0",),
          hoch_liste: list[float] | tuple[float, ...] = (0.55,),
          fusion_modi: list[str] | tuple[str, ...] = ("aus",),
          k_rrf_liste: list[int] | tuple[int, ...] = (60,),
          trigger_listen: dict[str, list[str]] | None = None,
          trigger_latenz: dict[str, float] | None = None,
          negativ_faelle: list[tuple[str, str]] | None = None,
          ) -> list[dict[str, Any]]:
    """Kartesisches Produkt aller Konfigurationen über die GLEICHEN Rohdaten.

    ``faelle``: (gold_name, query, ziele). ``negativ_faelle``: (query,
    kategorie) — je Konfiguration wird zusätzlich die Fehlvorschlagsquote auf
    den Negativen gemessen (``negativ``-Block je Ergebnis; None ohne Negativ-
    Gold). ``fusion_modi``: "aus" und/oder "rrf" (braucht ``trigger_listen``);
    "rrf" wird über ``k_rrf_liste`` gesweept. Rückgabe absteigend sortiert
    nach Recall@5, dann MRR (feste Tie-Breaker ⇒ deterministische Reihenfolge).
    """
    fusionen: list[tuple[str, int | None]] = []
    for fusion in sorted(fusion_modi):
        if fusion == "aus":
            fusionen.append(("aus", None))
        elif fusion == "rrf":
            if trigger_listen is None:
                raise ValueError("fusion=rrf im Sweep, aber keine Trigger-Listen gesammelt")
            fusionen.extend(("rrf", k) for k in sorted(k_rrf_liste))
        else:
            raise ValueError(f"Unbekannter Fusion-Modus {fusion!r} (aus|rrf)")

    def _kandidaten(q: str, *, floor, topk, modus, pol, schwelle, hoch, k_rrf):
        return konfig_kandidaten(
            roh[q]["treffer"], floor=floor, topk=topk, query=q,
            rerank_scores=rerank_scores if modus == "an" else None,
            schwelle=schwelle, rerank_topn=rerank_topn,
            policy=pol or "P0", hoch=hoch,
            trigger_ids=(trigger_listen or {}).get(q), k_rrf=k_rrf)

    ergebnisse: list[dict[str, Any]] = []
    for topk, floor, (fusion, k_rrf), modus in product(
            sorted(topk_liste), sorted(floor_liste), fusionen,
            sorted(rerank_modi)):
        if modus == "an":
            if rerank_scores is None:
                raise ValueError("rerank=an im Sweep, aber keine Reranker-Scores gesammelt")
            kombis = _policy_kombis(list(policies), sorted(schwellen_liste),
                                    sorted(hoch_liste))
        else:
            kombis = [(None, None, None)]
        for pol, schwelle, hoch in kombis:
            pro_fall: list[dict[str, Any]] = []
            pro_gold: dict[str, list[dict[str, Any]]] = {}
            for gold_name, q, ziele in faelle:
                kand = _kandidaten(q, floor=floor, topk=topk, modus=modus,
                                   pol=pol, schwelle=schwelle, hoch=hoch,
                                   k_rrf=k_rrf)
                lat = roh[q]["latenz_ms"]
                if k_rrf is not None:
                    lat += (trigger_latenz or {}).get(q, 0.0)
                if modus == "an":
                    lat += (rerank_latenz or {}).get(q, 0.0)
                fall = {"rang": treffer_rang(kand, ziele),
                        "abstention": not kand, "latenz_ms": lat}
                pro_fall.append(fall)
                pro_gold.setdefault(gold_name, []).append(fall)
            negativ = None
            if negativ_faelle:
                negativ = negativ_metriken([
                    {"kategorie": kat,
                     "n_kandidaten": len(_kandidaten(
                         q, floor=floor, topk=topk, modus=modus, pol=pol,
                         schwelle=schwelle, hoch=hoch, k_rrf=k_rrf))}
                    for q, kat in negativ_faelle])
            params = {"fusion": fusion, "k_rrf": k_rrf, "rerank": modus,
                      "policy": pol, "floor": floor, "schwelle": schwelle,
                      "hoch": hoch, "topk": topk,
                      "rerank_topn": rerank_topn if modus == "an" else None}
            ergebnisse.append({
                "params": params,
                "ist_default": ist_default_konfig(params),
                "aggregat": metriken(pro_fall),
                "pro_gold": {n: metriken(f) for n, f in sorted(pro_gold.items())},
                "negativ": negativ,
            })
    # Beste Konfiguration zuerst: Recall@5, dann MRR; feste weitere Tie-Breaker
    # (R@10, R@1, weniger Abstention, Parameter-String) für Determinismus.
    ergebnisse.sort(key=lambda e: (
        -e["aggregat"]["recall@5"], -e["aggregat"]["mrr"],
        -e["aggregat"]["recall@10"], -e["aggregat"]["recall@1"],
        e["aggregat"]["abstention"], _params_schluessel(e["params"])))
    return ergebnisse


# --- Bericht ---------------------------------------------------------------------

def baue_bericht(*, konfiguration: dict[str, Any], gold_dateien: list[dict[str, Any]],
                 encoder_name: str, index: EmbeddingIndex,
                 reranker_name: str | None, ergebnisse: list[dict[str, Any]],
                 zeitstempel: str,
                 negativ_gold: dict[str, Any] | None = None) -> dict[str, Any]:
    """Vollständiger, reproduzierbarer Bericht: Konfiguration + Gold-Versionen +
    Encoder + Index-Fingerprint + Zeitstempel + sortierte Sweep-Ergebnisse
    (+ Negativ-Gold-Version, wenn die Präzisionsmetrik mitgemessen wurde)."""
    return {
        "harness": "eval_harness/2",
        "erstellt_am": zeitstempel,
        "konfiguration": konfiguration,
        "gold": {g["name"]: {"version": g["version"], "stand": g["stand"],
                             "n": len(g["faelle"])} for g in gold_dateien},
        "negativ_gold": (None if negativ_gold is None else
                         {"name": negativ_gold["name"],
                          "version": negativ_gold["version"],
                          "stand": negativ_gold["stand"],
                          "n": len(negativ_gold["faelle"])}),
        "encoder": encoder_name,
        "reranker": reranker_name,
        "index": {"n_vektoren": len(index.ids), "dim": index.dim,
                  # Alt-Indizes (vor der Staleness-Erkennung gebaut): None.
                  "graph_fingerprint": index.graph_fingerprint},
        "n_faelle": sum(len(g["faelle"]) for g in gold_dateien),
        # Ist-Default der Produktion (je Modus EINE Zeile: aus/0,35 bzw. an/0,35/0,10)
        # zum direkten Vergleich gegen die beste Konfiguration.
        "ist_default": [e for e in ergebnisse if e["ist_default"]],
        "ergebnisse": ergebnisse,
    }


def _tabelle(ergebnisse: list[dict[str, Any]]) -> str:
    """Menschliche Sweep-Tabelle (beste Konfiguration zuerst, Default mit *).
    Mit Negativ-Gold zusätzlich Fehl% (Fehlvorschlagsquote) + mKand (mittlere
    Kandidatenzahl auf den Negativen)."""
    mit_negativ = any(e.get("negativ") for e in ergebnisse)
    kopf = (f"{'':1} {'fusion':6} {'rerank':6} {'pol':3} {'floor':>5} "
            f"{'schwelle':>8} {'hoch':>5} {'topk':>4} "
            f"{'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@50':>6} {'MRR':>6} "
            f"{'Abst':>6}")
    if mit_negativ:
        kopf += f" {'Fehl%':>6} {'mKand':>6}"
    kopf += f" {'Lat ms':>7}"
    zeilen = [kopf, "-" * len(kopf)]
    for e in ergebnisse:
        p, a = e["params"], e["aggregat"]
        fusion = ("aus" if p.get("fusion", "aus") == "aus"
                  else f"rrf{p.get('k_rrf')}")
        schwelle = "-" if p["schwelle"] is None else f"{p['schwelle']:.2f}"
        hoch = "-" if p.get("hoch") is None else f"{p['hoch']:.2f}"
        zeile = (
            f"{'*' if e['ist_default'] else '':1} {fusion:6} {p['rerank']:6} "
            f"{p.get('policy') or '-':3} {p['floor']:>5.2f} {schwelle:>8} "
            f"{hoch:>5} {p['topk']:>4} "
            f"{a['recall@1']:>6.3f} {a['recall@5']:>6.3f} {a['recall@10']:>6.3f} "
            f"{a['recall@50']:>6.3f} {a['mrr']:>6.3f} {a['abstention']:>6.2f}")
        if mit_negativ:
            ng = e.get("negativ")
            zeile += ((f" {ng['fehlvorschlagsquote'] * 100:>5.1f}% "
                       f"{ng['mittlere_kandidaten']:>6.1f}") if ng
                      else f" {'-':>6} {'-':>6}")
        zeile += f" {a['median_latenz_ms']:>7.1f}"
        zeilen.append(zeile)
    return "\n".join(zeilen)


# --- CLI ---------------------------------------------------------------------------

def _geraet_waehlen(wunsch: str) -> str:
    """auto → cuda, wenn verfügbar, sonst cpu (torch optional)."""
    if wunsch != "auto":
        return wunsch
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv: list[str] | None = None) -> int:
    utf8_konsole()
    ap = argparse.ArgumentParser(
        description="Eval-Harness: Gold-Retrieval-Sweep über Floor/Reranker-Schwelle")
    ap.add_argument("--gold", nargs="+", default=list(DEFAULT_GOLD),
                    help="Gold-YAMLs (Retrieval-Gold: query+ziele)")
    ap.add_argument("--negativ-gold",
                    help="Negativ-Gold-YAML (Sätze ohne steuerlichen Hinweis; "
                         "misst je Konfiguration die Fehlvorschlagsquote)")
    ap.add_argument("--corpus", default="corpus",
                    help="Korpus-Verzeichnis (graph.sqlite + trigger.sqlite + embedding/)")
    ap.add_argument("--device", default="auto",
                    help="auto|cuda|cpu (auto: cuda, wenn verfügbar)")
    ap.add_argument("--topk-liste", nargs="+", type=int, default=[50],
                    help="Vorschlagstiefen k (Default 50 = volle Recall@50-Messung)")
    ap.add_argument("--floor-liste", nargs="+", type=float,
                    default=[0.30, 0.35, 0.40, 0.45, 0.50],
                    help="Embedding-Mindest-Sim (Nachfilter-Sweep)")
    ap.add_argument("--rerank", choices=("aus", "an", "beides"), default="aus",
                    help="Cross-Encoder-Reranking messen (an/beides braucht GPU-Modell)")
    ap.add_argument("--rerank-schwellen-liste", nargs="+", type=float,
                    default=[0.05, 0.10, 0.20],
                    help="Abstention-Schwellen für den Rerank-Score")
    ap.add_argument("--rerank-topn", type=int, default=RERANK_TOPN,
                    help=f"Bi-Encoder-Top-N für den Reranker (Default {RERANK_TOPN})")
    ap.add_argument("--reranker", default="bge-reranker-v2-m3",
                    help="Reranker-Modellname (s. risk_layer.reranker)")
    ap.add_argument("--policies", nargs="+", default=["P0"], choices=POLICIES,
                    help="Rerank-Abstention-Policies (P0=heute, P1=ordering-only, "
                         "P2=oder-Abstention, P3=top1-rescue)")
    ap.add_argument("--hoch-liste", nargs="+", type=float,
                    default=[0.50, 0.55, 0.60],
                    help="sim-hoch-Schwellen für P2/P3 (Sweep)")
    ap.add_argument("--fusion", choices=("aus", "rrf", "beides"), default="aus",
                    help="RRF-Fusion Embedding × Trigger-Index messen")
    ap.add_argument("--k-rrf-liste", nargs="+", type=int, default=[20, 60],
                    help="RRF-Konstanten k (Sweep; score=Σ 1/(k+rang))")
    ap.add_argument("--trigger-k", type=int, default=20,
                    help="Tiefe der lexikalischen Trigger-Kandidatenliste (Default 20)")
    ap.add_argument("--bericht",
                    help="Verzeichnis für den datierten JSON-Bericht (out/berichte/)")
    ap.add_argument("--benchmark-out",
                    help="Commit-würdige Ergebnisdatei (z. B. benchmarks/harness-baseline.json)")
    ap.add_argument("--max-faelle", type=int,
                    help="Schnelllauf: höchstens N Fälle je Gold-Datei")
    args = ap.parse_args(argv)

    # Determinismus: keine echte Zufallsquelle im Harness, Seeds trotzdem fixiert
    # (Encoder-/Reranker-Backends könnten welche konsultieren).
    random.seed(0)
    np.random.seed(0)

    try:
        gold_dateien = [lade_gold(Path(p), args.max_faelle) for p in args.gold]
        negativ_gold = (lade_negativ_gold(Path(args.negativ_gold), args.max_faelle)
                        if args.negativ_gold else None)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Gold-Fehler: {exc}", file=sys.stderr)
        return 2

    corpus = Path(args.corpus)
    idx = EmbeddingIndex.load(corpus / "embedding")
    if idx is None:
        print(f"Kein Embedding-Index unter {corpus / 'embedding'} — erst bauen "
              "(risk-layer embedding build) oder --corpus prüfen.", file=sys.stderr)
        return 2
    device = _geraet_waehlen(args.device)
    try:
        enc = encoder_oeffnen(idx.encoder_name, device=device)
    except (ImportError, ValueError) as exc:
        print(f"Encoder-Fehler: {exc}", file=sys.stderr)
        return 2

    modi = ["aus", "an"] if args.rerank == "beides" else [args.rerank]
    fusion_modi = (["aus", "rrf"] if args.fusion == "beides" else [args.fusion])
    faelle = [(g["name"], q, z) for g in gold_dateien for q, z in g["faelle"]]
    negativ_faelle = negativ_gold["faelle"] if negativ_gold else None
    k_roh = max(max(args.topk_liste), args.rerank_topn if "an" in modi else 0)

    print(f"Eval-Harness: {len(faelle)} Fälle aus {len(gold_dateien)} Gold-Dateien"
          + (f" + {len(negativ_faelle)} Negativ-Sätze" if negativ_faelle else "")
          + f" | Encoder {enc.name} (device={device}) | Index {len(idx.ids)} Vektoren "
          f"(fingerprint={idx.graph_fingerprint or '—'})")

    # Warmlauf VOR der Zeitmessung (Lazy-Load des Modells gehört nicht in die
    # Query-Latenz; gemessen wird der eingeschwungene Zustand).
    try:
        enc.encode(["Warmlauf"])
    except RuntimeError as exc:
        print(f"Encoder lädt nicht: {exc}", file=sys.stderr)
        return 2
    alle_queries = ([q for _, q, _ in faelle]
                    + [q for q, _ in (negativ_faelle or [])])
    roh = sammle_rohdaten(idx, enc, alle_queries, k_roh)

    # Lexikalische Trigger-Kandidaten (RRF-Fusion): Index EINMAL öffnen,
    # reiner Lesepfad (nur kandidaten(); bestaetige wird nie aufgerufen).
    trigger_listen = trigger_latenz = None
    if "rrf" in fusion_modi:
        trigger_pfad = corpus / "trigger.sqlite"
        try:
            trigger_idx = TriggerIndex(trigger_pfad)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Trigger-Index-Fehler: {exc}", file=sys.stderr)
            return 2
        trigger_listen, trigger_latenz = sammle_trigger_kandidaten(
            trigger_idx, list(roh), args.trigger_k)
        trigger_idx.close()

    rerank_scores = rerank_latenz = None
    reranker_name = None
    if "an" in modi:
        graph = graph_oeffnen(corpus / "graph.sqlite")
        if graph is None:
            print(f"Kein Normgraph unter {corpus / 'graph.sqlite'} — der Reranker "
                  "braucht ihn für die anker-bewussten Norm-Docs.", file=sys.stderr)
            return 2
        from risk_layer.catalog import lade_katalog
        try:
            rr = reranker_oeffnen(args.reranker, device=device)
        except Exception as exc:  # noqa: BLE001 — klare Meldung statt Stacktrace
            print(f"Reranker-Fehler: {exc}", file=sys.stderr)
            return 2
        if rr is None:
            print(f"Reranker {args.reranker!r} nicht verfügbar "
                  "(sentence-transformers installiert?).", file=sys.stderr)
            return 2
        reranker_name = rr.name
        doc_fn = baue_norm_doc_fn(lade_katalog(), graph)
        try:
            rr.score([("Warmlauf", "Warmlauf")])    # Lazy-Load raus aus der Messung
        except RuntimeError as exc:
            print(f"Reranker lädt nicht: {exc}", file=sys.stderr)
            return 2
        # Fusion: auch tiefere Embedding-Ränge + Trigger-only-Kandidaten können
        # in den Rerank-Pool gelangen → Score-Cache auf die Vereinigung erweitern
        # (ungemessener Zweit-Pass; s. sammle_rerank_scores).
        extra_ids = None
        if trigger_listen is not None:
            extra_ids = {}
            for q, daten in roh.items():
                roh_ids = [t["norm_id"] for t in daten["treffer"]]
                extra_ids[q] = (roh_ids[args.rerank_topn:]
                                + [n for n in trigger_listen.get(q, [])
                                   if n not in set(roh_ids)])
        print(f"Reranker: {reranker_name} (Top-N {args.rerank_topn}, "
              f"{len(roh)} Queries"
              + (", + Fusion-Kandidaten" if extra_ids else "") + ") …")
        rerank_scores, rerank_latenz = sammle_rerank_scores(
            rr, doc_fn, roh, args.rerank_topn, extra_ids)

    ergebnisse = sweep(
        faelle, roh, topk_liste=args.topk_liste, floor_liste=args.floor_liste,
        rerank_modi=modi, schwellen_liste=args.rerank_schwellen_liste,
        rerank_scores=rerank_scores, rerank_latenz=rerank_latenz,
        rerank_topn=args.rerank_topn,
        policies=args.policies, hoch_liste=args.hoch_liste,
        fusion_modi=fusion_modi, k_rrf_liste=args.k_rrf_liste,
        trigger_listen=trigger_listen, trigger_latenz=trigger_latenz,
        negativ_faelle=negativ_faelle)

    print()
    print(_tabelle(ergebnisse))
    beste = ergebnisse[0]
    print(f"\nBeste Konfiguration: {_params_schluessel(beste['params'])}")
    for name, m in beste["pro_gold"].items():
        print(f"  {name}: n={m['n']}  R@5={m['recall@5']:.3f}  "
              f"MRR={m['mrr']:.3f}  Abst={m['abstention']:.2f}")
    if beste.get("negativ"):
        ng = beste["negativ"]
        print(f"  negativ: n={ng['n']}  Fehlvorschlag={ng['fehlvorschlagsquote']:.2%}  "
              f"mKand={ng['mittlere_kandidaten']:.1f}  "
              f"pro Kategorie={ng['pro_kategorie']}")
    for default in (e for e in ergebnisse if e["ist_default"]):
        a = default["aggregat"]
        zeile = (f"Ist-Default ({_params_schluessel(default['params'])}): "
                 f"R@5={a['recall@5']:.3f}  MRR={a['mrr']:.3f}  "
                 f"Abst={a['abstention']:.2f}")
        if default.get("negativ"):
            zeile += (f"  Fehlvorschlag="
                      f"{default['negativ']['fehlvorschlagsquote']:.2%}")
        print(zeile)

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    konfiguration = {
        "gold": list(args.gold), "negativ_gold": args.negativ_gold,
        "corpus": str(corpus), "device": device,
        "topk_liste": sorted(args.topk_liste),
        "floor_liste": sorted(args.floor_liste), "rerank": args.rerank,
        "rerank_schwellen_liste": sorted(args.rerank_schwellen_liste),
        "rerank_topn": args.rerank_topn, "k_roh": k_roh,
        "policies": sorted(args.policies), "hoch_liste": sorted(args.hoch_liste),
        "fusion": args.fusion, "k_rrf_liste": sorted(args.k_rrf_liste),
        "trigger_k": args.trigger_k,
        "max_faelle": args.max_faelle,
    }
    bericht = baue_bericht(
        konfiguration=konfiguration, gold_dateien=gold_dateien,
        encoder_name=enc.name, index=idx, reranker_name=reranker_name,
        ergebnisse=ergebnisse, zeitstempel=ts, negativ_gold=negativ_gold)
    inhalt = json.dumps(bericht, ensure_ascii=False, indent=2) + "\n"
    if args.bericht:
        pfad = Path(args.bericht) / f"eval_harness_{zeitstempel_dateiname(ts)}.json"
        atomar_schreiben(pfad, inhalt)
        print(f"Bericht: {pfad}", file=sys.stderr)
    if args.benchmark_out:
        atomar_schreiben(Path(args.benchmark_out), inhalt)
        print(f"Benchmark: {args.benchmark_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
