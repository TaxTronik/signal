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

Aufruf (Beispiele):
  python tools/eval_harness.py --corpus C:/dev/taxtronikexctract/corpus \\
      --rerank aus --bericht out/berichte
  python tools/eval_harness.py --corpus C:/dev/taxtronikexctract/corpus \\
      --rerank beides --benchmark-out benchmarks/harness-baseline.json

Exit-Codes: 0 = ok, 2 = Fehler (Index/Gold/Reranker fehlt o. Ä.).
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

# Recall-Tiefen des Berichts (fest, damit Berichte über Läufe vergleichbar bleiben).
ZIEL_KS = (1, 5, 10, 50)

# Ist-Default der Produktion (pipeline.py / rerank_schwelle) — wird in der
# Tabelle markiert und im Bericht separat ausgewiesen.
IST_DEFAULT_FLOOR = 0.35
IST_DEFAULT_SCHWELLE = 0.10

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


def konfig_kandidaten(treffer: list[dict[str, Any]], *, floor: float, topk: int,
                      query: str | None = None,
                      rerank_scores: dict[tuple[str, str], float] | None = None,
                      schwelle: float | None = None,
                      rerank_topn: int = RERANK_TOPN) -> list[str]:
    """Finale Kandidaten-IDs EINER Query unter EINER Konfiguration.

    rerank=aus:  Floor-Nachfilter → Top-k.
    rerank=an :  Floor-Nachfilter → Top-N → nach Cross-Encoder-Score absteigend
                 (stabil: Gleichstand behält Embedding-Reihenfolge, wie
                 ``rerank_normen``) → Abstention-Schwelle → Top-k.
    """
    kand = floor_filter(treffer, floor)
    if rerank_scores is None:
        return [t["norm_id"] for t in kand[:topk]]
    if schwelle is None:
        raise ValueError("rerank=an braucht eine Schwelle")
    kand = kand[:rerank_topn]
    geordnet = sorted(kand, key=lambda t: rerank_scores[(query, t["norm_id"])],
                      reverse=True)
    return [t["norm_id"] for t in geordnet
            if rerank_scores[(query, t["norm_id"])] >= schwelle][:topk]


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


def sammle_rerank_scores(reranker, doc_fn, roh: dict[str, dict[str, Any]],
                         topn: int) -> tuple[dict[tuple[str, str], float],
                                             dict[str, float]]:
    """Cross-Encoder-Scores je (query, kandidat) — EIN Pass über die rohen
    Top-N, gecacht für alle Floor/Schwellen-Kombinationen (jede gefilterte
    Kandidatenmenge ist Teilmenge der rohen Top-N)."""
    scores: dict[tuple[str, str], float] = {}
    latenz: dict[str, float] = {}
    for q in sorted(roh):                  # deterministische Abarbeitung
        kand = roh[q]["treffer"][:topn]
        t0 = time.perf_counter()
        sc = reranker.score([(q, doc_fn(t["norm_id"])) for t in kand]) if kand else []
        latenz[q] = (time.perf_counter() - t0) * 1000.0
        for t, s in zip(kand, sc):
            scores[(q, t["norm_id"])] = float(s)
    return scores, latenz


# --- Sweep (pur über Rohdaten — testbar ohne Modelle) ---------------------------

def _params_schluessel(p: dict[str, Any]) -> str:
    return f"{p['rerank']}|floor={p['floor']}|schwelle={p['schwelle']}|topk={p['topk']}"


def ist_default_konfig(p: dict[str, Any]) -> bool:
    """Markiert die Ist-Default-Zeile der Produktion (0,35 / 0,10)."""
    if abs(p["floor"] - IST_DEFAULT_FLOOR) > 1e-9:
        return False
    if p["rerank"] == "an":
        return abs((p["schwelle"] or -1) - IST_DEFAULT_SCHWELLE) < 1e-9
    return True


def sweep(faelle: list[tuple[str, str, set[str]]],
          roh: dict[str, dict[str, Any]], *,
          topk_liste: list[int], floor_liste: list[float],
          rerank_modi: list[str], schwellen_liste: list[float],
          rerank_scores: dict[tuple[str, str], float] | None = None,
          rerank_latenz: dict[str, float] | None = None,
          rerank_topn: int = RERANK_TOPN) -> list[dict[str, Any]]:
    """Kartesisches Produkt aller Konfigurationen über die GLEICHEN Rohdaten.

    ``faelle``: (gold_name, query, ziele). Rückgabe absteigend sortiert nach
    Recall@5, dann MRR (feste Tie-Breaker ⇒ deterministische Reihenfolge).
    """
    ergebnisse: list[dict[str, Any]] = []
    for topk, floor, modus in product(sorted(topk_liste), sorted(floor_liste),
                                      sorted(rerank_modi)):
        schwellen = sorted(schwellen_liste) if modus == "an" else [None]
        for schwelle in schwellen:
            if modus == "an" and rerank_scores is None:
                raise ValueError("rerank=an im Sweep, aber keine Reranker-Scores gesammelt")
            pro_fall: list[dict[str, Any]] = []
            pro_gold: dict[str, list[dict[str, Any]]] = {}
            for gold_name, q, ziele in faelle:
                kand = konfig_kandidaten(
                    roh[q]["treffer"], floor=floor, topk=topk, query=q,
                    rerank_scores=rerank_scores if modus == "an" else None,
                    schwelle=schwelle, rerank_topn=rerank_topn)
                lat = roh[q]["latenz_ms"]
                if modus == "an":
                    lat += (rerank_latenz or {}).get(q, 0.0)
                fall = {"rang": treffer_rang(kand, ziele),
                        "abstention": not kand, "latenz_ms": lat}
                pro_fall.append(fall)
                pro_gold.setdefault(gold_name, []).append(fall)
            params = {"rerank": modus, "floor": floor, "schwelle": schwelle,
                      "topk": topk,
                      "rerank_topn": rerank_topn if modus == "an" else None}
            ergebnisse.append({
                "params": params,
                "ist_default": ist_default_konfig(params),
                "aggregat": metriken(pro_fall),
                "pro_gold": {n: metriken(f) for n, f in sorted(pro_gold.items())},
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
                 zeitstempel: str) -> dict[str, Any]:
    """Vollständiger, reproduzierbarer Bericht: Konfiguration + Gold-Versionen +
    Encoder + Index-Fingerprint + Zeitstempel + sortierte Sweep-Ergebnisse."""
    return {
        "harness": "eval_harness/1",
        "erstellt_am": zeitstempel,
        "konfiguration": konfiguration,
        "gold": {g["name"]: {"version": g["version"], "stand": g["stand"],
                             "n": len(g["faelle"])} for g in gold_dateien},
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
    """Menschliche Sweep-Tabelle (beste Konfiguration zuerst, Default mit *)."""
    kopf = (f"{'':1} {'rerank':6} {'floor':>5} {'schwelle':>8} {'topk':>4} "
            f"{'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@50':>6} {'MRR':>6} "
            f"{'Abst':>6} {'Lat ms':>7}")
    zeilen = [kopf, "-" * len(kopf)]
    for e in ergebnisse:
        p, a = e["params"], e["aggregat"]
        schwelle = "-" if p["schwelle"] is None else f"{p['schwelle']:.2f}"
        zeilen.append(
            f"{'*' if e['ist_default'] else '':1} {p['rerank']:6} {p['floor']:>5.2f} "
            f"{schwelle:>8} "
            f"{p['topk']:>4} "
            f"{a['recall@1']:>6.3f} {a['recall@5']:>6.3f} {a['recall@10']:>6.3f} "
            f"{a['recall@50']:>6.3f} {a['mrr']:>6.3f} {a['abstention']:>6.2f} "
            f"{a['median_latenz_ms']:>7.1f}")
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
    ap.add_argument("--corpus", default="corpus",
                    help="Korpus-Verzeichnis (graph.sqlite + embedding/)")
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
    faelle = [(g["name"], q, z) for g in gold_dateien for q, z in g["faelle"]]
    k_roh = max(max(args.topk_liste), args.rerank_topn if "an" in modi else 0)

    print(f"Eval-Harness: {len(faelle)} Fälle aus {len(gold_dateien)} Gold-Dateien | "
          f"Encoder {enc.name} (device={device}) | Index {len(idx.ids)} Vektoren "
          f"(fingerprint={idx.graph_fingerprint or '—'})")

    # Warmlauf VOR der Zeitmessung (Lazy-Load des Modells gehört nicht in die
    # Query-Latenz; gemessen wird der eingeschwungene Zustand).
    try:
        enc.encode(["Warmlauf"])
    except RuntimeError as exc:
        print(f"Encoder lädt nicht: {exc}", file=sys.stderr)
        return 2
    roh = sammle_rohdaten(idx, enc, [q for _, q, _ in faelle], k_roh)

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
        print(f"Reranker: {reranker_name} (Top-N {args.rerank_topn}, "
              f"{len(roh)} Queries) …")
        rerank_scores, rerank_latenz = sammle_rerank_scores(
            rr, doc_fn, roh, args.rerank_topn)

    ergebnisse = sweep(
        faelle, roh, topk_liste=args.topk_liste, floor_liste=args.floor_liste,
        rerank_modi=modi, schwellen_liste=args.rerank_schwellen_liste,
        rerank_scores=rerank_scores, rerank_latenz=rerank_latenz,
        rerank_topn=args.rerank_topn)

    print()
    print(_tabelle(ergebnisse))
    beste = ergebnisse[0]
    print(f"\nBeste Konfiguration: {_params_schluessel(beste['params'])}")
    for name, m in beste["pro_gold"].items():
        print(f"  {name}: n={m['n']}  R@5={m['recall@5']:.3f}  "
              f"MRR={m['mrr']:.3f}  Abst={m['abstention']:.2f}")
    for default in (e for e in ergebnisse if e["ist_default"]):
        a = default["aggregat"]
        print(f"Ist-Default ({_params_schluessel(default['params'])}): "
              f"R@5={a['recall@5']:.3f}  MRR={a['mrr']:.3f}  "
              f"Abst={a['abstention']:.2f}")

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    konfiguration = {
        "gold": list(args.gold), "corpus": str(corpus), "device": device,
        "topk_liste": sorted(args.topk_liste),
        "floor_liste": sorted(args.floor_liste), "rerank": args.rerank,
        "rerank_schwellen_liste": sorted(args.rerank_schwellen_liste),
        "rerank_topn": args.rerank_topn, "k_roh": k_roh,
        "max_faelle": args.max_faelle,
    }
    bericht = baue_bericht(
        konfiguration=konfiguration, gold_dateien=gold_dateien,
        encoder_name=enc.name, index=idx, reranker_name=reranker_name,
        ergebnisse=ergebnisse, zeitstempel=ts)
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
