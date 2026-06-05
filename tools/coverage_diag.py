"""Coverage-Backfire-Diagnose: WARUM senkt der 1015er-Bestand die Decke gegenüber 786?

Drei Gegenhypothesen sauber trennen (statt „Flat-Index" zu behaupten):
  H1  Cutoff vor Aggregation  -> aus dem Code WIDERLEGT (anker_max aggregiert über
      ALLE Anker vor jedem Schnitt; none R@50 = Norm-Rang). Hier zusätzlich der
      Graph-Fenster-Test: none R@50 bei graph-k=300 vs. k=voll (Reachability).
  H2  Hub-Verdrängung          -> wenige breite Schema-Normen (UStG:1/3/10/15,
      EStG:10/20/32 …) verdrängen die feineren Ziele. Test: Verdränger-Konzentration.
  H3  echte Flat-Verdünnung    -> Verdränger streuen diffus über viele Normen.

Methode: identische Scoring-Logik wie scoped_eval (graph_sim ∪ anker_max per Norm,
max), einmal mit 786, einmal mit 1015 Ankern. Für Ziele, deren Norm-Rang sich
verschlechtert, werden die VERDRÄNGER (Normen, die über das Ziel rückten) getallyt
und nach Konzentration / Hub-Identität / neu-hinzugekommen klassifiziert.

Aufruf:
  python tools/coverage_diag.py --gold discovery/klausuren_v2.yaml \
      --anker-alt discovery/api/cards_training_pairs.yaml \
      --anker-neu discovery/api/cards_training_pairs_merged.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.graph import graph_oeffnen                        # noqa: E402
from risk_layer.idschema import para_level                        # noqa: E402

# Breite USt/ESt-Schema-Normen (Omnipräsenz-Verdacht, H2)
HUBS = {"norm:UStG:1", "norm:UStG:3", "norm:UStG:3a", "norm:UStG:4", "norm:UStG:10",
        "norm:UStG:13", "norm:UStG:15", "norm:EStG:2", "norm:EStG:10", "norm:EStG:15",
        "norm:EStG:19", "norm:EStG:20", "norm:EStG:32", "norm:EStG:4", "norm:EStG:9"}


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _gebiet(p: str) -> str:
    t = p.split(":")
    return t[1] if len(t) > 1 else "?"


def _lade_gold(p: Path):
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for f in d.get("faelle", []):
        q = _norm(str(f.get("query", "")))
        ziele = {para_level(z) for z in (f.get("ziele") or []) if z}
        if q and ziele:
            out.append((q, ziele))
    return out


def _lade_anker(p: Path, gold_q: set[str]):
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    inh, ziel = [], []
    for k in d.get("kandidaten", []):
        t = str(k.get("inhalt", "")).strip()
        z = para_level(str(k.get("ziel_norm_id") or k.get("quell_norm") or "").strip())
        if t and z and _norm(t) not in gold_q:
            inh.append(t)
            ziel.append(z)
    return inh, ziel


def _rang(ranking: list[str], ziele: set[str]):
    for i, p in enumerate(ranking):
        if p in ziele:
            return i + 1, p
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Coverage-Backfire-Diagnose")
    ap.add_argument("--gold", default="discovery/klausuren_v2.yaml")
    ap.add_argument("--anker-alt", default="discovery/api/cards_training_pairs.yaml")
    ap.add_argument("--anker-neu", default="discovery/api/cards_training_pairs_merged.yaml")
    ap.add_argument("--k", type=int, default=300, help="Graph-§-Text-Fenster")
    ap.add_argument("--save", default="benchmarks/coverage-diag.json")
    args = ap.parse_args()

    gold = _lade_gold(Path(args.gold))
    gq = {q for q, _ in gold}
    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen("bge-m3", device=dev)
    graph = graph_oeffnen()
    print(f"Gold {len(gold)} | baue Graph-Index …", file=sys.stderr)
    idx = EmbeddingIndex.build(graph, enc, katalog=None)
    n_nodes = 3000   # > Korpus (~2899 §§): "voll" = keine Graph-Reachability-Grenze

    saetze = [{"alt": {}, "neu": {}} for _ in gold]
    besta = {}
    for tag, pfad in (("alt", args.anker_alt), ("neu", args.anker_neu)):
        inh, ziel = _lade_anker(Path(pfad), gq)
        A = np.asarray(enc.encode(inh), dtype=np.float32)
        besta[tag] = (A, ziel, set(ziel))
        print(f"{tag}: {len(inh)} Anker / {len(set(ziel))} Normen", file=sys.stderr)

    def ranking(satz, tag, k):
        g = idx.suche(satz, enc, k=k, min_sim=0.0)
        sc = {}
        for t in g:
            sc.setdefault(para_level(t["norm_id"]), t["sim"])
        A, ziel, _ = besta[tag]
        qv = np.asarray(enc.encode([satz])[0], dtype=np.float32)
        sims = A @ qv
        amax: dict[str, float] = {}
        for i, p in enumerate(ziel):
            if sims[i] > amax.get(p, -2.0):
                amax[p] = float(sims[i])
        for p, s in amax.items():
            sc[p] = max(sc.get(p, 0.0), s)
        return sorted(sc, key=sc.get, reverse=True)

    neu_norms = besta["neu"][2]
    alt_norms = besta["alt"][2]
    r50 = {"alt": 0, "neu": 0, "neu_kfull": 0}
    recall_miss = {"alt": set(), "neu": set()}
    verdraenger = Counter()         # Norm -> wie oft über ein verschlechtertes Ziel gerückt
    verdraenger_faelle = Counter()  # Norm -> in wie vielen Fällen
    neu_miss_detail = []
    rangverlust = []

    for i, (satz, ziele) in enumerate(gold):
        ra = ranking(satz, "alt", args.k)
        rn = ranking(satz, "neu", args.k)
        rank_a, _ = _rang(ra, ziele)
        rank_n, ziel_n = _rang(rn, ziele)
        rank_nf, _ = _rang(ranking(satz, "neu", n_nodes), ziele)
        if rank_a and rank_a <= 50:
            r50["alt"] += 1
        if rank_n and rank_n <= 50:
            r50["neu"] += 1
        if rank_nf and rank_nf <= 50:
            r50["neu_kfull"] += 1
        if not rank_a or rank_a > 50:
            recall_miss["alt"].add(i)
        if not rank_n or rank_n > 50:
            recall_miss["neu"].add(i)
        # Verdrängung: Ziel-Rang verschlechtert -> wer rückte über das (best-platzierte) Ziel?
        if rank_a and rank_n and rank_n > rank_a:
            ziel_best = ra[rank_a - 1]
            alt_above = set(ra[:rank_a - 1])
            neu_above = set(rn[:rank_n - 1])
            disp = neu_above - alt_above - ziele
            rangverlust.append((i, rank_a, rank_n, sorted(disp)[:8]))
            for d in disp:
                verdraenger[d] += 1
            for d in set(disp):
                verdraenger_faelle[d] += 1

    n = len(gold)
    print(f"\n# Coverage-Diagnose (n={n}, graph-k={args.k})\n")
    print(f"none R@50:  alt {r50['alt']/n:.3f}  ->  neu {r50['neu']/n:.3f}"
          f"   | neu@graph-k=voll {r50['neu_kfull']/n:.3f}")
    print(f"recall_miss(>50): alt {len(recall_miss['alt'])}  neu {len(recall_miss['neu'])}"
          f"  | NEU dazugekommen: {len(recall_miss['neu']-recall_miss['alt'])}"
          f"  | weg: {len(recall_miss['alt']-recall_miss['neu'])}")

    print("\n# Neu herausgefallene Ziele (war ≤50 mit alt, jetzt >50/raus):")
    for i in sorted(recall_miss["neu"] - recall_miss["alt"]):
        ziele = gold[i][1]
        ra = ranking(gold[i][0], "alt", args.k)
        rank_a, _ = _rang(ra, ziele)
        rn = ranking(gold[i][0], "neu", args.k)
        rank_n, _ = _rang(rn, ziele)
        rnf = ranking(gold[i][0], "neu", n_nodes)
        rank_nf, _ = _rang(rnf, ziele)
        hat_anker = bool(ziele & neu_norms)
        z = sorted(z.replace("norm:", "") for z in ziele)
        print(f"  Fall {i:2d} {z}  rang alt {rank_a} -> neu {rank_n} (k=voll {rank_nf})"
              f"  | Ziel hat Anker(neu): {hat_anker}")

    print(f"\n# Verdränger-Struktur (H2 Hub vs H3 diffus)  — {sum(verdraenger.values())} Verdrängungs-Ereignisse")
    top = verdraenger.most_common(15)
    gesamt = sum(verdraenger.values()) or 1
    top5_share = sum(c for _, c in verdraenger.most_common(5)) / gesamt
    print(f"  Konzentration: Top-5 Verdränger = {top5_share:.0%} aller Ereignisse | "
          f"distinkte Verdränger-Normen: {len(verdraenger)}")
    print(f"  {'Norm':16} {'Ereign.':>7} {'Fälle':>5}  neu?  hub?")
    for d, c in top:
        print(f"  {d.replace('norm:',''):16} {c:>7} {verdraenger_faelle[d]:>5}  "
              f"{'NEU' if d not in alt_norms else '   '}  {'HUB' if d in HUBS else ''}")
    hub_ev = sum(c for d, c in verdraenger.items() if d in HUBS)
    neu_ev = sum(c for d, c in verdraenger.items() if d not in alt_norms)
    print(f"\n  Anteil Verdrängungen durch HUB-Normen: {hub_ev/gesamt:.0%}")
    print(f"  Anteil Verdrängungen durch NEU-hinzugekommene Normen: {neu_ev/gesamt:.0%}")
    print(f"  -> Befund: " + (
        "HUB-getrieben (H2)" if top5_share >= 0.4 or hub_ev/gesamt >= 0.3
        else "diffus/Flat (H3)" if len(verdraenger) > 0.6 * gesamt else "gemischt"))

    out = {"n": n, "k": args.k, "r50": {k: v / n for k, v in r50.items()},
           "recall_miss": {k: len(v) for k, v in recall_miss.items()},
           "neu_miss": sorted(recall_miss["neu"] - recall_miss["alt"]),
           "top_verdraenger": [{"norm": d, "ereignisse": c, "faelle": verdraenger_faelle[d],
                                "neu": d not in alt_norms, "hub": d in HUBS} for d, c in top],
           "top5_share": top5_share, "hub_share": hub_ev / gesamt, "neu_share": neu_ev / gesamt}
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
