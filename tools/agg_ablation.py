"""Aggregations-Ablation auf rerank-on-oracle: ist der EStG-Knick / der 786->1015-
Verlust ein `max`-Ordnungsstatistik-Artefakt statt echter Intra-Law-Granularität?

`max` über die Doc-Scores einer Norm bevorteilt anker-reiche Normen rein
kombinatorisch (mehr Ziehungen → höheres Maximum). Geprüft als ABLATION auf dem
GLEICHEN Reranker-Pass (Doc-Scores einmal berechnet, mehrfach aggregiert — billig):

  max          max(scores)                         (aktueller Default)
  top2_mean    mean(Top-2 scores)                  (param-frei; gegen Einzel-Glückstreffer)
  normed       max / log(2 + n_docs)               (param-frei; Ordnungsstatistik-Korrektur)
  pen0.5/1.0   max - λ·log(1 + n_docs)             (λ exploratorisch, NICHT auf Test getunt)

Architektur = rerank-on-oracle (Scope = laws(alle Gold-Ziele)). Beide Bestände.
Frage: schrumpft EStG-Knick (rerank50 1015 EStG 0,321 -> oracle_rr 0,292) und der
Coverage-Verlust (oracle_rr 786 0,444 -> 1015 0,381)?  -> dann teils Aggregationsartefakt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.discovery import iter_paragraphen                 # noqa: E402
from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.graph import graph_oeffnen                        # noqa: E402
from risk_layer.idschema import para_level                        # noqa: E402
from risk_layer.reranker import reranker_oeffnen                  # noqa: E402


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _law(p: str) -> str:
    t = p.split(":")
    return t[1] if len(t) > 1 else "?"


def _lade_gold(p: Path):
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for f in d.get("faelle", []):
        q = str(f.get("query", ""))
        ziele = {para_level(z) for z in (f.get("ziele") or []) if z}
        if _norm(q) and ziele:
            out.append((q, ziele))
    return out


def _lade_anker(p: Path, gold_q):
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    inh, ziel = [], []
    for k in d.get("kandidaten", []):
        t = str(k.get("inhalt", "")).strip()
        z = para_level(str(k.get("ziel_norm_id") or k.get("quell_norm") or "").strip())
        if t and z and _norm(t) not in gold_q:
            inh.append(t); ziel.append(z)
    return inh, ziel


AGGS = {
    "max":       lambda ss: max(ss),
    "top2_mean": lambda ss: float(np.mean(sorted(ss, reverse=True)[:2])),
    "normed":    lambda ss: max(ss) / math.log(2 + len(ss)),
    "pen0.5":    lambda ss: max(ss) - 0.5 * math.log(1 + len(ss)),
    "pen1.0":    lambda ss: max(ss) - 1.0 * math.log(1 + len(ss)),
}


def _mrr(r): n = len(r) or 1; return round(sum(1.0/x for x in r if x)/n, 3)  # noqa: E704
def _rk(r, k): n = len(r) or 1; return round(sum(1 for x in r if x and x <= k)/n, 3)  # noqa: E704


def lauf(anker_pfad, gold, enc, idx, reranker, node_text, K):
    inh, ziel = _lade_anker(Path(anker_pfad), {_norm(q) for q, _ in gold})
    A = np.asarray(enc.encode(inh), dtype=np.float32)
    a_law = [_law(z) for z in ziel]
    card_docs = defaultdict(list)
    for t, z in zip(inh, ziel):
        card_docs[z].append(t)

    def docs_of(nm):
        ds = list(card_docs.get(nm, []))
        if node_text.get(nm):
            ds.append(node_text[nm][:400])
        return ds or [nm]

    raenge = {a: [] for a in AGGS}
    pro_law = {a: defaultdict(list) for a in AGGS}
    for satz, ziele in gold:
        g = idx.suche(satz, enc, k=300, min_sim=0.0)
        graph_sim = {}
        for t in g:
            graph_sim.setdefault(para_level(t["norm_id"]), t["sim"])
        qv = np.asarray(enc.encode([satz])[0], dtype=np.float32)
        sims = A @ qv
        anker_max = {}
        for i, p in enumerate(ziel):
            if sims[i] > anker_max.get(p, (-2.0, ""))[0]:
                anker_max[p] = (float(sims[i]), a_law[i])
        gold_laws = {_law(z) for z in ziele}
        sc = dict(graph_sim)
        for p, (s, l) in anker_max.items():
            if l in gold_laws:
                sc[p] = max(sc.get(p, 0.0), s)
        cand = [n for n in sorted(sc, key=sc.get, reverse=True) if _law(n) in gold_laws][:K]
        # EIN Reranker-Pass -> Doc-Scores je Norm
        pairs, owner = [], []
        for nm in cand:
            for d in docs_of(nm):
                pairs.append((satz, d)); owner.append(nm)
        scores = reranker.score(pairs) if pairs else []
        per = defaultdict(list)
        for nm, s in zip(owner, scores):
            per[nm].append(float(s))
        prim = _law(sorted(ziele)[0])
        for a, fn in AGGS.items():
            ranked = sorted(per, key=lambda nm: fn(per[nm]), reverse=True)
            r = next((i + 1 for i, p in enumerate(ranked) if p in ziele), None)
            raenge[a].append(r); pro_law[a][prim].append(r)
    res = {}
    for a in AGGS:
        res[a] = {"mrr": _mrr(raenge[a]), "r5": _rk(raenge[a], 5), "r20": _rk(raenge[a], 20),
                  "r50": _rk(raenge[a], 50),
                  "pro_law": {l: _mrr(v) for l, v in sorted(pro_law[a].items())}}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregations-Ablation (rerank-on-oracle)")
    ap.add_argument("--gold", default="discovery/klausuren_v2.yaml")
    ap.add_argument("--anker-alt", default="discovery/api/cards_training_pairs.yaml")
    ap.add_argument("--anker-neu", default="discovery/api/cards_training_pairs_merged.yaml")
    ap.add_argument("--reranker", default="bge-reranker-v2-m3")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--save", default="benchmarks/agg-ablation.json")
    args = ap.parse_args()

    gold = _lade_gold(Path(args.gold))
    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen("bge-m3", device=dev)
    graph = graph_oeffnen()
    print(f"Gold {len(gold)} | baue Index + Reranker …", file=sys.stderr)
    idx = EmbeddingIndex.build(graph, enc, katalog=None)
    reranker = reranker_oeffnen(args.reranker)
    node_text = {para_level(n["id"]): (n["text"] or "") for n in iter_paragraphen(graph)}

    out = {}
    for tag, pfad in (("786", args.anker_alt), ("1015", args.anker_neu)):
        print(f"--- {tag} ---", file=sys.stderr)
        out[tag] = lauf(pfad, gold, enc, idx, reranker, node_text, args.k)

    print(f"\n# Aggregations-Ablation rerank-on-oracle (n={len(gold)}, K={args.k})\n")
    print(f"{'Aggregation':11} {'786 MRR':>8} {'1015 MRR':>9} {'Δcov':>7} "
          f"{'786 EStG':>9} {'1015 EStG':>10} {'ΔEStG':>7}")
    base = out["786"]["max"]["mrr"]
    for a in AGGS:
        m786, m1015 = out["786"][a]["mrr"], out["1015"][a]["mrr"]
        e786 = out["786"][a]["pro_law"].get("EStG", 0.0)
        e1015 = out["1015"][a]["pro_law"].get("EStG", 0.0)
        print(f"{a:11} {m786:>8.3f} {m1015:>9.3f} {m1015-m786:>+7.3f} "
              f"{e786:>9.3f} {e1015:>10.3f} {e1015-e786:>+7.3f}")
    print(f"\n# R@5/R@20/R@50 @1015")
    for a in AGGS:
        r = out["1015"][a]
        print(f"  {a:11} R@5 {r['r5']:.2f}  R@20 {r['r20']:.2f}  R@50 {r['r50']:.2f}")
    print(f"\n# Referenz (max): EStG-Knick rerank50→oracle 0,321→0,292; Coverage-Verlust 0,444→0,381")
    print("# Frage: schrumpft ΔEStG / Δcov unter top2_mean/normed?  -> teils Aggregationsartefakt")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
