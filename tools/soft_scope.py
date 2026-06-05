"""Schritt 2 — graph-basierter Soft-Scope (deploybar; oracle nutzte das wahre Gesetz).

Scope-Schätzung (gewählt): **Graph-Top-N Gesetzes-Set (weich)** — die Gesetze der
anker-FREIEN §-Text-Top-N der Query (unverdünnt), nach Sim gewichtet -> Gesetzes-Set S.
Erweiterung: 1-Hop typisierte Relationen (verweist_auf/anwendbar_auf/ausgenommen_von/
konkretisiert) der Top-in-scope-Normen = Scharniernormen. Weiche Gewichtung der
Anker-Beiträge (graph_sim bleibt immer voll):
  alaw in S            -> Faktor 1.0
  norm graph-adjacent  -> Faktor β  (Scharnier-Allowance, kein Drop)
  sonst                -> Faktor γ  (Malus, kein Drop)
Dann Reranker auf Top-K. Architektur = Scope -> Kandidaten -> Reranker -> Aggregation.

Disziplin: β/γ NUR auf Dev (Organschaft v1 + AO v1, disjunkt von n=77) im Grid; dann
EINE Messung auf n=77. Vergleich gegen Schranken routed_rr (hart) und oracle_rr (perfekt).
N=50, top_laws=2, M=20 per Prinzip fixiert (nicht auf Test getunt).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
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

N_GRAPH, TOP_LAWS, M_ADJ, K = 50, 2, 20, 200
RELATIONEN = ("verweist_auf", "anwendbar_auf", "ausgenommen_von", "konkretisiert")
GRID = [(b, g) for b in (0.6, 0.8) for g in (0.2, 0.4)]


def _norm(s): return " ".join((s or "").lower().split())  # noqa: E704
def _law(p): t = p.split(":"); return t[1] if len(t) > 1 else "?"  # noqa: E704


def _lade_gold(p: Path):
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for f in d.get("faelle", []):
        q = str(f.get("query", ""))
        ziele = {para_level(z) for z in (f.get("ziele") or []) if z}
        if _norm(q) and ziele:
            out.append((q, ziele))
    return out


def _lade_anker(p, gold_q):
    d = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
    inh, ziel = [], []
    for k in d.get("kandidaten", []):
        t = str(k.get("inhalt", "")).strip()
        z = para_level(str(k.get("ziel_norm_id") or k.get("quell_norm") or "").strip())
        if t and z and _norm(t) not in gold_q:
            inh.append(t); ziel.append(z)
    return inh, ziel


def _adjazenz(graph):
    adj = defaultdict(set)
    q = "SELECT src,dst,typ FROM kanten WHERE typ IN ({})".format(
        ",".join("'%s'" % r for r in RELATIONEN))
    for src, dst, _typ in graph.con.execute(q):
        if src and dst:
            adj[para_level(src)].add(para_level(dst))
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description="Soft-Scope (graph-Top-N Gesetzes-Set)")
    ap.add_argument("--dev", nargs="+",
                    default=["fixtures/gold/organschaft_v1.yaml", "fixtures/gold/ao_v1.yaml"])
    ap.add_argument("--test", default="discovery/klausuren_v2.yaml")
    ap.add_argument("--anker", default="discovery/api/cards_training_pairs_merged.yaml")
    ap.add_argument("--reranker", default="bge-reranker-v2-m3")
    ap.add_argument("--save", default="benchmarks/soft-scope.json")
    args = ap.parse_args()

    dev = []
    for p in args.dev:
        dev += _lade_gold(Path(p))
    test = _lade_gold(Path(args.test))
    gq = {_norm(q) for q, _ in dev + test}
    dev_q = {_norm(q) for q, _ in dev}                 # Leckage: Dev-Queries aus Ankern raus

    devc = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen("bge-m3", device=devc)
    graph = graph_oeffnen()
    print(f"Dev {len(dev)} | Test {len(test)} | baue Index + Reranker …", file=sys.stderr)
    idx = EmbeddingIndex.build(graph, enc, katalog=None)
    reranker = reranker_oeffnen(args.reranker)
    node_text = {para_level(n["id"]): (n["text"] or "") for n in iter_paragraphen(graph)}
    adj = _adjazenz(graph)

    inh, ziel = _lade_anker(args.anker, gq)
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

    def ctx_of(satz):
        g = idx.suche(satz, enc, k=300, min_sim=0.0)
        graph_sim, hits = {}, []
        for t in g:
            p = para_level(t["norm_id"])
            graph_sim.setdefault(p, t["sim"]); hits.append((p, t["sim"]))
        qv = np.asarray(enc.encode([satz])[0], dtype=np.float32)
        sims = A @ qv
        amax = {}
        for i, p in enumerate(ziel):
            if sims[i] > amax.get(p, (-2.0, ""))[0]:
                amax[p] = (float(sims[i]), a_law[i])
        law_mass = Counter()
        for p, s in hits[:N_GRAPH]:
            law_mass[_law(p)] += s
        S = {l for l, _ in law_mass.most_common(TOP_LAWS)}
        inscope_top = [p for p, _ in hits[:N_GRAPH] if _law(p) in S][:M_ADJ]
        adjacent = set().union(*(adj[p] for p in inscope_top)) if inscope_top else set()
        return {"satz": satz, "graph_sim": graph_sim, "amax": amax, "S": S,
                "adjacent": adjacent, "cache": {}}

    def rerank_cached(ctx, cands):
        cache = ctx["cache"]
        todo = [nm for nm in cands if nm not in cache]
        pairs, owner = [], []
        for nm in todo:
            for d in docs_of(nm):
                pairs.append((ctx["satz"], d)); owner.append(nm)
        if pairs:
            scores = reranker.score(pairs)
            tmp = defaultdict(lambda: -1e18)
            for nm, s in zip(owner, scores):
                tmp[nm] = max(tmp[nm], float(s))
            cache.update(tmp)
        return sorted(cands, key=lambda nm: cache.get(nm, -1e18), reverse=True)

    def soft_rank(ctx, beta, gamma):
        sc = dict(ctx["graph_sim"])
        for norm, (asim, alaw) in ctx["amax"].items():
            f = 1.0 if alaw in ctx["S"] else (beta if norm in ctx["adjacent"] else gamma)
            sc[norm] = max(sc.get(norm, 0.0), asim * f)
        return sorted(sc, key=sc.get, reverse=True)

    def rank_of(ranked, ziele):
        return next((i + 1 for i, p in enumerate(ranked) if p in ziele), None)

    def mrr(rs): n = len(rs) or 1; return round(sum(1.0/r for r in rs if r)/n, 3)  # noqa: E704

    print("baue Kontexte (Dev+Test) …", file=sys.stderr)
    dev_ctx = [(ctx_of(q), z) for q, z in dev]
    test_ctx = [(ctx_of(q), z) for q, z in test]

    # --- Dev-Grid (β/γ) ---
    print("\n# Dev-Kalibrierung (β/γ), MRR auf Dev:")
    best, best_mrr = None, -1.0
    for beta, gamma in GRID:
        rs = [rank_of(rerank_cached(c, soft_rank(c, beta, gamma)[:K]), z) for c, z in dev_ctx]
        m = mrr(rs)
        print(f"  β={beta} γ={gamma}  Dev-MRR {m:.3f}")
        if m > best_mrr:
            best, best_mrr = (beta, gamma), m
    print(f"  -> gewählt: β={best[0]} γ={best[1]} (Dev-MRR {best_mrr:.3f})")

    # --- EINE Test-Messung mit gewählter Config + Schranken ---
    def eval_test(rank_fn):
        rs, cc5, pro = [], [], defaultdict(list)
        for c, z in test_ctx:
            ranked = rank_fn(c, z)
            r = rank_of(ranked, z); rs.append(r)
            gl = {_law(x) for x in z}
            cc5.append(sum(1 for p in ranked[:5] if _law(p) not in gl) / max(len(ranked[:5]), 1))
            pro[_law(sorted(z)[0])].append(r)
        rk = lambda k: round(sum(1 for r in rs if r and r <= k)/(len(rs) or 1), 3)  # noqa: E731
        return {"mrr": mrr(rs), "r5": rk(5), "r20": rk(20), "r50": rk(50),
                "cc5": round(float(np.mean(cc5)), 3),
                "pro_law": {l: mrr(v) for l, v in sorted(pro.items())}}

    beta, gamma = best
    soft = eval_test(lambda c, z: rerank_cached(c, soft_rank(c, beta, gamma)[:K]))

    def oracle_rank(c, z):
        gl = {_law(x) for x in z}
        sc = dict(c["graph_sim"])
        for nm, (asim, al) in c["amax"].items():
            if al in gl:
                sc[nm] = max(sc.get(nm, 0.0), asim)
        cand = [n for n in sorted(sc, key=sc.get, reverse=True) if _law(n) in gl][:K]
        return rerank_cached(c, cand)
    oracle = eval_test(oracle_rank)

    def none_rank(c, z):
        sc = dict(c["graph_sim"])
        for nm, (asim, al) in c["amax"].items():
            sc[nm] = max(sc.get(nm, 0.0), asim)
        return rerank_cached(c, sorted(sc, key=sc.get, reverse=True)[:50])
    flat = eval_test(none_rank)

    print(f"\n# Test (n={len(test)}) — eine Messung, β={beta} γ={gamma}")
    print(f"{'Modus':12} {'MRR':>6} {'R@5':>5} {'R@20':>5} {'R@50':>5} {'cc5':>5}")
    for name, r in (("flat rerank50", flat), ("soft_scope", soft), ("oracle_rr", oracle)):
        print(f"{name:12} {r['mrr']:>6.3f} {r['r5']:>5.2f} {r['r20']:>5.2f} {r['r50']:>5.2f} {r['cc5']:>5.2f}")
    print(f"\n# Pro Gesetz (MRR)  flat -> soft -> oracle")
    for l in sorted(soft["pro_law"]):
        print(f"  {l:7} {flat['pro_law'].get(l,0):.3f} -> {soft['pro_law'][l]:.3f} -> {oracle['pro_law'].get(l,0):.3f}")
    spanne = (soft["mrr"] - flat["mrr"]) / max(oracle["mrr"] - flat["mrr"], 1e-9)
    print(f"\n# Soft-Scope schließt {spanne:.0%} der flat→oracle-Lücke "
          f"(flat {flat['mrr']:.3f} · soft {soft['mrr']:.3f} · oracle {oracle['mrr']:.3f})")
    print(f"  cross-class Top-5: flat {flat['cc5']:.2f} -> soft {soft['cc5']:.2f} -> oracle {oracle['cc5']:.2f}")

    out = {"dev_grid": {f"b{b}_g{g}": None for b, g in GRID}, "gewaehlt": {"beta": beta, "gamma": gamma,
           "dev_mrr": best_mrr}, "fix": {"N": N_GRAPH, "top_laws": TOP_LAWS, "M": M_ADJ, "K": K},
           "test": {"flat": flat, "soft_scope": soft, "oracle_rr": oracle},
           "luecke_geschlossen": spanne}
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
