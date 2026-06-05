"""Test 3 — rerank-on-oracle: Obergrenze von Scope+Rerank (vorab fixierte Schwellen).

Frage: Holt man den Intra-Law-Vorteil des Rerankers MIT der Cross-Law-Immunität des
Scopings, wenn der Reranker NUR in-scope-Kandidaten sieht (perfektes Law-Scoping)?

Oracle-Scope = laws(ALLE Gold-Ziele) je Fall (cross-law lässt alle Zielgesetze zu —
sonst künstlich schlechte Obergrenze). Kandidaten = Normen im Scope (graph ∪ in-scope-
Anker), Top-K -> Cross-Encoder -> Norm-Aggregation (max). Vergleich gegen rerank50 (flach).

VORAB FIXIERTE SCHWELLEN (nicht nachträglich justiert):
  stark      MRR >= 0.34
  brauchbar  MRR 0.31..0.34 bei stabiler Coverage
  schwach    MRR < 0.31  oder  Gesetz-Splits kippen
Architektur-Beleg (alle 5):
  1) MRR > rerank50@786 UND > rerank50@1015
  2) Coverage 786->1015 schadet nicht (Delta >= -0.01)
  3) UStG UND EStG nicht schlechter als rerank50
  4) cross-law-Subset nicht beschädigt
  5) cross-class-Quote Top-5/Top-20 sinkt ggü. rerank50

Eine Messung über beide Bestände. Report: MRR/R@5/R@20/R@50, pro Gesetz, cross- vs
intra-law, cross-class Top-5/20, Residuen.
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

STARK, BRAUCHBAR = 0.34, 0.31


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


def _rank(ranked: list[str], ziele: set[str]):
    return next((i + 1 for i, p in enumerate(ranked) if p in ziele), None)


def _mrr(raenge):
    n = len(raenge) or 1
    return round(sum(1.0 / r for r in raenge if r) / n, 3)


def _rk(raenge, k):
    n = len(raenge) or 1
    return round(sum(1 for r in raenge if r and r <= k) / n, 3)


def _crossclass(ranked: list[str], gold_laws: set[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if _law(p) not in gold_laws) / len(top)


def lauf(anker_pfad, gold, enc, idx, graph, reranker, node_text, K):
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

    def rerank(query, cands):
        pairs, owner = [], []
        for nm in cands:
            for d in docs_of(nm):
                pairs.append((query, d)); owner.append(nm)
        if not pairs:
            return list(cands)
        scores = reranker.score(pairs)
        best = {}
        for nm, s in zip(owner, scores):
            if s > best.get(nm, -1e18):
                best[nm] = s
        return sorted(best, key=best.get, reverse=True)

    modi = ["rerank50", "oracle_rr"]
    raenge = {m: [] for m in modi}
    cc5 = {m: [] for m in modi}
    cc20 = {m: [] for m in modi}
    pro_law = {m: defaultdict(list) for m in modi}
    cross_sub = {m: [] for m in modi}
    intra_sub = {m: [] for m in modi}
    residual = []

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

        sc_none = dict(graph_sim)
        for p, (s, _l) in anker_max.items():
            sc_none[p] = max(sc_none.get(p, 0.0), s)
        none_rank = sorted(sc_none, key=sc_none.get, reverse=True)

        sc_or = dict(graph_sim)
        for p, (s, l) in anker_max.items():
            if l in gold_laws:
                sc_or[p] = max(sc_or.get(p, 0.0), s)
        oracle_rank = sorted(sc_or, key=sc_or.get, reverse=True)
        cand_or = [n for n in oracle_rank if _law(n) in gold_laws][:K]

        ranked = {"rerank50": rerank(satz, none_rank[:50]),
                  "oracle_rr": rerank(satz, cand_or)}
        prim_law = _law(sorted(ziele)[0])
        ist_cross = len(gold_laws) > 1
        for m in modi:
            r = _rank(ranked[m], ziele)
            raenge[m].append(r)
            cc5[m].append(_crossclass(ranked[m], gold_laws, 5))
            cc20[m].append(_crossclass(ranked[m], gold_laws, 20))
            pro_law[m][prim_law].append(r)
            (cross_sub if ist_cross else intra_sub)[m].append(r)
        r_or = _rank(ranked["oracle_rr"], ziele)
        if not r_or or r_or > 5:
            residual.append((sorted(z.replace("norm:", "") for z in ziele), r_or,
                             "cross" if ist_cross else "intra"))

    res = {"n": len(gold), "n_anker": len(inh)}
    for m in modi:
        res[m] = {"mrr": _mrr(raenge[m]), "r5": _rk(raenge[m], 5), "r20": _rk(raenge[m], 20),
                  "r50": _rk(raenge[m], 50),
                  "cc5": round(float(np.mean(cc5[m])), 3), "cc20": round(float(np.mean(cc20[m])), 3),
                  "cross_mrr": _mrr(cross_sub[m]), "intra_mrr": _mrr(intra_sub[m]),
                  "pro_law": {l: _mrr(v) for l, v in sorted(pro_law[m].items())}}
    res["residual_oracle_rr"] = residual
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Test 3: rerank-on-oracle (Obergrenze)")
    ap.add_argument("--gold", default="discovery/klausuren_v2.yaml")
    ap.add_argument("--anker-alt", default="discovery/api/cards_training_pairs.yaml")
    ap.add_argument("--anker-neu", default="discovery/api/cards_training_pairs_merged.yaml")
    ap.add_argument("--reranker", default="bge-reranker-v2-m3")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--save", default="benchmarks/test3-oracle-rerank.json")
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
        out[tag] = lauf(pfad, gold, enc, idx, graph, reranker, node_text, args.k)

    # --- Report ---
    def zeile(tag, m):
        r = out[tag][m]
        print(f"  {tag:4} {m:10} MRR {r['mrr']:.3f}  R@5 {r['r5']:.2f}  R@20 {r['r20']:.2f}  "
              f"R@50 {r['r50']:.2f}  | cc5 {r['cc5']:.2f} cc20 {r['cc20']:.2f}  "
              f"| cross {r['cross_mrr']:.3f} intra {r['intra_mrr']:.3f}")

    print(f"\n# Test 3 — rerank-on-oracle (n={gold and len(gold)}, K={args.k}, vorab fixierte Schwellen)\n")
    for tag in ("786", "1015"):
        for m in ("rerank50", "oracle_rr"):
            zeile(tag, m)
    print("\n# Pro Gesetz (MRR)  rerank50 -> oracle_rr")
    for tag in ("786", "1015"):
        laws = sorted(out[tag]["rerank50"]["pro_law"])
        for l in laws:
            a = out[tag]["rerank50"]["pro_law"][l]
            b = out[tag]["oracle_rr"]["pro_law"].get(l, 0.0)
            print(f"  {tag:4} {l:7} {a:.3f} -> {b:.3f}")

    # --- Schwellen / Architektur-Belege (vorab fixiert) ---
    ref50_786 = out["786"]["rerank50"]["mrr"]
    ref50_1015 = out["1015"]["rerank50"]["mrr"]
    orr_786 = out["786"]["oracle_rr"]
    orr_1015 = out["1015"]["oracle_rr"]
    mrr = orr_1015["mrr"]
    cov_delta = round(orr_1015["mrr"] - orr_786["mrr"], 3)

    krit = {
        "1_mrr>beide_rerank50": mrr > ref50_786 and mrr > ref50_1015,
        "2_coverage_haelt(>=-0.01)": cov_delta >= -0.01,
        "3_UStG&EStG_nicht_schlechter": (
            orr_1015["pro_law"].get("UStG", 0) >= out["1015"]["rerank50"]["pro_law"].get("UStG", 0) and
            orr_1015["pro_law"].get("EStG", 0) >= out["1015"]["rerank50"]["pro_law"].get("EStG", 0)),
        "4_crosslaw_nicht_beschaedigt": orr_1015["cross_mrr"] >= out["1015"]["rerank50"]["cross_mrr"],
        "5_crossclass_top5_sinkt": orr_1015["cc5"] < out["1015"]["rerank50"]["cc5"],
    }
    stufe = ("STARK" if mrr >= STARK else
             "BRAUCHBAR" if mrr >= BRAUCHBAR and cov_delta >= -0.01 else "SCHWACH")

    print(f"\n# Urteil (vorab fixiert)")
    print(f"  oracle_rr MRR @1015 = {mrr:.3f}  (Ref rerank50: 786 {ref50_786:.3f} / 1015 {ref50_1015:.3f})")
    print(f"  Coverage-Delta oracle_rr 786->1015 = {cov_delta:+.3f}")
    print(f"  cross-class Top-5: rerank50 {out['1015']['rerank50']['cc5']:.2f} -> "
          f"oracle_rr {orr_1015['cc5']:.2f} | Top-20: "
          f"{out['1015']['rerank50']['cc20']:.2f} -> {orr_1015['cc20']:.2f}")
    print(f"  STUFE: {stufe}")
    for k, v in krit.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  Architektur-Beleg (alle 5): {'JA' if all(krit.values()) else 'NEIN'}")

    resid = out["1015"]["residual_oracle_rr"]
    print(f"\n# Residuen oracle_rr @1015 (Rang>5/raus): {len(resid)}")
    for z, r, kl in resid[:15]:
        print(f"  [{kl}] {z}  rang={r}")

    out["urteil"] = {"stufe": stufe, "mrr_1015": mrr, "cov_delta": cov_delta,
                     "ref_rerank50": {"786": ref50_786, "1015": ref50_1015},
                     "kriterien": krit, "architektur_beleg": all(krit.values()),
                     "schwellen": {"stark": STARK, "brauchbar": BRAUCHBAR}}
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
