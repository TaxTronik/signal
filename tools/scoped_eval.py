"""Reranker-Ablation (Deployment-Schicht): kann ein anker-bewusster Reranker den
verdünnten Card-Lift zurückholen, ohne harten Router? Beweiskräftig, nicht ermutigend.

Methodik (alle gegen graph-only, auf EINGEFRORENEM Gold, das der Reranker nie sah):
  graph    nur Gesetzestext (Recall-Boden)
  none     graph + ALLE Card-Anker global (der gemessene Crosstalk)
  routed   harter Law-Scope (graph-Top-k-Gebiete) — die tote Variante (Referenz)
  oracle   Law-Scope = Ziel-Gebiet (Obergrenze bei perfektem Routing)
  rerank20 none-Top-20 → bge-Reranker, NORM-aggregiert (max über die Docs der Norm)
  rerank50 none-Top-50 → dito

DECKE ZUERST: der Reranker kann nur sortieren, was im none-Top-N steht. none-R@20/R@50
ist sein theoretisches Maximum. Niedrig → Recall-Problem (eine Stufe früher), nicht
Ranking. Gepaarte Bootstrap-CIs über die Queries (n klein → Unsicherheit ausweisen).
Residualfälle klassifiziert: recall_miss / rerank_miss (cross_law|intra_law).

Aufruf:
  python tools/scoped_eval.py --gold discovery/opus/kstg/organschaft_gold.yaml \
      --anker discovery/api/cards_training_pairs.yaml --reranker bge-reranker-v2-m3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.discovery import iter_paragraphen                 # noqa: E402
from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.graph import graph_oeffnen                        # noqa: E402
from risk_layer.idschema import para_level                        # noqa: E402
from risk_layer.reranker import reranker_oeffnen                  # noqa: E402


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _gebiet(para: str) -> str:
    t = para.split(":")
    return t[1] if len(t) > 1 else "?"


def _lade_gold(pfad: Path):
    daten = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
    out = []
    for f in daten.get("faelle", []):
        q = _norm(str(f.get("query", "")))
        ziele = {para_level(z) for z in (f.get("ziele") or []) if z}
        if q and ziele:
            out.append((q, ziele))
    return out


def _lade_anker(pfad: Path, gold_q: set[str]):
    daten = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
    anker = []
    for d in daten.get("kandidaten", []):
        inh = str(d.get("inhalt", "")).strip()
        ziel = para_level(str(d.get("ziel_norm_id") or d.get("quell_norm") or "").strip())
        if inh and ziel and _norm(inh) not in gold_q:
            anker.append((inh, ziel, _gebiet(ziel)))
    return anker


def _rank_of(ranked: list[str], ziele: set[str]) -> int | None:
    return next((i + 1 for i, p in enumerate(ranked) if p in ziele), None)


def _metrik(raenge: list[int | None]) -> dict:
    n = len(raenge) or 1
    rk = lambda k: round(sum(1 for r in raenge if r and r <= k) / n, 3)  # noqa: E731
    return {"n": len(raenge), "r5": rk(5), "r10": rk(10), "r20": rk(20), "r50": rk(50),
            "mrr": round(sum(1 / r for r in raenge if r) / n, 3)}


def _rr(raenge: list[int | None]) -> np.ndarray:
    return np.array([1.0 / r if r else 0.0 for r in raenge], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reranker-Ablation (Deployment-Schicht)")
    ap.add_argument("--gold", default="fixtures/gold/organschaft_v1.yaml")
    ap.add_argument("--anker", default="discovery/api/cards_training_pairs.yaml")
    ap.add_argument("--route-top", type=int, default=3)
    ap.add_argument("--reranker", default="bge-reranker-v2-m3")
    ap.add_argument("--rerank-k", type=int, default=200, dest="rerank_k",
                    help="großes Rerank-Fenster (Diagnose-Wert aus beobachteter Verdrängung, "
                         "KEIN Produktiv-Default — der wird auf Dev/Latenz gewählt)")
    ap.add_argument("--boot", type=int, default=2000, help="Bootstrap-Resamples für die CIs")
    ap.add_argument("--save", default="benchmarks/rerank-ablation.json")
    args = ap.parse_args()
    rk = args.rerank_k

    gold = _lade_gold(Path(args.gold))
    anker = _lade_anker(Path(args.anker), {q for q, _ in gold})
    if not gold or not anker:
        print("Gold oder Anker leer.", file=sys.stderr)
        return 2

    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen("bge-m3", device=dev)
    graph = graph_oeffnen()
    print(f"Gold {len(gold)} | Anker {len(anker)} | Reranker {args.reranker} | baue Index …",
          file=sys.stderr)
    idx = EmbeddingIndex.build(graph, enc, katalog=None)
    reranker = reranker_oeffnen(args.reranker)
    if reranker is None:
        print(f"Reranker '{args.reranker}' nicht ladbar.", file=sys.stderr)
        return 2

    A = np.asarray(enc.encode([a[0] for a in anker]), dtype=np.float32)  # normalisiert -> Cosine
    a_ziel, a_law = [a[1] for a in anker], [a[2] for a in anker]
    card_docs: dict[str, list[str]] = {}
    for txt, ziel, _law in anker:
        card_docs.setdefault(ziel, []).append(txt)
    node_text = {para_level(n["id"]): (n["text"] or "") for n in iter_paragraphen(graph)}

    def docs_of(norm: str) -> list[str]:           # Norm-Aggregation: jede Quelle ein Doc
        ds = list(card_docs.get(norm, []))
        if node_text.get(norm):
            ds.append(node_text[norm][:400])
        return ds or [norm]

    def rerank(query: str, cands: list[str]) -> list[str]:
        pairs, owner = [], []
        for nm in cands:
            for d in docs_of(nm):
                pairs.append((query, d)); owner.append(nm)
        if not pairs:
            return list(cands)
        scores = reranker.score(pairs)
        best: dict[str, float] = {}
        # norm_score = max über die Docs der Norm. NOTE: max bevorteilt bei wachsendem
        # Bestand Hub-Normen mit vielen Docs (mehr Lose im Topf, höheres Maximum) — falls
        # Hubs nach Gold-100+ systematisch überranken, ist das der erste Verdächtige.
        for nm, s in zip(owner, scores):
            if s > best.get(nm, -1e18):
                best[nm] = s
        return sorted(best, key=best.get, reverse=True)

    modi = ["graph", "none", "routed", "oracle", "rerank20", "rerank50"]
    if rk not in (20, 50):
        modi.append(f"rerank{rk}")
    modi.append(f"scoped{rk}")          # Test 2: anker-freies First-Stage, Anker nur im Rerank
    raenge = {m: [] for m in modi}
    pro_gebiet: dict[str, dict[str, list]] = {}
    resid = {"hit": 0, "recall_miss": 0, "rerank_miss_cross": 0, "rerank_miss_intra": 0}

    for satz, ziele in gold:
        g = idx.suche(satz, enc, k=300, min_sim=0.0)
        graph_sim: dict[str, float] = {}
        for t in g:
            graph_sim.setdefault(para_level(t["norm_id"]), t["sim"])
        qv = np.asarray(enc.encode([satz])[0], dtype=np.float32)
        sims = A @ qv
        anker_max: dict[str, tuple[float, str]] = {}
        for i, para in enumerate(a_ziel):
            if sims[i] > anker_max.get(para, (-2.0, ""))[0]:
                anker_max[para] = (float(sims[i]), a_law[i])
        routed = {_gebiet(para_level(t["norm_id"])) for t in g[:args.route_top]}
        oracle = {_gebiet(z) for z in ziele}

        def scored(scope):
            sc = dict(graph_sim)
            for para, (s, law) in anker_max.items():
                if scope is None or law in scope:
                    sc[para] = max(sc.get(para, 0.0), s)
            return sorted(sc, key=sc.get, reverse=True)

        ranked = {"graph": sorted(graph_sim, key=graph_sim.get, reverse=True),
                  "none": scored(None), "routed": scored(routed), "oracle": scored(oracle)}
        ranked["rerank20"] = rerank(satz, ranked["none"][:20])
        ranked["rerank50"] = rerank(satz, ranked["none"][:50])
        if rk not in (20, 50):
            ranked[f"rerank{rk}"] = rerank(satz, ranked["none"][:rk])   # Test 1: großes Fenster
        # Test 2: First-Stage anker-frei (graph-§-Text mischt Gesetze kaum) -> Anker NUR im Rerank
        ranked[f"scoped{rk}"] = rerank(satz, ranked["graph"][:rk])

        geb = _gebiet(sorted(ziele)[0])
        pg = pro_gebiet.setdefault(geb, {m: [] for m in modi})
        for m in modi:
            r = _rank_of(ranked[m], ziele)
            raenge[m].append(r); pg[m].append(r)

        # Residual-Attribution (Hauptbedingung rerank50)
        in_cands = bool(set(ranked["none"][:50]) & ziele)
        r50 = _rank_of(ranked["rerank50"], ziele)
        if not in_cands:
            resid["recall_miss"] += 1
        elif r50 and r50 <= 5:
            resid["hit"] += 1
        else:
            top1 = ranked["rerank50"][0] if ranked["rerank50"] else ""
            same = _gebiet(top1) in oracle
            resid["rerank_miss_intra" if same else "rerank_miss_cross"] += 1

    # --- Report: R@k-Matrix + gepaarte Bootstrap-CIs ---
    n = len(gold)
    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, n, n) for _ in range(args.boot)]   # gepaart: gleiche Resamples

    def ci(raenge_m):
        rr = _rr(raenge_m)
        samp = np.array([rr[ix].mean() for ix in boot_idx])
        return round(float(np.percentile(samp, 2.5)), 3), round(float(np.percentile(samp, 97.5)), 3)

    print(f"\n# Reranker-Ablation  (n={n}, eingefrorenes Gold)\n")
    print(f"{'Modus':9} {'R@5':>5} {'R@10':>5} {'R@20':>5} {'R@50':>5} {'MRR':>6}   MRR 95%-CI")
    gesamt = {}
    for m in modi:
        mm = _metrik(raenge[m]); lo, hi = ci(raenge[m]); mm["ci"] = [lo, hi]; gesamt[m] = mm
        print(f"{m:9} {mm['r5']:5.2f} {mm['r10']:5.2f} {mm['r20']:5.2f} {mm['r50']:5.2f} "
              f"{mm['mrr']:6.3f}   [{lo:.3f}, {hi:.3f}]")
    print(f"\n  DECKE: none R@20={gesamt['none']['r20']:.2f}  R@50={gesamt['none']['r50']:.2f}"
          f"  (= Reranker-Maximum) | Ziellinie oracle MRR={gesamt['oracle']['mrr']:.3f}")

    print(f"\n# Residual rerank50 (n={n}): "
          f"hit≤5 {resid['hit']} | recall_miss {resid['recall_miss']} | "
          f"rerank_miss cross {resid['rerank_miss_cross']} / intra {resid['rerank_miss_intra']}")

    print(f"\n# Je Gebiet (MRR " + " → ".join(modi) + ")")
    geb_json = {}
    for geb in sorted(pro_gebiet):
        ms = {k: _metrik(v) for k, v in pro_gebiet[geb].items()}
        geb_json[geb] = ms
        print(f"  {geb:7} n={ms['graph']['n']:2d}  " + " → ".join(f"{ms[m]['mrr']:.2f}" for m in modi))

    out = {"stand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "gold": args.gold, "n": n, "anker": len(anker), "reranker": args.reranker,
           "route_top": args.route_top, "boot": args.boot,
           "gesamt": gesamt, "residual_rerank50": resid, "pro_gebiet": geb_json}
    ziel = Path(args.save)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
