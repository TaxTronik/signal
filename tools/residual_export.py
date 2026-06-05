"""Residual-Export für die Normpaket-Ablation (Schritt 1, NUR Export+Klassifikation).

Exportiert die oracle_rr-Residuen (Rang>5/raus) auf dem Replacement-Stand (1241) mit:
fall_id · Gold-Norm(en) · aktueller Rang · Top-3-Verdränger (Norm/Gesetz/Score) ·
Ziel im Kandidatensatz? (Coverage) · klasse · n_ziele. Reine Diagnose, keine Änderung.

Identische oracle_rr-Logik wie tools/test3_oracle_rerank.py (Scope=laws(Gold-Ziele),
Kandidaten in-scope Top-K, Cross-Encoder, Norm-max).
"""

from __future__ import annotations

import argparse
import json
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

K = 200


def _norm(s): return " ".join((s or "").lower().split())  # noqa: E704
def _law(p): t = p.split(":"); return t[1] if len(t) > 1 else "?"  # noqa: E704


def main() -> int:
    ap = argparse.ArgumentParser(description="Residual-Export (oracle_rr, Replacement)")
    ap.add_argument("--gold", default="discovery/klausuren_v2.yaml")
    ap.add_argument("--anker", default="discovery/api/cards_training_pairs_absatz_replace.yaml")
    ap.add_argument("--reranker", default="bge-reranker-v2-m3")
    ap.add_argument("--save", default="benchmarks/residuals-export.json")
    args = ap.parse_args()

    gd = yaml.safe_load(Path(args.gold).read_text(encoding="utf-8")) or {}
    gold = []
    for f in gd.get("faelle", []):
        q = str(f.get("query", ""))
        ziele = {para_level(z) for z in (f.get("ziele") or []) if z}
        if _norm(q) and ziele:
            gold.append({"fall_id": f.get("fall_id", ""), "query": q, "ziele": ziele,
                         "klasse": f.get("klasse", "")})
    gq = {_norm(g["query"]) for g in gold}

    d = yaml.safe_load(Path(args.anker).read_text(encoding="utf-8")) or {}
    inh, ziel = [], []
    for k in d.get("kandidaten", []):
        t = str(k.get("inhalt", "")).strip()
        z = para_level(str(k.get("ziel_norm_id") or k.get("quell_norm") or "").strip())
        if t and z and _norm(t) not in gq:
            inh.append(t); ziel.append(z)

    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen("bge-m3", device=dev)
    graph = graph_oeffnen()
    print(f"Gold {len(gold)} | Anker {len(inh)} | baue Index + Reranker …", file=sys.stderr)
    idx = EmbeddingIndex.build(graph, enc, katalog=None)
    reranker = reranker_oeffnen(args.reranker)
    node_text = {para_level(n["id"]): (n["text"] or "") for n in iter_paragraphen(graph)}
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

    def rerank_scored(query, cands):
        pairs, owner = [], []
        for nm in cands:
            for x in docs_of(nm):
                pairs.append((query, x)); owner.append(nm)
        if not pairs:
            return [(nm, 0.0) for nm in cands]
        sc = reranker.score(pairs)
        best = {}
        for nm, s in zip(owner, sc):
            best[nm] = max(best.get(nm, -1e18), float(s))
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    residuen = []
    for g in gold:
        ziele = g["ziele"]
        gl = {_law(z) for z in ziele}
        gsuche = idx.suche(g["query"], enc, k=300, min_sim=0.0)
        graph_sim = {}
        for t in gsuche:
            graph_sim.setdefault(para_level(t["norm_id"]), t["sim"])
        qv = np.asarray(enc.encode([g["query"]])[0], dtype=np.float32)
        sims = A @ qv
        amax = {}
        for i, p in enumerate(ziel):
            if sims[i] > amax.get(p, (-2.0, ""))[0]:
                amax[p] = (float(sims[i]), a_law[i])
        sc = dict(graph_sim)
        for p, (s, l) in amax.items():
            if l in gl:
                sc[p] = max(sc.get(p, 0.0), s)
        cand = [n for n in sorted(sc, key=sc.get, reverse=True) if _law(n) in gl][:K]
        in_cand = bool(set(cand) & ziele)
        ranked = rerank_scored(g["query"], cand)
        rang = next((i + 1 for i, (nm, _s) in enumerate(ranked) if nm in ziele), None)
        if rang and rang <= 5:
            continue
        # Verdränger = die Norm-Pakete über dem (best-platzierten) Ziel; sonst Top-3
        cutoff = rang - 1 if rang else 3
        displacer = [(nm, _law(nm), round(s, 3)) for nm, s in ranked[:max(cutoff, 3)]
                     if nm not in ziele][:3]
        ziel_score = next((round(s, 3) for nm, s in ranked if nm in ziele), None)
        residuen.append({
            "fall_id": g["fall_id"], "klasse": g["klasse"],
            "ziele": sorted(z.replace("norm:", "") for z in ziele), "n_ziele": len(ziele),
            "rang": rang, "ziel_im_kandidatensatz": in_cand, "ziel_score": ziel_score,
            "verdraenger": [{"norm": nm.replace("norm:", ""), "gesetz": l, "score": s}
                            for nm, l, s in displacer],
        })

    print(f"\n# oracle_rr-Residuen (Replacement 1241, K={K}): {len(residuen)}\n")
    for r in residuen:
        v = " · ".join(f"{d['norm']}({d['gesetz']},{d['score']})" for d in r["verdraenger"])
        print(f"[{r['klasse']:9}] {r['fall_id'] or '?':10} ziel={r['ziele']} rang={r['rang']} "
              f"in_cand={r['ziel_im_kandidatensatz']} ziel_score={r['ziel_score']}\n"
              f"            verdränger: {v}")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps({"n_residuen": len(residuen), "k": K,
                                           "residuen": residuen}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(f"\ngesichert: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
