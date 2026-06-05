"""Schritt-2-Ablation: Normpaket-Reranking ALLEIN (kein Guard/Scope/LLM/neue Cards).

Kandidat = Normpaket statt Einzelanker. Paket-Dokument (stabile Reihenfolge, kein LLM):
  norm:<id> | <Titel> | <§-Textauszug[:N] aus Graphknoten> | <Top-k Cards der Norm
  nach First-Stage-(Bi-Encoder)-Score zur Query>   (k=3 fix)
Der Cross-Encoder scort EIN (query, Paket)-Paar je Norm; KEINE per-Anker-max-Aggregation.
First-Stage (oracle-Scope-Kandidaten) unverändert. Vergleich gegen Modus `max` (Status quo)
im selben Harness (gleiches Tie-Handling).

Primärmetrik: KStG-Dev (motiviert durch den KStG-Knick), Schwelle >= 0,55.
Wächter: v2-oracle_rr im 0,506-Band, flat nicht schlechter. Bewegung soll aus den
paketfähigen Residuen (Topf 1+5) kommen.
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

K, TOPK_CARDS, EXCERPT = 200, 3, 400
# Paketfähig aus Schritt 1 (Topf 1 + Topf 5 = 14/21)
PAKETFAEHIG = {"AO-01d", "AO-04", "EST1-A1b2", "EST1-A1b4", "EST1-A1b5", "EST1-A2",
               "EST1-A3", "EST2-A1-3", "KST-10", "UST1-08", "UST1-11", "UST2-01",
               "UST2-03", "UST2-10"}


def _norm(s): return " ".join((s or "").lower().split())  # noqa: E704
def _law(p): t = p.split(":"); return t[1] if len(t) > 1 else "?"  # noqa: E704


def _lade_gold(p):
    d = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
    out = []
    for f in d.get("faelle", []):
        q = str(f.get("query", ""))
        ziele = {para_level(z) for z in (f.get("ziele") or []) if z}
        if _norm(q) and ziele:
            out.append({"fall_id": f.get("fall_id", ""), "query": q, "ziele": ziele})
    return out


def _mrr(rs): n = len(rs) or 1; return round(sum(1.0/r for r in rs if r)/n, 3)  # noqa: E704
def _rk(rs, k): n = len(rs) or 1; return round(sum(1 for r in rs if r and r <= k)/n, 3)  # noqa: E704


def main() -> int:
    ap = argparse.ArgumentParser(description="Normpaket-Reranking (Schritt-2-Ablation)")
    ap.add_argument("--gold-v2", default="discovery/klausuren_v2.yaml")
    ap.add_argument("--gold-dev", default="discovery/dev_v1.yaml")
    ap.add_argument("--anker", default="discovery/api/cards_training_pairs_absatz_replace.yaml")
    ap.add_argument("--reranker", default="bge-reranker-v2-m3")
    ap.add_argument("--save", default="benchmarks/normpaket.json")
    args = ap.parse_args()

    v2, dev = _lade_gold(args.gold_v2), _lade_gold(args.gold_dev)
    gq = {_norm(g["query"]) for g in v2 + dev}
    d = yaml.safe_load(Path(args.anker).read_text(encoding="utf-8")) or {}
    inh, ziel = [], []
    for k in d.get("kandidaten", []):
        t = str(k.get("inhalt", "")).strip()
        z = para_level(str(k.get("ziel_norm_id") or k.get("quell_norm") or "").strip())
        if t and z and _norm(t) not in gq:
            inh.append(t); ziel.append(z)

    dvc = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen("bge-m3", device=dvc)
    graph = graph_oeffnen()
    print(f"v2 {len(v2)} | dev {len(dev)} | Anker {len(inh)} | baue Index + Reranker …", file=sys.stderr)
    idx = EmbeddingIndex.build(graph, enc, katalog=None)
    reranker = reranker_oeffnen(args.reranker)
    node_text = {para_level(n["id"]): (n["text"] or "") for n in iter_paragraphen(graph)}
    node_titel = {para_level(n["id"]): (n.get("titel") or "") for n in iter_paragraphen(graph)}
    A = np.asarray(enc.encode(inh), dtype=np.float32)
    a_law = [_law(z) for z in ziel]
    norm_idx = defaultdict(list)                 # para-norm -> Anker-Zeilenindizes (für Top-k Cards)
    for i, z in enumerate(ziel):
        norm_idx[z].append(i)

    def docs_max(nm):                            # Status-quo: cards + §-Text[:400], je eigenes Doc
        ds = [inh[i] for i in norm_idx.get(nm, [])]
        if node_text.get(nm):
            ds.append(node_text[nm][:EXCERPT])
        return ds or [nm]

    def paket_doc(nm, sims):                     # EIN Paket-Dokument je Norm, stabile Reihenfolge
        cidx = sorted(norm_idx.get(nm, []), key=lambda i: (-float(sims[i]), i))[:TOPK_CARDS]
        teile = [f"norm:{nm.split(':',1)[1] if nm.startswith('norm:') else nm}",
                 node_titel.get(nm, ""), (node_text.get(nm, "") or "")[:EXCERPT]]
        teile += [inh[i] for i in cidx]
        return " | ".join(t for t in teile if t)

    def rerank(query, cands, modus, sims):
        if modus == "paket":
            pairs = [(query, paket_doc(nm, sims)) for nm in cands]
            if not pairs:
                return list(cands)
            sc = reranker.score(pairs)
            return [nm for nm, _ in sorted(zip(cands, sc), key=lambda kv: kv[1], reverse=True)]
        # max
        pairs, owner = [], []
        for nm in cands:
            for x in docs_max(nm):
                pairs.append((query, x)); owner.append(nm)
        if not pairs:
            return list(cands)
        sc = reranker.score(pairs)
        best = {}
        for nm, s in zip(owner, sc):
            best[nm] = max(best.get(nm, -1e18), float(s))
        return sorted(best, key=best.get, reverse=True)

    def kandidaten(query, scope):                # scope=None -> flat(none-Top-50); sonst oracle
        g = idx.suche(query, enc, k=300, min_sim=0.0)
        graph_sim = {}
        for t in g:
            graph_sim.setdefault(para_level(t["norm_id"]), t["sim"])
        qv = np.asarray(enc.encode([query])[0], dtype=np.float32)
        sims = A @ qv
        amax = {}
        for i, p in enumerate(ziel):
            if sims[i] > amax.get(p, (-2.0, ""))[0]:
                amax[p] = (float(sims[i]), a_law[i])
        sc = dict(graph_sim)
        for p, (s, l) in amax.items():
            if scope is None or l in scope:
                sc[p] = max(sc.get(p, 0.0), s)
        ranked = sorted(sc, key=sc.get, reverse=True)
        if scope is None:
            return ranked[:50], sims
        return [n for n in ranked if _law(n) in scope][:K], sims

    def evaluiere(gold, scope_mode, rerank_mode, residual=False):
        rs, pro = [], defaultdict(list)
        moved = []
        for g in gold:
            scope = {_law(z) for z in g["ziele"]} if scope_mode == "oracle" else None
            cands, sims = kandidaten(g["query"], scope)
            ranked = rerank(g["query"], cands, rerank_mode, sims)
            r = next((i + 1 for i, nm in enumerate(ranked) if nm in g["ziele"]), None)
            rs.append(r)
            pro[_law(sorted(g["ziele"])[0])].append(r)
            if residual and r and r <= 5 and g["fall_id"]:
                moved.append(g["fall_id"])
        return {"mrr": _mrr(rs), "r5": _rk(rs, 5), "r20": _rk(rs, 20),
                "pro_law": {l: _mrr(v) for l, v in sorted(pro.items())}, "moved": moved}

    res = {}
    print("v2 oracle max …", file=sys.stderr);  res["v2_oracle_max"] = evaluiere(v2, "oracle", "max", residual=True)
    print("v2 oracle paket …", file=sys.stderr); res["v2_oracle_paket"] = evaluiere(v2, "oracle", "paket", residual=True)
    print("v2 flat max …", file=sys.stderr);    res["v2_flat_max"] = evaluiere(v2, "flat", "max")
    print("v2 flat paket …", file=sys.stderr);  res["v2_flat_paket"] = evaluiere(v2, "flat", "paket")
    print("dev oracle max …", file=sys.stderr); res["dev_oracle_max"] = evaluiere(dev, "oracle", "max")
    print("dev oracle paket …", file=sys.stderr); res["dev_oracle_paket"] = evaluiere(dev, "oracle", "paket")

    def line(tag):
        r = res[tag]
        print(f"  {tag:18} MRR {r['mrr']:.3f}  R@5 {r['r5']:.2f}  R@20 {r['r20']:.2f}")

    print(f"\n# Normpaket-Reranking (k={TOPK_CARDS}, Excerpt {EXCERPT}, ein Lauf)\n")
    for t in ("v2_oracle_max", "v2_oracle_paket", "v2_flat_max", "v2_flat_paket",
              "dev_oracle_max", "dev_oracle_paket"):
        line(t)
    kstg_max = res["dev_oracle_max"]["pro_law"].get("KStG", 0.0)
    kstg_pak = res["dev_oracle_paket"]["pro_law"].get("KStG", 0.0)
    print(f"\n# PRIMÄR: KStG-Dev  max {kstg_max:.3f} -> paket {kstg_pak:.3f}  (Schwelle >= 0,55; Baseline-1015 0,587)")
    print(f"# v2-oracle_rr  max {res['v2_oracle_max']['mrr']:.3f} -> paket {res['v2_oracle_paket']['mrr']:.3f}  (Band ~0,506)")
    print(f"# v2-flat       max {res['v2_flat_max']['mrr']:.3f} -> paket {res['v2_flat_paket']['mrr']:.3f}  (nicht schlechter)")
    print(f"# v2 pro Gesetz oracle paket: {res['v2_oracle_paket']['pro_law']}")

    # Bewegung: welche Residuen sind unter paket neu <=5? aus Topf 1+5?
    neu_max = set(res["v2_oracle_max"]["moved"])
    neu_pak = set(res["v2_oracle_paket"]["moved"])
    bewegt = neu_pak - neu_max                   # vorher nicht <=5, jetzt <=5
    verloren = neu_max - neu_pak
    aus_paketfaehig = bewegt & PAKETFAEHIG
    print(f"\n# Bewegung v2-oracle (max->paket): +{len(bewegt)} neu ≤5, -{len(verloren)} verloren")
    print(f"  neu ≤5: {sorted(bewegt)}")
    print(f"  davon paketfähig (Topf1+5): {len(aus_paketfaehig)}/{len(bewegt)} -> {sorted(aus_paketfaehig)}")
    if verloren:
        print(f"  VERLOREN (Regression): {sorted(verloren)}")

    # PASS/FAIL (vorab fixiert)
    pass_kstg = kstg_pak >= 0.55
    pass_v2 = res["v2_oracle_paket"]["mrr"] >= res["v2_oracle_max"]["mrr"] - 0.02
    pass_flat = res["v2_flat_paket"]["mrr"] >= res["v2_flat_max"]["mrr"] - 0.005
    pass_bew = len(bewegt) == 0 or len(aus_paketfaehig) >= 0.6 * len(bewegt)
    print(f"\n# URTEIL (vorab fixiert)")
    print(f"  [{'PASS' if pass_kstg else 'FAIL'}] KStG-Dev ≥ 0,55  ({kstg_pak:.3f})")
    print(f"  [{'PASS' if pass_v2 else 'FAIL'}] v2-oracle im Band  ({res['v2_oracle_paket']['mrr']:.3f} vs {res['v2_oracle_max']['mrr']:.3f})")
    print(f"  [{'PASS' if pass_flat else 'FAIL'}] flat nicht schlechter  ({res['v2_flat_paket']['mrr']:.3f} vs {res['v2_flat_max']['mrr']:.3f})")
    print(f"  [{'PASS' if pass_bew else 'FAIL'}] Bewegung überwiegend paketfähig")
    gesamt = pass_kstg and pass_v2 and pass_flat and pass_bew
    print(f"  GESAMT: {'PASS — Normpaket wird Freeze-Stand' if gesamt else 'FAIL — Freeze bleibt 1241-Replacement'}")

    res["urteil"] = {"kstg_dev_max": kstg_max, "kstg_dev_paket": kstg_pak,
                     "pass_kstg": pass_kstg, "pass_v2": pass_v2, "pass_flat": pass_flat,
                     "pass_bewegung": pass_bew, "gesamt_pass": gesamt,
                     "bewegt": sorted(bewegt), "bewegt_paketfaehig": sorted(aus_paketfaehig),
                     "verloren": sorted(verloren)}
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
