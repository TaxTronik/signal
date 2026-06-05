"""Anker-Promiskuität: trennt H3 (diffuse Volumen-Verdünnung) von H4 (wenige zu
breite Neu-Anker) am ANKER-Level.

Ein Anker ist „promiskuitiv", wenn seine Norm in der norm-aggregierten Top-N einer
Gold-Query landet, OHNE Zielnorm dieser Query zu sein — und dieser Anker der
verantwortliche (max-sim) Anker seiner Norm ist. Gezählt je Anker:
  - events      : # Queries, in denen er eine Fremd-Norm in die Top-N hebt
  - cross_class : davon mit Anker-Gesetz ≠ Gold-Gesetz(e) der Query
Vergleich ALT vs. NEU-Bestand; NEU-Anker = Text nicht im ALT-Bestand (= 137-Batch).
  wenige Neu-Anker mit sehr hoher Promiskuität  -> H4 (Granularität/zu breit)
  viele Neu-Anker mit je niedriger Promiskuität  -> H3 (diffuse Volumen-Verdünnung)

Identische Scoring-Logik wie scoped_eval (graph_sim ∪ anker_max per Norm, max).

Aufruf:
  python tools/anchor_promiskuitaet.py --gold discovery/klausuren_v2.yaml \
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
            out.append((str(f.get("query", "")), ziele))
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Anker-Promiskuität (H3 vs H4)")
    ap.add_argument("--gold", default="discovery/klausuren_v2.yaml")
    ap.add_argument("--anker-alt", default="discovery/api/cards_training_pairs.yaml")
    ap.add_argument("--anker-neu", default="discovery/api/cards_training_pairs_merged.yaml")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--save", default="benchmarks/anchor-promiskuitaet.json")
    args = ap.parse_args()

    gold = _lade_gold(Path(args.gold))
    gq = {_norm(q) for q, _ in gold}
    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen("bge-m3", device=dev)
    graph = graph_oeffnen()
    print(f"Gold {len(gold)} | baue Graph-Index …", file=sys.stderr)
    idx = EmbeddingIndex.build(graph, enc, katalog=None)
    qvs = np.asarray(enc.encode([q for q, _ in gold]), dtype=np.float32)
    graph_cache = []
    for q, _ in gold:
        g = idx.suche(q, enc, k=300, min_sim=0.0)
        sc = {}
        for t in g:
            sc.setdefault(para_level(t["norm_id"]), t["sim"])
        graph_cache.append(sc)

    alt_inh, _ = _lade_anker(Path(args.anker_alt), gq)
    alt_set = {_norm(t) for t in alt_inh}

    def analyse(pfad: str):
        inh, ziel = _lade_anker(Path(pfad), gq)
        A = np.asarray(enc.encode(inh), dtype=np.float32)
        law = [_gebiet(z) for z in ziel]
        is_new = [_norm(t) not in alt_set for t in inh]
        # je Norm: Liste der Anker-Indizes
        norm_anker: dict[str, list[int]] = {}
        for i, z in enumerate(ziel):
            norm_anker.setdefault(z, []).append(i)
        events = Counter()       # anker_idx -> # Fremd-Top-N
        cross = Counter()
        for qi, (q, ziele) in enumerate(gold):
            sims = A @ qvs[qi]
            sc = dict(graph_cache[qi])
            best_anker: dict[str, int] = {}
            for z, idxs in norm_anker.items():
                bi = max(idxs, key=lambda i: sims[i])
                s = float(sims[bi])
                if s > sc.get(z, 0.0):
                    sc[z] = s
                    best_anker[z] = bi
                elif z not in sc:
                    sc[z] = s
                    best_anker[z] = bi
            top = sorted(sc, key=sc.get, reverse=True)[:args.topn]
            gold_laws = {_gebiet(z) for z in ziele}
            for nm in top:
                if nm in ziele or nm not in best_anker:
                    continue                          # Zieltreffer / graph-getrieben: nicht promisk
                bi = best_anker[nm]
                events[bi] += 1
                if law[bi] not in gold_laws:
                    cross[bi] += 1
        total = sum(events.values())
        neu_events = sum(c for i, c in events.items() if is_new[i])
        promisk = [(i, c) for i, c in events.most_common()]
        return {"inh": inh, "ziel": ziel, "is_new": is_new, "events": events,
                "cross": cross, "total": total, "neu_events": neu_events,
                "n_anker": len(inh), "n_promisk": len(events), "promisk": promisk}

    print("analysiere ALT …", file=sys.stderr)
    a = analyse(args.anker_alt)
    print("analysiere NEU …", file=sys.stderr)
    b = analyse(args.anker_neu)

    print(f"\n# Anker-Promiskuität (Top-{args.topn}, n={len(gold)} Queries)\n")
    for tag, r in (("ALT", a), ("NEU", b)):
        cr = sum(r["cross"].values())
        print(f"{tag}: {r['n_anker']} Anker | promisk. Anker {r['n_promisk']} "
              f"({100*r['n_promisk']/max(r['n_anker'],1):.0f}%) | Fremd-Top-N-Events {r['total']} "
              f"| cross-class {cr} ({100*cr/max(r['total'],1):.0f}%)")
    print(f"\nNEU-Anker-Anteil an den NEU-Events: {b['neu_events']}/{b['total']} "
          f"({100*b['neu_events']/max(b['total'],1):.0f}%)")

    # Konzentration: top-20 promiskuitive Anker -> Anteil; H3 vs H4
    top20 = b["promisk"][:20]
    top20_share = sum(c for _, c in top20) / max(b["total"], 1)
    print(f"\nKonzentration NEU: Top-20 Anker = {100*top20_share:.0f}% aller Events "
          f"| Ø Events/promisk-Anker = {b['total']/max(b['n_promisk'],1):.1f}")
    print(f"\n# Top-20 promiskuitive NEU-Anker (Norm | events | cross | Text)")
    gezeigt = 0
    for i, c in b["promisk"]:
        if not b["is_new"][i]:
            continue
        z = b["ziel"][i].replace("norm:", "")
        cc = b["cross"][i]
        txt = " ".join(b["inh"][i].split())[:70]
        print(f"  {z:14} ev={c:3d} cx={cc:3d}  {txt}")
        gezeigt += 1
        if gezeigt >= 20:
            break
    befund = ("H4: wenige breite Neu-Anker dominieren" if top20_share >= 0.5
              else "H3: diffuse Volumen-Verdünnung" if b["total"]/max(b["n_promisk"],1) < 3
              else "gemischt")
    print(f"\n-> Befund: {befund} (Top-20-Anteil {100*top20_share:.0f}%, "
          f"Ø {b['total']/max(b['n_promisk'],1):.1f} Events/Anker)")

    out = {"topn": args.topn, "n_queries": len(gold),
           "alt": {"n_anker": a["n_anker"], "n_promisk": a["n_promisk"], "events": a["total"],
                   "cross": sum(a["cross"].values())},
           "neu": {"n_anker": b["n_anker"], "n_promisk": b["n_promisk"], "events": b["total"],
                   "cross": sum(b["cross"].values()), "neu_events": b["neu_events"],
                   "top20_share": top20_share},
           "top_neu_promisk": [{"norm": b["ziel"][i], "events": c, "cross": b["cross"][i],
                                "text": b["inh"][i][:120]}
                               for i, c in b["promisk"] if b["is_new"][i]][:20]}
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
