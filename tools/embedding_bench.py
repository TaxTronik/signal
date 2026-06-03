"""Gold-Benchmark für den Embedding-Index (Schicht 1.7) – encoder-vergleichbar.

Erweitert die schmale Schwellwert-Diagnose (embedding_eval.py) zu einer
belastbaren Bench-Harness für Encoder-Head-to-Heads (BGE-M3 vs. Granite vs. …).

Leitplanke (Projekt-Kernprinzip): KEINE erfundenen Norm-Zuordnungen. Die Gold-
Targets sind ausschließlich KATALOG-geerdet (kuratierte Paraphrasen + norm_anker/
normketten), plus die hand-kuratierten Held-out-Fälle aus embedding_eval.

Methodik gegen Zirkularität: gemessen wird gegen einen ANKER-FREIEN Index
(``katalog=None``). Damit sind die Katalog-Paraphrasen echte Held-out-Anfragen
(sie sind nicht eingebettet) und der Vergleich misst die reine Gesetzestext-
Retrieval-Stärke des Encoders – fair über alle Encoder.

Metriken: Recall@5, Recall@10, MRR, Abstention-Marge (tiefster korrekter Positiv
vs. höchster Negativ), Aufschlüsselung je Rechtsgebiet.

Aufruf:  python tools/embedding_bench.py [--encoder bge-m3|hash]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_layer.catalog import lade_katalog                       # noqa: E402
from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.graph import graph_oeffnen                        # noqa: E402
from risk_layer.idschema import para_level                        # noqa: E402
from risk_layer.reranker import (baue_norm_doc_fn,                # noqa: E402
                                 rerank_normen, reranker_oeffnen)
from risk_layer.zitate import parse_normzitat                     # noqa: E402
from tools.embedding_eval import NEGATIVE, POSITIVE               # noqa: E402


def _ziele(eintrag) -> set[str]:
    """§-Ebene-Zielnormen eines Katalogeintrags aus norm_anker + Normketten-Köpfen."""
    quellen = list(getattr(eintrag, "norm_anker", ()) or ())
    quellen += [k.primaer for k in (getattr(eintrag, "normketten", ()) or ()) if k.primaer]
    ziele: set[str] = set()
    for z in quellen:
        for a in parse_normzitat(z):
            if a.norm_id:
                ziele.add(para_level(a.norm_id.to_id()))
    return ziele


def katalog_gold(kat) -> list[tuple[str, set[str], str]]:
    """(Paraphrase, {Ziel-§}, begriff_id) aus dem kuratierten Katalog – geerdet."""
    gold: list[tuple[str, set[str], str]] = []
    for e in kat:
        ph = list(getattr(e, "paraphrasen", ()) or ())
        ziele = _ziele(e)
        if not ph or not ziele:
            continue
        for p in ph:
            gold.append((p, ziele, e.id))
    return gold


def _gebiet(norm_id: str) -> str:
    teile = norm_id.split(":")
    return teile[1] if len(teile) > 1 else "?"


def bewerte(idx, enc, gold, topk=10, reranker=None, doc_fn=None, topn=20):
    r5 = r10 = mrr = 0
    best_korrekt: list[float] = []
    verfehlt: list[str] = []
    pro_gebiet: dict[str, list[int]] = {}
    for satz, ziele, _bid in gold:
        treffer = idx.suche(satz, enc, k=(topn if reranker else topk), min_sim=0.0)
        if reranker is not None and doc_fn is not None:   # Top-N -> anker-bewusst rerank
            treffer = rerank_normen(reranker, satz, treffer, doc_fn, topk)
        paras = [para_level(t["norm_id"]) for t in treffer]
        rank = next((i + 1 for i, p in enumerate(paras) if p in ziele), None)
        g = pro_gebiet.setdefault(_gebiet(sorted(ziele)[0]), [0, 0])
        g[1] += 1
        if rank:
            mrr += 1 / rank
            r5 += rank <= 5
            r10 += rank <= 10
            if rank <= 5:
                g[0] += 1
            best_korrekt.append(treffer[0]["sim"])
        else:
            verfehlt.append(satz[:44])
    n = len(gold)
    return {"n": n, "recall@5": r5 / n, "recall@10": r10 / n, "mrr": mrr / n,
            "best_korrekt": sorted(best_korrekt), "verfehlt": verfehlt,
            "pro_gebiet": pro_gebiet}


def main() -> int:
    name = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else "bge-m3"
    enc = encoder_oeffnen(name)
    kat = lade_katalog()
    graph = graph_oeffnen()
    print(f"Encoder: {enc.name} (dim {enc.dim})")

    # Zwei Indizes: MIT Anker = Produktionsbedingung; OHNE Anker = reine Encoder-
    # Rohkraft (und macht die Katalog-Paraphrasen zu echten Held-out-Anfragen).
    idx_ohne = EmbeddingIndex.build(graph, enc, katalog=None)
    idx_mit = EmbeddingIndex.build(graph, enc, katalog=kat)
    print(f"Index: {len(idx_ohne.ids)} anker-frei | {len(idx_mit.ids)} mit Anker")

    # Optionaler Reranker: --reranker bge-reranker-v2-m3 (Top-20 -> anker-bewusst -> Top-k).
    rr = doc_fn = None
    if "--reranker" in sys.argv:
        rr = reranker_oeffnen(sys.argv[sys.argv.index("--reranker") + 1])
        doc_fn = baue_norm_doc_fn(kat, graph)
        print(f"Reranker: {rr.name if rr else None}")
    print()

    gold_hand = [(s, z, "hand") for s, z in POSITIVE]
    gold_kat = katalog_gold(kat)

    def kompakt(m):
        return {"n": m["n"], "recall@5": round(m["recall@5"], 3),
                "recall@10": round(m["recall@10"], 3), "mrr": round(m["mrr"], 3),
                "verfehlt": len(m["verfehlt"]),
                "pro_gebiet": {g: v for g, v in sorted(m["pro_gebiet"].items())}}

    hand_mit = bewerte(idx_mit, enc, gold_hand, reranker=rr, doc_fn=doc_fn)
    hand_ohne = bewerte(idx_ohne, enc, gold_hand, reranker=rr, doc_fn=doc_fn)
    kat_ohne = bewerte(idx_ohne, enc, gold_kat, reranker=rr, doc_fn=doc_fn)
    neg_best = sorted((idx_mit.suche(s, enc, k=1, min_sim=0.0)[0]["sim"] for s in NEGATIVE),
                      reverse=True)
    pos_best = hand_mit["best_korrekt"]
    marge = round(pos_best[0] - neg_best[0], 3) if pos_best else None

    print(f"# HAND-Fälle (n={len(gold_hand)}, held-out)")
    print(f"  [mit Anker ] R@5={hand_mit['recall@5']:.2f}  R@10={hand_mit['recall@10']:.2f}  MRR={hand_mit['mrr']:.3f}")
    print(f"  [anker-frei] R@5={hand_ohne['recall@5']:.2f}  R@10={hand_ohne['recall@10']:.2f}  MRR={hand_ohne['mrr']:.3f}")
    print(f"# KATALOG-Paraphrasen (n={len(gold_kat)}, anker-frei) "
          f"R@5={kat_ohne['recall@5']:.2f}  R@10={kat_ohne['recall@10']:.2f}  MRR={kat_ohne['mrr']:.3f}")
    print("  je Gebiet (R@5): " + "  ".join(
        f"{g}={t}/{ges}" for g, (t, ges) in sorted(kat_ohne["pro_gebiet"].items())))
    if marge is not None:
        print(f"Abstention (mit Anker): Marge={marge:+.3f} "
              f"(tiefster Pos {pos_best[0]:.3f} / höchster Neg {neg_best[0]:.3f})")

    ergebnis = {
        "encoder": enc.name, "dim": enc.dim,
        "reranker": rr.name if rr else None,
        "katalog_version": getattr(kat, "version", "?"),
        "stand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index": {"anker_frei": len(idx_ohne.ids), "mit_anker": len(idx_mit.ids)},
        "hand_mit_anker": kompakt(hand_mit),
        "hand_anker_frei": kompakt(hand_ohne),
        "katalog_anker_frei": kompakt(kat_ohne),
        "abstention_marge": marge,
    }
    if "--save" in sys.argv:
        ziel = Path(sys.argv[sys.argv.index("--save") + 1])
        ziel.parent.mkdir(parents=True, exist_ok=True)
        ziel.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBaseline gesichert: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
