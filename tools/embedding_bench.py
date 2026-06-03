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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_layer.catalog import lade_katalog                       # noqa: E402
from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.graph import graph_oeffnen                        # noqa: E402
from risk_layer.idschema import para_level                        # noqa: E402
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


def bewerte(idx, enc, gold, topk=10):
    r5 = r10 = mrr = 0
    best_korrekt: list[float] = []
    verfehlt: list[str] = []
    pro_gebiet: dict[str, list[int]] = {}
    for satz, ziele, _bid in gold:
        treffer = idx.suche(satz, enc, k=topk, min_sim=0.0)
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
    print(f"Index: {len(idx_ohne.ids)} anker-frei | {len(idx_mit.ids)} mit Anker\n")

    gold_hand = [(s, z, "hand") for s, z in POSITIVE]
    gold_kat = katalog_gold(kat)

    print(f"# HAND-Fälle (n={len(gold_hand)}, held-out, andere Phrasierung als die Anker)")
    for cond, idx in (("mit Anker  (Produktion)", idx_mit),
                      ("anker-frei (Encoder pur)", idx_ohne)):
        m = bewerte(idx, enc, gold_hand)
        print(f"  [{cond}] R@5={m['recall@5']:.2f}  R@10={m['recall@10']:.2f}  MRR={m['mrr']:.3f}")

    print(f"\n# KATALOG-Paraphrasen (n={len(gold_kat)}, anker-frei = echtes Held-out)")
    m = bewerte(idx_ohne, enc, gold_kat)
    print(f"  R@5={m['recall@5']:.2f}  R@10={m['recall@10']:.2f}  MRR={m['mrr']:.3f}")
    print("  je Gebiet (R@5): " + "  ".join(
        f"{g}={tref}/{ges}" for g, (tref, ges) in sorted(m["pro_gebiet"].items())))

    # Abstention in der Produktionsbedingung (mit Anker).
    neg_best = sorted((idx_mit.suche(s, enc, k=1, min_sim=0.0)[0]["sim"] for s in NEGATIVE),
                      reverse=True)
    pos_best = bewerte(idx_mit, enc, gold_hand)["best_korrekt"]
    if pos_best:
        print(f"\nAbstention (mit Anker): tiefster Positiv={pos_best[0]:.3f} | "
              f"höchster Negativ={neg_best[0]:.3f} | Marge={pos_best[0]-neg_best[0]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
