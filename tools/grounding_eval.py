"""Grounding-Eval: profitiert das LLM (Schicht 2) MESSBAR von besserem Kontext?

Vergleicht die Norm-Vorschlagsqualität des LLM in drei Bedingungen auf demselben
Gold-Satz (Bench-Gold: HAND + Katalog-Paraphrasen, mit Ziel-§):

  A   un-grounded                 — kein kontext_normen (LLM rät aus Text/Gedächtnis)
  B0  grounded (Basis-Index)      — Top-k aus dem Katalog-verankerten Embedding-Index
  B1  grounded (+ Discovery-Anker)— Top-k aus Index, der ZUSÄTZLICH die Discovery-
                                     Paraphrasen als Anker führt  → isoliert den
                                     Discovery-Beitrag zur LLM-Erdung.

Kette: Discovery-Anker → besseres Retrieval → besserer kontext_normen → besserer
norm_vorschlag. Wenn B1 > B0 > A, ist der LLM-Gewinn aus Discovery belegt.

Leckage: Discovery-Anker, die mit einer Gold-Anfrage identisch sind, werden
ausgeschlossen (sonst injiziert man die Antwort). Lokaler llama-server (granite)
reicht — KEIN großes Modell nötig.

Aufruf:
  python tools/grounding_eval.py --llm-url http://127.0.0.1:8080 --no-think
  python tools/grounding_eval.py --limit 20            # schneller Teil-Lauf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.catalog import lade_katalog                 # noqa: E402
from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.finetune import bench_query_set             # noqa: E402
from risk_layer.graph import graph_oeffnen                  # noqa: E402
from risk_layer.idschema import para_level                  # noqa: E402
from risk_layer.llm import LlamaServerProvider, server_verfuegbar  # noqa: E402
from risk_layer.zitate import parse_normzitat               # noqa: E402
from tools.embedding_bench import katalog_gold              # noqa: E402
from tools.embedding_eval import POSITIVE                   # noqa: E402


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _vorschlag_paras(kandidaten) -> set[str]:
    """§-Ebene aller norm_vorschlag-Felder der LLM-Kandidaten."""
    out: set[str] = set()
    for k in kandidaten:
        for a in parse_normzitat(str(k.get("norm_vorschlag", ""))):
            if a.norm_id:
                out.add(para_level(a.norm_id.to_id()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Grounding-Eval: LLM-Norm-Vorschlag mit/ohne Kontext")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8080")
    ap.add_argument("--discovery", default="discovery/qwen-36b/training_pairs.yaml",
                    help="Discovery-Trainingspaare (Quelle der Extra-Anker für B1)")
    ap.add_argument("--k", type=int, default=5, help="Top-k Kontext-Normen pro Fall")
    ap.add_argument("--limit", type=int, default=0, help="nur N Gold-Fälle (0 = alle)")
    ap.add_argument("--save", default="benchmarks/grounding-eval.json")
    args = ap.parse_args()

    if not server_verfuegbar(args.llm_url, timeout=3.0):
        print(f"Kein llama-server unter {args.llm_url} (lokal: risk-layer modell start).", file=sys.stderr)
        return 2

    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen("bge-m3", device=dev)
    kat = lade_katalog()
    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2

    # Gold: HAND-Positive + Katalog-Paraphrasen, je (Satz, {Ziel-§})
    gold = [(s, set(z)) for s, z in POSITIVE] + [(p, set(z)) for p, z, _ in katalog_gold(kat)]
    if args.limit:
        gold = gold[:args.limit]

    # Discovery-Anker für B1 (Leckage gegen Gold ausschließen)
    bench_qs = bench_query_set(kat, POSITIVE)
    disc_pfad = Path(args.discovery)
    extra = []
    if disc_pfad.exists():
        for d in (yaml.safe_load(disc_pfad.read_text(encoding="utf-8")) or {}).get("kandidaten", []):
            inhalt, ziel = str(d.get("inhalt", "")), str(d.get("ziel_norm_id", ""))
            if inhalt.strip() and ziel and _norm(inhalt) not in bench_qs:
                extra.append((inhalt, ziel))
    else:
        print(f"WARN: {disc_pfad} nicht gefunden — B1 = B0 (keine Discovery-Anker).", file=sys.stderr)

    print(f"Gold: {len(gold)} Fälle | Discovery-Extra-Anker: {len(extra)} | k={args.k}", file=sys.stderr)
    print("Baue Indizes (Basis + Discovery) …", file=sys.stderr)
    idx0 = EmbeddingIndex.build(graph, enc, katalog=kat)
    idx1 = EmbeddingIndex.build(graph, enc, katalog=kat, extra_anker=extra)

    prov = LlamaServerProvider(args.llm_url)   # /no_think setzt der Provider modell-bewusst selbst
    tally = {"A": 0, "B0": 0, "B1": 0}
    ceiling = {"B0": 0, "B1": 0}            # enthielt das Retrieval überhaupt die Zielnorm?
    n = len(gold)
    for i, (satz, ziele) in enumerate(gold, 1):
        t0 = idx0.suche(satz, enc, k=args.k, min_sim=0.0)
        t1 = idx1.suche(satz, enc, k=args.k, min_sim=0.0)
        if any(para_level(t["norm_id"]) in ziele for t in t0):
            ceiling["B0"] += 1
        if any(para_level(t["norm_id"]) in ziele for t in t1):
            ceiling["B1"] += 1
        for cond, ktx in (("A", None), ("B0", [t["zitat"] for t in t0]), ("B1", [t["zitat"] for t in t1])):
            kand = prov.kandidaten(satz, [], kontext_normen=ktx)
            if _vorschlag_paras(kand) & ziele:
                tally[cond] += 1
        if i % 10 == 0:
            print(f"  … {i}/{n}", file=sys.stderr)

    def pct(x):
        return round(x / n, 3) if n else 0.0

    print(f"\n# Grounding-Eval (n={n} Gold-Fälle, LLM-Norm-Vorschlag trifft Ziel-§)")
    print(f"  A  un-grounded            : {tally['A']:3d}/{n}  ({pct(tally['A']):.0%})")
    print(f"  B0 grounded (Basis)       : {tally['B0']:3d}/{n}  ({pct(tally['B0']):.0%})   "
          f"[Retrieval-Decke {pct(ceiling['B0']):.0%}]")
    print(f"  B1 grounded (+Discovery)  : {tally['B1']:3d}/{n}  ({pct(tally['B1']):.0%})   "
          f"[Retrieval-Decke {pct(ceiling['B1']):.0%}]")
    print(f"\n  Grounding-Lift  A→B0: {pct(tally['B0'])-pct(tally['A']):+.0%}   "
          f"Discovery-Lift B0→B1: {pct(tally['B1'])-pct(tally['B0']):+.0%}")

    ergebnis = {
        "stand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": n, "k": args.k, "extra_anker": len(extra),
        "modell": prov._modell_name() if hasattr(prov, "_modell_name") else "?",
        "treffer": {c: tally[c] for c in tally},
        "rate": {c: pct(tally[c]) for c in tally},
        "retrieval_decke": {c: pct(ceiling[c]) for c in ceiling},
    }
    ziel = Path(args.save)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
