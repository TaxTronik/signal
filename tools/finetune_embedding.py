"""Fine-Tune BGE-M3 auf Discovery-Trainingspaaren (B300, GATED Experiment).

Liest discovery/training_pairs.yaml, baut (anchor=Laiensprache, positive=Normtext)
und trainiert mit MultipleNegativesRankingLoss ein NEUES Artefakt. Überschreibt
NIE den Produktions-BGE-M3/-Index. Trainingsdaten disjunkt vom Bench-Gold.

Akzeptanz danach (NICHT automatisch übernehmen):
  python tools/embedding_bench.py --encoder <out> --save benchmarks/bge-m3-ft.json
  gegen benchmarks/bge-m3.json vergleichen — die harte Guardrail muss ganz erfüllt
  sein (Produktion-mit-Ankern >= Baseline, MRR >=, KATALOG held-out besser, HAND
  anker-frei besser, Negativ-Floor nicht hoeher, FP nicht schlechter), sonst Research.

Aufruf (auf der B300):
  python tools/finetune_embedding.py --pairs discovery/training_pairs.yaml --epochs 1
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.catalog import lade_katalog                 # noqa: E402
from risk_layer.finetune import baue_trainingspaare, bench_query_set  # noqa: E402
from risk_layer.graph import graph_oeffnen                  # noqa: E402
from tools.embedding_eval import POSITIVE                   # noqa: E402

PROD_INDEX = Path("corpus/embedding")


def _norm_text(graph, nid: str) -> str:
    r = graph.aufloesen(nid)
    if not r.get("gefunden"):
        return ""
    return f"{(r.get('titel') or '').strip()}. {(r.get('text') or '').strip()}".strip(". ").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="BGE-M3 Fine-Tune auf Discovery-Paaren (gated)")
    ap.add_argument("--pairs", default="discovery/training_pairs.yaml")
    ap.add_argument("--base", default="BAAI/bge-m3")
    ap.add_argument("--out", default=f"models/bge-m3-ft-{date.today().isoformat()}",
                    help="NEUES Artefakt-Verzeichnis (NIE der Prod-Index)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    if out.resolve() == PROD_INDEX.resolve():
        print("VERWEIGERT: --out ist der Produktions-Index. Neues Artefakt wählen.", file=sys.stderr)
        return 2

    rohdaten = yaml.safe_load(Path(args.pairs).read_text(encoding="utf-8")) or {}
    kandidaten = rohdaten.get("kandidaten") or []
    if not kandidaten:
        print(f"Keine Kandidaten in {args.pairs} — erst Discovery + Filter laufen lassen.", file=sys.stderr)
        return 2

    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2
    kat = lade_katalog()
    bench_qs = bench_query_set(kat, POSITIVE)
    paare = baue_trainingspaare(kandidaten, norm_text=lambda nid: _norm_text(graph, nid),
                                bench_queries=bench_qs)
    print(f"{len(kandidaten)} Kandidaten → {len(paare)} Trainingspaare "
          f"(nach Leckage-/Dublettenfilter; {len(bench_qs)} Bench-Anfragen ausgeschlossen)")
    if not paare:
        print("Keine verwendbaren Paare.", file=sys.stderr)
        return 2

    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        print(f"sentence-transformers/torch fehlt: {exc}", file=sys.stderr)
        return 2

    print(f"Lade Basis-Modell {args.base} (device={dev}) — NEUES Artefakt → {out}")
    model = SentenceTransformer(args.base, device=dev)
    beispiele = [InputExample(texts=[p.anchor, p.positive]) for p in paare]
    loader = DataLoader(beispiele, shuffle=True, batch_size=args.batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = max(1, int(len(loader) * args.epochs * 0.1))
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=warmup, show_progress_bar=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"\nFertig: {out}  ({len(paare)} Paare, {args.epochs} Epoche(n))")
    print("Akzeptanz-Gate (NICHT automatisch übernehmen):")
    print(f"  python tools/embedding_bench.py --encoder {out} --save benchmarks/bge-m3-ft.json")
    print("  dann gegen benchmarks/bge-m3.json vergleichen — nur bei voller Guardrail behalten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
