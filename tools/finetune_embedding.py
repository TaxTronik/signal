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
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole                      # noqa: E402

utf8_konsole()

from risk_layer.catalog import lade_katalog                 # noqa: E402
from risk_layer.finetune import baue_trainingspaare, bench_query_set  # noqa: E402
from risk_layer.graph import graph_oeffnen                  # noqa: E402
from risk_layer.persistenz import atomar_schreiben          # noqa: E402
from risk_layer.sicherung import zeitstempel_dateiname      # noqa: E402
from tools.embedding_eval import POSITIVE                   # noqa: E402

PROD_INDEX = Path("corpus/embedding")

# Protokollpflicht je Trainingslauf (PS-861-Mapping, Lücke L6 / Tz. 33): der
# Fine-Tuning-Pfad ist technisch abgesichert, war aber organisatorisch
# protokollfrei — ohne Lauf-Protokoll gäbe es für eine Freigabe-Entscheidung
# (Governance-Betriebsordnung) keine belastbare Faktenbasis.
GOVERNANCE_HINWEIS = (
    "HINWEIS: Fine-Tuning ist ein EXPERIMENTELLER Pfad. Produktiver Einsatz "
    "des erzeugten Artefakts NUR nach dokumentierter Freigabe laut "
    "docs/compliance/Governance-Betriebsordnung.md (PS-861, Lücke L6).")


def baue_protokoll(*, zeitstempel: str, argumente: dict, datenquelle: str,
                   n_kandidaten: int, n_paare: int, n_bench_ausgeschlossen: int,
                   basis_modell: str, out_pfad: str, autor: str | None) -> dict:
    """Lauf-Protokoll (reine Datenstruktur, getrennt testbar — das Training
    selbst braucht GPU/torch und bleibt untestbar leicht)."""
    return {
        "zeitstempel": zeitstempel,
        "werkzeug": "tools/finetune_embedding.py",
        "argumente": dict(argumente),
        "datenquelle": {"pfad": datenquelle, "n_kandidaten": n_kandidaten,
                        "n_trainingspaare": n_paare,
                        "n_bench_queries_ausgeschlossen": n_bench_ausgeschlossen},
        "basis_modell": basis_modell,
        "output": out_pfad,
        "autor": autor,
        "hinweis": GOVERNANCE_HINWEIS,
    }


def schreibe_protokoll(protokoll: dict, out: Path) -> Path:
    """Schreibt das Protokoll datiert NEBEN den Output (atomar, diffbar).

    Neben (nicht in) das Artefakt-Verzeichnis: model.save() kontrolliert dessen
    Inhalt — eine Fremddatei dort könnte von künftigen save-Implementierungen
    überschrieben werden.
    """
    pfad = (out.parent
            / f"finetune_protokoll_{zeitstempel_dateiname(protokoll['zeitstempel'])}.json")
    atomar_schreiben(pfad, json.dumps(protokoll, ensure_ascii=False,
                                      sort_keys=True, indent=2) + "\n")
    return pfad


def _norm_text(graph, nid: str) -> str:
    r = graph.aufloesen(nid)
    if not r.get("gefunden"):
        return ""
    return f"{(r.get('titel') or '').strip()}. {(r.get('text') or '').strip()}".strip(". ").strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BGE-M3 Fine-Tune auf Discovery-Paaren (gated)")
    ap.add_argument("--pairs", default="discovery/training_pairs.yaml")
    ap.add_argument("--base", default="BAAI/bge-m3")
    ap.add_argument("--out", default=f"models/bge-m3-ft-{date.today().isoformat()}",
                    help="NEUES Artefakt-Verzeichnis (NIE der Prod-Index)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--autor", help="wer den Lauf verantwortet (wandert ins Protokoll)")
    args = ap.parse_args(argv)

    # Governance-Hinweis VOR allen Validierungen: jeder Aufruf — auch ein
    # fehlschlagender — soll den experimentellen Status laut benennen (L6).
    print(GOVERNANCE_HINWEIS, file=sys.stderr)

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
    # Protokollpflicht (L6): JEDER abgeschlossene Lauf hinterlässt ein datiertes
    # Protokoll neben dem Artefakt — die Faktenbasis für die spätere
    # Freigabe-Entscheidung (Mehraugen-Prinzip, Governance-Betriebsordnung).
    protokoll = baue_protokoll(
        zeitstempel=datetime.now().isoformat(timespec="seconds"),
        argumente={"pairs": args.pairs, "base": args.base, "out": str(out),
                   "epochs": args.epochs, "batch": args.batch, "device": dev},
        datenquelle=args.pairs, n_kandidaten=len(kandidaten), n_paare=len(paare),
        n_bench_ausgeschlossen=len(bench_qs), basis_modell=args.base,
        out_pfad=str(out), autor=args.autor)
    p_pfad = schreibe_protokoll(protokoll, out)
    print(f"Lauf-Protokoll: {p_pfad}", file=sys.stderr)
    print(f"\nFertig: {out}  ({len(paare)} Paare, {args.epochs} Epoche(n))")
    print("Akzeptanz-Gate (NICHT automatisch übernehmen):")
    print(f"  python tools/embedding_bench.py --encoder {out} --save benchmarks/bge-m3-ft.json")
    print("  dann gegen benchmarks/bge-m3.json vergleichen — nur bei voller Guardrail behalten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
