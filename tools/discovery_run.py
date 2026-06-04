"""Discovery-Runner (Pass 1): streamt §-Knoten des Normgraphen durch ein LLM und
schreibt die gescorten Kandidaten als Review-Queue nach ``discovery/`` (gitignored).

KEINE Auto-Commits in Katalog/Graph — nur Vorschläge für den Berater.

Aufruf:
  python tools/discovery_run.py --limit 20            # Probelauf (lokal, granite)
  python tools/discovery_run.py --llm-url http://127.0.0.1:8080 --out discovery
  python tools/discovery_run.py --no-think            # für Qwen-Modelle
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_layer.discovery import (                       # noqa: E402
    besteht_review_gate, besteht_training_gate, discovery_paragraph,
    iter_paragraphen, server_verfuegbar,
)
from risk_layer.graph import graph_oeffnen               # noqa: E402


def _verweist_ids(graph, node_id: str) -> set[str]:
    """Ausgehende verweist_auf-Kanten eines §-Knotens (für die Relations-Erdung)."""
    res = graph.aufloesen(node_id)
    ids: set[str] = set()
    for v in res.get("verweist_auf") or []:
        i = (v.get("id") or v.get("dst") or v.get("ziel") or "") if isinstance(v, dict) else str(v)
        if i:
            ids.add(i)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description="Discovery Pass 1 über den Normtext-Korpus")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8080")
    ap.add_argument("--limit", type=int, default=0, help="nur N §-Knoten (0 = alle)")
    ap.add_argument("--out", default="discovery", help="Zielverzeichnis (gitignored)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--no-think", action="store_true", help="/no_think anhängen (Qwen)")
    args = ap.parse_args()

    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph (corpus/graph.sqlite) — erst 'graph build'.", file=sys.stderr)
        return 2
    if not server_verfuegbar(args.llm_url, timeout=3.0):
        print(f"Kein llama-server unter {args.llm_url}.", file=sys.stderr)
        return 2

    @lru_cache(maxsize=None)
    def ist_im_graph(nid: str) -> bool:
        return bool(graph.aufloesen(nid).get("gefunden"))

    alle = []
    nodes = iter_paragraphen(graph)
    n = 0
    for node in nodes:
        if args.limit and n >= args.limit:
            break
        n += 1
        kandidaten = discovery_paragraph(
            args.llm_url, node, ist_im_graph=ist_im_graph,
            verweist_ids=_verweist_ids(graph, node["id"]),
            timeout=args.timeout, no_think=args.no_think)
        alle.extend(kandidaten)
        if n % 50 == 0:
            print(f"  … {n} §  /  {len(alle)} Kandidaten", file=sys.stderr)

    review = [k for k in alle if besteht_review_gate(k)]
    training = [k for k in alle if besteht_training_gate(k)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stand = datetime.now(timezone.utc).isoformat(timespec="seconds")
    kopf = {"stand": stand, "paragraphen": n, "kandidaten_gesamt": len(alle)}

    (out / "review_queue.yaml").write_text(
        yaml.safe_dump({**kopf, "n": len(review), "kandidaten": [k.as_dict() for k in review]},
                       allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "training_pairs.yaml").write_text(
        yaml.safe_dump({**kopf, "n": len(training), "kandidaten": [k.as_dict() for k in training]},
                       allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "alle_kandidaten.yaml").write_text(
        yaml.safe_dump({**kopf, "kandidaten": [k.as_dict() for k in alle]},
                       allow_unicode=True, sort_keys=False), encoding="utf-8")

    vo = sum(1 for k in alle if k.quelle_typ == "verordnung")
    print(f"\n{n} § analysiert → {len(alle)} Kandidaten "
          f"({vo} aus Verordnungen) · Review {len(review)} · Training {len(training)}")
    print(f"geschrieben: {out}/review_queue.yaml, training_pairs.yaml, alle_kandidaten.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
