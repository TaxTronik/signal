"""Discovery-Runner (Pass 1): streamt §-Knoten des Normgraphen durch ein LLM und
schreibt die gescorten Kandidaten als Review-Queue nach ``discovery/`` (gitignored).

Concurrency + Shards/Checkpointing:
- Pro Shard werden die LLM-Calls PARALLEL ausgeführt (roh_kandidaten = reines HTTP,
  thread-safe); die graph-gebundene verarbeite_kandidaten läuft danach im
  HAUPTTHREAD (SQLite ist nicht thread-safe).
- Jeder fertige Shard wird ATOMAR nach discovery/partials/shard_NNNN.json
  geschrieben → crash-sicher. Bei Neustart mit gleichem --out werden fertige Shards
  ÜBERSPRUNGEN (resumebar). Am Ende Merge → review_queue/training_pairs/alle.

KEINE Auto-Commits in Katalog/Graph — nur Vorschläge für den Berater.

Aufruf:
  python tools/discovery_run.py --limit 20                          # Probelauf
  python tools/discovery_run.py --concurrency 8 --shard-size 200    # Voll-Lauf
  python tools/discovery_run.py --no-think                          # Qwen
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole, verweist_ids     # noqa: E402

from risk_layer.discovery import (                       # noqa: E402
    Kandidat, besteht_review_gate, besteht_training_gate, erkenne_modell,
    iter_paragraphen, roh_kandidaten, server_verfuegbar, verarbeite_kandidaten,
)
from risk_layer.graph import graph_oeffnen               # noqa: E402
from risk_layer.persistenz import atomar_schreiben       # noqa: E402

utf8_konsole()


def _schreibe_yaml(pfad: Path, kopf: dict, kandidaten: list[dict]) -> None:
    atomar_schreiben(pfad, yaml.safe_dump({**kopf, "n": len(kandidaten), "kandidaten": kandidaten},
                                          allow_unicode=True, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="Discovery Pass 1 über den Normtext-Korpus")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--limit", type=int, default=0, help="nur N §-Knoten (0 = alle)")
    ap.add_argument("--out", default="discovery", help="Zielverzeichnis (gitignored)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--no-think", action="store_true", help="/no_think anhängen (Qwen)")
    ap.add_argument("--model", default="",
                    help="Modellname für den Chat-Request (vLLM PFLICHT). Leer = Auto via /v1/models")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallele LLM-Calls je Shard (vorsichtig hochsetzen: 8→16→32)")
    ap.add_argument("--shard-size", type=int, default=200,
                    help="§ pro Shard (Checkpoint-Granularität, resumebar)")
    ap.add_argument("--enable-begriff", action="store_true",
                    help="begriff-Kandidaten zulassen (Default aus: Substantiv-Rauschen)")
    ap.add_argument("--enable-normrelation", action="store_true",
                    help="normrelation-Kandidaten zulassen (Default aus: Pass-1-Rauschen)")
    ap.add_argument("--split-ab", type=int, default=0,
                    help="§ länger als N Zeichen in Absatz-Knoten zerlegen (0 = aus; "
                         "gegen stille Format-/Timeout-Ausfälle bei dichten Normen)")
    args = ap.parse_args()

    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph (corpus/graph.sqlite) — erst 'graph build'.", file=sys.stderr)
        return 2
    if not server_verfuegbar(args.llm_url, timeout=3.0):
        print(f"Kein LLM-Server unter {args.llm_url}.", file=sys.stderr)
        return 2
    modell = args.model or erkenne_modell(args.llm_url)
    print(f"LLM-Modell: {modell or '(unbekannt — bei vLLM --model setzen)'}", file=sys.stderr)

    @lru_cache(maxsize=None)
    def ist_im_graph(nid: str) -> bool:
        return bool(graph.aufloesen(nid).get("gefunden"))

    nodes = list(iter_paragraphen(graph, split_ab=args.split_ab or None))
    if args.limit:
        nodes = nodes[:args.limit]
    out = Path(args.out)
    partials = out / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    shard_size = max(1, args.shard_size)
    shards = [nodes[i:i + shard_size] for i in range(0, len(nodes), shard_size)]
    print(f"{len(nodes)} § in {len(shards)} Shards à {shard_size}, concurrency {args.concurrency}",
          file=sys.stderr)

    fehler: list[dict] = []                         # Failure-Ledger statt stillem return []

    def _roh(node):
        return roh_kandidaten(args.llm_url, node, model=modell, timeout=args.timeout,
                              no_think=args.no_think, enable_begriff=args.enable_begriff,
                              enable_normrelation=args.enable_normrelation,
                              fehler_protokoll=fehler)

    kum = 0
    for si, shard in enumerate(shards):
        pfad = partials / f"shard_{si:04d}.json"
        if pfad.exists():                       # resume: fertigen Shard überspringen
            kum += len(json.loads(pfad.read_text(encoding="utf-8")))
            print(f"Shard {si + 1}/{len(shards)}: übersprungen (vorhanden)", file=sys.stderr)
            continue
        # Phase 1: LLM-Calls PARALLEL (thread-safe, kein Graph)
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            rohs = list(ex.map(_roh, shard))
        # Phase 2: Graph-Verarbeitung im HAUPTTHREAD (SQLite nicht thread-safe)
        shard_kand: list[dict] = []
        for node, roh in zip(shard, rohs):
            for k in verarbeite_kandidaten(
                    roh, quell_norm=node["id"], quell_law=node["law"],
                    node_text=node["text"], node_titel=node["titel"],
                    verweist_ids=verweist_ids(graph, node["id"]), ist_im_graph=ist_im_graph,
                    enable_begriff=args.enable_begriff, enable_normrelation=args.enable_normrelation):
                shard_kand.append(k.as_dict())
        atomar_schreiben(pfad, json.dumps(shard_kand, ensure_ascii=False))
        kum += len(shard_kand)
        print(f"Shard {si + 1}/{len(shards)} ({len(shard)} §) → {len(shard_kand)} Kand. "
              f"(kumuliert {kum})", file=sys.stderr)

    # Merge aller Shards → finale YAMLs
    alle_dicts: list[dict] = []
    for pfad in sorted(partials.glob("shard_*.json")):
        alle_dicts.extend(json.loads(pfad.read_text(encoding="utf-8")))
    alle = [Kandidat(**d) for d in alle_dicts]
    review = [k for k in alle if besteht_review_gate(k)]
    training = [k for k in alle if besteht_training_gate(k)]

    kopf = {"stand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "paragraphen": len(nodes), "kandidaten_gesamt": len(alle)}
    _schreibe_yaml(out / "review_queue.yaml", kopf, [k.as_dict() for k in review])
    _schreibe_yaml(out / "training_pairs.yaml", kopf, [k.as_dict() for k in training])
    _schreibe_yaml(out / "alle_kandidaten.yaml", kopf, alle_dicts)
    if fehler:
        atomar_schreiben(out / "failures.json", json.dumps(fehler, ensure_ascii=False, indent=2))

    vo = sum(1 for k in alle if k.quelle_typ == "verordnung")
    print(f"\n{len(nodes)} § → {len(alle)} Kandidaten ({vo} aus Verordnungen) · "
          f"Review {len(review)} · Training {len(training)}")
    if fehler:
        print(f"AUSFÄLLE: {len(fehler)} § ohne Antwort (Server/Format) → {out}/failures.json")
    print(f"geschrieben: {out}/review_queue.yaml, training_pairs.yaml, alle_kandidaten.yaml")
    print(f"(Shards in {partials}/ — Neustart mit gleichem --out überspringt fertige)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
