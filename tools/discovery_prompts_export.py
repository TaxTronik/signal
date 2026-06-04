"""Export der EXAKTEN Discovery-Prompts für eine stratifizierte §-Stichprobe.

Für den „gleiches mit gleichem"-A/B (Opus vs. Qwen als Discovery-Modell): beide
Arme müssen denselben Input bekommen. Dieses Tool erzeugt für eine geschichtete
Stichprobe (kurz/mittel/lang über die Gesetze gestreut) die byte-identischen
System+User-Prompts aus DISCOVERY_SYSTEM + baue_user_prompt — so wie der Harness
sie an das LLM schickt (Pass-1-Default: begriff/normrelation AUS).

Ausgabe:
  - <out>.md    menschenlesbar (Bruder beantwortet diese von Hand mit Opus)
  - <out>.jsonl je Zeile {id, zitat, law, n_chars, bucket, system, user}
                (reproduzierbar an Qwen/jedes Endpoint refütterbar)

Aufruf:
  python tools/discovery_prompts_export.py --per-bucket 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.discovery import (DISCOVERY_SYSTEM, baue_user_prompt,  # noqa: E402
                                  iter_paragraphen)
from risk_layer.graph import graph_oeffnen                            # noqa: E402


def _law_round_robin(nodes: list[dict], k: int, rng: random.Random) -> list[dict]:
    """k Knoten ziehen, möglichst breit über die Gesetze (Round-Robin je law)."""
    pro_law: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        pro_law[n["law"]].append(n)
    for v in pro_law.values():
        rng.shuffle(v)
    laws = sorted(pro_law)
    rng.shuffle(laws)
    out: list[dict] = []
    while len(out) < k and any(pro_law[l] for l in laws):
        for l in laws:
            if pro_law[l]:
                out.append(pro_law[l].pop())
                if len(out) >= k:
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Stratifizierter Discovery-Prompt-Export")
    ap.add_argument("--per-bucket", type=int, default=10, help="§ je Längen-Bucket (kurz/mittel/lang)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="discovery/prompts_sample")
    args = ap.parse_args()

    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2
    nodes = list(iter_paragraphen(graph))
    lens = sorted(len(n["text"]) for n in nodes)
    q1, q2 = lens[len(lens) // 3], lens[2 * len(lens) // 3]    # Terzile der Textlänge

    buckets: dict[str, list[dict]] = {"kurz": [], "mittel": [], "lang": []}
    for n in nodes:
        L = len(n["text"])
        buckets["kurz" if L < q1 else "lang" if L >= q2 else "mittel"].append(n)

    rng = random.Random(args.seed)
    gewaehlt: list[tuple[str, dict]] = []
    for b in ("kurz", "mittel", "lang"):
        for n in _law_round_robin(buckets[b], args.per_bucket, rng):
            gewaehlt.append((b, n))

    print(f"§ gesamt {len(nodes)} | Terzile <{q1} / {q1}-{q2} / ≥{q2} Zeichen | "
          f"gewählt {len(gewaehlt)}", file=sys.stderr)

    md = [f"# Discovery-Prompt-Export (stratifiziert, seed={args.seed})",
          "",
          f"{len(gewaehlt)} §§ · je {args.per_bucket} kurz/mittel/lang · über die Gesetze gestreut.",
          "Beantworte jeden Block unter Contract (NUR das JSON nach Schema; Deckel ≤3 "
          "Paraphrasen; Pass-1: begriff/normrelation AUS).", ""]
    jsonl = []
    for b, n in gewaehlt:
        user = baue_user_prompt(n["zitat"], n["titel"], n["text"])
        md += [f"## {n['zitat']}  ·  {b}  ·  {len(n['text'])} Zeichen  ·  `{n['id']}`",
               "", "```text", "SYSTEM:", DISCOVERY_SYSTEM, "", "USER:", user, "```", ""]
        jsonl.append({"id": n["id"], "zitat": n["zitat"], "law": n["law"],
                      "n_chars": len(n["text"]), "bucket": b,
                      "system": DISCOVERY_SYSTEM, "user": user})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    out.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in jsonl), encoding="utf-8")
    print(f"geschrieben: {out.with_suffix('.md')} · {out.with_suffix('.jsonl')}")
    # Verteilung je Gesetz zur Kontrolle
    from collections import Counter
    print("je Gesetz:", dict(sorted(Counter(n["law"] for _, n in gewaehlt).items())), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
