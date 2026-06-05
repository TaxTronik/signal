"""Card-QC — Vertrauens-Audit für Card-Discovery-Anker (gpt-5.5 o. a.).

trainingsdaten-unabhängig: wir glauben dem Modell nichts, wir prüfen es — und zwar
mechanisch, ohne LLM-Judge (sonst tauscht man nur ein Modellvertrauen gegen ein
anderes). Drei Maße:

 1. Faithfulness (Gate-Spiegel): Anker mit kritischem Befund (norm_unbelegt) — MUSS 0
    sein; zahl_unbelegt nur beratend. Bestätigt, dass kein halluziniertes §/Zahl-Zitat
    durchrutscht.
 2. Diskriminierung: jeder Anker als Query gegen den graph-only-Index — Rang SEINER
    EIGENEN Norm unter allen 2899 §§. WICHTIG zur Einordnung: ein niedriger Rang ist
    bei guten LAIEN-Ankern NORMAL (genau die Register-Lücke, die der Anker überbrückt).
    Das echte Warnsignal ist DRIFT: die eigene Norm tief vergraben UND eine ANDERE
    konkrete Norm dominiert (rank-1, hohe Similarity) → der Anker beschreibt evtl. die
    falsche Norm. Diese Fälle werden gelistet (für den menschlichen Blick).
 3. (separat) Lift auf UNABHÄNGIGEM Gold via organschaft_eval / embedding_bench
    --extra-anker-from — der stärkste Trust-Beleg: ein falscher Anker hebt das richtige
    Gold nicht.

Aufruf:
  python tools/card_qc.py --cards-dir discovery/api/cards
"""

from __future__ import annotations

import argparse
import glob
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

from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.graph import graph_oeffnen                        # noqa: E402
from risk_layer.idschema import para_level                        # noqa: E402


def _lade_anker(cards_dir: Path) -> list[dict]:
    """Alle paraphrase-Kandidaten aus den per-Norm-Karten (mit Befunden)."""
    anker = []
    for p in sorted(glob.glob(str(cards_dir / "norm_*.yaml"))):
        d = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        for k in d.get("kandidaten", []):
            if k.get("typ") == "paraphrase" and k.get("inhalt"):
                ziel = para_level(k.get("ziel_norm_id") or k.get("quell_norm") or "")
                anker.append({"inhalt": k["inhalt"], "ziel": ziel,
                              "befunde": k.get("befunde") or []})
    return anker


def main() -> int:
    ap = argparse.ArgumentParser(description="Card-QC: Vertrauens-Audit der Anker")
    ap.add_argument("--cards-dir", default="discovery/api/cards")
    ap.add_argument("--encoder", default="bge-m3")
    ap.add_argument("--k", type=int, default=50, help="Suchtiefe für den Eigen-Rang")
    ap.add_argument("--drift-rang", type=int, default=20,
                    help="eigene Norm schlechter als dieser Rang + andere Norm rank-1 = Drift-Verdacht")
    ap.add_argument("--save", default="benchmarks/card-qc.json")
    args = ap.parse_args()

    anker = _lade_anker(Path(args.cards_dir))
    if not anker:
        print(f"Keine paraphrase-Anker in {args.cards_dir}", file=sys.stderr)
        return 2

    # 1) Faithfulness
    krit = [a for a in anker if any(b.split(":", 1)[0] == "norm_unbelegt" for b in a["befunde"])]
    zahl = [a for a in anker if any(b.split(":", 1)[0] == "zahl_unbelegt" for b in a["befunde"])]

    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen(args.encoder, device=dev)
    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2
    print(f"Anker: {len(anker)} | Encoder: {enc.name} ({dev}) | baue graph-only-Index …",
          file=sys.stderr)
    idx = EmbeddingIndex.build(graph, enc, katalog=None)

    # 2) Diskriminierung
    r1 = r5 = r20 = miss = 0
    drift = []
    for a in anker:
        treffer = idx.suche(a["inhalt"], enc, k=args.k, min_sim=0.0)
        paras = [para_level(t["norm_id"]) for t in treffer]
        rang = next((i + 1 for i, p in enumerate(paras) if p == a["ziel"]), None)
        if rang == 1:
            r1 += 1
        elif rang and rang <= 5:
            r5 += 1
        elif rang and rang <= 20:
            r20 += 1
        else:
            miss += 1
        top1, top1_sim = (paras[0], treffer[0]["sim"]) if treffer else ("", 0.0)
        if (rang is None or rang > args.drift_rang) and top1 and top1 != a["ziel"]:
            drift.append({"anker": a["inhalt"][:90], "eigene_norm": a["ziel"],
                          "stattdessen": top1, "sim": round(top1_sim, 3),
                          "eigen_rang": rang})

    n = len(anker)
    def pct(x): return round(x / n, 3)
    print(f"\n# Card-QC  (n={n} Anker)\n")
    print("1) FAITHFULNESS")
    print(f"   kritisch (norm_unbelegt): {len(krit)}   {'OK' if not krit else '⚠ PRÜFEN'}")
    print(f"   beratend (zahl_unbelegt): {len(zahl)}")
    print("\n2) DISKRIMINIERUNG (Eigen-Norm-Rang im graph-only-Index)")
    print(f"   rank-1 : {r1:3d} ({pct(r1):.0%})   stark norm-spezifisch")
    print(f"   top-5  : {r5:3d} ({pct(r5):.0%})")
    print(f"   top-20 : {r20:3d} ({pct(r20):.0%})")
    print(f"   tiefer : {miss:3d} ({pct(miss):.0%})   (bei Laien-Ankern normal — Register-Lücke)")
    print(f"\n   DRIFT-VERDACHT ({len(drift)}): eigene Norm vergraben + andere Norm rank-1")
    for d in drift[:25]:
        print(f"   · {d['eigene_norm']:14} statt→ {d['stattdessen']:14} sim={d['sim']}  «{d['anker']}»")

    erg = {"stand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n": n, "encoder": enc.name,
           "faithfulness": {"norm_unbelegt": len(krit), "zahl_unbelegt": len(zahl)},
           "diskriminierung": {"rank1": r1, "top5": r5, "top20": r20, "tiefer": miss},
           "drift_verdacht": drift}
    ziel = Path(args.save)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(erg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
