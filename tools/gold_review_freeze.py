"""Brücke: Gold-v2-Review-Set (discovery/opus/gold_v2_review.yaml) -> Eval-Gold.

Das Review-Set trägt reiche Kuration (primaer mit Beleg-Zitatstelle, entscheid-
Block, klasse/stratum, unsicher). scoped_eval/organschaft_eval wollen schlank
`faelle: [{query, ziele}]`. Dieses Skript zieht die LABELS aus dem Review-Set:

  --quelle entscheid  vom Berater bestätigte `entscheid.primaer`. Verweigert
                      leeres entscheid (= noch nicht getrimmt).
  --quelle vorschlag  die `primaer`-Vorsortierung. Preview standardmäßig; als
                      Labels einfrierbar NUR mit --akzeptiere-vorschlag (= der
                      Berater akzeptiert die Vorsortierung pauschal).

COPYRIGHT: Das eingefrorene Gold trägt die Query-Texte = StBK-Examens-Sachverhalte
(quelle_lizenz: NICHT publizierbar). Es bleibt daher **gitignored unter discovery/**,
NICHT in fixtures/gold/ (getrackt -> Mirror/Forgejo). Reproduzierbar bleibt es über
das (ebenfalls lokale) gold_v2_review.yaml + dieses Skript (committet).

Negativ-Kandidaten (leere Ziele) gehören nicht in ein Recall-Eval -> separat
gezählt, nicht ausgespielt (Präzisionsmessung kommt mit echten Negativen v3).
Disziplin: NICHT auf diesem Set tunen; vor dem Freeze keinen Retrieval-Output
auf diese Fälle ansehen (Blindheit).

Aufruf:
  python tools/gold_review_freeze.py --quelle vorschlag                       # Preview
  python tools/gold_review_freeze.py --quelle vorschlag --akzeptiere-vorschlag --freeze
  python tools/gold_review_freeze.py --quelle entscheid --freeze              # nach Einzel-Trim
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.graph import graph_oeffnen          # noqa: E402
from risk_layer.idschema import para_level           # noqa: E402


def _norm_id(n: str) -> str:
    n = str(n).strip()
    return para_level(n if n.startswith("norm:") else f"norm:{n}")


def _ziele(fall: dict, quelle: str) -> list[str]:
    if quelle == "entscheid":
        roh = (fall.get("entscheid") or {}).get("primaer") or []
    else:
        roh = fall.get("primaer") or []
    out: list[str] = []
    for p in roh:
        n = p.get("norm") if isinstance(p, dict) else p
        if n:
            z = _norm_id(n)
            if z not in out:
                out.append(z)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Gold-v2-Review -> Eval-Gold (Brücke)")
    ap.add_argument("--in", dest="ein", default="discovery/opus/gold_v2_review.yaml")
    ap.add_argument("--quelle", choices=["entscheid", "vorschlag"], default="entscheid")
    ap.add_argument("--out", default=None, help="Default: discovery/ (preview) bzw. fixtures/gold/ (--freeze)")
    ap.add_argument("--freeze", action="store_true",
                    help="eingefrorenes Gold schreiben (gitignored discovery/, Copyright)")
    ap.add_argument("--akzeptiere-vorschlag", dest="akzeptiere_vorschlag", action="store_true",
                    help="Berater akzeptiert die primaer-Vorsortierung pauschal als Labels")
    args = ap.parse_args()

    review = yaml.safe_load(Path(args.ein).read_text(encoding="utf-8")) or {}
    faelle_roh = review.get("faelle", [])
    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2
    ist = lambda n: bool(graph.aufloesen(n).get("gefunden"))  # noqa: E731

    if args.freeze and args.quelle == "vorschlag" and not args.akzeptiere_vorschlag:
        print("--freeze aus --quelle vorschlag nur mit --akzeptiere-vorschlag "
              "(der Berater akzeptiert die Vorsortierung pauschal als Labels).", file=sys.stderr)
        return 2

    faelle: list[dict] = []
    negativ = unconfirmed = 0
    nicht_korpus: list[str] = []
    for f in faelle_roh:
        q = " ".join(str(f.get("query", "")).split())
        ziele = _ziele(f, args.quelle)
        if f.get("klasse") == "negativ_kandidat" and not ziele:
            negativ += 1
            continue
        if not ziele:
            unconfirmed += 1
            continue
        for z in ziele:
            if not ist(z):
                nicht_korpus.append(f"{f.get('fall_id')}:{z}")
        faelle.append({
            "query": q,
            "ziele": ziele,
            "fall_id": f.get("fall_id"),
            "quelle": f.get("quelle"),
            "klasse": f.get("klasse"),
            "stratum": f.get("stratum"),
            "unsicher": bool(f.get("unsicher")),
        })

    if args.quelle == "entscheid" and not faelle:
        print("entscheid.primaer ist überall leer — das Set ist noch nicht getrimmt.\n"
              "Erst der Berater füllt entscheid, dann --freeze. (Preview: --quelle vorschlag)",
              file=sys.stderr)
        return 1

    # Copyright: eingefrorenes Gold bleibt gitignored (discovery/), nie fixtures/gold.
    ziel = Path(args.out) if args.out else Path(
        "discovery/klausuren_v2.yaml" if args.freeze
        else f"discovery/klausuren_v2_eval_{args.quelle}.yaml")
    label_q = args.quelle + (" (pauschal akzeptiert)"
                             if args.quelle == "vorschlag" and args.akzeptiere_vorschlag else "")
    kopf = {
        "gold_version": review.get("meta", {}).get("version", "gold_v2"),
        "rolle": "test",
        "status": ("eingefroren" if args.freeze else f"PREVIEW ({args.quelle}, nicht messen)"),
        "herkunft": "StB-Examens-Musterklausuren (StBK Niedersachsen), kuratiert über gold_v2_review",
        "label_quelle": label_q,
        "erzeugt": str(date.today()),
        "n_faelle": len(faelle),
        "klassen": dict(Counter(x["klasse"] for x in faelle)),
        "negativ_kandidaten_separat": negativ,
        "unbestaetigt_uebersprungen": unconfirmed,
    }
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(yaml.safe_dump({**kopf, "faelle": faelle}, allow_unicode=True,
                                   sort_keys=False, width=100), encoding="utf-8")

    print(f"# Gold-v2 Eval-Brücke ({args.quelle})")
    print(f"  ausgespielt:   {len(faelle)} Fälle  {dict(Counter(x['klasse'] for x in faelle))}")
    print(f"  negativ separat: {negativ} | unbestätigt übersprungen: {unconfirmed}")
    print(f"  unsicher (Grenzfall): {sum(1 for x in faelle if x['unsicher'])}")
    if nicht_korpus:
        print(f"  ⚠ NICHT im Korpus: {nicht_korpus}")
    print(f"  -> {ziel}" + ("  [EINGEFROREN]" if args.freeze else "  [Preview, nicht messen]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
