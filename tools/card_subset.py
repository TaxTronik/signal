"""Deterministische Subset-Liste für die Card-Discovery (Pass 2).

Governance: das MODELL entscheidet NICHT, welche Normen eine Karte bekommen — das
tut ein nachvollziehbares Korpus-Signal. v1 nutzt zwei deterministische Quellen:

  - Katalog-Scope   : Normen, die der kuratierte Katalog bereits als Ziel führt
                      (norm_anker / normketten) — das sind die, die uns interessieren.
  - Cross-Ref-1-Hop : optional deren im Graph zitierte Nachbarnormen (--expand).

Lücken-Signal: Katalog-Normen OHNE Paraphrasen zuerst (dort fehlen Anker am ehesten).
Ausgabe: eine Textdatei (ein norm_id je Zeile, # = Kommentar) als --subset-Input für
card_discovery.

Aufruf:
  python tools/card_subset.py --seed tools/cards_seed.txt --expand --out discovery/cards_subset.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.catalog import lade_katalog          # noqa: E402
from risk_layer.graph import graph_oeffnen           # noqa: E402
from risk_layer.idschema import para_level           # noqa: E402
from risk_layer.zitate import parse_normzitat        # noqa: E402


def _ziele(eintrag) -> set[str]:
    quellen = list(getattr(eintrag, "norm_anker", ()) or ())
    quellen += [k.primaer for k in (getattr(eintrag, "normketten", ()) or ()) if k.primaer]
    out: set[str] = set()
    for z in quellen:
        for a in parse_normzitat(z):
            if a.norm_id:
                out.add(para_level(a.norm_id.to_id()))
    return out


def _verweist(graph, nid: str) -> set[str]:
    res = graph.aufloesen(nid)
    out: set[str] = set()
    for v in res.get("verweist_auf") or []:
        i = (v.get("id") or v.get("dst") or v.get("ziel") or "") if isinstance(v, dict) else str(v)
        if i:
            out.add(para_level(i))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministisches Card-Discovery-Subset")
    ap.add_argument("--seed", default="", help="Datei mit Prioritäts-norm_ids (ein je Zeile, "
                    "# Kommentar) — geht dem Korpus-Signal voran (gepflegte Liste)")
    ap.add_argument("--expand", action="store_true", help="1-Hop verweist_auf-Nachbarn aufnehmen")
    ap.add_argument("--limit", type=int, default=0, help="auf N Normen kürzen (0 = alle)")
    ap.add_argument("--out", default="discovery/cards_subset.txt")
    args = ap.parse_args()

    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2
    kat = lade_katalog()

    katalog_normen: set[str] = set()
    ohne_paraphrasen: set[str] = set()
    for e in kat:
        ziele = _ziele(e)
        katalog_normen |= ziele
        if not (getattr(e, "paraphrasen", ()) or ()):
            ohne_paraphrasen |= ziele

    def im_graph(nid: str) -> bool:
        return bool(graph.aufloesen(nid).get("gefunden"))

    # Rang: 0) gepflegte Seed-Prioritätsliste 1) Katalog-Lücken (kein Paraphrasen-Anker)
    #       2) übrige Katalog-Normen 3) optional Cross-Ref-Nachbarn. Stabil sortiert.
    seed: list[str] = []
    if args.seed:
        for zeile in Path(args.seed).read_text(encoding="utf-8").splitlines():
            z = zeile.split("#", 1)[0].strip()
            if z and im_graph(z):
                seed.append(z)
    luecken = sorted(n for n in (ohne_paraphrasen & katalog_normen) if im_graph(n))
    rest_kat = sorted(n for n in katalog_normen if n not in ohne_paraphrasen and im_graph(n))
    gewaehlt = list(dict.fromkeys(seed + luecken + rest_kat))

    nachbarn: list[str] = []
    if args.expand:
        seen = set(gewaehlt)
        for n in gewaehlt:
            for m in sorted(_verweist(graph, n)):
                if m not in seen and im_graph(m):
                    seen.add(m)
                    nachbarn.append(m)
        gewaehlt += nachbarn

    if args.limit:
        gewaehlt = gewaehlt[:args.limit]

    zeilen = [f"# Card-Discovery-Subset (deterministisch) — {len(gewaehlt)} Normen",
              f"# Seed {len(seed)} · Katalog-Lücken {len(luecken)} · übrige Katalog "
              f"{len(rest_kat)} · Cross-Ref-Nachbarn {len(nachbarn)}",
              "# Auswahl = Seed + Korpus-Signal, NICHT Modellurteil.", ""]
    seed_set, luecken_set = set(seed), set(luecken)
    for n in gewaehlt:
        marke = ("seed" if n in seed_set else "luecke" if n in luecken_set
                 else "katalog" if n in katalog_normen else "nachbar")
        zeilen.append(f"{n}    # {marke}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    print(f"{len(gewaehlt)} Normen -> {out}  (Seed {len(seed)} / Lücken {len(luecken)} / "
          f"Katalog {len(rest_kat)} / Nachbarn {len(nachbarn)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
