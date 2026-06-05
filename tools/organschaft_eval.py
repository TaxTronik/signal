"""Organschaft-Retrieval-Eval (Ebene 1): hebt der Opus-Katalog das Retrieval?

Misst auf echtem, fachexperten-geschriebenem Organschaft-Gold (aus der Masterthesis,
Berater-Phrasierung -> Ziel-§) die reine Retrieval-Stärke in zwei Bedingungen:

  graph-only   — anker-freier Index (nur Gesetzestext der 2899 §)
  graph+Opus   — derselbe Index PLUS die Opus-Katalog-Paraphrasen als §-Anker

Isoliert den Opus-Beitrag exakt wie die dokumentierte anker-freie Bench-Methodik
(katalog=None, extra_anker=…). Lift = graph+Opus − graph-only, je Rechtsgebiet.
Opus deckt nur KStG (§§14–19) ab -> dort ist Lift erwartbar; GewSt/USt/AktG/AO
zeigen die Abdeckungslücke (kein Anker -> kein Lift). Content ≠ Lift wird so messbar.

Leckage: Opus-Anker, deren inhalt (normalisiert) == einer Gold-Anfrage ist, fliegen
raus (sonst injiziert man die Antwort). Gold-Anfragen sind ohnehin Laien-/Fallsätze,
die Anker sind §-Paraphrasen -> faktisch disjunkt.

Aufruf:
  python tools/organschaft_eval.py
  python tools/organschaft_eval.py --gold discovery/opus/kstg/organschaft_gold.yaml \
      --opus discovery/opus/kstg/discovery_kstg_organschaft.yaml
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

from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.graph import graph_oeffnen                        # noqa: E402
from risk_layer.idschema import para_level                        # noqa: E402


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _gebiet(norm_id: str) -> str:
    teile = norm_id.split(":")
    return teile[1] if len(teile) > 1 else "?"


def lade_gold(pfad: Path) -> list[tuple[str, set[str], str]]:
    daten = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
    gold: list[tuple[str, set[str], str]] = []
    for f in daten.get("faelle", []):
        q = _norm(str(f.get("query", "")))
        ziele = {para_level(z) for z in (f.get("ziele") or []) if z}
        if q and ziele:
            gold.append((q, ziele, str(f.get("quelle", ""))))
    return gold


def lade_opus_anker(pfad: Path, gold_queries: set[str]) -> tuple[list[tuple[str, str]], int]:
    """Opus-Eintraege -> (inhalt, quell_norm). Anker an die Norm, DIE der Eintrag
    charakterisiert (quell_norm) — Cross-Refs damit auf ihrer KStG-Quelle, nicht auf
    der referenzierten Fremdnorm (kein AktG/AO-Lift durch Meta-Sätze). Leckage raus."""
    daten = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
    anker: list[tuple[str, str]] = []
    leck = 0
    for d in daten.get("kandidaten", []):
        inhalt = str(d.get("inhalt", "")).strip()
        ziel = para_level(str(d.get("quell_norm", "")).strip())
        if not inhalt or not ziel:
            continue
        if _norm(inhalt) in gold_queries:       # Leckage: Anker == Gold-Anfrage
            leck += 1
            continue
        anker.append((inhalt, ziel))
    return anker, leck


def rang(idx, enc, satz: str, ziele: set[str], k: int = 10) -> int | None:
    treffer = idx.suche(satz, enc, k=k, min_sim=0.0)
    for i, t in enumerate(treffer):
        if para_level(t["norm_id"]) in ziele:
            return i + 1
    return None


def metriken(raenge: list[int | None]) -> dict:
    n = len(raenge)
    r5 = sum(1 for r in raenge if r and r <= 5)
    r10 = sum(1 for r in raenge if r and r <= 10)
    mrr = sum(1 / r for r in raenge if r)
    return {"n": n, "recall@5": round(r5 / n, 3) if n else 0.0,
            "recall@10": round(r10 / n, 3) if n else 0.0,
            "mrr": round(mrr / n, 3) if n else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Organschaft-Retrieval-Eval: Opus-Anker-Lift")
    ap.add_argument("--gold", default="fixtures/gold/organschaft_v1.yaml")
    ap.add_argument("--opus", default="discovery/opus/kstg/discovery_kstg_organschaft.yaml")
    ap.add_argument("--encoder", default="bge-m3")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--save", default="benchmarks/organschaft-opus-vs-graph.json")
    args = ap.parse_args()

    gold = lade_gold(Path(args.gold))
    if not gold:
        print(f"Kein Gold in {args.gold}", file=sys.stderr)
        return 2
    anker, leck = lade_opus_anker(Path(args.opus), {q for q, _, _ in gold})

    dev = os.environ.get("RISK_LAYER_EMB_DEVICE", "cuda")
    enc = encoder_oeffnen(args.encoder, device=dev)
    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2

    print(f"Gold: {len(gold)} Fälle | Opus-Anker: {len(anker)} "
          f"(Leckage verworfen: {leck}) | Encoder: {enc.name} ({dev})", file=sys.stderr)
    print("Baue Indizes (graph-only + graph+Opus) …", file=sys.stderr)
    idx_g = EmbeddingIndex.build(graph, enc, katalog=None)
    idx_o = EmbeddingIndex.build(graph, enc, katalog=None, extra_anker=anker)

    rows = []
    rg_all: list[int | None] = []
    ro_all: list[int | None] = []
    pro_gebiet: dict[str, list[list[int | None]]] = {}
    for satz, ziele, quelle in gold:
        rg = rang(idx_g, enc, satz, ziele, k=args.k)
        ro = rang(idx_o, enc, satz, ziele, k=args.k)
        rg_all.append(rg)
        ro_all.append(ro)
        geb = _gebiet(sorted(ziele)[0])
        pro_gebiet.setdefault(geb, [[], []])
        pro_gebiet[geb][0].append(rg)
        pro_gebiet[geb][1].append(ro)
        rows.append((geb, "|".join(sorted(ziele)), rg, ro, quelle, satz))

    m_g, m_o = metriken(rg_all), metriken(ro_all)

    def fmt(r):
        return "—" if r is None else str(r)

    print(f"\n# Organschaft-Retrieval (n={len(gold)}, Rang der Ziel-§; — = außerhalb Top-{args.k})\n")
    print(f"{'Gebiet':7} {'Ziel-§':22} {'graph':5} {'+Opus':5}  Quelle")
    print("-" * 78)
    for geb, ziel, rg, ro, quelle, _ in rows:
        mark = "  ▲" if (ro and (rg is None or ro < rg)) else ("  ▼" if (rg and (ro is None or ro > rg)) else "")
        print(f"{geb:7} {ziel:22} {fmt(rg):>5} {fmt(ro):>5}{mark}  {quelle[:30]}")

    print(f"\n# Gesamt        graph-only      graph+Opus      Lift")
    print(f"  Recall@5      {m_g['recall@5']:.2f}            {m_o['recall@5']:.2f}            "
          f"{m_o['recall@5']-m_g['recall@5']:+.2f}")
    print(f"  Recall@10     {m_g['recall@10']:.2f}            {m_o['recall@10']:.2f}            "
          f"{m_o['recall@10']-m_g['recall@10']:+.2f}")
    print(f"  MRR           {m_g['mrr']:.3f}           {m_o['mrr']:.3f}           "
          f"{m_o['mrr']-m_g['mrr']:+.3f}")

    print(f"\n# Je Rechtsgebiet (R@5 / MRR)  [Opus deckt nur KStG]")
    geb_json = {}
    for geb in sorted(pro_gebiet):
        rg_l, ro_l = pro_gebiet[geb]
        mg, mo = metriken(rg_l), metriken(ro_l)
        geb_json[geb] = {"n": mg["n"], "graph": mg, "opus": mo}
        print(f"  {geb:7} n={mg['n']:2d}  graph R@5={mg['recall@5']:.2f} MRR={mg['mrr']:.3f}"
              f"   +Opus R@5={mo['recall@5']:.2f} MRR={mo['mrr']:.3f}"
              f"   ΔMRR={mo['mrr']-mg['mrr']:+.3f}")

    ergebnis = {
        "stand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "encoder": enc.name, "n": len(gold), "k": args.k,
        "opus_anker": len(anker), "leckage_verworfen": leck,
        "graph_only": m_g, "graph_plus_opus": m_o,
        "pro_gebiet": geb_json,
        "raenge": [{"gebiet": g, "ziel": z, "graph": rg, "opus": ro, "quelle": q}
                   for g, z, rg, ro, q, _ in rows],
    }
    ziel = Path(args.save)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
