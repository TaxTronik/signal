"""Lemma-Eval /LF50/: misst den Lemma-Robustheitspfad gegen fixtures/gold/lemma_v1.yaml.

Läuft den Schicht-1-Matcher in zwei Bedingungen über jeden Gold-Satz:

  ohne Lemma — nur der wörtliche Bestand (``Schicht1Matcher.match``); per
               Gold-Konstruktion ist hier Recall 0 erwartet (jeder Fall wurde
               beim Erstellen programmatisch als heute-unauffindbar verifiziert),
  mit Lemma  — wörtlich PLUS Lemma-Pfad (``match_lemma`` + Dedupe exakt-vor-
               Lemma), mit ECHTEM spaCy-Modell (de_core_news_lg, Fallback sm).

Berichtet Recall beider Modi, Span-Exaktheit der Lemma-Treffer (Treffer-Span ==
Gold-Span) und Latenz. ``bekannt_hart``-Fälle (dokumentierte Modell-Grenzen,
s. Gold-Header) werden zusätzlich getrennt ausgewiesen – sie zählen im
Gesamt-Recall ehrlich mit.

Aufruf:
  python tools/lemma_eval.py
  python tools/lemma_eval.py --gold fixtures/gold/lemma_v1.yaml --bericht out/lemma_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole  # noqa: E402

utf8_konsole()

from risk_layer import ingest as ing                                   # noqa: E402
from risk_layer import lemma as lemma_mod                              # noqa: E402
from risk_layer.catalog import lade_katalog                            # noqa: E402
from risk_layer.matcher import Schicht1Matcher, entferne_lemma_dubletten  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_GOLD_DEFAULT = _REPO / "fixtures" / "gold" / "lemma_v1.yaml"


def _lade_gold(pfad: Path) -> list[dict]:
    d = yaml.safe_load(pfad.read_text(encoding="utf-8"))
    faelle = d.get("faelle") or []
    if d.get("n") != len(faelle):
        raise SystemExit(f"Gold inkonsistent: n={d.get('n')} != {len(faelle)} Fälle")
    for f in faelle:
        if f["satz"][f["start"]:f["end"]] != f["span_text"]:
            raise SystemExit(f"Gold {f['id']}: start/end passt nicht zu span_text")
        if f["satz"].count(f["span_text"]) != 1:
            raise SystemExit(f"Gold {f['id']}: span_text nicht eindeutig im Satz")
    return faelle


def main() -> int:
    ap = argparse.ArgumentParser(description="Lemma-Robustheitspfad /LF50/ gegen Gold messen")
    ap.add_argument("--gold", default=str(_GOLD_DEFAULT), help="Gold-Datei (lemma_v1.yaml)")
    ap.add_argument("--katalog", default=None, help="Katalogdatei (Default: catalog/begriffe.yaml)")
    ap.add_argument("--bericht", default=None, help="Verzeichnis für JSON-Bericht (optional)")
    args = ap.parse_args()

    faelle = _lade_gold(Path(args.gold))
    katalog = lade_katalog(args.katalog)
    matcher = Schicht1Matcher(katalog)

    # Echtes Modell laden (einmalig, gemessen): ohne Modell ist der Lemma-Pfad
    # per Design ein No-Op – das wäre hier ein Messfehler, kein Ergebnis.
    t0 = time.perf_counter()
    nlp = lemma_mod._lade_nlp()
    t_load = time.perf_counter() - t0
    if nlp is None:
        raise SystemExit("Kein spaCy-Modell ladbar — Eval braucht das echte Modell "
                         "(pip install --require-hashes --only-binary=:all: "
                         "-r requirements-ner-lock.txt).")
    meta = getattr(nlp, "meta", {}) or {}
    modell = f"{meta.get('lang', '?')}_{meta.get('name', '?')} {meta.get('version', '?')}"

    # Katalog-Oberflächen einmalig lemmatisieren/kompilieren (gemessen, lazy).
    t0 = time.perf_counter()
    n_muster = len(matcher._lemma_muster())
    t_katalog = time.perf_counter() - t0

    ergebnisse: list[dict] = []
    t_lemma_gesamt = 0.0
    for f in faelle:
        res = ing.ingest(f["satz"])
        exakt = matcher.match(res)
        ohne = any(t.begriff_id == f["begriff_id"] for t in exakt)
        t0 = time.perf_counter()
        lem = entferne_lemma_dubletten(exakt, matcher.match_lemma(res))
        t_lemma_gesamt += time.perf_counter() - t0
        alle = exakt + lem
        mit = any(t.begriff_id == f["begriff_id"] for t in alle)
        span_ok = any(t.begriff_id == f["begriff_id"]
                      and t.start == f["start"] and t.end == f["end"] for t in alle)
        ergebnisse.append({
            "id": f["id"], "begriff_id": f["begriff_id"], "kategorie": f.get("kategorie"),
            "span_text": f["span_text"], "hart": bool(f.get("bekannt_hart")),
            "ohne": ohne, "mit": mit, "span_exakt": span_ok,
        })

    n = len(ergebnisse)
    n_hart = sum(1 for e in ergebnisse if e["hart"])
    r_ohne = sum(e["ohne"] for e in ergebnisse)
    r_mit = sum(e["mit"] for e in ergebnisse)
    r_span = sum(e["span_exakt"] for e in ergebnisse)
    kern = [e for e in ergebnisse if not e["hart"]]
    r_mit_kern = sum(e["mit"] for e in kern)

    print(f"Lemma-Eval /LF50/  —  Gold: {Path(args.gold).name} (n={n}, davon {n_hart} bekannt_hart)")
    print(f"Modell             : {modell}  (Ladezeit {t_load:.1f}s; "
          f"{n_muster} Katalog-Muster in {t_katalog * 1000:.0f} ms)")
    print(f"Recall OHNE Lemma  : {r_ohne}/{n}  = {r_ohne / n:.1%}")
    print(f"Recall MIT  Lemma  : {r_mit}/{n}  = {r_mit / n:.1%}")
    print(f"  davon Kernfälle  : {r_mit_kern}/{len(kern)}  = {r_mit_kern / len(kern):.1%}"
          f"  (ohne bekannt_hart)")
    print(f"Span-exakt         : {r_span}/{r_mit}  der gefundenen Fälle"
          + (f"  = {r_span / r_mit:.1%}" if r_mit else ""))
    print(f"Latenz Lemma-Pfad  : {t_lemma_gesamt / n * 1000:.1f} ms/Satz "
          f"(gesamt {t_lemma_gesamt:.2f}s für {n} Sätze)")

    fehl = [e for e in ergebnisse if not e["mit"]]
    if fehl:
        print("\nNicht gefunden (mit Lemma):")
        for e in fehl:
            hart = " [bekannt_hart]" if e["hart"] else ""
            print(f"  {e['id']}  {e['begriff_id']}  {e['span_text']!r}{hart}")
    unscharf = [e for e in ergebnisse if e["mit"] and not e["span_exakt"]]
    if unscharf:
        print("\nGefunden, aber Span unscharf:")
        for e in unscharf:
            print(f"  {e['id']}  {e['begriff_id']}  {e['span_text']!r}")

    if args.bericht:
        out_dir = Path(args.bericht)
        out_dir.mkdir(parents=True, exist_ok=True)
        bericht = {
            "gold": str(args.gold), "n": n, "n_hart": n_hart, "modell": modell,
            "recall_ohne": r_ohne / n, "recall_mit": r_mit / n,
            "recall_mit_kern": r_mit_kern / len(kern) if kern else None,
            "span_exakt": (r_span / r_mit) if r_mit else None,
            "latenz_ms_pro_satz": t_lemma_gesamt / n * 1000,
            "modell_ladezeit_s": t_load,
            "faelle": ergebnisse,
        }
        pfad = out_dir / "lemma_eval.json"
        pfad.write_text(json.dumps(bericht, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"\n→ Bericht: {pfad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
