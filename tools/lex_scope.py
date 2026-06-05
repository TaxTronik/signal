"""Lexikalische Scope-Stufe (billigste ungemessene Sprosse vor dem LLM).

Begründung (aus der Graph-Probe): das Gold-Signal ist lexikalisch stark
(Vorsteuer→UStG, AfA→EStG), nur Graph-Embedding trennt es nicht. Also Scope rein
lexikalisch schätzen, KEINE GPU:
  1. Gesetzesname-/Abkürzungs-Regex im Query  (starkes, eindeutiges Signal)
  2. matcher_terms-Votum: je Gesetz die kuratierten Card-matcher_terms; präsente Terme
     stimmen ab, gewichtet per Inverse-Law-Frequency (gesetzes-spezifisch zählt, generisch ~0).

Passagen-Ebene: Schätzung pro Query (= Gold-Segment, bereits passagen-granular); für
ganze Dokumente würden die Segment-Verteilungen aggregiert (Produktionsrealität Signal).

Gemessen gegen die bekannte Latte (Graph-Top-N): top_laws=2 Recall 66 % / Präzision 0,36.

Aufruf:
  python tools/lex_scope.py --gold discovery/klausuren_v2.yaml --save benchmarks/lex-scope-acc.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from risk_layer.idschema import para_level   # noqa: E402

# Gesetzesname / Abkürzung -> Gesetz (starkes Signal; Wortgrenzen)
NAME_SIGNALE = {
    "UStG": [r"umsatzsteuer", r"\bustg\b", r"\bust\b"],
    "EStG": [r"einkommensteuer", r"\bestg\b", r"\best\b", r"lohnsteuer"],
    "KStG": [r"körperschaftsteuer", r"koerperschaftsteuer", r"\bkstg\b", r"\bkst\b"],
    "GewStG": [r"gewerbesteuer", r"\bgewstg\b", r"\bgewst\b"],
    "AO": [r"abgabenordnung", r"\bao\b", r"finanzgericht", r"\bfgo\b", r"einspruch",
           r"festsetzungsverjährung", r"bekanntgabe"],
    "ErbStG": [r"erbschaftsteuer", r"schenkungsteuer", r"\berbstg\b"],
    "GrEStG": [r"grunderwerbsteuer", r"\bgrestg\b"],
    "BewG": [r"bewertungsgesetz", r"\bbewg\b"],
    "UmwStG": [r"umwandlungssteuer", r"\bumwstg\b"],
}
W_NAME = 3.0   # Gewicht eines Gesetzesname-Treffers (relativ zu Term-idf)


def _norm(s): return " ".join((s or "").lower().split())  # noqa: E704
def _law(p): t = p.split(":"); return t[1] if len(t) > 1 else "?"  # noqa: E704


def _lade_gold(p: Path):
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for f in d.get("faelle", []):
        q = str(f.get("query", ""))
        ziele = {para_level(z) for z in (f.get("ziele") or []) if z}
        if _norm(q) and ziele:
            out.append((q, {_law(z) for z in ziele}))
    return out


def _lade_matcher_terms(card_dirs):
    """law -> {term_lower}. Aus den per-Norm-Card-Files (matcher_terms)."""
    law_terms = defaultdict(set)
    for d in card_dirs:
        for f in Path(d).glob("norm_*.yaml"):
            try:
                c = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            law = _law(para_level(c.get("norm_id", "")))
            for t in (c.get("matcher_terms") or []):
                t = str(t).strip().lower()
                if len(t) >= 4:                       # triviale Kurzterme raus
                    law_terms[law].add(t)
    return law_terms


def baue_scoper(card_dirs):
    law_terms = _lade_matcher_terms(card_dirs)
    # Inverse-Law-Frequency je Term
    term_laws = defaultdict(set)
    for law, terms in law_terms.items():
        for t in terms:
            term_laws[t].add(law)
    n_laws = len(law_terms) or 1
    idf = {t: math.log(1 + n_laws / len(ls)) for t, ls in term_laws.items()}
    name_rx = {law: re.compile("|".join(pats)) for law, pats in NAME_SIGNALE.items()}

    def scope(query: str):
        q = _norm(query)
        vote = Counter()
        for law, rx in name_rx.items():
            if rx.search(q):
                vote[law] += W_NAME
        for t, ls in term_laws.items():
            if t in q:                                # Substring-Präsenz (matcher = Presence)
                for law in ls:
                    vote[law] += idf[t]
        return [l for l, _ in vote.most_common()], dict(vote)
    return scope, law_terms


def main() -> int:
    ap = argparse.ArgumentParser(description="Lexikalische Scope-Treffsicherheit")
    ap.add_argument("--gold", default="discovery/klausuren_v2.yaml")
    ap.add_argument("--card-dirs", nargs="+",
                    default=["discovery/api/cards", "discovery/api2/cards"])
    ap.add_argument("--save", default="benchmarks/lex-scope-acc.json")
    args = ap.parse_args()

    gold = _lade_gold(Path(args.gold))
    scope, law_terms = baue_scoper(args.card_dirs)
    print(f"Gold {len(gold)} | matcher_terms je Gesetz: "
          f"{ {l: len(t) for l, t in sorted(law_terms.items())} }", file=sys.stderr)

    rows = [(gl, scope(q)[0]) for q, gl in gold]
    print(f"\n# Lexikalische Scope-Treffsicherheit (n={len(gold)})  — Latte Graph: top2 R 66 % / P 0,36\n")
    res = {}
    for topk in (1, 2, 3):
        voll = sum(1 for gl, rank in rows if gl <= set(rank[:topk]))
        teil = sum(1 for gl, rank in rows if gl & set(rank[:topk]))
        prec = sum(len(gl & set(rank[:topk])) / topk for gl, rank in rows) / len(rows)
        res[f"top{topk}"] = {"voll_recall": round(voll/len(rows), 3),
                             "teil_recall": round(teil/len(rows), 3), "praezision": round(prec, 3)}
        graph = {1: "39 %/0,44", 2: "66 %/0,36", 3: "70 %/0,26"}[topk]
        print(f"  top_laws={topk}: voll-Recall {voll}/{len(rows)} ({voll/len(rows):.0%}) | "
              f"teil-Recall {teil/len(rows):.0%} | Präzision {prec:.2f}   (Graph: {graph})")
    leer = sum(1 for _, rank in rows if not rank)
    print(f"\n  Queries ohne jedes Lexsignal: {leer}/{len(rows)}")
    # Fehlklassifikationen (gold nicht in top-2) — fürs Verständnis
    miss = [(q[:70], sorted(gl), rank[:2]) for (q, gl), (_, rank2) in
            zip(gold, [(g, scope(q)[0]) for q, g in gold]) for rank in [rank2] if not (gl <= set(rank[:2]))]
    print(f"\n# Fälle ohne Gold-Gesetz in lex-top-2 ({len(miss)}):")
    for qt, gl, top2 in miss[:12]:
        print(f"  ziel={gl} lex-top2={top2}  | {qt}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngesichert: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
