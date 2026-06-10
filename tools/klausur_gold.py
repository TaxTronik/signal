"""Gold v2 aus StB-Examens-Musterklausuren (BLINDES Test-Set).

Disziplin (vgl. docs/adr/ENTSCHEIDUNG_Card-Retrieval.md, [[card-discovery-pass2]]):
  Query   = der Sachverhalt / die Aufgabe (NIE die Musterlösung) — unabhängig vom
            Card-Design, also blind gegenüber unseren Ankern.
  Ziele   = die §§, die die MUSTERLÖSUNG anführt — autoritatives Label, nicht
            unser Urteil. Nur Korpus-Normen zählen (der Rest wird ausgewiesen).
  Status  = KANDIDAT. Dieses Skript finalisiert NICHTS: primär-vs-inzidentell und
            die Negativfälle bestätigt der Berater (faelle[].review).

Quellformat (StBK Niedersachsen):
  niedersachsen/  je Klausur eine SV-Datei + separate Lösungsdatei; Aufgaben
                  nummeriert ("N." / "N)" / "Aufgabe N" / "Sachverhalt N") in beiden
                  -> Segmentierung wählt die Achse mit max. SV↔Lösung-Überlappung.

Aufruf:
  python tools/klausur_gold.py --txt-dir discovery/klausuren_txt \
      --out discovery/klausuren_v2_kandidat.yaml --report discovery/klausuren_gold_review.md
"""

from __future__ import annotations

import argparse
import re
import sys
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

LAWS = ("AO|FGO|EStG|EStDV|KStG|KStDV|GewStG|GewStDV|UStG|UStDV|AStG|UmwStG|UmwG|"
        "BewG|ErbStG|ErbStDV|GrEStG|GrStG|SolzG|InvStG|HGB|AktG|GmbHG|StBerG|BGB|LStDV")
_ZIT = re.compile(rf"§§?\s*(\d+[a-z]?)[^§]{{0,45}}?\b({LAWS})\b")
_PUNKTE = re.compile(r"\(\s*\d+(?:[.,]\d+)?\s*\)")          # Punkte-Annotation (0,5)/(1)
_NEG = re.compile(r"nicht steuerbar|nicht steuerpflichtig|keine Änderung|nicht "
                  r"änderbar|unzulässig|nicht möglich|keine Steuer|steuerfrei", re.I)
STEUERARTEN = {
    "Abgabenordnung": "AO", "Finanzgericht": "FGO", "Umsatzsteuer": "UStG",
    "Einkommensteuer": "EStG", "Körperschaftsteuer": "KStG", "Gewerbesteuer": "GewStG",
    "Erbschaft": "ErbStG", "Schenkungsteuer": "ErbStG", "Grunderwerb": "GrEStG",
}


def _clean(t: str) -> str:
    t = _PUNKTE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _read(p: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="replace")


def _steuerart(text: str) -> str:
    for k, v in STEUERARTEN.items():
        if k.lower() in text.lower():
            return v
    return "?"


def _seg_numbered(text: str) -> dict:
    """Führend nummeriert: '1.', '7)', '1.a)' — NICHT Daten '13.06' (Lookahead)."""
    secs: dict = {}
    cur = None
    for ln in text.splitlines():
        m = re.match(r"^\s*([1-9]\d?)[.)](?=\s|[a-z]\)|$)", ln)
        if m:
            cur = int(m.group(1))
            secs.setdefault(cur, [])
        if cur is not None:
            secs[cur].append(ln)
    return {k: "\n".join(v) for k, v in secs.items()}


def _seg_marker(stichwort: str):
    def f(text: str) -> dict:
        out: dict = {}
        for p in re.split(rf"(?={stichwort}\s+\d+)", text):
            m = re.match(rf"{stichwort}\s+(\d+)", p)
            if m:
                out[int(m.group(1))] = p
        return out
    return f


STRATEGIEN = [_seg_numbered, _seg_marker("Aufgabe"), _seg_marker("Sachverhalt")]


def align(sv: str, loes: str) -> list[tuple]:
    """Wähle die Segmentierung, die SV↔Lösung am besten alignt (max. gemeinsame
    Schlüssel). <2 gemeinsame -> grob (ein Block), Berater segmentiert von Hand."""
    best: tuple | None = None
    for strat in STRATEGIEN:
        a, b = strat(sv), strat(loes)
        shared = sorted(set(a) & set(b))
        if best is None or len(shared) > len(best[0]):
            best = (shared, a, b)
    if best and len(best[0]) >= 2:
        shared, a, b = best
        return [(k, a[k], b[k]) for k in shared]
    return [(0, sv, loes)]


def ziele_aus(loesung: str, ist_korpus, primary: str | None = None):
    """Explizite Zitate ('§ N … LAW', jedes Gesetz) PLUS — in einsteuerlichen
    Tabellen-Lösungen — bare '§ N' der Primär-Steuerart der Einheit. Cross-Refs
    stehen in Examensklausuren explizit MIT Gesetz, das eigene Gesetz bare; die
    Zuordnung ist daher robust. Implizite werden separat ausgewiesen (Review).

    Korpus-Ziele werden nach ZENTRALITÄT gerankt (Häufigkeit in der Lösung, dann
    Erst-Position) — der Primär-Vorschlag steht so vorn, der Berater trimmt den
    inzidentellen Schwanz (Anwendungs-/Anrechnungs-§§)."""
    t = _clean(loesung)
    stat: dict[str, list] = {}                 # norm -> [count, first_pos]
    expl: set[str] = set()

    def add(norm: str, pos: int) -> None:
        if norm not in stat:
            stat[norm] = [0, pos]
        stat[norm][0] += 1
        stat[norm][1] = min(stat[norm][1], pos)

    for m in _ZIT.finditer(t):
        norm = f"norm:{m.group(2)}:{m.group(1)}"
        add(norm, m.start())
        expl.add(norm)
    if primary and primary != "?":
        rest = _ZIT.sub(lambda mm: " " * len(mm.group(0)), t)   # Positionen erhalten
        for m in re.finditer(r"§§?\s*(\d+[a-z]?)", rest):
            add(f"norm:{primary}:{m.group(1)}", m.start())

    korpus = sorted((n for n in stat if ist_korpus(n)),
                    key=lambda n: (-stat[n][0], stat[n][1]))     # zentral zuerst
    ausser = sorted(n for n in stat if not ist_korpus(n))
    implizit = sorted(n for n in korpus if n not in expl)        # nur impliz. hergeleitet
    return korpus, ausser, implizit


VORSCHLAG_N = 3            # so viele zentralste §§ werden als Primär vorgeschlagen


def klasse_von(vorschlag: list[str], loes: str) -> str:
    if not vorschlag:
        return "negativ?" if _NEG.search(loes) else "leer"
    laws = {z.split(":")[1] for z in vorschlag}
    if len(vorschlag) == 1:
        return "single"
    return "cross_law" if len(laws) > 1 else "intra_law"


def faelle_einer_einheit(sv: str, loes: str, quelle: str, steuerart: str, ist_korpus):
    segmente = align(sv, loes)
    grob = len(segmente) == 1 and segmente[0][0] == 0
    out = []
    for k, sv_txt, lo_txt in segmente:
        q = _clean(sv_txt)
        if len(q) < 25:
            continue
        kor, ausser, impl = ziele_aus(lo_txt, ist_korpus, steuerart)
        vorschlag = kor[:VORSCHLAG_N]                 # zentralste -> Primär-Vorschlag
        out.append({
            "query": q,
            "ziele": vorschlag,                       # Berater bestätigt/trimmt DIES
            "weitere": kor[VORSCHLAG_N:],             # Inzidentell-Verdacht (ggf. hochstufen)
            "implizit": [z for z in impl if z in vorschlag],  # impliz. attribuiert (prüfen)
            "ausserhalb_korpus": ausser,
            "klasse": klasse_von(vorschlag, _clean(lo_txt)),
            "quelle": f"{quelle}#{k}" if k else quelle,
            "steuerart": steuerart,
            "mehrteilig": len(kor) > 5,              # vermutl. Mehr-Issue-Aufgabe -> splitten
            "grob": grob,
            "review": "offen",          # Berater: primär-ziele bestätigen / negativ klären
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Gold v2 aus StB-Musterklausuren (blind)")
    ap.add_argument("--txt-dir", default="discovery/klausuren_txt")
    ap.add_argument("--out", default="discovery/klausuren_v2_kandidat.yaml")
    ap.add_argument("--report", default="discovery/klausuren_gold_review.md")
    args = ap.parse_args()

    graph = graph_oeffnen()
    if graph is None:
        print("Kein Normgraph — erst 'graph build'.", file=sys.stderr)
        return 2
    ist_korpus = lambda nid: bool(graph.aufloesen(nid).get("gefunden"))  # noqa: E731

    txt = Path(args.txt_dir)
    faelle: list[dict] = []

    # --- Niedersachsen: SV-Datei + separate Lösungsdatei ---
    nds = txt / "niedersachsen"
    for kl in sorted(nds.glob("*Klausur*.txt")):
        lo = None
        for cand in (kl.name.replace("Klausur", "Lösungen"),
                     kl.name.replace("Klausur", "Lösung")):
            if (nds / cand).exists():
                lo = nds / cand
                break
        if lo is None:
            print(f"  ! keine Lösung zu {kl.name}", file=sys.stderr)
            continue
        name = kl.name.replace("2024_", "").replace(" Klausur Rechtstand 2023", "")
        faelle += faelle_einer_einheit(_read(kl), _read(lo), f"nds:{name}",
                                       _steuerart(kl.name), ist_korpus)

    # --- Card-Bestand für Coverage-Flag ---
    pairs = yaml.safe_load(_read(Path("discovery/api/cards_training_pairs.yaml"))) or {}
    bestand = {para_level(k.get("quell_norm", "")) for k in pairs.get("kandidaten", [])
               if k.get("quell_norm")}
    for fa in faelle:
        fa["abgedeckt"] = sorted(z for z in fa["ziele"] if z in bestand)
        fa["luecke"] = sorted(z for z in fa["ziele"] if z not in bestand)

    # --- Ausgabe ---
    from collections import Counter
    kl_zähl = Counter(f["klasse"] for f in faelle)
    art_zähl = Counter(f["steuerart"] for f in faelle)
    messbar = [f for f in faelle if f["ziele"]]
    kopf = {
        "gold_version": "v2-kandidat",
        "rolle": "test",
        "status": "UNREVIEWED — Berater bestätigt primär-ziele + negativ",
        "herkunft": "StB-Examens-Musterklausuren (StBK Niedersachsen)",
        "erzeugt": str(date.today()),
        "disziplin": "Query=Sachverhalt (blind), Ziele=Musterlösungs-§§ (korpus-gefiltert). "
                     "NICHT auf diesem Set tunen.",
        "n_faelle": len(faelle),
        "n_messbar": len(messbar),
        "klassen": dict(kl_zähl),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        yaml.safe_dump({**kopf, "faelle": faelle}, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8")

    # Review-Report (gruppiert, fürs schnelle Trimmen optimiert)
    kurz = lambda L: ", ".join(z.replace("norm:", "") for z in L)  # noqa: E731
    zeilen = [f"# Gold v2 — Kandidaten-Review ({len(faelle)} Fälle, {len(messbar)} messbar)\n",
              f"Klassen: {dict(kl_zähl)}  ·  Steuerart: {dict(art_zähl)}\n",
              "Berater je Fall: **Vorschlag** = Primär bestätigen/trimmen · **weitere** = "
              "ggf. hochstufen · `impl` = implizit (Tabelle, prüfen) · `mehrteilig` = splitten "
              "· `negativ?` = ist es ein echter Nicht-Anker-Fall?\n"]
    quelle = None
    for fa in faelle:
        if fa["quelle"].split("#")[0] != quelle:
            quelle = fa["quelle"].split("#")[0]
            zeilen.append(f"\n## {quelle}")
        flags = " ".join(t for t, on in (("⚠mehrteilig", fa.get("mehrteilig")),
                                         ("⚠grob", fa.get("grob"))) if on)
        zeilen.append(
            f"\n- **[{fa['klasse']}] {fa['quelle'].split('#')[-1]}** {flags}\n"
            f"  Q: {fa['query'][:160]}\n"
            f"  Vorschlag: {kurz(fa['ziele']) or '— (negativ?)'}"
            + (f"  ·  impl: {kurz(fa['implizit'])}" if fa['implizit'] else "")
            + (f"\n  weitere: {kurz(fa['weitere'])}" if fa['weitere'] else "")
            + (f"\n  LÜCKE (kein Card): {kurz(fa['luecke'])}" if fa['luecke'] else ""))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(zeilen), encoding="utf-8")

    print(f"\n# Gold v2 Kandidaten: {len(faelle)} Fälle ({len(messbar)} messbar)")
    print(f"  Klassen:   {dict(kl_zähl)}")
    print(f"  Steuerart: {dict(art_zähl)}")
    luecke_n = len({z for f in faelle for z in f["luecke"]})
    abged_n = len({z for f in faelle for z in f["abgedeckt"]})
    print(f"  Ziel-Normen: {abged_n} im Card-Bestand, {luecke_n} Lücke")
    print(f"  -> {args.out}\n  -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
