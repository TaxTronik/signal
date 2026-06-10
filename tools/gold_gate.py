"""Gold-Gate: deterministisches Regressionsgate für die CI (PS-861 L3).

WARUM: Das IDW-PS-861-Mapping (docs/compliance/IDW-PS-861-Mapping.md, Lücke L3)
verlangt ein FESTGELEGTES, in der CI verankertes Qualitätsmaß statt nur
gelegentlicher Offline-Messungen. Dieses Gate führt den DETERMINISTISCHEN Teil
der Pipeline (Schicht 1 Matcher, Schicht 1b Muster, Streit-Detektor,
Zitat-Parser) gegen eingefrorene Gold-Erwartungen aus und bricht bei
Regression unter der Schwelle mit Exit 1 ab — artefakt-frei (nur Katalog +
examples/), ohne Embedding/LLM/GPU und ohne Korpus-Build, damit es im
normalen Test-Job der CI laufen kann.

ABGRENZUNG (Faktenlage, gemessen am 2026-06-10): Die bestehenden Gold-Sets
fixtures/gold/ao_v1.yaml und organschaft_v1.yaml sind Embedding-Retrieval-Gold
(Laien-Paraphrasen → Ziel-Normen). Ihr deterministisch prüfbarer Anteil ist
NULL — 0 von 25 Queries erzeugen eine deterministische Marke (die Queries sind
bewusst blind gegenüber Katalog-Wortlaut). Sie werden hier deshalb erkannt und
mit Hinweis ÜBERSPRUNGEN, nicht gemessen. Gemessen wird das eigene
deterministische Gold (fixtures/gold/deterministisch_v1.yaml): examples/*.txt
gegen erwartete Katalog-Marken, Zitat-IDs und Streit-Signale.

Schicht 1.5 (Trigger-Index) ist zwar deterministisch, braucht aber das
Korpus-Artefakt corpus/trigger.sqlite — die CI-Suite ist artefakt-frei, also
läuft das Gate bewusst ohne Detektor (detektor=False, wie der degradierte
Betrieb ohne Index).

Aufruf:
  python tools/gold_gate.py                      # Exit 0 ok / 1 Regression / 2 Fehler
  python tools/gold_gate.py --bericht out/berichte   # + datierter JSON-Bericht (L9-Baustein)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole                      # noqa: E402

from risk_layer.catalog import lade_katalog                 # noqa: E402
from risk_layer.idschema import para_level                  # noqa: E402
from risk_layer.persistenz import atomar_schreiben          # noqa: E402
from risk_layer.pipeline import AnalyseOptionen, Pipeline   # noqa: E402
from risk_layer.sicherung import zeitstempel_dateiname      # noqa: E402
from risk_layer.zitate import finde_normzitate              # noqa: E402

GOLD_DIR = Path("fixtures/gold")

# Schwelle der Trefferquote (gefundene erwartete Marken / alle erwarteten
# Marken). GEMESSENER Ist-Stand am 2026-06-10 (Katalog 0.3.0): 36/36 = 1.0 —
# der Kern ist deterministisch, also ist JEDE verlorene Marke eine echte
# Regression. Die Schwelle steht deshalb auf dem Ist-Stand (bindend ab sofort,
# nicht aspirational). Wer den Katalog bewusst ändert, schreibt das Gold mit
# fort — eine abgesenkte Schwelle wäre der falsche Weg (sie versteckte künftige
# Regressionen hinter der gewollten Änderung).
SCHWELLE = 1.0


def ist_deterministisches_gold(daten: dict) -> bool:
    """Erkennt das hier messbare Gold-Format (Fälle mit datei + erwartete_*).

    Retrieval-Gold (query/ziele, modellgebunden) fällt durch und wird vom
    Aufrufer mit Hinweis übersprungen statt falsch gemessen.
    """
    faelle = daten.get("faelle") or []
    return bool(faelle) and all(
        isinstance(f, dict) and "datei" in f and "erwartete_katalog" in f
        for f in faelle)


def pruefe_fall(pipe: Pipeline, fall: dict) -> dict:
    """Führt EINEN Gold-Fall deterministisch aus und vergleicht die Marken.

    Liefert je Erwartungsart (katalog/zitate/streit) fehlende und — für die
    Katalog-Marken — zusätzliche IDs; 'praezision: true' macht Zusätzliche zum
    Gate-Bruch (Negativ-Kontrolle).
    """
    text = Path(fall["datei"]).read_text(encoding="utf-8")
    # Fester Zeitstempel + nur deterministische Schichten: gleicher Stand →
    # gleiches Ergebnis (keine Uhr, kein Modell, kein Index).
    rep = pipe.run(text, optionen=AnalyseOptionen(
        erzeugt_am="gold-gate", detektor=False, streit=True))

    gef_katalog = {k.eintrag.id for k in rep.karten}
    gef_zitate = {para_level(a.norm_id.to_id())
                  for a in finde_normzitate(text) if a.norm_id}
    gef_streit = {r.streit_signal for r in rep.risiken if r.quelle == "streit"}

    erw_katalog = set(fall.get("erwartete_katalog") or ())
    erw_zitate = set(fall.get("erwartete_zitate") or ())
    erw_streit = set(fall.get("erwartete_streit_signale") or ())

    fehlend = (sorted(erw_katalog - gef_katalog)
               + sorted(erw_zitate - gef_zitate)
               + sorted(erw_streit - gef_streit))
    zusaetzlich = sorted(gef_katalog - erw_katalog)
    return {
        "datei": fall["datei"],
        "erwartet": len(erw_katalog) + len(erw_zitate) + len(erw_streit),
        "gefunden": (len(erw_katalog & gef_katalog) + len(erw_zitate & gef_zitate)
                     + len(erw_streit & gef_streit)),
        "fehlend": fehlend,
        "zusaetzlich": zusaetzlich,
        # Präzisionsverstoß nur, wo das Gold es ausdrücklich verlangt —
        # andernorts ist Katalog-Wachstum legitim und wird nur berichtet.
        "praezisions_verstoss": bool(fall.get("praezision")) and bool(zusaetzlich),
    }


def main(argv: list[str] | None = None) -> int:
    utf8_konsole()
    ap = argparse.ArgumentParser(
        description="Deterministisches Gold-Gate (Regressionsschranke für die CI)")
    ap.add_argument("--gold-dir", default=str(GOLD_DIR),
                    help="Verzeichnis der Gold-YAMLs (Default: fixtures/gold)")
    ap.add_argument("--bericht",
                    help="Verzeichnis für einen datierten JSON-Bericht "
                         "(gold_gate_<zeitstempel>.json; Empfehlung: out/berichte/)")
    ap.add_argument("--zeitstempel",
                    help="Fester ISO-Zeitstempel für den Bericht (reproduzierbare Läufe)")
    args = ap.parse_args(argv)

    gold_dir = Path(args.gold_dir)
    dateien = sorted(gold_dir.glob("*.yaml"))
    if not dateien:
        print(f"Keine Gold-Dateien unter {gold_dir}.", file=sys.stderr)
        return 2

    pipe = Pipeline(lade_katalog())
    ergebnisse: list[dict] = []
    uebersprungen: list[str] = []
    for pfad in dateien:
        daten = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
        if not ist_deterministisches_gold(daten):
            # ehrlich benennen statt still 0 messen (s. Modul-Docstring).
            uebersprungen.append(pfad.name)
            continue
        for fall in daten["faelle"]:
            ergebnisse.append(pruefe_fall(pipe, fall))

    if not ergebnisse:
        print("Kein deterministisch messbares Gold gefunden — Gate ohne Substanz.",
              file=sys.stderr)
        return 2

    erwartet = sum(e["erwartet"] for e in ergebnisse)
    gefunden = sum(e["gefunden"] for e in ergebnisse)
    quote = (gefunden / erwartet) if erwartet else 1.0
    verstoesse = [e for e in ergebnisse if e["praezisions_verstoss"]]

    print("Gold-Gate (deterministische Schichten: Matcher/Muster/Streit/Zitate)")
    for name in uebersprungen:
        print(f"  übersprungen: {name} (Retrieval-Gold, kein deterministischer Anteil)")
    for e in ergebnisse:
        status = "ok " if not e["fehlend"] and not e["praezisions_verstoss"] else "REG"
        print(f"  [{status}] {e['datei']}: {e['gefunden']}/{e['erwartet']} erwartete Marken")
        if e["fehlend"]:
            print(f"        fehlend   : {', '.join(e['fehlend'])}")
        if e["zusaetzlich"]:
            tag = "REGRESSION (Negativ-Kontrolle)" if e["praezisions_verstoss"] else "Hinweis"
            print(f"        zusätzlich ({tag}): {', '.join(e['zusaetzlich'])}")
    print(f"  Trefferquote: {gefunden}/{erwartet} = {quote:.3f}  (Schwelle {SCHWELLE:.3f})")

    bestanden = quote >= SCHWELLE and not verstoesse

    if args.bericht:
        # L9-Baustein: Kennzahlen datiert archivieren (gleiches Muster wie
        # 'risk-layer radar --archiv'); Uhr nur hier am Rand.
        ts = args.zeitstempel or datetime.now().isoformat(timespec="seconds")
        bericht_pfad = Path(args.bericht) / f"gold_gate_{zeitstempel_dateiname(ts)}.json"
        atomar_schreiben(bericht_pfad, json.dumps({
            "erstellt_am": ts,
            "schwelle": SCHWELLE,
            "trefferquote": round(quote, 6),
            "erwartet": erwartet,
            "gefunden": gefunden,
            "bestanden": bestanden,
            "uebersprungen": uebersprungen,
            "faelle": ergebnisse,
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        print(f"  Bericht: {bericht_pfad}", file=sys.stderr)

    if not bestanden:
        print("GOLD-GATE VERLETZT: Regression im deterministischen Kern.",
              file=sys.stderr)
        return 1
    print("Gold-Gate bestanden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
