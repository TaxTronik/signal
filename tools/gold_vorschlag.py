"""Gold-Kandidaten-Generator — erzeugt MENSCHLICH ZU PRÜFENDE Vorschläge.

KEIN bindendes Gold! Dieses Skript läuft satzweise über Beispieltexte und
sammelt je Satz die Embedding-Top-5 plus die Trigger-Index-Kandidaten als
ENTWURF einer Gold-Datei. Die ``ziele``-Felder bleiben LEER — sie füllt die
fachliche Leitung beim Review aus. Erst eine geprüfte, eingefrorene Kopie
(ohne den UNGEPRÜFT-Status) darf in Evals einfließen; die Vorschlags-Dateien
selbst liegen bewusst in fixtures/gold/vorschlaege/ (Unterordner — werden von
gold_gate.py und dem Eval-Harness-Default NICHT mitgelesen).

Projekt-Leitplanke: die Kandidaten sind KORPUS-fundiert (Embedding über
Gesetzestexte + deterministischer Trigger-Index), kein Modellwissen als Norm.
Achtung Skala: ``sim`` ist bei quelle=embedding die Kosinus-Ähnlichkeit
(0..1), bei quelle=trigger der TF-IDF-Score des Trigger-Index (offen nach
oben) — NICHT vergleichbar, nur je Quelle interpretierbar.

Aufruf:  python tools/gold_vorschlag.py --corpus C:/dev/taxtronikexctract/corpus
Exit-Codes: 0 = ok, 2 = Fehler.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole                            # noqa: E402

from risk_layer.embedding import EmbeddingIndex, encoder_oeffnen  # noqa: E402
from risk_layer.muster import saetze_mit_offset                   # noqa: E402
from risk_layer.persistenz import atomar_schreiben                # noqa: E402
from risk_layer.sicherung import zeitstempel_dateiname            # noqa: E402
from risk_layer.triggerindex import TriggerIndex                  # noqa: E402

STATUS = ("UNGEPRÜFT — Entwurf, von der Leitung zu reviewen; "
          "NICHT in Evals einbeziehen")
DEFAULT_OUT = Path("fixtures") / "gold" / "vorschlaege"


def texte_sammeln(pfade: list[str]) -> list[Path]:
    """Eingaben (Datei ODER Verzeichnis) zu einer sortierten .txt-Liste auflösen."""
    dateien: list[Path] = []
    for roh in pfade:
        p = Path(roh)
        if p.is_dir():
            dateien.extend(sorted(p.glob("*.txt")))
        elif p.is_file():
            dateien.append(p)
        else:
            raise FileNotFoundError(f"Text-Quelle nicht gefunden: {p}")
    if not dateien:
        raise FileNotFoundError(f"Keine .txt-Dateien unter: {', '.join(pfade)}")
    return dateien


def kandidaten_je_satz(satz: str, idx: EmbeddingIndex, enc,
                       trigger: TriggerIndex | None, *, topk: int,
                       trigger_topk: int) -> list[dict]:
    """Embedding-Top-k + Trigger-Kandidaten EINES Satzes, review-fertig sortiert."""
    out: list[dict] = []
    for rang, t in enumerate(idx.suche(satz, enc, k=topk, min_sim=0.0), start=1):
        out.append({"norm_id": t["norm_id"], "zitat": t["zitat"],
                    "titel": t["titel"], "sim": t["sim"], "rang": rang,
                    "quelle": "embedding"})
    if trigger is not None:
        for rang, t in enumerate(trigger.kandidaten(satz, k=trigger_topk), start=1):
            out.append({"norm_id": t["norm_id"], "zitat": t["zitat"],
                        "titel": t["titel"], "sim": t["score"], "rang": rang,
                        "quelle": "trigger"})
    return out


def main(argv: list[str] | None = None) -> int:
    utf8_konsole()
    ap = argparse.ArgumentParser(
        description="Erzeugt UNGEPRÜFTE Gold-Kandidaten (Entwurf für das Review)")
    ap.add_argument("--texte", nargs="+", default=["examples"],
                    help="Textdatei(en) oder Verzeichnis(se) mit *.txt (Default: examples/)")
    ap.add_argument("--corpus", default="corpus",
                    help="Korpus-Verzeichnis (embedding/ + trigger.sqlite)")
    ap.add_argument("--device", default=None, help="cuda|cpu (Default: Encoder-Wahl)")
    ap.add_argument("--topk", type=int, default=5,
                    help="Embedding-Kandidaten je Satz (Default 5)")
    ap.add_argument("--trigger-topk", type=int, default=5,
                    help="Trigger-Kandidaten je Satz (Default 5)")
    ap.add_argument("--min-laenge", type=int, default=15,
                    help="Sätze kürzer als N Zeichen überspringen (Default 15)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help=f"Zielverzeichnis (Default {DEFAULT_OUT})")
    args = ap.parse_args(argv)

    try:
        dateien = texte_sammeln(args.texte)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    corpus = Path(args.corpus)
    idx = EmbeddingIndex.load(corpus / "embedding")
    if idx is None:
        print(f"Kein Embedding-Index unter {corpus / 'embedding'}.", file=sys.stderr)
        return 2
    try:
        enc = encoder_oeffnen(idx.encoder_name, device=args.device)
    except (ImportError, ValueError) as exc:
        print(f"Encoder-Fehler: {exc}", file=sys.stderr)
        return 2

    # Trigger-Index ist optional-degradierend: ohne ihn nur Embedding-Kandidaten
    # (laut gemeldet, nicht still verschluckt).
    trigger: TriggerIndex | None = None
    try:
        trigger = TriggerIndex(corpus / "trigger.sqlite")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Hinweis: kein Trigger-Index ({exc}) — nur Embedding-Kandidaten.",
              file=sys.stderr)

    saetze_aus: list[dict] = []
    for datei in dateien:
        text = datei.read_text(encoding="utf-8")
        for s in saetze_mit_offset(text):
            satz = s.text.strip()
            if len(satz) < args.min_laenge:
                continue
            saetze_aus.append({
                "datei": str(datei).replace("\\", "/"),
                "query": satz,
                "kandidaten": kandidaten_je_satz(
                    satz, idx, enc, trigger,
                    topk=args.topk, trigger_topk=args.trigger_topk),
                # LEER — füllt das menschliche Review aus (das ist der Punkt).
                "ziele": [],
            })

    if not saetze_aus:
        print("Keine auswertbaren Sätze gefunden.", file=sys.stderr)
        return 2

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc = {
        "status": STATUS,
        "erstellt_am": ts,
        "encoder": enc.name,
        "corpus": str(corpus),
        "quellen": [str(d).replace("\\", "/") for d in dateien],
        "n_saetze": len(saetze_aus),
        "saetze": saetze_aus,
    }
    kopf = (
        f"# {STATUS}\n"
        "# Automatisch erzeugte Gold-KANDIDATEN (tools/gold_vorschlag.py).\n"
        "# Review-Anleitung: je Satz die fachlich korrekten Ziel-Normen in\n"
        "# 'ziele: []' eintragen (Kandidaten sind nur Vorschläge — auch andere\n"
        "# Normen sind zulässig), Status-Zeilen entfernen und als eingefrorenes\n"
        "# Gold nach fixtures/gold/ verschieben. 'sim' ist je Quelle skaliert:\n"
        "# embedding = Kosinus (0..1), trigger = TF-IDF-Score (offene Skala).\n")
    ziel = Path(args.out_dir) / f"embedding_gold_vorschlag_{zeitstempel_dateiname(ts)}.yaml"
    atomar_schreiben(ziel, kopf + yaml.safe_dump(
        doc, allow_unicode=True, sort_keys=False, width=100))
    print(f"{len(saetze_aus)} Satz-Vorschläge aus {len(dateien)} Datei(en) → {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
