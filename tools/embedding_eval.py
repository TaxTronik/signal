"""Ehrliche Schwellwert-Diagnose für den Embedding-Index (Schicht 1.7).

NICHT auf 3 Punkten kalibrieren. Dieses Skript misst zwei VERTEILUNGEN:
  * POSITIVE – Laien-/Fakt-Paraphrasen, die das Embedding fangen SOLL, je mit den
    fachlich akzeptablen Ziel-Normen.
  * NEGATIVE – neutrale UND themennah-aber-irrelevante Sätze, die NICHT feuern sollen.

Ausgegeben wird die Überlappung der best-sim-Verteilungen. Genau diese Überlappung
ist die Antwort auf die Frage, ob ein robuster Schwellwert existiert – und wenn ja,
wo. Liegt der höchste Negativ-Wert über dem niedrigsten korrekten Positiv-Wert,
gibt es KEINEN sauberen Schwellwert; dann ist der nächste Schritt nicht ein
gequälter Threshold, sondern katalog-verankerte Paraphrasen oder Fine-Tuning.

Aufruf:  python tools/embedding_eval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole                           # noqa: E402

from risk_layer.embedding import index_oeffnen, encoder_oeffnen  # noqa: E402

utf8_konsole()

# (Satz, {akzeptable Ziel-Norm-IDs})  – Treffer = eine davon in Top-k.
POSITIVE: list[tuple[str, set[str]]] = [
    ("Die täglichen Bareinnahmen tauchen im Kassenbuch gar nicht auf.", {"norm:AO:146", "norm:AO:146a", "norm:AO:158"}),
    ("Die Kasse wurde über Jahre überhaupt nicht ordentlich geführt.", {"norm:AO:146", "norm:AO:146a", "norm:AO:158"}),
    ("Belege für die Einnahmen fehlen vollständig.", {"norm:AO:146", "norm:AO:147"}),
    ("Das Finanzamt setzt die Umsätze einfach nach eigenem Gutdünken an.", {"norm:AO:162"}),
    ("Weil die Zahlen nicht prüfbar sind, schätzt das Amt die Gewinne.", {"norm:AO:162"}),
    ("Nach Bestandskraft werden dem Amt neue Tatsachen bekannt.", {"norm:AO:173"}),
    ("Erst später kamen Umstände ans Licht, die vorher niemand kannte.", {"norm:AO:173"}),
    ("Eine künstliche Konstruktion allein zur Steuerersparnis.", {"norm:AO:42"}),
    ("Eine substanzlose Auslandsfirma ohne Personal schiebt nur Geld durch.", {"norm:AO:42", "norm:AStG:7", "norm:AStG:8"}),
    ("Der Geschäftsführer soll persönlich für die Steuerschulden einstehen.", {"norm:AO:69", "norm:AO:34"}),
    ("Trotz vorhandener Mittel wurden andere Gläubiger zuerst bezahlt.", {"norm:AO:69", "norm:AO:34"}),
    ("Der Arbeitgeber behielt die Lohnsteuer ein, führte sie aber nicht ab.", {"norm:EStG:42d", "norm:EStG:41a", "norm:EStG:38"}),
    ("Wem das Wirtschaftsgut tatsächlich wirtschaftlich gehört.", {"norm:AO:39"}),
    ("Im Bescheid steckt ein simpler Schreib- oder Rechenfehler.", {"norm:AO:129"}),
    ("Mutter- und Tochtergesellschaft sind finanziell, wirtschaftlich und organisatorisch verbunden.", {"norm:KStG:14", "norm:UStG:2"}),
]

# Negative: sollen NICHT feuern. Mischung aus neutral und themennah-irrelevant.
NEGATIVE: list[str] = [
    "Das Wetter ist heute sonnig und warm.",
    "Wir treffen uns am Freitag zum Mittagessen.",
    "Der Hund läuft schnell über die Wiese.",
    "Sie hat gestern einen guten Film im Kino gesehen.",
    "Der Zug hatte heute zwanzig Minuten Verspätung.",
    "Das neue Smartphone hat eine bessere Kamera.",
    # themennah, aber ohne konkreten Subsumtionsauslöser:
    "Steuern sind insgesamt ein wichtiges gesellschaftliches Thema.",
    "Das Unternehmen hat im letzten Jahr ordentlich verdient.",
    "Unsere Buchhaltung erledigt seit Jahren ein externes Büro zuverlässig.",
    "Der Steuerberater hat ein freundliches und kompetentes Team.",
    "Die Firma plant nächstes Jahr eine neue Filiale zu eröffnen.",
    "Das Finanzamt hat moderne, gut erreichbare Büroräume.",
    "Die Mitarbeiter bekommen pünktlich ihr Gehalt überwiesen.",
    "Im Jahresbericht stehen viele Diagramme und Tabellen.",
    "Die Steuererklärung wurde fristgerecht und vollständig eingereicht.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Schwellwert-Diagnose für den Embedding-Index")
    # k = ausgelieferte Vorschlagstiefe (System: Top-5). Über --topk überschreibbar.
    ap.add_argument("--topk", type=int, default=5,
                    help="Vorschlagstiefe k für die Positiv-Treffer (Default 5)")
    args = ap.parse_args()
    topk = args.topk

    idx = index_oeffnen()
    if idx is None:
        print("Kein Embedding-Index. Erst: python -m risk_layer.cli embedding build", file=sys.stderr)
        return 1
    enc = encoder_oeffnen(idx.encoder_name if idx.encoder_name != "hash512" else "hash")

    pos_korrekt: list[float] = []   # best-sim, wenn Zielnorm in Top-k
    pos_verfehlt: list[tuple[str, float]] = []
    for satz, ziele in POSITIVE:
        treffer = idx.suche(satz, enc, k=topk, min_sim=0.0)
        ids = [t["norm_id"] for t in treffer]
        best = treffer[0]["sim"] if treffer else 0.0
        if set(ids) & ziele:
            pos_korrekt.append(best)
        else:
            pos_verfehlt.append((satz[:40], best))

    neg_best: list[float] = []
    for s in NEGATIVE:
        treffer = idx.suche(s, enc, k=1, min_sim=0.0)
        if not treffer:                 # leerer Index → kein [0]-Zugriff (IndexError)
            print("Embedding-Index liefert keine Treffer (leer/defekt) — "
                  "erst: python -m risk_layer.cli embedding build", file=sys.stderr)
            return 2
        neg_best.append(treffer[0]["sim"])

    pos_korrekt.sort()
    neg_best.sort(reverse=True)
    print(f"POSITIVE: {len(pos_korrekt)}/{len(POSITIVE)} mit Zielnorm in Top-{topk}")
    print(f"  korrekte best-sim (sortiert): {[round(x,3) for x in pos_korrekt]}")
    if pos_verfehlt:
        print(f"  VERFEHLT ({len(pos_verfehlt)}):")
        for s, b in pos_verfehlt:
            print(f"     sim={b:.3f}  {s}")
    print(f"NEGATIVE best-sim (höchste zuerst): {[round(x,3) for x in neg_best]}")

    if pos_korrekt:
        tiefster_pos = pos_korrekt[0]
        hoechster_neg = neg_best[0]
        print(f"\nÜberlappung: tiefster korrekter Positiv = {tiefster_pos:.3f}"
              f" | höchster Negativ = {hoechster_neg:.3f}")
        if hoechster_neg < tiefster_pos:
            mitte = (hoechster_neg + tiefster_pos) / 2
            print(f"  → SAUBERE TRENNUNG. Robuster Schwellwert ≈ {mitte:.3f} "
                  f"(Marge {tiefster_pos - hoechster_neg:.3f}).")
        else:
            # Wie viele Positive lägen über dem höchsten Negativ?
            ueber = sum(1 for x in pos_korrekt if x > hoechster_neg)
            print(f"  → KEINE saubere Trennung. Bei Schwelle knapp über {hoechster_neg:.3f} "
                  f"blieben nur {ueber}/{len(pos_korrekt)} korrekte Positive übrig.")
            print("    Konsequenz: kein robuster Schwellwert. Nächster Schritt: "
                  "katalog-verankerte Paraphrasen oder Fine-Tuning – NICHT Threshold quälen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
