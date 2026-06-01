"""Hub-Diagnose für den Embedding-Index – Ursache statt Symptom.

Ein "Hub" ist ein Dokumentvektor, der zu VIELEN unterschiedlichen Queries
mittelhoch ähnlich ist. Das ist die definierende Eigenschaft – unabhängig davon,
ob der Paragraph einen Titel hat. Dieses Skript misst sie direkt:

  hubness(doc) = mittlere Kosinus-Ähnlichkeit des Dok-Vektors über ein breites,
                 gemischtes Query-Sample (legal + neutral).

Ausgegeben werden die Top-Hubs mit ihren Merkmalen (Titel? Textlänge?), damit
sichtbar wird, ob "titellos" und "hubness" deckungsgleich sind oder ob es
betitelte Hubs / valide titellose Normen gibt. Daraus leitet sich die richtige,
GENERALISIERBARE Filterregel ab – nicht "diese drei HGB-Paragraphen", sondern
"Knoten unterhalb sinnvoller Informationsdichte gehören nicht in den Index".

Aufruf:  python tools/hub_diagnose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_layer.embedding import index_oeffnen, encoder_oeffnen  # noqa: E402

# Breites, gemischtes Query-Sample (legal + neutral). Hubs matchen alle mittelhoch.
QUERIES = [
    "Die Bareinnahmen tauchen im Kassenbuch nicht auf.",
    "Das Amt schätzt die Umsätze nach eigenem Ermessen.",
    "Nach Bestandskraft werden neue Tatsachen bekannt.",
    "Eine substanzlose Auslandsfirma ohne Personal.",
    "Der Geschäftsführer haftet persönlich für die Steuern.",
    "Die Lohnsteuer wurde einbehalten, aber nicht abgeführt.",
    "Ein Schreibfehler im Steuerbescheid.",
    "Wem das Wirtschaftsgut wirtschaftlich gehört.",
    "Die Organgesellschaft ist finanziell eingegliedert.",
    "Das Wetter ist heute sonnig und warm.",
    "Wir treffen uns am Freitag zum Essen.",
    "Das Unternehmen plant eine neue Filiale.",
    "Die Mitarbeiter bekommen pünktlich Gehalt.",
    "Ein freundliches und kompetentes Beraterteam.",
]


def main() -> int:
    idx = index_oeffnen()
    if idx is None:
        print("Kein Index. Erst: python -m risk_layer.cli embedding build", file=sys.stderr)
        return 1
    enc = encoder_oeffnen(idx.encoder_name if idx.encoder_name != "hash512" else "hash")
    Q = enc.encode(QUERIES)                      # (q, d)
    S = idx.vectors @ Q.T                        # (n, q) Kosinus
    hubness = S.mean(axis=1)                     # mittlere Ähnlichkeit je Dok
    order = np.argsort(-hubness)

    n = len(idx.ids)
    titellos = sum(1 for t in idx.titel if not t.strip())
    print(f"Index: {n} Normen, davon titellos: {titellos}")
    print(f"hubness: mean={hubness.mean():.3f} std={hubness.std():.3f} "
          f"max={hubness.max():.3f} min={hubness.min():.3f}")

    print("\nTOP-20 HUBS (höchste mittlere Query-Ähnlichkeit):")
    print(f"  {'hubness':>7}  {'titel?':6}  {'txtlen':>6}  zitat / titel")
    top = order[:20]
    titellos_in_top = 0
    for i in top:
        hat_titel = bool(idx.titel[i].strip())
        if not hat_titel:
            titellos_in_top += 1
        # Textlänge schätzen wir aus dem Index nicht direkt; Titel als Proxy zeigen.
        print(f"  {hubness[i]:7.3f}  {'JA' if hat_titel else 'nein':6}  "
              f"{'-':>6}  {idx.zitate[i]:12} {idx.titel[i][:45]}")
    print(f"\n  von Top-20 Hubs sind {titellos_in_top} titellos, "
          f"{20 - titellos_in_top} HABEN einen Titel.")
    print("  → Ist hubness deckungsgleich mit 'titellos'? Wenn nein, ist 'titellos' "
          "das falsche Filterkriterium und Informationsdichte das richtige.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
