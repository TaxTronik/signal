"""Embedding-Fine-Tuning (B300, GATED Experiment) — Datenaufbereitung.

Baut aus gefilterten Discovery-Trainingspaaren ``(anchor, positive)``:
anchor = Laien-/Kanzlei-Formulierung (Discovery-``inhalt``), positive = Normtext
der Zielnorm. MultipleNegativesRankingLoss zieht Alltagssprache näher an
Normsprache — der Root-Fix gegen das gemessene Modell-Verständnis-Limit.

Sicherheitslinien (siehe Memory b300-discovery-finetune-run-order):
- NEUES Artefakt, überschreibt NIE den Produktions-BGE-M3.
- Trainingsdaten DISJUNKT vom Bench-Gold (Katalog-Paraphrasen + HAND-POSITIVE):
  identische Bench-Anfragen werden ausgeschlossen, sonst ist die held-out-
  Messung keine echte Akzeptanz-Gate (Leckage). Andere Paraphrasen DESSELBEN §
  bleiben erlaubt — das ist erwünschte Generalisierung, keine Leckage.
- Das Artefakt bleibt Research, bis embedding_bench die harte Guardrail erfüllt
  (Produktion-mit-Ankern ≥ Baseline, MRR ≥, KATALOG held-out besser, HAND
  anker-frei besser, Negativ-Floor nicht höher, FP nicht schlechter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


@dataclass
class Trainingspaar:
    anchor: str          # Laien-/Kanzlei-Formulierung (Discovery-inhalt)
    positive: str        # Normtext der Zielnorm
    ziel_norm_id: str


def baue_trainingspaare(
    kandidaten: Iterable[dict[str, Any]],
    *,
    norm_text: Callable[[str], str],
    bench_queries: set[str],
) -> list[Trainingspaar]:
    """``kandidaten``: Discovery-Kandidaten (dicts; idealerweise schon
    training-gate-gefiltert). ``norm_text``: ID → Normtext der Zielnorm ('' wenn
    fehlt). ``bench_queries``: normalisierte Bench-Gold-Anfragen → Leckage-Ausschluss."""
    paare: list[Trainingspaar] = []
    gesehen: set[tuple[str, str]] = set()
    for k in kandidaten:
        anchor = str(k.get("inhalt", "")).strip()
        ziel = str(k.get("ziel_norm_id", "")).strip()
        if not anchor or not ziel:
            continue
        if _norm(anchor) in bench_queries:        # identisch mit Bench-Anfrage → Leckage
            continue
        positive = (norm_text(ziel) or "").strip()
        if not positive:
            continue
        schluessel = (_norm(anchor), ziel)
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)
        paare.append(Trainingspaar(anchor=anchor, positive=positive, ziel_norm_id=ziel))
    return paare


def bench_query_set(katalog, hand_positive: Iterable[tuple[str, Any]]) -> set[str]:
    """Normalisierte Bench-Gold-Anfragen (für den Leckage-Ausschluss): die
    kuratierten Katalog-Paraphrasen + die HAND-POSITIVE-Sätze der Gold-Bench."""
    qs: set[str] = set()
    for e in katalog:
        for p in (getattr(e, "paraphrasen", ()) or ()):
            qs.add(_norm(p))
    for satz, _ziele in hand_positive:
        qs.add(_norm(satz))
    return qs
