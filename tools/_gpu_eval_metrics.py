"""Modellfreie Metriken fuer den A/B-Vergleich der echten Satz-Pipeline.

Eine Query kann mehrere Saetze mit eigenen Ranglisten liefern. Hit@k und MRR
verwenden den BESTEN satzweisen Rang, nie eine kuenstliche globale Top-k-Liste.
Zielabdeckung ist die je Query getroffene Fraktion ihrer eindeutigen Gold-Ziele.
Negative Queries sind falsch positiv, sobald irgendein Satz eine Norm liefert.
Diese Fehlvorschlagsquote misst keine Praezision auf positivem Retrieval-Gold:
dessen Ziel-Listen enthalten nicht zwingend alle fachlich akzeptablen Normen.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _matches(candidate: str, target: str) -> bool:
    """Ein Ziel trifft sich selbst oder tiefere Pfadsegmente, nie Nachbar-IDs."""
    return candidate == target or candidate.startswith(target + ":")


def evaluate_variant(suggestions: Sequence[Mapping[str, Any]],
                     targets: Iterable[str]) -> dict[str, Any]:
    """Bewertet eine Query mit satzweisen ``normen: [{id: ...}]``-Ranglisten.

    ``n_norms`` zaehlt die angezeigten Eintraege, auch wenn eine Norm in mehreren
    Saetzen steht. Die Zielabdeckung dedupliziert dagegen Gold-Ziele und Treffer.
    Ohne Ziele (Negativ-Gold) sind ``best_rank=None`` und ``target_coverage=0``.
    Es werden keine Scores oder Ranglisten veraendert und keine Daten gerundet.
    """
    target_ids = set(targets)
    matched: set[str] = set()
    best_rank: int | None = None
    n_norms = 0
    for suggestion in suggestions:
        norms = suggestion["normen"]
        n_norms += len(norms)
        for rank, norm in enumerate(norms, start=1):
            hits = {target for target in target_ids if _matches(norm["id"], target)}
            if hits:
                matched.update(hits)
                best_rank = rank if best_rank is None else min(best_rank, rank)
    return {
        "best_rank": best_rank,
        "target_coverage": len(matched) / len(target_ids) if target_ids else 0.0,
        "n_suggestions": len(suggestions),
        "n_norms": n_norms,
    }


def _aggregate(evaluations: Sequence[Mapping[str, Any]], *,
               negative: bool) -> dict[str, Any]:
    n = len(evaluations)

    def mean(values: Iterable[float]) -> float | None:
        return sum(values) / n if n else None

    result: dict[str, Any] = {
        "n": n,
        "mean_suggestions": mean(e["n_suggestions"] for e in evaluations),
        "mean_norms": mean(e["n_norms"] for e in evaluations),
    }
    if negative:
        result["false_suggestion_rate"] = mean(
            float(e["n_norms"] > 0) for e in evaluations)
    else:
        ranks = [e["best_rank"] for e in evaluations]
        result.update({
            "hit@1": mean(float(rank is not None and rank <= 1) for rank in ranks),
            "hit@5": mean(float(rank is not None and rank <= 5) for rank in ranks),
            "mrr_best_sentence": mean(1.0 / rank if rank else 0.0 for rank in ranks),
            "mean_target_coverage": mean(e["target_coverage"] for e in evaluations),
        })
    return result


def summarize(rows: Iterable[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    """Gesamt- und Gruppenmetriken fuer ``before`` oder ``after``.

    Jede Zeile enthaelt ``negative``, ``group``, ``targets`` und je Variante
    ``{'suggestions': [...]}``. Jeder Gold-Fall zaehlt einmal, unabhaengig von
    seiner Satzanzahl. Leere Aggregate tragen ``n=0`` und ``None`` fuer Raten.
    Die Gruppen werden deterministisch sortiert; leeres positives Gold in einer
    einzelnen Zeile wird als Datenfehler abgewiesen statt als Verfehlung gezaehlt.
    """
    all_cases: dict[bool, list[dict[str, Any]]] = {False: [], True: []}
    groups: dict[str, dict[bool, list[dict[str, Any]]]] = {}
    for row in rows:
        negative = row["negative"]
        targets = list(row["targets"])
        if not negative and not targets:
            raise ValueError(f"Positiver Gold-Fall ohne Ziele: {row.get('case_id', '?')}")
        evaluation = evaluate_variant(row[variant]["suggestions"], targets)
        all_cases[negative].append(evaluation)
        group = groups.setdefault(row["group"], {False: [], True: []})
        group[negative].append(evaluation)

    def aggregate_pair(cases: Mapping[bool, Sequence[Mapping[str, Any]]]) -> dict:
        return {
            "positive": _aggregate(cases[False], negative=False),
            "negative": _aggregate(cases[True], negative=True),
        }

    return {
        **aggregate_pair(all_cases),
        "groups": {group: aggregate_pair(cases) for group, cases in sorted(groups.items())},
    }


def compare_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Fall-IDs der gepaarten Hit@5-/Fehlvorschlags-Gewinne und -Verluste.

    Bei positiven Faellen ist Hit@5 besser als kein Hit@5; bei negativen Faellen
    ist Abstention besser als irgendein Vorschlag. Rang- oder Umfangsaenderungen
    innerhalb derselben Klasse gelten hier weder als Gewinn noch als Verlust.
    """
    result: dict[str, list[str]] = {
        "positive_hit5_wins": [], "positive_hit5_losses": [],
        "negative_abstention_wins": [], "negative_abstention_losses": [],
    }
    for row in rows:
        targets = list(row["targets"])
        before = evaluate_variant(row["before"]["suggestions"], targets)
        after = evaluate_variant(row["after"]["suggestions"], targets)
        if row["negative"]:
            was_good = before["n_norms"] == 0
            now_good = after["n_norms"] == 0
            prefix = "negative_abstention"
        else:
            was_good = before["best_rank"] is not None and before["best_rank"] <= 5
            now_good = after["best_rank"] is not None and after["best_rank"] <= 5
            prefix = "positive_hit5"
        if was_good != now_good:
            result[f"{prefix}_{'wins' if now_good else 'losses'}"].append(row["case_id"])
    return result
