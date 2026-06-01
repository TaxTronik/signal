"""Audit-Log /LF210/ /AB60/.

Jeder Lauf erzeugt ein vollständiges, maschinenlesbares Protokoll: Eingabe-Hash,
Katalogversion, Modell- + Quantisierungsversion (für Schicht 1: null), Zeitstempel,
Trefferliste. Dient als Nachweis für die PS-980-Kontrolldokumentation und zur
Haftungsbegrenzung.

Das Audit-Log ist die maßgebliche Reproduzierbarkeits- und Nachvollziehbarkeits-
Spur (/NF20/, /NF30/): Aus Eingabe-Hash + Katalogversion ist das Schicht-1-Ergebnis
bit-identisch rekonstruierbar.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import RunReport, Status
from .zitate import kanonisiere_fundstelle, parse_normzitat


def _norm_anker_ids(norm_anker: list[str]) -> list[str]:
    ids: list[str] = []
    for zitat in norm_anker:
        for anker in parse_normzitat(zitat):
            if anker.norm_id:
                ids.append(anker.norm_id.to_id())
    return ids


def _quellen_ids(quellen) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for q in quellen:
        out.append({
            "typ": q.typ,
            "fundstelle": q.fundstelle,
            "id": kanonisiere_fundstelle(q.typ, q.fundstelle),
        })
    return out


def baue_audit(
    report: RunReport,
    *,
    schicht1_engine: str = "deterministic-catalog-matcher",
    modell: Optional[str] = None,
    quantisierung: Optional[str] = None,
    seed: Optional[int] = None,
    temperatur: Optional[float] = None,
) -> dict[str, Any]:
    """Baut das Audit-Dict eines Laufs.

    Für reine Schicht 1 bleiben Modellangaben ``None`` – belegt, dass KEIN
    Modell beteiligt war. Bei aktiver Schicht 2 sind Modell/Quantisierung/Seed/
    Temperatur zwingend zu füllen (/LF140/).
    """
    info = report.schicht2_info or {}
    return {
        "schema": "risk-layer/audit/v1",
        "erzeugt_am": report.erzeugt_am,
        "eingabe_hash": report.text_hash,
        "katalog_version": report.katalog_version,
        "schicht1": {
            "engine": schicht1_engine,
            "deterministisch": True,
        },
        "schicht2": {
            "aktiv": bool(report.kandidaten),
            "engine": info.get("engine"),
            "modell": info.get("modell", modell),
            "quantisierung": info.get("quantisierung", quantisierung),
            "seed": info.get("seed", seed),
            "temperatur": info.get("temperatur", temperatur),
        },
        "zusammenfassung": report.zusammenfassung(),
        "treffer": [
            {
                "begriff_id": k.eintrag.id,
                "status": k.status.value,
                "herkunft": k.herkunft(),
                "start": k.treffer.start,
                "end": k.treffer.end,
                "matched_text": k.treffer.matched_text,
                "surface_beleg": k.treffer.surface,
                "norm_anker": list(k.eintrag.norm_anker),
                "norm_anker_ids": _norm_anker_ids(list(k.eintrag.norm_anker)),
                "definitionsquelle": _quellen_ids(k.quellen),
                "ist_luecke": k.status is Status.LUECKE,
            }
            for k in report.karten
        ],
        "kandidaten": report.kandidaten,
        "risiken": [r.as_dict() for r in report.risiken],
    }


def schreibe_audit(audit: dict[str, Any], pfad: str | Path) -> Path:
    """Schreibt das Audit-Log als JSON (UTF-8, stabile Schlüsselreihenfolge)."""
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return pfad


def jetzt_iso() -> str:
    """Zeitstempel in ISO-8601 (UTC)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
