"""Gemeinsame Helfer für die tools/-Skripte (bewusst KEIN Teil des risk_layer-Pakets).

tools/ hat kein __init__.py: die Skripte laufen direkt (``python tools/x.py``)
und schieben die Repo-Wurzel selbst auf den sys.path. ``tools`` ist damit ein
Namespace-Paket (PEP 420) — ``from tools._common import ...`` funktioniert nach
dem sys.path-Insert ohne Paket-Deklaration.

Hier liegt nur, was zuvor in mehreren Skripten identisch dupliziert war:
UTF-8-Konsole und die verweist_auf-Kanten-Extraktion. Atomares Schreiben kommt
NICHT von hier, sondern aus risk_layer.persistenz.atomar_schreiben (fsync +
Windows-Retry + tmp-Cleanup — die frühere tools-Kopie konnte das alles nicht).
"""

from __future__ import annotations

import sys


def utf8_konsole() -> None:
    """stdout/stderr auf UTF-8 umstellen (Windows-Konsole ist sonst cp1252:
    Umlaute/Pfeile in den Skript-Ausgaben würden mit UnicodeEncodeError sterben).
    Zuvor als identischer 5-Zeilen-Block in ~20 Skripten dupliziert."""
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def verweist_ids(graph, node_id: str) -> set[str]:
    """Ausgehende verweist_auf-Kanten eines §-Knotens (für die Relations-Erdung)."""
    ids: set[str] = set()
    for v in graph.aufloesen(node_id).get("verweist_auf") or []:
        i = (v.get("id") or v.get("dst") or v.get("ziel") or "") if isinstance(v, dict) else str(v)
        if i:
            ids.add(i)
    return ids
