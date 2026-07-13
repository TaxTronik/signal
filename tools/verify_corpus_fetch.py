#!/usr/bin/env python3
"""Verwirf partielle, veraltete oder nicht parsebare gii-Korpus-Builds."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from risk_layer import corpus, gii


LAW_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def verify(korpus: Path) -> None:
    if not korpus.is_dir() or korpus.is_symlink():
        raise ValueError(f"kein reguläres Korpus-Verzeichnis: {korpus}")
    config = corpus.lade_config(korpus)
    if not config:
        raise ValueError("gesetze.yaml ist leer")

    expected: set[str] = set()
    for gesetz in config:
        if not LAW_RE.fullmatch(gesetz.law):
            raise ValueError(f"nicht portabler law-Code in gesetze.yaml: {gesetz.law!r}")
        folded = gesetz.law.casefold()
        if folded in expected:
            raise ValueError(f"doppelter law-Code in gesetze.yaml: {gesetz.law!r}")
        expected.add(folded)

    gii_dir = korpus / "gii"
    if not gii_dir.is_dir() or gii_dir.is_symlink():
        raise ValueError("gii-Verzeichnis fehlt oder ist ein Symlink")
    entries = list(gii_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ValueError("gii enthält Verzeichnisse, Links oder Sonderdateien")
    if any(path.suffix.casefold() != ".xml" for path in entries):
        raise ValueError("gii enthält unerwartete Nicht-XML-Dateien")

    actual = {path.stem.casefold() for path in entries}
    if len(actual) != len(entries):
        raise ValueError("gii enthält kollidierende XML-Dateinamen")
    if actual != expected:
        missing = sorted(expected - actual)
        stale = sorted(actual - expected)
        raise ValueError(
            f"Korpus-Fetch ist nicht vollständig (fehlend={missing}, veraltet={stale})"
        )

    for path in sorted(entries, key=lambda item: item.name.casefold()):
        try:
            nodes = gii.parse_datei(path)
        except (OSError, ValueError, ET.ParseError) as exc:
            raise ValueError(f"gii-XML nicht parsebar: {path.name}: {exc}") from exc
        if not nodes:
            raise ValueError(f"gii-XML enthält keine nutzbaren Normen: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("korpus", type=Path)
    args = parser.parse_args()
    verify(args.korpus)
    print(f"vollständiger frischer Korpus: {args.korpus}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"Korpus-Build abgelehnt: {exc}") from exc
