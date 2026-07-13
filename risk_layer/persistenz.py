"""Gemeinsame Store-Primitive: Slug-Bildung + atomares Schreiben.

Zuvor lag beides dreifach (fall/berater/aufgabe) leicht abweichend implementiert:
unterschiedliche Umlaut-Behandlung und Separatoren ließen denselben Namen je
Store anders verschlüsseln, und ``write_text`` schrieb nicht-atomar (paralleler
Leser sah halb-geschriebene JSON). Beide Primitive leben jetzt EINMAL hier; die
Stores reichen nur Separator/Default durch.

Grenze ehrlich benannt: ``atomar_schreiben`` verhindert torn reads/halbe
Dateien, NICHT verlorene Updates konkurrierender Read-Modify-Write-Zyklen.
Dagegen müssen die Stores selbst vorgehen (frischen Datei-Stand vor dem
Schreiben neu laden, s. BeraterStore); ohne File-Locking (bewusst weggelassen –
Windows/Unix-Portabilität) bleibt ein schmales Cross-Prozess-Restrisiko.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, cast

# Umlaut-Transliteration: filesystem- und ID-sicher (ASCII), über alle Stores
# IDENTISCH. "Müller" → "mueller" – egal welcher Store den Slug bildet.
_UML = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "Ä": "ae", "Ö": "oe", "Ü": "ue"}


def als_int(v: object, default: int | None = 0) -> int | None:
    """Robuste int-Coercion für Store-Eingaben aus dem Request-Body.

    Ungültige Werte (z. B. ``'abc'``, ``None``, ``''``) ergeben ``default`` statt
    einen unbehandelten ``ValueError``/``TypeError`` (→ HTTP 500) auszulösen.
    """
    try:
        # ``object`` ist an der API-Grenze bewusst breit; ``int`` akzeptiert
        # zur Laufzeit mehr Protokolltypen als seine statischen Overloads
        # ausdruecken. Der Cast aendert nichts am Verhalten, die Fehler bleiben
        # durch den folgenden Guard sauber auf ``default`` abgebildet.
        return int(cast(Any, v))
    except (TypeError, ValueError):
        return default


def slugify(text: str, *, default: str, sep: str = "-", maxlen: int = 60) -> str:
    """Kanonischer Slug: ASCII-Kleinbuchstaben/Ziffern, Rest → ``sep``.

    ``sep`` ist pro Store wählbar (fall: '-', berater/aufgabe: '_'), die
    Algorithmik (Transliteration, Filterung, Längen-Cap) ist überall dieselbe.
    Leeres Ergebnis fällt auf ``default`` zurück.
    """
    umgewandelt = "".join(_UML.get(c, c) for c in str(text).lower())
    roh = "".join(c if (c.isalnum() and c.isascii()) else sep for c in umgewandelt)
    # Der Längen-Cap kann mitten am Separator schneiden ("…abc_") – Rest-``sep``
    # abstreifen. Slugs sind persistente IDs: das ändert nur KÜNFTIG gebildete
    # Slugs, bereits persistierte Daten bleiben unberührt.
    teil = sep.join(p for p in roh.split(sep) if p)[:maxlen].rstrip(sep)
    return teil or default


def atomar_schreiben(pfad: str | Path, inhalt: str, *, encoding: str = "utf-8") -> None:
    """Schreibt ``inhalt`` atomar nach ``pfad`` (temp im selben Verzeichnis + replace).

    ``os.replace`` ist auf demselben Filesystem atomar (auch unter Windows) und
    ersetzt eine bestehende Datei ungeteilt – ein paralleler Leser sieht immer
    ENTWEDER die alte ODER die neue, nie eine halb-geschriebene Datei.
    """
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(pfad.parent), prefix=f".{pfad.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(inhalt)
            f.flush()
            os.fsync(f.fileno())
        # os.replace kann unter Windows kurz mit PermissionError scheitern, wenn ein
        # paralleler (ungelockter) Leser die Zieldatei gerade offen hat – Python öffnet
        # ohne FILE_SHARE_DELETE. Kurzer Retry überbrückt das schmale Fenster, statt
        # den Save fehlschlagen zu lassen.
        for versuch in range(5):
            try:
                os.replace(tmp, pfad)
                break
            except PermissionError:
                if versuch == 4:
                    raise
                time.sleep(0.02)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
