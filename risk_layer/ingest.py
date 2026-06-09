"""Ingest /LF10/ /LF20/ /LF30/.

Nimmt Rohtext entgegen, normalisiert ihn *offset-erhaltend* und zerlegt ihn in
adressierbare Segmente. Zentrale Eigenschaft: Jeder Zeichenindex des
normalisierten Textes lässt sich über ``offset_map`` exakt auf den Originaltext
zurückführen – damit Markierungen verlustfrei zurückgemappt werden können.

Bewusst KEINE globale NFC-Transformation des Gesamtstrings (würde Offsets
verschieben). Stattdessen explizite, prüfbare Zeichen-Transformationen für die
typischen PDF-/DOCX-Extrakt-Artefakte (Ligaturen, Trennstriche, Sonderspatien).
"""

from __future__ import annotations

import hashlib
import re

from .models import IngestErgebnis, Segment

# Ligaturen → Klartext (häufig in PDF-Extrakten)
_LIGATUREN: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
    "Ĳ": "IJ",
    "ĳ": "ij",
}

# Zeichen, die ersatzlos entfernt werden (zero-width). Der Word Joiner ist
# ebenfalls zero-width und gehoert HIERHER: ein Mapping auf ein sichtbares
# Leerzeichen (frueher in _SPATIEN) zerschnitt Woerter ("Ein⁠gliederung"
# -> "Ein gliederung") und kostete Katalogtreffer.
_LOESCHEN = {
    "​",  # zero width space
    "‌",  # zero width non-joiner
    "‍",  # zero width joiner
    "⁠",  # word joiner
    "﻿",  # zero width no-break space / BOM
}

# Weiches Trennzeichen (U+00AD): wird ersatzlos geloescht – ausser am
# Zeilenumbruch, dort fuegt es wie ein regulaerer Trennstrich das Wort
# zusammen (klassisches PDF-Artefakt, eigener Pfad in normalisiere()).
_WEICHES_TRENNZEICHEN = "­"

# Exotische Leerzeichen → normales Leerzeichen
_SPATIEN = {
    " ": " ",  # no-break space
    " ": " ",  # figure space
    " ": " ",  # thin space
    " ": " ",  # narrow no-break space
    "\t": " ",
    "": "\n",
    "": "\n",  # form feed → Absatztrenner
    " ": "\n",  # line separator
    " ": "\n",  # paragraph separator
}

_TRENNSTRICHE = {"-", "‐", "‑"}  # Bindestrich, hyphen, non-breaking hyphen


def _ist_buchstabe(ch: str) -> bool:
    return ch.isalpha()


def _umbruch_fuge(text: str, i: int, out_chars: list[str]) -> int | None:
    """Trennstrich an Position ``i`` + Zeilenumbruch: Wort zusammenfügen?

    Liefert den Index HINTER Umbruch und Folge-Einrückung, wenn links wie
    rechts Buchstaben stehen (Strich + Umbruch + Einrückung verwerfen) – sonst
    None. Gemeinsamer Pfad für reguläre Trennstriche und das weiche
    Trennzeichen; rein lesend, die Offset-Treue bleibt beim Aufrufer.
    """
    n = len(text)
    j = i + 1
    if j < n and text[j] == "\r":
        j += 1
    if j >= n or text[j] != "\n":
        return None
    # nachfolgende Einrückung (Spaces/Tabs) der Folgezeile überspringen
    k = j + 1
    while k < n and text[k] in " \t":
        k += 1
    vorher = out_chars[-1] if out_chars else ""
    nachher = text[k] if k < n else ""
    if _ist_buchstabe(vorher) and _ist_buchstabe(nachher):
        return k
    return None


def normalisiere(text: str) -> tuple[str, list[int]]:
    """Normalisiert ``text`` offset-erhaltend.

    Returns:
        (normalized_text, offset_map) mit
        ``offset_map[i]`` = Originalindex des Zeichens ``normalized_text[i]``.
    """
    out_chars: list[str] = []
    out_map: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]

        # 1) Zeilenumbruch-Trennung: "Einglie-\nderung" → "Eingliederung"
        if ch in _TRENNSTRICHE:
            k = _umbruch_fuge(text, i, out_chars)
            if k is not None:
                i = k
                continue
            # gewöhnlicher Bindestrich: unverändert übernehmen
            out_chars.append(ch)
            out_map.append(i)
            i += 1
            continue

        # 1b) Weiches Trennzeichen: am Zeilenumbruch wie ein regulärer Trenn-
        # strich zusammenfügen ("Einglie­\nderung" → "Eingliederung",
        # klassisches PDF-Artefakt) – sonst ersatzlos löschen. Vorher lief es
        # über _LOESCHEN und das Wort blieb am Umbruch zerschnitten.
        if ch == _WEICHES_TRENNZEICHEN:
            k = _umbruch_fuge(text, i, out_chars)
            i = k if k is not None else i + 1
            continue

        # 2) ersatzlos zu löschende Zeichen
        if ch in _LOESCHEN:
            i += 1
            continue

        # 3) Ligaturen expandieren (jedes Ersatzzeichen zeigt auf die Ligaturquelle)
        if ch in _LIGATUREN:
            for sub in _LIGATUREN[ch]:
                out_chars.append(sub)
                out_map.append(i)
            i += 1
            continue

        # 4) Sonderspatien vereinheitlichen
        if ch in _SPATIEN:
            out_chars.append(_SPATIEN[ch])
            out_map.append(i)
            i += 1
            continue

        # 5) CR aus CRLF schlucken (LF bleibt als Absatzsignal erhalten)
        if ch == "\r":
            if i + 1 < n and text[i + 1] == "\n":
                i += 1
                continue
            out_chars.append("\n")
            out_map.append(i)
            i += 1
            continue

        # 6) unverändert
        out_chars.append(ch)
        out_map.append(i)
        i += 1

    return "".join(out_chars), out_map


# Absatzgrenze = LEERZEILE (ggf. mit Spaces/Tabs darin), wie in
# pipeline._absatz_grenzen – nicht jeder einzelne Zeilenumbruch.
_ABSATZ_TRENNER = re.compile(r"\n[ \t]*\n+")


def _segmentiere(normalized: str, offset_map: list[int]) -> list[Segment]:
    """Zerlegt in Absatz-Segmente (Trennung an Leerzeilen).

    Einfache Zeilenumbrüche INNERHALB eines Absatzes (Fließtext mit hartem
    Umbruch, PDF-Extrakte) trennen NICHT – vorher wurde an jedem ``\\n``
    geschnitten, entgegen dem dokumentierten Absatz-Begriff.
    Offsets der Segmente werden auf den ORIGINALTEXT abgebildet.
    """
    segments: list[Segment] = []
    idx = 0
    pos = 0
    n = len(normalized)
    grenzen = [(m.start(), m.end()) for m in _ABSATZ_TRENNER.finditer(normalized)]
    grenzen.append((n, n))                  # letzter Absatz bis Textende
    for g_start, g_ende in grenzen:
        roh = normalized[pos:g_start]
        seg_text = roh.strip()
        if seg_text:
            # auf nicht-leeren Bereich trimmen, Offsets exakt halten
            lead = len(roh) - len(roh.lstrip())
            real_start = pos + lead
            real_end = real_start + len(seg_text)
            orig_start = offset_map[real_start]
            orig_end = offset_map[real_end - 1] + 1
            segments.append(
                Segment(
                    index=idx,
                    start=orig_start,
                    end=orig_end,
                    text=normalized[real_start:real_end],
                )
            )
            idx += 1
        pos = g_ende
    return segments


def text_hash(text: str) -> str:
    """SHA-256 des Originaltextes (Reproduzierbarkeit/Audit, /LF30/)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest(text: str) -> IngestErgebnis:
    """Vollständiger Ingest-Schritt (/LF10/–/LF30/)."""
    normalized, offset_map = normalisiere(text)
    segments = _segmentiere(normalized, offset_map)
    return IngestErgebnis(
        original_text=text,
        normalized_text=normalized,
        offset_map=offset_map,
        segments=segments,
        text_hash=text_hash(text),
    )
