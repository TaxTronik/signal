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

# Zeichen, die ersatzlos entfernt werden (zero-width, weiches Trennzeichen)
_LOESCHEN = {
    "­",  # soft hyphen
    "​",  # zero width space
    "‌",  # zero width non-joiner
    "‍",  # zero width joiner
    "﻿",  # zero width no-break space / BOM
}

# Exotische Leerzeichen → normales Leerzeichen
_SPATIEN = {
    " ": " ",  # no-break space
    " ": " ",  # figure space
    " ": " ",  # thin space
    " ": " ",  # narrow no-break space
    "⁠": " ",  # word joiner
    "\t": " ",
    "": "\n",
    "": "\n",  # form feed → Absatztrenner
    " ": "\n",  # line separator
    " ": "\n",  # paragraph separator
}

_TRENNSTRICHE = {"-", "‐", "‑"}  # Bindestrich, hyphen, non-breaking hyphen


def _ist_buchstabe(ch: str) -> bool:
    return ch.isalpha()


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
            j = i + 1
            if j < n and text[j] == "\r":
                j += 1
            if j < n and text[j] == "\n":
                # nachfolgende Einrückung (Spaces/Tabs) der Folgezeile überspringen
                k = j + 1
                while k < n and text[k] in " \t":
                    k += 1
                vorher = out_chars[-1] if out_chars else ""
                nachher = text[k] if k < n else ""
                if _ist_buchstabe(vorher) and _ist_buchstabe(nachher):
                    # Trennstrich + Umbruch + Einrückung verwerfen → Wort zusammenfügen
                    i = k
                    continue
            # gewöhnlicher Bindestrich: unverändert übernehmen
            out_chars.append(ch)
            out_map.append(i)
            i += 1
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


def _segmentiere(normalized: str, offset_map: list[int]) -> list[Segment]:
    """Zerlegt in Absatz-Segmente (Trennung an Leerzeilen / Zeilenumbrüchen).

    Offsets der Segmente werden auf den ORIGINALTEXT abgebildet.
    """
    segments: list[Segment] = []
    idx = 0
    pos = 0
    n = len(normalized)
    while pos < n:
        # führende Zeilenumbrüche / Leerzeichen überspringen
        while pos < n and normalized[pos] in "\n \t":
            pos += 1
        if pos >= n:
            break
        start = pos
        # Segmentende: nächste (Doppel-)Leerzeile bzw. Zeilenumbruch
        nl = normalized.find("\n", pos)
        end = nl if nl != -1 else n
        seg_text = normalized[start:end].strip()
        if seg_text:
            # auf nicht-leeren Bereich trimmen, Offsets exakt halten
            lead = len(normalized[start:end]) - len(normalized[start:end].lstrip())
            real_start = start + lead
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
        pos = end + 1 if nl != -1 else n
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
