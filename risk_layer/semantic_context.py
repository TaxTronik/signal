"""Satzkontext fuer semantische Suche mit unveraenderten Originalspans.

Fokus und Suchtext bleiben getrennt: Ein Modell darf Nachbarsaetze sehen, seine
Vorschlaege werden weiterhin dem urspruenglichen Satzspan zugeordnet. Dieser
Span ist kein Beweis, dass jede vorgeschlagene Norm gerade dort einschlaegig ist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
from typing import Any, Literal


ContextMode = Literal["sentence", "neighbors", "paragraph", "coherent"]
_BLANK_PARAGRAPH = re.compile(r"\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*")
_COHERENT_BLANK_PARAGRAPH = re.compile(r"\r?\n[^\S\r\n]*\r?\n(?:[^\S\r\n]*\r?\n)*")
# These abbreviations introduce a following phrase; they do not finish it.
# This repair is local to coherent semantic context, not the hard-mark splitter.
_CONTINUING_ABBREVIATION = re.compile(
    r"\b(?:lt|betr|bzgl|insb|entspr|einschl|ausschl)$", re.IGNORECASE,
)


@dataclass(frozen=True)
class ContextQuery:
    focus_start: int
    focus_end: int
    focus_text: str
    query_text: str
    context_start: int
    context_end: int
    mode: ContextMode
    context_truncated: bool = False
    focus_truncated: bool = False


@dataclass(frozen=True)
class _FocusSpan:
    text: str
    start: int
    end: int


def _coherent_sentences(text: str) -> list[_FocusSpan]:
    from .muster import saetze_mit_offset

    # The legacy detector recognizes these closing-quote shapes. Its shadow
    # keeps one code point per original character; every returned text is then
    # sliced from the unmodified source, never from this detection-only copy.
    shadow = text.translate(str.maketrans("«‹", "»›"))
    out: list[_FocusSpan] = []
    for region_start, region_end in _paragraphs(text, coherent=True):
        for sentence in saetze_mit_offset(shadow[region_start:region_end]):
            start, end = region_start + sentence.start, region_start + sentence.end
            closing = re.match(r'''[«»‹›“”‘’"')\]]+''', text[start:end])
            if out and closing and re.fullmatch(r"[.!?]+", text[out[-1].end:start]):
                start += closing.end()
                previous = out[-1]
                out[-1] = _FocusSpan(text[previous.start:start], previous.start, start)
            if text[start:end].strip():
                out.append(_FocusSpan(text[start:end], start, end))
    return out


def _coherent_foci(text: str, sentences: Sequence[Any]) -> list[_FocusSpan]:
    """Restore continuation boundaries using only untouched source slices.

    Colons and semicolons connect a heading/value or list within a sentence.
    True sentence endings and blank paragraphs remain focus boundaries. Validate
    every supplied span before merging, so a bad splitter cannot hide its error.
    """
    out: list[_FocusSpan] = []
    previous_end = 0
    for sentence in sentences:
        if (type(sentence.start) is not int or type(sentence.end) is not int
                or not previous_end <= sentence.start <= sentence.end <= len(text)
                or text[sentence.start:sentence.end] != sentence.text):
            raise ValueError("Satzsplitter liefert keinen offsettreuen Originaltext")
        previous_end = sentence.end
        current = _FocusSpan(sentence.text, sentence.start, sentence.end)
        if out:
            previous = out[-1]
            first_content, _ = _trim_span(text, current.start, current.end)
            separator = text[previous.end:first_content]
            continuation = (
                re.fullmatch(r"[;:]\s*", separator) is not None
                or (_CONTINUING_ABBREVIATION.search(previous.text.rstrip()) is not None
                    and re.fullmatch(r"\.\s*", separator) is not None)
            )
            if continuation and not _COHERENT_BLANK_PARAGRAPH.search(separator):
                out[-1] = _FocusSpan(text[previous.start:current.end],
                                     previous.start, current.end)
                continue
        out.append(current)
    return out


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    raw = text[start:end]
    left = start + len(raw) - len(raw.lstrip())
    return left, max(left, end - len(raw) + len(raw.rstrip()))


def _paragraphs(text: str, *, coherent: bool = False) -> list[tuple[int, int]]:
    spans = []
    start = 0
    boundaries = _COHERENT_BLANK_PARAGRAPH if coherent else _BLANK_PARAGRAPH
    for separator in boundaries.finditer(text):
        if text[start:separator.start()].strip():
            spans.append((start, separator.start()))
        start = separator.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _bounded_span(start: int, end: int, focus_start: int, focus_end: int,
                  max_chars: int) -> tuple[int, int]:
    if end - start <= max_chars:
        return start, end
    # Rund um den Fokus zentrieren; an Absatzraendern die freie Seite ausnutzen.
    # Passt der Fokus selbst nicht ins Budget, wird sein Zentrum verwendet.
    left = max(start, (focus_start + focus_end - max_chars) // 2)
    right = min(end, left + max_chars)
    return max(start, right - max_chars), right


def build_context_queries(
    text: str, *, mode: ContextMode = "sentence", max_chars: int = 900,
    neighbor_radius: int = 1, min_focus_chars: int = 12,
    sentence_splitter: Callable[[str], Sequence[Any]] | None = None,
    cross_paragraphs: bool = False,
) -> list[ContextQuery]:
    """Erzeugt einen Suchtext je geeignetem Satz mit exakten Originaloffsets.

    ``sentence`` ist die Baseline: Suchtext = unveraenderter ``Satz.text`` des
    Splitters, einschliesslich aeusserem Leerraum. Wie in der echten Pipeline
    gilt die Zeichenbegrenzung hier NICHT; so bleibt ihr Encoderinput identisch.

    ``neighbors`` nimmt bis zu ``neighbor_radius`` Saetze auf beiden Seiten mit;
    kurze Saetze bleiben als Kontext erhalten, auch wenn sie selbst keinen
    Fokus bilden. ``paragraph`` verwendet den ganzen umgebenden Absatz.
    ``coherent`` verwendet denselben Absatzkontext, verbindet aber im Fokus
    Abkuerzungsfortsetzungen (z. B. ``lt.``) und durch Doppelpunkt/Semikolon
    getrennte Satzteile. Dadurch bleiben Listeneinleitung und Werte zusammen.
    Vergleichssaetze bleiben eigene Foki mit gemeinsamem Kontext. Das bestaetigt
    keine Fakten oder Rechtsfolge fuer jeden einzelnen Fokus.
    Alle Kontextmodi begrenzen ihren Suchtext auf ``max_chars``. Ein passender
    Fokus bleibt vollstaendig darin; ist schon er laenger, kennzeichnet
    ``focus_truncated`` den zentrierten Ausschnitt. Die Fokusoffsets bleiben
    auch dann vollstaendig und unveraendert. Beschnitt erfolgt nach Zeichen.

    Ohne ``cross_paragraphs`` werden Kontextmodi an Leerzeilen getrennt, auch
    bei CRLF. Ein einzelner Zeilenumbruch begrenzt den Kontext nicht. Mit dem
    expliziten Schalter darf Nachbarschaft absatzuebergreifend sein; im Modus
    ``paragraph``/``coherent`` wird dann das ganze Dokument zum begrenzten
    Kontextbereich; Foki werden auch dann nicht ueber Leerabsatzgrenzen vereint.
    Ein injizierter Splitter muss ``text/start/end`` relativ zu seinem Argument
    liefern (z. B. ``saetze_mit_offset`` eines eingefrorenen Quellstands).
    """
    if mode not in ("sentence", "neighbors", "paragraph", "coherent"):
        raise ValueError(f"Unbekannter Kontextmodus: {mode!r}")
    if max_chars < 1 or min_focus_chars < 1 or neighbor_radius < 0:
        raise ValueError("Zeichenbudgets muessen positiv, neighbor_radius nichtnegativ sein")
    if sentence_splitter is None:
        from .muster import saetze_mit_offset
        sentence_splitter = _coherent_sentences if mode == "coherent" else saetze_mit_offset

    regions = ([(0, len(text))] if mode == "sentence" or cross_paragraphs
               else _paragraphs(text, coherent=mode == "coherent"))
    out: list[ContextQuery] = []
    for region_start, region_end in regions:
        sentences = sentence_splitter(text[region_start:region_end])
        if mode == "coherent":
            sentences = _coherent_foci(text[region_start:region_end], sentences)
        for i, sentence in enumerate(sentences):
            raw_start, raw_end = region_start + sentence.start, region_start + sentence.end
            if not (0 <= sentence.start <= sentence.end <= region_end - region_start) \
                    or text[raw_start:raw_end] != sentence.text:
                raise ValueError("Satzsplitter liefert keinen offsettreuen Originaltext")
            focus_start, focus_end = _trim_span(text, raw_start, raw_end)
            focus = text[focus_start:focus_end]
            if len(focus) < min_focus_chars or not any(char.isalpha() for char in focus):
                continue
            if mode == "sentence":
                out.append(ContextQuery(focus_start, focus_end, focus,
                                        sentence.text, raw_start, raw_end, mode))
                continue
            if mode in ("paragraph", "coherent"):
                start, end = region_start, region_end
            else:
                first = sentences[max(0, i - neighbor_radius)]
                last = sentences[min(len(sentences) - 1, i + neighbor_radius)]
                start, end = region_start + first.start, region_start + last.end
            desired_start, desired_end = _trim_span(text, start, end)
            truncated = desired_end - desired_start > max_chars
            start, end = _bounded_span(
                desired_start, desired_end, focus_start, focus_end, max_chars)
            start, end = _trim_span(text, start, end)
            if start == end:
                # Ein langer PDF-Leerraum im Fokuszentrum darf keinen leeren
                # Encoderinput erzeugen. Den naechsten Original-Token waehlen.
                center = (focus_start + focus_end) / 2
                token = min(re.finditer(r"\S+", focus), key=lambda match: abs(
                    focus_start + (match.start() + match.end()) / 2 - center))
                start, end = _bounded_span(
                    desired_start, desired_end, focus_start + token.start(),
                    focus_start + token.end(), max_chars)
                start, end = _trim_span(text, start, end)
            out.append(ContextQuery(
                focus_start, focus_end, focus, text[start:end], start, end, mode,
                context_truncated=truncated,
                focus_truncated=not (start <= focus_start and end >= focus_end),
            ))
    return out
