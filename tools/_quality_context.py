"""Reine Kontextvarianten fuer Retrieval-Experimente, ohne harte Marken zu aendern.

Fokus und Suchtext bleiben getrennt: Ein Modell darf Nachbarsaetze sehen, seine
Vorschlaege werden weiterhin dem urspruenglichen Satzspan zugeordnet. Dieser
Span ist kein Beweis, dass jede vorgeschlagene Norm gerade dort einschlaegig ist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
from typing import Any, Literal


ContextMode = Literal["sentence", "neighbors", "paragraph"]
_BLANK_PARAGRAPH = re.compile(r"\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*")


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


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    raw = text[start:end]
    left = start + len(raw) - len(raw.lstrip())
    return left, max(left, end - len(raw) + len(raw.rstrip()))


def _paragraphs(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for separator in _BLANK_PARAGRAPH.finditer(text):
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
    Beide Kontextmodi begrenzen ihren Suchtext auf ``max_chars``. Ein passender
    Fokus bleibt vollstaendig darin; ist schon er laenger, kennzeichnet
    ``focus_truncated`` den zentrierten Ausschnitt. Die Fokusoffsets bleiben
    auch dann vollstaendig und unveraendert. Beschnitt erfolgt nach Zeichen.

    Ohne ``cross_paragraphs`` werden Kontextmodi an Leerzeilen getrennt, auch
    bei CRLF. Ein einzelner Zeilenumbruch begrenzt den Kontext nicht. Mit dem
    expliziten Schalter darf Nachbarschaft absatzuebergreifend sein; im Modus
    ``paragraph`` wird dann das ganze Dokument zum begrenzten Kontextbereich.
    Ein injizierter Splitter muss ``text/start/end`` relativ zu seinem Argument
    liefern (z. B. ``saetze_mit_offset`` eines eingefrorenen Quellstands).
    """
    if mode not in ("sentence", "neighbors", "paragraph"):
        raise ValueError(f"Unbekannter Kontextmodus: {mode!r}")
    if max_chars < 1 or min_focus_chars < 1 or neighbor_radius < 0:
        raise ValueError("Zeichenbudgets muessen positiv, neighbor_radius nichtnegativ sein")
    if sentence_splitter is None:
        from risk_layer.muster import saetze_mit_offset
        sentence_splitter = saetze_mit_offset

    regions = ([(0, len(text))] if mode == "sentence" or cross_paragraphs
               else _paragraphs(text))
    out: list[ContextQuery] = []
    for region_start, region_end in regions:
        sentences = sentence_splitter(text[region_start:region_end])
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
            if mode == "paragraph":
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


_SOURCE_WORKFLOW = re.compile(
    r"^(AO|ESt|KSt|USt|GewSt)(?:[-–—](?:I|II))?[-–—]Klausur(?=\s*(?:,|\(|$))",
    re.IGNORECASE,
)
_WORKFLOW_LAWS = {
    "ao": ("AO",),
    "est": ("EStG", "EStDV", "AO"),
    "kst": ("KStG", "KStDV", "AO"),
    "ust": ("UStG", "UStDV", "AO"),
    "gewst": ("GewStG", "GewStDV", "AO"),
}


def source_workflow_laws(case_source: str) -> tuple[str, ...] | None:
    """Experimenteller Suchraum aus dem expliziten Klausur-Quelldokumenttyp.

    Nutzt ausschliesslich den Titelpraefix der gelieferten Herkunftsangabe,
    niemals Query, Gold-Ziele, Fall-ID, Stratum oder Loesungskommentare. Das
    Hauptgesetz, seine gleichnamige DV und AO bilden eine feste Suchheuristik;
    diese Liste behauptet keine fachliche Vollstaendigkeit des Rechtsraums.
    Unbekannte Quellen liefern None (= unbeschraenkt). Ein Ergebnis damit ist
    als ``workflow_scope`` mit bekanntem Quelldokumenttyp separat auszuweisen,
    nicht als automatisch geroutete Qualitaet fuer unbekannte Dokumente.
    """
    match = _SOURCE_WORKFLOW.match(case_source.strip())
    return _WORKFLOW_LAWS[match.group(1).lower()] if match else None
