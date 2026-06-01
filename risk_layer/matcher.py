"""Schicht 1 – Deterministischer Katalog-Matcher /LF40/–/LF80/.

Höchste Priorität, reiner Code: identische Eingabe + identische Katalogversion
⇒ bit-identische Ausgabe (/LF80/, /NF20/). Kein Zufall, kein Modellaufruf.

Morphologische Robustheit (/LF50/) ohne spaCy:
  * je Oberflächenvariante (kanonisch + Aliase) wird ein Muster gebaut, das
    deutsche Flexionsendungen am Wortende toleriert,
  * Trenn-/Zusammenschreibung und Mehrfach-Leerzeichen zwischen den Wörtern
    werden über einen flexiblen Trenner abgedeckt,
  * Groß-/Kleinschreibung wird ignoriert.

Betriebspunkt recall-first (/LF60/): im Zweifel wird markiert. Ein optionaler
spaCy-Lemmapfad ist in ``matcher_spacy`` als Ausbaustufe vorgesehen, ändert aber
nichts am deterministischen Verhalten dieses Kerns.
"""

from __future__ import annotations

import re

from .catalog import Katalog
from .models import IngestErgebnis, Katalogeintrag, Segment, Treffer

# Wortzeichen für Wortgrenzen (inkl. deutscher Umlaute/ß und Akzente)
_WB = r"0-9A-Za-zÄÖÜäöüßÀ-ÿ"
_WB_OPEN = rf"(?<![{_WB}])"
_WB_CLOSE = rf"(?![{_WB}])"

# Trenner zwischen Wörtern: Leerraum und/oder Bindestrich (Trenn-/Zusammenschr.)
_SEP = r"[\s\-]+"

# Rein flexivische Endungen (KEINE Derivation wie -ung), längste zuerst
_FLEX_ENDUNGEN = ("en", "er", "es", "em", "e", "n", "s")
# nach dem Stamm erlaubte (optionale) Flexionsendung
_FLEX_OPT = r"(?:en|er|es|em|e|n|s)?"
_MIN_STAMM = 4  # Stamm nicht unter diese Länge kürzen (vermeidet Übergeneralisierung)


def _stamm(token: str) -> str:
    """Reduziert ein Wort auf seinen Stamm, indem eine flexivische Endung entfällt."""
    low = token.lower()
    for end in _FLEX_ENDUNGEN:
        if low.endswith(end) and len(low) - len(end) >= _MIN_STAMM:
            return token[: len(token) - len(end)]
    return token


def _token_muster(token: str) -> str:
    """Regex-Fragment für ein einzelnes Wort inkl. Flexionstoleranz."""
    return re.escape(_stamm(token)) + _FLEX_OPT


def _surface_muster(surface: str) -> str | None:
    """Baut das vollständige Regex-Muster einer Oberflächenvariante."""
    woerter = [w for w in re.split(r"[\s\-]+", surface.strip()) if w]
    if not woerter:
        return None
    kern = _SEP.join(_token_muster(w) for w in woerter)
    return f"{_WB_OPEN}{kern}{_WB_CLOSE}"


class CompiledTerm:
    """Vorkompilierte Muster eines Katalogeintrags."""

    __slots__ = ("eintrag", "muster")

    def __init__(self, eintrag: Katalogeintrag):
        self.eintrag = eintrag
        self.muster: list[tuple[str, re.Pattern[str]]] = []
        for surface in eintrag.oberflaechen:
            pat = _surface_muster(surface)
            if pat is None:
                continue
            self.muster.append((surface, re.compile(pat, re.IGNORECASE | re.UNICODE)))


class Schicht1Matcher:
    """Deterministischer Katalog-Matcher."""

    def __init__(self, katalog: Katalog):
        self.katalog = katalog
        self._compiled = [CompiledTerm(e) for e in katalog]

    def match(self, ingest: IngestErgebnis) -> list[Treffer]:
        """Findet alle Katalogtreffer im (normalisierten) Text.

        Offsets werden über ``ingest.offset_map`` exakt auf den Originaltext
        zurückgemappt (/LF20/). Die Ausgabe ist stabil sortiert (/LF80/).
        """
        text = ingest.normalized_text
        omap = ingest.offset_map
        segs = ingest.segments

        treffer: list[Treffer] = []
        gesehen: set[tuple[str, int, int]] = set()

        for ct in self._compiled:
            for surface, regex in ct.muster:
                for m in regex.finditer(text):
                    ns, ne = m.start(), m.end()
                    if ns >= ne:
                        continue
                    orig_start = omap[ns]
                    orig_end = omap[ne - 1] + 1
                    schluessel = (ct.eintrag.id, orig_start, orig_end)
                    if schluessel in gesehen:
                        continue
                    gesehen.add(schluessel)
                    treffer.append(
                        Treffer(
                            begriff_id=ct.eintrag.id,
                            start=orig_start,
                            end=orig_end,
                            matched_text=ingest.original_text[orig_start:orig_end],
                            surface=surface,
                            segment_index=_segment_fuer(orig_start, segs),
                        )
                    )

        treffer = _entferne_enthaltene(treffer)
        # stabile, deterministische Reihenfolge: Position, dann Begriffs-ID
        treffer.sort(key=lambda t: (t.start, t.end, t.begriff_id))
        return treffer


def _entferne_enthaltene(treffer: list[Treffer]) -> list[Treffer]:
    """Fasst denselben Begriff am selben Ort zusammen.

    Ist die Spanne eines Treffers vollständig in der eines anderen Treffers
    DESSELBEN Begriffs enthalten (z. B. Alias „Gesamtwürdigung" ⊂ kanonisch
    „Gesamtwürdigung der tatsächlichen Verhältnisse"), bleibt nur der maximale
    Span erhalten. Kein Recall-Verlust – der Begriff bleibt markiert; nur die
    redundante Doppelmarkierung entfällt. Treffer verschiedener Begriffe (etwa
    dieselbe Oberfläche unter UStG und KStG) bleiben unberührt.
    """
    verwerfen: set[int] = set()
    for i, a in enumerate(treffer):
        for j, b in enumerate(treffer):
            if i == j or a.begriff_id != b.begriff_id:
                continue
            # b ist echt in a enthalten ⇒ b verwerfen
            if a.start <= b.start and b.end <= a.end and (a.end - a.start) > (b.end - b.start):
                verwerfen.add(j)
    return [t for k, t in enumerate(treffer) if k not in verwerfen]


def _segment_fuer(orig_pos: int, segments: list[Segment]) -> int:
    """Ordnet eine Originalposition dem enthaltenden Segment zu (-1 falls keins)."""
    for seg in segments:
        if seg.start <= orig_pos < seg.end:
            return seg.index
    return -1
