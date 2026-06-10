"""Schicht 1 – Deterministischer Katalog-Matcher /LF40/–/LF80/.

Höchste Priorität, reiner Code: identische Eingabe + identische Katalogversion
⇒ bit-identische Ausgabe (/LF80/, /NF20/). Kein Zufall, kein Modellaufruf.

Morphologische Robustheit (/LF50/) ohne spaCy:
  * je Oberflächenvariante (kanonisch + Aliase) wird ein Muster gebaut, das
    deutsche Flexionsendungen am Wortende toleriert,
  * Getrennt- und Bindestrich-Schreibung sowie Mehrfach-Leerzeichen zwischen
    den Wörtern werden über einen flexiblen Trenner abgedeckt. ECHTE
    Zusammenschreibung ("Kassennachschau" für "Kassen-Nachschau") ist bewusst
    NICHT abgedeckt: der Trenner verlangt mindestens ein Trennzeichen –
    Betriebspunkt des recall/precision-Tradeoffs; solche Varianten gehören als
    eigene Aliase in den Katalog,
  * Groß-/Kleinschreibung wird ignoriert.

Betriebspunkt recall-first (/LF60/): im Zweifel wird markiert. Der optionale
spaCy-Lemmapfad (/LF50/, ``match_lemma``) läuft ZUSÄTZLICH über eine
``LemmaSicht`` (risk_layer/lemma.py) und ändert nichts am deterministischen
Verhalten dieses Kerns: ``match`` bleibt byte-identisch; ohne spaCy/Modell ist
``match_lemma`` ein No-Op (leere Liste).
"""

from __future__ import annotations

import re

from . import lemma as lemma_mod
from .catalog import Katalog
from .models import IngestErgebnis, Katalogeintrag, Segment, Treffer

# Wortzeichen für Wortgrenzen (inkl. deutscher Umlaute/ß und Akzente).
# À-ÿ würde auch × (U+00D7) und ÷ (U+00F7) einschließen – das sind Operatoren,
# keine Buchstaben; sie werden über die Lücken D7/F7 explizit ausgenommen.
_WB = r"0-9A-Za-zÄÖÜäöüßÀ-ÖØ-öø-ÿ"
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


# --- Lemma-Pfad /LF50/: Muster über der Lemma-Sicht --------------------------
# Beide Seiten (Katalog-Oberfläche und Text) sind hier bereits LEMMATISIERT –
# die Resttoleranz gleicht keine Flexion mehr aus, sondern ARTEFAKTE des
# statistischen Lemmatizers (de_core_news_lg, empirisch belegt):
#   * Steigerungsformen behalten ein st-Suffix („unangemessenste"→"unangemessenst",
#     „offenbarsten"→"offenbarst"),
#   * Komparative behalten -er („unangemessenere"→"unangemessener"),
#   * -nis-Plurale erzeugen ss/ß-Varianten („Schätzungsbefugnisse"→
#     "Schätzungsbefugniss", „Gesellschaftsverhältnisse"→"Gesellschaftsverhältniß"),
#   * Residualflexion, wenn der Lemmatizer eine Form unverändert durchreicht.
# UMLAUT-Artefakte („gröberes"→"gröb", „Gestaltungsmissbräuche"→
# "Gestaltungsmissbräuch") werden bewusst NICHT gefaltet – Umlaut-Folding wäre
# ein Präzisionsrisiko (schwül/schwul); solche Formen bleiben dokumentierte
# Lücken des Lemma-Pfads (siehe fixtures/gold/lemma_v1.yaml).
_LEMMA_ARTEFAKTE = ("sten", "stem", "ster", "stes", "ste", "st",
                    "ssen", "sse", "ss", "en", "er", "es", "em", "e", "n", "s", "ß")
_LEMMA_FLEX_OPT = "(?:" + "|".join(_LEMMA_ARTEFAKTE) + ")?"


def _lemma_token_muster(token: str) -> str:
    """Regex-Fragment eines Lemma-Tokens: volle Form ODER Stamm + Artefakt-Suffix."""
    stamm = _stamm(token)
    kern = (re.escape(token) if stamm == token
            else f"(?:{re.escape(token)}|{re.escape(stamm)})")
    return kern + _LEMMA_FLEX_OPT


def _lemma_surface_muster(lemma_surface: str) -> str | None:
    """Vollständiges Muster einer LEMMATISIERTEN Oberfläche (für lemma_text)."""
    woerter = [w for w in re.split(r"[\s\-]+", lemma_surface.strip()) if w]
    if not woerter:
        return None
    kern = _SEP.join(_lemma_token_muster(w) for w in woerter)
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
        # Lemma-Pfad (/LF50/): die Katalog-Oberflächen werden EINMALIG mit dem
        # gleichen nlp lemmatisiert und vorkompiliert. Lazy + instanzgebunden
        # (eine Instanz = eine Katalogversion); None = noch nicht gebaut.
        self._lemma_compiled: list[tuple[Katalogeintrag, str, re.Pattern[str]]] | None = None

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

    def _lemma_muster(self) -> list[tuple[Katalogeintrag, str, re.Pattern[str]]]:
        """Lemmatisierte + vorkompilierte Katalog-Oberflächen (einmalig je Instanz).

        Aufruf NUR, wenn eine LemmaSicht vorliegt (= nlp verfügbar) – sonst
        würde ein früher Degradations-Lauf eine leere Liste festschreiben.
        Muster wie im wörtlichen Pfad (Wortgrenzen, flexibler Trenner), aber
        über der Lemma-Normalform und mit ARTEFAKT-Toleranz statt Flexions-
        Toleranz (``_lemma_surface_muster``, s. Kommentar dort).
        """
        if self._lemma_compiled is None:
            compiled: list[tuple[Katalogeintrag, str, re.Pattern[str]]] = []
            for ct in self._compiled:
                for surface, _ in ct.muster:
                    lf = lemma_mod.lemma_form(surface)
                    if not lf:
                        continue
                    pat = _lemma_surface_muster(lf)
                    if pat is None:
                        continue
                    compiled.append((ct.eintrag, surface,
                                     re.compile(pat, re.IGNORECASE | re.UNICODE)))
            self._lemma_compiled = compiled
        return self._lemma_compiled

    def match_lemma(self, ingest: IngestErgebnis) -> list[Treffer]:
        """Lemma-Pfad /LF50/: Katalogtreffer über die Lemma-Sicht des Textes.

        Läuft ZUSÄTZLICH zu ``match`` (Bestand unverändert). Die Lemma-Sicht
        entsteht über dem NORMALISIERTEN Text; Treffer-Spans wandern zweistufig
        zurück: lemma_text → normalisierte Span (``LemmaSicht.rueckabbildung``)
        → Original-Span (``offset_map``, /LF20/) – jede Marke zeigt wörtlich
        auf den ORIGINALTEXT. Ohne spaCy/Modell oder mit RISK_LAYER_LEMMA=aus:
        leere Liste (Degradation = exakt heutiges Verhalten).
        """
        sicht = lemma_mod.lemma_sicht(ingest.normalized_text)
        if sicht is None:
            return []
        omap = ingest.offset_map
        segs = ingest.segments
        treffer: list[Treffer] = []
        gesehen: set[tuple[str, int, int]] = set()
        for eintrag, surface, regex in self._lemma_muster():
            for m in regex.finditer(sicht.lemma_text):
                if m.start() >= m.end():
                    continue
                ns, ne = sicht.rueckabbildung(m.start(), m.end())
                orig_start = omap[ns]
                orig_end = omap[ne - 1] + 1
                schluessel = (eintrag.id, orig_start, orig_end)
                if schluessel in gesehen:
                    continue
                gesehen.add(schluessel)
                treffer.append(
                    Treffer(
                        begriff_id=eintrag.id,
                        start=orig_start,
                        end=orig_end,
                        matched_text=ingest.original_text[orig_start:orig_end],
                        surface=surface,
                        segment_index=_segment_fuer(orig_start, segs),
                        via_lemma=True,
                    )
                )
        treffer = _entferne_enthaltene(treffer)
        treffer.sort(key=lambda t: (t.start, t.end, t.begriff_id))
        return treffer


def entferne_lemma_dubletten(exakt: list[Treffer], lemma_treffer: list[Treffer]) -> list[Treffer]:
    """Dedupe Lemma- gegen exakte Treffer: EXAKTE Treffer gewinnen (/LF50/).

    Ein Lemma-Treffer entfällt, sobald er einen exakten Treffer DERSELBEN
    Begriffs-ID überlappt – auch bei bloßer Teil-Überlappung (anders als
    ``entferne_enthaltene_spans``, das nur echtes Enthaltensein verwirft:
    hier darf der wörtliche Beleg nie von einer Modell-Normalisierung
    überlagert werden). Treffer ANDERER Begriffe bleiben unberührt.
    """
    out: list[Treffer] = []
    for lt in lemma_treffer:
        if any(t.begriff_id == lt.begriff_id and lt.start < t.end and t.start < lt.end
               for t in exakt):
            continue
        out.append(lt)
    return out


def entferne_enthaltene_spans(items: list, *, gruppe=None) -> list:
    """Entfernt echt enthaltene Spans; der maximale Span bleibt erhalten.

    Gemeinsamer Containment-Kern für Schicht 1 (Treffer) und das Unknown-Gate
    (Risikomarkierungen) – die Elemente brauchen nur ``start``/``end``.

    ``gruppe`` ist eine optionale Schlüsselfunktion: nur Elemente DERSELBEN
    Gruppe werden gegeneinander verglichen (Schicht 1 vergleicht je Begriffs-ID;
    ``None`` ⇒ alle Spans konkurrieren). Gleich lange / identische Spans bleiben
    beide erhalten (nur ECHTES Enthaltensein verwirft).
    """
    verwerfen: set[int] = set()
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i == j:
                continue
            if gruppe is not None and gruppe(a) != gruppe(b):
                continue
            # b ist echt in a enthalten ⇒ b verwerfen
            if a.start <= b.start and b.end <= a.end and (a.end - a.start) > (b.end - b.start):
                verwerfen.add(j)
    return [x for k, x in enumerate(items) if k not in verwerfen]


def _entferne_enthaltene(treffer: list[Treffer]) -> list[Treffer]:
    """Fasst denselben Begriff am selben Ort zusammen.

    Ist die Spanne eines Treffers vollständig in der eines anderen Treffers
    DESSELBEN Begriffs enthalten (z. B. Alias „Gesamtwürdigung" ⊂ kanonisch
    „Gesamtwürdigung der tatsächlichen Verhältnisse"), bleibt nur der maximale
    Span erhalten. Kein Recall-Verlust – der Begriff bleibt markiert; nur die
    redundante Doppelmarkierung entfällt. Treffer verschiedener Begriffe (etwa
    dieselbe Oberfläche unter UStG und KStG) bleiben unberührt.
    """
    return entferne_enthaltene_spans(treffer, gruppe=lambda t: t.begriff_id)


def _segment_fuer(orig_pos: int, segments: list[Segment]) -> int:
    """Ordnet eine Originalposition dem enthaltenden Segment zu (-1 falls keins)."""
    for seg in segments:
        if seg.start <= orig_pos < seg.end:
            return seg.index
    return -1
