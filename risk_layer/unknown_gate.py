"""„Unknown als Risiko" – Gate & deterministischer Proposer (Erweiterung §3).

Default-Umkehr: nicht „kenne ich nicht → Schweigen", sondern „kann ich nicht
sauber auflösen → Risiko". Das Gate trennt *juristisch aufgeladene* Unbekannte
von belanglosen Alltagswörtern – sonst ertrinkt man in Gelb.

Die Signale werden KOMBINIERT (nie auf eines allein verlassen, /LF270–310/):
  * Norm-Nähe          – Stelle steht im Kontext einer konkreten Norm (§/Art.)
  * Definitorische Form – Nominalkompositum + Auslöser („im Sinne des", „gilt als" …)
  * Schicht-2-Urteil    – optionales LLM-Signal (übermarkierend, danach gegated)
  * Negativfilter       – Stoppwort / Named Entity ⇒ hartes Veto (Fakt, kein Begriff)
  * Schwellenlogik      – kombinierter Score über Schwelle ⇒ unknown_risiko

Bewusst OHNE LLM lauffähig: die ersten beiden + Negativfilter sind reiner Code;
das LLM-Signal ist additiv und optional. Recall-first: im Zweifel eskalieren
(False Positive = ein Beraterblick). Span-Verifikation (/LF120/) gilt unverändert.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lawmap import DEFAULT_LAWMAP
from .models import Risikomarkierung, Status
from .zitate import finde_normzitate

# Gesetzes-/Gerichtskürzel sind keine Rechtsbegriffe, sondern Verweise (Veto)
_ABKUERZUNGEN = set(DEFAULT_LAWMAP.alle_aliase()) | {
    "BFH", "BGH", "EuGH", "EuG", "BVerfG", "BVerwG", "FG", "BMF",
    "UStAE", "AEAO", "KStR", "GewStR", "EStR", "BStBl", "DStR",
}

# Fenstergröße (Zeichen) für Kontextsignale links/rechts der Stelle
_FENSTER = 140

# Definitorische Auslöser (/LF280/)
_DEFINITORISCH = re.compile(
    r"\b(?:im\s+Sinne\s+(?:des|der|von)|i\.?\s?S\.?\s?[dv]\.?|gilt(?:en)?\s+als|"
    r"ist\s+anzunehmen,?\s+wenn|setzt\s+voraus,?\s+dass|liegt\s+vor,?\s+wenn|"
    r"gelten\s+als|ist\s+gegeben,?\s+wenn|bedarf\s+(?:einer|der)|"
    r"erfordert|voraussetzung)\b",
    re.IGNORECASE,
)

# § / Art.-Marker grob, danach via Zitat-Parser bestätigt (/LF270/)
_NORM_MARKER = re.compile(r"§|\bArt\.|\bArtikel\b")

# Nominalkompositum / juristische Wortbildung (Endungen) (/LF280/)
_NOMINAL_ENDUNG = re.compile(r"(?:ung|keit|heit|barkeit|schaft|nis|tum|werb|gliederung)$")

# Firmen-/Entity-Suffixe ⇒ Named Entity, kein Rechtsbegriff (/LF300/)
_ENTITY_SUFFIX = re.compile(
    r"\b(?:GmbH|AG|KG|KGaA|SE|UG|OHG|GbR|e\.?\s?V\.?|mbH|Co\.?\s?KG|Ltd|Inc)\b"
)

# kleine deutsche Stoppwort-/Alltagsliste (Negativfilter, /LF300/)
_STOPP = {
    "der", "die", "das", "und", "oder", "ist", "sind", "war", "wird", "werden",
    "ein", "eine", "einen", "einem", "einer", "auch", "nicht", "noch", "schon",
    "hier", "dort", "dann", "wenn", "weil", "dass", "für", "mit", "ohne", "über",
    "unter", "nach", "vor", "bei", "zum", "zur", "den", "dem", "des", "frage",
    "sachverhalt", "gesellschaft", "unternehmen", "jahr", "monat", "april",
    "februar", "herbst", "investor", "tranche",
    # Seed-Alltagslexikon (generische Nomen, kein Rechtsbegriff)
    "wirtschaftsjahr", "immobilie", "immobilien", "folge", "geschäftsführer",
    "gesellschaften", "anteile", "beginn", "geltungsdauer",
    # Satzanfangs-Adverbien / Konnektoren (großgeschrieben, kein Begriff)
    "schließlich", "außerdem", "insoweit", "ferner", "zudem", "fraglich",
    "sinne", "besitz", "parallel", "hier", "damit", "zwischen",
}


def ist_belanglos(span: str) -> bool:
    """Negativfilter (/LF300/): Stoppwort / Named Entity / Gesetzeskürzel ⇒ verwerfen.

    Hartes Veto, gemeinsam genutzt von Gate (deterministisch) und LLM-Pfad.
    """
    span = span.strip()
    if not span or not any(c.isalpha() for c in span):
        return True
    kern = span.split()[-1] if span.split() else span
    return (
        span.lower() in _STOPP
        or kern.lower() in _STOPP
        or span in _ABKUERZUNGEN
        or kern in _ABKUERZUNGEN
        or bool(_ENTITY_SUFFIX.search(span))
    )


@dataclass
class GateErgebnis:
    angenommen: bool
    score: float
    status: Status
    signale: dict[str, float | bool] = field(default_factory=dict)
    norm_anker: list[str] = field(default_factory=list)


class UnknownGate:
    """Kombiniert die Signale zu einem Score und entscheidet (recall-first)."""

    def __init__(self, schwelle: float = 0.4):
        self.schwelle = schwelle
        # Gewichte – Norm-Nähe am stärksten (Spez. „stark gewichtet")
        self.w_norm = 0.40
        self.w_definitorisch = 0.35
        self.w_nominal = 0.20
        # LLM ist ein starkes Signal: ein hochsicherer Kandidat überschreitet die
        # Schwelle auch allein (Schicht 2 bewusst übermarkierend, /LF130/).
        self.w_llm = 0.45

    def _kontext(self, text: str, start: int, end: int) -> str:
        return text[max(0, start - _FENSTER): min(len(text), end + _FENSTER)]

    def bewerte(
        self, text: str, span_text: str, offset: int, *, llm_signal: float = 0.0
    ) -> GateErgebnis:
        """Bewertet eine (bereits span-verifizierte) Stelle."""
        span = span_text.strip()
        signale: dict[str, float | bool] = {}

        # Negativfilter (hartes Veto) -------------------------------------
        if ist_belanglos(span):
            return GateErgebnis(False, 0.0, Status.VERWORFEN, {"veto": True})
        kern = span.split()[-1] if span.split() else span

        kontext = self._kontext(text, offset, offset + len(span_text))
        score = 0.0

        # Norm-Nähe: nur bei vorhandenem § / Art.-Marker den (teuren) Zitat-Scanner
        # bemühen – EINMAL berechnen und für Score UND Norm-Anker wiederverwenden.
        norm_zitate = finde_normzitate(kontext) if _NORM_MARKER.search(kontext) else []
        norm = any(a.norm_id for a in norm_zitate)
        signale["norm_naehe"] = norm
        if norm:
            score += self.w_norm

        # definitorische Auslöser nur LOKAL um die Stelle (Term folgt/umgibt Trigger)
        lokal = text[max(0, offset - 80): min(len(text), offset + len(span_text) + 80)]
        definitorisch = bool(_DEFINITORISCH.search(lokal))
        signale["definitorische_form"] = definitorisch
        if definitorisch:
            score += self.w_definitorisch

        # Nominalkompositum: juristische Wortbildung am Kopfnomen oder langes Kompositum
        nominal = bool(_NOMINAL_ENDUNG.search(kern)) or len(kern) >= 14
        signale["nominal_kompositum"] = nominal
        if nominal:
            score += self.w_nominal

        if llm_signal > 0:
            signale["llm_signal"] = round(llm_signal, 3)
            score += self.w_llm * min(1.0, llm_signal)

        score = min(1.0, score)
        angenommen = score >= self.schwelle
        status = Status.UNKNOWN_RISIKO if angenommen else Status.VERWORFEN
        # Norm-Anker aus den bereits gescannten Zitaten mitführen (für Definition/Graph)
        anker = [a.norm_id.to_id() for a in norm_zitate if a.norm_id][:3]
        return GateErgebnis(angenommen, score, status, dict(signale), anker)


# ---------------------------------------------------------------------------
# Deterministischer Proposer (offline „Schicht-2-lite")
# ---------------------------------------------------------------------------

# Kandidatenform: groß beginnendes (zusammengesetztes) Nomen oder Nominalphrase,
# ggf. mit vorangehendem klein-/großgeschriebenem Adjektiv.
_KANDIDAT_RE = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ]{3,}"              # (Kompositum-)Nomen, mind. 4 Zeichen
    r"(?:[ \t]+[A-ZÄÖÜ][a-zäöüß]{3,})?"           # optionales zweites Nomen (nicht über Zeilenumbruch)
)


class DefinitorischProposer:
    """Schlägt offline Kandidatenspans vor, die dann das Gate durchlaufen.

    Ohne LLM: enumeriert nominale Formen; die eigentliche Trennung leistet das
    Gate (Norm-Nähe / definitorische Form / Negativfilter / Schwelle). Stellen,
    die bereits durch Schicht 1 erkannt sind, werden ausgelassen.
    """

    def __init__(self, gate: UnknownGate | None = None):
        self.gate = gate or UnknownGate()

    def vorschlagen(
        self,
        text: str,
        *,
        belegte_spans: list[tuple[int, int]] | None = None,
        llm_signale: dict[str, float] | None = None,
        zusatz_spans: list[tuple[str, int]] | None = None,
    ) -> list[Risikomarkierung]:
        belegt = belegte_spans or []
        llm_signale = llm_signale or {}
        risiken: list[Risikomarkierung] = []
        gesehen: set[tuple[int, int]] = set()

        # Kandidatenstellen: deterministische Nominalformen + (optional) LLM-Spans
        stellen: list[tuple[int, int, str]] = [
            (m.start(), m.end(), text[m.start():m.end()]) for m in _KANDIDAT_RE.finditer(text)
        ]
        for span_text, off in (zusatz_spans or []):
            if 0 <= off and text[off:off + len(span_text)] == span_text:
                stellen.append((off, off + len(span_text), span_text))

        for s, e, span_text in stellen:
            if (s, e) in gesehen:
                continue
            if any(not (e <= bs or s >= be) for bs, be in belegt):
                continue
            erg = self.gate.bewerte(text, span_text, s, llm_signal=llm_signale.get(span_text, 0.0))
            if not erg.angenommen:
                continue
            gesehen.add((s, e))
            risiken.append(
                Risikomarkierung(
                    start=s, end=e, matched_text=span_text, status=Status.UNKNOWN_RISIKO,
                    titel=span_text, score=erg.score, signale=erg.signale,
                    schadensintensitaet="mittel", norm_anker=erg.norm_anker,
                )
            )
        risiken.sort(key=lambda r: (r.start, r.end))
        return _entferne_enthaltene_risiken(risiken)


def _entferne_enthaltene_risiken(risiken: list[Risikomarkierung]) -> list[Risikomarkierung]:
    """Behält bei Überlappung die längere (maximale) Stelle."""
    verwerfen: set[int] = set()
    for i, a in enumerate(risiken):
        for j, b in enumerate(risiken):
            if i == j:
                continue
            if a.start <= b.start and b.end <= a.end and (a.end - a.start) > (b.end - b.start):
                verwerfen.add(j)
    return [r for k, r in enumerate(risiken) if k not in verwerfen]
