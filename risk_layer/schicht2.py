"""Schicht 2 – LLM-Kandidatenerkennung (Ausbaustufe 1) /LF90/–/LF140/.

Das LLM darf in dieser Architektur ausschließlich *zeigen* („dieser Begriff
könnte definitionsbedürftig sein"), nie *definieren* oder *entscheiden*. Zwei
Code-Gates erzwingen das – beide sind hier bereits implementiert und ohne GPU
testbar, sodass der Sicherheitsvertrag steht, bevor ein Modell angebunden wird:

  1. Constrained Decoding (/LF110/): JSON-Schema bzw. GBNF-Grammatik, die die
     Ausgabe strukturell nicht verlassen kann.
  2. Span-Verifikation (Pflicht-Gate, /LF120/): Jede zurückgegebene Spanne wird
     gegen den Quelltext geprüft. Nicht verifizierbare Spannen werden VERWORFEN.
     Damit kann eine Halluzination höchstens einen Fehlalarm erzeugen – niemals
     einen erfundenen Rechtssatz.

Die eigentliche Inferenz steckt hinter ``KandidatenProvider`` (Protokoll). Der
deterministische MVP nutzt den ``NullProvider`` (kein Modell). Ein llama.cpp-
Provider ist als optionaler, klar markierter Stub vorgesehen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# --- /LF110/ Strukturzwang -------------------------------------------------

# JSON-Schema des erzwungenen Schicht-2-Ausgabeformats (vgl. Lastenheft 6.2).
KANDIDATEN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kandidaten"],
    "properties": {
        "kandidaten": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["span_text", "char_offset", "begruendung_kurz", "konfidenz"],
                "properties": {
                    "span_text": {"type": "string", "minLength": 1},
                    "char_offset": {"type": "integer", "minimum": 0},
                    "begruendung_kurz": {"type": "string"},
                    "konfidenz": {"type": "string", "enum": ["niedrig", "mittel", "hoch"]},
                    # optionaler Norm-VORSCHLAG (Einordnung, keine Subsumtion); der
                    # Graph prüft nur, ob es den Paragraphen real gibt – kein festes Mapping.
                    "norm_vorschlag": {"type": "string"},
                },
            },
        }
    },
}

# Äquivalente GBNF-Grammatik für llama.cpp (alternativ zum JSON-Schema, /LF110/).
KANDIDATEN_GBNF = r'''
root        ::= "{" ws "\"kandidaten\"" ws ":" ws "[" ws (kandidat (ws "," ws kandidat)*)? ws "]" ws "}"
kandidat    ::= "{" ws
                "\"span_text\"" ws ":" ws string ws "," ws
                "\"char_offset\"" ws ":" ws integer ws "," ws
                "\"begruendung_kurz\"" ws ":" ws string ws "," ws
                "\"konfidenz\"" ws ":" ws konfidenz ws
              "}"
konfidenz   ::= "\"niedrig\"" | "\"mittel\"" | "\"hoch\""
string      ::= "\"" ([^"\\] | "\\" .)* "\""
integer     ::= [0-9]+
ws          ::= [ \t\n]*
'''

# Systemprompt-Rahmen (/LF100/): strikt begrenzte Aufgabe.
SYSTEMPROMPT = (
    "Du bist ein Hinweisgeber für Steuerberater. Aufgabe: markiere in einem deutschen "
    "steuerrechtlichen Text Stellen, die einen UNBESTIMMTEN RECHTSBEGRIFF ausdrücken – "
    "auch wenn er nur UMSCHRIEBEN ist.\n"
    "Worked Example – Textstelle: eine Firma im Niedrigsteuerland ohne eigenes "
    "Personal, die nur Zahlungen weiterleitet. Erwartetes JSON-Element: "
    "{span_text: 'eine Firma im Niedrigsteuerland ohne eigenes Personal', "
    "begruendung_kurz: 'Gestaltungsmissbrauch', konfidenz: 'hoch', norm_vorschlag: '§ 42 AO'}.\n"
    "Regeln:\n"
    "(1) span_text MUSS wörtlich und zusammenhängend im Text stehen (exakt kopieren).\n"
    "(2) Erfinde KEINE Definitionen/Rechtsfolgen; subsumiere nicht.\n"
    "(3) Keine Alltagswörter (Sachverhalt, Anteile, Stimmen, Gesellschaft, Jahr) und "
    "nichts aus der Liste bereits erkannter Begriffe.\n"
    "(4) begruendung_kurz = das einschlägige Rechtskonzept in höchstens 6 Wörtern.\n"
    "(5) norm_vorschlag: WÄHLE die Norm möglichst aus der unten genannten "
    "Kandidatenliste (Trigger-Index) – das ist eine kuratierte Vorauswahl. GENAU "
    "EINE konkrete Norm (z. B. '§ 146 AO'), kein Bereich. Nur wenn KEINE passt, "
    "nenne eine eigene; bist du unsicher, lass das Feld weg. NICHT frei raten.\n"
    "(6) Höchstens 8 Treffer. Strikt nach JSON-Schema.\n"
    "Achte besonders auf häufige Auslöser (auch wenn nur umschrieben): "
    "Buchführungs-/Aufzeichnungs-/Kassenmängel (Schätzungsanlass), nachträglich "
    "bekannt gewordene Tatsachen, künstliche/funktionslose Gestaltungen, "
    "Haftungsauslöser (Steuern trotz Mitteln nicht abgeführt), abweichende "
    "wirtschaftliche Zurechnung. Markiere GROSSZÜGIG und EINZELN: jede umschriebene "
    "Stelle als eigenen Treffer, lieber mehr als weniger (Schicht 2 ist bewusst "
    "übermarkierend; das nachgelagerte Gate filtert)."
)


@dataclass(frozen=True)
class Kandidat:
    """Ein (verifizierter) Schicht-2-Kandidat außerhalb des Katalogs."""

    span_text: str
    char_offset: int
    begruendung_kurz: str
    konfidenz: str
    norm_vorschlag: str = ""        # optionaler LLM-Einordnungsvorschlag (z. B. „§ 42 AO")

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_text": self.span_text,
            "char_offset": self.char_offset,
            "begruendung_kurz": self.begruendung_kurz,
            "konfidenz": self.konfidenz,
            "norm_vorschlag": self.norm_vorschlag,
            "status": "kandidat",
        }


# --- /LF120/ Span-Verifikation (Pflicht-Gate) ------------------------------

def verifiziere_span(text: str, span_text: str, char_offset: int) -> bool:
    """Pflichtprüfung: ``text[offset : offset+len(span_text)] == span_text``.

    Reiner Code, kein Modell. Schlägt die Prüfung fehl ⇒ Kandidat wird verworfen.
    """
    if char_offset < 0 or not span_text:
        return False
    return text[char_offset : char_offset + len(span_text)] == span_text


def verifiziere_kandidaten(text: str, roh: list[dict[str, Any]]) -> list[Kandidat]:
    """Filtert LLM-Rohkandidaten auf wörtlich im Text belegbare Spannen (/LF120/).

    Nicht verifizierbare Einträge werden stillschweigend verworfen – sie können
    damit nur einen Fehlalarm verhindern, nie einen erfundenen Rechtssatz tragen.
    """
    verifiziert: list[Kandidat] = []
    for k in roh:
        try:
            span = str(k["span_text"])
            offset = int(k["char_offset"])
            konf = str(k.get("konfidenz", "niedrig"))
        except (KeyError, TypeError, ValueError):
            continue
        if konf not in ("niedrig", "mittel", "hoch"):
            konf = "niedrig"
        if verifiziere_span(text, span, offset):
            verifiziert.append(
                Kandidat(
                    span_text=span,
                    char_offset=offset,
                    begruendung_kurz=str(k.get("begruendung_kurz", "")),
                    konfidenz=konf,
                    norm_vorschlag=str(k.get("norm_vorschlag", "")).strip(),
                )
            )
    return verifiziert


# --- Provider-Schnittstelle -------------------------------------------------

@runtime_checkable
class KandidatenProvider(Protocol):
    """Liefert Roh-Kandidaten für ein Segment (vor Span-Verifikation)."""

    def kandidaten(self, segment_text: str, bekannte_begriffe: list[str],
                   kontext_normen: list[str] | None = ...,
                   abweichung: bool = ...) -> list[dict[str, Any]]:
        ...


class NullProvider:
    """Default für reine Schicht 1 (MVP): kein Modell, keine Kandidaten."""

    def kandidaten(self, segment_text: str, bekannte_begriffe: list[str],
                   kontext_normen: list[str] | None = None,
                   abweichung: bool = False) -> list[dict[str, Any]]:
        return []


def erkenne_kandidaten(
    text: str,
    bekannte_begriffe: list[str],
    provider: KandidatenProvider | None = None,
    kontext_normen: list[str] | None = None,
    abweichung: bool = False,
) -> list[Kandidat]:
    """Schicht-2-Lauf mit zwingender Span-Verifikation.

    ``kontext_normen`` erdet das Modell (im Text/Korpus reale Normen) – das ist
    KEINE Begriffsdefinition, nur eine Auswahlhilfe für ``norm_vorschlag``.
    ``abweichung`` erlaubt dem Prompt, eine vom Katalog abweichende Norm zu nennen.
    """
    provider = provider or NullProvider()
    roh = provider.kandidaten(text, bekannte_begriffe, kontext_normen, abweichung)
    return verifiziere_kandidaten(text, roh)
