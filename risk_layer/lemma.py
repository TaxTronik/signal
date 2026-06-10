"""Lemma-Robustheitspfad /LF50/ – offset-treue Lemma-Sicht (optional, spaCy).

Schicht 1 matcht wörtlich (mit begrenzter Endungs-Toleranz) und verpasst
stärker flektierte Formen (Umlaut-Plurale „Ergebnisabführungsverträge",
zu-Infinitive „einzugliedern", Artikel-Kasus „Haftung DER Geschäftsführer").
Dieses Modul liefert dafür eine LEMMA-SICHT des Textes: alle Token-Lemmata,
mit Einzel-Leerzeichen verbunden, plus die exakte Abbildung jeder Span im
Lemma-Text zurück auf die ORIGINAL-Zeichenspan (Offset-Treue ist heilig –
jede Marke zeigt am Ende wörtlich auf den Originaltext, /LF20/).

Degradations-Philosophie wie die NER-Schicht in anonymize.py:
  * Lazy-Load mit prozessweitem Modul-Cache (lg → sm-Fallback, einmaliges
    Warning mit Install-Hinweis),
  * ohne spaCy/Modell liefert ``lemma_sicht`` None – der Aufrufer verhält
    sich dann EXAKT wie heute (kein Lemma-Pfad, kein Crash),
  * hart abschaltbar per RISK_LAYER_LEMMA=aus (auch off/none/0): für
    Determinismus-kritische Läufe wird das Modell nicht einmal geladen.

Hinweis Provenienz: die Lemmatisierung ist modellgestützt (statistischer
EditTree-Lemmatizer) – für fixe Modellversion deterministisch, aber ein
Modellwechsel kann Lemmata ändern. Treffer aus diesem Pfad werden deshalb
gesondert gekennzeichnet (Treffer.via_lemma, Begriffskarte.via="lemma").
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("risk_layer.lemma")

_UNGELADEN = object()    # Sentinel: Laden wurde noch nie versucht
_NLP: Any = _UNGELADEN   # prozessweiter Modul-Cache (None = nicht verfügbar)

# Reihenfolge = Präferenz: lg (beste Lemma-Güte, /LF50/) vor sm (Notbehelf).
_LEMMA_MODELLE = ("de_core_news_lg", "de_core_news_sm")
_LEMMA_INSTALL_HINWEIS = ("Lemma-Pfad (/LF50/) inaktiv — pip install spacy + "
                          "python -m spacy download de_core_news_lg")
# Nur was die Lemmatisierung braucht: der EditTree-Lemmatizer der de_core_news-
# Modelle hängt an tok2vec/tagger/morphologizer/attribute_ruler – parser und
# ner sind dafür unnötig und werden fürs Tempo deaktiviert (geprüft: Lemmata
# sind mit und ohne parser/ner identisch).
_DISABLE = ("parser", "ner")


def _lemma_deaktiviert() -> bool:
    """RISK_LAYER_LEMMA=aus erzwingt den rein wörtlichen Pfad.

    Wie RISK_LAYER_NER in anonymize.py: Determinismus-kritische Läufe müssen
    die modellgestützte Schicht hart abschalten können, OHNE dass das Modell
    überhaupt geladen wird. Akzeptierte Werte: aus/off/none/0.
    """
    return os.environ.get("RISK_LAYER_LEMMA", "").strip().lower() in ("aus", "off", "none", "0")


def _lade_nlp() -> Any:
    """Lazy-Load des deutschen spaCy-Modells mit prozessweitem Modul-Cache.

    Muster ist ``anonymize._lade_nlp`` (eigener Cache: die NER-Schicht lädt
    das VOLLE Modell, dieser Pfad nur die Lemma-Komponenten – die Pipelines
    sind verschieden konfiguriert und dürfen sich nicht teilen). Rückgabe:
    nlp-Objekt oder None (= Lemma-Pfad inaktiv). Der Install-Hinweis wird dank
    Cache nur EINMAL pro Prozess geloggt. Tests injizieren über genau diesen
    Seam ein Fake-NLP (tests/conftest.py patcht ihn autouse auf None).
    """
    global _NLP
    if _NLP is not _UNGELADEN:
        return _NLP
    try:
        import spacy  # lazy: optionales Extra
    except ImportError:
        _log.warning(_LEMMA_INSTALL_HINWEIS)
        _NLP = None
        return None
    for modell in _LEMMA_MODELLE:
        try:
            nlp = spacy.load(modell, disable=list(_DISABLE))
        except OSError:
            continue    # Modell nicht installiert → nächste Stufe probieren
        except Exception as exc:  # noqa: BLE001
            # Defensiv: eine defekte Installation darf Schicht 1 nicht crashen –
            # der wörtliche Pfad bleibt als Mindestverhalten bestehen.
            _log.warning("spaCy-Modell %s nicht ladbar (%s: %s) — versuche Fallback",
                         modell, type(exc).__name__, exc)
            continue
        if modell != _LEMMA_MODELLE[0]:
            _log.warning("de_core_news_lg fehlt — nutze %s (geringere Lemma-Güte). "
                         "Abhilfe: python -m spacy download de_core_news_lg", modell)
        _NLP = nlp
        return nlp
    _log.warning(_LEMMA_INSTALL_HINWEIS)
    _NLP = None
    return None


def _token_lemma(tok: Any) -> str:
    """Lemma eines Tokens – mit Schutz vor Lemmatizer-Artefakten.

    Der deutsche Lemmatizer liefert für Interpunktion das Artefakt „--";
    Tokens ohne brauchbares Lemma (leer oder ohne ein einziges alphanumerisches
    Zeichen) behalten ihren Originaltext. Bewusst KEIN ``tok.is_alpha``:
    Test-Fakes müssen nur text/idx/lemma_ tragen (Seam-Vertrag).
    """
    lemma = (getattr(tok, "lemma_", "") or "").strip()
    if lemma and any(ch.isalnum() for ch in lemma):
        return lemma
    return tok.text


@dataclass(frozen=True)
class LemmaSicht:
    """Lemma-Text + Token-Spans in BEIDEN Koordinatensystemen.

    ``lemma_text``: alle Token-Lemmata, mit Einzel-Leerzeichen verbunden.
    ``lemma_spans[i]``/``quell_spans[i]``: Span des i-ten Tokens im lemma_text
    bzw. im analysierten QUELLTEXT (token.idx, token.idx+len(token.text)) –
    die Brücke, über die Lemma-Treffer offset-treu zurückwandern.
    """

    lemma_text: str
    lemma_spans: tuple[tuple[int, int], ...]
    quell_spans: tuple[tuple[int, int], ...]

    def rueckabbildung(self, start: int, ende: int) -> tuple[int, int]:
        """Bildet eine [start, ende)-Span im lemma_text auf den Quelltext ab.

        Rückgabe ist die UMSCHLIESSENDE Quellspan: vom ersten bis zum letzten
        Token, dessen Lemma-Span die Eingabespan berührt. Eine Span ohne
        Token-Berührung ist ein Programmierfehler (Treffer entstehen nur über
        Wortzeichen) → ValueError statt stiller Falsch-Offsets.
        """
        erster: int | None = None
        letzter: int | None = None
        for i, (ls, le) in enumerate(self.lemma_spans):
            if le > start and ls < ende:
                if erster is None:
                    erster = i
                letzter = i
        if erster is None or letzter is None:
            raise ValueError(f"Lemma-Span [{start}, {ende}) berührt kein Token")
        return (self.quell_spans[erster][0], self.quell_spans[letzter][1])


def lemma_sicht(text: str) -> LemmaSicht | None:
    """Lemmatisiert ``text`` offset-treu – oder None (Degradation/Env-aus).

    None ⇒ der Aufrufer bleibt beim exakt heutigen Verhalten (kein Lemma-Pfad).
    Leerraum-Tokens werden übersprungen; jede Position im lemma_text gehört
    damit eindeutig zu genau einem Token oder einem Einzel-Leerzeichen-Fugenraum.
    """
    if not text or not text.strip():
        return None
    # Env-Check VOR dem Loader: RISK_LAYER_LEMMA=aus darf das Modell gar nicht
    # erst laden (Determinismus + Startzeit; Muster anonymize._ner_pass).
    if _lemma_deaktiviert():
        return None
    nlp = _lade_nlp()
    if nlp is None:
        return None
    try:
        doc = nlp(text)
    except Exception as exc:  # noqa: BLE001
        # Fail-degraded statt Crash: ein kaputtes Modell darf Schicht 1 nicht
        # legen – der wörtliche Pfad lief bereits/läuft unabhängig weiter.
        _log.warning("Lemmatisierung fehlgeschlagen (%s: %s) — Lemma-Pfad inaktiv",
                     type(exc).__name__, exc)
        return None
    teile: list[str] = []
    lemma_spans: list[tuple[int, int]] = []
    quell_spans: list[tuple[int, int]] = []
    pos = 0
    for tok in doc:
        roh = tok.text
        if not roh.strip():
            continue    # reine Leerraum-Tokens tragen weder Lemma noch Treffer
        lemma = _token_lemma(tok)
        if teile:
            pos += 1    # Einzel-Leerzeichen-Fuge
        teile.append(lemma)
        lemma_spans.append((pos, pos + len(lemma)))
        quell_spans.append((tok.idx, tok.idx + len(roh)))
        pos += len(lemma)
    if not teile:
        return None
    return LemmaSicht(lemma_text=" ".join(teile),
                      lemma_spans=tuple(lemma_spans),
                      quell_spans=tuple(quell_spans))


def lemma_form(text: str) -> str | None:
    """Lemma-Normalform einer kurzen Oberfläche (für den Katalog-Abgleich).

    Gleiches nlp, gleiche Artefakt-Behandlung wie ``lemma_sicht`` – Katalog-
    Oberflächen und Text werden damit in DASSELBE Koordinatensystem gebracht.
    None bei fehlendem spaCy/Env-aus.
    """
    sicht = lemma_sicht(text)
    return sicht.lemma_text if sicht is not None else None
