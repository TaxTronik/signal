"""Lokale Abstraktion/Anonymisierung (§ 203 StGB) – Geheimnisschutz-Gate.

Bevor ein Sachverhalt an ein externes (Cloud-)LLM gegeben wird, MUSS er lokal
abstrahiert werden: mandantenidentifizierende Angaben (Firmen, Personen, Beträge,
Daten, Steuer-/Konto-Nummern) werden durch stabile Platzhalter ersetzt. Die
juristische Struktur bleibt erhalten – die Rechtsbegriffe selbst sind kein
Geheimnis.

Vollständig lokal. Das Ergebnis ist ein ENTWURF, den der Berater vor Freigabe
manuell prüft und anpasst. Die Rückübersetzungstabelle (``ersetzungen``)
bleibt lokal und wird NIEMALS mitexportiert.

Zulässiger Fehlermodus wie beim Risiko-Layer: lieber zu viel schwärzen
(Über-Abstraktion) als ein Geheimnis preisgeben.

ZWEISCHICHTIGES Modell:
  1. Deterministische Regexe (FÜHREND, immer aktiv): Anrede+Name, Firmen mit
     Rechtsform, Beträge mit Währungskontext, Daten, Steuer-/Kontonummern,
     IBAN, E-Mail, Telefon, Anschrift. Gleiche Eingabe ⇒ gleiches Ergebnis.
  2. Optionale NER-Schicht (spaCy ``de_core_news_lg``, Fallback ``_sm``;
     Extra ``nlp``, /LF50/): läuft NACH den Regexen und ersetzt verbliebene
     PER/ORG/LOC-Entitäten mit denselben Platzhalter-Konventionen
     ([Person X]/[Gesellschaft X]/[Ort X]); die Nummerierung läuft über den
     gemeinsamen Zähler pro Dokument weiter (gleiche Entität → gleicher
     Platzhalter). Bereits vergebene Platzhalter sind für NER tabu – der
     Regex-Pass bleibt führend. Fehlen spaCy oder das Modell, bleibt exakt
     der bisherige Regex-Pfad aktiv (einmaliges Warning pro Prozess).
     Abschaltbar per Umgebungsvariable RISK_LAYER_NER=aus (auch off/none/0)
     für Determinismus-kritische Läufe – dann wird das Modell nicht einmal
     geladen.

Bekannte GRENZEN (ehrlich dokumentiert, weil ``pruefe_rest`` in recherche.py
und das 422-Gate von /v1/katalog/definiere als DURCHSETZUNG darauf bauen):
  * Personen OHNE Anrede („Mueller kam zur Besprechung") und Firmen OHNE
    Rechtsform-Suffix („die Mustermann-Gruppe") erkennen NUR die OPTIONALE
    NER-Schicht – ohne installiertes spaCy-Modell weiterhin gar nicht.
  * NER ist statistisch: kein Recall-Versprechen. Seltene Namen, Tippfehler,
    Kleinschreibung oder ungewöhnliche Kontexte können durchrutschen; ein
    Modellwechsel kann Ergebnisse verändern (deshalb bleibt der Regex-Pass
    führend und die Schicht per RISK_LAYER_NER=aus abschaltbar).
  * Keine Koreferenz-Auflösung: „Herr Mueller" (Regex) und „Mueller" (NER)
    erhalten ZWEI verschiedene Platzhalter.
  * MISC-Entitäten (Produkte, Projekte, Nationalitäten) werden nicht ersetzt.
  * Großgeschriebene Nicht-Namen direkt vor einer Rechtsform werden mitgeschwärzt
    (Über-Abstraktion, z. B. „Unsere Beta GmbH" → komplett ersetzt) – das ist der
    zulässige Fehlermodus; nur Satzanfang-Artikel (Die/Eine …) sind ausgenommen.
  * Beträge brauchen Währungskontext (€/EUR/Euro/TEUR): nackte Zahlen („zahlte
    50000") bleiben stehen, damit Paragraphen-/Ordnungsnummern („§ 146 AO",
    „Nr. 21") nicht zerstört werden.
Die manuelle Prüfung des Entwurfs durch den Berater bleibt deshalb Pflichtteil
des Verfahrens – Regex+NER sind Sicherheitsnetz, kein Ersatz.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from importlib.util import find_spec as _find_spec
from typing import Any

_log = logging.getLogger("risk_layer.anonymize")

# Rechtsformen – „X GmbH", „M-GmbH & Co. KG", „Alpha SE" → eine Gesellschaft
_RECHTSFORM = (
    r"GmbH\s*&\s*Co\.?\s*KGaA|GmbH\s*&\s*Co\.?\s*KG|AG\s*&\s*Co\.?\s*KG|"
    r"GmbH|gGmbH|UG(?:\s*\(haftungsbeschränkt\))?|KGaA|AG|SE|KG|OHG|GbR|"
    r"e\.\s?G\.|e\.\s?V\.|e\.\s?K\.|mbH|Ltd\.?|Inc\.?|S\.\s?A\.|N\.\s?V\.|plc"
)
_RF = rf"(?:{_RECHTSFORM})"
# „M-GmbH", „T-GmbH" – Rechtsform per Bindestrich angehängt
_FIRMA_KOMPAKT = re.compile(rf"\b[A-ZÄÖÜ][\wäöüßÄÖÜ.&]*-{_RF}\b")
# „Alpha SE", „Beta GmbH", „Mueller und Schulze GmbH" – ein bis mehrere
# großgeschriebene Namenswörter (auch per „und"/„&" verbunden) + Rechtsform.
# Satzanfang-Artikel („Die Mueller … GmbH") zählen NICHT zum Firmennamen.
_ARTIKEL = r"(?:Die|Der|Das|Den|Dem|Des|Ein|Eine|Einer|Eines|Einem|Einen)"
_FIRMA_WORT = rf"(?!{_ARTIKEL}\b)[A-ZÄÖÜ][\wäöüßÄÖÜ.&]+"
_FIRMA_SPACED = re.compile(
    rf"\b{_FIRMA_WORT}(?:(?:\s+(?:und|&))?\s+{_FIRMA_WORT}){{0,5}}\s+{_RF}\b")
# Anrede + (Titel) + Name; Bindestrich-Doppelnamen („Mueller-Lüdenscheid")
# werden vollständig erfasst, nicht nur das erste Glied.
_NAME = r"[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)*"
_PERSON = re.compile(rf"\b(?:Herr|Frau|Hr\.|Fr\.)\s+(?:Dr\.\s+|Prof\.\s+){{0,2}}{_NAME}(?:\s+{_NAME})?")
# Beträge: mit Tausendertrennern (1.500 EUR, 1 500 000 €) ODER ungetrennt
# (50000 EUR, 1500 Euro). Der WÄHRUNGSKONTEXT ist Pflicht – nackte Zahlen
# (Paragraphen „§ 146 AO", Ordnungsnummern „Nr. 21") bleiben unangetastet.
_WAEHRUNG = r"(?:€|EUR|Euro|TEUR|Mio\.?\s*(?:€|EUR|Euro))"
_BETRAG = re.compile(rf"\b(?:\d{{1,3}}(?:[.\s]\d{{3}})+|\d+)(?:,\d+)?\s*{_WAEHRUNG}")
_DATUM = re.compile(
    r"\b(?:\d{1,2}\.\s?(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s?\d{4}"
    r"|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)
_STEUERNR = re.compile(r"\b\d{2,3}/\d{3}/\d{4,5}\b")
# DE hat nach der Prüfziffer 18 Ziffern. Die letzte Zweiergruppe gehört
# auch in der Druckdarstellung vollständig zur Kontonummer.
_IBAN = re.compile(
    r"\b(?i:DE)\d{2}(?:[ \u00a0\u202f]?\d{4}){4}[ \u00a0\u202f]?\d{2}\b"
    r"|\b[A-Z]{2}\d{2}(?:\s?\d{4}){3,}\d{0,4}\b"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Telefon: +49/0-Vorwahl + >=6 weitere Ziffern/Trenner. Kein '.'/',' in der Klasse,
# damit Daten/Betraege (1.000,00) und Steuernummern nicht faelschlich greifen; der
# Lookbehind verhindert einen Treffer mitten in einer laengeren Zahl.
_TELEFON = re.compile(r"(?<!\d)(?:\+49|0)[\d /()-]{6,}\d\b")
# Anschrift (§ 203): Strasse + Hausnummer, optional direkt + PLZ + Ort -> EINE
# Anschrift ("Johannes-Gutwein Str. 8, 36304 Alsfeld"). Alle Quantoren bounded
# (kein '*'/'+' vor einem ueberspringbaren Anker) -> ReDoS-frei.
_STRASSE_SUFFIX = r"(?i:stra(?:ß|ss)e|str\.|weg|platz|allee|gasse|ring|damm|ufer|chaussee)"
_PLZ_ORT_TAIL = (r"(?:\s*,?\s+\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+"
                 r"(?:[ -][A-ZÄÖÜ][a-zäöüß]+){0,2})?")
# angehaengtes Suffix: "Hauptstraße 5", "Bahnhofstr. 12", "Gartenweg 3"
_ADRESSE_ATTACHED = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüßA-Za-z\-]{1,38}" + _STRASSE_SUFFIX
    + r"\s+\d{1,4}\s?[a-z]?\b" + _PLZ_ORT_TAIL)
# separates Suffix: "Johannes-Gutwein Str. 8", "Am Alten Markt Platz 2"
_ADRESSE_SEPARAT = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüßA-Za-z.\-]+(?:\s+[A-ZÄÖÜ][a-zäöüßA-Za-z.\-]+){0,2}\s+"
    + _STRASSE_SUFFIX + r"\s+\d{1,4}\s?[a-z]?\b" + _PLZ_ORT_TAIL)
# Standalone PLZ + Ort (ohne vorausgehende Strasse): "36304 Alsfeld"
_PLZ_ORT = re.compile(r"\b\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+(?:[ -][A-ZÄÖÜ][a-zäöüß]+){0,2}\b")
# „M-GmbH" → der reine Buchstaben-Stub als Mandantenkürzel (nach Rechtsform-Ersetzung übrig)
# (durch _FIRMA bereits erfasst)


@dataclass
class Ersetzung:
    original: str
    platzhalter: str
    typ: str


@dataclass
class AnonymisierungsErgebnis:
    abstrahiert: str
    ersetzungen: list[Ersetzung] = field(default_factory=list)
    # ``True`` nur, wenn ein NER-Modell diesen konkreten Text erfolgreich
    # verarbeitet hat. Installationsprognosen reichen fuer Export-Gates nicht.
    ner_verwendet: bool = False
    ner_grund: str | None = None

    def as_dict(self) -> dict:
        return {
            "abstrahiert": self.abstrahiert,
            "ersetzungen": [{"original": e.original, "platzhalter": e.platzhalter, "typ": e.typ}
                            for e in self.ersetzungen],
            "ner": {"verwendet": self.ner_verwendet, "grund": self.ner_grund},
        }


class _Zaehler:
    """Vergibt stabile Platzhalter je Typ; identische Originale → gleicher Platzhalter."""

    _LABEL = {"firma": "Gesellschaft", "person": "Person", "betrag": "Betrag",
              "datum": "Datum", "steuernr": "Steuernummer", "iban": "Konto",
              "email": "E-Mail", "telefon": "Telefon", "adresse": "Adresse",
              "ort": "Ort"}    # „ort" liefert nur die NER-Schicht (LOC)
    _BUCHST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self) -> None:
        self._map: dict[tuple[str, str], str] = {}
        self._n: dict[str, int] = {}
        self.ersetzungen: list[Ersetzung] = []

    def platzhalter(self, original: str, typ: str) -> str:
        schluessel = (typ, original.strip().lower())
        if schluessel in self._map:
            return self._map[schluessel]
        idx = self._n.get(typ, 0)
        self._n[typ] = idx + 1
        label = self._LABEL.get(typ, typ.capitalize())
        suffix = self._BUCHST[idx] if idx < 26 else str(idx + 1)
        ph = f"[{label} {suffix}]"
        self._map[schluessel] = ph
        self.ersetzungen.append(Ersetzung(original=original.strip(), platzhalter=ph, typ=typ))
        return ph


# Reihenfolge ist wichtig: spezifische Muster zuerst (IBAN/Steuernr vor Beträgen).
_REGELN: list[tuple[str, re.Pattern]] = [
    ("iban", _IBAN),
    ("steuernr", _STEUERNR),
    ("email", _EMAIL),
    ("adresse", _ADRESSE_ATTACHED),   # Strasse+Nr.(+PLZ/Ort) VOR Firma/Person/Betrag
    ("adresse", _ADRESSE_SEPARAT),
    ("adresse", _PLZ_ORT),
    ("firma", _FIRMA_KOMPAKT),
    ("firma", _FIRMA_SPACED),
    ("person", _PERSON),
    ("betrag", _BETRAG),
    ("datum", _DATUM),
    ("telefon", _TELEFON),    # zuletzt: numerische Spezialmuster (Betrag/Datum/Steuernr) zuerst
]


# --- Optionale NER-Schicht (spaCy, Extra ``nlp``) ----------------------------
# Zweite Verteidigungslinie HINTER den Regexen: erkennt Personen ohne Anrede,
# Firmen ohne Rechtsform und Orte. Fehlen spaCy oder das Modell, bleibt der
# deterministische Regex-Pfad allein aktiv (Degradations-Philosophie wie bei
# llm.py/embedding.py: funktional bleiben, aber SICHTBAR loggen).

_UNGELADEN = object()    # Sentinel: Laden wurde noch nie versucht
_NLP: Any = _UNGELADEN   # prozessweiter Modul-Cache (None = nicht verfügbar)

# Reihenfolge = Präferenz: lg (beste NER-Güte, /LF50/) vor sm (Notbehelf).
_NER_MODELLE = ("de_core_news_lg", "de_core_news_sm")
_NER_INSTALL_HINWEIS = ("NER-Schicht inaktiv — pip install --require-hashes "
                        "--only-binary=:all: -r requirements-ner-lock.txt")
# spaCy-Label → Platzhalter-Typ des _Zaehler. MISC wird bewusst NICHT ersetzt:
# das Label ist zu unscharf (Produkte, Nationalitäten, Gesetzesnamen) – zerstörte
# Rechtsbegriffe wären teurer als die ohnehin verpflichtende manuelle Prüfung.
_NER_TYP = {"PER": "person", "ORG": "firma", "LOC": "ort"}


def _ner_deaktiviert() -> bool:
    """RISK_LAYER_NER=aus erzwingt den rein deterministischen Regex-Pfad.

    Warum: NER ist statistisch und modellabhängig – Determinismus-kritische
    Läufe (Reproduzierbarkeit /NF20/, Golden-File-Vergleiche) müssen die
    Schicht hart abschalten können, OHNE dass das Modell überhaupt geladen
    wird. Akzeptierte Werte wie beim Reranker: aus/off/none/0.
    """
    return os.environ.get("RISK_LAYER_NER", "").strip().lower() in ("aus", "off", "none", "0")


def _lade_nlp() -> Any:
    """Lazy-Load des deutschen spaCy-Modells mit prozessweitem Modul-Cache.

    Vorbild ist der Lazy-Load in embedding.py – das dortige double-checked
    Locking braucht es hier NICHT: spaCy lädt CPU-seitig und idempotent (kein
    GPU-Forward, kein doppelter VRAM-Load möglich). Schlimmster Fall bei
    parallelen Erst-Aufrufen: zwei Threads laden einmal doppelt, ein Ergebnis
    wird verworfen – harmlos.

    Rückgabe: nlp-Objekt oder None (= NER inaktiv, Regex-Pfad allein). Dank
    Cache wird der Install-Hinweis nur EINMAL pro Prozess geloggt, nicht pro
    Dokument. Tests injizieren über genau diesen Seam ein Fake-NLP.
    """
    global _NLP
    if _NLP is not _UNGELADEN:
        return _NLP
    try:
        import spacy  # lazy: optionales Extra ``nlp``
    except ImportError:
        _log.warning(_NER_INSTALL_HINWEIS)
        _NLP = None
        return None
    for modell in _NER_MODELLE:
        try:
            nlp = spacy.load(modell)
        except OSError:
            continue    # Modell nicht installiert → nächste Stufe probieren
        except Exception as exc:  # noqa: BLE001
            # Defensiv: eine defekte Installation darf das § 203-Gate nicht
            # crashen – der Regex-Pfad bleibt als Mindestschutz bestehen.
            _log.warning("spaCy-Modell %s nicht ladbar (%s: %s) — versuche Fallback",
                         modell, type(exc).__name__, exc)
            continue
        if modell != _NER_MODELLE[0]:
            _log.warning("de_core_news_lg fehlt — nutze %s (geringere NER-Güte). "
                         "Abhilfe: pip install --require-hashes --only-binary=:all: "
                         "-r requirements-ner-lock.txt", modell)
        _NLP = nlp
        return nlp
    _log.warning(_NER_INSTALL_HINWEIS)
    _NLP = None
    return None


def _paket_installiert(name: str) -> bool:
    """``find_spec`` ohne Import: prüft nur die Installations-Metadaten.

    ValueError fängt den Sonderfall ab, dass ein Paket in ``sys.modules`` auf
    None steht (kaputter/abgebrochener Import) – ``find_spec`` wirft dann,
    statt None zu liefern. Für den Status zählt beides schlicht als „nicht
    verfügbar" (der Regex-Pfad bleibt ohnehin der Mindestschutz)."""
    try:
        return _find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def ner_status() -> dict[str, Any]:
    """Zustand der NER-Schicht für den Engine-Status – OHNE das Modell zu laden.

    Form: ``{"verfuegbar": bool, "modell": str | None, "grund": str | None}``.

    Feldname bewusst ``verfuegbar`` statt „aktiv": gemeldet wird, ob die
    Schicht laufen KÖNNTE. Vor dem ersten ``anonymisiere()``-Aufruf ist das
    nur eine Installations-Prognose (``find_spec``) – das tatsächliche Laden
    kann trotzdem scheitern (defekte Installation; ``_lade_nlp`` degradiert
    dann laut). Erst der Modul-Cache ``_NLP`` macht die Aussage hart: ist er
    befüllt, wird der TATSÄCHLICH geladene Modellname gemeldet. Bewusst KEIN
    ``_lade_nlp()``-Aufruf: eine Status-Abfrage darf weder Ladezeit kosten
    noch den Determinismus-Schalter (RISK_LAYER_NER) unterlaufen.
    """
    if _ner_deaktiviert():
        return {"verfuegbar": False, "modell": None,
                "grund": "deaktiviert (RISK_LAYER_NER)"}
    if _NLP is not _UNGELADEN:
        # Laden wurde bereits versucht – der Cache ist die härteste Evidenz
        # (genauer als jede find_spec-Prognose, in beide Richtungen).
        if _NLP is None:
            return {"verfuegbar": False, "modell": None, "grund": _NER_INSTALL_HINWEIS}
        meta = getattr(_NLP, "meta", None) or {}
        modell = f"{meta.get('lang', '')}_{meta['name']}" if meta.get("name") else None
        return {"verfuegbar": True, "modell": modell, "grund": None}
    # Noch nie geladen: Prognose über die Installations-Metadaten, in derselben
    # Präferenz-Reihenfolge wie _lade_nlp (lg → sm).
    if not _paket_installiert("spacy"):
        return {"verfuegbar": False, "modell": None, "grund": _NER_INSTALL_HINWEIS}
    for modell in _NER_MODELLE:
        if _paket_installiert(modell):
            return {"verfuegbar": True, "modell": modell, "grund": None}
    return {"verfuegbar": False, "modell": None, "grund": _NER_INSTALL_HINWEIS}


def _ner_pass(text: str, z: _Zaehler) -> tuple[str, bool, str | None]:
    """NER-Nachlauf über den bereits regex-ersetzten Text.

    Der Regex-Pass bleibt FÜHREND: Spans, die einen bereits vergebenen
    Platzhalter überlappen, sind tabu – NER ergänzt nur, was die Regexe nicht
    fanden. Der gemeinsame ``_Zaehler`` führt die Nummerierung pro Dokument
    fort und hält sie stabil (gleicher Wortlaut + Typ → gleicher Platzhalter,
    auch über die Regex-/NER-Grenze hinweg).
    """
    # Env-Check VOR dem Loader: RISK_LAYER_NER=aus darf das Modell gar nicht
    # erst laden (Determinismus + Startzeit).
    if _ner_deaktiviert():
        return text, False, "deaktiviert (RISK_LAYER_NER)"
    nlp = _lade_nlp()
    if nlp is None:
        return text, False, _NER_INSTALL_HINWEIS
    try:
        doc = nlp(text)
    except Exception as exc:  # noqa: BLE001
        # Fail-degraded statt Crash: das Gate bleibt mindestens so stark wie
        # der Regex-Pfad; ein kaputtes Modell darf den Export-Pfad nicht legen.
        # Modellfehler können Ausschnitte des eingereichten Texts enthalten.
        _log.warning("NER-Pass fehlgeschlagen (%s) — nur Regex-Ergebnis",
                     type(exc).__name__)
        return text, False, f"NER-Pass fehlgeschlagen ({type(exc).__name__})"
    # Tabu-Spans: alle Vorkommen der von UNS vergebenen Platzhalter. Warum kein
    # generisches „[…]"-Regex: eckige Klammern können legitim im Sachverhalt
    # stehen – nur echte Platzhalter dürfen vor NER geschützt sein.
    tabu: list[tuple[int, int]] = []
    for ph in {e.platzhalter for e in z.ersetzungen}:
        pos = text.find(ph)
        while pos != -1:
            tabu.append((pos, pos + len(ph)))
            pos = text.find(ph, pos + len(ph))

    def _ueberlappt(start: int, ende: int) -> bool:
        return any(start < t_ende and ende > t_start for t_start, t_ende in tabu)

    # Von hinten nach vorn ersetzen, damit die Offsets der weiter links
    # liegenden, noch ausstehenden Entitäten gültig bleiben.
    grenze = len(text)
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        typ = _NER_TYP.get(ent.label_)
        if typ is None:
            continue    # MISC u. Ä. bewusst ignorieren (s. _NER_TYP)
        start, ende = ent.start_char, ent.end_char
        original = ent.text
        # Defensiv: inkonsistente Offsets (Modell-/Fake-Fehler) NICHT ersetzen –
        # ein falsch getroffener Span zerstörte die juristische Struktur UND
        # ließe das eigentliche Geheimnis stehen.
        if not original or text[start:ende] != original:
            continue
        if ende > grenze:    # Überlappung mit bereits ersetztem NER-Span
            continue
        if _ueberlappt(start, ende):    # Platzhalter aus dem Regex-Pass: tabu
            continue
        text = text[:start] + z.platzhalter(original, typ) + text[ende:]
        grenze = start
    return text, True, None


def anonymisiere(text: str) -> AnonymisierungsErgebnis:
    """Erzeugt eine abstrahierte Fassung + Rückübersetzungstabelle (lokal).

    Schicht 1: deterministische Regexe (immer, führend). Schicht 2: optionaler
    NER-Nachlauf – nur was die Regexe nicht fanden (s. Moduldocstring;
    abschaltbar per RISK_LAYER_NER=aus).
    """
    z = _Zaehler()
    ergebnis = text
    for typ, regex in _REGELN:
        def _ersetze(m: re.Match, _typ: str = typ) -> str:
            return z.platzhalter(m.group(0), _typ)
        ergebnis = regex.sub(_ersetze, ergebnis)
    # NER ergänzt NACH den Regexen; ohne spaCy/Modell oder mit
    # RISK_LAYER_NER=aus ist das ein No-Op (exakt bisheriges Verhalten).
    ergebnis, ner_verwendet, ner_grund = _ner_pass(ergebnis, z)
    return AnonymisierungsErgebnis(
        abstrahiert=ergebnis,
        ersetzungen=z.ersetzungen,
        ner_verwendet=ner_verwendet,
        ner_grund=ner_grund,
    )
