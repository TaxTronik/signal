"""Lokale Abstraktion/Anonymisierung (§ 203 StGB) – Geheimnisschutz-Gate.

Bevor ein Sachverhalt an ein externes (Cloud-)LLM gegeben wird, MUSS er lokal
abstrahiert werden: mandantenidentifizierende Angaben (Firmen, Personen, Beträge,
Daten, Steuer-/Konto-Nummern) werden durch stabile Platzhalter ersetzt. Die
juristische Struktur bleibt erhalten – die Rechtsbegriffe selbst sind kein
Geheimnis.

Reiner Code, vollständig lokal. Das Ergebnis ist ein ENTWURF, den der Berater
vor Freigabe manuell prüft und anpasst. Die Rückübersetzungstabelle
(``ersetzungen``) bleibt lokal und wird NIEMALS mitexportiert.

Zulässiger Fehlermodus wie beim Risiko-Layer: lieber zu viel schwärzen
(Über-Abstraktion) als ein Geheimnis preisgeben.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Rechtsformen – „X GmbH", „M-GmbH & Co. KG", „Alpha SE" → eine Gesellschaft
_RECHTSFORM = (
    r"GmbH\s*&\s*Co\.?\s*KGaA|GmbH\s*&\s*Co\.?\s*KG|AG\s*&\s*Co\.?\s*KG|"
    r"GmbH|gGmbH|UG(?:\s*\(haftungsbeschränkt\))?|KGaA|AG|SE|KG|OHG|GbR|"
    r"e\.\s?G\.|e\.\s?V\.|e\.\s?K\.|mbH|Ltd\.?|Inc\.?|S\.\s?A\.|N\.\s?V\.|plc"
)
_RF = rf"(?:{_RECHTSFORM})"
# „M-GmbH", „T-GmbH" – Rechtsform per Bindestrich angehängt
_FIRMA_KOMPAKT = re.compile(rf"\b[A-ZÄÖÜ][\wäöüßÄÖÜ.&]*-{_RF}\b")
# „Alpha SE", „Beta GmbH", „X GmbH & Co. KG" – Name + Leerzeichen + Rechtsform
_FIRMA_SPACED = re.compile(rf"\b[A-ZÄÖÜ][\wäöüßÄÖÜ.&]+\s+{_RF}\b")
_PERSON = re.compile(r"\b(?:Herr|Frau|Hr\.|Fr\.)\s+(?:Dr\.\s+|Prof\.\s+){0,2}[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?")
_BETRAG = re.compile(r"\b\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?\s*(?:€|EUR|Euro|TEUR|Mio\.?\s*(?:€|EUR|Euro))")
_DATUM = re.compile(
    r"\b(?:\d{1,2}\.\s?(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s?\d{4}"
    r"|\d{1,2}\.\d{1,2}\.\d{2,4})\b"
)
_STEUERNR = re.compile(r"\b\d{2,3}/\d{3}/\d{4,5}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?\d{4}){3,}\d{0,4}\b")
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

    def as_dict(self) -> dict:
        return {
            "abstrahiert": self.abstrahiert,
            "ersetzungen": [{"original": e.original, "platzhalter": e.platzhalter, "typ": e.typ}
                            for e in self.ersetzungen],
        }


class _Zaehler:
    """Vergibt stabile Platzhalter je Typ; identische Originale → gleicher Platzhalter."""

    _LABEL = {"firma": "Gesellschaft", "person": "Person", "betrag": "Betrag",
              "datum": "Datum", "steuernr": "Steuernummer", "iban": "Konto",
              "email": "E-Mail", "telefon": "Telefon", "adresse": "Adresse"}
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


def anonymisiere(text: str) -> AnonymisierungsErgebnis:
    """Erzeugt eine abstrahierte Fassung + Rückübersetzungstabelle (lokal)."""
    z = _Zaehler()
    ergebnis = text
    for typ, regex in _REGELN:
        def _ersetze(m: re.Match, _typ: str = typ) -> str:
            return z.platzhalter(m.group(0), _typ)
        ergebnis = regex.sub(_ersetze, ergebnis)
    return AnonymisierungsErgebnis(abstrahiert=ergebnis, ersetzungen=z.ersetzungen)
