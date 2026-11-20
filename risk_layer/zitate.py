"""Zitat-Parser (Parser-Spez. Kap. 2 & 3) – korpus-unabhängiger Kern.

Wandelt Norm- und Entscheidungszitate in kanonische IDs (siehe ``idschema``).
Reiner Code, ohne gii-XML lauffähig und testbar. Listen/Bereiche/„ff." werden
expandiert; relative Verweise (ohne Gesetz/§) werden als Risiko geflaggt
(``relativ=True``, niedrige Konfidenz) – ihre Auflösung über den Kontext-Stack
gehört in die Graph-Stufe (Spez. 2.3).
"""

from __future__ import annotations

import itertools
import re
import weakref
from dataclasses import dataclass, replace
from typing import Literal

from .idschema import NormID, Quellentyp, UrteilID, VerwID
from .lawmap import DEFAULT_LAWMAP, LawMap

# Konfidenz-Skala (aufsteigend) – EINE zentrale Quelle für Annotation UND
# Vergleiche (graph nutzt die Reihenfolge für „höchste Konfidenz gewinnt").
# Bewusst Literal statt Enum: die Strings sind API (as_dict, SQLite, Fixtures).
Konfidenz = Literal["niedrig", "mittel", "hoch"]
KONFIDENZ_STUFEN: tuple[Konfidenz, ...] = ("niedrig", "mittel", "hoch")

# Kompilierte Freitext-Scan-Regex je LawMap. finde_normzitate baute die große
# Law-Alternation und kompilierte sie bei JEDEM Aufruf neu – input-unabhängige
# Verschwendung in heißen Schleifen (pro Kandidat im Gate, pro Text der Pipeline).
# Eine LawMap ist nach Konstruktion immutabel, daher pro Instanz sicher cachebar;
# WeakKeyDictionary verhindert ein Festhalten kurzlebiger (merge_harvest-)Maps.
_SCAN_PATTERN_CACHE: "weakref.WeakKeyDictionary[LawMap, re.Pattern[str]]" = (
    weakref.WeakKeyDictionary())

# Ausgeschriebene Namen der Korpusgesetze, wie sie im laufenden Normtext stehen.
_LANGNAMEN = {
    "Abgabenordnung": "AO", "Aktiengesetz": "AktG", "Außensteuergesetz": "AStG",
    "Bewertungsgesetz": "BewG", "Einkommensteuergesetz": "EStG",
    "Erbschaftsteuer- und Schenkungsteuergesetz": "ErbStG",
    "Finanzgerichtsordnung": "FGO", "Gewerbesteuergesetz": "GewStG",
    "Grunderwerbsteuergesetz": "GrEStG", "Grundsteuergesetz": "GrStG",
    "Grundgesetz": "GG", "Handelsgesetzbuch": "HGB", "Investmentsteuergesetz": "InvStG",
    "Körperschaftsteuergesetz": "KStG", "Solidaritätszuschlaggesetz": "SolzG",
    "Steuerberatungsgesetz": "StBerG", "Umsatzsteuergesetz": "UStG",
    "Umwandlungsgesetz": "UmwG", "Umwandlungssteuergesetz": "UmwStG",
    "Gesetz betreffend die Gesellschaften mit beschränkter Haftung": "GmbHG",
    "Gesetzes betreffend die Gesellschaften mit beschränkter Haftung": "GmbHG",
    "Gesetz über die Besteuerung bei Auslandsbeziehungen": "AStG",
    "Gesetzes über die Besteuerung bei Auslandsbeziehungen": "AStG",
    "Einkommensteuer-Durchführungsverordnung": "EStDV",
    "Erbschaftsteuer-Durchführungsverordnung": "ErbStDV",
    "Gewerbesteuer-Durchführungsverordnung": "GewStDV",
    "Körperschaftsteuer-Durchführungsverordnung": "KStDV",
    "Lohnsteuer-Durchführungsverordnung": "LStDV",
    "Umsatzsteuer-Durchführungsverordnung": "UStDV",
}
for _name, _law in list(_LANGNAMEN.items()):
    if _name.endswith(("gesetz", "gesetzbuch")):
        _LANGNAMEN[_name + "es"] = _law
    if _name.endswith("gesetzbuch"):
        _LANGNAMEN[_name + "s"] = _law
_LANGNAME_AM_ENDE = re.compile(r"\b(" + "|".join(
    re.escape(name) for name in sorted(_LANGNAMEN, key=len, reverse=True)) + r")$")


def _scan_pattern(lawmap: LawMap) -> re.Pattern[str]:
    pat = _SCAN_PATTERN_CACHE.get(lawmap)
    if pat is None:
        namen = sorted(set(lawmap.alle_aliase()) | _LANGNAMEN.keys(), key=len, reverse=True)
        law_alt = "|".join(re.escape(a) for a in namen)
        # Zwischen der Nummer und dem Gesetz dürfen nur Bestandteile eines
        # Zitats stehen. Ein später im Satz genanntes Gesetz gehört sonst
        # fälschlich zu einem vorangehenden relativen Verweis.
        pat = re.compile(
            rf"{_ZITAT_MIT_ZUSAETZEN}(?:\s+(?:des|der))?\s*\b(?P<quelle>{law_alt})\b(?![-\w])")
        _SCAN_PATTERN_CACHE[lawmap] = pat
    return pat


# --- Normzitat-Parser -------------------------------------------------------

# Tiefenmarker-Grammatik (Spez. 2.1) – die EINE Quelle für Parser (_DEPTH_RE)
# UND Fließtext-Span-Scanner (_ZITAT_SPAN). Vorher war sie doppelt definiert
# und divergierte: der Span kannte keine reinen Buchstabenwerte („Buchst. a"),
# brach das Zitat davor ab und attribuierte den Rest ans falsche Gesetz.
# Längere Marker-Varianten zuerst (z. B. "Satz" vor "S").
_TIEFE_KINDS = r"Absatz|Abs|Nummer|Nr|Satz|S|Buchstabe|Buchst|lit|Halbsatz|Halbs|Alt|Var"
# Wert: Zahl mit optionalem Buchstabensuffix („8c") ODER reiner Buchstabe („a").
_TIEFE_WERT = r"\d+[a-z]?|[a-z]"

_BEREICHS_TRENNER = r"(?:bis\b|[–—-])"
_LISTENTRENNER = rf"(?:,|und\b|{_BEREICHS_TRENNER})"
_PARA_NUM = re.compile(r"\d+[a-z]*")
_PARA_KETTE = rf"{_PARA_NUM.pattern}(?:\s*{_LISTENTRENNER}\s*{_PARA_NUM.pattern})*"
_TIEFEN_NUM = re.compile(rf"(?<![a-z])(?:{_TIEFE_WERT})(?!\w)")
_TIEFEN_WERT = rf"(?:{_TIEFE_WERT})(?!\w)"
_DEPTH_RE = re.compile(
    rf"(?P<kind>{_TIEFE_KINDS})\.?\s*(?P<folge>{_TIEFEN_WERT}(?:\s*{_LISTENTRENNER}\s*{_TIEFEN_WERT})*)"
)
_NORMZEICHEN = r"(?:§§?|\bArt(?:ikel|\.)?)"
_ZITATKERN = rf"{_NORMZEICHEN}\s*{_PARA_KETTE}(?:\s*{_DEPTH_RE.pattern})*"
_ZITAT_MIT_ZUSAETZEN = rf"{_ZITATKERN}(?:(?<=[a-z])\.)?(?:\s+ff?\.)?"
_MAX_ANKER_PRO_ZITAT = 4096


class _ExpansionZuGross(ValueError):
    """Die angegebene Liste kann nicht vollständig expandiert werden."""


@dataclass(frozen=True)
class GeparsterAnker:
    """Ergebnis je expandierter Norm-ID: ID + Beleg + Konfidenz/Flags (Spez. 2.5)."""

    norm_id: NormID | None
    roh: str
    konfidenz: Konfidenz    # niedrig | mittel | hoch (s. KONFIDENZ_STUFEN)
    relativ: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.norm_id.to_id() if self.norm_id else None,
            "roh": self.roh,
            "konfidenz": self.konfidenz,
            "relativ": self.relativ,
            "offener_bereich": bool(self.norm_id and self.norm_id.offener_bereich),
        }


def _label(kind: str) -> str:
    k = kind.lower()
    if k.startswith("abs"):
        return "abs"
    if k.startswith("n"):       # Nr, Nummer
        return "nr"
    if k.startswith("s"):       # Satz, S
        return "s"
    if k.startswith("buchst") or k == "lit":
        return "lit"
    if k.startswith("halbs"):
        return "hs"
    return "alt"                # Alt, Var


def _expand_range(v1: str, v2: str) -> list[str]:
    """Inklusiver Bereich für numerische oder einbuchstabige Grenzen."""
    if v1.isdigit() and v2.isdigit():
        if max(len(v1), len(v2)) > 10:
            raise _ExpansionZuGross
        a, b = int(v1), int(v2)
        if a <= b:
            if b - a + 1 > _MAX_ANKER_PRO_ZITAT:
                raise _ExpansionZuGross
            return [str(x) for x in range(a, b + 1)]
    if len(v1) == 1 and len(v2) == 1 and v1.isalpha() and v2.isalpha():
        a, b = ord(v1), ord(v2)
        if a <= b:
            return [chr(x) for x in range(a, b + 1)]
    return [v1, v2]


def _expand_werte(text: str, muster: re.Pattern[str]) -> tuple[list[str], bool]:
    """Expandiert nur benachbarte Werte mit einem ausdrücklichen Bereichstrenner."""
    teile: list[str] = []
    vorher: re.Match[str] | None = None
    expandiert = False
    for nummer in muster.finditer(text):
        if vorher and re.fullmatch(_BEREICHS_TRENNER, text[vorher.end():nummer.start()].strip()):
            teile.extend(_expand_range(vorher.group(), nummer.group())[1:])
        else:
            teile.append(nummer.group())
        if len(teile) > _MAX_ANKER_PRO_ZITAT:
            raise _ExpansionZuGross
        expandiert = vorher is not None
        vorher = nummer
    return list(dict.fromkeys(teile)), expandiert


def _parse_depth(depth_part: str, anzahl_paras: int) -> tuple[list[tuple[tuple[str, str], ...]], bool]:
    """Parst Tiefenmarker, kartesisch über Bereiche/Listen expandiert.

    Returns (Liste von Tiefen-Tupeln, expandiert?).
    """
    achsen: list[list[tuple[str, str]]] = []
    expandiert = False
    kombinationen = anzahl_paras
    for m in _DEPTH_RE.finditer(depth_part):
        label = _label(m.group("kind"))
        werte, erweitert = _expand_werte(m.group("folge"), _TIEFEN_NUM)
        expandiert |= erweitert
        kombinationen *= len(werte)
        if kombinationen > _MAX_ANKER_PRO_ZITAT:
            raise _ExpansionZuGross
        achsen.append([(label, w) for w in werte])
    if not achsen:
        return [()], False
    varianten = [tuple(kombi) for kombi in itertools.product(*achsen)]
    return varianten, expandiert


def parse_normzitat(zitat: str, lawmap: LawMap = DEFAULT_LAWMAP) -> list[GeparsterAnker]:
    """Parst einen Normzitat-String zu einer Liste kanonischer Anker (Spez. 2)."""
    roh = zitat.strip()
    s = roh

    # 1) Gesetzes-Abkürzung am Ende (Longest-Match)
    law: str | None = None
    treffer = lawmap.match_am_ende(s)
    if treffer:
        law, alias = treffer
        s = s[: len(s) - len(alias)].strip()
    elif langname := _LANGNAME_AM_ENDE.search(s):
        law = _LANGNAMEN[langname[1]]
        s = s[:langname.start()].strip()

    # 2) "ff."/"f." → offener Bereich. AUSNAHME: direkt hinter einem Buchstaben-
    # Marker ist „f." ein lit-WERT („§ 33 Abs. 1 Buchst. f. EStG") und bleibt
    # für _DEPTH_RE im String – sonst ginge litf verloren und das Zitat würde
    # fälschlich als offener Bereich geflaggt.
    ff_treffer = [m for m in re.finditer(r"\bff?\.", s)
                  if not re.search(r"(?:Buchst(?:abe)?|lit)\.?\s*$", s[: m.start()])]
    offener = bool(ff_treffer)
    for m in reversed(ff_treffer):          # rückwärts entfernen → Offsets stabil
        s = s[: m.start()] + s[m.end():]
    s = s.strip()

    # 3) Modus: Art. vs § vs (impliziter §, z. B. „162 AO") vs relativ
    artikel_mode = bool(re.match(r"^(Art\.?|Artikel)\b", s))
    if artikel_mode:
        s = re.sub(r"^(Artikel|Art\.?)\s*", "", s).strip()
    elif s.startswith("§"):
        s = re.sub(r"^§+\s*", "", s).strip()
    elif re.match(r"^\d", s):
        pass  # ohne „§" geschrieben („162 AO", „162 Abs. 1 AO") – Zahl = Paragraph
    else:
        # relativer Verweis (kein Gesetz, kein §) – flaggen, nicht raten (Spez. 2.3)
        return [GeparsterAnker(norm_id=None, roh=roh, konfidenz="niedrig", relativ=True)]

    # 4) Para-Teil vs. Tiefe-Teil trennen (erster Tiefenmarker)
    dm = _DEPTH_RE.search(s)
    if dm:
        para_part, depth_part = s[: dm.start()], s[dm.start():]
    else:
        para_part, depth_part = s, ""

    try:
        paras, p_exp = _expand_werte(para_part, _PARA_NUM)
        tiefen, t_exp = _parse_depth(depth_part, max(1, len(paras)))
    except _ExpansionZuGross:
        return [GeparsterAnker(norm_id=None, roh=roh, konfidenz="niedrig", relativ=law is None)]
    expandiert = p_exp or t_exp

    if not paras and not artikel_mode:
        return [GeparsterAnker(norm_id=None, roh=roh, konfidenz="niedrig", relativ=law is None)]

    konf: Konfidenz
    if law:
        konf = "mittel" if (offener or expandiert) else "hoch"
    else:
        konf = "niedrig"

    anker: list[GeparsterAnker] = []
    para_iter = paras if not artikel_mode else [None]
    art_nums = paras if artikel_mode else [None]
    for para, art in itertools.product(para_iter, art_nums):
        for tiefe in tiefen:
            nid = NormID(
                law=law or "?",
                para=para,
                artikel=art,
                tiefe=tiefe,
                offener_bereich=offener,
            )
            anker.append(GeparsterAnker(norm_id=nid, roh=roh, konfidenz=konf, relativ=law is None))
    return anker


# Zitat-Span im Fließtext (mit/ohne Gesetzesnamen), für intra-/cross-law-Kanten.
# Aus DERSELBEN Tiefenmarker-Grammatik gebaut wie _DEPTH_RE (keine zweite,
# divergierende Definition mehr). (?!\w) nach dem Wert verhindert, dass der
# Einzelbuchstaben-Wert ins Folgewort läuft („… Alt e‹rnative›"). Bewusst OHNE
# „und"-Aufzählungen bei Verweisen ohne Gesetzesnamen: „Satz 1 und 2 weitere
# Personen" kann laufender Text sein. Ein nachfolgendes explizites Gesetz
# verankert hingegen die gesamte Kette über _scan_pattern.
_ZITAT_SPAN = re.compile(
    rf"(?:§§\s*{_PARA_KETTE}|(?:§|\bArt(?:ikel|\.)?)\s*{_PARA_NUM.pattern})"
    rf"(?:\s+(?:{_TIEFE_KINDS})\.?"
    rf"\s*(?:{_TIEFE_WERT})(?!\w)(?:\s*(?:bis|–|—|-)\s*(?:{_TIEFE_WERT})(?!\w))?)*"
)

# Ein explizit genanntes Fremdgesetz ohne registrierte Kennung lässt sich
# nicht als Eigenverweis auswerten. Nur unmittelbar anschließende Namen prüfen.
_FREMDE_QUELLE = re.compile(
    rf"{_ZITAT_MIT_ZUSAETZEN}\s+(?:(?:des|der)\s+)?"
    r"(?:(?:[A-ZÄÖÜ][\w-]*\s+){0,3}(?:[A-ZÄÖÜ][\w-]*?)?"
    r"(?i:gesetz(?:buch)?(?:e?s)?|verordnung|ordnung|richtlinie|übereinkommen(?:s)?)"
    r"|Zollkodex(?:es)?\s+der\s+Union|[A-Z][A-Za-z0-9]*(?:GB|DV|RL|VO|G|O))\b"
)
# Unmittelbar koordinierte Zitate können einen einzigen nachgestellten
# Gesetzesnamen teilen („§ 72a Abs. 4 und § 93c Abs. 4 der Abgabenordnung“).
# Das nächste Normzeichen muss wirklich folgen. Punkte und laufender Text
# dürfen den Geltungsbereich einer Quellenangabe nicht ausweiten.
_KETTENGLIED = re.compile(
    rf"(?P<glied>{_ZITATKERN}(?:\s+ff?\.)?)(?:\s*,\s*|\s+(?:und|oder|sowie)\s+)"
    rf"(?=(?P<weiter>{_NORMZEICHEN})\s*\d)"
)


def extrahiere_anker(
    text: str, default_law: str | None = None, lawmap: LawMap = DEFAULT_LAWMAP
) -> list[GeparsterAnker]:
    """Extrahiert Norm-Anker aus Fließtext – inkl. intra-law-Verweisen.

    Ein bekannter Gesetzesname oder ein Kürzel kann für mehrere koordinierte
    Zitate gelten. Ein ausdrücklicher Verweis auf eine unbekannte Quelle bleibt
    unaufgelöst. Ohne Quellenangabe gilt ``default_law`` (typisch für interne
    Verweise wie „§ 15“). Damit entstehen die ``verweist_auf``-Kanten des Graphen.
    """
    out: list[GeparsterAnker] = []
    offen: list[str] = []
    mit_gesetz = _scan_pattern(lawmap)
    for m in _ZITAT_SPAN.finditer(text):
        direkt: list[GeparsterAnker] = []
        explizit = mit_gesetz.match(text, m.start())
        if explizit:
            direkt = parse_normzitat(explizit.group(), lawmap)
            quelle = explizit["quelle"]
            gesetz = lawmap.resolve(quelle) or _LANGNAMEN.get(quelle)
        else:
            glied = _KETTENGLIED.match(text, m.start())
            if glied and m[0].startswith("§") == glied["weiter"].startswith("§"):
                offen.append(glied["glied"])
                continue
            offen.append(m[0])
            gesetz = default_law if default_law and not _FREMDE_QUELLE.match(text, m.start()) else None
        if gesetz:
            for zitat in offen:
                for a in parse_normzitat(zitat, lawmap):
                    if a.norm_id:
                        out.append(GeparsterAnker(replace(a.norm_id, law=gesetz), a.roh, "mittel"))
        offen.clear()
        out.extend(direkt)
    return out


def finde_normzitate(text: str, lawmap: LawMap = DEFAULT_LAWMAP) -> list[GeparsterAnker]:
    """Scannt FREITEXT nach Normzitaten (§…LAW / Art.…LAW) und parst jedes.

    Anders als ``parse_normzitat`` (das einen einzelnen, sauberen Zitatstring
    erwartet) findet diese Funktion Zitate inmitten von Fließtext – nötig für die
    Norm-Nähe des Unknown-Gates und später für den Kantenaufbau im Graphen.
    """
    return extrahiere_anker(text, lawmap=lawmap)


# --- Rekonstruktion (Round-Trip /T10/) --------------------------------------

_TIEFE_WORT = {"abs": "Abs.", "nr": "Nr.", "s": "Satz", "lit": "Buchst.", "hs": "Halbs.", "alt": "Alt."}


def norm_zu_zitat(nid: NormID) -> str:
    """Rekonstruiert ein menschenlesbares Zitat aus einer NormID (/T10/)."""
    kopf = f"Art. {nid.artikel}" if nid.artikel else f"§ {nid.para}"
    teile = [f"{_TIEFE_WORT[label]} {wert}" for label, wert in nid.tiefe]
    ff = " ff." if nid.offener_bereich else ""
    return " ".join([kopf, *teile]).strip() + ff + f" {nid.law}"


def knoten_zitat(law: str, para: str, *, absatz: str | None = None,
                 nummer: str | None = None) -> str:
    """Lesbares Zitat für die im Graph gespeicherten Kopf- und Tiefenfelder."""
    tiefe = tuple((stufe, wert) for stufe, wert in (("abs", absatz), ("nr", nummer)) if wert)
    if para.startswith("art"):
        return norm_zu_zitat(NormID(law=law, artikel=para[3:], tiefe=tiefe))
    return norm_zu_zitat(NormID(law=law, para=para, tiefe=tiefe))


# --- Entscheidungszitat-Parser (Spez. 3) ------------------------------------

_GERICHT_RE = re.compile(
    r"\s*(?P<ger>BVerfG|BVerwG|BFH|BGH|EuGH|EuG|FG\s+[A-ZÄÖÜ][^\s,]+|FG)"
)
_DATUM_RE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}")
# Aktenzeichen am Senats-/Registerzeichen VERANKERT (kein Lazy-Vorlauf mehr, der
# führende Wörter wie „Urteil" ins Az zog und den Join mit urteil:BFH:XI-R-29/22
# brach): römische Senatszahl (XI, III), arabische Kammer-/Senatszahl (2 BvL,
# 2 K), EuG-Register (T-…; C ist als römische Zahl ohnehin abgedeckt) oder
# kurzes Sonderregister (GrS, VGS). Lange Wörter („Urteil", „Beschluss")
# scheitern an der 4-Zeichen-Grenze des Sonderregister-Zweigs.
_AZ_RE = re.compile(
    r"\b(?:[IVXLC]{1,5}|\d{1,3}|T|[A-Z][A-Za-z]{1,3})"   # Senat/Register-Anker
    r"(?:[ \-][A-Za-z]{1,5}\.?)*"                         # Registerzeichen (R, ZR, BvL, K …)
    r"[ \-]\d{1,4}/\d{2,4}"                               # lfd. Nummer / Jahr
)
_FUNDSTELLE_RE = re.compile(r"[A-ZÄÖÜ][A-Za-zÄÖÜ.]*\.?\s*[IVX]*\s*\d{4},\s*\d+")

_GERICHT_QUELLENTYP = {
    "BFH": Quellentyp.RSPR_BFH,
    "EuGH": Quellentyp.RSPR_EUGH,
}


def gericht_quellentyp(gericht: str) -> Quellentyp:
    return _GERICHT_QUELLENTYP.get(gericht, Quellentyp.RSPR_SONSTIGE)


def parse_entscheidung(zitat: str) -> UrteilID | None:
    """Parst ein Entscheidungszitat zu einer Urteils-ID (Spez. 3.1/3.2).

    Aktenzeichen ist der dauerhafte Schlüssel; Fundstellen werden als Aliase
    geführt (Spez. 1.3/3.2).
    """
    gm = _GERICHT_RE.match(zitat)
    if not gm:
        return None
    ger_raw = gm.group("ger")
    if ger_raw.startswith("FG") and len(ger_raw) > 2:
        gericht = "FG" + re.sub(r"\s+", "", ger_raw[2:])
    else:
        gericht = ger_raw.replace(" ", "")

    rest = zitat[gm.end():]
    dm = _DATUM_RE.search(rest)
    after = rest[dm.end():] if dm else rest
    after = re.sub(r"^\s*[–—-]\s*", "", after.strip())

    am = _AZ_RE.search(after)
    if not am:
        return None
    az = re.sub(r"\s+", "-", am.group(0).strip())

    tail = after[am.end():]
    aliase: list[str] = []
    par = re.search(r"\(([^)]*\d+/\d{2,4}[^)]*)\)", tail)
    if par:
        aliase.append(par.group(1).strip())
    for f in _FUNDSTELLE_RE.finditer(tail):
        aliase.append(re.sub(r"\s+", " ", f.group(0)).strip())

    return UrteilID(gericht=gericht, aktenzeichen=az, aliase=tuple(dict.fromkeys(aliase)))


# --- Verwaltungs-Parser (Spez. 1.2) -----------------------------------------

_VERW_QUELLEN = ("UStAE", "AEAO", "GewStR", "KStR", "EStR", "LStR", "ErbStR", "KStH", "EStH")


def parse_verwaltung(text: str) -> VerwID | None:
    """Parst Verwaltungsfundstellen (UStAE-Abschnitt, Richtlinien-R, AEAO zu §)."""
    quelle = next((q for q in _VERW_QUELLEN if re.search(rf"\b{q}\b", text)), None)
    if not quelle:
        return None
    m = re.search(r"Abschn\.?\s*([0-9]+(?:\.[0-9]+)*[a-z]?)", text)
    if m:
        return VerwID(quelle, m.group(1))
    m = re.search(r"\bR\s*([0-9]+(?:\.[0-9]+)*[a-z]?)", text)
    if m:
        return VerwID(quelle, "R" + m.group(1))
    m = re.search(r"zu\s*§\s*(\d+\w*)", text)
    if m:
        return VerwID(quelle, "zu§" + m.group(1))
    return VerwID(quelle, "?")


# --- Fundstellen-Kanonisierung (Katalog-Ankopplung, Spez. Kap. 6 Schritt 5) --

def kanonisiere_fundstelle(typ: str, fundstelle: str) -> str | None:
    """Versucht, eine Katalog-Fundstelle auf eine kanonische ID zu heben.

    Liefert die ID-Zeichenkette oder None, wenn nicht eindeutig parsebar
    (z. B. BMF-Schreiben ohne eigenes Schema). Niemals geraten.
    """
    text = fundstelle.strip()
    norm = _scan_pattern(DEFAULT_LAWMAP).match(text)
    if norm:
        # Das Feld trägt genau eine ID. Eine Erläuterung in Klammern ist
        # zulässig, mehrere zitierte Normen haben keine eindeutige Einzel-ID.
        anker = parse_normzitat(norm.group())
        if len(anker) == 1 and anker[0].norm_id and not _ZITAT_SPAN.search(text, norm.end()):
            return anker[0].norm_id.to_id()
        return None
    if _GERICHT_RE.match(text):
        uid = parse_entscheidung(text)
        if uid:
            return uid.to_id()
    verw = parse_verwaltung(text)
    if verw:
        return verw.to_id()
    return None
