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
from dataclasses import dataclass
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


def _scan_pattern(lawmap: LawMap) -> re.Pattern[str]:
    pat = _SCAN_PATTERN_CACHE.get(lawmap)
    if pat is None:
        law_alt = "|".join(re.escape(a) for a in lawmap.alle_aliase())
        # {0,80}? statt *?: echte Zitate zwischen §-Nummer und Gesetz sind kurz
        # (laengste Kette < 60 Zeichen). Die Begrenzung verhindert einen polynomialen
        # Lazy-Scan auf gecraftetem Nutzertext ("§1 " + langer Nicht-Gesetz-Schwanz),
        # da finde_normzitate im mitLLM-Pfad ueber den GANZEN Text laeuft.
        # Der (?!…)-Wächter stoppt am SATZENDE (Punkt + Whitespace + Großbuchstabe):
        # „… § 17 Abs. 2. Die AO regelt …" darf NICHT zu norm:AO:17:abs2 werden –
        # das Gesetz gehört zum Folgesatz. Abkürzungspunkte in echten Zitaten
        # („Abs. 2", „S. 1") folgt immer eine Ziffer, die bleiben erlaubt.
        # Bewusster Verzicht: Ordinalformen („2. Halbsatz AO") werden dadurch
        # nicht gescannt – die parste auch _DEPTH_RE nie (kein Recall-Verlust).
        pat = re.compile(
            rf"(?:§§?|\bArt\.)\s*\d+"
            rf"(?:(?!\.\s+[A-ZÄÖÜ])[\w.\s,–\-]){{0,80}}?"
            rf"\b(?:{law_alt})\b")
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

_DEPTH_RE = re.compile(
    rf"(?P<kind>{_TIEFE_KINDS})\.?\s*"
    rf"(?P<v1>{_TIEFE_WERT})"
    rf"(?:\s*(?:bis|–|—|-)\s*(?P<v2>{_TIEFE_WERT})"
    # „Satz 1 und 2" / „Nr. 2, 3 und 5": Aufzählung von Tiefenwerten (Spez. 2.2,
    # Listen-Expansion). (?!\w) verhindert, dass der Einzelbuchstaben-Wert das
    # erste Zeichen eines Folgeworts frisst („Satz 1 und folgende").
    rf"|(?P<liste>(?:\s*(?:,|und)\s*(?:{_TIEFE_WERT})(?!\w))+))?"
)

_PARA_NUM = re.compile(r"\d+[a-z]*")


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
        a, b = int(v1), int(v2)
        if a <= b:
            return [str(x) for x in range(a, b + 1)]
    if len(v1) == 1 and len(v2) == 1 and v1.isalpha() and v2.isalpha():
        a, b = ord(v1), ord(v2)
        if a <= b:
            return [chr(x) for x in range(a, b + 1)]
    return [v1, v2]


# Bereichsmarker zwischen §-Nummern: „bis", Gedankenstriche – und der einfache
# Bindestrich, aber NUR zwischen zwei Nummern(-Suffixen) („14-17", „64a-65").
# Der Lookahead auf eine Ziffer schützt Wortkopplungen („§ 6b-Rücklage") davor,
# als Bereich gelesen zu werden.
_PARA_BEREICH = re.compile(r"\bbis\b|–|—|[0-9a-z]\s*-\s*(?=\d)")


def _expand_paras(para_part: str) -> tuple[list[str], bool]:
    """Expandiert die §-Nummern (Liste/Bereich). Returns (paras, expandiert?)."""
    nums = _PARA_NUM.findall(para_part)
    if not nums:
        return [], False
    if _PARA_BEREICH.search(para_part) and len(nums) >= 2:
        return _expand_range(nums[0], nums[-1]), True
    if len(nums) > 1:                       # Liste: "14, 17"
        return nums, True
    return [nums[0]], False


def _parse_depth(depth_part: str) -> tuple[list[tuple[tuple[str, str], ...]], bool]:
    """Parst Tiefenmarker, kartesisch über Bereiche/Listen expandiert.

    Returns (Liste von Tiefen-Tupeln, expandiert?).
    """
    achsen: list[list[tuple[str, str]]] = []
    expandiert = False
    for m in _DEPTH_RE.finditer(depth_part):
        label = _label(m.group("kind"))
        if m.group("v2"):
            expandiert = True
            werte = _expand_range(m.group("v1"), m.group("v2"))
        elif m.group("liste"):
            # „Satz 1 und 2" → s1, s2; „Nr. 2, 3 und 5" → nr2, nr3, nr5.
            # findall mit Gruppe holt nur die Werte, nicht die Trenner.
            expandiert = True
            werte = [m.group("v1")] + re.findall(
                rf"(?:,|und)\s*({_TIEFE_WERT})", m.group("liste"))
        else:
            werte = [m.group("v1")]
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
        s = re.sub(r"^(Art\.?|Artikel)\s*", "", s).strip()
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

    paras, p_exp = _expand_paras(para_part)
    tiefen, t_exp = _parse_depth(depth_part)
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
# „und"-Aufzählungen: im Fließtext wäre „Satz 1 und 2 weitere Personen" sonst
# falsch als Listen-Zitat gelesen – die Listen-Expansion macht parse_normzitat.
_ZITAT_SPAN = re.compile(
    rf"§§?\s*\d+[a-z]*"
    rf"(?:\s+(?:{_TIEFE_KINDS})\.?"
    rf"\s*(?:{_TIEFE_WERT})(?!\w)(?:\s*(?:bis|–|—|-)\s*(?:{_TIEFE_WERT})(?!\w))?)*"
)


def extrahiere_anker(
    text: str, default_law: str | None = None, lawmap: LawMap = DEFAULT_LAWMAP
) -> list[GeparsterAnker]:
    """Extrahiert Norm-Anker aus Fließtext – inkl. intra-law-Verweisen.

    Steht hinter einem „§…"-Zitat ein bekanntes Gesetzeskürzel, wird cross-law
    aufgelöst; fehlt es, wird ``default_law`` angenommen (typisch in Gesetzestexten,
    die intern „§ 15" ohne Gesetzesnamen zitieren). Damit entstehen die
    ``verweist_auf``-Kanten des Normgraphen.
    """
    out: list[GeparsterAnker] = []
    for m in _ZITAT_SPAN.finditer(text):
        zitat = m.group(0)
        nach = text[m.end(): m.end() + 40]
        law_m = re.match(r"\s+([A-ZÄÖÜ][A-Za-zÄÖÜ0-9]+)", nach)
        law = lawmap.resolve(law_m.group(1)) if law_m else None
        if law and law_m is not None:
            out.extend(parse_normzitat(f"{zitat} {law_m.group(1)}", lawmap))
        elif default_law:
            for a in parse_normzitat(zitat, lawmap):
                if a.norm_id:
                    nid = NormID(
                        law=default_law, para=a.norm_id.para, artikel=a.norm_id.artikel,
                        tiefe=a.norm_id.tiefe, offener_bereich=a.norm_id.offener_bereich,
                    )
                    out.append(GeparsterAnker(norm_id=nid, roh=a.roh, konfidenz="mittel", relativ=False))
    return out


def finde_normzitate(text: str, lawmap: LawMap = DEFAULT_LAWMAP) -> list[GeparsterAnker]:
    """Scannt FREITEXT nach Normzitaten (§…LAW / Art.…LAW) und parst jedes.

    Anders als ``parse_normzitat`` (das einen einzelnen, sauberen Zitatstring
    erwartet) findet diese Funktion Zitate inmitten von Fließtext – nötig für die
    Norm-Nähe des Unknown-Gates und später für den Kantenaufbau im Graphen.
    """
    pat = _scan_pattern(lawmap)
    anker: list[GeparsterAnker] = []
    for m in pat.finditer(text):
        anker.extend(parse_normzitat(m.group(0), lawmap))
    return anker


# --- Rekonstruktion (Round-Trip /T10/) --------------------------------------

_TIEFE_WORT = {"abs": "Abs.", "nr": "Nr.", "s": "Satz", "lit": "Buchst.", "hs": "Halbs.", "alt": "Alt."}


def norm_zu_zitat(nid: NormID) -> str:
    """Rekonstruiert ein menschenlesbares Zitat aus einer NormID (/T10/)."""
    kopf = f"Art. {nid.artikel}" if nid.artikel else f"§ {nid.para}"
    teile = [f"{_TIEFE_WORT[label]} {wert}" for label, wert in nid.tiefe]
    ff = " ff." if nid.offener_bereich else ""
    return " ".join([kopf, *teile]).strip() + ff + f" {nid.law}"


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
    if re.match(r"^\s*(§|Art\.?|Artikel)\b", text):
        anker = parse_normzitat(text)
        if anker and anker[0].norm_id:
            return anker[0].norm_id.to_id()
    if _GERICHT_RE.match(text):
        uid = parse_entscheidung(text)
        if uid:
            return uid.to_id()
    verw = parse_verwaltung(text)
    if verw:
        return verw.to_id()
    return None
