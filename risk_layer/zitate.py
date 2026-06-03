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

from .idschema import NormID, Quellentyp, UrteilID, VerwID
from .lawmap import DEFAULT_LAWMAP, LawMap

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
        pat = re.compile(rf"(?:§§?|\bArt\.)\s*\d+[\w.\s,–\-]{{0,80}}?\b(?:{law_alt})\b")
        _SCAN_PATTERN_CACHE[lawmap] = pat
    return pat


# --- Normzitat-Parser -------------------------------------------------------

# Tiefenmarker (Spez. 2.1). Längere Varianten zuerst (z. B. "Satz" vor "S").
_DEPTH_RE = re.compile(
    r"(?P<kind>Absatz|Abs|Nummer|Nr|Satz|S|Buchstabe|Buchst|lit|Halbsatz|Halbs|Alt|Var)\.?\s*"
    r"(?P<v1>\d+[a-z]?|[a-z])"
    r"(?:\s*(?:bis|–|—|-)\s*(?P<v2>\d+[a-z]?|[a-z]))?"
)

_PARA_NUM = re.compile(r"\d+[a-z]*")


@dataclass(frozen=True)
class GeparsterAnker:
    """Ergebnis je expandierter Norm-ID: ID + Beleg + Konfidenz/Flags (Spez. 2.5)."""

    norm_id: NormID | None
    roh: str
    konfidenz: str          # niedrig | mittel | hoch
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


def _expand_paras(para_part: str) -> tuple[list[str], bool]:
    """Expandiert die §-Nummern (Liste/Bereich). Returns (paras, expandiert?)."""
    nums = _PARA_NUM.findall(para_part)
    if not nums:
        return [], False
    if re.search(r"\bbis\b|–|—", para_part) and len(nums) >= 2:
        return _expand_range(nums[0], nums[-1]), True
    if len(nums) > 1:                       # Liste: "14, 17"
        return nums, True
    return [nums[0]], False


def _parse_depth(depth_part: str) -> tuple[list[tuple[tuple[str, str], ...]], bool]:
    """Parst Tiefenmarker, kartesisch über Bereiche expandiert.

    Returns (Liste von Tiefen-Tupeln, expandiert?).
    """
    achsen: list[list[tuple[str, str]]] = []
    expandiert = False
    for m in _DEPTH_RE.finditer(depth_part):
        label = _label(m.group("kind"))
        if m.group("v2"):
            expandiert = True
            werte = _expand_range(m.group("v1"), m.group("v2"))
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

    # 2) "ff."/"f." → offener Bereich
    offener = bool(re.search(r"\bff?\.", s))
    s = re.sub(r"\bff?\.", "", s).strip()

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


# Zitat-Span im Fließtext (mit/ohne Gesetzesnamen), für intra-/cross-law-Kanten
_ZITAT_SPAN = re.compile(
    r"§§?\s*\d+[a-z]*"
    r"(?:\s+(?:Abs(?:atz)?|Nr(?:ummer)?|Satz|S|Buchst(?:abe)?|lit|Halbs(?:atz)?|Alt|Var)\.?"
    r"\s*\d+[a-z]?(?:\s*(?:bis|–|—|-)\s*\d+[a-z]?)?)*"
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
        if law:
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
_AZ_RE = re.compile(r"[A-Za-z0-9][\w .\-]{0,40}?\d{1,4}/\d{2,4}")  # begrenzt (kein Lazy-Blowup)
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
