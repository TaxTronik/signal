"""Komposita-Mining: kuratierbare Alias-Vorschläge aus dem Gesetzes-Korpus (/R40/).

WARUM: Der Schicht-1-Matcher toleriert Flexion sowie Getrennt-/Bindestrich-
Schreibung, aber bewusst KEINE echte Zusammenschreibung („Kassennachschau" für
„Kassen-Nachschau", s. matcher.py) – solche Varianten gehören als eigene Aliase
in den kuratierten Katalog. Dieses OFFLINE-Skript schließt die /R40/-Lücke von
der Datenseite: Es sammelt deutsche Komposita aus den Normtexten des Graphen,
deren KOPF (Grundwort; Kopf-Prinzip des deutschen Kompositums) eine
Katalog-Oberfläche ist, und schreibt sie als ausdrücklich UNGEPRÜFTE Vorschläge
nach catalog/vorschlaege/. Es schreibt NIE in den Katalog und NIE in den
Graphen – der Katalog bleibt Single Source of Truth, Übernahme nur kuratiert.

Kandidaten-Logik (Kopf-Richtung, Default):
  Token T ist Kandidat für Begriff B mit Ein-Wort-Oberfläche O (bzw. dem KOPF
  einer Mehrwort-Oberfläche, s. u.), wenn
    * T case-insensitiv auf O endet – flexionstolerant nach den Konventionen
      des Matchers (Stamm + optionale Flexionsendung, risk_layer.matcher),
      und len(T) > len(O)+2,
    * der Vorsatz (T minus O) >= 3 Zeichen hat und plausibel ist: direkt
      angefügt oder mit Fugenelement (s|es|n|en|e|er|-),
    * der Vorsatz im Korpus belegt ist: als VOLLES Wort, oder – nur wenn ein
      Fugenelement die Kompositumsgrenze markiert – ent-fugt als Wortanfang
      eines ANDEREN Wortes (Selbstbeleg durch T/Flexionsformen von T zählt
      nicht, sonst wäre der Filter vakuum-wahr). Genau dieser Filter sortiert
      Zufallsenden aus („Bewirtschaftung" endet auf „…haftung", aber
      „Bewirtsc" ist weder Wort noch ent-fugbarer Wortanfang),
    * der Vorsatz KEINE bloße Verbpartikel ist („Einschätzung" ist Derivation
      von einschätzen, kein Kompositum mit Kopf „Schätzung"),
    * T nicht bereits exakt als Oberfläche/Alias im Katalog steht.

  KOPF einer Mehrwort-Oberfläche (Iteration am echten Korpus, s. u.): das
  letzte Wort – AUSSER die Oberfläche enthält ein Funktionswort (Genitiv-/
  Präpositionalgefüge): dann ist der semantische Kopf das Wort DAVOR
  („Mehrheit der Stimmrechte" → Kopf „Mehrheit", nicht „Stimmrechte";
  „finanzielle Eingliederung in das Unternehmen" → „Eingliederung", nicht
  „Unternehmen"). Das blanke letzte Wort lieferte am echten Korpus fast nur
  Zufallstreffer („Finanzunternehmen", „Handelsvertreter", „…verhältnisse").

  Träg die Oberfläche inhaltliche Begleiter (Modifikatoren wie „verdeckte",
  „gewerbliche"), trägt der Kopf allein die Bedeutung nicht: der Vorsatz muss
  dann an ein Inhaltswort IRGENDEINER Oberfläche desselben Begriffs anklingen
  (gemeinsamer Wortanfang >= 4; „Zwischen|einkünfte" klingt an Alias
  „Zwischengesellschaft" an, „Berufs|tätigkeit" an nichts von „gewerbliche
  Tätigkeit" → fällt). Ein-Wort-Oberflächen brauchen keinen Anklang.

Bestimmungswort-Richtung (--auch-praefix, konservativer): T beginnt mit O,
das Grundwort (Rest) muss als VOLLES Korpuswort belegt sein. Semantisch
schwächer (das Kompositum bezeichnet dann etwas ANDERES als B – das
„Kassennachschauprotokoll" ist nicht die Kassennachschau), daher nur auf Flag.

Scoring je Vorschlag: Häufigkeit im Korpus / Länge des Vorsatzes (kürzer =
näher am Begriffskern = besser); Flexionsformen desselben Kandidaten werden
zu EINEM Vorschlag aggregiert (Häufigkeitssumme); je Katalogbegriff die Top-N.

Aufruf:
  python tools/komposita_mining.py --graph corpus/graph.sqlite
  python tools/komposita_mining.py --graph ... --auch-praefix --top 5 --json

Exit-Codes: 0 = ok, 2 = Fehler (Graph fehlt, Katalog kaputt, leeres Vokabular).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._common import utf8_konsole                      # noqa: E402

from risk_layer.catalog import Katalog, KatalogFehler, lade_katalog  # noqa: E402
from risk_layer.matcher import _FLEX_ENDUNGEN, _stamm        # noqa: E402
from risk_layer.persistenz import atomar_schreiben           # noqa: E402
from risk_layer.sicherung import zeitstempel_dateiname       # noqa: E402

DEFAULT_GRAPH = Path("corpus") / "graph.sqlite"
DEFAULT_AUSGABE_DIR = Path("catalog") / "vorschlaege"

HINWEIS = ("UNGEPRÜFT — Alias-Vorschläge aus Korpus-Mining (/R40/); "
           "Übernahme nur kuratiert in begriffe.yaml; NICHT maschinell eingebunden")

# Substantivisches Wort-Token, optional mit Binnen-Bindestrich(en)
# („Umsatzsteuer-Nachschau"). Bewusst NUR großgeschriebene Glieder – Komposita
# sind Substantive; alles andere ist für die Kandidatensuche Rauschen.
_TOKEN = re.compile(r"[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)*")
# Alle Wortformen (auch kleingeschrieben) für den Vorsatz-Belegfilter.
_WORTFORM = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")
# Form eines gültigen Kopf-Wortes (sortiert z. B. „AStG"/„vGA"-Kürzel aus).
_KOPF_FORM = re.compile(r"[A-ZÄÖÜ][a-zäöüß]+")

MIN_TOKEN_LAENGE = 8     # Vokabular: kürzere Tokens sind selten echte Komposita
MIN_VORSATZ = 3          # Vorsatz unterhalb ist kein tragfähiges Bestimmungswort
MIN_KOPF = 4             # zu kurze Katalog-Köpfe erzeugten nur Zufallsenden
MIN_ANKLANG = 4          # gemeinsamer Wortanfang Vorsatz↔Begleiter
KONTEXT_MAX = 120        # Belegausschnitt-Obergrenze (Zeichen)

# Fugenelemente, LÄNGSTE zuerst (sonst fräste „s" das „es" an).
_FUGEN = ("es", "en", "er", "s", "n", "e")

# Verbpartikeln/Präfixe: „Ein|schätzung", „Nach|prüfung" sind DERIVATIONEN des
# Verbs (einschätzen, nachprüfen), keine Komposita mit dem Katalogkopf –
# semantisch verschoben, daher kein Alias-Vorschlag. BEWUSST OHNE „zwischen"
# (Zwischeneinkünfte/Zwischengesellschaft sind echte nominale Komposita).
_VERBPARTIKELN = frozenset({
    "ab", "an", "auf", "aus", "bei", "dar", "durch", "ein", "ent", "fort",
    "ge", "her", "hin", "hinzu", "los", "mit", "nach", "neben", "nieder",
    "rück", "ueber", "über", "um", "unter", "ur", "ver", "vor", "weg",
    "weiter", "wider", "wieder", "zer", "zu", "zurück", "zusammen",
})

# Funktionswörter: markieren Genitiv-/Präpositionalgefüge in Mehrwort-
# Oberflächen – der semantische Kopf steht DAVOR (s. Modul-Docstring).
_FUNKTIONSWOERTER = frozenset({
    "der", "die", "das", "des", "dem", "den", "von", "vom", "in", "im",
    "an", "am", "auf", "aus", "bei", "durch", "für", "mit", "nach", "über",
    "um", "unter", "zu", "zum", "zur", "als", "ohne", "gegen",
})

_WORT_PUTZ = ",.;:()/%\"'„“‚‘"   # Satzzeichen an Oberflächen-Wörtern


# ---------------------------------------------------------------------------
# Korpuszugriff (READ-ONLY)
# ---------------------------------------------------------------------------

def lade_paragraph_texte(graph_pfad: Path) -> list[tuple[str, str]]:
    """Liest (norm_id, text) der §-Knoten – strikt read-only (URI mode=ro).

    Nur §-Ebene (absatz/nummer IS NULL): Absatz-/Nummer-Knoten wiederholen den
    Text ihres §-Elterns (s. graph.py, Bau-Hinweis) und würden die Häufigkeiten
    künstlich vervielfachen. Schema laut risk_layer/graph.py (_SCHEMA: knoten).
    ORDER BY id: deterministische Belegauswahl über Läufe hinweg.
    """
    con = sqlite3.connect(f"file:{graph_pfad.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT id, text FROM knoten "
            "WHERE absatz IS NULL AND nummer IS NULL AND id NOT LIKE 'norm:%:%:%' ORDER BY id").fetchall()
    finally:
        con.close()
    return [(r[0], r[1] or "") for r in rows]


# ---------------------------------------------------------------------------
# Vokabular
# ---------------------------------------------------------------------------

def satz_ausschnitt(text: str, start: int, ende: int, max_laenge: int = KONTEXT_MAX) -> str:
    """Satzausschnitt um [start:ende), auf ``max_laenge`` Zeichen gekürzt.

    Satzgrenzen heuristisch an .!?; und Zeilenumbruch; ein zu langer Satz wird
    symmetrisch um das Token gefenstert und mit „…" markiert.
    """
    grenzen = ".!?;\n"
    a = max((text.rfind(z, 0, start) for z in grenzen), default=-1) + 1
    naechste = [text.find(z, ende) for z in grenzen]
    naechste = [n for n in naechste if n != -1]
    b = (min(naechste) + 1) if naechste else len(text)
    satz = " ".join(text[a:b].split())  # Whitespace normalisieren
    if len(satz) <= max_laenge:
        return satz
    # Fenster um das Token zentrieren (Offsets im normalisierten Satz suchen)
    token = " ".join(text[start:ende].split())
    pos = satz.find(token)
    if pos == -1:
        return satz[: max_laenge - 1] + "…"
    rand = max(0, (max_laenge - len(token) - 2) // 2)
    von = max(0, pos - rand)
    bis = min(len(satz), pos + len(token) + rand)
    aus = satz[von:bis]
    if von > 0:
        aus = "…" + aus
    if bis < len(satz):
        aus = aus + "…"
    return aus


def baue_vokabular(zeilen: list[tuple[str, str]]) -> tuple[dict, set, list]:
    """Vokabular + Wortform-Beleg aus (norm_id, text)-Zeilen.

    Liefert:
      * vokabular: {token_lower: {"original", "haeufigkeit", "beleg"}}
        – nur substantivische Tokens >= MIN_TOKEN_LAENGE; Beleg = ERSTE
        Fundstelle (norm_id + Satzausschnitt), deterministisch durch die
        sortierte Zeilenreihenfolge des Aufrufers,
      * wortformen: set ALLER Wortformen (lower, >= 3 Zeichen) – „ist Wort?",
      * wortformen_sortiert: dieselben sortiert – „ist Wortanfang?" via bisect.
    """
    vokabular: dict[str, dict] = {}
    wortformen: set[str] = set()
    for norm_id, text in zeilen:
        for w in _WORTFORM.findall(text):
            wortformen.add(w.lower())
        for m in _TOKEN.finditer(text):
            token = m.group(0)
            if len(token) < MIN_TOKEN_LAENGE:
                continue
            schluessel = token.lower()
            eintrag = vokabular.get(schluessel)
            if eintrag is None:
                vokabular[schluessel] = {
                    "original": token,
                    "haeufigkeit": 1,
                    "beleg": {"norm_id": norm_id,
                              "kontext": satz_ausschnitt(text, m.start(), m.end())},
                }
            else:
                eintrag["haeufigkeit"] += 1
    return vokabular, wortformen, sorted(wortformen)


def ist_wortanfang(praefix: str, wortformen_sortiert: list[str],
                   *, ausschluss_praefix: str = "") -> bool:
    """True, wenn ein Korpuswort mit ``praefix`` beginnt – Selbstbeleg zählt nicht.

    ``ausschluss_praefix`` ist der Stamm des Kandidaten-Tokens: Wörter, die
    damit beginnen (das Token selbst und seine Flexionsformen), bestätigen den
    Vorsatz NICHT – sonst bestätigte jedes Kompositum seinen eigenen Vorsatz.
    """
    i = bisect_left(wortformen_sortiert, praefix)
    while i < len(wortformen_sortiert) and wortformen_sortiert[i].startswith(praefix):
        wort = wortformen_sortiert[i]
        if not (ausschluss_praefix and wort.startswith(ausschluss_praefix)):
            return True
        i += 1
    return False


def vorsatz_plausibel(vorsatz: str, wortformen: set, wortformen_sortiert: list[str],
                      *, token_lower: str = "") -> bool:
    """Prüft das Bestimmungswort eines Kandidaten (Rausch-Filter, s. Docstring).

    Direkt angefügte Vorsätze müssen ein VOLLES Korpuswort sein; nur wenn ein
    Fugenelement (s|es|n|en|e|er|-) die Kompositumsgrenze markiert, genügt der
    ent-fugte Stamm als Wortanfang eines anderen Korpuswortes. Verbpartikeln
    („ein", „nach", „über") fallen – Derivation, kein Kompositum.
    """
    if len(vorsatz) < MIN_VORSATZ or not vorsatz[:1].isupper():
        return False
    v = vorsatz.lower()
    if v in _VERBPARTIKELN:
        return False
    ausschluss = _stamm(token_lower) if token_lower else ""
    # direkt angefügt: nur ein volles Korpuswort trägt
    if v in wortformen:
        return True
    # Fugenelement am Ende: ent-fugter Stamm darf auch bloßer Wortanfang sein
    entfugt = []
    if v.endswith("-"):
        entfugt.append(v.rstrip("-"))
    for fuge in _FUGEN:
        if v.endswith(fuge) and len(v) - len(fuge) >= MIN_VORSATZ:
            entfugt.append(v[: -len(fuge)])
    for kandidat in entfugt:
        if kandidat in _VERBPARTIKELN:
            continue
        if kandidat in wortformen or ist_wortanfang(
                kandidat, wortformen_sortiert, ausschluss_praefix=ausschluss):
            return True
    return False


# ---------------------------------------------------------------------------
# Kandidatensuche
# ---------------------------------------------------------------------------

def _surface_woerter(surface: str) -> list[str]:
    """Wörter einer Oberfläche – Trennung wie im Matcher (Leerraum/Bindestrich),
    Satzzeichen abgestreift."""
    woerter = []
    for w in re.split(r"[\s\-]+", surface.strip()):
        w = w.strip(_WORT_PUTZ)
        if w:
            woerter.append(w)
    return woerter


def _gemeinsamer_anfang(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def kopf_einer_oberflaeche(surface: str) -> tuple[str | None, bool]:
    """(Kopf, braucht_anklang) einer Oberfläche – Kopf-Prinzip des Kompositums.

    Kopf = letztes Wort; bei Genitiv-/Präpositionalgefügen (erstes Funktions-
    wort) das Wort DAVOR („Mehrheit der Stimmrechte" → „Mehrheit"). None, wenn
    der Kopf kein Substantiv ist (kleingeschrieben, Kürzel wie „AStG") oder zu
    kurz. ``braucht_anklang`` = die Oberfläche trägt NEBEN dem Kopf weitere
    Nicht-Funktionswörter, der Kopf allein also nicht die Bedeutung – auch ein
    Normzitat zählt als solches: „Haftung nach § 69 AO" meint nicht jede
    „…haftung" (sonst kämen „Sachhaftung" § 76 AO und „Veranlasserhaftung"
    § 10b EStG als Vorschläge für die Vertreterhaftung durch).
    """
    woerter = _surface_woerter(surface)
    if not woerter:
        return None, False
    idx = len(woerter) - 1
    for i, w in enumerate(woerter):
        if i >= 1 and w.lower() in _FUNKTIONSWOERTER:
            idx = i - 1
            break
    kopf = woerter[idx]
    if len(kopf) < MIN_KOPF or not _KOPF_FORM.fullmatch(kopf):
        return None, False
    begleiter = [w for i, w in enumerate(woerter)
                 if i != idx and w.lower() not in _FUNKTIONSWOERTER]
    return kopf, bool(begleiter)


def _koepfe_je_eintrag(eintrag) -> list[tuple[str, str, str, bool]]:
    """(stamm_lower, kopf, oberflaeche, braucht_anklang) je Eintrag, dedupliziert.

    Der Kopf wird wie im Matcher flexionstolerant behandelt (``_stamm``).
    Liefert mehrere Oberflächen denselben Kopf-Stamm, gewinnt die SCHWÄCHSTE
    Anforderung (eine Ein-Wort-Oberfläche macht den Kopf anklang-frei).
    """
    je_stamm: dict[str, tuple[str, str, str, bool]] = {}
    for surface in eintrag.oberflaechen:
        kopf, braucht_anklang = kopf_einer_oberflaeche(surface)
        if kopf is None:
            continue
        stamm = _stamm(kopf).lower()
        alt = je_stamm.get(stamm)
        if alt is None or (alt[3] and not braucht_anklang):
            je_stamm[stamm] = (stamm, kopf, surface, braucht_anklang)
    return list(je_stamm.values())


def _anklang_pool(eintrag) -> set[str]:
    """Inhaltswörter ALLER Oberflächen eines Eintrags (für den Anklang-Filter).

    Eintragsweit statt je Oberfläche: „Zwischen|einkünfte" (zu „passive
    Einkünfte") klingt am Alias „Zwischengesellschaft" desselben Begriffs an.
    """
    pool: set[str] = set()
    for surface in eintrag.oberflaechen:
        for w in _surface_woerter(surface):
            wl = w.lower()
            if len(wl) >= MIN_ANKLANG and wl not in _FUNKTIONSWOERTER:
                pool.add(wl)
    return pool


def _vorsatz_varianten(vorsatz_lower: str) -> list[str]:
    """Vorsatz roh + ent-fugte Stämme (für den Anklang-Vergleich)."""
    v = vorsatz_lower.rstrip("-")
    varianten = [v]
    for fuge in _FUGEN:
        if v.endswith(fuge) and len(v) - len(fuge) >= MIN_VORSATZ:
            varianten.append(v[: -len(fuge)])
    return varianten


def _klingt_an(vorsatz_lower: str, pool: set[str]) -> bool:
    return any(_gemeinsamer_anfang(var, w) >= MIN_ANKLANG
               for var in _vorsatz_varianten(vorsatz_lower) for w in pool)


def _einwort_oberflaechen(eintrag) -> list[tuple[str, str]]:
    """(oberflaeche_norm, oberflaeche) der Ein-Wort-Oberflächen (--auch-praefix).

    Normalisiert ohne Bindestriche, damit „Kassen-Nachschau" das zusammen-
    geschriebene „Kassennachschauprotokoll" findet.
    """
    out = []
    for surface in eintrag.oberflaechen:
        s = surface.strip()
        if not s or any(c.isspace() for c in s):
            continue
        norm = s.replace("-", "").lower()
        if len(norm) >= 6 and s[:1].isupper():
            out.append((norm, surface))
    return out


def finde_vorschlaege(katalog: Katalog, vokabular: dict, wortformen: set,
                      wortformen_sortiert: list[str], *, auch_praefix: bool = False,
                      min_haeufigkeit: int = 2, top: int = 10) -> list[dict]:
    """Alle kuratierbaren Vorschläge, je Katalogbegriff die Top-N nach Score.

    Score = Häufigkeit / Länge des Vorsatzes: häufige Komposita mit KURZEM
    Vorsatz liegen näher am Begriffskern. Flexionsformen desselben Kandidaten
    („Beteiligungserwerb/-erwerbe/-erwerbs") werden zu EINEM Vorschlag
    aggregiert (Häufigkeitssumme; Originalschreibweise der häufigsten Form).
    Reihenfolge deterministisch: Begriffe in Katalogreihenfolge, Vorschläge
    nach Score absteigend, dann alphabetisch.
    """
    # bereits kuratierte Oberflächen: dafür braucht es keinen Vorschlag
    im_katalog = {s.strip().lower() for e in katalog for s in e.oberflaechen}

    vorschlaege: list[dict] = []
    for eintrag in katalog:
        koepfe = _koepfe_je_eintrag(eintrag)
        pool = _anklang_pool(eintrag)
        einwort = _einwort_oberflaechen(eintrag) if auch_praefix else []
        je_familie: dict[str, dict] = {}     # Flexionsfamilie → aggregierter Vorschlag

        for t_low in sorted(vokabular):      # sortiert ⇒ deterministische Aggregation
            vok = vokabular[t_low]
            if t_low in im_katalog:
                continue

            # -- Kopf-Richtung: Katalogbegriff ist GRUNDWORT des Kompositums --
            for stamm, kopf, surface, braucht_anklang in koepfe:
                rest = _flex_rest(t_low, stamm)
                if rest is None or len(t_low) <= len(kopf) + 2:
                    continue
                vorsatz = vok["original"][: len(t_low) - len(stamm) - len(rest)]
                if len(vorsatz) < MIN_VORSATZ:
                    continue
                if not vorsatz_plausibel(vorsatz, wortformen, wortformen_sortiert,
                                         token_lower=t_low):
                    continue
                if braucht_anklang and not _klingt_an(vorsatz.lower(), pool):
                    continue
                _merke(je_familie, t_low, vok, eintrag.id, surface,
                       richtung="kopf", vorsatz_laenge=len(vorsatz.rstrip("-")))

            # -- Bestimmungswort-Richtung (nur --auch-praefix, konservativ) --
            t_norm = t_low.replace("-", "")
            for o_norm, surface in einwort:
                if not t_norm.startswith(o_norm) or len(t_norm) <= len(o_norm) + 2:
                    continue
                grundwort = t_norm[len(o_norm):]
                varianten = [grundwort] + [grundwort[len(f):] for f in _FUGEN
                                           if grundwort.startswith(f)]
                # konservativer als die Kopf-Richtung: das Grundwort muss als
                # VOLLES Korpuswort belegt sein (kein bloßer Wortanfang).
                if not any(len(g) >= 4 and g in wortformen for g in varianten):
                    continue
                _merke(je_familie, t_low, vok, eintrag.id, surface,
                       richtung="praefix", vorsatz_laenge=len(grundwort))

        # erst die Häufigkeitsschwelle, DANN Top-N – sonst belegten seltene
        # Kandidaten Top-Plätze, nur um anschließend herauszufallen.
        kandidaten = sorted(
            (v for v in je_familie.values() if v["haeufigkeit"] >= min_haeufigkeit),
            key=lambda v: (-v["score"], v["alias_vorschlag"].lower()))
        for v in kandidaten[:top]:
            v.pop("_haeufigkeit_beste", None)
            vorschlaege.append(v)
    return vorschlaege


def _flex_rest(t_low: str, stamm: str) -> str | None:
    """Endet ``t_low`` auf ``stamm`` (+ optionale Flexionsendung)? → die Endung.

    Flexionstoleranz wie im Matcher (_FLEX_OPT): „mehrstimmrechten" endet auf
    Stamm „stimmrecht" + „en". None, wenn der Kopf nicht am Wortende steht.
    """
    if t_low.endswith(stamm):
        return ""
    for endung in _FLEX_ENDUNGEN:
        if t_low.endswith(stamm + endung):
            return endung
    return None


def _merke(je_familie: dict, t_low: str, vok: dict, katalog_id: str, surface: str,
           *, richtung: str, vorsatz_laenge: int) -> None:
    """Aggregiert einen Treffer in seine Flexionsfamilie (Stamm-Schlüssel).

    Die Familie summiert die Häufigkeit; Originalschreibweise und Beleg kommen
    von der häufigsten Form. Score wird nach jeder Aggregation neu berechnet.
    """
    schluessel = f"{richtung}:{_stamm(t_low)}"
    eintrag = je_familie.get(schluessel)
    if eintrag is None:
        eintrag = je_familie[schluessel] = {
            "katalog_id": katalog_id,
            "alias_vorschlag": vok["original"],
            "haeufigkeit": 0,
            "beleg": dict(vok["beleg"]),
            # Kuratierhilfen (additiv): welche Oberfläche trug, welche Richtung,
            # wie gescort wurde – erleichtert die fachliche Prüfung.
            "oberflaeche": surface,
            "richtung": richtung,
            "score": 0.0,
            "_haeufigkeit_beste": 0,
        }
    eintrag["haeufigkeit"] += vok["haeufigkeit"]
    if vok["haeufigkeit"] > eintrag["_haeufigkeit_beste"]:
        eintrag["_haeufigkeit_beste"] = vok["haeufigkeit"]
        eintrag["alias_vorschlag"] = vok["original"]
        eintrag["beleg"] = dict(vok["beleg"])
        eintrag["oberflaeche"] = surface
    eintrag["score"] = round(eintrag["haeufigkeit"] / max(1, vorsatz_laenge), 4)


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def baue_dokument(vorschlaege: list[dict], *, katalog: Katalog, graph_pfad: Path,
                  zeitstempel: str, parameter: dict) -> dict:
    return {
        "hinweis": HINWEIS,
        "erstellt_am": zeitstempel,
        "graph": str(graph_pfad),
        "katalog_version": katalog.version,
        "parameter": parameter,
        "anzahl": len(vorschlaege),
        "vorschlaege": vorschlaege,
    }


def schreibe_yaml(dokument: dict, pfad: Path) -> None:
    """Schreibt das Vorschlagsdokument atomar – Header-Kommentar zuerst."""
    kopf = ("# =========================================================================\n"
            f"# {HINWEIS}\n"
            "# =========================================================================\n")
    atomar_schreiben(pfad, kopf + yaml.safe_dump(
        dokument, allow_unicode=True, sort_keys=False, width=100))


def main(argv: list[str] | None = None) -> int:
    utf8_konsole()
    ap = argparse.ArgumentParser(
        description="Komposita-Mining: kuratierbare Alias-Vorschläge aus dem "
                    "Normgraph-Korpus (/R40/). Schreibt NUR nach catalog/vorschlaege/.")
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH),
                    help="Pfad zum (read-only geöffneten) graph.sqlite")
    ap.add_argument("--katalog", default=None,
                    help="Katalog-YAML (Default: catalog/begriffe.yaml)")
    ap.add_argument("--top", type=int, default=10,
                    help="maximale Vorschläge je Katalogbegriff (Default: 10)")
    ap.add_argument("--min-haeufigkeit", type=int, default=2,
                    help="Mindest-Häufigkeit eines Kandidaten im Korpus (Default: 2)")
    ap.add_argument("--auch-praefix", action="store_true",
                    help="zusätzlich Bestimmungswort-Richtung (Begriff als "
                         "Wortanfang) – semantisch schwächer, konservativ gefiltert")
    ap.add_argument("--json", action="store_true",
                    help="Dokument zusätzlich als JSON nach stdout")
    ap.add_argument("--ausgabe-dir", default=str(DEFAULT_AUSGABE_DIR),
                    help="Zielverzeichnis (Default: catalog/vorschlaege)")
    ap.add_argument("--zeitstempel",
                    help="fester ISO-Zeitstempel (reproduzierbare Läufe)")
    args = ap.parse_args(argv)

    graph_pfad = Path(args.graph)
    if not graph_pfad.exists():
        print(f"Graph nicht gefunden: {graph_pfad}", file=sys.stderr)
        return 2
    try:
        katalog = lade_katalog(args.katalog)
    except KatalogFehler as exc:
        print(f"Katalogfehler: {exc}", file=sys.stderr)
        return 2

    try:
        zeilen = lade_paragraph_texte(graph_pfad)
    except sqlite3.Error as exc:
        print(f"Graph nicht lesbar ({graph_pfad}): {exc}", file=sys.stderr)
        return 2
    if not zeilen:
        print("Graph enthält keine §-Knoten – nichts zu minen.", file=sys.stderr)
        return 2

    vokabular, wortformen, sortiert = baue_vokabular(zeilen)
    if not vokabular:
        print("Leeres Vokabular – Korpustexte ohne verwertbare Tokens.", file=sys.stderr)
        return 2

    vorschlaege = finde_vorschlaege(
        katalog, vokabular, wortformen, sortiert,
        auch_praefix=args.auch_praefix,
        min_haeufigkeit=args.min_haeufigkeit, top=args.top)

    ts = args.zeitstempel or datetime.now().isoformat(timespec="seconds")
    dokument = baue_dokument(
        vorschlaege, katalog=katalog, graph_pfad=graph_pfad, zeitstempel=ts,
        parameter={"top": args.top, "min_haeufigkeit": args.min_haeufigkeit,
                   "auch_praefix": bool(args.auch_praefix)})
    pfad = Path(args.ausgabe_dir) / f"komposita_{zeitstempel_dateiname(ts)}.yaml"
    schreibe_yaml(dokument, pfad)

    if args.json:
        print(json.dumps(dokument, ensure_ascii=False, indent=2))

    print(f"Komposita-Mining: {len(zeilen)} §-Knoten, {len(vokabular)} Vokabeln, "
          f"{len(vorschlaege)} Vorschläge → {pfad}", file=sys.stderr)
    je_begriff: dict[str, int] = {}
    for v in vorschlaege:
        je_begriff[v["katalog_id"]] = je_begriff.get(v["katalog_id"], 0) + 1
    for bid, n in je_begriff.items():
        print(f"  {bid}: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
