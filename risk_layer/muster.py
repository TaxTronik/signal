"""Schicht 1b – deterministische Beschreibungs-Muster + Streit-Verstärker.

Schicht 1 (matcher.py) findet nur WÖRTLICHE Schlüsselbegriffe. Reale Sachverhalte
benennen die Rechtsbegriffe aber selten – sie UMSCHREIBEN sie, oft als Negativbefund
("Belege fehlen", "Kassenführung nicht nachvollziehbar", "Abgaben nicht
weitergeleitet"). Diese Schicht schließt genau diese Lücke, weiterhin rein
deterministisch (kein Modell, reproduzierbar, kuratierbar):

  * MusterMatcher: Ko-Okkurrenz-Muster je Begriff. Ein Muster trifft, wenn in
    EINEM Satz aus JEDER geforderten Wortgruppe mindestens ein Glied vorkommt.
    Die Ko-Okkurrenz (Objekt + Mangel/Abwesenheit) erzeugt Präzision: "Kasse"
    allein feuert nicht, "Kasse" + "nicht nachvollziehbar" schon.

  * StreitDetektor: Subsumtions-Brennpunkte. Signalwörter wie "streitig",
    "fraglich", "hätte … vermeiden müssen", "nachlässig" markieren die Stellen,
    an denen der Sachverhalt selbst sagt, wo die Subsumtion kippt – und verstärken
    (ist_streitig) die im selben Satz liegenden Begriffsmarken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Satzgrenzen: NUR an Satzzeichen, NICHT an Zeilenumbrüchen – Korpus-/Sachverhalts-
# texte sind oft mitten im Satz hart umbrochen; ein \n darf eine Ko-Okkurrenz
# ("Belege\nfehlen") nicht zerreißen. Einzel-Newlines zählen als Whitespace.
_SATZ_TRENNER = re.compile(r"[.!?;:]+")

# Ein PUNKT ist nur dann Satzgrenze, wenn er nicht zu einer juristischen
# Abkürzung gehört („gem. § 146 Abs. 1 AO" ist EIN Satz – sonst zerfällt die
# Ko-Okkurrenz und kein Muster feuert) UND danach ein plausibler Satzanfang
# folgt. !?;: trennen weiterhin immer.
#
# Abkürzungsschutz: bekannte Kurzformen + JEDER Einzelbuchstabe vor dem Punkt
# (deckt i. S. d. / z. B. / u. a. / d. h. / S. ab – ein Satz endet praktisch
# nie auf ein alleinstehendes Zeichen).
_ABK_VOR_PUNKT = re.compile(
    r"(?:\b(?:gem|Abs|Nr|Buchst|Satz|Art|Tz|Rz|Rn|Anm|Az|vgl|Vgl|bzw|ggf|Ggf|"
    r"sog|inkl|zzgl|ca|evtl|etc|Co)"
    r"|(?<![0-9A-Za-zÄÖÜäöüß])[A-Za-zÄÖÜäöü])$"
)
# plausibler Satzanfang nach dem Punkt: Whitespace + Großbuchstabe (ggf. nach
# öffnendem Anführungszeichen/Klammer) – oder Zeilen-/Textende.
_SATZ_ANFANG_DANACH = re.compile(r"\s+[„\"‚'(]?[A-ZÄÖÜ]")


def _ist_satzgrenze(text: str, m: re.Match) -> bool:
    """Entscheidet, ob ein Satzzeichen-Fund wirklich eine Satzgrenze ist."""
    if any(c in "!?;:" for c in m.group()):
        return True  # harte Trenner – wie bisher immer Grenze
    # nur Punkte: Abkürzung davor ⇒ keine Grenze. Kleines Suchfenster reicht
    # (längste Kurzform 6 Zeichen); Lookbehind/\b sehen trotz pos das Zeichen davor.
    if _ABK_VOR_PUNKT.search(text, max(0, m.start() - 8), m.start()):
        return False
    rest = text[m.end():]
    # Zeilen-/Textende oder Whitespace+Großbuchstabe ⇒ Grenze; alles andere
    # (z. B. „§ 146", „Abs. 1", Kleinbuchstabe) bleibt im Satz (recall-first).
    return not rest or rest.startswith("\n") or not rest.strip() \
        or bool(_SATZ_ANFANG_DANACH.match(rest))


@dataclass(frozen=True)
class Satz:
    text: str
    start: int
    end: int


def saetze_mit_offset(text: str) -> list[Satz]:
    """Zerlegt den Text in Sätze und behält die Originaloffsets.

    Offset-treu: es wird ausschließlich an Originalpositionen geschnitten,
    nie Text verändert – ``text[s.start:s.end] == s.text`` gilt immer.
    """
    out: list[Satz] = []
    pos = 0
    for m in _SATZ_TRENNER.finditer(text):
        if not _ist_satzgrenze(text, m):
            continue
        seg = text[pos:m.start()]
        if seg.strip():
            out.append(Satz(seg, pos, m.start()))
        pos = m.end()
    if pos < len(text) and text[pos:].strip():
        out.append(Satz(text[pos:], pos, len(text)))
    # Doppelte Zeilenumbrüche (Absatzgrenzen) trennen zusätzlich, damit nicht ein
    # ganzer Absatz ohne Satzzeichen zu einem Riesen-„Satz" verschmilzt.
    fein: list[Satz] = []
    for s in out:
        teil_pos = s.start
        for m in re.finditer(r"\n[ \t]*\n+", s.text):
            seg = text[teil_pos:s.start + m.start()]
            if seg.strip():
                fein.append(Satz(seg, teil_pos, s.start + m.start()))
            teil_pos = s.start + m.end()
        rest = text[teil_pos:s.end]
        if rest.strip():
            fein.append(Satz(rest, teil_pos, s.end))
    return fein


def _glied_pattern(glied: str) -> re.Pattern[str] | None:
    """Kompiliert ein Gruppen-Glied zu einem wiederverwendbaren Suchmuster.

    Auf Tokens splitten und mit \\s+ verbinden: tolerant gegen jede Folge von
    Leerraum (auch Zeilenumbruch/Doppelspatien) zwischen den Wörtern – PDF-/DOCX-
    Extrakte haben oft unregelmäßige Abstände. \\b erzwingt Wortanfang.

    WICHTIG: case-insensitiv über re.IGNORECASE auf dem ORIGINAL-Satz suchen,
    NICHT über satz.lower(). .lower() ist nicht längen-erhaltend (z. B.
    "İ".lower() → 2 Codepoints) und würde die zurückgemappten Offsets verschieben
    – Offset-Treue ist aber die load-bearing Integrationsgarantie.
    """
    teile = glied.split()
    if not teile:
        return None
    muster = r"\b" + r"\s+".join(re.escape(t) for t in teile)
    return re.compile(muster, re.IGNORECASE)


def _glied_funde(satz: str, glied: str) -> list[tuple[int, int]]:
    """Findet ein Gruppen-Glied im Satz (an Wortanfang, Stamm-/Präfix-tolerant).

    Mehrwort-Glieder ("nicht nachvollziehbar") werden als Phrase gesucht;
    Einzelwort-Glieder als Wortanfang ("fehl" ⇒ fehlt/fehlen/fehlende; "kasse"
    ⇒ Kasse/Kassenführung, aber NICHT Unterstützungs|kasse – kein Wortanfang).
    """
    rx = _glied_pattern(glied)
    if rx is None:
        return []
    return [(m.start(), m.end()) for m in rx.finditer(satz)]


class MusterMatcher:
    """Findet Beschreibungs-/Negativbefund-Muster aus dem Katalog (Schicht 1b)."""

    def __init__(self, katalog):
        # (begriff_id, [gruppen]) – nur Begriffe mit ausloeser. Die Glied-Muster
        # werden HIER einmal vorkompiliert (statt pro match()-Aufruf neu gebaut
        # und dem re-Modul-Cache überlassen).
        self._regeln: list[tuple[str, list[list[re.Pattern[str]]]]] = []
        for b in katalog:
            for spec in getattr(b, "ausloeser", []) or []:
                gruppen = spec.get("gruppen") if isinstance(spec, dict) else spec
                if gruppen and all(gruppen):
                    kompiliert = [
                        [rx for glied in gruppe if (rx := _glied_pattern(str(glied))) is not None]
                        for gruppe in gruppen
                    ]
                    self._regeln.append((b.id, kompiliert))

    def match(self, text: str) -> list["MusterTreffer"]:
        treffer: list[MusterTreffer] = []
        for satz in saetze_mit_offset(text):
            for begriff_id, gruppen in self._regeln:
                spans: list[tuple[int, int]] = []
                ok = True
                for gruppe in gruppen:
                    funde = [(m.start(), m.end())
                             for rx in gruppe for m in rx.finditer(satz.text)]
                    if not funde:
                        ok = False
                        break
                    spans.extend(funde)
                if not ok:
                    continue
                lo = min(s for s, _ in spans)
                hi = max(e for _, e in spans)
                treffer.append(MusterTreffer(
                    begriff_id=begriff_id,
                    start=satz.start + lo,
                    end=satz.start + hi,
                    matched_text=satz.text[lo:hi].strip(),
                    satz_start=satz.start, satz_end=satz.end,
                ))
        return treffer


@dataclass(frozen=True)
class MusterTreffer:
    begriff_id: str
    start: int
    end: int
    matched_text: str
    satz_start: int
    satz_end: int


# --- Streit-Verstärker -------------------------------------------------------

# Subsumtions-Brennpunkte: der Sachverhalt sagt selbst, wo es kippt. Diese
# Signale tragen KEINE Norm – sie heben FF/IN-Stellen hervor und verstärken
# benachbarte Begriffsmarken.
STREIT_SIGNALE: list[tuple[str, str]] = [
    (r"\b(?:zwischen den Beteiligten\s+)?streitig\b", "streitig"),
    (r"\bstrittig\b", "strittig"),
    (r"\bumstritten\b", "umstritten"),
    (r"\bfraglich\b", "fraglich"),
    (r"\bnachlässig\b", "nachlässig"),
    (r"\bhätte\b[^.;:!?\n]{0,60}?\b(?:vermeiden|erkennen|wissen|prüfen)\s+müssen", "hätte … müssen"),
    (r"\bnicht zu vertreten\b", "Vertretenmüssen streitig"),
    (r"\bob ihn ein Verschulden trifft\b", "Verschulden offen"),
]
_STREIT = [(re.compile(p, re.IGNORECASE), label) for p, label in STREIT_SIGNALE]


@dataclass(frozen=True)
class StreitFund:
    start: int
    end: int
    text: str
    signal: str


class StreitDetektor:
    """Findet Streit-/Subsumtions-Signalwörter (deterministisch)."""

    def funde(self, text: str) -> list[StreitFund]:
        out: list[StreitFund] = []
        belegt: list[tuple[int, int]] = []
        for rx, label in _STREIT:
            for m in rx.finditer(text):
                s, e = m.start(), m.end()
                if any(not (e <= bs or s >= be) for bs, be in belegt):
                    continue
                out.append(StreitFund(s, e, text[s:e], label))
                belegt.append((s, e))
        out.sort(key=lambda f: f.start)
        return out
