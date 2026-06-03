"""Datenmodell des Risk-Layer.

Reine Daten-Container (dataclasses), bewusst ohne Logik, damit Schicht 1
reproduzierbar und prüfungsfest bleibt (/NF20/, /NF30/).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class GovernanceTyp(str, Enum):
    """FP = formal-periodisch | FF = fortlaufend-faktisch | IN = inhärent."""

    FP = "FP"
    FF = "FF"
    IN = "IN"


# Kanonische Validierungsmengen – EINMAL hier, von Berater-Store UND Fall-Store
# geteilt (vorher dupliziert/teilweise gar nicht geprüft).
GUELTIGE_GOVERNANCE = frozenset(t.value for t in GovernanceTyp)   # {"FP","FF","IN"}
GUELTIGE_SCHADEN = frozenset({"niedrig", "mittel", "hoch"})


class Status(str, Enum):
    """Status einer erkannten Textstelle (Lastenheft /LF180/ + Erweiterung §2).

    Zustandsmodell der Erweiterung „Unknown als Risiko":
    Abdeckung ist nicht mehr Eingangskriterium, sondern Ausgabewert – das System
    muss nicht alles kennen, sondern zuverlässig erkennen, was es nicht auflöst.
    """

    TREFFER = "treffer"                    # Katalogbegriff mit aufgelöster Quelle (Schicht 1)
    KANDIDAT = "kandidat"                  # LLM-Vorschlag außerhalb des Katalogs (Schicht 2)
    LUECKE = "luecke"                      # im Katalog, aber Definitionsquelle fehlt (= kandidat_lücke)
    UNKNOWN_RISIKO = "unknown_risiko"      # juristisch aufgeladen, nicht auflösbar → eskaliert (/LF250/)
    BERATER_DEFINIERT = "berater_definiert"  # von einem Berater definiert (/LF260/)
    VERWORFEN = "verworfen"                # Gate nicht bestanden (Alltagswort) – nicht angezeigt


# ---------------------------------------------------------------------------
# Katalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Definitionsquelle:
    typ: str           # "Gesetz" | "Verwaltung" | "Rspr" | "Literatur" ...
    fundstelle: str

    def as_dict(self) -> dict[str, str]:
        return {"typ": self.typ, "fundstelle": self.fundstelle}


@dataclass(frozen=True)
class Normkette:
    """Eine Fundstelle als zusammenhängende NORMKETTE (z. B. § 14 KStG iVm § 2 GewStG).

    Im Steuerrecht ist eine Fundstelle oft KEINE einzelne Norm, sondern eine Kette
    verzahnter Normen – der steuerartenübergreifende Durchschlag (Kaskade). Diese
    Struktur hält die Glieder GEORDNET zusammen (``glieder[0]`` ist die primäre
    Norm, der Rest die Folgeglieder), damit Karte und Vorschlags-Modus die ganze
    Kaskade zeigen – nicht nur die Spitze des Eisbergs.
    """

    glieder: tuple[str, ...]          # Zitate in Reihenfolge; [0] = primär
    verknuepfung: str = "iVm"         # iVm | iSd | Durchschlag auf | ...
    hinweis: str = ""                 # menschliche Erläuterung der Kaskade

    @property
    def primaer(self) -> str:
        return self.glieder[0] if self.glieder else ""

    def anzeige(self) -> str:
        """Lesbare Kette, z. B. '§ 14 KStG iVm § 2 Abs. 2 GewStG'."""
        return f" {self.verknuepfung} ".join(self.glieder)

    def as_dict(self) -> dict[str, Any]:
        return {"glieder": list(self.glieder), "verknuepfung": self.verknuepfung,
                "hinweis": self.hinweis, "anzeige": self.anzeige()}


@dataclass(frozen=True)
class Katalogeintrag:
    """Ein kuratierter Schlüsselbegriff (Single Source of Truth, Schicht 1)."""

    id: str
    begriff_kanonisch: str
    aliase: tuple[str, ...] = ()
    # Flache Liste ALLER berührten Normen (Graph-Anreicherung, Rückwärtskompat.).
    norm_anker: tuple[str, ...] = ()
    # Fundstellen als KETTEN (geordnet, mit Verknüpfung) – die fachlich korrekte
    # Kaskaden-Abbildung. ``norm_anker`` bleibt die flache Union; ``normketten``
    # trägt die Verzahnung (iVm/iSd/Durchschlag), an der Paraphrasen & Vorschläge hängen.
    normketten: tuple[Normkette, ...] = ()
    definitionsquelle: tuple[Definitionsquelle, ...] = ()
    governance_typ: GovernanceTyp = GovernanceTyp.IN
    schadensintensitaet: str = "mittel"
    kaskadenreichweite: int = 1
    trigger_ereignisse: tuple[str, ...] = ()
    gefaehrdungsnotiz: str = ""
    gueltig_ab: Optional[str] = None
    gueltig_bis: Optional[str] = None
    # Schicht 1b – Beschreibungs-/Negativbefund-Muster (Ko-Okkurrenz je Satz).
    # Jede Spec: {"gruppen": [[glied, ...], ...]}; trifft, wenn aus JEDER Gruppe
    # mindestens ein Glied im selben Satz vorkommt. Rein deterministisch.
    ausloeser: tuple[Any, ...] = ()
    # Schicht 1.7 – kuratierte LAIEN-Umschreibungen für den Embedding-Index. Werden
    # MIT den Norm-Ankern eingebettet, damit alltagssprachliche Sachverhalte den
    # passenden § im selben Register treffen (hebt echte Treffer, ohne den
    # Rauschboden zu senken). Kuratiert, nicht Modellwissen – trainingsdaten-unabh.
    paraphrasen: tuple[str, ...] = ()

    @property
    def hat_definitionsquelle(self) -> bool:
        return len(self.definitionsquelle) > 0

    @property
    def oberflaechen(self) -> tuple[str, ...]:
        """Kanonische Form + alle Aliase – die zu suchenden Oberflächenvarianten."""
        return (self.begriff_kanonisch, *self.aliase)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """Adressierbares Textsegment mit stabilen Offsets auf dem ORIGINALTEXT."""

    index: int
    start: int   # Zeichen-Offset (inkl.) im Originaltext
    end: int     # Zeichen-Offset (exkl.) im Originaltext
    text: str    # normalisierter Segmenttext


@dataclass
class IngestErgebnis:
    original_text: str
    normalized_text: str
    # offset_map[i] = Originalindex des Zeichens normalized_text[i]
    offset_map: list[int]
    segments: list[Segment]
    text_hash: str           # sha256 des Originaltextes (/LF30/)


# ---------------------------------------------------------------------------
# Treffer & Markierung
# ---------------------------------------------------------------------------

@dataclass
class Treffer:
    """Ein Schicht-1-Treffer: deterministisch belegt durch eine Katalog-ID (/LF70/)."""

    begriff_id: str
    start: int                  # Offset im Originaltext (inkl.)
    end: int                    # Offset im Originaltext (exkl.)
    matched_text: str           # wörtlich aus dem Originaltext
    surface: str                # die getroffene Oberflächenvariante (Regelbeleg)
    segment_index: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Begriffskarte:
    """Aufgelöste Markierung für Report & Audit (/LF190/)."""

    treffer: Treffer
    eintrag: Katalogeintrag
    status: Status
    # aufgelöste Definitionsquellen (leer ⇒ Lücke)
    quellen: list[Definitionsquelle] = field(default_factory=list)
    rechercheauftrag: Optional[str] = None  # gesetzt bei Status.LUECKE
    # Stufe 2: Gesetzestext aus dem Normgraph (Korpus, nie Modell), /LF170/
    gesetzestext: list[dict[str, Any]] = field(default_factory=list)
    # Berater-Annotationen an dieser Karte (eigene Infos, Quellen, Notizen)
    berater_notizen: list[dict[str, Any]] = field(default_factory=list)
    # Katalog-weite Norm-Kuratierung dieses Begriffs (verworfene/ergänzte Normen);
    # vom Resolver aus dem BeraterStore gesetzt: {"verworfen": set[id], "ergaenzt": [...]}
    norm_kuratierung: dict[str, Any] = field(default_factory=dict)
    # Abweichungs-Check: das LLM ordnet dieselbe Stelle einer ANDEREN Norm zu (Hinweis)
    abweichung: list[dict[str, Any]] = field(default_factory=list)
    # Herkunft des Treffers: "wörtlich" (Surface-Form) oder "muster" (Schicht 1b)
    via: str = "wörtlich"
    # Streit-Verstärker: liegt die Karte in einem Subsumtions-Brennpunkt-Satz?
    ist_streitig: bool = False
    streit_signal: str = ""

    @property
    def wertungsoffen(self) -> bool:
        """Inhärent wertungsoffener Begriff (governance_typ IN): braucht Subsumtion,
        UNABHÄNGIG davon, ob im Text ein Streit-Wort fällt. Eigener Flag, NICHT Streit
        (= Disput im Text) — die fachliche Sollbruchstelle, die der Berater prüfen muss."""
        return self.eintrag.governance_typ is GovernanceTyp.IN

    def herkunft(self) -> dict[str, Any]:
        """Epistemische Provenienz – Haftungssoftware muss die Schicht ausweisen."""
        if self.via == "muster":
            return {
                "schicht": "1b", "methode": "deterministisch", "interpretation": False,
                "kurz": "① Katalog · Beschreibungsmuster",
                "art": "deterministischer Katalogtreffer über Beschreibungsmuster",
                "hinweis": "Der Begriff ist nicht wörtlich genannt, aber durch ein "
                           "kuratiertes Ko-Okkurrenz-Muster (kein Modell) belegt. "
                           "Inhaltlich vom Berufsträger zu prüfen.",
            }
        if self.status is Status.LUECKE:
            return {
                "schicht": "1", "methode": "deterministisch", "interpretation": False,
                "kurz": "① wörtlich · ohne Definition",
                "art": "wörtlich erkannt, ohne hinterlegte Definition",
                "hinweis": "Begriff steht wörtlich im Text, wird aber (noch) nicht erklärt "
                           "– Rechercheauftrag. Keine Wertung durch das System.",
            }
        return {
            "schicht": "1", "methode": "deterministisch", "interpretation": False,
            "kurz": "① wörtlich · Katalog",
            "art": "wörtlicher Katalogtreffer",
            "hinweis": "Begriff steht wörtlich im Text; Definitionsquelle hinterlegt. "
                       "Die Subsumtion schuldet der Berufsträger.",
        }

    def as_dict(self) -> dict[str, Any]:
        e = self.eintrag
        return {
            "begriff_id": e.id,
            "begriff_kanonisch": e.begriff_kanonisch,
            "status": self.status.value,
            "herkunft": self.herkunft(),
            "matched_text": self.treffer.matched_text,
            "start": self.treffer.start,
            "end": self.treffer.end,
            "surface_beleg": self.treffer.surface,
            "norm_anker": list(e.norm_anker),
            "definitionsquelle": [q.as_dict() for q in self.quellen],
            "governance_typ": e.governance_typ.value,
            "schadensintensitaet": e.schadensintensitaet,
            "kaskadenreichweite": e.kaskadenreichweite,
            "gefaehrdungsnotiz": e.gefaehrdungsnotiz.strip(),
            "rechercheauftrag": self.rechercheauftrag,
            "gesetzestext": self.gesetzestext,
            "berater_notizen": self.berater_notizen,
            "abweichung": self.abweichung,
            "via": self.via,
            "ist_streitig": self.ist_streitig,
            "streit_signal": self.streit_signal,
        }


# ---------------------------------------------------------------------------
# Risikomarkierung (Erweiterung §2/§3): Stellen OHNE Katalogeintrag
# ---------------------------------------------------------------------------

@dataclass
class Risikomarkierung:
    """Markierung außerhalb des Katalogs: unknown_risiko oder berater_definiert.

    Trägt – anders als die Begriffskarte – keinen Katalogeintrag, sondern den
    Gate-Score bzw. die Provenienz der Berater-Definition.
    """

    start: int
    end: int
    matched_text: str
    status: Status
    titel: str
    score: float = 0.0
    signale: dict[str, Any] = field(default_factory=dict)     # Unknown-Gate-Signale
    governance_typ: Optional[str] = None
    schadensintensitaet: Optional[str] = None
    quellen: list[Definitionsquelle] = field(default_factory=list)  # bei berater_definiert
    provenienz: dict[str, Any] = field(default_factory=dict)   # Autor/Review/Scope
    norm_anker: list[str] = field(default_factory=list)
    quelle: str = "heuristik"   # "llm" | "heuristik" | "berater" | "trigger" | "streit"
    # LLM-Einordnungs-VORSCHLAG (vom Graph nur auf Existenz geprüft, kein festes Mapping)
    norm_vorschlag: list[dict[str, Any]] = field(default_factory=list)
    # Streit-Verstärker (Subsumtions-Brennpunkt)
    ist_streitig: bool = False
    streit_signal: str = ""

    def herkunft(self) -> dict[str, Any]:
        """Epistemische Provenienz der Markierung (Layer-Kennzeichnung)."""
        if self.quelle == "streit":
            return {
                "schicht": "1b", "methode": "deterministisch", "interpretation": False,
                "kurz": "⚑ Subsumtions-Brennpunkt", "art": "Streit-/Subsumtionssignal",
                "hinweis": "Der Sachverhalt signalisiert hier selbst eine streitige "
                           "Subsumtion (Signalwörter wie streitig oder haette vermeiden "
                           "muessen). Deterministisch erkannt; kein Modellurteil.",
            }
        if self.quelle == "trigger":
            return {
                "schicht": "1.5", "methode": "deterministisch", "interpretation": False,
                "kurz": "① Trigger-Index (ohne Modell)",
                "art": "korpus-trennscharfes Trigger-Wort mit Norm-Kandidaten",
                "hinweis": "Deterministisch aus dem Gesetzeskorpus abgeleitet "
                           "(trainingsdaten-unabhängig). Norm-Kandidaten sind Vorschläge, "
                           "kein festes Mapping.",
            }
        if self.status is Status.BERATER_DEFINIERT:
            return {
                "schicht": "berater", "methode": "human", "interpretation": False,
                "kurz": "Berater · Definition", "art": "Berater-Definition",
                "hinweis": "Vom Berater hinterlegte Arbeitsdefinition (kein Modellurteil).",
            }
        if self.quelle == "llm":
            return {
                "schicht": "2", "methode": "llm", "interpretation": True,
                "kurz": "② LLM-Interpretation", "art": "LLM-Interpretation (ggf. umschrieben)",
                "konfidenz": self.signale.get("konfidenz"),
                "hinweis": "Das Modell INTERPRETIERT diese – evtl. nur umschriebene – Stelle "
                           "als definitionsbedürftig. Wörtlich im Text belegt, inhaltlich "
                           "vom Berufsträger zu prüfen. Keine Definition aus dem Modell.",
            }
        return {
            "schicht": "2", "methode": "heuristik", "interpretation": True,
            "kurz": "② heuristischer Hinweis", "art": "heuristischer Hinweis (Norm-Nähe/Form)",
            "hinweis": "Regelbasierter Verdacht (Norm-Nähe, definitorische Form). "
                       "Zu prüfen; keine Wertung durch das System.",
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "herkunft": self.herkunft(),
            "quelle": self.quelle,
            "matched_text": self.matched_text,
            "start": self.start,
            "end": self.end,
            "titel": self.titel,
            "score": round(self.score, 3),
            "signale": self.signale,
            "governance_typ": self.governance_typ,
            "schadensintensitaet": self.schadensintensitaet,
            "norm_anker": self.norm_anker,
            "norm_vorschlag": self.norm_vorschlag,
            "definitionsquelle": [q.as_dict() for q in self.quellen],
            "provenienz": self.provenienz,
            "ist_streitig": self.ist_streitig,
            "streit_signal": self.streit_signal,
        }


# ---------------------------------------------------------------------------
# Lauf-Report
# ---------------------------------------------------------------------------

@dataclass
class RunReport:
    text_hash: str
    katalog_version: str
    erzeugt_am: str                         # ISO-8601 Zeitstempel
    karten: list[Begriffskarte] = field(default_factory=list)
    # Stufe 1: verifizierte Schicht-2-Kandidaten (roh, vor Gate)
    kandidaten: list[dict[str, Any]] = field(default_factory=list)
    # Erweiterung §2/§3: eskalierte Unbekannte + Berater-Definitionen
    risiken: list[Risikomarkierung] = field(default_factory=list)
    # Stufe 1: Modell-/Quantisierungs-/Seed-Angaben des LLM-Laufs (Audit /LF140/)
    schicht2_info: Optional[dict[str, Any]] = None
    # Schicht 1.7: semantische Norm-VORSCHLÄGE je Satz (kein hartes Markieren –
    # kuratierbarer Hinweis + LLM-Erdung; sim-Wert wird mitgeführt)
    semantik_vorschlaege: list[dict[str, Any]] = field(default_factory=list)

    @property
    def treffer(self) -> list[Begriffskarte]:
        return [k for k in self.karten if k.status is Status.TREFFER]

    @property
    def luecken(self) -> list[Begriffskarte]:
        return [k for k in self.karten if k.status is Status.LUECKE]

    @property
    def unknowns(self) -> list[Risikomarkierung]:
        return [r for r in self.risiken if r.status is Status.UNKNOWN_RISIKO]

    @property
    def berater(self) -> list[Risikomarkierung]:
        return [r for r in self.risiken if r.status is Status.BERATER_DEFINIERT]

    def zusammenfassung(self) -> dict[str, Any]:
        """Zählung nach Risikoklasse + Lücken/Kandidaten/Unknowns (/LF200/)."""
        nach_governance: dict[str, int] = {}
        nach_intensitaet: dict[str, int] = {}
        for k in self.karten:
            g = k.eintrag.governance_typ.value
            i = k.eintrag.schadensintensitaet
            nach_governance[g] = nach_governance.get(g, 0) + 1
            nach_intensitaet[i] = nach_intensitaet.get(i, 0) + 1
        return {
            "markierungen_gesamt": len(self.karten) + len(self.risiken),
            "treffer": len(self.treffer),
            "luecken": len(self.luecken),
            "kandidaten": len(self.kandidaten),
            "unknown_risiko": len(self.unknowns),
            "berater_definiert": len(self.berater),
            "nach_governance_typ": nach_governance,
            "nach_schadensintensitaet": nach_intensitaet,
        }
