"""Kanonisches Knoten-ID-Schema (Parser-Spez. Kap. 1).

Das eine Schema, das Katalog, Gesetze, Verwaltung und Urteile zusammenhält –
der universelle Join über alle Korpora. Reine Daten-/Formatlogik, kein Korpus
nötig.

    norm:<LAW>:<para>[:abs<n>][:nr<n>][:s<n>][:lit<x>]      Norm
    verw:<QUELLE>:<abschnitt>                                Verwaltung
    urteil:<GERICHT>:<aktenzeichen-normalisiert>             Urteil
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Quellentyp(str, Enum):
    """Pflicht-Property jedes Knotens (Parser-Spez. 1.4) – trägt die Rangfolge.

    Norm/Unionsrecht = Tatbestand, BFH/EuGH = Anerkennungskorridor,
    Verwaltung = Streit-/Prozessrisiko. Jede ausgespielte Definition MUSS mit
    ihrem Quellentyp gelabelt werden (Verwaltungsauffassung ≠ Gerichtsentscheidung).
    """

    NORM = "norm"
    UNIONSRECHT = "unionsrecht"
    RSPR_BFH = "rspr_bfh"
    RSPR_EUGH = "rspr_eugh"
    RSPR_SONSTIGE = "rspr_sonstige"
    VERWALTUNG = "verwaltung"


# Reihenfolge der Tiefenlabels, wie in der ID geschrieben (Quell-Reihenfolge bleibt
# erhalten; dies ist nur die Menge der erlaubten Labels).
TIEFE_LABELS = ("abs", "nr", "s", "lit", "hs", "alt")


def para_level(norm_id: str) -> str:
    """Reduziert eine volle Knoten-ID auf die PARAGRAPH-Ebene ``norm:<LAW>:<para>``.

    Schneidet die Tiefenlabels (abs/nr/s/lit …) ab – das ist der Join-Schlüssel,
    auf dem Katalog, Trigger-Index und Embedding ihre §§ zusammenführen. Liegt
    EINMAL hier (war zuvor dreifach: _para_level/_para_id/inline-split).
    """
    return ":".join(str(norm_id).split(":")[:3])


@dataclass(frozen=True)
class NormID:
    """Kanonische Norm-ID. Bewahrt die Quell-Reihenfolge des Zitats (Spez. 1.1)."""

    law: str
    para: str | None = None          # z. B. "2", "2a", "14"
    artikel: str | None = None       # z. B. "11" bei Art.-Zitaten
    tiefe: tuple[tuple[str, str], ...] = ()   # geordnet: (("abs","2"),("nr","2"))
    offener_bereich: bool = False    # ff. → offener Bereich (Spez. 2.2)

    def __post_init__(self) -> None:
        if not str(self.law):
            raise ValueError("NormID braucht ein nicht-leeres law (unbekannt: '?')")
        if not (self.para or self.artikel):
            raise ValueError("NormID braucht para oder artikel")
        if self.para and self.artikel:
            raise ValueError(
                f"NormID: para ({self.para!r}) und artikel ({self.artikel!r}) schließen "
                "sich aus – to_id() würde para sonst still ignorieren")
        for label, _wert in self.tiefe:
            if label not in TIEFE_LABELS:
                raise ValueError(
                    f"NormID: unbekanntes Tiefen-Label {label!r} (erlaubt: {TIEFE_LABELS})")

    def to_id(self) -> str:
        kopf = f"art{self.artikel}" if self.artikel else self.para
        teile = [f"{label}{wert}" for label, wert in self.tiefe]
        return ":".join(["norm", self.law, kopf, *teile])

    def __str__(self) -> str:
        return self.to_id()


@dataclass(frozen=True)
class VerwID:
    """Kanonische Verwaltungs-ID (UStAE, AEAO, Richtlinien) (Spez. 1.2)."""

    quelle: str       # UStAE | AEAO | GewStR | ...
    abschnitt: str    # z. B. "2.8", "zu§153", "R2.3"

    def to_id(self) -> str:
        return f"verw:{self.quelle}:{self.abschnitt}"

    def __str__(self) -> str:
        return self.to_id()


@dataclass(frozen=True)
class UrteilID:
    """Kanonische Urteils-ID; Fundstellen sind Aliase, nie ID (Spez. 1.3)."""

    gericht: str            # BFH | EuGH | BGH | BVerfG | BVerwG | FG<Ort>
    aktenzeichen: str       # normalisiert, z. B. "I-R-37/19", "C-141/20"
    aliase: tuple[str, ...] = field(default_factory=tuple)  # Fundstellen

    def to_id(self) -> str:
        return f"urteil:{self.gericht}:{self.aktenzeichen}"

    def __str__(self) -> str:
        return self.to_id()
