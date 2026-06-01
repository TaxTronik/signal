"""Berater-Definitionen (Erweiterung §4).

Verteilt die Pflegelast (löst /R10/ teilweise): Der Katalog wächst durch Nutzung
statt durch Vordefinition. Harte Trennung zweier Ebenen – im Haftungskontext
nicht optional (/LF320/, /LF330/):

  * scope=personal : nur für den Autor, frei beschreibbar, kein Review.
  * scope=geteilt  : für andere sichtbar; wird erst nach Freigabe kanonisch
                     (Lebenszyklus entwurf → geprüft → freigegeben), trägt
                     Autor/Review-Status/Konfidenz sichtbar mit.

Das System validiert NICHT die rechtliche Richtigkeit (das wäre Subsumtion);
es führt Provenienz und Lebenszyklus prüfbar mit (PS-980-freundlich).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .idschema import NormID  # noqa: F401  (Dokumentationsbezug)
from .models import (Definitionsquelle, GUELTIGE_GOVERNANCE, GUELTIGE_SCHADEN,
                     Risikomarkierung, Status)
from .persistenz import atomar_schreiben, slugify
from .zitate import parse_normzitat

_WB = r"0-9A-Za-zÄÖÜäöüßÀ-ÿ"

REVIEW_REIHENFOLGE = ("entwurf", "geprüft", "freigegeben")


class BeraterFehler(ValueError):
    """Strukturfehler / unzulässiger Lebenszyklus-Übergang."""


def _slug(text: str) -> str:
    # Separator '_' (passt zu den ASCII-Katalog-IDs wie kst_finanzielle_…);
    # Algorithmik (Transliteration, Cap) kommt zentral aus persistenz.slugify.
    return slugify(text, default="eintrag", sep="_")


@dataclass
class BeraterDefinition:
    """Eine Berater-Definition (Schema gem. Erweiterung §4.3)."""

    id: str
    begriff: str
    definition_text: str
    norm_anker: tuple[str, ...] = ()
    definitionsquelle: tuple[Definitionsquelle, ...] = ()
    governance_typ: str = "IN"
    schadensintensitaet: str = "mittel"
    kaskadenreichweite: int = 1
    trigger_ereignisse: tuple[str, ...] = ()
    # Provenienz & Lebenszyklus
    scope: str = "personal"           # personal | geteilt
    autor: str = ""
    eingetragen_am: str = ""
    review_status: str = "entwurf"    # nur bei scope=geteilt relevant
    reviewer: str | None = None
    konfidenz: str = "mittel"
    # Abgleich (Stufe 2)
    geankert_an: tuple[str, ...] = ()
    zuletzt_geprueft_gegen_rspr_am: str | None = None
    abgleich_status: str = "aktuell"  # aktuell | pruefen | veraltet_verdacht
    # Annotation an einen BESTEHENDEN Katalogbegriff (statt eigenständiger Begriff)
    katalog_id: str | None = None

    @property
    def ist_kanonisch(self) -> bool:
        """Geteilte Einträge sind erst nach Freigabe kanonisch (/LF330/)."""
        return self.scope == "personal" or self.review_status == "freigegeben"

    def as_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "begriff": self.begriff,
            "definition_text": self.definition_text,
            "norm_anker": list(self.norm_anker),
            "definitionsquelle": [q.as_dict() for q in self.definitionsquelle],
            "governance_typ": self.governance_typ,
            "schadensintensitaet": self.schadensintensitaet,
            "kaskadenreichweite": self.kaskadenreichweite,
            "trigger_ereignisse": list(self.trigger_ereignisse),
            "scope": self.scope,
            "autor": self.autor,
            "eingetragen_am": self.eingetragen_am,
            "review_status": self.review_status,
            "reviewer": self.reviewer,
            "konfidenz": self.konfidenz,
            "geankert_an": list(self.geankert_an),
            "zuletzt_geprueft_gegen_rspr_am": self.zuletzt_geprueft_gegen_rspr_am,
            "abgleich_status": self.abgleich_status,
            "katalog_id": self.katalog_id,
        }
        return d


def _anker_zu_knoten(norm_anker: tuple[str, ...]) -> tuple[str, ...]:
    """Hebt Norm-Anker via Zitat-Parser auf Knoten-IDs (Schema-Feld geankert_an)."""
    ids: list[str] = []
    for zitat in norm_anker:
        for a in parse_normzitat(zitat):
            if a.norm_id:
                ids.append(a.norm_id.to_id())
    return tuple(dict.fromkeys(ids))


def _validiere(d: BeraterDefinition) -> None:
    if d.governance_typ not in GUELTIGE_GOVERNANCE:
        raise BeraterFehler(f"[{d.id}] governance_typ muss FP|FF|IN sein")
    if d.schadensintensitaet not in GUELTIGE_SCHADEN:
        raise BeraterFehler(f"[{d.id}] schadensintensitaet ungültig")
    if d.scope not in ("personal", "geteilt"):
        raise BeraterFehler(f"[{d.id}] scope muss personal|geteilt sein")
    if d.scope == "geteilt" and d.review_status not in REVIEW_REIHENFOLGE:
        raise BeraterFehler(f"[{d.id}] review_status muss {REVIEW_REIHENFOLGE} sein")


class BeraterStore:
    """Lädt/persistiert Berater-Definitionen (je Scope eine YAML-Datei)."""

    def __init__(self, verzeichnis: str | Path):
        self.dir = Path(verzeichnis)
        self._eintraege: list[BeraterDefinition] = []
        self._laden()

    # --- Persistenz -----------------------------------------------------

    def _datei(self, scope: str) -> Path:
        return self.dir / f"{scope}.yaml"

    def _laden(self) -> None:
        self._eintraege = []
        for scope in ("personal", "geteilt"):
            pfad = self._datei(scope)
            if not pfad.exists():
                continue
            data = yaml.safe_load(pfad.read_text(encoding="utf-8")) or []
            for raw in data:
                self._eintraege.append(self._aus_dict(raw, scope))

    def _speichern(self, scope: str) -> None:
        rows = [d.as_dict() for d in self._eintraege if d.scope == scope]
        atomar_schreiben(
            self._datei(scope),
            yaml.safe_dump(rows, allow_unicode=True, sort_keys=False),
        )

    @staticmethod
    def _aus_dict(raw: dict[str, Any], scope: str) -> BeraterDefinition:
        quellen = tuple(
            Definitionsquelle(typ=str(q.get("typ", "")), fundstelle=str(q.get("fundstelle", "")))
            for q in (raw.get("definitionsquelle") or [])
        )
        d = BeraterDefinition(
            id=str(raw["id"]),
            begriff=str(raw["begriff"]),
            definition_text=str(raw.get("definition_text", "")),
            norm_anker=tuple(raw.get("norm_anker") or ()),
            definitionsquelle=quellen,
            governance_typ=str(raw.get("governance_typ", "IN")),
            schadensintensitaet=str(raw.get("schadensintensitaet", "mittel")),
            kaskadenreichweite=int(raw.get("kaskadenreichweite", 1)),
            trigger_ereignisse=tuple(raw.get("trigger_ereignisse") or ()),
            scope=str(raw.get("scope", scope)),
            autor=str(raw.get("autor", "")),
            eingetragen_am=str(raw.get("eingetragen_am", "")),
            review_status=str(raw.get("review_status", "entwurf")),
            reviewer=raw.get("reviewer"),
            konfidenz=str(raw.get("konfidenz", "mittel")),
            geankert_an=tuple(raw.get("geankert_an") or ()),
            zuletzt_geprueft_gegen_rspr_am=raw.get("zuletzt_geprueft_gegen_rspr_am"),
            abgleich_status=str(raw.get("abgleich_status", "aktuell")),
            katalog_id=raw.get("katalog_id"),
        )
        _validiere(d)
        return d

    # --- Operationen ----------------------------------------------------

    def alle(self) -> list[BeraterDefinition]:
        return list(self._eintraege)

    def definiere(
        self,
        begriff: str,
        definition_text: str,
        *,
        autor: str,
        datum: str,
        scope: str = "personal",
        governance_typ: str = "IN",
        schadensintensitaet: str = "mittel",
        norm_anker: tuple[str, ...] = (),
        konfidenz: str = "mittel",
        katalog_id: str | None = None,
    ) -> BeraterDefinition:
        """Legt eine Definition/Annotation an (define-on-the-fly, /LF260/, §6 Schritt 3).

        Mit ``katalog_id`` wird ein BESTEHENDER Katalogbegriff annotiert statt ein
        neuer Begriff definiert.
        """
        slug = _slug(f"{katalog_id}_{begriff}" if katalog_id else begriff)
        d = BeraterDefinition(
            id=slug,
            begriff=begriff,
            definition_text=definition_text,
            norm_anker=tuple(norm_anker),
            governance_typ=governance_typ,
            schadensintensitaet=schadensintensitaet,
            scope=scope,
            autor=autor,
            eingetragen_am=datum,
            review_status="entwurf",   # Lebenszyklus startet immer im Entwurf
            konfidenz=konfidenz,
            geankert_an=_anker_zu_knoten(tuple(norm_anker)),
            katalog_id=katalog_id,
        )
        _validiere(d)
        # vorhandenen gleichen Eintrag (id+scope) ersetzen
        self._eintraege = [e for e in self._eintraege if not (e.id == d.id and e.scope == d.scope)]
        self._eintraege.append(d)
        self._speichern(scope)
        return d

    def setze_review_status(self, eintrag_id: str, neuer_status: str, *, reviewer: str) -> None:
        """Lebenszyklus-Übergang für geteilte Einträge (/LF320/).

        Erlaubt nur Vorwärts-Schritte entwurf → geprüft → freigegeben.
        """
        d = next((e for e in self._eintraege if e.id == eintrag_id and e.scope == "geteilt"), None)
        if d is None:
            raise BeraterFehler(f"kein geteilter Eintrag {eintrag_id!r}")
        if neuer_status not in REVIEW_REIHENFOLGE:
            raise BeraterFehler(f"unbekannter review_status {neuer_status!r}")
        alt = REVIEW_REIHENFOLGE.index(d.review_status)
        neu = REVIEW_REIHENFOLGE.index(neuer_status)
        if neu <= alt:
            raise BeraterFehler(
                f"unzulässiger Übergang {d.review_status} → {neuer_status} (nur vorwärts)"
            )
        d.review_status = neuer_status
        d.reviewer = reviewer
        self._speichern("geteilt")

    def _sichtbare(self, nutzer: str | None) -> list[BeraterDefinition]:
        out: list[BeraterDefinition] = []
        for d in self._eintraege:
            if d.scope == "personal" and (nutzer is None or d.autor == nutzer):
                out.append(d)
            elif d.scope == "geteilt" and d.ist_kanonisch:
                out.append(d)
        return out

    def markierungen(
        self,
        text: str,
        *,
        belegte_spans: list[tuple[int, int]] | None = None,
        nutzer: str | None = None,
    ) -> list[Risikomarkierung]:
        """Erkennt sichtbare Berater-Begriffe im Text (Katalog wächst durch Nutzung).

        Stellen, die bereits durch Schicht 1 / andere Markierungen belegt sind,
        werden ausgelassen.
        """
        belegt = belegte_spans or []
        out: list[Risikomarkierung] = []
        for d in self._sichtbare(nutzer):
            if d.katalog_id:           # Annotationen sind keine eigenständigen Begriffe
                continue
            pat = re.compile(
                rf"(?<![{_WB}]){re.escape(d.begriff)}(?![{_WB}])", re.IGNORECASE
            )
            for m in pat.finditer(text):
                s, e = m.start(), m.end()
                if any(not (e <= bs or s >= be) for bs, be in belegt):
                    continue
                out.append(
                    Risikomarkierung(
                        start=s,
                        end=e,
                        matched_text=text[s:e],
                        status=Status.BERATER_DEFINIERT,
                        titel=d.begriff,
                        governance_typ=d.governance_typ,
                        schadensintensitaet=d.schadensintensitaet,
                        quellen=list(d.definitionsquelle),
                        norm_anker=list(d.norm_anker),
                        quelle="berater",
                        provenienz={
                            "autor": d.autor,
                            "scope": d.scope,
                            "review_status": d.review_status,
                            "konfidenz": d.konfidenz,
                            "definition_text": d.definition_text,
                        },
                    )
                )
        out.sort(key=lambda r: (r.start, r.end))
        return out

    def annotationen(self, katalog_id: str, *, nutzer: str | None = None) -> list[BeraterDefinition]:
        """Sichtbare Berater-Annotationen zu einem bestehenden Katalogbegriff."""
        return [d for d in self._sichtbare(nutzer) if d.katalog_id == katalog_id]

    def finde(self, span_text: str, *, nutzer: str | None = None) -> BeraterDefinition | None:
        """Findet eine zur Stelle passende, sichtbare Definition.

        Sichtbar = eigene personal-Einträge ODER kanonische (freigegebene) geteilte.
        """
        norm = " ".join(span_text.lower().split())
        for d in self._eintraege:
            if " ".join(d.begriff.lower().split()) != norm:
                continue
            if d.scope == "personal":
                if nutzer is None or d.autor == nutzer:
                    return d
            elif d.ist_kanonisch:
                return d
        return None
