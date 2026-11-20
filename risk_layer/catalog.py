"""Katalog-Loader /LF220/ /LF230/ /NF50/.

Lädt die versionierte YAML-Datendatei, validiert sie und stellt sie als
unveränderliche Katalogeinträge bereit. Der Katalog ist die Single Source of
Truth der Schicht 1 und strikt vom Code getrennt: eine Katalogänderung erfordert
keinen Code-Deploy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (Definitionsquelle, GovernanceTyp, GUELTIGE_SCHADEN,
                     Katalogeintrag, Normkette)

# Default-Katalog relativ zum Projekt-Root (Repo-Layout: <root>/catalog/begriffe.yaml)
DEFAULT_KATALOG = Path(__file__).resolve().parent.parent / "catalog" / "begriffe.yaml"


def _standard_katalog() -> Path:
    """Findet Checkout-Daten auch bei einem nicht-editierbar installierten Wheel."""
    if DEFAULT_KATALOG.is_file():
        return DEFAULT_KATALOG
    checkout = Path.cwd() / "catalog" / "begriffe.yaml"
    return checkout if checkout.is_file() else DEFAULT_KATALOG


class KatalogFehler(ValueError):
    """Strukturfehler im Begriffskatalog (Pflege-/Validierungsfehler)."""


class Katalog:
    """Geladener, validierter Begriffskatalog mit Versionsangabe."""

    def __init__(self, version: str, eintraege: list[Katalogeintrag], *, quelle: str = ""):
        self.version = version
        self.quelle = quelle
        self.eintraege = eintraege
        # Duplikatprüfung auch bei DIREKTER Konstruktion (nicht nur in lade_katalog):
        # sonst verlöre _by_id still Einträge.
        self._by_id: dict[str, Katalogeintrag] = {}
        for e in eintraege:
            if e.id in self._by_id:
                raise KatalogFehler(f"Doppelte Begriffs-ID im Katalog: {e.id!r}")
            self._by_id[e.id] = e

    def __len__(self) -> int:
        return len(self.eintraege)

    def __iter__(self):
        return iter(self.eintraege)

    def get(self, begriff_id: str) -> Katalogeintrag:
        return self._by_id[begriff_id]

    def __contains__(self, begriff_id: str) -> bool:
        return begriff_id in self._by_id


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _parse_ausloeser(raw: Any, *, begriff_id: str) -> tuple[dict[str, Any], ...]:
    """Normalisiert Gruppen und optionale wortbezogene Ausschlüsse (Schicht 1b)."""
    if not raw:
        return ()
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise KatalogFehler(
            f"[{begriff_id}] ausloeser muss eine Liste von Specs sein: {raw!r}")
    specs: list[dict[str, Any]] = []
    for spec in raw:
        gruppen = spec.get("gruppen") if isinstance(spec, dict) else spec
        if isinstance(gruppen, str) or not isinstance(gruppen, (list, tuple)) or not gruppen:
            raise KatalogFehler(
                f"[{begriff_id}] ausloeser-Spec braucht nicht-leere 'gruppen' "
                f"(Liste von Listen): {spec!r}")
        for gruppe in gruppen:
            # jede Gruppe MUSS eine Liste von Strings sein – ein blanker String
            # würde sonst zeichenweise zerlegt und das Muster matchte jeden Satz.
            if isinstance(gruppe, str) or not isinstance(gruppe, (list, tuple)) \
                    or not gruppe or not all(isinstance(wort, str) and wort.strip() for wort in gruppe):
                raise KatalogFehler(
                    f"[{begriff_id}] jede ausloeser-Gruppe muss eine nicht-leere "
                    f"LISTE von Strings sein (kein blanker String): {gruppe!r}")
        normalisiert: dict[str, Any] = {
            "gruppen": [list(gruppe) for gruppe in gruppen]}
        for feld in ("ausschlusswoerter", "verneinbare_nomen"):
            if not isinstance(spec, dict) or feld not in spec:
                continue
            woerter = spec[feld]
            if not isinstance(woerter, (list, tuple)) or any(
                not isinstance(wort, str) or len(wort.split()) != 1
                for wort in woerter
            ):
                raise KatalogFehler(
                    f"[{begriff_id}] {feld} muss eine Liste einzelner "
                    f"nicht-leerer Wörter sein: {woerter!r}")
            normalisiert[feld] = [wort.strip() for wort in woerter]
        specs.append(normalisiert)
    return tuple(specs)


def _parse_normketten(raw: Any, *, begriff_id: str) -> tuple[Normkette, ...]:
    """Liest Fundstellen-KETTEN (geordnete Glieder + Verknüpfung).

    YAML je Kette:  {glieder: ["§ 14 KStG", "§ 2 Abs. 2 GewStG"], verknuepfung: "iVm",
                     hinweis: "Durchschlag auf die GewSt"}
    Eine bloße Liste von Zitaten wird als Kette mit Default-Verknüpfung „iVm" gelesen.
    """
    if not raw:
        return ()
    ketten: list[Normkette] = []
    for k in raw:
        if isinstance(k, dict):
            glieder = tuple(str(g) for g in (k.get("glieder") or []) if str(g).strip())
            verkn = str(k.get("verknuepfung", "iVm"))
            hinweis = str(k.get("hinweis", ""))
        elif isinstance(k, (list, tuple)):
            glieder = tuple(str(g) for g in k if str(g).strip())
            verkn, hinweis = "iVm", ""
        else:
            raise KatalogFehler(
                f"[{begriff_id}] normkette muss dict oder Liste sein: {k!r}")
        if not glieder:
            raise KatalogFehler(f"[{begriff_id}] normkette braucht nicht-leere 'glieder'")
        ketten.append(Normkette(glieder=glieder, verknuepfung=verkn, hinweis=hinweis))
    return tuple(ketten)


def _parse_quellen(raw: Any, *, begriff_id: str) -> tuple[Definitionsquelle, ...]:
    if not raw:
        return ()
    quellen: list[Definitionsquelle] = []
    for q in raw:
        if not isinstance(q, dict) or "typ" not in q or "fundstelle" not in q:
            raise KatalogFehler(
                f"[{begriff_id}] definitionsquelle braucht 'typ' und 'fundstelle': {q!r}"
            )
        quellen.append(Definitionsquelle(typ=str(q["typ"]), fundstelle=str(q["fundstelle"])))
    return tuple(quellen)


def _parse_eintrag(raw: dict[str, Any]) -> Katalogeintrag:
    # Typ ZUERST prüfen: bei einem String wäre `"id" not in raw` nur ein
    # Substring-Check und raw["id"] ein TypeError statt einer klaren Meldung.
    if not isinstance(raw, dict):
        raise KatalogFehler(f"Katalog-Eintrag muss ein Mapping sein: {raw!r}")
    if "id" not in raw or "begriff_kanonisch" not in raw:
        raise KatalogFehler(f"Eintrag braucht 'id' und 'begriff_kanonisch': {raw!r}")
    begriff_id = str(raw["id"])

    gov_raw = str(raw.get("governance_typ", "IN")).upper()
    try:
        governance = GovernanceTyp(gov_raw)
    except ValueError as exc:
        raise KatalogFehler(
            f"[{begriff_id}] governance_typ muss FP|FF|IN sein, nicht {gov_raw!r}"
        ) from exc

    intensitaet = str(raw.get("schadensintensitaet", "mittel")).lower()
    if intensitaet not in GUELTIGE_SCHADEN:
        raise KatalogFehler(
            f"[{begriff_id}] schadensintensitaet muss niedrig|mittel|hoch sein, "
            f"nicht {intensitaet!r}"
        )

    try:
        kaskade = int(raw.get("kaskadenreichweite", 1))
    except (TypeError, ValueError):
        raise KatalogFehler(f"[{begriff_id}] kaskadenreichweite muss eine Zahl 1|2|3 sein") from None
    if kaskade not in (1, 2, 3):
        raise KatalogFehler(f"[{begriff_id}] kaskadenreichweite muss 1|2|3 sein")

    return Katalogeintrag(
        id=begriff_id,
        begriff_kanonisch=str(raw["begriff_kanonisch"]),
        aliase=_as_tuple(raw.get("aliase")),
        norm_anker=_as_tuple(raw.get("norm_anker")),
        definitionsquelle=_parse_quellen(raw.get("definitionsquelle"), begriff_id=begriff_id),
        governance_typ=governance,
        schadensintensitaet=intensitaet,
        kaskadenreichweite=kaskade,
        trigger_ereignisse=_as_tuple(raw.get("trigger_ereignisse")),
        gefaehrdungsnotiz=str(raw.get("gefaehrdungsnotiz", "")),
        # str()-Coercion wie bei den anderen Feldern: ein unquotiertes
        # YAML-Datum käme sonst als datetime.date statt Optional[str] an.
        gueltig_ab=str(raw["gueltig_ab"]) if raw.get("gueltig_ab") is not None else None,
        gueltig_bis=str(raw["gueltig_bis"]) if raw.get("gueltig_bis") is not None else None,
        ausloeser=_parse_ausloeser(raw.get("ausloeser"), begriff_id=begriff_id),
        paraphrasen=_as_tuple(raw.get("paraphrasen")),
        normketten=_parse_normketten(raw.get("normketten"), begriff_id=begriff_id),
    )


def lade_katalog(pfad: str | Path | None = None) -> Katalog:
    """Lädt und validiert den Begriffskatalog.

    Raises:
        KatalogFehler: bei strukturellen Fehlern oder doppelten IDs.
    """
    pfad = Path(pfad) if pfad is not None else _standard_katalog()
    if not pfad.exists():
        raise KatalogFehler(f"Katalogdatei nicht gefunden: {pfad}")

    with pfad.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict) or "begriffe" not in data:
        raise KatalogFehler("Katalog braucht ein Top-Level-Feld 'begriffe'.")
    begriffe = data["begriffe"]
    if not isinstance(begriffe, list):
        raise KatalogFehler(
            f"'begriffe' muss eine Liste sein, nicht {type(begriffe).__name__}.")

    version = str(data.get("version", "0.0.0"))
    quelle = str(data.get("quelle", ""))

    eintraege: list[Katalogeintrag] = []
    gesehen: set[str] = set()
    for raw in begriffe:
        eintrag = _parse_eintrag(raw)
        if eintrag.id in gesehen:
            raise KatalogFehler(f"Doppelte Begriffs-ID im Katalog: {eintrag.id!r}")
        gesehen.add(eintrag.id)
        eintraege.append(eintrag)

    return Katalog(version=version, eintraege=eintraege, quelle=quelle)
