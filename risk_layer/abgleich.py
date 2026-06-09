"""Abgleich-Schleife (Erweiterung §5, /LF340–370/).

Beobachtet neue BFH-Urteile und BMF-Schreiben (via RSS) und konfrontiert sie mit
den im Katalog/in Berater-Definitionen verankerten Normen: Dockt eine neue
Entscheidung/Verwaltungsanweisung an denselben Norm-Anker an, wird der betroffene
Begriff zur Prüfung vorgelegt.

Das System ENTSCHEIDET nicht, ob sich die Auslegung ändert – es macht die
Spannung sichtbar („konfrontieren, nicht entscheiden"). Reines lokales Matching
über den Zitat-Parser; nur der RSS-Abruf geht ins Netz.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import Katalog
from .idschema import para_level as _para_id
from .rss import RssEintrag, hole_alle
from .zitate import finde_normzitate, kanonisiere_fundstelle, parse_entscheidung

DEFAULT_STATE = Path("abgleich") / "gesehen.json"

_log = logging.getLogger(__name__)


@dataclass
class TrackingIndex:
    norm: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    az: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))


def baue_index(katalog: Katalog, berater=None) -> TrackingIndex:
    """Index: welche Begriffe hängen an welcher Norm (§-Ebene) bzw. welchem Urteil."""
    idx = TrackingIndex()
    from .zitate import parse_normzitat
    for e in katalog:
        for zitat in e.norm_anker:
            for a in parse_normzitat(zitat):
                if a.norm_id:
                    idx.norm[_para_id(a.norm_id.to_id())].append(
                        {"ref": e.id, "begriff": e.begriff_kanonisch, "quelle": "katalog", "anker": zitat})
        for q in e.definitionsquelle:
            kid = kanonisiere_fundstelle(q.typ, q.fundstelle)
            if kid and kid.startswith("urteil:"):
                idx.az[kid].append({"ref": e.id, "begriff": e.begriff_kanonisch, "quelle": "katalog"})
    if berater is not None:
        for d in berater.alle():
            for nid in d.geankert_an:
                idx.norm[_para_id(nid)].append(
                    {"ref": d.id, "begriff": d.begriff, "quelle": "berater", "anker": nid})
    return idx


@dataclass
class AbgleichTreffer:
    eintrag: RssEintrag
    norm_paras: list[str]
    az: list[str]
    betroffene: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "quelle": self.eintrag.quelle,
            "titel": self.eintrag.titel,
            "link": self.eintrag.link,
            "datum": self.eintrag.datum,
            "norm_paras": self.norm_paras,
            "az": self.az,
            "betroffene": self.betroffene,
        }


def _ids_aus_eintrag(eintrag: RssEintrag) -> tuple[set[str], set[str]]:
    paras = {_para_id(a.norm_id.to_id()) for a in finde_normzitate(eintrag.volltext) if a.norm_id}
    azs: set[str] = set()
    if eintrag.quelle == "BFH":
        uid = parse_entscheidung("BFH " + eintrag.titel)
        if uid:
            azs.add(uid.to_id())
    uid2 = parse_entscheidung(eintrag.volltext)
    if uid2:
        azs.add(uid2.to_id())
    return paras, azs


def match_eintrag(eintrag: RssEintrag, index: TrackingIndex) -> AbgleichTreffer | None:
    paras, azs = _ids_aus_eintrag(eintrag)
    betroffene: list[dict[str, Any]] = []
    for p in sorted(paras):
        for ref in index.norm.get(p, []):
            betroffene.append({**ref, "via": p})
    for a in sorted(azs):
        for ref in index.az.get(a, []):
            betroffene.append({**ref, "via": a})
    if not betroffene:
        return None
    return AbgleichTreffer(eintrag, sorted(paras), sorted(azs), betroffene)


def abgleich(eintraege: list[RssEintrag], index: TrackingIndex) -> list[AbgleichTreffer]:
    treffer = [t for e in eintraege if (t := match_eintrag(e, index)) is not None]
    return treffer


# --- Persistenz (nur Neues melden) -----------------------------------------

def lade_gesehen(state_pfad: Path = DEFAULT_STATE) -> set[str]:
    """Lädt die gesehenen GUIDs; eine korrupte State-Datei blockiert NICHT.

    Eine kaputte gesehen.json (abgebrochener Schreibvorgang, Handbearbeitung)
    würde sonst JEDEN weiteren Lauf unmöglich machen. Stattdessen: Warnung
    loggen, Original als ``.corrupt`` sichern (Diagnose) und als leer behandeln
    – die Folge sind schlimmstenfalls Doppelmeldungen, kein Datenverlust.
    """
    p = Path(state_pfad)
    if not p.exists():
        return set()
    try:
        roh = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(roh, list):
            raise ValueError(f"Liste erwartet, nicht {type(roh).__name__}")
        return {str(g) for g in roh}
    except (ValueError, OSError) as exc:  # json.JSONDecodeError ⊂ ValueError
        sicherung = p.with_suffix(p.suffix + ".corrupt")
        try:
            p.replace(sicherung)
            _log.warning("Korrupte State-Datei %s (%s) – gesichert nach %s, "
                         "starte mit leerem Status", p, exc, sicherung)
        except OSError:
            _log.warning("Korrupte State-Datei %s (%s) – Sicherung fehlgeschlagen, "
                         "starte mit leerem Status", p, exc)
        return set()


def speichere_gesehen(guids: set[str], state_pfad: Path = DEFAULT_STATE) -> None:
    p = Path(state_pfad)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(guids), ensure_ascii=False, indent=0), encoding="utf-8")


def laufe(
    katalog: Katalog, *, berater=None, eintraege: list[RssEintrag] | None = None,
    state_pfad: Path = DEFAULT_STATE, nur_neue: bool = True, fetch: bool = True,
) -> list[AbgleichTreffer]:
    """Vollständiger Lauf: (Feeds holen) → nur neue Einträge → Match → Status fortschreiben.

    Melde-Semantik AT-MOST-ONCE (bewusste Entscheidung): der gesehen-Status wird
    direkt nach dem Match fortgeschrieben – BEVOR der Aufrufer die Treffer
    konsumiert hat. Stürzt der Aufrufer danach ab, gelten die Einträge dennoch
    als gesehen und werden NICHT erneut gemeldet (Verlust möglich, keine
    Doppelmeldung). Das passt zum Werkzeugcharakter: ein verlorener Hinweis ist
    beim nächsten fachlichen Anlass wieder auffindbar, wiederholte Meldungen
    desselben Urteils würden dagegen das Vertrauen in die Liste erodieren.
    Ausnahme: ist die State-Datei korrupt, startet ``lade_gesehen`` leer – dann
    sind einmalige Doppelmeldungen möglich (siehe dort).
    """
    if eintraege is None:
        eintraege = hole_alle() if fetch else []
    index = baue_index(katalog, berater)
    gesehen = lade_gesehen(state_pfad) if nur_neue else set()
    zu_pruefen = [e for e in eintraege if e.guid not in gesehen]
    treffer = abgleich(zu_pruefen, index)
    if nur_neue:
        speichere_gesehen(gesehen | {e.guid for e in eintraege}, state_pfad)
    return treffer
