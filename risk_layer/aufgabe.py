"""Rechercheaufgaben – Delegation als kollaborativer Zusammenbau des Wissens.

Eine Rechercheaufgabe ist die menschliche Delegation („Junior, kläre die
Definition von X für diesen Fall") – im Unterschied zum Recherche*auftrag* an ein
LLM (`recherche.py`, Maschinen-Delegation, § 203-anonymisiert). Sie ist hier ein
ERSTKLASSIGES Objekt, weil ihr Abschluss der Mechanismus ist, durch den Katalog
und Wissensgraph wachsen: viele Hände bauen das gemeinsame Wissen zusammen.

Bewusste Trennlinie zur TaxTronik-Integration: Diese Struktur definiert das WAS
(von/an/Scope/Status/Ergebnis/Kontext-Anker) – integrationsfest und graph-kompatibel.
Das WER und die ZUSTELLUNG (Identität, Inbox, Benachrichtigung) gehören NICHT
hierher; sie docken an TaxTroniks vorhandenes Aufgaben-/Nutzersystem an. ``von``
und ``an`` sind hier freie Strings (Name/Rolle), kein eigenes Nutzermodell.

Verankerung (der Graph-Bezug): Jede Aufgabe trägt einen Kontext-Anker – Mandant,
Fall, optional die Textstelle (start/end) und der/die betroffene Begriff bzw.
Norm-Anker. So ist später eine Aufgabe ein Knoten, der an genau die Stelle im
Mandanten-Akten-/Wissensgraphen hängt, die sie klären soll.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .persistenz import als_int, atomar_schreiben, slugify

# Lebenszyklus – bewusst parallel zum Berater-Review (entwurf→geprüft→freigegeben),
# nur aufgabenseitig: offen → in_bearbeitung → erledigt → geprueft (Abnahme).
STATUS = ("offen", "in_bearbeitung", "erledigt", "geprueft", "verworfen")
_AKTIV = ("offen", "in_bearbeitung")


def _jetzt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_ID_ZAEHLER = 0


def _kurz_id() -> str:
    # kompakte, sortierbare ID aus Zeitstempel + Prozess-Zähler. Der Zähler
    # verhindert ID-Kollisionen bei mehreren Aufgaben in derselben Millisekunde
    # (zwei schnelle Delegationen würden sich sonst gegenseitig überschreiben).
    global _ID_ZAEHLER
    _ID_ZAEHLER += 1
    stamp = datetime.now(timezone.utc).strftime("a-%Y%m%d-%H%M%S-%f")
    return f"{stamp}-{_ID_ZAEHLER:03d}"


def _norm_aufgabe(a: dict[str, Any], *, vorhanden: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validiert/normalisiert eine Aufgabe; füllt Defaults, bewahrt erstellt_am."""
    status = str(a.get("status", "offen"))
    if status not in STATUS:
        status = "offen"
    anker = a.get("anker") or {}
    return {
        "id": str(a.get("id") or _kurz_id()),
        # Kontext-Verankerung (Graph-Bezug):
        "mandant": str(a.get("mandant", "")),
        "fall_id": str(a.get("fall_id", "")),
        "anker": {
            "start": als_int(anker.get("start"), None),
            "end": als_int(anker.get("end"), None),
            "stelle_text": str(anker.get("stelle_text", ""))[:200],
            "begriff": str(anker.get("begriff", "")),
            "norm": str(anker.get("norm", "")),
        },
        # Delegation (freie Strings – KEIN eigenes Nutzermodell, s. Modul-Doc):
        "von": str(a.get("von", "")),
        "an": str(a.get("an", "")),
        # Auftrag:
        "titel": str(a.get("titel", "")).strip() or "Rechercheaufgabe",
        "frage": str(a.get("frage", "")),
        "prioritaet": str(a.get("prioritaet", "normal")),   # niedrig|normal|hoch
        # Workflow:
        "status": status,
        # Ergebnis (fließt beim Abschluss in eine Berater-Definition zurück):
        "ergebnis": str(a.get("ergebnis", "")),
        "ergebnis_quellen": list(a.get("ergebnis_quellen") or []),
        "berater_def_id": (a.get("berater_def_id") or None),
        # Zeitstempel:
        "erstellt_am": (vorhanden or {}).get("erstellt_am") or _jetzt(),
        "geaendert_am": _jetzt(),
    }


class AufgabenStore:
    """Persistiert Rechercheaufgaben als JSON (eine Datei je Aufgabe).

    Liegt unter ``<basis>/<id>.json`` (z. B. ``aufgaben/``). Mandanten-/Fall-Bezug
    steckt IM Objekt (Felder ``mandant``/``fall_id``), nicht im Pfad – Aufgaben
    sind quer zu Akten filterbar („alle offenen Aufgaben", „meine Aufgaben").
    """

    def __init__(self, verzeichnis: str | Path):
        self.dir = Path(verzeichnis)
        # Serialisiert read-modify-write (speichern/status_setzen) im ThreadingHTTPServer.
        self._lock = threading.Lock()

    def _pfad(self, aid: str) -> Path:
        # Zentrale, traversal-sichere Slug-Bildung (war ein dritter Ad-hoc-Sanitizer).
        return self.dir / f"{slugify(aid, default='aufgabe', sep='-')}.json"

    def speichern(self, a: dict[str, Any]) -> dict[str, Any]:
        aid = str(a.get("id") or "")
        with self._lock:
            vorhanden = self.laden(aid) if aid else None
            eintrag = _norm_aufgabe(a, vorhanden=vorhanden)
            atomar_schreiben(self._pfad(eintrag["id"]),
                             json.dumps(eintrag, ensure_ascii=False, indent=2))
        return eintrag

    def laden(self, aid: str) -> dict[str, Any] | None:
        p = self._pfad(aid)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def liste(self, *, mandant: str | None = None, fall_id: str | None = None,
              an: str | None = None, nur_aktiv: bool = False) -> list[dict[str, Any]]:
        """Aufgaben gefiltert (Mandant/Fall/Bearbeiter/aktiv) – jüngste zuerst."""
        out: list[dict[str, Any]] = []
        if not self.dir.exists():
            return out
        for p in self.dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if mandant and d.get("mandant") != mandant:
                continue
            if fall_id and d.get("fall_id") != fall_id:
                continue
            if an and d.get("an") != an:
                continue
            if nur_aktiv and d.get("status") not in _AKTIV:
                continue
            out.append(d)
        out.sort(key=lambda x: x.get("geaendert_am", ""), reverse=True)
        return out

    def status_setzen(self, aid: str, status: str) -> dict[str, Any] | None:
        d = self.laden(aid)
        if d is None:
            return None
        if status in STATUS:
            d["status"] = status
        return self.speichern(d)

    def loeschen(self, aid: str) -> bool:
        with self._lock:
            p = self._pfad(aid)
            if p.exists():
                p.unlink()
                return True
        return False
