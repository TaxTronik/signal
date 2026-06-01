"""Fall-Datenbank – Fälle speichern und später weiterbearbeiten.

Ein „Fall" bündelt den Sachverhaltstext, die Analyse-Optionen und die EIGENEN
(manuellen) Markierungen des Beraters. So wird der Fall zur Arbeitsumgebung:
öffnen, weiterarbeiten, erneut speichern.

ACHTUNG: Fälle enthalten Mandantendaten (§ 203 StGB) – das Verzeichnis ist in
.gitignore ausgenommen und verlässt die Maschine nicht.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import GUELTIGE_GOVERNANCE, GUELTIGE_SCHADEN
from .persistenz import als_int as _as_int, atomar_schreiben, slugify

# Markierungsfarbe: NUR Hex erlaubt. Sonst landet ein gespeicherter Fall-Wert wie
# `red" onmouseover="…` ungeprüft im style-Attribut des Overlays (XSS). Ungültiges
# fällt auf den Default zurück – autoritativ hier, weil jeder Save hier durchläuft.
_DEFAULT_FARBE = "#fde68a"
_HEX_FARBE = re.compile(r"^#[0-9a-fA-F]{3,8}$")


def _safe_farbe(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    return s if _HEX_FARBE.match(s) else _DEFAULT_FARBE


def _jetzt() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(name: str) -> str:
    # Separator '-' (Datei-/ID-Stil dieses Stores); Algorithmik zentral in slugify.
    return slugify(name, default="fall", sep="-")


def _norm_risiko(r: Any) -> dict[str, Any] | None:
    """Vierdimensionale Risikomatrix + TCMS-Felder einer eigenen Markierung.

    Bewusst eine eigene, getrennte Struktur (nicht in die Markierung gemischt),
    damit ein späteres TCMS / der Wissensgraph sie als erstklassiges Risiko-Objekt
    aufgreifen kann. None, wenn die Markierung kein Risiko trägt.
    """
    if not isinstance(r, dict):
        return None
    # governance_typ/schadensintensitaet gegen die KANONISCHEN Enums prüfen
    # (dieselben wie im Berater-Store). Ungültige Werte → None, statt ungeprüft
    # als FP/FF/IN durchzureichen – der Downstream-TCMS-Consumer würde ihnen sonst
    # vertrauen. kaskadenreichweite robust coercen (kein 500 bei 'abc').
    gov = r.get("governance_typ") or None
    schaden = r.get("schadensintensitaet") or None
    return {
        "governance_typ": gov if gov in GUELTIGE_GOVERNANCE else None,
        "schadensintensitaet": schaden if schaden in GUELTIGE_SCHADEN else None,
        "wahrscheinlichkeit": r.get("wahrscheinlichkeit") or None,
        "kaskadenreichweite": _as_int(r.get("kaskadenreichweite"), None),
        "kontrolle": str(r.get("kontrolle", "")),
        "verantwortlich": str(r.get("verantwortlich", "")),
        "status": str(r.get("status", "offen")),
    }


def _norm_norm(n: Any) -> dict[str, Any] | None:
    """Optionaler Norm-Anker einer eigenen Markierung (id/zitat/titel)."""
    if not isinstance(n, dict):
        return None
    return {"id": str(n.get("id", "")), "zitat": str(n.get("zitat", "")),
            "titel": str(n.get("titel", ""))}


def _norm_markierung(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": _as_int(m.get("start"), 0),
        "end": _as_int(m.get("end"), 0),
        "label": str(m.get("label", "")),
        "farbe": _safe_farbe(m.get("farbe")),
        "kategorie": str(m.get("kategorie", "eigen")),
        "notiz": str(m.get("notiz", "")),
        "norm": _norm_norm(m.get("norm")),
        "risiko": _norm_risiko(m.get("risiko")),
    }


def _clamp_mark(m: dict[str, Any], textlen: int) -> dict[str, Any]:
    """Hält Span-Offsets in [0, textlen] und start <= end – verhindert, dass das
    Highlight-Overlay über das Textende hinaus oder rückwärts rendert."""
    s = max(0, min(m["start"], textlen))
    e = max(s, min(m["end"], textlen))
    m["start"], m["end"] = s, e
    return m


class FallStore:
    """Digitale Mandantenakte: Fälle (Sachverhalte) gruppiert nach Mandant.

    Physische Struktur – bewusst als ORDNER-pro-Mandant, damit sie sich 1:1 in
    einem Dateispeicher (TaxTronik/SeaweedFS) als Mandantenakte wiederfindet:

        faelle/<mandant-slug>/<fall-slug>.json   ← ein Sachverhalt
        faelle/<mandant-slug>/_mandant.json      ← Mandant-Metadaten (optional)

    Die Fall-ID ist der relative Pfad ``<mandant>/<fall>`` (ein Slash), damit die
    bestehende laden/speichern/loeschen-API über eine einzige ID adressiert.
    Alt-Fälle (flach unter faelle/*.json) bleiben lesbar und werden dem virtuellen
    Mandanten ``allgemein`` zugeschlagen.
    """

    ALLGEMEIN = "allgemein"

    def __init__(self, verzeichnis: str | Path):
        self.dir = Path(verzeichnis)
        # Serialisiert das read-modify-write in speichern()/loeschen() – im
        # ThreadingHTTPServer können zwei Saves derselben ID sonst Updates verlieren.
        self._lock = threading.Lock()

    # -- Pfad-Helfer (mit Traversal-Schutz) --------------------------------
    def _split(self, fid: str) -> tuple[str, str]:
        """Zerlegt eine Fall-ID in (mandant_slug, fall_slug); slug-gesichert."""
        teile = str(fid).replace("\\", "/").split("/")
        if len(teile) >= 2:
            return _slug(teile[0]), _slug(teile[-1])
        return self.ALLGEMEIN, _slug(teile[0])

    def _pfad(self, fid: str) -> Path:
        mand, fall = self._split(fid)
        return self.dir / mand / f"{fall}.json"

    def _altpfad(self, fall_slug: str) -> Path:
        return self.dir / f"{fall_slug}.json"   # Rückwärtskompat: flacher Alt-Fall

    # -- Mandanten ---------------------------------------------------------
    def mandanten(self) -> list[dict[str, Any]]:
        """Das Aktenregal: ein Eintrag je Mandant mit Fallzahl + letzter Änderung."""
        regal: dict[str, dict[str, Any]] = {}
        if not self.dir.exists():
            return []
        # Unterordner = Mandanten
        for ordner in self.dir.iterdir():
            if not ordner.is_dir():
                continue
            meta = self._mandant_meta(ordner.name)
            faelle = [p for p in ordner.glob("*.json") if p.name != "_mandant.json"]
            geaendert = max((self._stamp(p) for p in faelle), default="")
            regal[ordner.name] = {
                "slug": ordner.name, "name": meta.get("name", ordner.name),
                "mandantennummer": meta.get("mandantennummer", ""),
                "faelle": len(faelle), "geaendert_am": geaendert,
            }
        # flache Alt-Fälle → virtueller Mandant "allgemein"
        flach = [p for p in self.dir.glob("*.json")]
        if flach:
            r = regal.setdefault(self.ALLGEMEIN, {"slug": self.ALLGEMEIN, "name": "Allgemein",
                                                  "mandantennummer": "", "faelle": 0, "geaendert_am": ""})
            r["faelle"] += len(flach)
            r["geaendert_am"] = max([r["geaendert_am"], *(self._stamp(p) for p in flach)], default="")
        return sorted(regal.values(), key=lambda m: m.get("geaendert_am", ""), reverse=True)

    def mandant_speichern(self, slug: str, name: str, mandantennummer: str = "") -> dict[str, Any]:
        slug = _slug(slug or name)
        meta = {"slug": slug, "name": name.strip() or slug, "mandantennummer": mandantennummer.strip()}
        atomar_schreiben(self.dir / slug / "_mandant.json",
                         json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    def _mandant_meta(self, slug: str) -> dict[str, Any]:
        p = self.dir / slug / "_mandant.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    @staticmethod
    def _stamp(pfad: Path) -> str:
        # Letzte Änderung über st_mtime statt die ganze JSON zu parsen: mandanten()
        # braucht nur den jüngsten Zeitstempel je Mandant, nicht den Fallinhalt.
        # mtime ≈ geaendert_am (beim Speichern gesetzt) – für Anzeige/Sortierung gleich.
        try:
            return datetime.fromtimestamp(
                pfad.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        except OSError:
            return ""

    # -- Fälle -------------------------------------------------------------
    def liste(self, mandant: str | None = None) -> list[dict[str, Any]]:
        """Fälle eines Mandanten (oder aller) – jüngste zuerst, ohne Volltext."""
        eintraege: list[dict[str, Any]] = []
        if not self.dir.exists():
            return eintraege
        if mandant:
            mslug = _slug(mandant)
            if mslug == self.ALLGEMEIN:
                quellen = list(self.dir.glob("*.json"))   # flache Alt-Fälle
            else:
                quellen = [p for p in (self.dir / mslug).glob("*.json") if p.name != "_mandant.json"]
        else:
            quellen = [p for p in self.dir.rglob("*.json") if p.name != "_mandant.json"]
        for pfad in quellen:
            try:
                d = json.loads(pfad.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            mslug = pfad.parent.name if pfad.parent != self.dir else self.ALLGEMEIN
            eintraege.append({
                "id": d.get("id") or f"{mslug}/{pfad.stem}",
                "mandant": mslug,
                "name": d.get("name", pfad.stem),
                "erstellt_am": d.get("erstellt_am", ""),
                "geaendert_am": d.get("geaendert_am", ""),
                "vorschau": (d.get("text", "") or "")[:120],
                "markierungen": len(d.get("markierungen", [])),
            })
        eintraege.sort(key=lambda e: e.get("geaendert_am", ""), reverse=True)
        return eintraege

    def laden(self, fid: str) -> dict[str, Any] | None:
        pfad = self._pfad(fid)
        if not pfad.exists():
            # Fallback: Alt-Fall flach unter faelle/<fall>.json
            _, fall = self._split(fid)
            alt = self._altpfad(fall)
            if not alt.exists():
                return None
            pfad = alt
        return json.loads(pfad.read_text(encoding="utf-8"))

    def speichern(self, fall: dict[str, Any]) -> dict[str, Any]:
        """Legt einen Fall an oder aktualisiert ihn (stabile ID je Mandant+Name)."""
        name = str(fall.get("name") or "Unbenannter Fall").strip()
        mandant = _slug(str(fall.get("mandant") or self.ALLGEMEIN))
        # ID bestimmen: vorhandene ID respektieren, sonst <mandant>/<name>
        fid = str(fall.get("id") or f"{mandant}/{_slug(name)}")
        if "/" not in fid:
            fid = f"{mandant}/{_slug(fid)}"
        text = str(fall.get("text", ""))
        markierungen = [_clamp_mark(_norm_markierung(m), len(text))
                        for m in (fall.get("markierungen") or [])]
        # Lock: laden→schreiben ist EIN kritischer Abschnitt, sonst überschreiben
        # zwei gleichzeitige Saves derselben ID einander (lost update). Geschrieben
        # wird atomar – ein paralleler Leser sieht nie eine halbe Datei.
        with self._lock:
            vorhanden = self.laden(fid)
            eintrag = {
                "id": fid,
                "mandant": mandant,
                "name": name,
                "text": text,
                "optionen": fall.get("optionen", {}) or {},
                "markierungen": markierungen,
                "erstellt_am": (vorhanden or {}).get("erstellt_am") or _jetzt(),
                "geaendert_am": _jetzt(),
            }
            atomar_schreiben(self._pfad(fid),
                             json.dumps(eintrag, ensure_ascii=False, indent=2))
        return eintrag

    def loeschen(self, fid: str) -> bool:
        with self._lock:
            for pfad in (self._pfad(fid), self._altpfad(self._split(fid)[1])):
                if pfad.exists():
                    pfad.unlink()
                    return True
        return False
