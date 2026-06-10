"""Norm-Änderungs-Radar: Re-Review-Trigger nach Korpus-/Graph-Rebuild.

WARUM: Freigegebene Berater-Definitionen (berater.py) veralten heute STILL,
wenn der Gesetzgeber den verankerten Paragraphen ändert – der Review-
Lebenszyklus (entwurf → geprüft → freigegeben) kennt nur Vorwärts-Schritte
und keinen Re-Review-Trigger. Das Radar schließt diese Lücke datenseitig:
Bei jedem ``graph build`` wird ein Manifest der Knoten-Inhalts-Hashes
(``NormGraph.knoten_hashes()``) persistiert; nach dem nächsten Rebuild lässt
sich daraus benennen, WELCHE Normen sich geändert haben / entfallen sind und
welche Festwissens-Anker davon betroffen sind (Berater-Definitionen über
``geankert_an``, Norm-Kuratierungs-Records, Katalog-Anker).

Die Engine MUTIERT NICHTS: sie berichtet nur – der Host (TaxTronik) entscheidet
über Aufgaben/Re-Review und protokolliert die Handlung in SEINER Audit-Chain.
Bewusst KEINE automatische Status-Rücksetzung (freigegeben → entwurf): das
bräche die Nur-vorwärts-Invariante des Lebenszyklus
(``BeraterStore.setze_review_status``) und nähme dem Berufsträger die fachliche
Entscheidung ab, ob die Gesetzesänderung die Definition überhaupt berührt
(§§ 33, 57 StBerG: Subsumtion bleibt beim Menschen).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .persistenz import atomar_schreiben
from .zitate import parse_normzitat

MANIFEST_VERSION = 1
# Dateinamen NEBEN der Graph-Datei (gleiches Verzeichnis wie graph.sqlite):
# der aktuelle Stand + der beim letzten Build rotierte Vorher-Stand.
MANIFEST_NAME = "norm_manifest.json"
MANIFEST_VORHER_NAME = "norm_manifest.vorher.json"


class RadarFehler(ValueError):
    """Strukturfehler im Norm-Manifest (korrupte Datei / falsche Version)."""


def manifest_pfad(graph_db: str | Path) -> Path:
    """Pfad des aktuellen Manifests, abgeleitet vom Graph-Pfad (selbes Verzeichnis)."""
    return Path(graph_db).parent / MANIFEST_NAME


def manifest_vorher_pfad(graph_db: str | Path) -> Path:
    """Pfad des rotierten Vorher-Manifests (Vergleichsstand für das Radar)."""
    return Path(graph_db).parent / MANIFEST_VORHER_NAME


# -- Manifest -----------------------------------------------------------------

def erstelle_manifest(graph, erstellt_am: str) -> dict[str, Any]:
    """Schnappschuss des Korpus-Stands: Fingerprint + Inhalts-Hash je Knoten.

    ``erstellt_am`` kommt vom Aufrufer (ISO-String) – keine eigene Uhr in der
    Engine, damit Läufe reproduzierbar bleiben (Determinismus-Disziplin).
    """
    return {
        "version": MANIFEST_VERSION,
        "erstellt_am": erstellt_am,
        "fingerprint": graph.fingerprint(),
        "knoten": graph.knoten_hashes(),
    }


def speichere_manifest(pfad: str | Path, manifest: dict[str, Any]) -> None:
    """Schreibt das Manifest atomar (kein torn read) und DETERMINISTISCH:
    sortierte Keys ⇒ gleicher Stand → byte-gleiche Datei (diff-/git-freundlich)."""
    atomar_schreiben(
        Path(pfad),
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def lade_manifest(pfad: str | Path) -> dict[str, Any] | None:
    """Lädt ein Manifest. Tolerant NUR gegenüber Abwesenheit:

    * Datei fehlt → ``None`` (es gibt schlicht keinen Vergleichsstand – das
      ist der Normalzustand vor dem zweiten ``graph build``, kein Fehler).
    * korrupte Datei / falsche Version → ``RadarFehler`` mit Pfad (laut
      scheitern statt still einen leeren Diff zu melden).
    """
    pfad = Path(pfad)
    if not pfad.exists():
        return None
    try:
        data = json.loads(pfad.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RadarFehler(f"{pfad}: Manifest nicht lesbar (korrupt?): {exc}") from exc
    if (not isinstance(data, dict) or not isinstance(data.get("knoten"), dict)
            or not isinstance(data.get("fingerprint"), str)):
        raise RadarFehler(
            f"{pfad}: kein Norm-Manifest (erwartet Mapping mit 'version', "
            f"'fingerprint' und 'knoten')")
    if data.get("version") != MANIFEST_VERSION:
        raise RadarFehler(
            f"{pfad}: Manifest-Version {data.get('version')!r} wird nicht "
            f"unterstützt (erwartet {MANIFEST_VERSION})")
    return data


def diff_manifeste(alt: dict[str, Any], neu: dict[str, Any]) -> dict[str, list[str]]:
    """Knoten-Diff zweier Manifeste: geändert / neu / entfallen (sortiert).

    Sortierte Listen ⇒ deterministische Berichte (gleiches Paar → gleiche
    Ausgabe), unabhängig von Dict-Reihenfolgen der geladenen JSON-Dateien.
    """
    ka: dict[str, str] = alt.get("knoten", {})
    kn: dict[str, str] = neu.get("knoten", {})
    return {
        "geaendert": sorted(i for i in ka.keys() & kn.keys() if ka[i] != kn[i]),
        "neu": sorted(kn.keys() - ka.keys()),
        "entfallen": sorted(ka.keys() - kn.keys()),
    }


# -- Anker-Kreuzung -------------------------------------------------------------

def anker_trifft(anker_id: str, knoten_id: str) -> bool:
    """Trifft eine Knoten-Änderung einen Norm-Anker? Pfad-SEGMENT-Vergleich.

    Betroffen, wenn Anker == Knoten ODER der Knoten unterhalb des Ankers liegt
    (Anker ``norm:KStG:8c`` ↔ geänderter Satz ``norm:KStG:8c:abs1:s2``) ODER
    der Anker unterhalb des Knotens liegt (Anker auf Satz-Ebene, der Text hängt
    aber am §-Knoten). Verglichen werden ':'-Segmente, KEIN Substring-Match:
    ``norm:AO:14`` darf ``norm:AO:140`` nicht treffen.
    """
    ta, tk = str(anker_id).split(":"), str(knoten_id).split(":")
    n = min(len(ta), len(tk))
    return n > 0 and ta[:n] == tk[:n]


# Dringlichkeits-Rang (kleiner = dringender): freigegebene GETEILTE Definitionen
# sind kanonisches Wissen für ALLE Nutzer – ihre stille Veralterung ist der
# haftungsrelevanteste Fall, deshalb stehen sie vorn. Kuratierungen/Katalog
# folgen: sie tragen keinen Review-Lebenszyklus (der Katalog wird über die
# Git-Pflege der YAML fortgeschrieben, nicht über Review-Status).
_REVIEW_RANG = {"freigegeben": 0, "geprüft": 1, "entwurf": 2}


def _dringlichkeit(t: dict[str, Any]) -> tuple:
    if t["typ"] == "berater":
        rang = (_REVIEW_RANG.get(t["review_status"], 2) if t["scope"] == "geteilt"
                else 3)                                       # personal zuletzt
    elif t["typ"] == "kuratierung":
        rang = 4 if t["scope"] == "geteilt" else 5
    else:                                                     # katalog
        rang = 6
    # Sekundärschlüssel machen die Reihenfolge VOLLSTÄNDIG deterministisch.
    return (rang, t["typ"], t["id"], t["norm_id"], t["ausloeser"], t["aenderung"])


def _treffer(typ: str, id_: str, begriff: str | None, scope: str | None,
             review_status: str | None, aktion: str | None,
             norm_id: str, ausloeser: str, aenderung: str) -> dict[str, Any]:
    """Ein betroffener Anker – FESTES Schema (alle Keys immer vorhanden, ggf.
    None), damit der Host nicht je Typ andere Felder behandeln muss."""
    return {"typ": typ, "id": id_, "begriff": begriff, "scope": scope,
            "review_status": review_status, "aktion": aktion,
            "norm_id": norm_id, "ausloeser": ausloeser, "aenderung": aenderung}


def betroffene_anker(diff: dict[str, Any], berater_store, katalog) -> dict[str, Any]:
    """Kreuzt geänderte/entfallene Norm-Knoten mit allen Festwissens-Ankern.

    Quellen: (a) Berater-Definitionen über ``geankert_an`` (Knoten-IDs),
    (b) Norm-Kuratierungs-Records (``norm_id``), (c) Katalog-Anker (Zitate,
    hier via Zitat-Parser auf Knoten-IDs gehoben). ``berater_store``/``katalog``
    dürfen ``None`` sein – die ausgelassene Quelle wird in ``hinweise``
    benannt (kein stilles „nichts betroffen").

    NEUE Knoten lösen bewusst KEINEN Befund aus: ein neuer Unterabsatz macht
    eine bestehende Definition nicht automatisch falsch – geändert/entfallen
    dagegen schon potenziell.

    Rückgabe: ``{"betroffen": [Treffer …], "anzahl": {...}, "hinweise": [...]}``,
    Treffer sortiert nach Review-Dringlichkeit (freigegeben → geprüft →
    entwurf → personal, dann Kuratierung, dann Katalog).
    """
    aenderungen = ([(k, "geaendert") for k in diff.get("geaendert", ())]
                   + [(k, "entfallen") for k in diff.get("entfallen", ())])
    treffer: list[dict[str, Any]] = []
    hinweise: list[str] = []

    def _pruefe(anker_ids, baue) -> None:
        for anker in anker_ids:
            for knoten, art in aenderungen:
                if anker_trifft(anker, knoten):
                    treffer.append(baue(anker, knoten, art))

    if berater_store is None:
        hinweise.append("Berater-Store nicht übergeben – Berater-Anker und "
                        "Norm-Kuratierungen ungeprüft.")
    else:
        for d in berater_store.alle():
            # geankert_an trägt bereits Knoten-IDs (beim definiere() via
            # Zitat-Parser aus norm_anker gehoben).
            _pruefe(d.geankert_an, lambda a, k, art, d=d: _treffer(
                "berater", d.id, d.begriff, d.scope, d.review_status, None,
                a, k, art))
        # ALLE Kuratierungs-Records (geteilt UND personal): das Radar ist eine
        # Wartungssicht über den GESAMTEN Festwissens-Bestand. Die öffentliche
        # ``kuratierungen()``-API filtert dagegen auf die Sicht EINES Nutzers
        # (geteilt + eigene) – deshalb hier der Roh-Bestand; der Scope steht an
        # jedem Treffer, filtern kann der Host. Fallback auf die öffentliche
        # API hält Duck-Typing-Stubs ohne Privatfeld funktionsfähig.
        records = getattr(berater_store, "_kuratierungen", None)
        if records is None:
            records = berater_store.kuratierungen()
        for rec in records:
            _pruefe((rec.norm_id,), lambda a, k, art, rec=rec: _treffer(
                "kuratierung", rec.katalog_id, None, rec.scope, None,
                rec.aktion, a, k, art))

    if katalog is None:
        hinweise.append("Katalog nicht übergeben – Katalog-Anker ungeprüft.")
    else:
        mit_anker = 0
        for e in katalog:
            # Katalog-Anker sind ZITATE ("§ 2 Abs. 2 Nr. 2 UStG"), keine
            # Knoten-IDs – erst via Zitat-Parser heben (wie berater._anker_zu_knoten).
            ids = tuple(dict.fromkeys(
                a.norm_id.to_id()
                for zitat in e.norm_anker
                for a in parse_normzitat(zitat) if a.norm_id))
            if ids:
                mit_anker += 1
            _pruefe(ids, lambda a, k, art, e=e: _treffer(
                "katalog", e.id, e.begriff_kanonisch, None, None, None,
                a, k, art))
        if not mit_anker:
            hinweise.append("Katalogeinträge tragen keine auflösbaren "
                            "Norm-Anker – Katalog faktisch ungeprüft.")

    # Duplikate deterministisch entfernen (z. B. derselbe Anker doppelt am
    # Eintrag) – Schlüssel über ALLE Felder, Reihenfolge danach per Sortierung.
    eindeutig = list({tuple(sorted(t.items(), key=lambda kv: kv[0])): t
                      for t in treffer}.values())
    eindeutig.sort(key=_dringlichkeit)
    anzahl = {
        "betroffen": len(eindeutig),
        # „davon freigegeben" = die Fälle mit höchster Review-Dringlichkeit
        "freigegeben": sum(1 for t in eindeutig
                           if t["typ"] == "berater" and t["scope"] == "geteilt"
                           and t["review_status"] == "freigegeben"),
    }
    return {"betroffen": eindeutig, "anzahl": anzahl, "hinweise": hinweise}


# -- Komfort: kompletter Bericht -------------------------------------------------

def radar_bericht(alt_manifest: dict[str, Any], graph, berater_store, katalog,
                  erstellt_am: str) -> dict[str, Any]:
    """Kompletter Radar-Lauf: neues Manifest aus dem Graph, Diff, Anker-Kreuzung.

    Reiner Funktionskern ohne I/O außer Graph-LESEN – direkt vom Host (z. B.
    einem künftigen /v1-Endpunkt) nutzbar; Persistenz/Rotation macht der
    Aufrufer (CLI: ``graph build`` rotiert, ``radar`` liest).
    """
    neu = erstelle_manifest(graph, erstellt_am)
    diff = diff_manifeste(alt_manifest, neu)
    kreuz = betroffene_anker(diff, berater_store, katalog)
    return {
        "ok": True,
        "stand_alt": {"erstellt_am": alt_manifest.get("erstellt_am"),
                      "fingerprint": alt_manifest.get("fingerprint")},
        "stand_neu": {"erstellt_am": neu["erstellt_am"],
                      "fingerprint": neu["fingerprint"]},
        "normen": diff,
        "betroffen": kreuz["betroffen"],
        "hinweise": kreuz["hinweise"],
        "anzahl": {"geaendert": len(diff["geaendert"]),
                   "neu": len(diff["neu"]),
                   "entfallen": len(diff["entfallen"]),
                   **kreuz["anzahl"]},
    }
