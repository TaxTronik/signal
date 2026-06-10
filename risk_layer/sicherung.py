"""Sicherung/Wiederherstellung des NICHT-regenerierbaren Festwissens (PS-861, L5).

WARUM dieses Modul existiert: Graph, Trigger-Index und Embedding-Index sind aus
amtlichen Quellen jederzeit regenerierbar ('korpus fetch' + 'graph build' +
'embedding build') — die Berater-Definitionen und Norm-Kuratierungen sind es
NICHT. Sie sind kuratierte menschliche Arbeit (personal.yaml / geteilt.yaml /
norm_kuratierung.yaml im BeraterStore-Verzeichnis, s. berater.py) und gingen
bei Datenträgerverlust ersatzlos verloren. Das IDW-PS-861-Mapping weist genau
das als Lücke L5 aus (Tz. 43: Sicherungs-/Wiederherstellungsverfahren für
kritische Bestände); dieses Modul schließt sie engine-seitig.

Bewusst NICHT enthalten: Korpus, Normgraph, Trigger- und Embedding-Index —
regenerierbare Build-Artefakte gehören nicht ins Backup (sie würden es nur
aufblähen und die Wiederherstellung verschleiern: Restore heißt hier exakt
"kuratiertes Wissen zurück", alles andere wird neu gebaut).

Sicherheits-Disziplin beim Entpacken: NUR die erwarteten YAML-Namen werden
wiederhergestellt; jeder andere Archiv-Inhalt bricht laut ab (Whitelist) und
zusätzlich wird jeder Member-Pfad gegen das Zielverzeichnis normalisiert —
dieselbe Zip-Slip-Abwehr, die modell._lade_und_entpacke vormacht.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

# Die Dateien, die der BeraterStore hält (berater.py: _datei('personal'/'geteilt')
# + norm_kuratierung.yaml). FESTE Tupel-Reihenfolge = deterministische Reihenfolge
# der Archiv-Mitglieder (gleicher Store-Stand → gleiche Member-Liste).
FESTWISSEN_DATEIEN: tuple[str, ...] = ("personal.yaml", "geteilt.yaml",
                                       "norm_kuratierung.yaml")
ARCHIV_PREFIX = "festwissen_"

# Hinweis im Zip-Kommentar eines leeren Archivs: ein leerer Store ist KEIN
# Fehler (frische Installation), aber das Archiv soll selbsterklärend sein.
_LEER_HINWEIS = ("leerer Berater-Store: keine Festwissens-YAMLs vorhanden "
                 "(gültiges, leeres Archiv)")


class SicherungsFehler(ValueError):
    """Bedienungs-/Strukturfehler bei Sicherung oder Wiederherstellung."""


def zeitstempel_dateiname(zeitstempel: str) -> str:
    """Macht einen (ISO-)Zeitstempel dateinamenstauglich.

    ':' ist unter Windows in Dateinamen verboten ('2026-06-10T12:00:00' wäre
    kein gültiger Name) — alle Nicht-[Alnum ._-]-Zeichen werden zu '-'.
    Zentral hier, damit alle datierten Artefakte (Festwissen-Archiv,
    Radar-Archiv, Berichte) dieselbe Namensbildung nutzen.
    """
    s = "".join(c if (c.isalnum() or c in "._-") else "-"
                for c in str(zeitstempel).strip())
    s = s.strip("-.")
    if not s:
        raise SicherungsFehler(f"Zeitstempel ergibt keinen Dateinamen: {zeitstempel!r}")
    return s


def sichere_festwissen(berater_dir: str | Path, ziel_dir: str | Path,
                       zeitstempel: str) -> Path:
    """Sichert die Festwissens-YAMLs des Berater-Stores als datiertes Zip.

    Archivname: ``festwissen_<zeitstempel>.zip`` (Zeitstempel dateinamen-
    stauglich gemacht). Mitglieder in der festen ``FESTWISSEN_DATEIEN``-
    Reihenfolge — deterministisch, damit zwei Sicherungen desselben Stands
    vergleichbar bleiben. Ein leerer Store (Verzeichnis existiert, aber keine
    der YAMLs) ergibt ein GÜLTIGES leeres Archiv mit Hinweis im Zip-Kommentar
    — kein Fehler, sonst scheiterte das Backup ausgerechnet auf frischen
    Installationen und würde aus Betriebsplänen wieder gestrichen.

    Geschrieben wird atomar (temp im Zielverzeichnis + ``os.replace``, Muster
    aus persistenz.atomar_schreiben): ein halb geschriebenes Backup wäre
    schlimmer als keins — es sähe vorhanden aus, wäre aber unbrauchbar.
    """
    berater_dir = Path(berater_dir)
    if not berater_dir.is_dir():
        # Ein Tippfehler im Pfad darf NICHT still ein leeres "Backup" erzeugen —
        # das würde im Ernstfall als vorhandene Sicherung missverstanden.
        raise SicherungsFehler(
            f"Berater-Verzeichnis existiert nicht: {berater_dir}")
    ziel_dir = Path(ziel_dir)
    ziel_dir.mkdir(parents=True, exist_ok=True)
    pfad = ziel_dir / f"{ARCHIV_PREFIX}{zeitstempel_dateiname(zeitstempel)}.zip"

    vorhanden = [n for n in FESTWISSEN_DATEIEN if (berater_dir / n).is_file()]
    fd, tmp = tempfile.mkstemp(dir=str(ziel_dir), prefix=f".{pfad.name}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            with zipfile.ZipFile(fh, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in vorhanden:                  # feste Reihenfolge
                    zf.write(berater_dir / name, arcname=name)
                zf.comment = (
                    f"Festwissen-Sicherung {zeitstempel} aus {berater_dir.name}"
                    if vorhanden else _LEER_HINWEIS).encode("utf-8")
        os.replace(tmp, pfad)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return pfad


def stelle_wieder_her(archiv: str | Path, ziel_dir: str | Path,
                      force: bool = False) -> list[str]:
    """Stellt Festwissens-YAMLs aus einem Sicherungsarchiv wieder her.

    Entpackt AUSSCHLIESSLICH die erwarteten Namen (``FESTWISSEN_DATEIEN``):
    jeder fremde Member bricht laut ab — ein Backup-Restore darf nie zum
    Einfallstor für beliebige Dateien werden (Whitelist + Zip-Slip-Abwehr wie
    in modell._lade_und_entpacke; die Pfad-Normalisierung bleibt als zweite
    Verteidigungslinie, obwohl die Whitelist nur nackte Dateinamen zulässt).

    Ein nicht-leeres Ziel wird ohne ``force`` NICHT angefasst: Wiederherstellen
    überschreibt kuratiertes Festwissen — das muss eine ausdrückliche,
    bewusste Entscheidung sein, kein Nebeneffekt.

    Rückgabe: die wiederhergestellten Dateinamen (in Archiv-Reihenfolge).
    """
    archiv = Path(archiv)
    if not archiv.is_file():
        raise SicherungsFehler(f"Archiv nicht gefunden: {archiv}")
    ziel_dir = Path(ziel_dir)
    if ziel_dir.exists() and any(ziel_dir.iterdir()) and not force:
        raise SicherungsFehler(
            f"Ziel {ziel_dir} ist nicht leer — Wiederherstellen würde "
            "bestehendes Festwissen überschreiben. Mit force=True (CLI: "
            "--force) ausdrücklich bestätigen.")
    try:
        zf = zipfile.ZipFile(archiv)
    except zipfile.BadZipFile as exc:
        raise SicherungsFehler(f"{archiv}: kein gültiges Zip-Archiv ({exc})") from exc
    with zf:
        ziel = ziel_dir.resolve()
        for m in zf.namelist():
            if m not in FESTWISSEN_DATEIEN:
                raise SicherungsFehler(
                    f"{archiv}: unerwarteter Archiv-Inhalt abgewehrt: {m!r} "
                    f"(erlaubt sind nur {', '.join(FESTWISSEN_DATEIEN)})")
            try:
                (ziel_dir / m).resolve().relative_to(ziel)
            except ValueError:
                raise SicherungsFehler(
                    f"{archiv}: unsicherer Pfad im Archiv (Zip-Slip): {m!r}") from None
        ziel_dir.mkdir(parents=True, exist_ok=True)
        wiederhergestellt: list[str] = []
        for m in zf.namelist():
            zf.extract(m, ziel_dir)
            wiederhergestellt.append(m)
    return wiederhergestellt
