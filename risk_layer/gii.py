"""gii-Knotenextraktion (Parser-Spez. Kap. 4).

Parst das DTD-basierte gesetze-im-internet-XML in kanonische Norm-Knoten.
Granularität: Paragraph + Absatz + Nummer (zuverlässig aus der Struktur).
Satz-Ebene ist bewusst NICHT enthalten (heuristisch, unsicher – Spez. 4.2/4.3).

Reiner Lesepfad über die lokale XML – kein Netzaufruf.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .idschema import Quellentyp

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormKnoten:
    """Ein Norm-Knoten auf §/Absatz/Nummer-Granularität.

    ``gueltig_bis`` wird von DIESEM Parser nie gesetzt (die gii-XML liefert nur
    die geltende Fassung, kein Außerkrafttreten je Norm). Das Feld bleibt für
    die Fassungslogik /R30/ reserviert; ``graph._gueltig`` behandelt ``None``
    korrekt als „offen gültig".
    """

    id: str                 # kanonische ID, z. B. norm:UStG:2:abs2:nr1
    law: str
    para: str
    text: str
    titel: str = ""
    absatz: str | None = None
    nummer: str | None = None
    quellentyp: str = Quellentyp.NORM.value
    gueltig_ab: str | None = None
    gueltig_bis: str | None = None   # reserviert für /R30/, hier immer None


_WS = re.compile(r"\s+")
_ABS = re.compile(r"^\((\d+[a-z]*)\)")


def _text(el: ET.Element) -> str:
    return _WS.sub(" ", "".join(el.itertext())).strip()


def _law_aus_metadaten(root: ET.Element) -> tuple[str, str | None]:
    """LAW aus <amtabk> (Fallback <jurabk> ohne Jahres-Suffix) + Ausfertigungsdatum."""
    amtabk = None
    jurabk = None
    ausfertigung = None
    for md in root.iter("metadaten"):
        amtabk = amtabk or md.findtext("amtabk")
        jurabk = jurabk or md.findtext("jurabk")
        ausfertigung = ausfertigung or md.findtext("ausfertigung-datum")
        if amtabk and ausfertigung:
            break
    law = (amtabk or jurabk or "?").strip()
    # „UStG 1980" → „UStG" (Jahres-Suffix entfernen)
    law = re.sub(r"\s+\d{4}$", "", law)
    return law, ausfertigung


def _oberste_dls(el: ET.Element) -> list[ET.Element]:
    """Liefert die OBERSTE DL-Ebene unterhalb von ``el``.

    Steigt durch Zwischenelemente (LA …) ab, hält aber an jeder gefundenen DL
    an – verschachtelte DLs (Buchstaben unter Nummern, realer Fall § 4 UStG)
    werden NICHT mitgeliefert, sondern später mit ihrem nr-Kontext rekursiv
    verarbeitet. (`.//DL` fand auch die verschachtelten und ließ deren lit-IDs
    ohne nr-Kontext kollidieren.)
    """
    out: list[ET.Element] = []
    for kind in el:
        if kind.tag == "DL":
            out.append(kind)
        else:
            out.extend(_oberste_dls(kind))
    return out


def _dt_dd_paare(dl: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    """Paart DT/DD über die KIND-REIHENFOLGE statt ``zip`` getrennter Listen.

    Ein DT ohne folgendes DD fällt einzeln weg; ``zip(dts, dds)`` hätte ab dort
    ALLE weiteren Paarungen still verschoben (Marke n bekäme Text von Marke n+1).
    """
    paare: list[tuple[ET.Element, ET.Element]] = []
    offen: ET.Element | None = None
    for kind in dl:
        if kind.tag == "DT":
            if offen is not None:
                _log.debug("DT ohne DD übersprungen: %r", _text(offen))
            offen = kind
        elif kind.tag == "DD":
            if offen is not None:
                paare.append((offen, kind))
                offen = None
    if offen is not None:
        _log.debug("DT ohne DD übersprungen: %r", _text(offen))
    return paare


def _dl_knoten(
    dl: ET.Element, *, law: str, para: str, absn: str, titel: str,
    gueltig_ab: str | None, praefix: str, nummer: str | None,
) -> list[NormKnoten]:
    """Knoten einer DL (rekursiv): nr-/lit-Marken unterhalb von ``praefix``.

    ``praefix`` trägt den bereits aufgebauten ID-Pfad (idschema-Tiefenlabels),
    sodass ein Buchst. a unter Nr. 1 die ID ``…:abs1:nr1:lita`` erhält – distinkt
    zu Buchst. a unter Nr. 2 (sonst überschreibt INSERT OR REPLACE im Graph still).
    """
    knoten: list[NormKnoten] = []
    typ = (dl.get("Type") or "").lower()
    for dt, dd in _dt_dd_paare(dl):
        marke = _text(dt).rstrip(".)").strip()
        if not marke:
            continue
        if typ.startswith("alpha"):
            kid = f"{praefix}:lit{marke}"
            sub_nummer = nummer
        elif re.match(r"^\d+[a-z]?$", marke):
            kid = f"{praefix}:nr{marke}"
            sub_nummer = marke
        else:
            continue
        knoten.append(NormKnoten(
            id=kid, law=law, para=para, absatz=absn, nummer=sub_nummer,
            text=_text(dd), titel=titel, gueltig_ab=gueltig_ab,
        ))
        # verschachtelte Listen im DD (z. B. Buchstaben unter dieser Nummer)
        for sub in _oberste_dls(dd):
            knoten.extend(_dl_knoten(
                sub, law=law, para=para, absn=absn, titel=titel,
                gueltig_ab=gueltig_ab, praefix=kid, nummer=sub_nummer,
            ))
    return knoten


def parse_datei(pfad: str | Path) -> list[NormKnoten]:
    """Parst eine gii-XML-Datei zu Norm-Knoten."""
    root = ET.parse(pfad).getroot()
    law, gueltig_ab = _law_aus_metadaten(root)
    knoten: list[NormKnoten] = []

    for norm in root.findall("norm"):
        md = norm.find("metadaten")
        if md is None:
            continue
        enbez = md.findtext("enbez")
        if not enbez or not re.match(r"^§\s*\d", enbez):
            # Rahmen-/Gliederungsnorm überspringen (Struktur bleibt ungenutzt).
            # „§§ x bis y"-Einheiten fallen hier ebenfalls durch – das soll
            # sichtbar sein statt still geschluckt zu werden.
            if enbez and enbez.startswith("§"):
                _log.info("enbez ohne Einzel-§ übersprungen (%s): %r", law, enbez)
            continue
        para = re.sub(r"^§\s*", "", enbez).strip()
        titel = (md.findtext("titel") or "").strip()
        content = norm.find("textdaten/text/Content")
        if content is None:
            continue

        # §-Knoten (Volltext)
        para_text = _text(content)
        if not para_text:
            continue
        knoten.append(NormKnoten(
            id=f"norm:{law}:{para}", law=law, para=para, text=para_text,
            titel=titel, gueltig_ab=gueltig_ab,
        ))

        # Absatz- und Nummern-Knoten
        for p in content.findall("P"):
            ptext = _text(p)
            m = _ABS.match(ptext)
            if not m:
                continue
            absn = m.group(1)
            abs_id = f"norm:{law}:{para}:abs{absn}"
            knoten.append(NormKnoten(
                id=abs_id, law=law, para=para, absatz=absn,
                text=ptext, titel=titel, gueltig_ab=gueltig_ab,
            ))
            for dl in _oberste_dls(p):
                knoten.extend(_dl_knoten(
                    dl, law=law, para=para, absn=absn, titel=titel,
                    gueltig_ab=gueltig_ab, praefix=abs_id, nummer=None,
                ))
    return knoten


def parse_verzeichnis(verzeichnis: str | Path) -> list[NormKnoten]:
    """Parst alle *.xml eines Korpus-Verzeichnisses.

    Robust gegen einzelne korrupte Dateien: ein Parse-Fehler bricht NICHT den
    gesamten Korpus-Import ab, sondern wird je Datei geloggt; die übrigen
    Dateien werden weiterverarbeitet.
    """
    knoten: list[NormKnoten] = []
    dateien = sorted(Path(verzeichnis).glob("*.xml"))
    fehler: list[tuple[str, str]] = []
    for pfad in dateien:
        try:
            knoten.extend(parse_datei(pfad))
        except (ET.ParseError, OSError) as exc:
            fehler.append((str(pfad), str(exc)))
            _log.warning("gii-Datei übersprungen (%s): %s", pfad, exc)
    if fehler:
        _log.warning("%d von %d gii-Dateien fehlerhaft übersprungen",
                     len(fehler), len(dateien))
    return knoten
