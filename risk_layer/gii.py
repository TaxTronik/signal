"""gii-Knotenextraktion (Parser-Spez. Kap. 4).

Parst das DTD-basierte gesetze-im-internet-XML in kanonische Norm-Knoten.
Granularität: Paragraph + Absatz + Nummer (zuverlässig aus der Struktur).
Satz-Ebene ist bewusst NICHT enthalten (heuristisch, unsicher – Spez. 4.2/4.3).

Reiner Lesepfad über die lokale XML – kein Netzaufruf.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .idschema import Quellentyp


@dataclass(frozen=True)
class NormKnoten:
    """Ein Norm-Knoten auf §/Absatz/Nummer-Granularität."""

    id: str                 # kanonische ID, z. B. norm:UStG:2:abs2:nr1
    law: str
    para: str
    text: str
    titel: str = ""
    absatz: str | None = None
    nummer: str | None = None
    quellentyp: str = Quellentyp.NORM.value
    gueltig_ab: str | None = None
    gueltig_bis: str | None = None


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
            continue  # Rahmen-/Gliederungsnorm überspringen (Struktur bleibt ungenutzt)
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
            knoten.append(NormKnoten(
                id=f"norm:{law}:{para}:abs{absn}", law=law, para=para, absatz=absn,
                text=ptext, titel=titel, gueltig_ab=gueltig_ab,
            ))
            for dl in p.findall(".//DL"):
                typ = (dl.get("Type") or "").lower()
                dts = dl.findall("DT")
                dds = dl.findall("DD")
                for dt, dd in zip(dts, dds):
                    marke = _text(dt).rstrip(".)").strip()
                    if not marke:
                        continue
                    if typ.startswith("alpha"):
                        kid = f"norm:{law}:{para}:abs{absn}:lit{marke}"
                        knoten.append(NormKnoten(
                            id=kid, law=law, para=para, absatz=absn, nummer=None,
                            text=_text(dd), titel=titel, gueltig_ab=gueltig_ab,
                        ))
                    elif re.match(r"^\d+[a-z]?$", marke):
                        kid = f"norm:{law}:{para}:abs{absn}:nr{marke}"
                        knoten.append(NormKnoten(
                            id=kid, law=law, para=para, absatz=absn, nummer=marke,
                            text=_text(dd), titel=titel, gueltig_ab=gueltig_ab,
                        ))
    return knoten


def parse_verzeichnis(verzeichnis: str | Path) -> list[NormKnoten]:
    """Parst alle *.xml eines Korpus-Verzeichnisses."""
    knoten: list[NormKnoten] = []
    for pfad in sorted(Path(verzeichnis).glob("*.xml")):
        knoten.extend(parse_datei(pfad))
    return knoten
