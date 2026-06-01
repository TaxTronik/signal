"""RSS-Feeds von BFH und BMF holen und parsen (Quelle der Abgleich-Schleife).

Öffentliche Rechtsquellen (keine Mandantendaten) – der Abruf ist ein erlaubter
Online-Schritt. Das Matching gegen Katalog/Definitionen läuft anschließend lokal.
"""

from __future__ import annotations

import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

BFH_URL = "https://www.bundesfinanzhof.de/de/precedent.rss"
BMF_URL = "https://www.bundesfinanzministerium.de/SiteGlobals/Functions/RSSFeed/DE/Steuern/RSSSteuern.xml"

# Obergrenze für eine Feed-Antwort: bounded read gegen eine (kompromittierte/
# fehlkonfigurierte) Riesen-Response. 16 MiB ist weit über jedem realen RSS.
# XXE selbst ist kein Thema – ElementTree expandiert keine externen Entitäten.
_MAX_FEED = 16 * 1024 * 1024

_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class RssEintrag:
    quelle: str          # "BFH" | "BMF"
    titel: str
    link: str
    guid: str
    datum: str
    beschreibung: str

    @property
    def volltext(self) -> str:
        return f"{self.titel}. {self.beschreibung}"


def _text(s: str | None) -> str:
    return html.unescape(_TAG.sub(" ", s or "")).strip()


def parse_rss(raw: bytes | str, quelle: str) -> list[RssEintrag]:
    root = ET.fromstring(raw)
    out: list[RssEintrag] = []
    for it in root.iter("item"):
        titel = _text(it.findtext("title"))
        link = (it.findtext("link") or "").strip()
        guid = (it.findtext("guid") or link).strip()
        datum = (it.findtext("pubDate") or "").strip()
        besch = _text(it.findtext("description"))
        if titel or besch:
            out.append(RssEintrag(quelle, titel, link, guid, datum, besch))
    return out


def hole_feed(url: str, quelle: str, *, timeout: float = 25.0) -> list[RssEintrag]:
    req = urllib.request.Request(url, headers={"User-Agent": "risk-layer/0.1 (lokal)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (öffentliche Rechtsquelle)
        return parse_rss(r.read(_MAX_FEED), quelle)


def hole_alle(*, timeout: float = 25.0) -> list[RssEintrag]:
    eintraege: list[RssEintrag] = []
    for url, quelle in ((BFH_URL, "BFH"), (BMF_URL, "BMF")):
        try:
            eintraege.extend(hole_feed(url, quelle, timeout=timeout))
        except Exception:  # noqa: BLE001 – ein Feed darf ausfallen
            continue
    return eintraege
