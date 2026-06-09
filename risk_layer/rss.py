"""RSS-Feeds von BFH und BMF holen und parsen (Quelle der Abgleich-Schleife).

Öffentliche Rechtsquellen (keine Mandantendaten) – der Abruf ist ein erlaubter
Online-Schritt. Das Matching gegen Katalog/Definitionen läuft anschließend lokal.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

_log = logging.getLogger(__name__)

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
    # strippen → unescapen → NOCHMAL strippen: Feeds liefern HTML oft escaped
    # ("&lt;p&gt;…"); nach dem Unescape stünden sonst literale Tags im Klartext.
    # Der zweite Strip ist idempotent (auf tagfreiem Text ein No-Op).
    return _TAG.sub(" ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def parse_rss(raw: bytes | str, quelle: str) -> list[RssEintrag]:
    # Defense-in-depth gegen Entity-Expansion (billion laughs): legitimes RSS hat
    # keine DTD/Entities. Ein Feed, der welche im Prolog mitbringt, wird abgelehnt,
    # statt von expat expandiert zu werden (unabhaengig von der libexpat-Version).
    # BEWUSSTE GRENZE: geprüft werden nur die ersten 64 KiB – DOCTYPE/ENTITY sind
    # Prolog-Konstrukte und stehen am Dokumentanfang; ein erst später deklariertes
    # Entity fangen darunter die eingebauten expat-Limits ab. Kein Vollscan nötig.
    head = (raw[:65536] if isinstance(raw, bytes) else raw[:65536].encode("utf-8", "replace")).lower()
    if b"<!doctype" in head or b"<!entity" in head:
        raise ValueError("RSS mit DTD/Entity abgelehnt (Entity-Expansion-Schutz)")
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
        roh = r.read(_MAX_FEED)
        # Exakt am Cap UND es kommen noch Daten: kein stilles Kappen (das endete
        # als kryptischer ParseError = stiller Ausfall), sondern ein expliziter
        # Fehler – der landet sichtbar im Fehlerkanal von hole_alle_mit_fehlern.
        if len(roh) == _MAX_FEED and r.read(1):
            raise ValueError(f"Feed {quelle} größer als {_MAX_FEED} Bytes – abgelehnt")
    return parse_rss(roh, quelle)


@dataclass(frozen=True)
class FeedErgebnis:
    """Sammelergebnis über alle Feeds (Fehlerkanal wie corpus.FetchErgebnis)."""

    eintraege: list[RssEintrag]
    fehler: dict[str, str]    # Quelle ("BFH"/"BMF") → "ExcType: Meldung"


def hole_alle_mit_fehlern(*, timeout: float = 25.0) -> FeedErgebnis:
    """Holt alle Feeds; ein Feed darf ausfallen, aber NICHT mehr unsichtbar.

    Vorher verschluckte ``hole_alle`` jeden Feed-Fehler kommentarlos – ein toter
    BFH-Feed sah aus wie „keine neuen Entscheidungen". Jetzt: Warnung im Log und
    strukturierter Fehlerkanal für Aufrufer/CLI (Muster: corpus.hole_alle).
    """
    eintraege: list[RssEintrag] = []
    fehler: dict[str, str] = {}
    for url, quelle in ((BFH_URL, "BFH"), (BMF_URL, "BMF")):
        try:
            eintraege.extend(hole_feed(url, quelle, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 – ein Feed darf ausfallen, sichtbar
            fehler[quelle] = f"{type(exc).__name__}: {exc}"
            _log.warning("RSS-Feed %s (%s) fehlgeschlagen: %s", quelle, url, fehler[quelle])
    return FeedErgebnis(eintraege=eintraege, fehler=fehler)


def hole_alle(*, timeout: float = 25.0) -> list[RssEintrag]:
    """Abwärtskompatible Fassade (abgleich.laufe erwartet eine Liste).

    Fehler werden geloggt; wer sie strukturiert braucht, nutzt
    ``hole_alle_mit_fehlern``.
    """
    return hole_alle_mit_fehlern(timeout=timeout).eintraege
