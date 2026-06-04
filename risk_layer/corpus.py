"""Normtext-Korpus: lokaler XML-Store + optionaler gii-Fetcher.

Trennung von Laufzeit und Build (/NF10/, /NF60/):
  * Zur LAUFZEIT wird ausschließlich aus dem lokalen XML-Store gelesen – kein
    Netzaufruf. Der gebaute Normgraph liegt lokal.
  * Das BEFÜLLEN des Stores (dieser Fetcher) ist ein expliziter, vom Nutzer
    ausgelöster Wartungsschritt, der einmalig online geht.

Quelle: gesetze-im-internet.de stellt je Gesetz ein ``<slug>/xml.zip`` bereit
(DTD-basiertes gii-Format). „Welche Gesetze" wird aus den ``norm_anker`` des
Katalogs abgeleitet und in ``corpus/gesetze.yaml`` als editierbare Config abgelegt.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from .catalog import Katalog
from .zitate import parse_normzitat

GII_BASIS = "https://www.gesetze-im-internet.de"
DEFAULT_KORPUS = Path("corpus")

# LAW-Code → gii-Slug (Sonderfälle; Default ist die Kleinschreibung des Codes).
GII_SLUG = {
    "AO": "ao_1977",
    "UStG": "ustg_1980",
    "KStG": "kstg_1977",
    "GewStG": "gewstg",
    "GewStDV": "gewstdv_1955",
    "EStG": "estg",
    "EStDV": "estdv_1955",
    "LStDV": "lstdv",
    "AStG": "astg",
    "AktG": "aktg",
    "GmbHG": "gmbhg",
    "HGB": "hgb",
    "UmwStG": "umwstg_2006",
    "UmwG": "umwg_1995",
    "BewG": "bewg",
    "ErbStG": "erbstg_1974",
    "GrEStG": "grestg_1983",
    "GG": "gg",
    "FGO": "fgo",
    "StBerG": "stberg",
    "BGB": "bgb",
    "UStDV": "ustdv_1980",
    "KStDV": "kstdv_1977",
    "ErbStDV": "erbstdv_1998",
    "GrStG": "grstg_1973",
    "SolzG": "solzg_1995",
    "InvStG": "invstg_2018",
    "FVG": "fvg",
    # Kein gii-Download (EU-/Verfassungsrecht in eigener Quelle):
    "MwStSystRL": None,
    "AEUV": None,
    "EUV": None,
}

# Breiter Steuer-Korpus: relevante Gesetze einer Steuerkanzlei (Mehrgesetz-Sachverhalte).
STEUER_KORPUS: tuple[str, ...] = (
    "AO", "FGO", "EStG", "EStDV", "LStDV", "KStG", "KStDV", "GewStG", "GewStDV",
    "UStG", "UStDV", "AStG", "UmwStG", "UmwG", "BewG", "ErbStG", "ErbStDV", "GrEStG",
    "GrStG", "SolzG", "InvStG", "HGB", "AktG", "GmbHG", "StBerG", "GG",
)

# Authority-Klasse: welche Korpus-Gesetze Rechtsverordnungen sind. gii
# klassifiziert Gesetz/Verordnung nicht zuverlässig in EINEM Feld, daher eine
# kuratierte Liste als Quelle der Wahrheit; gesetze.yaml 'art' kann überschreiben.
VERORDNUNGEN: frozenset[str] = frozenset(
    {"EStDV", "UStDV", "LStDV", "KStDV", "GewStDV", "ErbStDV"}
)


def art_fuer(law: str) -> str:
    """Authority-Klasse eines LAW-Codes: 'verordnung' oder (Default) 'gesetz'."""
    return "verordnung" if law in VERORDNUNGEN else "gesetz"


def source_rank(art: str) -> str:
    """Abgeleiteter Rang fürs Discovery-Output (keine doppelte Wahrheit in der Config)."""
    return "statutory_regulation" if art == "verordnung" else "statute"


def steuer_korpus() -> list["Gesetz"]:
    """Die kuratierte Liste relevanter Steuergesetze (für den vollen Graphen)."""
    aus: list[Gesetz] = []
    for law in STEUER_KORPUS:
        slug = slug_fuer(law)
        if slug:
            aus.append(Gesetz(law=law, slug=slug, url=gii_url(slug), art=art_fuer(law)))
    return aus


def slug_fuer(law: str) -> str | None:
    if law in GII_SLUG:
        return GII_SLUG[law]
    return law.lower()


def gii_url(slug: str) -> str:
    return f"{GII_BASIS}/{slug}/xml.zip"


@dataclass(frozen=True)
class Gesetz:
    law: str
    slug: str
    url: str
    art: str = "gesetz"   # Authority-Klasse: gesetz | verordnung


def ableiten_aus_katalog(katalog: Katalog) -> list[Gesetz]:
    """Leitet die zu pflegenden Gesetze aus den Norm-Ankern des Katalogs ab."""
    laws: list[str] = []
    for eintrag in katalog:
        for zitat in eintrag.norm_anker:
            for anker in parse_normzitat(zitat):
                if anker.norm_id:
                    laws.append(anker.norm_id.law)
    gesehen: dict[str, Gesetz] = {}
    for law in laws:
        if law in gesehen:
            continue
        slug = slug_fuer(law)
        if not slug:
            continue
        gesehen[law] = Gesetz(law=law, slug=slug, url=gii_url(slug), art=art_fuer(law))
    return list(gesehen.values())


def config_pfad(korpus: Path = DEFAULT_KORPUS) -> Path:
    return Path(korpus) / "gesetze.yaml"


def schreibe_config(gesetze: list[Gesetz], korpus: Path = DEFAULT_KORPUS) -> Path:
    pfad = config_pfad(korpus)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in gesetze:
        row = {"law": g.law, "slug": g.slug, "url": g.url}
        if g.art != "gesetz":          # nur die Abweichung (Verordnung) markieren
            row["art"] = g.art
        rows.append(row)
    pfad.write_text(
        "# Normtext-Korpus – editierbare Gesetzesliste (aus dem Katalog abgeleitet).\n"
        "# 'slug' ist die gesetze-im-internet-Kennung; eigene Gesetze einfach ergänzen.\n"
        + yaml.safe_dump(rows, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return pfad


def lade_config(korpus: Path = DEFAULT_KORPUS) -> list[Gesetz]:
    pfad = config_pfad(korpus)
    if not pfad.exists():
        return []
    rows = yaml.safe_load(pfad.read_text(encoding="utf-8")) or []
    return [Gesetz(law=r["law"], slug=r["slug"], url=r.get("url") or gii_url(r["slug"]),
                   art=(r.get("art") or art_fuer(r["law"]))) for r in rows]


def xml_verzeichnis(korpus: Path = DEFAULT_KORPUS) -> Path:
    return Path(korpus) / "gii"


def hole_gesetz(g: Gesetz, korpus: Path = DEFAULT_KORPUS, *, timeout: float = 30.0) -> Path:
    """Lädt ein einzelnes gii-Gesetz (xml.zip) und entpackt es in den Store.

    NETZZUGRIFF – nur im Build-Schritt aufzurufen. Raises bei Fehlern.
    """
    # g.law stammt aus der editierbaren gesetze.yaml und bildet den Dateinamen ->
    # Pfad-Traversal verhindern, BEVOR ueberhaupt etwas geladen/geschrieben wird.
    ziel = xml_verzeichnis(korpus)
    pfad = ziel / f"{g.law}.xml"
    if pfad.resolve().parent != ziel.resolve():
        raise ValueError(f"unzulässiger law-Code (Pfad-Traversal): {g.law!r}")
    req = urllib.request.Request(g.url, headers={"User-Agent": "risk-layer/0.1 (lokal)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (feste, vom Nutzer bestätigte Quelle)
        roh = resp.read()
    with zipfile.ZipFile(io.BytesIO(roh)) as zf:
        xml_namen = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_namen:
            raise ValueError(f"{g.law}: kein XML im Archiv {g.url}")
        daten = zf.read(xml_namen[0])
    ziel.mkdir(parents=True, exist_ok=True)
    pfad.write_bytes(daten)
    return pfad


@dataclass
class FetchErgebnis:
    geholt: list[str]
    fehler: dict[str, str]


def hole_alle(gesetze: list[Gesetz], korpus: Path = DEFAULT_KORPUS, *, timeout: float = 30.0) -> FetchErgebnis:
    geholt: list[str] = []
    fehler: dict[str, str] = {}
    for g in gesetze:
        try:
            hole_gesetz(g, korpus, timeout=timeout)
            geholt.append(g.law)
        except Exception as exc:  # noqa: BLE001 – Sammelreport, einzelnes Gesetz darf scheitern
            fehler[g.law] = f"{type(exc).__name__}: {exc}"
    return FetchErgebnis(geholt=geholt, fehler=fehler)
