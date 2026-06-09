"""Law-Alias-Map (Parser-Spez. 2.4).

Im Vollbetrieb wird die Map beim gii-Ingest automatisch aus den `<amtabk>`/
`<jurabk>`-Feldern jeder Gesetzesdatei geharvestet. Ohne Korpus liefert dieses
Modul eine gepflegte Seed-Map für die im Katalog/Artikel vorkommenden Gesetze
plus die Sonderliste (MwStSystRL, GG, EU-Recht) und einen Harvest-Hook, der eine
geharvestete Map additiv einspielt.

Beim Parsen gilt Longest-Match (z. B. „GewStG" vor „StG").
"""

from __future__ import annotations

# Seed-Aliase → kanonische LAW-Abkürzung. Schlüssel werden case-sensitiv per
# Longest-Match gegen das Zitatende geprüft.
SEED_ALIASE: dict[str, str] = {
    "UStG": "UStG",
    "KStG": "KStG",
    "GewStG": "GewStG",
    "EStG": "EStG",
    "AStG": "AStG",
    "AktG": "AktG",
    "GmbHG": "GmbHG",
    "HGB": "HGB",
    "AO 1977": "AO",
    "AO": "AO",
    "UmwStG": "UmwStG",
    "UmwG": "UmwG",
    "BewG": "BewG",
    "ErbStG": "ErbStG",
    "GrEStG": "GrEStG",
    "SolzG": "SolzG",
    "FGO": "FGO",
    # Durchführungsverordnungen + übrige Korpus-Gesetze. Longest-Match trennt die
    # DV vom Stammgesetz (z. B. "GewStDV" vor "GewStG"). Diese Liste MUSS den
    # Korpus (corpus/gesetze.yaml) abdecken — sonst lösen Zitate auf diese Gesetze
    # nicht auf (LAW=?); test_zitate.test_seed_deckt_korpus sichert das ab.
    "EStDV": "EStDV",
    "LStDV": "LStDV",
    "KStDV": "KStDV",
    "GewStDV": "GewStDV",
    "UStDV": "UStDV",
    "ErbStDV": "ErbStDV",
    "GrStG": "GrStG",
    "InvStG": "InvStG",
    "StBerG": "StBerG",
    # Sonderliste (keine gii-amtabk): Art.-zitierte / EU- / Verfassungsrecht
    "GG": "GG",
    "MwStSystRL": "MwStSystRL",
    "AEUV": "AEUV",
    "EUV": "EUV",
}


class LawMap:
    """Alias→LAW-Auflösung mit Longest-Match."""

    def __init__(self, aliase: dict[str, str] | None = None):
        self._aliase: dict[str, str] = dict(SEED_ALIASE)
        if aliase:
            self._aliase.update(aliase)
        # längste Aliase zuerst (Longest-Match)
        self._sortiert = sorted(self._aliase, key=len, reverse=True)

    def merge_harvest(self, geharvestet: dict[str, str]) -> "LawMap":
        """Spielt eine aus dem Korpus geharvestete Alias-Map ADDITIV ein.

        Additiv heißt: geharvestete Aliase ergänzen nur – bei Schlüssel-Kollision
        gewinnen die kuratierten Seeds/bestehenden Aliase, der Harvest
        überschreibt sie nicht.
        """
        return LawMap({**geharvestet, **self._aliase})

    def alle_aliase(self) -> list[str]:
        return list(self._sortiert)

    def resolve(self, alias: str) -> str | None:
        return self._aliase.get(alias)

    def match_am_ende(self, text: str) -> tuple[str, str] | None:
        """Findet die längste Gesetzes-Abkürzung am Ende von ``text``.

        Returns (LAW, alias) oder None. Achtet auf eine Wortgrenze links.
        """
        for alias in self._sortiert:
            if text.endswith(alias):
                vor = text[: len(text) - len(alias)]
                if vor == "" or not vor[-1].isalnum():
                    return self._aliase[alias], alias
        return None


DEFAULT_LAWMAP = LawMap()
