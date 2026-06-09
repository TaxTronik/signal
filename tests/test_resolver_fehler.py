"""resolver: unbekannte Begriffs-ID ⇒ KatalogFehler mit Kontext (kein nackter KeyError)."""

import pytest

from risk_layer.catalog import KatalogFehler, lade_katalog
from risk_layer.models import Treffer
from risk_layer.resolver import resolve


def _treffer(begriff_id: str) -> Treffer:
    return Treffer(begriff_id=begriff_id, start=0, end=4, matched_text="Test",
                   surface="Test", segment_index=0)


def test_unbekannte_begriffs_id_wirft_katalogfehler_mit_kontext():
    with pytest.raises(KatalogFehler, match="gibt_es_nicht"):
        resolve(_treffer("gibt_es_nicht"), lade_katalog())


def test_bekannte_id_bleibt_unveraendert():
    kat = lade_katalog()
    eintrag = next(iter(kat))
    karte = resolve(_treffer(eintrag.id), kat)
    assert karte.eintrag.id == eintrag.id
