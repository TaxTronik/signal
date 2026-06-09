"""Projektweite Test-Fixtures.

Die autouse-Fixture ``_kein_embedding`` lag früher nur in test_web.py; auch
test_api_v1.py und test_web_surface.py instanziieren aber ``Anwendung`` und
würden auf Entwickler-Maschinen mit gebautem ``corpus/embedding/`` das echte
BGE-M3-Modell (~2,3 GB, Default-Device cuda) laden — umgebungsabhängiges
Testverhalten. Deshalb projektweit: kein Test lädt den echten Index.

Gepatcht wird der Lade-Seam ``embedding.index_oeffnen`` (liefert None = „kein
Index gebaut"), NICHT ``EmbeddingIndex.load`` selbst: test_embedding.py prüft
load/build mit dem HashEncoder direkt und bleibt unberührt. Tests, die einen
Index brauchen, patchen ``index_oeffnen`` lokal (siehe test_web_surface.py) —
test-lokale monkeypatches überschreiben diese Fixture.
"""

import pytest


@pytest.fixture(autouse=True)
def _kein_embedding(monkeypatch):
    monkeypatch.setattr("risk_layer.embedding.index_oeffnen",
                        lambda *a, **k: None)
