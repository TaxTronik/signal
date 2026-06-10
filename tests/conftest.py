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


@pytest.fixture(autouse=True)
def _kein_ner(monkeypatch):
    """Kein Test lädt implizit das echte spaCy-Modell (optionale NER-Schicht).

    anonymize.anonymisiere ruft über den Lade-Seam ``_lade_nlp`` (Modul-Cache
    ``_NLP``) lazy das deutsche spaCy-Modell – und steckt via recherche.pruefe_rest
    im § 203-Gate von /v1/katalog/definiere und der CLI. tests/test_api_v1.py,
    test_web.py und test_cli.py liefen auf Dev-Maschinen MIT installiertem spaCy
    sonst gegen das echte Modell (langsam, umgebungsabhängiges Testverhalten).
    Muster wie ``_kein_spacy`` in tests/test_anonymize.py: Loader auf None
    patchen + Env aufräumen. NER-Tests injizieren ihren Fake über denselben
    Seam – test-lokale monkeypatches überschreiben diese Fixture.
    """
    monkeypatch.setattr("risk_layer.anonymize._lade_nlp", lambda: None)
    monkeypatch.delenv("RISK_LAYER_NER", raising=False)
