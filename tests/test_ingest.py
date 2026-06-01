"""Tests für den Ingest (/LF10/–/LF30/): offset-erhaltende Normalisierung."""

from risk_layer import ingest as ing


def test_offset_map_laenge_passt():
    text = "Die organisatorische Eingliederung besteht."
    res = ing.ingest(text)
    assert len(res.offset_map) == len(res.normalized_text)
    # jeder normalisierte Index zeigt auf ein gültiges Originalzeichen
    for i, src in enumerate(res.offset_map):
        assert 0 <= src < len(text)


def test_offsets_mappen_zurueck_auf_original():
    text = "finanzielle Eingliederung"
    res = ing.ingest(text)
    # Identitätstext: normalisiert == original, Map ist die Identität
    assert res.normalized_text == text
    assert res.offset_map == list(range(len(text)))


def test_zeilenumbruch_trennung_wird_zusammengefuegt():
    text = "organisato-\nrische Eingliederung"
    res = ing.ingest(text)
    assert "organisatorische" in res.normalized_text
    # das 'r' nach der Trennung mappt auf seine Originalposition
    idx = res.normalized_text.index("rische")
    assert text[res.offset_map[idx]] == "r"


def test_ligatur_wird_expandiert():
    text = "Pﬂicht zur Prüfung"  # 'ﬂ'-Ligatur
    res = ing.ingest(text)
    assert "Pflicht" in res.normalized_text


def test_weiches_trennzeichen_entfernt():
    text = "Ein­glie­derung"
    res = ing.ingest(text)
    assert res.normalized_text == "Eingliederung"


def test_hash_stabil_und_eingabeabhaengig():
    a = ing.ingest("Text A")
    b = ing.ingest("Text A")
    c = ing.ingest("Text B")
    assert a.text_hash == b.text_hash
    assert a.text_hash != c.text_hash


def test_segmente_haben_originaloffsets():
    text = "Erster Absatz.\n\nZweiter Absatz."
    res = ing.ingest(text)
    assert len(res.segments) == 2
    for seg in res.segments:
        assert text[seg.start:seg.end].strip() == seg.text.strip()
