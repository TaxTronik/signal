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


def test_word_joiner_wird_geloescht_nicht_sichtbar():
    # U+2060 ist zero-width – ein Mapping auf ein sichtbares Leerzeichen
    # zerschnitt das Wort und kostete den Katalogtreffer.
    text = "Ein⁠gliederung"
    res = ing.ingest(text)
    assert res.normalized_text == "Eingliederung"
    assert len(res.offset_map) == len(res.normalized_text)
    # das 'g' nach dem Word Joiner mappt exakt auf seine Originalposition
    idx = res.normalized_text.index("gliederung")
    assert text[res.offset_map[idx]] == "g"


def test_soft_hyphen_am_zeilenumbruch_fuegt_zusammen():
    # klassisches PDF-Artefakt: weiches Trennzeichen + Umbruch (+ Einrückung)
    text = "Einglie­\nderung"
    res = ing.ingest(text)
    assert res.normalized_text == "Eingliederung"
    idx = res.normalized_text.index("derung")
    assert text[res.offset_map[idx]] == "d"


def test_soft_hyphen_mit_crlf_und_einrueckung():
    text = "Einglie­\r\n  derung besteht"
    res = ing.ingest(text)
    assert res.normalized_text == "Eingliederung besteht"


def test_soft_hyphen_ohne_buchstaben_nur_geloescht():
    # rechts kein Buchstabe → keine Fuge; der Soft Hyphen verschwindet,
    # der Zeilenumbruch bleibt erhalten
    text = "Tel.­\n123"
    res = ing.ingest(text)
    assert res.normalized_text == "Tel.\n123"


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


def test_segmentierung_trennt_an_leerzeilen_nicht_an_jedem_umbruch():
    # Absatz-Semantik (Docstring): einfacher Zeilenumbruch INNERHALB eines
    # Absatzes trennt nicht – nur die Leerzeile.
    text = "Zeile eins\nzeile zwei desselben Absatzes.\n\nZweiter Absatz."
    res = ing.ingest(text)
    assert len(res.segments) == 2
    assert "zeile zwei" in res.segments[0].text
    for seg in res.segments:
        assert text[seg.start:seg.end] == seg.text


def test_segmentierung_leerzeile_mit_spaces_trennt():
    text = "Erster Absatz.\n   \nZweiter Absatz."
    res = ing.ingest(text)
    assert [s.text for s in res.segments] == ["Erster Absatz.", "Zweiter Absatz."]
