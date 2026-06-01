"""Schicht-2-Sicherheitsvertrag /LF120/ /AB30/ – ohne GPU prüfbar.

Span-Treue = 100 %: Jede ausgegebene Spanne existiert wörtlich im Quelltext.
Halluzinierte Spannen werden verworfen.
"""

from risk_layer.schicht2 import (
    Kandidat,
    NullProvider,
    erkenne_kandidaten,
    verifiziere_kandidaten,
    verifiziere_span,
)


TEXT = "Die Angemessenheit der Vergütung ist anhand eines Fremdvergleichs zu prüfen."


def test_verifiziere_span_positiv():
    off = TEXT.index("Angemessenheit")
    assert verifiziere_span(TEXT, "Angemessenheit", off)


def test_verifiziere_span_falscher_offset():
    assert not verifiziere_span(TEXT, "Angemessenheit", 0)


def test_verifiziere_span_nicht_im_text():
    assert not verifiziere_span(TEXT, "Liebhaberei", 5)


def test_ab30_nur_woertliche_spannen_ueberleben():
    off = TEXT.index("Fremdvergleichs")
    roh = [
        {"span_text": "Fremdvergleichs", "char_offset": off,
         "begruendung_kurz": "wertungsoffen", "konfidenz": "hoch"},          # gültig
        {"span_text": "Angemessenheit", "char_offset": 0,
         "begruendung_kurz": "falscher offset", "konfidenz": "mittel"},      # falscher Offset
        {"span_text": "verdeckte Einlage", "char_offset": 4,
         "begruendung_kurz": "halluziniert", "konfidenz": "hoch"},           # nicht im Text
    ]
    verifiziert = verifiziere_kandidaten(TEXT, roh)
    assert len(verifiziert) == 1
    assert verifiziert[0].span_text == "Fremdvergleichs"
    # alle überlebenden Spannen stehen wörtlich im Text
    for k in verifiziert:
        assert TEXT[k.char_offset:k.char_offset + len(k.span_text)] == k.span_text


def test_ab30_span_treue_ueber_gold_set():
    """100 % Span-Treue: jeder akzeptierte Kandidat ist wörtlich belegt."""
    import random
    rng = random.Random(42)
    woerter = TEXT.split()
    roh = []
    for w in woerter:
        off = TEXT.index(w)
        roh.append({"span_text": w, "char_offset": off,
                    "begruendung_kurz": "", "konfidenz": "niedrig"})
        # halluzinierter Müll mit zufälligem Offset
        roh.append({"span_text": w + "XYZ", "char_offset": rng.randint(0, len(TEXT)),
                    "begruendung_kurz": "", "konfidenz": "niedrig"})
    verifiziert = verifiziere_kandidaten(TEXT, roh)
    assert verifiziert  # einige echte überleben
    for k in verifiziert:
        assert TEXT[k.char_offset:k.char_offset + len(k.span_text)] == k.span_text


def test_nullprovider_liefert_nichts():
    assert erkenne_kandidaten(TEXT, bekannte_begriffe=[], provider=NullProvider()) == []
    # Default-Provider ist der NullProvider
    assert erkenne_kandidaten(TEXT, bekannte_begriffe=[]) == []
