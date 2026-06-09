"""Report-Schicht: HTML-Escaping von Mandantentext + Annotations-Korrektheit.

Kein Netz, kein LLM: Reports entstehen aus der deterministischen Pipeline
(Schicht 1) bzw. aus synthetischen Risikomarkierungen.
"""

import html as html_mod

from risk_layer.catalog import lade_katalog
from risk_layer.models import Risikomarkierung, RunReport, Status
from risk_layer.pipeline import Pipeline
from risk_layer.report import _annotiere, report_html, report_text


def _lauf(text: str) -> RunReport:
    return Pipeline(lade_katalog(None)).run(text, erzeugt_am="2026-01-01T00:00:00")


def _mini_report(risiken: list[Risikomarkierung]) -> RunReport:
    return RunReport(text_hash="0" * 64, katalog_version="test",
                     erzeugt_am="2026-01-01T00:00:00", risiken=risiken)


# -- HTML-Escaping (Mandantentext darf nie als Markup enden) ------------------

def test_report_html_escapt_mandantentext():
    boese = "<script>alert('Mandant & Co')</script>"
    text = f"Es droht die Schätzung der Besteuerungsgrundlagen. {boese}"
    seite = report_html(_lauf(text), text)
    assert "<script>alert" not in seite                  # kein injizierbares Markup
    assert "&lt;script&gt;alert" in seite                # escaped sichtbar


def test_report_html_escapt_titel():
    text = "Es droht die Schätzung der Besteuerungsgrundlagen."
    seite = report_html(_lauf(text), text, titel="<img src=x onerror=alert(1)>")
    assert "<img src=x" not in seite
    assert "&lt;img" in seite


def test_report_html_traegt_meta_und_karten():
    text = "Es droht die Schätzung der Besteuerungsgrundlagen."
    report = _lauf(text)
    seite = report_html(report, text)
    assert report.text_hash[:16] in seite
    assert "Schätzung der Besteuerungsgrundlagen" in seite
    assert "id='karte-0'" in seite                       # Karte + Anker vorhanden


# -- Annotations-Korrektheit (_annotiere) -------------------------------------

def test_annotiere_markiert_exakt_die_treffer_spans():
    text = "Es droht die Schätzung der Besteuerungsgrundlagen im Betrieb."
    report = _lauf(text)
    assert report.karten, "Fixture-Text muss einen Katalog-Treffer liefern"
    annot = _annotiere(text, report.karten, report.risiken)
    k = report.karten[0]
    # der markierte Text steht escaped in einem <a>-Anker auf die Karte
    assert f">{html_mod.escape(k.treffer.matched_text)}</a>" in annot
    assert "href='#karte-0'" in annot
    # Text außerhalb der Spans bleibt unmarkiert erhalten
    assert "Es droht die " in annot


def test_annotiere_ohne_eintraege_gibt_escapten_text():
    annot = _annotiere("Nur <b>Text</b> ohne Treffer.", [], [])
    assert annot == "<p class='kein-treffer'>Nur &lt;b&gt;Text&lt;/b&gt; ohne Treffer.</p>"


def test_annotiere_ueberlappung_deterministisch_und_vollstaendig():
    """Mini-Fall mit Überlappung: Abschnitte werden überlappungssicher zerlegt,
    Titel-Reihenfolge folgt der Eintrags-Reihenfolge (Determinismus), und die
    Konkatenation aller Chunks ergibt wieder den ganzen (escapten) Text."""
    text = "abcdefghij"
    risiken = [
        Risikomarkierung(start=2, end=8, matched_text="cdefgh",
                         status=Status.UNKNOWN_RISIKO, titel="Aussen"),
        Risikomarkierung(start=4, end=6, matched_text="ef",
                         status=Status.UNKNOWN_RISIKO, titel="Innen"),
    ]
    annot = _annotiere(text, [], risiken)
    # Überlappungsbereich e–f trägt BEIDE Titel in stabiler Reihenfolge
    assert "Aussen [Unknown-Risiko] | Innen [Unknown-Risiko]" in annot
    # zweimal identischer Lauf → identisches Ergebnis (Determinismus)
    assert annot == _annotiere(text, [], risiken)
    # alle Zeichen des Originaltexts bleiben in Reihenfolge erhalten
    import re
    assert "".join(re.findall(r">([^<>]*)<", f">{annot}<")).replace("</a>", "") \
        .startswith("ab")
    nur_text = re.sub(r"<[^>]+>", "", annot)
    assert nur_text == text


def test_annotiere_rand_spans_anfang_und_ende():
    """Spans an Textanfang/-ende (Grenzen 0 und len) werden korrekt behandelt."""
    text = "Start Mitte Ende"
    risiken = [
        Risikomarkierung(start=0, end=5, matched_text="Start",
                         status=Status.UNKNOWN_RISIKO, titel="Anfang"),
        Risikomarkierung(start=12, end=16, matched_text="Ende",
                         status=Status.UNKNOWN_RISIKO, titel="Schluss"),
    ]
    annot = _annotiere(text, [], risiken)
    import re
    assert re.sub(r"<[^>]+>", "", annot) == text
    assert ">Start</a>" in annot and ">Ende</a>" in annot
    assert " Mitte " in annot                            # Zwischenraum unmarkiert


# -- report_text (Konsole) -----------------------------------------------------

def test_report_text_zusammenfassung_und_marker():
    text = "Es droht die Schätzung der Besteuerungsgrundlagen."
    out = report_text(_lauf(text))
    assert "Risk-Layer" in out and "Markierungen" in out
    assert "Schätzung der Besteuerungsgrundlagen" in out


def test_report_text_risikomarkierung_zeile():
    out = report_text(_mini_report([
        Risikomarkierung(start=0, end=4, matched_text="Wort",
                         status=Status.UNKNOWN_RISIKO, titel="Testrisiko", score=0.5),
    ]))
    assert "Risikomarkierungen" in out and "Testrisiko" in out and "UNKNOWN" in out
