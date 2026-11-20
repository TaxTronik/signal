"""Semantic suggestions remain visible, source-based and distinct from markings."""

from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

from risk_layer.models import Risikomarkierung, RunReport, Status
from risk_layer.report import report_html, report_text


TEXT = "Die erste Aussage beschreibt die Person. Die zweite beschreibt den Vorgang."
SOURCE = "Die Aufzeichnungen müssen vollständig sein.\nDie zweite Quellenzeile bleibt erhalten."


def norm():
    return {
        "id": "norm:AO:146", "zitat": "§ 146 AO", "titel": "Ordnungsvorschriften",
        "gestuetzt": True, "im_korpus": True, "sim": .73, "rerank": .88,
        "ketten": [{"anzeige": "§ 146 AO iVm § 147 AO"}],
        "evidenz": {
            "fingerprint": "f" * 64, "gekuerzt": True, "anker": ["Buchführung als Kataloganker"],
            "passagen": [{"id": "norm:AO:146@4:900", "norm_id": "norm:AO:146",
                          "quell_start": 4, "quell_end": 4 + len(SOURCE), "text": SOURCE}],
        },
    }


def suggestions():
    split = TEXT.index("Die zweite")
    return [{"start": start, "end": end, "satz": TEXT[start:end], "profil": "dokument",
             "kontext": {"start": 0, "end": len(TEXT), "text": TEXT}, "normen": [norm()]}
            for start, end in [(0, split - 1), (split, len(TEXT))]]


def report(items=None):
    return RunReport(text_hash="0" * 64, katalog_version="test", erzeugt_am="2026-01-01T00:00:00",
                     semantik_vorschlaege=items or [])


def render(kind, items):
    r = report(items)
    return report_text(r) if kind == "text" else report_html(r, TEXT)


@pytest.mark.parametrize("kind", ["text", "html"])
def test_context_list_appears_once_with_both_focus_references_and_original_source(kind):
    items = suggestions()
    before = deepcopy(items)
    out = render(kind, items)
    assert out.count("Vorschlag: § 146 AO") == 1
    assert "Semantische Normvorschläge" in out
    assert "dokument" in out and "Normquelle" in out and "§ 146 AO" in out
    assert "norm:AO:146" not in out
    assert "Quelloffset 4–" + str(4 + len(SOURCE)) in out
    assert "Die Aufzeichnungen müssen vollständig sein." in out
    assert "Die zweite Quellenzeile bleibt erhalten." in out
    assert "f" * 64 not in out and "Gekürzte Auswahl" in out
    assert "Kataloganker" in out and "§ 146 AO iVm § 147 AO" in out
    for item in items:
        assert f"@{item['start']}–{item['end']}" in out and item["satz"] in out
    assert "Wahrscheinlichkeit" not in out and "88%" not in out and "0.88" not in out
    assert items == before
    assert report(items).zusammenfassung()["markierungen_gesamt"] == 0


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.ids = []
        self.links = []
        self.details = []
        self.detail_depth = 0
        self.detail_text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        if tag == "details":
            self.details.append(attrs)
            self.detail_depth += 1
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if "href" in attrs:
            self.links.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "details":
            self.detail_depth -= 1

    def handle_data(self, data):
        if self.detail_depth:
            self.detail_text.append(data)


def test_html_reference_links_resolve_and_do_not_create_document_risk_spans():
    out = render("html", suggestions())
    structure = Structure()
    structure.feed(out)
    assert structure.links == ["#semantik-0-bezug-1", "#semantik-0-bezug-2"]
    assert all(link[1:] in structure.ids for link in structure.links)
    assert len(structure.ids) == len(set(structure.ids))
    assert "class='doc'><p class='kein-treffer'>" + TEXT in out
    assert "href='#karte-" not in out and "href='#risiko-" not in out
    assert out.count("class='disclaimer'") == 1  # only the existing report banner


def test_html_source_excerpts_and_anchors_are_in_initially_closed_native_details():
    out = render("html", suggestions())
    structure = Structure()
    structure.feed(out)
    assert structure.details == [{}]  # one deduplicated norm, no open attribute
    assert "<summary>Normquelle und Belege</summary>" in out
    detail_text = "".join(structure.detail_text)
    assert SOURCE in detail_text and "Buchführung als Kataloganker" in detail_text
    assert "Quelloffset 4–" in detail_text
    assert "Die erste Aussage" not in detail_text


@pytest.mark.parametrize("kind", ["text", "html"])
def test_duplicate_focus_does_not_repeat_norms_or_references(kind):
    item = suggestions()[0]
    out = render(kind, [item, deepcopy(item)])
    assert out.count("Vorschlag: § 146 AO") == 1
    if kind == "text":
        assert out.count("Textstelle 1 @") == 1 and "Textstelle 2 @" not in out
    else:
        assert out.count("id='semantik-0-bezug-1'") == 1 and "semantik-0-bezug-2" not in out


@pytest.mark.parametrize("kind", ["text", "html"])
def test_distinct_source_evidence_for_same_norm_is_preserved_with_its_focus(kind):
    items = suggestions()
    items[1]["normen"][0]["evidenz"]["passagen"][0]["text"] = "Eine andere Quellenauswahl."
    out = render(kind, items)
    assert out.count("Vorschlag: § 146 AO") == 2
    assert "Die Aufzeichnungen müssen vollständig sein." in out
    assert "Eine andere Quellenauswahl." in out


@pytest.mark.parametrize("kind", ["text", "html"])
def test_equal_context_text_at_different_positions_or_profiles_stays_separate(kind):
    items = suggestions()
    moved = deepcopy(items[0])
    moved["start"] += 100
    moved["end"] += 100
    moved["kontext"]["start"] += 100
    moved["kontext"]["end"] += 100
    other_profile = {**deepcopy(items[0]), "profil": "kontext"}
    out = render(kind, [*items, moved, other_profile])
    assert out.count("Vorschlag: § 146 AO") == 3


@pytest.mark.parametrize("kind", ["text", "html"])
def test_different_norm_lists_keep_their_own_focus_references(kind):
    items = suggestions()
    items[1]["normen"] = [{"id": "norm:AO:147", "zitat": "§ 147 AO", "titel": "Aufbewahrung"}]
    out = render(kind, items)
    first = out.split("Vorschlag: § 146 AO", 1)[1].split("Vorschlag: § 147 AO", 1)[0]
    second = out.split("Vorschlag: § 147 AO", 1)[1]
    if kind == "text":
        assert "Bezug: Textstellen 1\n" in first and "Bezug: Textstellen 2\n" in second
    else:
        assert "href='#semantik-0-bezug-1'" in first and "href='#semantik-0-bezug-2'" not in first
        assert "href='#semantik-0-bezug-2'" in second and "href='#semantik-0-bezug-1'" not in second


@pytest.mark.parametrize("kind", ["text", "html"])
def test_legacy_suggestions_are_visible_without_fabricated_source_excerpts(kind):
    items = [{"start": 0, "end": len(TEXT), "satz": TEXT,
              "normen": [{"id": "norm:AO:146", "zitat": "§ 146 AO", "titel": ""}]}]
    out = render(kind, items)
    assert "standard" in out and "Bezugssatz" in out and "Vorschlag: § 146 AO" in out
    assert "Nicht im Analyseergebnis mitgeliefert." in out
    assert "Originalnormpassage" not in out and "Quellenstand" not in out


@pytest.mark.parametrize("kind", ["text", "html"])
def test_norm_id_is_only_used_when_readable_citation_is_missing(kind):
    item = suggestions()[0]
    item["normen"][0].pop("zitat")
    out = render(kind, [item])
    assert "Vorschlag: norm:AO:146" in out
    assert ("Normquelle: norm:AO:146" if kind == "text" else
            "<dt>Normquelle</dt><dd class='kein-treffer'>norm:AO:146</dd>") in out


def test_html_escapes_all_semantic_text_and_metadata_without_links_from_payload():
    bad = "<script>alert('x' & 1)</script>"
    item = suggestions()[0]
    item.update(satz=bad, profil=bad)
    item["kontext"]["text"] = bad
    n = item["normen"][0]
    n.update(id=bad, zitat=bad, titel=bad)
    n["ketten"][0]["anzeige"] = bad
    n["evidenz"].update(fingerprint=bad, anker=[bad])
    n["evidenz"]["passagen"][0].update(text=bad, norm_id=bad)
    out = render("html", [item])
    structure = Structure()
    structure.feed(out)
    assert "script" not in structure.tags and bad not in out
    assert out.count("&lt;script&gt;") >= 9
    assert structure.links == ["#semantik-0-bezug-1"]


@pytest.mark.parametrize("kind, expected", [
    ("text", "391634d913e9ab23c0117e34f8e7eb80fbdaf2266da2413e17f88f719c0faaf3"),
    ("html", "2a214e29e72514655f10f28600efe31fe867560b60f6ddcb0a46ac3fdffda970"),
])
def test_reports_without_semantics_match_pre_change_bytes(kind, expected):
    # Captured from report.py before adding semantic rendering, including an
    # existing risk so unchanged formatting/escaping is checked as well.
    r = report()
    r.risiken = [Risikomarkierung(start=0, end=4, matched_text="Wort", status=Status.UNKNOWN_RISIKO,
                                 titel="Test & Risiko", score=.5)]
    output = report_text(r) if kind == "text" else report_html(r, "Wort & Kontext.")
    assert sha256(output.encode("utf-8")).hexdigest() == expected


def test_cli_emits_semantic_suggestions_in_stdout_and_html(monkeypatch, tmp_path, capsys):
    from risk_layer import cli, embedding, triggerindex

    r = report(suggestions())
    monkeypatch.setattr(cli, "lade_katalog", lambda _: object())
    monkeypatch.setattr(cli, "graph_oeffnen", lambda *a: None)
    monkeypatch.setattr(cli, "Pipeline", lambda _: SimpleNamespace(run=lambda *a, **kw: r))
    monkeypatch.setattr(triggerindex, "index_oeffnen", lambda: None)
    monkeypatch.setattr(embedding, "index_oeffnen", lambda *a: None)
    monkeypatch.setattr(embedding, "pruefe_aktualitaet", lambda *a: True)
    source = tmp_path / "input.txt"
    source.write_text(TEXT, encoding="utf-8")
    destination = tmp_path / "nested" / "report.html"
    assert cli.main(["analyse", str(source), "--no-schicht2", "--html", str(destination)]) == 0
    assert capsys.readouterr().out.count("Vorschlag: § 146 AO") == 1
    assert destination.read_text(encoding="utf-8").count("Vorschlag: § 146 AO") == 1
