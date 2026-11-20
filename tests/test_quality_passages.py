"""Corpus passage coverage and evidence assembly, without a model or network."""

from types import SimpleNamespace

import pytest

from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from tools._quality_passages import (
    assemble_norm_package, build_passages, catalog_anchors, passage_spans,
)


def _node(para, text, *, title="", **kwargs):
    return NormKnoten(id=f"norm:EStG:{para}", law="EStG", para=para,
                     titel=title, text=text, **kwargs)


@pytest.fixture
def graph(tmp_path):
    text = "\n  (1) " + "Der erste Absatz regelt einen eigenen Sachverhalt. " * 30
    text += "\n\n (2) " + "Ein zweiter Absatz beschreibt weitere Voraussetzungen. " * 30
    text += "\n\n (3) Nur am Ende steht der entscheidende Sonderfall.  \n"
    nodes = [
        _node("10", text, title="Sonderausgaben"),
        _node("3", "Diese Einkünfte sind unter den genannten Bedingungen steuerfrei."),
        _node("4", "Eine weitere eigenständige Rechtsfolge bleibt anwendbar.", title="§ 4"),
        _node("5", "(weggefallen)"),
        _node("6", "(1) (weggefallen)\n(2) (aufgehoben)"),
        _node("7", "  \n "),
        _node("8", "(1) (weggefallen)\n(2) Für diese Fälle gilt eine besondere Regelung."),
        # Same parent norm, both properly marked and legacy depth metadata.
        _node("10:abs1", "Der doppelte Absatz darf nicht eigenständig eingebettet werden.", absatz="1"),
        _node("10:abs2", "Auch ohne Metadaten bleibt dies ein Kindknoten."),
    ]
    result = NormGraph.build(reversed(nodes), tmp_path / "graph.sqlite")
    try:
        yield result
    finally:
        result.close()


def test_long_norm_covers_tail_and_every_source_character(graph):
    passages = [p for p in build_passages(graph) if p.norm_id == "norm:EStG:10"]
    source = graph.node("norm:EStG:10")["text"]
    covered = set()
    assert len(passages) > 3
    assert any("entscheidende Sonderfall" in p.text for p in passages)
    assert any(p.start > 1500 for p in passages)
    for passage in passages:
        assert 0 < len(passage.text) <= 900
        assert passage.text == source[passage.start:passage.end]
        covered.update(range(passage.start, passage.end))
    assert all(i in covered for i, c in enumerate(source) if not c.isspace())
    assert any(a.end > b.start for a, b in zip(passages, passages[1:]))


def test_titleless_norms_included_dead_and_duplicate_nodes_excluded(graph):
    passages = build_passages(graph)
    assert {p.norm_id for p in passages} == {
        "norm:EStG:10", "norm:EStG:3", "norm:EStG:4", "norm:EStG:8"}
    assert len({p.passage_id for p in passages}) == len(passages)
    assert [(p.norm_id, p.start) for p in passages] == sorted(
        (p.norm_id, p.start) for p in passages)
    titleless = next(p for p in passages if p.norm_id == "norm:EStG:3")
    assert titleless.title == ""
    assert titleless.embedding_text == titleless.text
    titled = next(p for p in passages if p.norm_id == "norm:EStG:10")
    assert titled.embedding_text == f"Sonderausgaben\n{titled.text}"
    assert passages == build_passages(graph)


@pytest.mark.parametrize("text", [
    "  Vor der Zeile steht die Voraussetzung.\nNach der Zeile folgt die Rechtsfolge.  ",
    "Erster Absatz mit eigenem Sachverhalt.\n\nZweiter Absatz mit eigener Rechtsfolge.",
    "(1) Vor der Zeile steht die Voraussetzung.\n(2) Danach folgt die Rechtsfolge.",
    "X" * 500,
    "\n\n\t",
])
def test_source_offsets_and_coverage_survive_newlines_and_long_tokens(text):
    spans = passage_spans(text, max_chars=60, overlap_chars=12)
    covered = set()
    for start, end in spans:
        assert 0 <= start < end <= len(text)
        assert end - start <= 60
        assert text[start:end] == text[start:end].strip()
        covered.update(range(start, end))
    assert all(i in covered for i, c in enumerate(text) if not c.isspace())
    assert all(a[0] < b[0] for a, b in zip(spans, spans[1:]))


def test_chunk_prefers_paragraph_boundary_over_splitting_next_paragraph():
    first = "Eine Voraussetzung wird in diesem ersten Absatz vollständig beschrieben."
    second = "Eine zweite Voraussetzung wird im nächsten Absatz beschrieben."
    text = first + "\n\n" + second
    spans = passage_spans(text, max_chars=100, overlap_chars=0)
    assert [text[s:e] for s, e in spans] == [first, second]


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (60, -1), (60, 60)])
def test_invalid_window_parameters_fail_clearly(max_chars, overlap):
    with pytest.raises(ValueError):
        passage_spans("Ein ausreichend langer Text.", max_chars=max_chars,
                      overlap_chars=overlap)


def test_catalog_anchors_keep_ownership_and_deduplicate():
    entries = [SimpleNamespace(
        norm_anker=("§ 3 EStG", "§ 3 Abs. 1 EStG", "§ 4 EStG"),
        normketten=(SimpleNamespace(primaer="§ 3 EStG"),),
        paraphrasen=("Einnahmen bleiben steuerfrei.", " Einnahmen bleiben steuerfrei. ", ""),
    )]
    assert catalog_anchors(entries) == {
        "norm:EStG:3": ("Einnahmen bleiben steuerfrei.",),
        "norm:EStG:4": ("Einnahmen bleiben steuerfrei.",),
    }


def test_package_uses_best_passage_and_limited_anchors_without_foreign_norms(graph):
    passages = build_passages(graph)
    own = [p for p in passages if p.norm_id == "norm:EStG:10"]
    foreign = next(p for p in passages if p.norm_id == "norm:EStG:3")
    package = assemble_norm_package(
        "norm:EStG:10:abs3", [foreign, own[-1], own[-1], own[0]],
        ranked_anchors=["Ein passender Kataloganker.", "Ein passender Kataloganker.",
                        "Ein zweiter passender Kataloganker.", "Ein unnötiger weiterer Anker."],
        max_chars=3000,
    )
    assert foreign.text not in package
    assert package.index("entscheidende Sonderfall") < package.index(own[0].text)
    assert package.count("entscheidende Sonderfall") == 1
    assert package.count("Ein passender Kataloganker.") == 1
    assert "Ein zweiter passender Kataloganker." in package
    assert "Ein unnötiger weiterer Anker." not in package
    assert package.startswith("norm:EStG:10\n\nSonderausgaben")
    assert len(assemble_norm_package("norm:EStG:10", own, max_chars=200)) <= 200


def test_package_limits_zero_and_empty_inputs():
    assert assemble_norm_package("norm:EStG:3", [], ranked_anchors=["Anker"],
                                 max_anchors=0) == "norm:EStG:3"
    with pytest.raises(ValueError):
        assemble_norm_package("norm:EStG:3", [], max_passages=-1)
