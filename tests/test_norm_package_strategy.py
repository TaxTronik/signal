"""Source-first packages preserve source identity without changing old ranking."""
import numpy as np
import pytest

from risk_layer.embedding import EmbeddingIndex
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.passages import Passage, assemble_source_first_package


class SpecialCaseEncoder:
    name = "source-first-test"
    revision = None
    dim = 2

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.array([[0., 1.] if "SONDERFALL" in text else [1., 0.] for text in texts],
                        dtype=np.float32)


def passage(source, start=0, end=None, *, norm_id="norm:AO:146", title="Aufzeichnungen"):
    end = len(source) if end is None else end
    return Passage(f"{norm_id}@{start}:{end}", norm_id, norm_id.split(":")[1],
                   norm_id.split(":")[2], title, start, end, source[start:end])


def test_first_rule_survives_wrong_query_passage_and_index_roundtrip(tmp_path):
    general = "(1) Die Aufzeichnungen sind vollständig und geordnet vorzunehmen. "
    text = ("  " + general + "Dieser Grundsatz beschreibt die laufende Dokumentation. " * 12 +
            "\n\n(2) " + "SONDERFALL Auslandsbuchführung und Rückverlagerung. " * 65)
    graph = NormGraph.build([NormKnoten(id="norm:AO:146", law="AO", para="146",
                                      titel="Aufzeichnungen", text=text)], tmp_path / "graph.sqlite")
    try:
        encoder = SpecialCaseEncoder()
        index = EmbeddingIndex.build(graph, encoder, passagen=True)
        vector = np.array([0., 1.])
        before_metadata, before_vectors = index.passage_evidence.storage()
        calls = encoder.calls
        legacy = index.norm_package("norm:AO:146", vector)
        assert general.strip() not in legacy.text
        assert legacy == index.norm_package("norm:AO:146", vector, strategy="ranked")
        package = index.norm_package("norm:AO:146", vector, strategy="source_first")
        assert general.strip() in package.text and "SONDERFALL" in package.text
        first = index.passage_evidence.passages[0]
        assert package.evidence[0] == first
        assert len(package.evidence) == 2
        assert package.evidence[0].end <= package.evidence[1].start
        assert len(package.text) <= 2200
        assert package.source_fingerprint == index.passage_evidence.source_fingerprint
        for p in package.evidence:
            assert text[p.start:p.end] == p.text
        after_metadata, after_vectors = index.passage_evidence.storage()
        assert after_metadata == before_metadata and np.array_equal(after_vectors, before_vectors)
        assert encoder.calls == calls
        assert index.norm_package("norm:AO:146", vector) == legacy
        index.save(tmp_path / "index")
        restored = EmbeddingIndex.load(tmp_path / "index")
        assert restored.input_fingerprint == index.input_fingerprint
        assert restored.norm_package("norm:AO:146", vector, strategy="source_first") == package
        assert restored.norm_package("norm:AO:146", vector) == legacy
        old = EmbeddingIndex.build(graph, encoder)
        assert old.norm_package("norm:AO:146", vector, strategy="source_first") is None
    finally:
        graph.close()


def test_overlapping_source_characters_are_not_repeated():
    source = "Erste Regel. Gemeinsame Grenze.\n\nEine spätere Sonderregel folgt."
    first_end = source.index("\n\n")
    first = passage(source, 0, first_end)
    overlapping = passage(source, source.index("Gemeinsame"))
    covered = passage(source, source.index("Gemeinsame"), first_end)
    package = assemble_source_first_package("norm:AO:146", [covered, overlapping, first])
    assert package.evidence[0] == first
    assert package.evidence[1].passage_id == overlapping.passage_id
    assert package.evidence[1].start == source.index("Eine spätere")
    assert package.text.count("Gemeinsame Grenze.") == 1
    for p in package.evidence:
        assert source[p.start:p.end] == p.text


def test_foreign_retired_and_duplicate_windows_do_not_displace_live_source():
    source = "(weggefallen)\n\nErste lebende Regel.\n\nSpätere Sonderregel."
    first_start = source.index("Erste")
    later_start = source.index("Spätere")
    dead = passage(source, 0, source.index("\n"))
    first = passage(source, first_start, source.index("\n", first_start), title="")
    later = passage(source, later_start, title="")
    foreign = passage("Fremde Rechtsfolge.", norm_id="norm:AO:147")
    package = assemble_source_first_package("norm:AO:146:abs1", [foreign, dead, later, later, first],
                                            ranked_anchors=["Vorhandener Anker."] * 3 +
                                            ["Zweiter vorhandener Anker.", "Dritter Anker."],
                                            source_fingerprint="original-source-hash")
    assert package.evidence == (first, later)
    assert package.anchors == ("Vorhandener Anker.", "Zweiter vorhandener Anker.")
    assert package.source_fingerprint == "original-source-hash"
    assert package.text == ("norm:AO:146\n\nErste lebende Regel.\n\nSpätere Sonderregel.\n\n"
                            "Vorhandener Anker.\n\nZweiter vorhandener Anker.")
    assert not package.truncated


def test_truncated_evidence_uses_rendered_positions_not_matches_in_heading():
    source = "Erste Quellenpassage.\n\nDie spätere Quellenpassage ist länger."
    later_start = source.index("Die spätere")
    title = "Die spätere Quellenpassage ist länger."
    first = passage(source, 0, source.index("\n"), title=title)
    later = passage(source, later_start, title=title)
    prefix = "\n\n".join((first.norm_id, title, first.text)) + "\n\n"
    package = assemble_source_first_package(first.norm_id, [later, first], max_chars=len(prefix) + 11,
                                            ranked_anchors=["Nicht im Paket enthaltener Anker."])
    assert package.text == (prefix + later.text[:11]).rstrip()
    assert package.truncated and package.anchors == ()
    assert package.evidence[0] == first
    tail = package.evidence[1]
    assert tail.text == later.text[:11].rstrip()
    assert source[tail.start:tail.end] == tail.text
    assert tail.end < later.end


def test_long_title_cannot_displace_first_source_and_short_budgets_are_honest():
    first = passage("Eine echte Grundregel.", title="Lange Überschrift " * 400)
    package = assemble_source_first_package(first.norm_id, [first])
    assert len(package.text) <= 2200 and first.text in package.text
    assert package.evidence == (first,) and package.truncated
    short = assemble_source_first_package(first.norm_id, [first], max_chars=len(first.norm_id) + 8)
    assert short.text == first.norm_id + "\n\n" + first.text[:6]
    assert short.truncated and short.evidence[0].text == first.text[:6]


def test_unknown_norm_has_no_invented_source_or_anchor():
    package = assemble_source_first_package("norm:AO:999", [passage("Nur eine fremde Norm.")],
                                            ranked_anchors=["Ohne Quellpassage kein Ersatz."])
    assert package.text == "norm:AO:999"
    assert package.evidence == () and package.anchors == ()


@pytest.mark.parametrize("budget", [0, -1, True, 2200.5])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="max_chars"):
        assemble_source_first_package("norm:AO:146", [], max_chars=budget)


def test_unknown_strategy_rejected_before_query_scoring(tmp_path):
    graph = NormGraph.build([NormKnoten(id="norm:AO:146", law="AO", para="146",
                                      text="Eine reale Quellpassage.")], tmp_path / "graph.sqlite")
    try:
        index = EmbeddingIndex.build(graph, SpecialCaseEncoder(), passagen=True)
        with pytest.raises(ValueError, match="strategy"):
            index.norm_package("norm:AO:146", np.array([1., 0.]), strategy="invented")
    finally:
        graph.close()
