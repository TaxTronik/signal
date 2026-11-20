"""Whole original source units and packaging, without model execution."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.grounded_signal.sources import (
    SourceCorpus, SourceError, canonical_json, package_fingerprint, render_package,
)
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph


def node(nid, text, *, title="Regel", paragraph=None):
    return NormKnoten(id=nid, law=nid.split(":")[1], para=nid.split(":")[2],
                     text=text, titel=title, absatz=paragraph)


def graph_at(tmp_path, nodes):
    path = tmp_path / "original.sqlite"
    graph = NormGraph.build(nodes, path)
    graph.close()
    return path


@pytest.fixture
def corpus(tmp_path):
    first = "(1) Eine Aufzeichnungspflicht verlangt richtige Belege. " * 12
    second = "(2) Abweichend von Absatz 1 besteht bei vollständigem Nachweis eine Ausnahme."
    third = "(3) Die Ausnahme gilt nur im Inland."
    text = first + second + third
    path = graph_at(tmp_path, [
        node("norm:TEST:1", text),
        node("norm:TEST:1:abs1", first, paragraph="1"),
        node("norm:TEST:1:abs2", second, paragraph="2"),
        node("norm:TEST:1:abs3", third, paragraph="3"),
        node("norm:TEST:2", "Eine andere Regel betrifft die eindeutige Kennzeichnung."),
    ])
    with SourceCorpus(path) as value:
        yield value, text


def test_full_paragraphs_preserve_original_prefixes_offsets_and_exception(corpus):
    source, text = corpus
    units = source.units("norm:TEST:1")
    assert len(units) == 3 and all(u.kind == "paragraph" for u in units)
    assert len(units[0].text) > 220
    assert "".join(u.text for u in units) == text
    assert all(text[u.start:u.end] == u.text for u in units)
    assert units[1].text.startswith("(2) Abweichend")
    assert units[2].text.startswith("(3) Die Ausnahme gilt nur")
    assert units[0].whole_rule_id == units[1].whole_rule_id == units[2].whole_rule_id


def test_a_found_subunit_requires_the_entire_parent_rule(corpus):
    source, text = corpus
    payload = render_package(source.package(["norm:TEST:1:abs1"]))
    assert payload["status"] == "ready" and payload["complete_local_norms"]
    assert payload["candidate_norm_ids"] == ["norm:TEST:1"]
    assert "".join(u["text"] for u in payload["units"]) == text
    assert payload["required_groups"][0]["unit_ids"] == payload["unit_ids"]
    assert "requested_subunits_expanded_to_whole_parent_norms" in payload["limitations"]


def test_rule_larger_than_old_cursor_page_remains_whole(tmp_path):
    text = "(1) Der vollständige Grundsatz bleibt sichtbar. " * 150
    text += "(2) Die Ausnahme darf nicht abgeschnitten werden."
    path = graph_at(tmp_path, [node("norm:TEST:1", text)])
    with SourceCorpus(path) as source:
        payload = render_package(source.package(["norm:TEST:1"]))
    assert len(text) > 4000
    assert payload["units"][0]["text"] == text
    assert payload["source_chars"] == len(text)
    assert payload["legal_completeness"] == "unassessed"


def test_overflow_is_atomic_and_keeps_required_counts(corpus):
    source, text = corpus
    package = source.package(["norm:TEST:1", "norm:TEST:2"], max_chars=len(text))
    payload = render_package(package)
    assert payload["status"] == "overflow"
    assert payload["units"] == payload["unit_ids"] == []
    assert not payload["complete_local_norms"] and not payload["reference_closure_complete"]
    assert payload["source_chars"] == 0 and payload["required_source_chars"] > len(text)
    assert payload["candidate_norm_ids"] == ["norm:TEST:1", "norm:TEST:2"]
    assert payload["required_groups"][0]["unit_ids"]


def test_missing_parent_cannot_be_reported_as_complete(corpus):
    source, _ = corpus
    payload = render_package(source.package(["norm:TEST:1", "norm:TEST:999"]))
    assert payload["status"] == "missing" and payload["units"] == []
    assert payload["missing_norm_ids"] == ["norm:TEST:999"]
    assert not payload["complete_local_norms"]


@pytest.mark.parametrize("parent, children, warning", [
    ("abcdef", [("abc", "1"), ("ef", "2")], "paragraph_structure_does_not_cover_whole_parent"),
    ("abcabc", [("abc", "1")], "paragraph_embedding_missing_or_ambiguous"),
    ("abcdef", [("not present", "1")], "paragraph_embedding_missing_or_ambiguous"),
    ("abcdef", [("abcd", "1"), ("cdef", "2")], "paragraph_structure_does_not_cover_whole_parent"),
])
def test_uncertain_structure_falls_back_to_exact_whole_parent(tmp_path, parent, children, warning):
    path = graph_at(tmp_path, [node("norm:TEST:1", parent)] + [
        node(f"norm:TEST:1:abs{number}", text, paragraph=number) for text, number in children])
    with SourceCorpus(path) as source:
        payload = render_package(source.package(["norm:TEST:1"]))
    assert payload["units"][0]["kind"] == "whole_norm"
    assert payload["units"][0]["text"] == parent
    assert any(warning in value for value in payload["limitations"])


def test_original_whitespace_between_paragraphs_is_preserved(tmp_path):
    path = graph_at(tmp_path, [node("norm:TEST:1", "(1) Alpha.\n\n(2) Beta."),
                              node("norm:TEST:1:abs1", "(1) Alpha.", paragraph="1"),
                              node("norm:TEST:1:abs2", "(2) Beta.", paragraph="2")])
    with SourceCorpus(path) as source:
        units = source.units("norm:TEST:1")
    assert len(units) == 2 and units[0].text.endswith("\n\n")
    assert units[1].start == len(units[0].text)


def test_evidence_is_immutable_and_quotes_default_to_the_full_unit(corpus):
    source, _ = corpus
    unit = source.units("norm:TEST:1")[1]
    with pytest.raises(FrozenInstanceError):
        unit.text = "changed"
    quote = source.quote(unit.unit_id)
    assert quote.whole_unit and quote.text == unit.text
    assert (quote.start, quote.end) == (unit.start, unit.end)
    part = source.quote(unit.unit_id, 0, 3)
    assert not part.whole_unit and part.text == "(2)" and part.start == unit.start


@pytest.mark.parametrize("start,end", [(True, 3), (0, False), (-1, 4), (0, 999999), (2, 2)])
def test_quote_offsets_fail_closed(corpus, start, end):
    source, _ = corpus
    with pytest.raises(SourceError):
        source.quote(source.units("norm:TEST:1")[0].unit_id, start, end)


def test_package_identity_is_order_independent_and_rendering_cannot_mutate_it(corpus):
    source, _ = corpus
    one = source.package(["norm:TEST:1", "norm:TEST:2"])
    two = source.package(["norm:TEST:2", "norm:TEST:1", "norm:TEST:1"])
    assert one == two
    payload = render_package(one)
    assert payload["fingerprint"] == package_fingerprint(payload)
    assert payload["package_id"] == "pkg:" + package_fingerprint(payload)
    payload["units"][0]["text"] = "mutated copy"
    assert render_package(one)["units"][0]["text"] != "mutated copy"
    with pytest.raises(SourceError):
        render_package(replace(one, units=()))


def test_real_graph_references_and_open_dependency_frontier(tmp_path):
    path = graph_at(tmp_path, [node("norm:TEST:1", "Es gilt § 2 TEST."),
                              node("norm:TEST:2", "Es gilt § 3 TEST.")])
    with SourceCorpus(path) as source:
        shallow = render_package(source.package(["norm:TEST:1"]))
        deeper = render_package(source.package(["norm:TEST:1"], reference_depth=1))
    assert shallow["references"][0]["target_id"] == "norm:TEST:2"
    assert shallow["references"][0]["exact_target_available"]
    assert not shallow["reference_closure_complete"]
    assert deeper["norm_ids"] == ["norm:TEST:1", "norm:TEST:2"]
    assert deeper["candidate_norm_ids"] == ["norm:TEST:1"]
    dangling = next(r for r in deeper["references"] if r["target_id"] == "norm:TEST:3")
    assert not dangling["exact_target_available"] and dangling["resolved_norm_id"] is None
    assert not deeper["reference_closure_complete"]
    assert deeper["complete_local_norms"]


def test_bm25_search_is_generic_and_packages_all_exception_units(corpus):
    source, text = corpus
    hits = source.search("Aufzeichnungspflicht richtige Belege", limit=2)
    assert hits[0].norm_id == "norm:TEST:1" and hits[0].score > 0
    assert source.search("Aufzeichnungspflicht richtige Belege", limit=2) == hits
    package = render_package(source.package([hits[0].norm_id]))
    assert "".join(u["text"] for u in package["units"]) == text
    assert source.search("zzzzzzzzzzzz") == ()
    assert source.retrieval_metadata()["policy"]["anchors"] is False
    with pytest.raises(TypeError):
        source.search("Belege", targets=["norm:TEST:2"])


def test_optional_hybrid_delegates_to_existing_cpu_path_and_expands_to_full_norm(tmp_path, monkeypatch):
    import risk_layer.research_sources
    calls = []

    class StubRetriever:
        def __init__(self, graph, index, **options):
            calls.append(("init", options))

        def prepare(self):
            calls.append(("prepare",))

        def _search(self, query, limit):
            calls.append(("search", query, limit))
            return [SimpleNamespace(norm_id="norm:TEST:1", start=8, end=15)]

        def retrieval_metadata(self):
            return {"mode": "hybrid", "model_identity": {"device": "cpu"}}

        @property
        def retrieval_metrics(self):
            return {"query_journal": [{"query": "Beleg"}]}

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(risk_layer.research_sources, "ResearchCorpus", StubRetriever)
    text = "(1) Eine vollständige Regel bleibt als Quelle erhalten."
    path = graph_at(tmp_path, [node("norm:TEST:1", text)])
    with SourceCorpus(path, tmp_path / "index", retrieval="hybrid", encoder_model_path=tmp_path / "cpu-model") as source:
        source.prepare_lookup()
        hits = source.search("Beleg", 1)
        assert hits[0].score is None and hits[0].matched_unit_ids
        assert render_package(source.package([hits[0].norm_id]))["units"][0]["text"] == text
        assert source.retrieval_metrics["query_journal"][0]["query"] == "Beleg"
    assert calls[0][1]["retrieval"] == "hybrid" and calls[-1] == ("close",)


def test_export_is_an_exact_portable_snapshot_without_model_weights(corpus, tmp_path):
    source, _ = corpus
    manifest = source.export_snapshot(tmp_path / "snapshot")
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((tmp_path / "snapshot" / name).read_bytes()).hexdigest() == expected
    nodes = json.loads((tmp_path / "snapshot/nodes.json").read_text(encoding="utf-8"))
    assert any(n["id"] == "norm:TEST:1" for n in nodes)
    assert not manifest["contains_model_or_index_vectors"]
    with pytest.raises(SourceError):
        source.export_snapshot(tmp_path / "snapshot")
    assert canonical_json(manifest) == (tmp_path / "snapshot/manifest.json").read_text(encoding="utf-8")


def test_unit_ids_change_with_original_source_content(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    a = graph_at(first, [node("norm:TEST:1", "Die Ausnahme besteht.")])
    b = graph_at(second, [node("norm:TEST:1", "Die Ausnahme besteht nicht.")])
    with SourceCorpus(a) as left, SourceCorpus(b) as right:
        assert left.units("norm:TEST:1")[0].unit_id != right.units("norm:TEST:1")[0].unit_id


def test_empty_original_norm_is_a_missing_source(tmp_path):
    path = graph_at(tmp_path, [node("norm:TEST:1", " ")])
    with SourceCorpus(path) as source:
        assert render_package(source.package(["norm:TEST:1"]))["status"] == "missing"


def test_closed_corpus_cannot_issue_evidence(corpus):
    source, _ = corpus
    source.close()
    with pytest.raises(SourceError):
        source.package(["norm:TEST:1"])


@pytest.mark.parametrize("options", [{"max_chars": True}, {"max_chars": 0},
                                    {"reference_depth": True}, {"reference_depth": 4}])
def test_packaging_budgets_are_strict(corpus, options):
    source, _ = corpus
    with pytest.raises(SourceError):
        source.package(["norm:TEST:1"], **options)


def test_real_local_astg8_has_the_entire_exception_group_if_archive_available():
    graph = Path(__file__).resolve().parents[1] / "out/quality-research-moe-additional-20260908-r1/inputs/graph.sqlite"
    if not graph.is_file():
        pytest.skip("Private frozen original graph is not installed")
    with SourceCorpus(graph) as source:
        units = source.units("norm:AStG:8")
        package = render_package(source.package(["norm:AStG:8", "norm:AStG:7", "norm:AO:42"]))
    assert len(units) == 5
    assert units[1].start == 6488 and units[1].text.startswith("(2) Ungeachtet")
    assert units[2].text.startswith("(3) Absatz 2 gilt nur")
    assert package["status"] == "ready" and package["source_chars"] < 32000
