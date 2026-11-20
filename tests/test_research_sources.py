"""Real source/index snapshots and bounded evidence-only research sessions."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from risk_layer.embedding import EmbeddingIndex, HashEncoder
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.research_sources import ResearchBudgetExceeded, ResearchCorpus, ResearchSourceError


@pytest.fixture
def corpus_files(tmp_path):
    text = ("Allgemeine Voraussetzungen werden im ersten Abschnitt beschrieben. " * 95
            + "\n\nSchlussspezialregel zur seltenen Verrechnung steht ausschließlich am Ende. "
            + "Für diesen Vorgang gelten § 2 Abs. 99 AO und § 999 AO.")
    nodes = [NormKnoten(id="norm:AO:1", law="AO", para="1", titel="Allgemeine Regel", text=text)]
    nodes += [NormKnoten(id=f"norm:AO:{i}", law="AO", para=str(i), titel=f"Norm {i}",
                        text=f"Diese Regelung behandelt den Vorgang und seine Voraussetzung Nummer {i}."
                             + (" Ergänzend gilt § 3 AO." if i == 2 else "")) for i in range(2, 8)]
    nodes += [NormKnoten(id="norm:AO:8", law="AO", para="8", titel="",
                        text="Titelloserbegriff bezeichnet einen vollständig belegten Sachverhalt."),
              NormKnoten(id="norm:AO:9", law="AO", para="9", titel="Aufgehoben", text="(weggefallen)")]
    graph_path, index_dir = tmp_path / "graph.sqlite", tmp_path / "index"
    graph = NormGraph.build(nodes, graph_path)
    try:
        EmbeddingIndex.build(graph, HashEncoder(dim=32), passagen=True).save(index_dir)
    finally:
        graph.close()
    return graph_path, index_dir, {n.id: n.text for n in nodes}


@pytest.fixture
def corpus(corpus_files):
    graph_path, index_dir, _ = corpus_files
    value = ResearchCorpus(graph_path, index_dir)
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def catalog_corpus_files(corpus_files, tmp_path):
    from risk_layer.catalog import lade_katalog
    graph_path, index_dir, texts = corpus_files
    path = tmp_path / "research-catalog.yaml"
    path.write_text(json.dumps({"version": "hints-1", "begriffe": [{
        "id": "formelle_ordnung", "begriff_kanonisch": "formelle Ordnung",
        "norm_anker": ["§ 2 AO"], "definitionsquelle": [{"typ": "Gesetz", "fundstelle": "§ 2 AO"}],
        "normketten": [{"glieder": ["§ 2 AO", "§ 7 AO"], "verknuepfung": "iVm"}]}]}), encoding="utf-8")
    graph = NormGraph(graph_path)
    try:
        EmbeddingIndex.build(graph, HashEncoder(dim=32), katalog=lade_katalog(path), passagen=True).save(index_dir)
    finally:
        graph.close()
    return graph_path, index_dir, texts, path


def test_catalog_hints_do_not_change_original_sources_or_source_budget(catalog_corpus_files):
    graph, index, _, path = catalog_corpus_files
    disabled, enabled = ResearchCorpus(graph, index), ResearchCorpus(graph, index, catalog_path=path)
    try:
        text = "Die formelle Ordnung dieses Vorgangs wird untersucht."
        old, new = disabled.new_session(text), enabled.new_session(text)
        assert disabled.catalog_metadata() is None and old.catalog_hints is None
        assert disabled.identity_metadata() == enabled.identity_metadata()
        assert old.initial_sources() == new.initial_sources()
        assert old.budget == new.budget and new.budget["used_calls"] == 0
        hints = new.catalog_hints
        assert hints["catalog_fingerprint"] == enabled.catalog_metadata()["fingerprint"]
        assert hints["concepts"][0]["hint_id"] == "k1"
        assert hints["concepts"][0]["norm_anchors"] == ["§ 2 AO"]
        assert enabled.new_session("Ein neutraler Satz.").catalog_hints["concepts"] == []
        hints["concepts"][0]["norm_anchors"].append("Invented")
        assert new.catalog_hints["concepts"][0]["norm_anchors"] == ["§ 2 AO"]
    finally:
        disabled.close()
        enabled.close()


def test_catalog_hints_cannot_be_used_as_evidence_or_original_graph_edges(catalog_corpus_files):
    graph, index, _, path = catalog_corpus_files
    corpus = ResearchCorpus(graph, index, catalog_path=path)
    try:
        text = "Die formelle Ordnung dieses Vorgangs wird untersucht."
        unread, reader = corpus.new_session(text), corpus.new_session(text)
        hint = unread.catalog_hints["concepts"][0]
        record = reader.execute("read_source", {"norm_id": "norm:AO:2"})["source"]
        with pytest.raises(ResearchSourceError, match="never supplied"):
            finish(unread, [finding(unread, record, evidence_id=hint["hint_id"])])
        with pytest.raises(ResearchSourceError, match="never supplied"):
            finish(unread, [finding(unread, record)])
        citations = reader.execute("follow_citations", {"norm_id": "norm:AO:2"})["citations"]
        assert all(edge["norm_id"] != "norm:AO:7" for edge in citations)
        source = unread.execute("read_source", {"norm_id": "norm:AO:2"})["source"]
        assert finish(unread, [finding(unread, source)])[0]["normen"][0]["research"]["source_verified"] is True
    finally:
        corpus.close()


def test_catalog_index_mismatch_fails_without_silent_disable(catalog_corpus_files):
    graph, index, _, path = catalog_corpus_files
    data = json.loads(path.read_text(encoding="utf-8"))
    data["begriffe"][0]["norm_anker"] = ["§ 3 AO"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="different catalog generations"):
        ResearchCorpus(graph, index, catalog_path=path)


def test_catalog_corpus_uses_frozen_yaml_snapshot(catalog_corpus_files):
    graph, index, _, path = catalog_corpus_files
    corpus = ResearchCorpus(graph, index, catalog_path=path)
    try:
        metadata = corpus.catalog_metadata()
        path.write_text("invalid changed yaml", encoding="utf-8")
        assert corpus.catalog_metadata() == metadata
        assert corpus.new_session("Formelle Ordnung.").catalog_hints["concepts"][0]["term"] == "formelle Ordnung"
        metadata["policy"]["max_concepts"] = 0
        assert corpus.catalog_metadata()["policy"]["max_concepts"] == 4
    finally:
        corpus.close()


@pytest.fixture
def semantic_catalog_corpus(catalog_corpus_files, monkeypatch):
    import hashlib
    from risk_layer import research_hybrid
    from risk_layer.catalog import lade_katalog
    graph_path, index_dir, _, path = catalog_corpus_files
    data = json.loads(path.read_text(encoding="utf-8"))
    data["begriffe"][0]["paraphrasen"] = ["Die Ordnung der Belege ist ein Prüfgegenstand."]
    path.write_text(json.dumps(data), encoding="utf-8")
    graph = NormGraph(graph_path)
    try:
        EmbeddingIndex.build(graph, HashEncoder(dim=32), katalog=lade_katalog(path), passagen=True).save(index_dir)
    finally:
        graph.close()
    instances = []

    class Retriever:
        def __init__(self, evidence, model_path):
            self.evidence, self.model_path, self.calls = evidence, Path("unused"), []
            instances.append(self)
        def identity_metadata(self):
            return {"model": {"files": {}}, "fingerprint": "retrieval-generation", "policy": {},
                    "format": "research_cpu_hybrid/2",
                    "query_vector": {"dtype": "float32", "dimension": 32, "byte_order": "little", "order": "C"},
                    "evidence_fingerprint": self.evidence.fingerprint,
                    "source_fingerprint": self.evidence.source_fingerprint}
        @property
        def metrics(self):
            return {"query_count": len(self.calls), "queries": deepcopy(self.calls)}
        def search_with_vector(self, query, limit=6):
            vector = self.evidence.anchor_vectors[0].copy()
            vector.flags.writeable = False
            self.calls.append({"query_sha256": hashlib.sha256(query.encode()).hexdigest(), "status": "complete",
                               "query_vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(), "limit": limit})
            by_norm = {}
            for passage in self.evidence.passages:
                by_norm.setdefault(passage.norm_id, passage)
            return list(by_norm.values())[:limit], vector
        def search(self, query, limit=6):
            raise AssertionError("Semantic initial search must not encode a second query")
        def close(self):
            pass

    monkeypatch.setattr(research_hybrid, "HybridResearchRetriever", Retriever)
    corpus = ResearchCorpus(graph_path, index_dir, catalog_path=path, retrieval="hybrid")
    try:
        yield corpus, instances[0]
    finally:
        corpus.close()


def test_catalog_semantic_initial_pool_reuses_one_actual_vector_and_keeps_it_private(semantic_catalog_corpus):
    corpus, retriever = semantic_catalog_corpus
    session = corpus.new_session("Ein indirekt beschriebener Vorgang soll untersucht werden.")
    with pytest.raises(ResearchSourceError, match="initial search first"):
        _ = session.catalog_hints
    assert session.catalog_query_vector is None
    first = session.initial_sources(1)
    full = session.initial_sources(6)
    assert first == full[:1] and len(retriever.calls) == 1 and retriever.calls[0]["limit"] == 6
    assert session.budget["used_calls"] == 0
    assert session.catalog_hints["concepts"][0]["matches"] == []
    assert session.catalog_hints["concepts"][0]["matching"] == "semantic_anchor"
    vector = session.catalog_query_vector
    assert vector == retriever.evidence.anchor_vectors[0].tolist()
    vector[0] += 1.
    assert session.catalog_query_vector != vector
    assert "query_vector" not in json.dumps(session.catalog_hints)
    assert corpus.retrieval_metadata()["query_vector"] == corpus.catalog_metadata()["semantic"]["query_vector"]
    assert corpus.retrieval_metadata()["format"] == "research_cpu_hybrid/2"


def test_catalog_semantic_real_agent_receives_hints_but_not_private_vector(semantic_catalog_corpus):
    from risk_layer.research_agent import ResearchAgent
    corpus, retriever = semantic_catalog_corpus

    class Agent(ResearchAgent):
        def _prompt_tokens(self, payload, deadline, result):
            return 100
        def _completion(self, payload, deadline, result, count=None):
            self.sent = deepcopy(payload)
            result["usage"]["model_calls"] += 1
            return {"role": "assistant", "tool_calls": [{"id": "finish-one", "type": "function",
                    "function": {"name": "finish", "arguments": '{"findings":[],"unresolved":[]}'}}]}

    agent = Agent("http://127.0.0.1:12345", "synthetic-local")
    result = agent.run("Ein indirekt beschriebener Vorgang soll untersucht werden.", corpus)
    assert result["status"] == "complete", result["errors"]
    assert len(retriever.calls) == 1
    assert result["catalog_query_vector"] == retriever.evidence.anchor_vectors[0].tolist()
    user = json.loads(agent.sent["messages"][1]["content"])
    assert user["KATALOGHINWEISE"] == result["catalog_hints"]
    assert "catalog_query_vector" not in user and "query_vector" not in json.dumps(user["KATALOGHINWEISE"])
    assert all(not identifier.startswith("k") for identifier in result["quote_references"])


def finding(session, record, **changes):
    result = {"focus_id": "f0", "norm_id": record["norm_id"], "evidence_id": record["evidence_id"],
              "source_quote": session.focuses[0]["text"][:20], "norm_quote": record["text"][:40],
              "reason": "Die Originalregel erklärt den beschriebenen Vorgang."}
    result.update(changes)
    return result


def finish(session, records):
    return session.validate_finish({"findings": records, "unresolved": []})


def test_bm25_finds_original_tail_beyond_old_1500_character_prefix(corpus, corpus_files):
    session = corpus.new_session("Die Schlussspezialregel zur Verrechnung muss untersucht werden.")
    source = session.execute("search_sources", {"query": "Schlussspezialregel Verrechnung"})["results"][0]
    assert source["norm_id"] == "norm:AO:1" and source["source_start"] > 1500
    assert len(source["text"]) <= 900 and "Schlussspezialregel" in source["text"]
    assert source["text"] == corpus_files[2][source["norm_id"]][source["source_start"]:source["source_end"]]
    assert session.budget["used_calls"] == 1


def test_identical_initial_sources_are_budget_free_and_sessions_do_not_share_permissions(corpus):
    text = "Diese Regelung betrifft einen Vorgang und seine Voraussetzung."
    left, right, unread = [corpus.new_session(text) for _ in range(3)]
    records = left.initial_sources()
    assert records == right.initial_sources() == left.initial_sources()
    assert 1 <= len(records) <= 6 and left.budget["used_calls"] == 0
    with pytest.raises(ResearchSourceError, match="never supplied"):
        finish(unread, [finding(unread, records[0])])
    changed = deepcopy(records[0])
    records[0]["text"] = "Invented replacement"
    result = finish(left, [finding(left, changed)])
    assert result[0]["normen"][0]["evidenz"]["passagen"][0]["text"] == changed["text"]


def test_initial_limit_registers_only_sources_actually_returned(corpus):
    text = "Diese Regelung betrifft einen Vorgang und seine Voraussetzung."
    full, limited = corpus.new_session(text), corpus.new_session(text)
    all_records = full.initial_sources()
    assert len(all_records) > 1 and limited.initial_sources(1) == all_records[:1]
    with pytest.raises(ResearchSourceError, match="never supplied"):
        finish(limited, [finding(limited, all_records[1])])


def test_full_norm_pagination_preserves_newlines_and_explicit_continuation(corpus, corpus_files):
    session = corpus.new_session("Der gesamte Originaltext soll gelesen werden.")
    first = session.execute("read_source", {"norm_id": "norm:AO:1"})["source"]
    second = session.execute("read_source", {"norm_id": "norm:AO:1", "cursor": first["next_cursor"]})["source"]
    assert first["source_end"] == first["next_cursor"] == second["source_start"] == 4000
    assert first["text"] + second["text"] == corpus_files[2]["norm:AO:1"]
    assert not first["complete"] and second["next_cursor"] is None
    assert "\n\n" in second["text"] and first["text_sha256"] == second["text_sha256"]


def test_resolved_ancestor_cannot_be_misrepresented_as_requested_subsection(corpus):
    session = corpus.new_session("Die Regelung betrifft diesen Vorgang.")
    source = session.execute("read_source", {"norm_id": "norm:AO:2:abs99"})["source"]
    assert source["requested_id"] == "norm:AO:2:abs99" and source["norm_id"] == "norm:AO:2"
    assert source["fallback"] is True
    with pytest.raises(ResearchSourceError, match="resolved source"):
        finish(session, [finding(session, source, norm_id="norm:AO:2:abs99")])
    assert finish(session, [finding(session, source)])[0]["normen"][0]["id"] == "norm:AO:2"


def test_missing_and_retired_norms_are_explicit_without_fabricated_evidence(corpus):
    session = corpus.new_session("Ein fehlender Normtext soll geprüft werden.")
    missing = session.execute("read_source", {"norm_id": "norm:AO:999"})
    retired = session.execute("read_source", {"norm_id": "norm:AO:9"})
    assert missing["status"] == "missing" and missing["source"] is None
    assert retired["status"] == "not_live" and retired["source"] is None
    assert finish(session, []) == []


def test_titleless_live_norm_is_searchable(corpus):
    session = corpus.new_session("Der Titelloserbegriff muss erläutert werden.")
    results = session.initial_sources()
    assert results[0]["norm_id"] == "norm:AO:8" and results[0]["titel"] == ""
    assert all(r["norm_id"] != "norm:AO:9" for r in results)


def test_citation_tools_report_original_dangling_edges_and_do_not_read_their_targets(corpus):
    session = corpus.new_session("Ein Verweis auf eine andere Regelung wird geprüft.")
    outgoing = session.execute("follow_citations", {"norm_id": "norm:AO:1"})
    by_target = {r["norm_id"]: r for r in outgoing["citations"]}
    assert {"norm:AO:2:abs99", "norm:AO:999"} <= by_target.keys()
    assert not by_target["norm:AO:2:abs99"]["exact_target_present"]
    assert not by_target["norm:AO:999"]["declared_present"]
    assert all(e["src"] == "norm:AO:1" and e["confidence"] for e in by_target.values())
    capped = session.execute("follow_citations", {"norm_id": "norm:AO:1", "limit": 1})
    assert len(capped["citations"]) == 1 and capped["has_more"]
    other = corpus.new_session(session.focuses[0]["text"])
    page = other.execute("read_source", {"norm_id": "norm:AO:2"})["source"]
    with pytest.raises(ResearchSourceError, match="never supplied"):
        finish(session, [finding(session, page)])


def test_source_quote_from_another_sentence_keeps_its_actual_context_offsets(corpus):
    text = "Die Regelung betrifft diesen Vorgang. Die Tochter ist eine GmbH."
    session = corpus.new_session(text)
    source = session.execute("read_source", {"norm_id": "norm:AO:2"})["source"]
    result = finish(session, [finding(session, source, source_quote="Die Tochter")])
    assert result[0]["start"] == session.focuses[0]["start"]
    quote = result[0]["normen"][0]["research"]["source_quote"]
    assert quote["start"] == text.index("Die Tochter") > result[0]["end"]
    assert text[quote["start"]:quote["end"]] == quote["text"]
    assert result[0]["normen"][0]["research"]["legal_validity_verified"] is False


def test_incoming_citations_preserve_their_original_edge_direction(corpus):
    session = corpus.new_session("Ein Verweis auf eine andere Regelung wird geprüft.")
    incoming = session.execute("follow_citations", {"norm_id": "norm:AO:3", "direction": "in"})
    assert len(incoming["citations"]) == 1
    edge = incoming["citations"][0]
    assert edge["src"] == edge["norm_id"] == "norm:AO:2" and edge["dst"] == "norm:AO:3"
    assert edge["direction"] == "in" and edge["declared_present"] and edge["exact_target_present"]


def test_same_norm_evidence_can_support_two_explicit_distinct_foci(corpus):
    session = corpus.new_session("Die Regelung betrifft diesen Vorgang. Die Regelung betrifft einen weiteren Vorgang.")
    page = session.execute("read_source", {"norm_id": "norm:AO:2"})["source"]
    results = finish(session, [finding(session, page, focus_id=focus["id"], source_quote="Die Regelung")
                               for focus in session.focuses])
    assert len(results) == 2
    for result in results:
        quote = result["normen"][0]["research"]["source_quote"]
        assert quote["start"] == result["start"]


@pytest.mark.parametrize("field,value", [("norm_quote", "A made-up legal condition"),
                                        ("source_quote", "Invented original facts"),
                                        ("norm_quote", "Diese\u00a0Regelung"),
                                        ("source_quote", "Die\tRegelung"),
                                        ("reason", " "), ("norm_quote", "kurz"),
                                        ("source_quote", "ab"), ("reason", "x" * 401)])
def test_false_normalized_or_out_of_schema_quotes_fail_without_rewriting(corpus, field, value):
    session = corpus.new_session("Die Regelung betrifft diesen Vorgang.")
    source = session.execute("read_source", {"norm_id": "norm:AO:2"})["source"]
    with pytest.raises(ResearchSourceError):
        finish(session, [finding(session, source, **{field: value})])
    assert finish(session, [finding(session, source)])  # Validation is atomic.


def test_a_real_quote_from_an_unread_later_page_is_not_accepted(corpus):
    session = corpus.new_session("Der gesamte Originaltext soll gelesen werden.")
    source = session.execute("read_source", {"norm_id": "norm:AO:1"})["source"]
    with pytest.raises(ResearchSourceError, match="supplied original page"):
        finish(session, [finding(session, source, norm_quote="Schlussspezialregel zur seltenen Verrechnung")])


def test_ambiguous_quote_outside_focus_is_rejected(corpus):
    session = corpus.new_session("Dieser andere Fokus wird untersucht. Die Tochter handelt selbst. Die Tochter handelt später.")
    source = session.execute("read_source", {"norm_id": "norm:AO:2"})["source"]
    with pytest.raises(ResearchSourceError, match="ambiguous"):
        finish(session, [finding(session, source, source_quote="Die Tochter")])


def test_duplicates_and_six_norms_fail_instead_of_silent_top5_truncation(corpus):
    session = corpus.new_session("Diese Regelung betrifft einen Vorgang und seine Voraussetzung.")
    sources = session.initial_sources()
    assert len(sources) == 6
    items = [finding(session, r) for r in sources]
    with pytest.raises(ResearchSourceError, match="Duplicate"):
        finish(session, [items[0], items[0]])
    with pytest.raises(ResearchSourceError, match="five norms"):
        finish(session, items)
    result = finish(session, items[:5])
    assert len(result[0]["normen"]) == 5 and all(n["rerank"] is None for n in result[0]["normen"])


def test_finish_is_strict_and_success_closes_session(corpus):
    session = corpus.new_session("Dieser Sachverhalt wird zunächst untersucht.")
    for arguments in ({"findings": []}, {"findings": [], "unresolved": [], "ignored": True},
                      {"findings": [], "unresolved": ["unknown"] * 9}):
        with pytest.raises(ResearchSourceError):
            session.validate_finish(arguments)
    assert session.validate_finish({"findings": [], "unresolved": ["Die konkrete Rechtsform fehlt."]}) == []
    with pytest.raises(ResearchSourceError, match="finished"):
        session.execute("read_source", {"norm_id": "norm:AO:2"})


@pytest.mark.parametrize("name,arguments,limit", [("search_sources", {"query": "Regelung"}, 3),
    ("read_source", {"norm_id": "norm:AO:2"}, 5), ("follow_citations", {"norm_id": "norm:AO:1"}, 2)])
def test_individual_tool_budgets_are_enforced(corpus, name, arguments, limit):
    session = corpus.new_session("Dieser Sachverhalt wird zunächst untersucht.")
    for _ in range(limit):
        session.execute(name, arguments)
    with pytest.raises(ResearchBudgetExceeded):
        session.execute(name, arguments)
    assert session.budget["used_calls"] == limit and session.budget["per_tool"][name] == limit


def test_total_budget_and_invalid_attempts_are_not_bypassable(corpus):
    session = corpus.new_session("Dieser Sachverhalt wird zunächst untersucht.")
    for _ in range(2):
        session.execute("search_sources", {"query": "Regelung"})
        session.execute("follow_citations", {"norm_id": "norm:AO:1"})
    for _ in range(3):
        session.execute("read_source", {"norm_id": "norm:AO:2"})
    with pytest.raises(ResearchSourceError, match="Unknown"):
        session.execute("arbitrary_sql", {})
    assert session.budget["used_calls"] == 8
    with pytest.raises(ResearchBudgetExceeded):
        session.execute("read_source", {"norm_id": "norm:AO:2"})
    assert finish(session, []) == []


@pytest.mark.parametrize("name,args", [("search_sources", {"query": "Regelung", "limit": True}),
    ("search_sources", {"query": "Regelung", "limit": 7}), ("read_source", {"norm_id": "norm:AO:2", "cursor": -1}),
    ("read_source", {"norm_id": "norm:AO:2", "cursor": True}), ("read_source", {"norm_id": "norm:AO:2", "cursor": 99999}),
    ("read_source", {"norm_id": "../../secret"}), ("follow_citations", {"norm_id": "norm:AO:1", "direction": "both"})])
def test_invalid_tool_arguments_are_counted_and_never_silently_clamped(corpus, name, args):
    session = corpus.new_session("Dieser Sachverhalt wird zunächst untersucht.")
    with pytest.raises(ResearchSourceError):
        session.execute(name, args)
    assert session.budget["used_calls"] == session.budget["per_tool"][name] == 1


def test_graph_generation_mismatch_is_rejected_including_changed_metadata(corpus_files):
    graph_path, index_dir, _ = corpus_files
    with sqlite3.connect(graph_path) as con:
        con.execute("UPDATE knoten SET titel='Changed source title' WHERE id='norm:AO:1'")
    with pytest.raises(ResearchSourceError, match="different generations"):
        ResearchCorpus(graph_path, index_dir)


def test_source_snapshot_remains_immutable_across_sessions(corpus, corpus_files):
    with sqlite3.connect(corpus_files[0]) as con:
        con.execute("UPDATE knoten SET text='Changed external source' WHERE id='norm:AO:2'")
    session = corpus.new_session("Die Regelung betrifft diesen Vorgang.")
    source = session.execute("read_source", {"norm_id": "norm:AO:2"})["source"]
    assert source["text"] == corpus_files[2]["norm:AO:2"]
    focuses = session.focuses
    focuses[0]["start"] = 500
    budget = session.budget
    budget["used_calls"] = 0
    assert session.focuses[0]["start"] == 0 and session.budget["used_calls"] == 1
    corpus.close()
    corpus.close()
    with pytest.raises(ResearchSourceError, match="closed"):
        corpus.new_session("A new session")


def test_real_corpus_load_and_search_do_not_import_torch(corpus_files):
    graph_path, index_dir, _ = corpus_files
    code = """import json,sys
from pathlib import Path
from risk_layer.research_sources import ResearchCorpus
c=ResearchCorpus(Path(sys.argv[1]),Path(sys.argv[2]))
s=c.new_session('Die Schlussspezialregel zur Verrechnung wird untersucht.')
assert s.initial_sources()
c.close()
print(json.dumps({'torch': 'torch' in sys.modules}))
"""
    result = subprocess.run([sys.executable, "-c", code, str(graph_path), str(index_dir)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout) == {"torch": False}


def test_host_selected_quote_offset_preserves_second_identical_occurrence(corpus):
    session = corpus.new_session("Die allgemeine Regel muss geprüft werden.")
    record = session.execute("read_source", {"norm_id": "norm:AO:1"})["source"]
    quote = "Allgemeine Voraussetzungen"
    second = record["text"].index(quote, record["text"].index(quote) + 1)
    result = finish(session, [finding(session, record, norm_quote=quote,
                                      norm_quote_start=record["source_start"] + second)])
    selected = result[0]["normen"][0]["research"]["norm_quote"]
    assert selected["start"] == second and selected["text"] == quote


@pytest.mark.parametrize("offset", [-1, 1, 5000, True, "0"])
def test_selected_quote_offset_must_match_actual_original_page(corpus, offset):
    session = corpus.new_session("Die allgemeine Regel muss geprüft werden.")
    record = session.execute("read_source", {"norm_id": "norm:AO:1"})["source"]
    with pytest.raises(ResearchSourceError, match="Selected norm quote offset"):
        finish(session, [finding(session, record, norm_quote="Allgemeine Voraussetzungen", norm_quote_start=offset)])


@pytest.mark.parametrize("offset", [-1, 1, 5000, True, "0"])
def test_selected_case_quote_offset_must_match_original_text(corpus, offset):
    session = corpus.new_session("Dieser Sachverhalt enthält eine allgemeine Regel.")
    record = session.execute("read_source", {"norm_id": "norm:AO:1"})["source"]
    with pytest.raises(ResearchSourceError, match="Selected source quote offset"):
        finish(session, [finding(session, record, source_quote="Dieser Sachverhalt", source_quote_start=offset)])


def test_selected_case_quote_can_identify_one_repeated_original_occurrence(corpus):
    text = "Dieser Sachverhalt gilt. Dieser Sachverhalt gilt."
    session = corpus.new_session(text)
    record = session.execute("read_source", {"norm_id": "norm:AO:1"})["source"]
    quote = "Dieser Sachverhalt"
    second = text.index(quote, 1)
    result = finish(session, [finding(session, record, source_quote=quote, source_quote_start=second)])
    assert result[0]["normen"][0]["research"]["source_quote"]["start"] == second


def test_hybrid_corpus_uses_one_retriever_and_preserves_original_source_ledger(corpus_files, monkeypatch):
    from risk_layer import research_hybrid
    instances = []
    class Retriever:
        def __init__(self, evidence, model_path):
            self.evidence = evidence
            self.model_path = Path("model").resolve()
            self.loaded = 0
            self.closed = False
            self.calls = []
            instances.append(self)
        def identity_metadata(self):
            return {"model": {"files": {}}, "fingerprint": "retrieval-generation", "policy": {},
                    "evidence_fingerprint": self.evidence.fingerprint,
                    "source_fingerprint": self.evidence.source_fingerprint}
        @property
        def metrics(self):
            return {"query_count": len(self.calls)}
        def load(self):
            self.loaded += 1
        def search(self, query, limit=6):
            self.calls.append((query, limit))
            return self.evidence.passages[:limit]
        def close(self):
            self.closed = True
    monkeypatch.setattr(research_hybrid, "HybridResearchRetriever", Retriever)
    corpus = ResearchCorpus(corpus_files[0], corpus_files[1], retrieval="hybrid")
    corpus.prepare()
    corpus.prepare()
    left = corpus.new_session("Dieser Vorgang muss geprüft werden.").initial_sources()
    right = corpus.new_session("Dieser Vorgang muss geprüft werden.").initial_sources()
    assert left == right and instances[0].loaded == 1
    assert corpus.retrieval_metadata()["mode"] == "hybrid"
    assert corpus.retrieval_metrics["query_count"] == 3
    assert corpus.identity_metadata()["retrieval_fingerprint"] == "retrieval-generation"
    assert all(item["text"] == corpus_files[2][item["norm_id"]][item["source_start"]:item["source_end"]]
               for item in left)
    corpus.close()
    assert instances[0].closed


@pytest.mark.parametrize("options", [{"retrieval": "missing"}, {"encoder_model_path": Path("unused")}])
def test_invalid_or_unused_retrieval_options_do_not_silently_degrade(corpus_files, options):
    with pytest.raises(ResearchSourceError):
        ResearchCorpus(corpus_files[0], corpus_files[1], **options)
