"""Fixed one-hop citation expansion must preserve scope, identity and lifecycle."""
from types import SimpleNamespace

import pytest

from risk_layer.citation_candidates import citation_candidates
from risk_layer.passages import Passage


class Graph:
    def __init__(self, edges):
        self.edges = list(edges)
        self.reads = 0

    def _query(self, sql):
        assert "WHERE typ='verweist_auf'" in sql
        self.reads += 1
        return [{"src": src, "dst": dst, "vorhanden": present} for src, dst, present in self.edges]


def evidence(*ids, dead=()):
    def passage(nid):
        text = "(weggefallen)" if nid in dead else "Originale Quellvorschrift."
        return Passage(f"{nid}@0:{len(text)}", nid, nid.split(":")[1], nid.split(":")[2],
                       "Titel " + nid, 0, len(text), text)
    return SimpleNamespace(passages=tuple(map(passage, ids)), source_fingerprint="frozen-source")


def norm_ids(candidates):
    return [c["norm_id"] for c in candidates]


def test_both_directions_subsection_parents_and_exact_source_metadata():
    seed, outgoing, incoming = "norm:AO:1", "norm:AO:2", "norm:EStG:3"
    graph = Graph([(seed, outgoing + ":abs2:s1", 0), (incoming + ":abs1", seed, 1)])
    ev = evidence(seed, outgoing, incoming)
    result = citation_candidates(graph, ev, [seed], {seed})
    assert norm_ids(result) == [outgoing, incoming]
    assert result[0] == {"norm_id": outgoing, "zitat": "§ 2 AO", "titel": "Titel " + outgoing, "sim": None}
    assert all(set(c) == {"norm_id", "zitat", "titel", "sim"} for c in result)


def test_no_second_hop_missing_or_retired_sources_and_existing_subnodes():
    seed, neighbor, second, absent, dead = (f"norm:AO:{i}" for i in range(1, 6))
    graph = Graph([(seed, neighbor, 1), (neighbor, second, 1), (seed, absent, 0), (seed, dead, 1)])
    ev = evidence(seed, neighbor, second, dead, dead=(dead,))
    assert norm_ids(citation_candidates(graph, ev, [seed], {seed})) == [neighbor]
    assert citation_candidates(graph, ev, [seed], {seed, neighbor + ":abs2"}) == []


def test_scope_filter_precedes_cap_and_empty_scope_never_guesses_law():
    seed, ao, estg = "norm:AO:1", "norm:AO:2", "norm:EStG:3"
    graph = Graph([(seed, ao, 1), (seed, estg, 1)])
    ev = evidence(seed, ao, estg)
    assert norm_ids(citation_candidates(graph, ev, [seed], {seed}, k=1)) == [ao]
    assert norm_ids(citation_candidates(graph, ev, [seed], {seed}, k=1, laws=("EStG",))) == [estg]
    assert citation_candidates(graph, ev, [seed], {seed}, laws=("Unbekannt",)) == []


def test_first_seed_rank_precedes_id_order_and_duplicate_neighbors_are_removed():
    first, second, early, late = "norm:AO:1", "norm:AO:2", "norm:AO:3", "norm:EStG:9"
    graph = Graph([(first, late, 1), (second, early, 1), (second, late, 1), (first + ":abs1", late, 1)])
    ev = evidence(first, second, early, late)
    assert norm_ids(citation_candidates(graph, ev, [first, second], {first, second})) == [late, early]


def test_only_first_ten_supplied_seeds_and_no_refill_from_seed_eleven():
    seeds = [f"norm:AO:{i}" for i in range(1, 12)]
    neighbor = "norm:EStG:99"
    graph = Graph([(seeds[-1], neighbor, 1)])
    assert citation_candidates(graph, evidence(*seeds, neighbor), seeds, set(seeds)) == []


def test_fixed_cap50_and_stable_identity_order_for_large_neighborhood():
    seed = "norm:EStG:1"
    neighbors = [f"norm:AO:{i}" for i in range(1, 71)]
    graph = Graph([(seed, neighbor, 1) for neighbor in reversed(neighbors)])
    result = citation_candidates(graph, evidence(seed, *neighbors), [seed], {seed})
    assert norm_ids(result) == sorted(neighbors)[:50]


def test_mutable_graph_is_read_again_and_new_edges_are_visible():
    seed, first, later = "norm:AO:1", "norm:AO:2", "norm:AO:3"
    graph = Graph([(seed, first, 1)])
    ev = evidence(seed, first, later)
    assert norm_ids(citation_candidates(graph, ev, [seed], {seed})) == [first]
    graph.edges.append((seed, later, 1))
    assert norm_ids(citation_candidates(graph, ev, [seed], {seed})) == [first, later]
    assert graph.reads == 2 and not hasattr(graph, "_citation_candidates_snapshot")


def test_frozen_cache_reuses_one_snapshot_and_binds_evidence_and_lifecycle():
    seed, neighbor = "norm:AO:1", "norm:AO:2"
    graph = Graph([(seed, neighbor, 1)])
    graph._embedding_graph_fingerprint_memo = "immutable-generation-one"
    ev = evidence(seed, neighbor)
    first = citation_candidates(graph, ev, [seed], {seed})
    assert citation_candidates(graph, ev, [seed], {seed}) == first
    assert graph.reads == 1
    replacement = evidence(seed)
    assert citation_candidates(graph, replacement, [seed], {seed}) == []
    assert graph.reads == 2
    graph._embedding_graph_fingerprint_memo = "immutable-generation-two"
    citation_candidates(graph, replacement, [seed], {seed})
    assert graph.reads == 3


@pytest.mark.parametrize("k", [0, 51, -1, True, 1.5])
def test_bad_budgets_fail_before_graph_reads(k):
    graph = Graph([])
    with pytest.raises(ValueError, match="cap"):
        citation_candidates(graph, evidence("norm:AO:1"), ["norm:AO:1"], set(), k=k)
    assert graph.reads == 0


def test_empty_seeds_need_no_graph_read():
    graph = Graph([])
    assert citation_candidates(graph, evidence("norm:AO:1"), [], set()) == []
    assert graph.reads == 0
