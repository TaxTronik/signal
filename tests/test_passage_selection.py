"""Independent corpus views must retain exact source identity and raw ranking."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from risk_layer.passage_selection import lexical_candidates, package_ranked_passages, select_passage_candidates
from risk_layer.passages import Passage


def passage(source, start=0, end=None, norm_id="norm:AO:146", title=""):
    end = len(source) if end is None else end
    return Passage(f"{norm_id}@{start}:{end}", norm_id, norm_id.split(":")[1],
                   norm_id.split(":")[2], title, start, end, source[start:end])


def evidence(passages, similarities):
    vectors = np.array([[s, 0.] for s in similarities], dtype=np.float32)
    by_norm = {}
    for i, p in enumerate(passages):
        by_norm.setdefault(p.norm_id, []).append(i)
    return SimpleNamespace(passages=tuple(passages), vectors=vectors,
                           _by_norm=by_norm,
                           _query=lambda vector: vector)


def test_candidate_pool_preserves_first_semantic_and_lexical_tail_without_encoding():
    segments = ["Erste allgemeine Aufzeichnungsregel. "] + [f"Sonderregel zum Verfahren {i}. " for i in range(8)]
    segments.append("Unterhaltszahlungen sind hier als eigener Sachverhalt beschrieben.")
    source = "".join(segments)
    positions = np.cumsum([0] + list(map(len, segments))).tolist()
    rows = [passage(source, a, b) for a, b in zip(positions, positions[1:])]
    # Distinct numeric source content must not be replaced by fabricated text.
    ev = evidence(rows, [.0, .99, .98, .97, .96, .95, .94, .93, .92, .01])
    before = ev.vectors.copy()
    selected = select_passage_candidates(ev, rows[0].norm_id, np.array([1., 0.]), "Unterhaltszahlungen", 4)
    assert selected[0] == rows[0]
    assert rows[1] in selected and rows[-1] in selected
    assert len(selected) == 4
    assert all(p in rows and source[p.start:p.end] == p.text for p in selected)
    stats = ev._selection_lexical_stats
    assert select_passage_candidates(ev, rows[0].norm_id, np.array([1., 0.]), "Unterhaltszahlungen", 4) == selected
    assert ev._selection_lexical_stats is stats
    assert np.array_equal(before, ev.vectors)
    assert select_passage_candidates(ev, "norm:AO:99999", np.array([1., 0.]), "Quelle") == []


def test_titleless_umlaut_equivalence_and_source_ties_are_deterministic():
    source = "Erste Regel. Andere Vorschrift. Weitere Norm. Späte Prüfungspflicht."
    points = [0, source.index("Andere"), source.index("Weitere"), source.index("Späte"), len(source)]
    rows = [passage(source, a, b) for a, b in zip(points, points[1:])]
    ev = evidence(rows, [0., .8, .7, .1])
    selected = select_passage_candidates(ev, rows[0].norm_id, np.array([1., 0.]), "Pruefungspflicht", 3)
    assert selected == [rows[0], rows[1], rows[3]]
    assert all(not p.title for p in selected)


@pytest.mark.parametrize("limit", [0, 9, -1, True, 2.5])
def test_candidate_budget_rejects_unsupported_values(limit):
    row = passage("Tatsächlicher Quelltext.")
    with pytest.raises(ValueError, match="limit"):
        select_passage_candidates(evidence([row], [1.]), row.norm_id, np.array([1., 0.]), "Quelle", limit)


def test_crossencoder_scores_select_later_rule_and_trim_overlap_in_either_direction():
    source = "Erste allgemeine Regel. Gemeinsamer Mittelteil.\n\nSpäte spezielle Regel."
    start = source.index("Gemeinsamer")
    middle_end = source.index("\n\n")
    earlier, later = passage(source, 0, middle_end), passage(source, start)
    package = package_ranked_passages([earlier, later], [.49991, .49999], [], "frozen-source")
    assert package.evidence[0] == later  # Raw scores must not round into a tie.
    assert package.evidence[1].start == 0 and package.evidence[1].end <= later.start
    assert package.text.splitlines()[0] == later.norm_id
    assert package.text.count("Gemeinsamer Mittelteil.") == 1
    assert package.source_fingerprint == "frozen-source"
    for p in package.evidence:
        assert source[p.start:p.end] == p.text
    reverse = package_ranked_passages([earlier, later], [.9, .1], [], "frozen-source")
    assert reverse.evidence[0] == earlier and reverse.evidence[1].start > middle_end


def test_contained_duplicate_views_do_not_displace_second_novel_source():
    source = "Erste Regel. Noch dieselbe Regel.\n\nSpätere eigenständige Vorschrift."
    split = source.index("\n\n")
    first = passage(source, 0, split)
    contained = passage(source, 0, source.index("Noch"))
    later = passage(source, split + 2)
    package = package_ranked_passages([first, contained, first, later], [.9, .8, .7, .6], [], "fp")
    assert package.evidence == (first, later)


def test_large_surrounding_view_keeps_one_contiguous_novel_interval():
    source = "Linke Quelle ist lang. MITTE. Rechte Quelle ist noch deutlich länger."
    inner = passage(source, source.index("MITTE"), source.index(" Rechte"))
    outer = passage(source)
    package = package_ranked_passages([inner, outer], [.9, .8], [], "fp")
    assert len(package.evidence) == 2
    assert package.evidence[1].start > inner.end
    for p in package.evidence:
        assert source[p.start:p.end] == p.text


def test_budget_and_anchors_report_only_exact_present_sources():
    source = "Tatsächliche Quelle mit Belegen. " * 30
    row = passage(source, title="Lange Überschrift " * 100)
    package = package_ranked_passages([row], [.9], ["Anker eins", "Anker eins", "Anker zwei", "Anker drei"], "fp", max_chars=180)
    assert len(package.text) <= 180 and package.truncated
    assert len(package.evidence) == 1
    actual = package.evidence[0]
    assert source[actual.start:actual.end] == actual.text and actual.text in package.text
    assert package.anchors == ()
    short = package_ranked_passages([passage("Originale kurze Quelle.")], [.9],
                                   ["Anker eins", "Anker eins", "Anker zwei", "Anker drei"], "fp")
    assert short.anchors == ("Anker eins", "Anker zwei")


@pytest.mark.parametrize("scores", [[True], [float("nan")], [float("inf")], ["0.9"], []])
def test_bad_scores_never_produce_supported_packages(scores):
    with pytest.raises(ValueError):
        package_ranked_passages([passage("Originale Quelle.")], scores, [], "fp")


def test_foreign_norm_and_conflicting_source_offsets_are_rejected():
    first = passage("Originale Quelle.")
    foreign = passage("Andere Quelle.", norm_id="norm:AO:147")
    contradictory = passage("Erfundene Quelle.")
    for bad in (foreign, contradictory):
        with pytest.raises(ValueError):
            package_ranked_passages([first, bad], [.9, .8], [], "fp")
    with pytest.raises(ValueError):
        package_ranked_passages([], [], [], "fp")


def test_lexical_scope_precedes_cap_and_norms_are_unique_without_fake_similarity():
    ao = passage("Besondere Dokumentationspflicht.", title="Dokumentationspflicht")
    estg = passage("Dokumentationspflicht für besondere Vorgänge.", norm_id="norm:EStG:18", title="Quellen")
    ev = evidence([ao, ao, estg], [1., 1., 1.])
    both = lexical_candidates(ev, "Dokumentationspflicht", k=2)
    assert len(both) == 2 and {c["norm_id"] for c in both} == {ao.norm_id, estg.norm_id}
    assert all(c["sim"] is None for c in both)
    second_law = both[1]["norm_id"].split(":")[1]
    assert lexical_candidates(ev, "Dokumentationspflicht", k=1, laws=(second_law,)) == [both[1]]
    assert lexical_candidates(ev, "Dokumentationspflicht", laws=("Unbekannt",)) == []
    assert lexical_candidates(ev, "NichtvorhandenesQuellwort") == []
    assert lexical_candidates(ev, "... 123 und oder") == []


def test_lexical_norm_score_is_maximum_instead_of_sum_of_many_weak_passages():
    weak = passage("Dokumentation " + "Andere bekannte Vorgänge " * 25)
    strong = passage("Dokumentation " * 10, norm_id="norm:EStG:18")
    for copies in (1, 40):
        ev = evidence([weak] * copies + [strong], [0.] * (copies + 1))
        assert lexical_candidates(ev, "Dokumentation", k=1)[0]["norm_id"] == strong.norm_id


def test_lexical_equal_score_tie_uses_norm_identity_and_shares_view_statistics():
    source = "Quellenprüfung ist vorzunehmen."
    ao = passage(source)
    estg = passage(source, norm_id="norm:EStG:18")
    ev = evidence([estg, ao], [0., 0.])
    assert [c["norm_id"] for c in lexical_candidates(ev, "Quellenpruefung")] == [ao.norm_id, estg.norm_id]
    cached = ev._selection_lexical_stats
    select_passage_candidates(ev, estg.norm_id, np.array([1., 0.]), "Quellenpruefung")
    assert ev._selection_lexical_stats is cached


def test_unicode_and_crlf_in_overlapping_source_views_remain_exact_after_clipping():
    source = "Erster\u00a0Abschnitt.\r\nGemeinsame Angaben.\r\nZweiter\u202fAbschnitt."
    first = passage(source, 0, source.index("Zweiter"))
    second = passage(source, source.index("Gemeinsame"))
    package = package_ranked_passages([first, second], [.7, .8], [], "original-fingerprint")
    assert len(package.evidence) == 2
    assert package.text.count("Gemeinsame Angaben.") == 1
    assert "\u00a0" in package.text and "\u202f" in package.text
    assert all(source[p.start:p.end] == p.text for p in package.evidence)


@pytest.mark.parametrize("change", [{"start": False}, {"end": 26.0}])
def test_noninteger_offsets_fail_with_explicit_validation_error(change):
    original = passage("Ein echter Quellabschnitt.")
    with pytest.raises(ValueError, match="source lengths"):
        package_ranked_passages([replace(original, **change)], [.8], [], "fp")


def test_foreign_malformed_and_non_enclosing_parent_ids_are_rejected():
    original = passage("Ein echter Quellabschnitt.")
    bad_ids = ["norm:AO:147@0:999", "norm:AO:146@1:999", "norm:AO:146@0:3", "ungueltig"]
    for bad_id in bad_ids:
        with pytest.raises(ValueError, match="passage ID"):
            package_ranked_passages([replace(original, passage_id=bad_id)], [.8], [], "fp")


def test_clipped_view_may_retain_its_larger_original_parent_span():
    source = "Einleitung. Ein eigenstaendiger Quellabschnitt. Abschluss."
    parent = passage(source)
    start, end = source.index("Ein eigenstaendiger"), source.index(" Abschluss")
    clipped = replace(parent, start=start, end=end, text=source[start:end])
    package = package_ranked_passages([clipped], [.8], [], "fp")
    assert package.evidence == (clipped,)
    assert package.evidence[0].passage_id == parent.passage_id
