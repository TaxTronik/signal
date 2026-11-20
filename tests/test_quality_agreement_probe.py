"""Exact cached inputs, fixed score conjunctions, and the original Top20 gate."""
import json

import pytest

from tools.quality_agreement_probe import AGREEMENT_RULES, load_scores, pair_digest, plan_cases, select, shortlist


def candidate(norm_id="norm:AO:1", *, qwen=.5, bge=.01, sim=.99):
    return {"norm_id": norm_id, "qwen": qwen, "bge": bge, "sim": sim}


@pytest.mark.parametrize(("qwen", "bge", "accepted"), [
    (.4999, .99, False), (.5, .009999, False), (.5, .01, True),
])
def test_agreement_uses_unrounded_model_boundaries(qwen, bge, accepted):
    assert bool(select([candidate(qwen=qwen, bge=bge)], AGREEMENT_RULES["qwen_half_bge_floor"])) == accepted


@pytest.mark.parametrize(("bge", "sim", "accepted"), [
    (.01, .549999, False), (.01, .55, True), (.009999, .99, False),
    (.1, None, True), (.099999, None, False),
])
def test_bge_weak_rescue_is_an_exact_conjunction(bge, sim, accepted):
    assert bool(select([candidate(bge=bge, sim=sim)],
                       AGREEMENT_RULES["qwen_half_bge_weak_rescue"])) == accepted


def test_all_four_rules_are_distinct_without_label_inputs():
    candidates = [candidate("floor", sim=.4), candidate("rescue"),
                  candidate("both", bge=.1), candidate("tenth", qwen=.1, bge=.1)]
    actual = {name: {n["norm_id"] for n in select(candidates, rule)}
              for name, rule in AGREEMENT_RULES.items()}
    assert actual == {"qwen_half_bge_floor": {"floor", "rescue", "both"},
                      "qwen_half_bge_weak_rescue": {"rescue", "both"},
                      "qwen_half_bge_tenth": {"both"}, "qwen_tenth_bge_tenth": {"both", "tenth"}}


def test_qwen_ranking_then_similarity_then_id_precedes_top5():
    candidates = [candidate(f"norm:{i}", bge=1. - i / 100) for i in range(6, 0, -1)]
    candidates.append(candidate("low-sim-high-qwen", qwen=.6, bge=.1, sim=.1))
    candidates.append(candidate("a-low-sim", bge=.1, sim=.1))
    selected = select(candidates, AGREEMENT_RULES["qwen_half_bge_tenth"])
    assert [n["norm_id"] for n in selected] == ["low-sim-high-qwen", "norm:1", "norm:2", "norm:3", "norm:4"]
    assert len(candidates) == 8


def test_bge_top20_precedes_any_qwen_lookup():
    pairs = [("query", f"document {i}") for i in range(21)]
    sentence = {"candidates": [candidate(f"norm:{i:02}") for i in range(21)], "pairs": pairs}
    bge = {pair_digest(pair): .2 for pair in pairs}
    selected = shortlist(sentence, bge)
    assert [n["norm_id"] for n in selected] == [f"norm:{i:02}" for i in range(20)]
    # A missing score anywhere in the full BGE pool is an error, even though
    # its Qwen pair would not have been scored after the Top20 cut.
    bge.pop(pair_digest(pairs[-1]))
    with pytest.raises(KeyError):
        shortlist(sentence, bge)


def test_cache_matches_exact_strings_and_records_original_override_order(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_text(json.dumps(["query", "document", .1]) + "\n", encoding="utf-8")
    second.write_text(json.dumps(["query", "document", .2]) + "\n" +
                      json.dumps(["query ", "document", .3]) + "\n", encoding="utf-8")
    scores, overrides = load_scores([first, second])
    assert scores[pair_digest(("query", "document"))] == .2
    assert scores[pair_digest(("query ", "document"))] == .3
    assert len(overrides) == 1 and overrides[0]["previous"] == .1
    assert pair_digest(("a", "bc")) != pair_digest(("ab", "c"))


def test_streamed_cases_preserve_unicode_newlines_and_ignore_other_configs(tmp_path):
    cases = [{"case_id": "eins", "text": "Änderung\n\"Zitat\""}, {"case_id": "zwei"}]
    path = tmp_path / "plans.json"
    path.write_text(json.dumps({"other": [{}], "passage_package_200": cases, "last": [{}]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    assert list(plan_cases(path)) == cases
    path.write_text('{"passage_package_200": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        list(plan_cases(path))
