"""Small integrity tests for synthetic annotations, not automated legal adjudication."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from experiments.grounded_signal.cases import (
    ANNOTATION_STATUS,
    SOURCE_GRAPH_SHA256,
    dataset_manifest,
    load_cases,
)

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "out/gpu-ab-20260907-final/artifacts/graph.sqlite"
LABEL_GROUPS = (
    "target_norm_ids", "allowed_hint_norm_ids", "no_hint_norm_ids", "insufficient_context_norm_ids",
)


def test_synthetic_status_full_denominators_and_family_disjoint_splits():
    cases = load_cases()
    assert len(cases) == 16
    assert len({case["case_id"] for case in cases}) == 16
    assert len({case["text"] for case in cases}) == 16
    assert Counter(case["split"] for case in cases) == {"development": 8, "evaluation": 8}
    families = defaultdict(list)
    for case in cases:
        families[case["family_id"]].append(case)
        assert case["diagnostic_labels"]["annotation_status"] == ANNOTATION_STATUS == "assistant_reviewed_synthetic"
        assert case["text"].strip() == case["text"] and 40 <= len(case["text"]) <= 600
        assert "norm:" not in case["text"]
    assert len(families) == 8
    for pair in families.values():
        assert len(pair) == 2 and len({case["split"] for case in pair}) == 1
        assert {case["diagnostic_labels"]["variant"] for case in pair} == {"anchor", "contrast"}
        assert pair[0]["prepared_norm_ids"] == pair[1]["prepared_norm_ids"]
    assert {c["family_id"] for c in load_cases("development")}.isdisjoint(
        {c["family_id"] for c in load_cases("evaluation")}
    )


def test_labels_have_explicit_disjoint_candidates_and_prose_review_limits():
    for case in load_cases():
        assert set(case) == {"case_id", "family_id", "split", "text", "prepared_norm_ids", "diagnostic_labels"}
        prepared = case["prepared_norm_ids"]
        assert 2 <= len(prepared) <= 3 and len(set(prepared)) == len(prepared)
        labels = case["diagnostic_labels"]
        groups = [set(labels[name]) for name in LABEL_GROUPS]
        for group, name in zip(groups, LABEL_GROUPS):
            assert group <= set(prepared) and len(group) == len(labels[name])
        assert set.union(*groups) == set(prepared)
        for i, group in enumerate(groups):
            assert all(group.isdisjoint(other) for other in groups[i + 1:])
        assert labels["hint_expected"] is bool(labels["target_norm_ids"])
        assert labels["source_based_rationale"]
        assert labels["prohibited_factual_assertions"]
        for field in ("prohibited_factual_assertions", "expected_open_conditions"):
            assert all(isinstance(value, str) and value.strip() for value in labels[field])
        for support in labels["source_based_rationale"]:
            assert support["norm_id"] in prepared
            assert support["source_end"] - support["source_start"] == len(support["source_quote"])
            assert support["source_start"] >= 0
            assert len(support["source_text_sha256"]) == 64


def test_negative_and_unknown_context_do_not_lose_the_prepared_distractors():
    cases = load_cases()
    no_expected_hint = [c for c in cases if not c["diagnostic_labels"]["hint_expected"]]
    assert len(no_expected_hint) == 4
    for case in no_expected_hint:
        sibling = next(c for c in cases if c["family_id"] == case["family_id"] and c["case_id"] != case["case_id"])
        assert sibling["diagnostic_labels"]["target_norm_ids"]
        assert case["prepared_norm_ids"] == sibling["prepared_norm_ids"]
    unknown = next(c for c in no_expected_hint if c["family_id"] == "representative_role")["diagnostic_labels"]
    assert set(unknown["insufficient_context_norm_ids"]) == {"norm:AO:34", "norm:AO:69"}
    assert unknown["no_hint_norm_ids"] == ["norm:AO:158"]


def test_negation_and_exception_contrasts_keep_relevant_norms():
    by_id = {case["case_id"]: case for case in load_cases()}
    for family, target in (("ownership_negation", "norm:AO:39"),
                           ("foreign_substance_scope", "norm:AStG:8")):
        a, b = (by_id[f"gs-{family}-{variant}"] for variant in ("anchor", "contrast"))
        assert target in a["diagnostic_labels"]["target_norm_ids"]
        assert target in b["diagnostic_labels"]["target_norm_ids"]
        assert a["diagnostic_labels"]["prohibited_factual_assertions"] != b["diagnostic_labels"]["prohibited_factual_assertions"]
    assert "nicht für die gewöhnliche Nutzungsdauer" in by_id["gs-ownership_negation-contrast"]["text"]
    assert "außerhalb von EU und EWR" in by_id["gs-foreign_substance_scope-contrast"]["text"]
    cash = by_id["gs-cash_exception-contrast"]["diagnostic_labels"]
    assert cash["target_norm_ids"] == ["norm:AO:146", "norm:AO:146a"]


def test_correction_origin_does_not_label_a_possible_related_norm_as_false():
    by_id = {case["case_id"]: case for case in load_cases()}
    authority = by_id["gs-correction_origin-anchor"]["diagnostic_labels"]
    declaration = by_id["gs-correction_origin-contrast"]["diagnostic_labels"]
    assert authority["target_norm_ids"] == ["norm:AO:129"]
    assert "norm:AO:173a" in authority["no_hint_norm_ids"]
    assert declaration["target_norm_ids"] == ["norm:AO:173a"]
    assert "norm:AO:129" in declaration["allowed_hint_norm_ids"]
    assert "norm:AO:129" not in declaration["no_hint_norm_ids"]


def test_defensive_load_and_manifest_cannot_mutate_other_cases_or_subsequent_calls():
    original = load_cases()
    altered = load_cases()
    altered[0]["prepared_norm_ids"].clear()
    altered[0]["diagnostic_labels"]["source_based_rationale"][0]["source_quote"] = "changed"
    assert altered[1]["prepared_norm_ids"] == original[1]["prepared_norm_ids"]
    assert load_cases() == original
    manifest = dataset_manifest()
    manifest["family_splits"]["development"].clear()
    assert len(dataset_manifest()["family_splits"]["development"]) == 4


@pytest.mark.parametrize("split", ["train", "eval", "", "blind", 1, []])
def test_unknown_split_is_rejected(split):
    with pytest.raises(ValueError, match="split"):
        load_cases(split)


def test_manifest_binds_records_and_discloses_known_synthetic_status():
    manifest = dataset_manifest()
    raw = json.dumps(load_cases(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert manifest["dataset_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["source_graph_sha256"] == SOURCE_GRAPH_SHA256
    assert manifest["case_count"] == 16 and manifest["family_count"] == 8
    assert manifest["synthetic"] is True and manifest["known_failure_patterns"] is True
    assert manifest["blind_evaluation"] is False and manifest["expert_adjudicated"] is False
    assert manifest["limits"]


@pytest.mark.skipif(not GRAPH.is_file(), reason="Optional frozen local corpus is not distributed with synthetic cases")
def test_all_authored_rule_quotes_and_family_budgets_match_the_actual_frozen_graph():
    assert hashlib.sha256(GRAPH.read_bytes()).hexdigest() == SOURCE_GRAPH_SHA256
    with sqlite3.connect(GRAPH.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        norms = {}
        for case in load_cases():
            for nid in case["prepared_norm_ids"]:
                if nid not in norms:
                    row = connection.execute("SELECT text FROM knoten WHERE id=?", (nid,)).fetchone()
                    assert row is not None and row[0]
                    norms[nid] = row[0]
            assert sum(len(norms[nid]) for nid in case["prepared_norm_ids"]) <= 32000
            for support in case["diagnostic_labels"]["source_based_rationale"]:
                text = norms[support["norm_id"]]
                assert hashlib.sha256(text.encode()).hexdigest() == support["source_text_sha256"]
                assert text[support["source_start"]:support["source_end"]] == support["source_quote"]
    foreign = next(c for c in load_cases() if c["family_id"] == "foreign_substance_scope")
    rule = next(s for s in foreign["diagnostic_labels"]["source_based_rationale"] if s["norm_id"] == "norm:AStG:8")
    assert rule["source_start"] == 6488
    assert "(2) Ungeachtet" in rule["source_quote"] and "(3) Absatz 2 gilt nur" in rule["source_quote"]
    assert "(4) Die Absätze 2 und 3 gelten nicht" in rule["source_quote"]


@pytest.mark.skipif(not (ROOT / "out/quality-research-moe-additional-20260908-r1/report.json").is_file(),
                    reason="Optional private historical report is not a runtime dataset dependency")
def test_synthetic_case_texts_are_not_copied_historical_queries():
    old_paths = [ROOT / "out/quality-research-moe-additional-20260908-r1/report.json",
                 ROOT / "out/quality-research-moe-pilot-20260908-r3/report.json"]
    previous = set()
    for path in old_paths:
        if path.is_file():
            previous.update(row["query"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"])
    assert all(case["text"] not in previous for case in load_cases())
