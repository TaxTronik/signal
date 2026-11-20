"""Model-free safeguards for the independently executed real GPU validator."""
from copy import deepcopy
from dataclasses import asdict
import json
import subprocess
import sys

import pytest

from tools._gpu_eval_metrics import evaluate_variant, summarize
from tools.quality_recall_live import (
    compare, json_value, renamed_reference, validate_public_profile, validate_reference,
)


def output():
    return {"suggestions": [{"start": 0, "end": 4, "satz": "TEST", "profil": "public",
        "kontext": {"start": 0, "end": 4, "text": "TEST"}, "normen": [
            {"id": "norm:TEST:1", "sim": .8, "rerank": .9, "ketten": [], "evidenz": {"fingerprint": "source"}},
            {"id": "norm:TEST:2", "sim": .7, "rerank": .8, "ketten": [], "evidenz": {"fingerprint": "source"}}]}],
        "cards": [], "trigger_marks": []}


def reference():
    rows = []
    for dataset, positive, negative in (("main", 102, 30), ("additional", 15, 15)):
        for n in range(positive + negative):
            rows.append({"case_id": f"{dataset}:{n}", "dataset": dataset, "group": "TEST",
                "query": "Synthetic technical test", "negative": n >= positive,
                "targets": ["norm:TEST:1"] if n < positive else [], "category": None, "uncertain": False,
                "chosen": {"suggestions": []}})
    return {"format": "quality_passage_rerank_probe/1", "status": "complete", "rows": rows,
        "variants": {"chosen": {"cascade": True}}, "models": {k: {} for k in ("encoder", "bge", "qwen")},
        "summaries": {d: {"chosen": summarize([r for r in rows if r["dataset"] == d], "chosen")}
                      for d in ("main", "additional")}}


def test_complete_case_reference_accepted_and_incomplete_pilot_refused():
    report = reference()
    validate_reference(report, "chosen")
    report["rows"] = report["rows"][:61]
    with pytest.raises(ValueError, match="162"):
        validate_reference(report, "chosen")


@pytest.mark.parametrize("mutation", ("running", "failed", "subset", "missing_variant", "wrong_denominator"))
def test_invalid_selection_fails_before_gpu(mutation):
    report = reference()
    if mutation in ("running", "failed"):
        report["status"] = mutation
    elif mutation == "subset":
        report["diagnostic_subset"] = True
    elif mutation == "missing_variant":
        report["variants"] = {}
    else:
        report["rows"][0]["negative"] = True
    with pytest.raises(ValueError):
        validate_reference(report, "chosen")


def test_public_profile_must_exist_and_keep_every_selected_setting():
    config = {"cascade": True, "shortlist": 50, "instruction": "fixed"}
    validate_public_profile("public", config, ("standard", "public"), {"public": config})
    for name, actual in (("research_only", config), ("standard", config),
                         ("public", {**config, "shortlist": 20})):
        with pytest.raises(ValueError):
            validate_public_profile(name, config, ("standard", "public"), {name: actual})


def test_only_reference_profile_tag_is_renamed_without_mutation_or_hidden_metadata():
    original = output()["suggestions"]
    original[0]["profil"] = "chosen"
    renamed = renamed_reference(original, "chosen", "public")
    assert renamed[0]["profil"] == "public" and original[0]["profil"] == "chosen"
    assert renamed[0]["normen"] == original[0]["normen"]
    with pytest.raises(ValueError, match="profile"):
        renamed_reference(original, "wrong", "public")


@pytest.mark.parametrize(("mutation", "category"), (
    ("score", "score_only"), ("order", "order"), ("selection", "selection"),
    ("context", "context"), ("hard", "hard_marks"), ("chain", "metadata_other"),
))
def test_difference_categories_keep_selection_and_metadata_distinct(mutation, category):
    expected, actual = output(), output()
    sentence = actual["suggestions"][0]
    if mutation == "score":
        sentence["normen"][0]["rerank"] = .89
    elif mutation == "order":
        sentence["normen"].reverse()
    elif mutation == "selection":
        sentence["normen"][1]["id"] = "norm:TEST:3"
    elif mutation == "context":
        sentence["kontext"]["end"] = 5
    elif mutation == "hard":
        actual["cards"] = [{"id": "additional"}]
    else:
        sentence["normen"][0]["ketten"] = [{"glieder": ["TEST"]}]
    result = compare(actual, expected, ["norm:TEST:1"], evaluate_variant)
    assert category in result["categories"] and not result["all_equal"]
    assert result["actual_metrics"] == evaluate_variant(actual["suggestions"], ["norm:TEST:1"])


def test_identical_output_preserves_all_comparison_fields():
    result = compare(output(), deepcopy(output()), ["norm:TEST:1"], evaluate_variant)
    assert result["all_equal"] and result["categories"] == [] and result["selection_equal"]


def test_actual_dataclass_report_serializes_without_dropping_enum_fields(monkeypatch):
    monkeypatch.setenv("RISK_LAYER_NER", "aus")
    from risk_layer.pipeline import analyse
    report = analyse("Ein Gestaltungsmissbrauch liegt vor.")
    raw = json.loads(json.dumps(asdict(report), default=json_value))
    assert raw["karten"] and raw["karten"][0]["eintrag"]["governance_typ"]
    assert raw["text_hash"] == report.text_hash


def test_tool_import_does_not_import_torch_or_production_runtime():
    result = subprocess.run([sys.executable, "-c", "import sys; import tools.quality_recall_live; "
        "assert 'torch' not in sys.modules; assert 'risk_layer' not in sys.modules"], check=False, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
