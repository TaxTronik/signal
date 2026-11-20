"""Human labels remain separate from frozen predictions and historical targets."""
from copy import deepcopy
import json

import pytest

from tools.quality_adjudication import (
    collect_exports, encoded, evaluate_bundle, export_reports, export_train_pairs, main, read_json, sha,
)


def report_row(case_id="one", query="Alpha", norms=("norm:TEST:14", "norm:TEST:140"), **extra):
    return {"case_id": case_id, "dataset": "historical", "group": "fixture", "query": query,
            "targets": ["norm:TEST:14"], "negative": False, "uncertain": True,
            "kaskade": {"suggestions": [{"start": 0, "end": len(query), "satz": query,
                                          "normen": [{"id": norm, "titel": "Synthetic fixture", "rerank": .6,
                                                      "evidenz": {"text": "Archived source excerpt"}} for norm in norms]}]
                         if norms else []}, **extra}


def write_report(tmp_path, rows=None, name="source.json"):
    path = tmp_path / name
    path.write_bytes(encoded({"format": "test_fixture/1", "status": "pass", "profiles": ["kaskade"],
                             "source_hashes": {"core.py": "preserved"}, "rows": rows or [report_row()]}))
    return path


@pytest.fixture
def exported(tmp_path):
    source = write_report(tmp_path)
    out = tmp_path / "review"
    bundle = export_reports([source], out)
    return out, bundle, read_json(out / "labels.json")


def evaluate(out, labels):
    (out / "labels.json").write_bytes(encoded(labels))
    return evaluate_bundle(out / "bundle.json", out / "labels.json")


def adjudicate_case(labels, targets=("norm:TEST:14",), complete=True):
    labels["cases"][0].update(status="adjudicated", expected_targets=list(targets),
                              expected_targets_complete=complete, adjudicator="reviewer-1")


def label_pairs(labels, *values):
    for entry, value in zip(labels["suggestions"], values, strict=True):
        entry.update(label=value, adjudicator="reviewer-1" if value else None)


def test_export_preserves_outputs_but_never_manufactures_labels(exported):
    out, bundle, labels = exported
    assert bundle["cases"][0]["historical_metadata"]["targets"] == ["norm:TEST:14"]
    assert bundle["cases"][0]["historical_metadata"]["uncertain"] is True
    assert bundle["suggestions"][0]["norm"]["evidenz"] == {"text": "Archived source excerpt"}
    assert labels["cases"][0]["expected_targets"] is None
    assert all(entry["label"] is None for entry in labels["suggestions"])
    assert bundle["split_audit"]["independence_certified"] is False
    result = evaluate(out, labels)
    metric = result["metrics"][0]
    assert result["status"] == "awaiting_adjudication"
    assert metric["precision_applicable_reviewed"] is None
    assert metric["recall_applicable_adjudicated"] is None
    assert metric["counts"]["unlabeled"] == 2


def test_only_explicit_adjudicated_targets_enable_recall(exported):
    out, _, labels = exported
    label_pairs(labels, "applicable", "not_applicable")
    metric = evaluate(out, labels)["metrics"][0]
    assert metric["precision_applicable_reviewed"] == .5
    assert metric["recall_applicable_adjudicated"] is None
    adjudicate_case(labels)
    result = evaluate(out, labels)
    assert result["status"] == "adjudicated"
    assert result["metrics"][0]["recall_applicable_adjudicated"] == 1


@pytest.mark.parametrize("unresolved", [None, "insufficient_context"])
def test_unknown_and_insufficient_are_not_false_labels(exported, unresolved):
    out, _, labels = exported
    adjudicate_case(labels)
    label_pairs(labels, "applicable", unresolved)
    result = evaluate(out, labels)
    metric = result["metrics"][0]
    assert result["status"] == "partial_adjudication"
    assert metric["precision_applicable_reviewed"] == 1
    assert metric["review_coverage"] == .5
    assert metric["recall_applicable_adjudicated"] is None
    assert metric["counts"]["recall_excluded_unresolved_suggestions"] == 1


def test_missing_label_entries_remain_unknown(exported):
    out, _, labels = exported
    label_pairs(labels, "applicable", "not_applicable")
    labels["suggestions"].pop(0)
    labels["cases"].clear()
    result = evaluate(out, labels)
    assert result["missing_case_label_entries"] == 1
    assert result["missing_suggestion_label_entries"] == 1
    assert result["metrics"][0]["counts"]["unlabeled"] == 1
    assert result["metrics"][0]["review_coverage"] == .5


def test_helpful_definition_has_separate_precision_and_is_not_applicable_recall(exported):
    out, _, labels = exported
    adjudicate_case(labels)
    label_pairs(labels, "helpful_definition", "contradicted")
    metric = evaluate(out, labels)["metrics"][0]
    assert metric["precision_applicable_reviewed"] == 0
    assert metric["precision_helpful_reviewed"] == .5
    assert metric["recall_applicable_adjudicated"] == 0


def test_partial_expected_targets_do_not_enable_recall(exported):
    out, _, labels = exported
    adjudicate_case(labels, complete=False)
    label_pairs(labels, "applicable", "not_applicable")
    metric = evaluate(out, labels)["metrics"][0]
    assert metric["recall_applicable_adjudicated"] is None
    assert metric["counts"]["recall_excluded_no_complete_targets"] == 1


@pytest.mark.parametrize("targets,recall", [(["norm:TEST:14"], 0), ([], None)])
def test_empty_prediction_cases_are_retained_for_recall(tmp_path, targets, recall):
    source = write_report(tmp_path, [report_row(norms=())])
    out = tmp_path / "review"
    export_reports([source], out)
    labels = read_json(out / "labels.json")
    adjudicate_case(labels, targets)
    result = evaluate(out, labels)
    assert result["status"] == "adjudicated"
    metric = result["metrics"][0]
    assert metric["recall_applicable_adjudicated"] == recall
    assert metric["precision_applicable_reviewed"] is None
    assert metric["counts"]["cases"] == 1


def test_norm_matching_has_a_boundary_and_rejects_inconsistent_complete_targets(exported):
    out, _, labels = exported
    adjudicate_case(labels)
    label_pairs(labels, "not_applicable", "applicable")
    with pytest.raises(ValueError, match="absent from declared complete"):
        evaluate(out, labels)


def test_target_descendants_match_without_double_counting_sentences(tmp_path):
    row = report_row(norms=("norm:TEST:14:abs1",), query="Alpha Beta")
    row["kaskade"]["suggestions"] = [
        {"start": 0, "end": 5, "satz": "Alpha", "normen": [{"id": "norm:TEST:14:abs1"}]},
        {"start": 6, "end": 10, "satz": "Beta", "normen": [{"id": "norm:TEST:14:abs2"}]},
    ]
    source, out = write_report(tmp_path, [row]), tmp_path / "review"
    bundle = export_reports([source], out)
    assert len({item["suggestion_id"] for item in bundle["suggestions"]}) == 2
    labels = read_json(out / "labels.json")
    adjudicate_case(labels)
    label_pairs(labels, "applicable", "applicable")
    metric = evaluate(out, labels)["metrics"][0]
    assert metric["found_targets"] == metric["expected_targets"] == 1
    assert metric["counts"]["displayed_suggestions"] == 2


@pytest.mark.parametrize("field,value,match", [
    ("label", "unknown", "Unknown suggestion label"),
    ("adjudicator", None, "adjudicator"),
    ("norm_id", "norm:TEST:15", "suggestion context"),
    ("rank", 9, "suggestion context"),
])
def test_invalid_pair_labels_fail(exported, field, value, match):
    out, _, labels = exported
    label_pairs(labels, "applicable", "not_applicable")
    labels["suggestions"][0][field] = value
    with pytest.raises(ValueError, match=match):
        evaluate(out, labels)


@pytest.mark.parametrize("targets", [["norm:TEST:14", "norm:TEST:14"],
                                     ["norm:TEST:14", "norm:TEST:14:abs1"], ["bad id"]])
def test_duplicate_overlapping_or_invalid_expected_targets_fail(exported, targets):
    out, _, labels = exported
    adjudicate_case(labels, targets)
    with pytest.raises(ValueError):
        evaluate(out, labels)


def test_unreviewed_case_cannot_smuggle_targets(exported):
    out, _, labels = exported
    labels["cases"][0]["expected_targets"] = ["norm:TEST:14"]
    with pytest.raises(ValueError, match="Unreviewed"):
        evaluate(out, labels)


def test_duplicate_and_foreign_label_ids_fail(exported):
    out, _, labels = exported
    labels["suggestions"].append(deepcopy(labels["suggestions"][0]))
    with pytest.raises(ValueError, match="duplicate label ID"):
        evaluate(out, labels)
    labels["suggestions"][-1]["suggestion_id"] = "not-in-export"
    with pytest.raises(ValueError, match="Unknown"):
        evaluate(out, labels)


@pytest.mark.parametrize("mutate", ["hash", "source", "prediction", "case"])
def test_modified_provenance_is_rejected_even_with_recomputed_bundle_hash(exported, mutate):
    out, bundle, labels = exported
    if mutate == "source":
        (out / bundle["sources"][0]["frozen_path"]).write_text("{}", encoding="utf-8")
    elif mutate == "prediction":
        bundle["suggestions"][0]["rank"] = 9
    elif mutate == "case":
        bundle["cases"][0]["query"] = "Invented context"
    else:
        labels["bundle_sha256"] = "wrong"
    if mutate in ("prediction", "case"):
        (out / "bundle.json").write_bytes(encoded(bundle))
        labels["bundle_sha256"] = sha((out / "bundle.json").read_bytes())
    with pytest.raises(ValueError):
        evaluate(out, labels)


@pytest.mark.parametrize("mode", ["document", "group", "text"])
def test_declared_document_group_and_normalized_query_split_leakage_fail(tmp_path, mode):
    rows = [report_row("one", "Alpha"), report_row("two", "Beta", group="other")]
    assignments = []
    for i, row in enumerate(rows):
        assignments.append({"dataset": row["dataset"], "case_id": row["case_id"],
                            "document_id": "shared" if mode == "document" else f"doc-{i}",
                            "split_group_id": "shared" if mode == "group" else f"group-{i}",
                            "split": "development" if i == 0 else "test"})
    if mode == "text":
        rows[1] = report_row("two", "  ALPHA  ", group="other")
    source = write_report(tmp_path, rows)
    mapping = {(entry["dataset"], entry["case_id"]): entry for entry in assignments}
    with pytest.raises(ValueError, match="Split leakage"):
        collect_exports([source], assignments=mapping)


def test_document_split_assignments_are_frozen_and_group_metrics_separate(tmp_path):
    rows = [report_row("one", "Alpha"), report_row("two", "Beta")]
    source = write_report(tmp_path, rows)
    splits, out = tmp_path / "splits.json", tmp_path / "review"
    splits.write_bytes(encoded([{"dataset": "historical", "case_id": row["case_id"],
                                "document_id": f"doc-{i}", "split": "development" if i == 0 else "test"}
                               for i, row in enumerate(rows)]))
    bundle = export_reports([source], out, splits=splits)
    assert bundle["split_audit"]["assigned_cases"] == 2
    result = evaluate(out, read_json(out / "labels.json"))
    assert {row["split"] for row in result["metrics"]} == {"development", "test"}
    (out / "inputs/split-assignments.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="split assignments"):
        evaluate_bundle(out / "bundle.json", out / "labels.json")


def test_conservative_group_fallback_prevents_random_case_split(tmp_path):
    source = write_report(tmp_path, [report_row("one", "Alpha", split="train"),
                                    report_row("two", "Beta", split="test")])
    with pytest.raises(ValueError, match="Split leakage"):
        collect_exports([source])


def test_profiles_and_sources_keep_separate_denominators(tmp_path):
    first = write_report(tmp_path)
    row = report_row()
    row["kaskade"]["suggestions"][0]["normen"][0]["rerank"] = .7
    second = write_report(tmp_path, [row], "second.json")
    out = tmp_path / "review"
    bundle = export_reports([first, second], out)
    assert len(bundle["cases"]) == 1
    assert len(bundle["outputs"]) == 2
    assert len({item["suggestion_id"] for item in bundle["suggestions"]}) == 4
    assert len(evaluate(out, read_json(out / "labels.json"))["metrics"]) == 2


@pytest.mark.parametrize("problem", ["duplicate_case", "different_labels", "bad_span", "missing_profile", "unknown_profile"])
def test_inconsistent_source_data_fail_before_creating_directory(tmp_path, problem):
    row = report_row()
    source = write_report(tmp_path)
    paths, profiles = [source], []
    if problem == "duplicate_case":
        source = write_report(tmp_path, [row, row])
    elif problem == "different_labels":
        paths.append(write_report(tmp_path, [report_row(targets=["norm:TEST:15"])], "second.json"))
    elif problem == "bad_span":
        row["kaskade"]["suggestions"][0]["satz"] = "Invented"
        source = write_report(tmp_path, [row])
    elif problem == "missing_profile":
        row.pop("kaskade")
        source = write_report(tmp_path, [row])
    else:
        profiles = ["absent"]
    out = tmp_path / "review"
    with pytest.raises(ValueError):
        export_reports(paths, out, profiles)
    assert not out.exists()


def test_cli_exports_then_reports_pending_and_preserves_existing_files(tmp_path, capsys):
    source, out = write_report(tmp_path), tmp_path / "review"
    assert main(["export", "--report", str(source), "--out", str(out)]) == 0
    result = tmp_path / "evaluation.json"
    arguments = ["evaluate", "--bundle", str(out / "bundle.json"), "--labels", str(out / "labels.json"), "--out", str(result)]
    assert main(arguments) == 0
    assert json.loads(result.read_text("utf-8"))["status"] == "awaiting_adjudication"
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "exists" in capsys.readouterr().err


def test_duplicate_json_keys_are_not_silently_replaced(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"label": "applicable", "label": "contradicted"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_json(path)


def training_row(case_id="one", query="Alpha", **extra):
    row = report_row(case_id, query, split="train", document_id=f"document-{case_id}", **extra)
    for i, norm in enumerate(row["kaskade"]["suggestions"][0]["normen"]):
        text = f"Verbatim synthetic evidence {i}."
        norm["evidenz"] = {"anker": ["This model-related anchor must never enter the document"],
                           "passagen": [{"id": f"{norm['id']}@0:{len(text)}", "norm_id": norm["id"],
                                         "quell_start": 0, "quell_end": len(text), "text": text}]}
    return row


def prepared_training(tmp_path, rows=None):
    source, out = write_report(tmp_path, rows or [training_row()]), tmp_path / "review"
    export_reports([source], out)
    labels = read_json(out / "labels.json")
    return out, labels


def test_train_pairs_use_only_human_labels_exact_case_and_verbatim_source(tmp_path):
    out, labels = prepared_training(tmp_path)
    label_pairs(labels, "helpful_definition", "contradicted")
    evaluate(out, labels)
    train = tmp_path / "training"
    manifest = export_train_pairs(out / "bundle.json", out / "labels.json", train)
    pairs = [json.loads(line) for line in (train / "train.jsonl").read_text("utf-8").splitlines()]
    assert manifest["status"] == "ready"
    assert manifest["positive_pairs"] == manifest["negative_pairs"] == 1
    assert [pair["label"] for pair in pairs] == [1, 0]
    assert all(pair["query"] == "Alpha" for pair in pairs)
    assert [pair["document"] for pair in pairs] == ["Verbatim synthetic evidence 0.", "Verbatim synthetic evidence 1."]
    assert all(pair["origins"][0]["document_id"] == "document-one" for pair in pairs)
    assert manifest["train_jsonl_sha256"] == sha((train / "train.jsonl").read_bytes())
    assert manifest["labels_sha256"] == sha((train / "labels.json").read_bytes())
    assert manifest["bundle_sha256"] == sha((train / "bundle.json").read_bytes())


def test_training_excludes_validation_and_test_labels(tmp_path):
    train_row, test_row = training_row(), training_row("two", "Beta")
    test_row["split"] = "test"
    out, labels = prepared_training(tmp_path, [train_row, test_row])
    label_pairs(labels, "applicable", "not_applicable", "applicable", "not_applicable")
    evaluate(out, labels)
    manifest = export_train_pairs(out / "bundle.json", out / "labels.json", tmp_path / "training")
    assert manifest["pairs"] == 2
    assert manifest["exclusion_counts"] == {"not_train_split": 2}


@pytest.mark.parametrize("label", [None, "insufficient_context"])
def test_training_never_converts_unknown_or_insufficient_to_negative(tmp_path, label):
    out, labels = prepared_training(tmp_path)
    label_pairs(labels, label, label)
    evaluate(out, labels)
    manifest = export_train_pairs(out / "bundle.json", out / "labels.json", tmp_path / "training")
    assert manifest["status"] == "no_adjudicated_training_pairs"
    assert manifest["negative_pairs"] == 0
    assert manifest["exclusion_counts"] == {"unresolved_human_label": 2}


@pytest.mark.parametrize("missing,reason", [("document", "missing_document_identity"),
                                             ("evidence", "missing_source_passages")])
def test_training_excludes_missing_document_identity_or_evidence(tmp_path, missing, reason):
    row = training_row()
    if missing == "document":
        row.pop("document_id")
    else:
        for norm in row["kaskade"]["suggestions"][0]["normen"]:
            norm.pop("evidenz")
    out, labels = prepared_training(tmp_path, [row])
    label_pairs(labels, "applicable", "not_applicable")
    evaluate(out, labels)
    manifest = export_train_pairs(out / "bundle.json", out / "labels.json", tmp_path / "training")
    assert manifest["pairs"] == 0
    assert manifest["exclusion_counts"] == {reason: 2}


@pytest.mark.parametrize("conflict", [False, True])
def test_training_deduplicates_exact_pairs_and_rejects_conflicting_human_labels(tmp_path, conflict):
    first = write_report(tmp_path, [training_row()])
    row = training_row()
    row["kaskade"]["suggestions"][0]["normen"][0]["rerank"] = .7
    second = write_report(tmp_path, [row], "second.json")
    out = tmp_path / "review"
    export_reports([first, second], out)
    labels = read_json(out / "labels.json")
    label_pairs(labels, "applicable", "not_applicable", "not_applicable" if conflict else "helpful_definition", "contradicted")
    evaluate(out, labels)
    if conflict:
        with pytest.raises(ValueError, match="Conflicting human labels"):
            export_train_pairs(out / "bundle.json", out / "labels.json", tmp_path / "training")
        assert not (tmp_path / "training").exists()
    else:
        manifest = export_train_pairs(out / "bundle.json", out / "labels.json", tmp_path / "training")
        assert manifest["pairs"] == 2
        assert manifest["adjudicated_origins"] == 4


def test_training_rejects_unassigned_overlap_with_training_document(tmp_path):
    first, second = training_row(), training_row("two", "Beta")
    second.update(document_id="document-one", split="unassigned")
    out, labels = prepared_training(tmp_path, [first, second])
    label_pairs(labels, "applicable", "not_applicable", None, None)
    evaluate(out, labels)
    with pytest.raises(ValueError, match="overlap unassigned"):
        export_train_pairs(out / "bundle.json", out / "labels.json", tmp_path / "training")


@pytest.mark.parametrize("corruption", ["offset", "norm"])
def test_training_rejects_misbound_source_evidence(tmp_path, corruption):
    row = training_row()
    passage = row["kaskade"]["suggestions"][0]["normen"][0]["evidenz"]["passagen"][0]
    if corruption == "offset":
        passage["quell_end"] += 1
    else:
        passage["norm_id"] = "norm:TEST:999"
    out, labels = prepared_training(tmp_path, [row])
    label_pairs(labels, "applicable", "not_applicable")
    evaluate(out, labels)
    with pytest.raises(ValueError):
        export_train_pairs(out / "bundle.json", out / "labels.json", tmp_path / "training")
    assert not (tmp_path / "training").exists()
