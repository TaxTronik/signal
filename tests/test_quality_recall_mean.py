"""Audit guards use synthetic accounting cases, never new legal Gold labels."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from tools._quality_recall_mean import NAMES, RULE, _json, _replay, aggregate
from tools.quality_recall_report import variant_metrics


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def archive(tmp_path):
    rows, cases = [], []
    for dataset, positives, negatives in (("main", 102, 30), ("additional", 15, 15)):
        for i in range(positives + negatives):
            query = f"Synthetic accounting case {dataset} {i}."
            row = {"case_id": f"TEST:{dataset}:{i}", "dataset": dataset, "group": "synthetic",
                   "query": query, "negative": i >= positives, "targets": [] if i >= positives else ["norm:TEST:1"],
                   "category": None, "uncertain": False}
            document = "Invented neutral source for code accounting only."
            candidate = {"id": "norm:TEST:1", "sim": .8, "document": document,
                         "document_sha256": hashlib.sha256(document.encode()).hexdigest()}
            cases.append({**row, "groups": [{"start": 0, "end": len(query), "query": query, "candidates": [candidate]}]})
            suggestions = [{"start": 0, "end": len(query), "normen": [{"id": "norm:TEST:1", "sim": .8,
                "rerank": .6, "raw_instruction_scores": dict.fromkeys(NAMES, .6)}]}]
            rows.append({**row, "mean_english3": {"suggestions": suggestions}})
    inputs = {"format": "quality_recall_inputs/1", "status": "complete", "source_hashes": {}, "input_hashes": {}, "cases": cases}
    directories = [tmp_path / "one", tmp_path / "two"]
    for directory, names in zip(directories, (NAMES[:2], NAMES[2:]), strict=True):
        directory.mkdir()
        save(directory / "inputs.json", inputs)
        scores = [[name, 2048, case["query"], case["groups"][0]["candidates"][0]["document"], .6]
                  for case in cases for name in names]
        (directory / "scores.jsonl").write_text("\n".join(json.dumps(row) for row in scores), encoding="utf-8")
        save(directory / "report.json", {"format": "quality_instruction_probe/1", "status": "complete", "rows": rows,
             "source_hashes": {}, "max_length": 2048, "threshold": .5, "shortlist": 20, "topk": 5,
             "instructions": {name: name for name in names}, "model": {"revision": "synthetic"}, "runtime": {}})
    mean = tmp_path / "mean"
    mean.mkdir()
    (mean / "source.py").write_text("# Synthetic historical producer; never executed.\n", encoding="utf-8")
    save(mean / "protocol.json", {"format": "quality_instruction_mean_protocol/1", "status": "declared", **RULE})
    report = {"format": "quality_instruction_mean/1", "status": "complete", "rows": rows,
              "metrics": {dataset: variant_metrics([r for r in rows if r["dataset"] == dataset], "mean_english3")
                          for dataset in ("main", "additional")}}
    save(mean / "report.json", report)

    def rebind():
        for directory in directories:
            parent = json.loads((directory / "report.json").read_bytes())
            parent.update(inputs_sha256=digest(directory / "inputs.json"), scores_sha256=digest(directory / "scores.jsonl"))
            save(directory / "report.json", parent)
        protocol = json.loads((mean / "protocol.json").read_bytes())
        protocol.update(script_sha256=digest(mean / "source.py"), inputs={str(directory / name): digest(directory / name)
                        for directory in directories for name in ("report.json", "scores.jsonl", "inputs.json")})
        save(mean / "protocol.json", protocol)
        report = json.loads((mean / "report.json").read_bytes())
        report["protocol_sha256"] = digest(mean / "protocol.json")
        save(mean / "report.json", report)
    rebind()
    return SimpleNamespace(path=mean / "report.json", mean=mean, parents=directories, rebind=rebind)


def test_aggregate_recomputes_counts_without_exporting_case_payload(archive):
    result = aggregate(archive.path)
    assert result["datasets"]["main"]["hit5_cases"] == 102
    assert result["datasets"]["main"]["negative_unique_case_norm_pairs"] == 30
    assert result["audit"]["exact_case_outputs"] == 162
    assert result["decision"] == "rejected_for_runtime"
    assert "TEST:" not in json.dumps(result) and str(archive.mean) not in json.dumps(result)


@pytest.mark.parametrize("field,value", [("threshold", .48), ("topk", 6), ("shortlist", 50), ("instructions", list(NAMES[:2]))])
def test_changed_rule_cannot_reuse_mean_declaration(archive, field, value):
    path = archive.mean / "protocol.json"
    protocol = json.loads(path.read_bytes())
    protocol[field] = value
    save(path, protocol)
    archive.rebind()
    with pytest.raises(ValueError, match="fixed declaration"):
        aggregate(archive.path)


def test_changed_producer_source_is_rejected(archive):
    (archive.mean / "source.py").write_text("# Changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        aggregate(archive.path)


def test_bound_score_change_cannot_hide_behind_unchanged_aggregate_metrics(archive):
    path = archive.parents[0] / "scores.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    entries[0][-1] = .9
    path.write_text("\n".join(map(json.dumps, entries)), encoding="utf-8")
    archive.rebind()
    with pytest.raises(ValueError, match="exact score replay"):
        aggregate(archive.path)


def test_missing_or_duplicate_raw_scores_are_rejected(archive):
    path = archive.parents[0] / "scores.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[1:]), encoding="utf-8")
    archive.rebind()
    with pytest.raises(ValueError, match="Missing exact"):
        aggregate(archive.path)
    path.write_text("\n".join([*lines, lines[0]]), encoding="utf-8")
    archive.rebind()
    with pytest.raises(ValueError, match="Duplicate exact"):
        aggregate(archive.path)


def test_stored_metrics_are_checked_against_rows(archive):
    report = json.loads(archive.path.read_bytes())
    report["metrics"]["main"]["negative_unique_case_norm_pairs"] -= 1
    save(archive.path, report)
    with pytest.raises(ValueError, match="metrics"):
        aggregate(archive.path)


def test_historical_mean_archive_remains_valid_without_new_additive_target_counts(archive):
    report = json.loads(archive.path.read_bytes())
    for values in report["metrics"].values():
        values.pop("historical_norm_targets")
        values.pop("matched_historical_norm_targets")
    save(archive.path, report)
    result = aggregate(archive.path)
    assert result["datasets"]["main"]["historical_norm_targets"] == 102
    assert result["datasets"]["main"]["matched_historical_norm_targets"] == 102
    report["metrics"]["main"].pop("hit5_cases")
    save(archive.path, report)
    with pytest.raises(ValueError, match="metrics"):
        aggregate(archive.path)


def test_missing_case_is_not_removed_from_denominator(archive):
    report = json.loads(archive.path.read_bytes())
    report["rows"].pop()
    save(archive.path, report)
    with pytest.raises(ValueError, match="162"):
        aggregate(archive.path)


def test_replay_applies_top20_before_mean_and_sorts_ties_before_top5():
    candidates, scores = [], {}
    for i in range(21):
        document = f"Synthetic source {i}"
        candidates.append({"id": f"norm:TEST:{i:02}", "sim": None if i == 0 else .7, "document": document,
                           "document_sha256": hashlib.sha256(document.encode()).hexdigest()})
        for name in NAMES:
            scores[name, 2048, "test", document] = .99 if i == 20 else .5
    case = {"query": "test", "groups": [{"start": 0, "end": 4, "query": "test", "candidates": candidates}]}
    result = _replay(case, {name: name for name in NAMES}, scores)
    assert [norm["id"] for norm in result[0]["normen"]] == [f"norm:TEST:{i:02}" for i in range(1, 6)]
    assert all(norm["rerank"] == .5 for norm in result[0]["normen"])


@pytest.mark.parametrize("field,value", [("start", True), ("start", -1), ("end", 5), ("query", "fake")])
def test_replay_rejects_malformed_original_focus(field, value):
    case = {"query": "test", "groups": [{"start": 0, "end": 4, "query": "test", "candidates": []}]}
    changed = deepcopy(case)
    changed["groups"][0][field] = value
    with pytest.raises(ValueError, match="focus span"):
        _replay(changed, {}, {})


@pytest.mark.parametrize("text", ['{"status":1,"status":2}', '{"score":NaN}', '{"score":Infinity}'])
def test_nonfinite_and_duplicate_json_are_rejected(text):
    with pytest.raises(ValueError):
        _json(text)
