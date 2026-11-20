"""Freeze semantic suggestions for human adjudication; evaluate reviewed labels only.

This tool imports no model/runtime code and never creates legal gold labels.
Existing targets are retained as historical metadata, separate from adjudication.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

EXPORT_FORMAT = "quality_adjudication_export/1"
LABEL_FORMAT = "quality_adjudication_labels/1"
LABELS = ("applicable", "helpful_definition", "not_applicable", "contradicted", "insufficient_context")
DECIDED = frozenset(LABELS) - {"insufficient_context"}
SPLITS = ("unassigned", "train", "development", "validation", "test")
CASE_FIELDS = ("case_id", "dataset", "group", "query", "targets", "negative", "category", "uncertain",
               "document_id", "source_document", "source", "split", "split_group_id")
NOTES = [
    "Human adjudication template, not new or independently validated legal gold.",
    "Historical targets/negative flags never fill human labels or expected targets.",
    "Assigning a test split does not establish that data were unseen during development.",
    "Only exact/normalized text and declared document/group leakage can be checked; paraphrases may remain.",
    "Precision is conditional on reviewed suggestions; unresolved and missing labels are excluded and counted.",
    "Recall requires complete human expected targets and all displayed suggestions to have resolved labels.",
    "Semantic norm suggestions only; citation cards and trigger marks remain in the frozen source report.",
]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identifier(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8"))


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_bytes(), parse_constant=reject, object_pairs_hook=unique)


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty text: {field}")
    return value


def norm_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"norm:[^:\s]+:[^\s]+", value):
        raise ValueError(f"Invalid norm ID: {value!r}")
    return value


def matches(candidate, target):
    return candidate == target or candidate.startswith(target + ":")


def normalized_query(query):
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def check_splits(cases):
    """Conservative split-group fallback; never infer document identity from gold."""
    memberships = defaultdict(list)
    for case in cases:
        if case["split"] not in SPLITS:
            raise ValueError(f"Unknown split: {case['split']}")
        memberships[("group", case["split_group_id"])].append(case)
        if case["document_id"] is not None:
            memberships[("document", case["document_id"])].append(case)
        memberships[("text", normalized_query(case["query"]))].append(case)
    unassigned_overlap = []
    for (kind, value), members in memberships.items():
        assigned = {case["split"] for case in members} - {"unassigned"}
        if len(assigned) > 1:
            raise ValueError(f"Split leakage ({kind}): {[case['case_id'] for case in members]}")
        if assigned and any(case["split"] == "unassigned" for case in members):
            unassigned_overlap.append({"kind": kind, "key_sha256": identifier(value),
                                       "case_keys": [case["case_key"] for case in members]})
    return {"assigned_cases": sum(case["split"] != "unassigned" for case in cases),
            "unassigned_cases": sum(case["split"] == "unassigned" for case in cases),
            "assigned_unassigned_overlap": unassigned_overlap,
            "independence_certified": False}


def split_map(path):
    if path is None:
        return {}
    value = read_json(path)
    if not isinstance(value, list):
        raise ValueError("Split assignments must be a list")
    result = {}
    for entry in value:
        key = (require_text(entry.get("dataset"), "dataset"),
               require_text(entry.get("case_id"), "case_id"))
        if key in result:
            raise ValueError(f"Duplicate split assignment: {key}")
        if set(entry) - {"dataset", "case_id", "document_id", "split_group_id", "split"}:
            raise ValueError(f"Unknown split assignment fields: {key}")
        result[key] = entry
    return result


def collect_exports(paths, profiles=(), assignments=None):
    """Return the complete immutable bundle before creating any output directory."""
    assignments = assignments or {}
    sources, source_bytes, cases, outputs, suggestions = [], {}, {}, [], []
    found_profiles, used_assignments, identities = set(), set(), {}
    for path in paths:
        path = Path(path).resolve()
        data = path.read_bytes()
        report = read_json(path)
        if sha(path.read_bytes()) != sha(data):
            raise ValueError(f"Report changed while reading: {path}")
        if (report.get("error") or report.get("cache_misses")
                or report.get("status") not in (None, "pass", "mismatch", "complete")):
            raise ValueError(f"Incomplete or failed report: {path}")
        rows = report.get("rows")
        available = report.get("profiles", list(profiles))
        if not isinstance(rows, list) or not rows or not isinstance(available, list) or not available:
            raise ValueError("Report requires non-empty rows and profiles (or explicit --profile)")
        selected = [p for p in available if not profiles or p in profiles]
        if not selected:
            continue
        if len(selected) != len(set(selected)) or any(not isinstance(p, str) or not p for p in selected):
            raise ValueError("Profiles must be distinct non-empty strings")
        source_id = sha(data)
        if source_id in source_bytes:
            raise ValueError("The same report was supplied twice")
        source_bytes[source_id] = data
        sources.append({"source_id": source_id, "path": str(path), "sha256": source_id,
                        "frozen_path": f"inputs/{source_id}.json", "report_format": report.get("format"),
                        "report_status": report.get("status"), "profiles": selected,
                        "declared_source_hashes": report.get("source_hashes", {}),
                        "declared_input_hashes": report.get("input_hashes", report.get("input_reports", {})),
                        "diagnostic_subset": report.get("diagnostic_subset", False),
                        "omitted_case_ids": report.get("omitted_case_ids", [])})
        seen = set()
        for row in rows:
            case_id = require_text(row.get("case_id"), "case_id")
            dataset = require_text(row.get("dataset", "unspecified"), "dataset")
            group = require_text(row.get("group"), "group")
            query = require_text(row.get("query"), "query")
            identity = (dataset, case_id)
            if identity in seen:
                raise ValueError(f"Duplicate case in source: {identity}")
            seen.add(identity)
            historical = {key: row[key] for key in CASE_FIELDS if key in row}
            if identity in identities and identities[identity] != historical:
                raise ValueError(f"Case metadata differs across reports: {identity}")
            identities[identity] = historical
            case_key = identifier([dataset, case_id, group, query])
            assignment = assignments.get(identity, {})
            if assignment:
                used_assignments.add(identity)
            document = assignment.get("document_id", row.get("document_id", row.get("source_document")))
            if document is not None:
                require_text(document, "document_id")
            split_group = assignment.get("split_group_id", row.get("split_group_id"))
            if split_group is None:
                split_group = f"document:{document}" if document else f"dataset-group:{dataset}:{group}"
            require_text(split_group, "split_group_id")
            case = {"case_key": case_key, "case_id": case_id, "dataset": dataset, "group": group,
                    "query": query, "historical_metadata": historical, "document_id": document,
                    "split_group_id": split_group, "split": assignment.get("split", row.get("split", "unassigned")),
                    "split_provenance": "explicit_assignment" if assignment else "source_or_conservative_group"}
            if case_key in cases and case != cases[case_key]:
                raise ValueError("Inconsistent case split metadata")
            cases[case_key] = case
            for profile in selected:
                value = row.get(profile)
                if not isinstance(value, dict) or value.get("error") or not isinstance(value.get("suggestions"), list):
                    raise ValueError(f"Missing or failed profile output: {case_id}/{profile}")
                found_profiles.add(profile)
                output = {"source_id": source_id, "case_key": case_key, "profile": profile,
                          "suggestion_ids": []}
                for sentence_index, sentence in enumerate(value["suggestions"]):
                    start, end, text = sentence.get("start"), sentence.get("end"), sentence.get("satz")
                    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(query)
                            or not isinstance(text, str) or query[start:end] != text):
                        raise ValueError(f"Suggestion span differs from exact query: {case_id}/{profile}")
                    norms = sentence.get("normen")
                    if not isinstance(norms, list) or not norms:
                        raise ValueError("A displayed suggestion must contain norms")
                    for rank, norm in enumerate(norms, 1):
                        norm_id(norm.get("id"))
                        item = {"source_id": source_id, "case_key": case_key, "profile": profile,
                                "sentence_index": sentence_index, "rank": rank, "norm": norm,
                                "sentence": {k: v for k, v in sentence.items() if k != "normen"}}
                        item["suggestion_id"] = identifier(item)
                        suggestions.append(item)
                        output["suggestion_ids"].append(item["suggestion_id"])
                outputs.append(output)
    if set(profiles) - found_profiles:
        raise ValueError(f"Profiles not found: {sorted(set(profiles) - found_profiles)}")
    if set(assignments) - used_assignments:
        raise ValueError("Split assignment refers to a case absent from selected reports")
    if not cases:
        raise ValueError("No cases were selected")
    case_list = list(cases.values())
    bundle = {"format": EXPORT_FORMAT, "notes": NOTES, "sources": sources,
              "cases": case_list, "outputs": outputs, "suggestions": suggestions,
              "split_audit": check_splits(case_list)}
    return bundle, source_bytes


def labels_template(bundle, bundle_hash):
    cases = {case["case_key"]: case for case in bundle["cases"]}
    return {"format": LABEL_FORMAT, "bundle_sha256": bundle_hash, "allowed_labels": list(LABELS),
            "cases": [{"case_key": case["case_key"], "case_id": case["case_id"], "query": case["query"],
                       "status": "unreviewed", "expected_targets": None, "expected_targets_complete": False,
                       "adjudicator": None, "rationale": "", "evidence": ""} for case in bundle["cases"]],
            "suggestions": [{"suggestion_id": item["suggestion_id"],
                             "case_id": cases[item["case_key"]]["case_id"], "profile": item["profile"],
                             "sentence": item["sentence"]["satz"], "norm_id": item["norm"]["id"],
                             "title": item["norm"].get("titel"), "rank": item["rank"],
                             "label": None, "adjudicator": None, "rationale": "", "evidence": ""}
                            for item in bundle["suggestions"]]}


def export_reports(paths, out, profiles=(), splits=None):
    out = Path(out)
    if out.exists():
        raise ValueError("Output directory exists; choose a fresh directory")
    split_data = Path(splits).read_bytes() if splits is not None else None
    bundle, inputs = collect_exports(paths, profiles, split_map(splits))
    if splits is not None:
        if Path(splits).read_bytes() != split_data:
            raise ValueError("Split assignments changed while exporting")
        bundle["split_assignment_source"] = {"path": str(Path(splits).resolve()),
                                              "sha256": sha(split_data)}
    data = encoded(bundle)
    out.mkdir(parents=True)
    (out / "inputs").mkdir()
    for source_id, content in inputs.items():
        (out / "inputs" / f"{source_id}.json").write_bytes(content)
    if split_data is not None:
        (out / "inputs" / "split-assignments.json").write_bytes(split_data)
    (out / "bundle.json").write_bytes(data)
    (out / "labels.json").write_bytes(encoded(labels_template(bundle, sha(data))))
    return bundle


def index_labels(entries, field, known):
    if not isinstance(entries, list):
        raise ValueError(f"Labels must be a list: {field}")
    result = {}
    for entry in entries:
        key = entry.get(field)
        if key not in known or key in result:
            raise ValueError(f"Unknown or duplicate label ID: {key}")
        result[key] = entry
    return result


def evaluate_bundle(bundle_path, label_path):
    bundle_path, label_path = Path(bundle_path), Path(label_path)
    bundle, labels = read_json(bundle_path), read_json(label_path)
    if bundle.get("format") != EXPORT_FORMAT or labels.get("format") != LABEL_FORMAT:
        raise ValueError("Unsupported export or label schema")
    if labels.get("bundle_sha256") != sha(bundle_path.read_bytes()):
        raise ValueError("Labels do not match the frozen bundle hash")
    check_splits(bundle["cases"])
    source_paths = []
    for source in bundle["sources"]:
        # Frozen input names are content-addressed, never arbitrary manifest paths.
        expected = source["sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid frozen source digest")
        path = bundle_path.parent / "inputs" / f"{expected}.json"
        if sha(path.read_bytes()) != expected:
            raise ValueError("Frozen source report was modified")
        source_paths.append(path)
    # Reconstruct predictions from frozen reports to reject changed ranks, IDs,
    # evidence, spans, and outputs even if somebody recalculated a bundle hash.
    selected = {source["source_id"]: source["profiles"] for source in bundle["sources"]}
    assignments = {}
    if "split_assignment_source" in bundle:
        path = bundle_path.parent / "inputs" / "split-assignments.json"
        if sha(path.read_bytes()) != bundle["split_assignment_source"]["sha256"]:
            raise ValueError("Frozen split assignments were modified")
        assignments = split_map(path)
    reconstructed, original_outputs, original_cases = [], [], {}
    for path in source_paths:
        source_id = sha(path.read_bytes())
        source_cases = {(row.get("dataset", "unspecified"), row["case_id"]) for row in read_json(path)["rows"]}
        original, _ = collect_exports([path], selected[source_id],
                                      {key: value for key, value in assignments.items() if key in source_cases})
        reconstructed.extend(original["suggestions"])
        original_outputs.extend(original["outputs"])
        for case in original["cases"]:
            if case["case_key"] in original_cases and original_cases[case["case_key"]] != case:
                raise ValueError("Frozen source cases disagree")
            original_cases[case["case_key"]] = case
    if reconstructed != bundle["suggestions"]:
        raise ValueError("Export suggestions differ from the frozen source reports")
    if original_outputs != bundle["outputs"]:
        raise ValueError("Export outputs differ from the frozen source reports")
    if list(original_cases.values()) != bundle["cases"]:
        raise ValueError("Export cases differ from the frozen source reports and split assignments")
    if check_splits(bundle["cases"]) != bundle["split_audit"]:
        raise ValueError("Export split audit was modified")
    cases = {case["case_key"]: case for case in bundle["cases"]}
    suggestions = {item["suggestion_id"]: item for item in bundle["suggestions"]}
    case_labels = index_labels(labels.get("cases"), "case_key", cases)
    pair_labels = index_labels(labels.get("suggestions"), "suggestion_id", suggestions)
    complete_cases = {}
    for key, entry in case_labels.items():
        if entry.get("case_id", cases[key]["case_id"]) != cases[key]["case_id"] or entry.get("query", cases[key]["query"]) != cases[key]["query"]:
            raise ValueError("Editable case context differs from the frozen case")
        status = entry.get("status")
        if status not in ("unreviewed", "adjudicated") or type(entry.get("expected_targets_complete")) is not bool:
            raise ValueError("Invalid case adjudication status")
        targets = entry.get("expected_targets")
        if status == "unreviewed":
            if targets is not None or entry["expected_targets_complete"]:
                raise ValueError("Unreviewed cases cannot contain adjudicated targets")
            continue
        require_text(entry.get("adjudicator"), "case adjudicator")
        if not isinstance(targets, list):
            raise ValueError("Adjudicated cases require distinct explicit expected targets (possibly [])")
        for target in targets:
            norm_id(target)
        if len(targets) != len(set(targets)):
            raise ValueError("Expected targets must be distinct")
        if any(a != b and matches(a, b) for a in targets for b in targets):
            raise ValueError("Expected targets overlap hierarchically; choose one evaluation granularity")
        if entry["expected_targets_complete"]:
            complete_cases[key] = targets
    resolved = {}
    for key, entry in pair_labels.items():
        item = suggestions[key]
        expected_context = {"case_id": cases[item["case_key"]]["case_id"], "profile": item["profile"],
                            "sentence": item["sentence"]["satz"], "norm_id": item["norm"]["id"],
                            "title": item["norm"].get("titel"), "rank": item["rank"]}
        if any(entry.get(field, value) != value for field, value in expected_context.items()):
            raise ValueError("Editable suggestion context differs from the frozen suggestion")
        label = entry.get("label")
        if label is not None and label not in LABELS:
            raise ValueError(f"Unknown suggestion label: {label}")
        if label is not None:
            require_text(entry.get("adjudicator"), "suggestion adjudicator")
        resolved[key] = label
    metrics, grouped = [], defaultdict(list)
    for output in bundle["outputs"]:
        case = cases[output["case_key"]]
        key = (output["source_id"], output["profile"], case["dataset"], case["group"], case["split"])
        grouped[key].append(output)
    for key, outputs in grouped.items():
        counts = Counter(dict.fromkeys((*LABELS, "unlabeled", "cases", "displayed_suggestions",
                                        "recall_excluded_no_complete_targets", "recall_excluded_unresolved_suggestions",
                                        "recall_eligible_cases", "recall_empty_target_cases"), 0))
        expected_count, found_count = 0, 0
        recall_cases = []
        for output in outputs:
            ids = output["suggestion_ids"]
            counts["cases"] += 1
            counts["displayed_suggestions"] += len(ids)
            values = [resolved.get(item_id) for item_id in ids]
            for value in values:
                counts[value or "unlabeled"] += 1
            targets = complete_cases.get(output["case_key"])
            if targets is None:
                counts["recall_excluded_no_complete_targets"] += 1
                continue
            if any(value not in DECIDED for value in values):
                counts["recall_excluded_unresolved_suggestions"] += 1
                continue
            norms = [suggestions[item_id]["norm"]["id"] for item_id, label in zip(ids, values, strict=True)
                     if label == "applicable"]
            if any(not any(matches(norm, target) for target in targets) for norm in norms):
                raise ValueError("Applicable suggestion absent from declared complete expected targets")
            found = sum(any(matches(norm, target) for norm in norms) for target in targets)
            expected_count += len(targets)
            found_count += found
            counts["recall_eligible_cases"] += 1
            counts["recall_empty_target_cases"] += not targets
            recall_cases.append({"case_key": output["case_key"], "expected": len(targets), "found": found})
        decided = sum(counts[label] for label in DECIDED)
        displayed = counts["displayed_suggestions"]
        metrics.append({"source_id": key[0], "profile": key[1], "dataset": key[2], "group": key[3], "split": key[4],
                        "counts": dict(counts), "resolved_suggestions": decided,
                        "review_coverage": decided / displayed if displayed else None,
                        "precision_applicable_reviewed": counts["applicable"] / decided if decided else None,
                        "precision_helpful_reviewed": (counts["applicable"] + counts["helpful_definition"]) / decided if decided else None,
                        "recall_applicable_adjudicated": found_count / expected_count if expected_count else None,
                        "expected_targets": expected_count, "found_targets": found_count,
                        "recall_cases": recall_cases})
    any_metric = any(row["precision_applicable_reviewed"] is not None or row["recall_applicable_adjudicated"] is not None for row in metrics)
    complete = len(complete_cases) == len(cases) and all(resolved.get(key) in DECIDED for key in suggestions)
    status = "adjudicated" if complete else "partial_adjudication" if any_metric else "awaiting_adjudication"
    return {"format": "quality_adjudication_evaluation/1", "status": status,
            "bundle_sha256": sha(bundle_path.read_bytes()), "labels_sha256": sha(label_path.read_bytes()),
            "notes": NOTES, "split_audit": bundle["split_audit"], "metrics": metrics,
            "missing_case_label_entries": len(cases.keys() - case_labels.keys()),
            "missing_suggestion_label_entries": len(suggestions.keys() - pair_labels.keys())}


def training_document(item):
    """A new deterministic training document from verbatim frozen source passages.

    These outputs do not contain every original inference input. Do not invent
    its former query, title weighting, or catalogue anchors. Missing evidence
    is an explicit exclusion, never a substitute document made from a norm ID.
    """
    evidence = item["norm"].get("evidenz")
    if not isinstance(evidence, dict) or not evidence.get("passagen"):
        return None
    passages = evidence["passagen"]
    if not isinstance(passages, list):
        raise ValueError("Evidence passages must be a list")
    for passage in passages:
        if not isinstance(passage, dict) or passage.get("norm_id") != item["norm"]["id"]:
            raise ValueError("Evidence passage belongs to a different norm")
        require_text(passage.get("id"), "passage ID")
        text = require_text(passage.get("text"), "source passage text")
        start, end = passage.get("quell_start"), passage.get("quell_end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end or end - start != len(text):
            raise ValueError("Source passage offsets do not match its verbatim text")
    return "\n\n".join(passage["text"] for passage in passages), passages


def export_train_pairs(bundle_path, label_path, out):
    out, bundle_path, label_path = Path(out), Path(bundle_path), Path(label_path)
    if out.exists():
        raise ValueError("Training output directory exists; choose a fresh directory")
    evaluation = evaluate_bundle(bundle_path, label_path)
    bundle, labels = read_json(bundle_path), read_json(label_path)
    if (sha(bundle_path.read_bytes()) != evaluation["bundle_sha256"]
            or sha(label_path.read_bytes()) != evaluation["labels_sha256"]):
        raise ValueError("Adjudication inputs changed during validation")
    cases = {case["case_key"]: case for case in bundle["cases"]}
    pair_labels = {entry["suggestion_id"]: entry for entry in labels["suggestions"]}
    train_keys = {key for key, case in cases.items() if case["split"] == "train"}
    if any(train_keys.intersection(entry["case_keys"])
           for entry in bundle["split_audit"]["assigned_unassigned_overlap"]):
        raise ValueError("Training cases overlap unassigned document/group/text members; resolve their splits first")
    pairs, exclusions = {}, []
    for item in bundle["suggestions"]:
        case = cases[item["case_key"]]
        label = pair_labels.get(item["suggestion_id"], {})
        value = label.get("label")
        reason = None
        if case["split"] != "train":
            reason = "not_train_split"
        elif value not in DECIDED:
            reason = "unresolved_human_label"
        elif case["document_id"] is None:
            reason = "missing_document_identity"
        if reason is not None:
            exclusions.append({"suggestion_id": item["suggestion_id"], "reason": reason})
            continue
        evidence = training_document(item)
        if evidence is None:
            exclusions.append({"suggestion_id": item["suggestion_id"], "reason": "missing_source_passages"})
            continue
        document, passages = evidence
        query = case["query"]
        pair_id = identifier([query, document])
        binary_label = int(value in ("applicable", "helpful_definition"))
        origin = {"source_id": item["source_id"], "case_key": case["case_key"], "case_id": case["case_id"],
                  "dataset": case["dataset"], "group": case["group"], "document_id": case["document_id"],
                  "split_group_id": case["split_group_id"], "split": case["split"], "profile": item["profile"],
                  "suggestion_id": item["suggestion_id"], "sentence": item["sentence"],
                  "norm_id": item["norm"]["id"], "source_passages": passages,
                  "human_label": value, "adjudicator": label["adjudicator"],
                  "rationale": label.get("rationale", ""), "evidence": label.get("evidence", "")}
        if pair_id in pairs:
            if pairs[pair_id]["label"] != binary_label:
                raise ValueError("Conflicting human labels for an identical query/document training pair")
            pairs[pair_id]["origins"].append(origin)
        else:
            pairs[pair_id] = {"pair_id": pair_id, "query": query, "document": document,
                              "label": binary_label, "origins": [origin]}
    data = "".join(json.dumps(pair, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                   for pair in pairs.values()).encode("utf-8")
    bundle_data, label_data = bundle_path.read_bytes(), label_path.read_bytes()
    if sha(bundle_data) != evaluation["bundle_sha256"] or sha(label_data) != evaluation["labels_sha256"]:
        raise ValueError("Adjudication inputs changed before training export")
    manifest = {"format": "quality_adjudication_train_pairs/1",
                "status": "ready" if pairs else "no_adjudicated_training_pairs",
                "bundle_sha256": evaluation["bundle_sha256"], "labels_sha256": evaluation["labels_sha256"],
                "bundle_path": str(bundle_path.resolve()), "labels_path": str(label_path.resolve()),
                "train_jsonl_sha256": sha(data), "pairs": len(pairs),
                "positive_pairs": sum(pair["label"] == 1 for pair in pairs.values()),
                "negative_pairs": sum(pair["label"] == 0 for pair in pairs.values()),
                "adjudicated_origins": sum(len(pair["origins"]) for pair in pairs.values()),
                "exclusion_counts": dict(Counter(entry["reason"] for entry in exclusions)), "exclusions": exclusions,
                "notes": [
                    "Only resolved human labels on explicitly assigned train cases with known source-document identity.",
                    "Applicable/helpful_definition are positive; not_applicable/contradicted are negative.",
                    "The training objective is broad relevance, not a calibrated probability of legal applicability.",
                    "Query is the exact complete frozen case text; document joins verbatim source passages with two newlines.",
                    "These are new transparent training inputs, not claimed reconstructions of historical reranker inputs.",
                    "No catalogue anchors, model pseudo-labels, or unreviewed historical target labels are used.",
                    "Identical query/document pairs are deduplicated; conflicting binary human labels fail.",
                    "No model was trained. Split checks do not establish unseen or independent evaluation data.",
                ]}
    out.mkdir(parents=True)
    (out / "train.jsonl").write_bytes(data)
    (out / "bundle.json").write_bytes(bundle_data)
    (out / "labels.json").write_bytes(label_data)
    (out / "manifest.json").write_bytes(encoded(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Freeze outputs and create empty human label templates")
    export.add_argument("--report", type=Path, action="append", required=True)
    export.add_argument("--profile", action="append", default=[])
    export.add_argument("--splits", type=Path, help="Explicit dataset/case/document/split assignments JSON")
    export.add_argument("--out", type=Path, required=True, help="Fresh directory")
    evaluate = commands.add_parser("evaluate", help="Validate frozen provenance and evaluate reviewed labels")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True, help="New evaluation JSON; existing files are preserved")
    train = commands.add_parser("train-pairs", help="Export only adjudicated train labels with frozen source evidence")
    train.add_argument("--bundle", type=Path, required=True)
    train.add_argument("--labels", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True, help="Fresh training artifact directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            bundle = export_reports(args.report, args.out, args.profile, args.splits)
            print(f"Exported {len(bundle['cases'])} cases / {len(bundle['suggestions'])} norm suggestions; no human labels created.")
        elif args.command == "evaluate":
            if args.out.exists():
                raise ValueError("Evaluation output exists; choose a fresh path")
            result = evaluate_bundle(args.bundle, args.labels)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(encoded(result))
            print(f"{result['status']}; {len(result['metrics'])} source/profile/dataset/group/split summaries.")
        else:
            manifest = export_train_pairs(args.bundle, args.labels, args.out)
            print(f"{manifest['status']}; {manifest['pairs']} adjudicated training pairs; no model trained.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
