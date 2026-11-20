"""Model-free, aggregate-only audit of the single declared English-score mean."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from tools import quality_recall_report as audit

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("english_match", "english_retrieval", "english_issue")
RULE = {"rule": "arithmetic mean of all three English instruction raw scores",
        "instructions": list(NAMES), "threshold": .5, "shortlist": 20, "topk": 5,
        "ordering": "descending mean, descending retrieval similarity, ascending norm ID"}


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            audit.require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def _replay(case, instructions, scores):
    suggestions = []
    for group in case["groups"]:
        start, end = group["start"], group["end"]
        audit.require(type(start) is int and type(end) is int and 0 <= start < end <= len(case["query"])
                      and case["query"][start:end] == group["query"], "Invalid original focus span")
        candidates = group["candidates"]
        audit.require(len({c["id"] for c in candidates}) == len(candidates), "Duplicate norm candidate")
        eligible = []
        for candidate in candidates[:20]:
            audit.require(candidate["sim"] is None or _number(candidate["sim"]), "Invalid retrieval similarity")
            document = candidate["document"]
            audit.require(hashlib.sha256(document.encode()).hexdigest() == candidate["document_sha256"],
                          "Candidate document identity differs")
            keys = [(instructions[name], 2048, group["query"], document) for name in NAMES]
            audit.require(all(key in scores for key in keys), "Missing exact instruction score")
            values = [scores[key] for key in keys]
            mean = math.fsum(values) / 3
            if mean >= .5:
                eligible.append({"id": candidate["id"], "sim": candidate["sim"], "rerank": mean,
                                 "raw_instruction_scores": dict(zip(NAMES, values, strict=True))})
        eligible.sort(key=lambda n: (-n["rerank"], -(n["sim"] if n["sim"] is not None else -math.inf), n["id"]))
        if eligible:
            suggestions.append({"start": start, "end": end, "normen": eligible[:5]})
    return suggestions


def aggregate(path):
    """Verify bound raw-score replay and return no case, query or norm payload."""
    path = Path(path).resolve()
    hashes = {}

    def raw(filename, expected=None):
        filename = Path(filename).resolve()
        data = filename.read_bytes()
        value = hashlib.sha256(data).hexdigest()
        audit.require(expected is None or value == expected, "Bound mean input hash differs")
        audit.require(str(filename) not in hashes or hashes[str(filename)] == value, "Input changed while reading")
        hashes[str(filename)] = value
        return data

    report = _json(raw(path))
    audit.require(report.get("format") == "quality_instruction_mean/1" and report.get("status") == "complete"
                  and not report.get("error") and not report.get("diagnostic_subset"), "Requires a completed mean report")
    rows = audit.cases(report["rows"])
    protocol_path = path.parent / "protocol.json"
    protocol = _json(raw(protocol_path, report["protocol_sha256"]))
    audit.require(protocol.get("format") == "quality_instruction_mean_protocol/1" and protocol.get("status") == "declared"
                  and all(protocol.get(key) == value for key, value in RULE.items()), "Mean rule differs from fixed declaration")
    source_path = path.parent / "source.py"
    raw(source_path, protocol["script_sha256"])
    files = {}
    for name, expected in protocol["inputs"].items():
        filename = Path(name.replace("\\", "/"))
        if not filename.is_absolute():
            filename = ROOT / filename
        filename = filename.resolve()
        audit.require(filename not in files, "Duplicate bound input path")
        files[filename] = raw(filename, expected)
    parents = sorted({filename.parent for filename in files}, key=str)
    audit.require(len(parents) == 2 and len(files) == 6
                  and all({p.name for p in files if p.parent == directory} == {"report.json", "scores.jsonl", "inputs.json"}
                          for directory in parents), "Expected two complete instruction archives")
    instructions, scores, identity, inputs = {}, {}, None, None
    for directory in parents:
        parent = _json(files[directory / "report.json"])
        audit.require(parent.get("format") == "quality_instruction_probe/1" and parent.get("status") == "complete"
                      and not parent.get("error") and parent.get("max_length") == 2048 and parent.get("threshold") == .5
                      and parent.get("shortlist") == 20 and parent.get("topk") == 5, "Incompatible instruction parent")
        audit.same_cases(rows, audit.cases(parent["rows"]))
        parent_hashes = audit.bind_report(directory / "report.json", parent)
        for filename, expected in parent_hashes.items():
            audit.require(filename not in hashes or hashes[filename] == expected, "Parent binding differs")
            hashes[filename] = expected
        current_identity = (parent["model"], parent["runtime"], parent["inputs_sha256"])
        audit.require(identity is None or identity == current_identity, "Instruction model/runtime/input identities differ")
        identity = current_identity
        current_inputs = _json(files[directory / "inputs.json"])
        audit.require(current_inputs.get("format") == "quality_recall_inputs/1" and current_inputs.get("status") == "complete",
                      "Invalid instruction inputs")
        audit.same_cases(rows, audit.cases(current_inputs["cases"]))
        audit.require(inputs is None or inputs == current_inputs, "Instruction case inputs differ")
        inputs = current_inputs
        for field in ("source_hashes", "input_hashes"):
            for filename, expected in current_inputs[field].items():
                audit.check_hash(filename, expected)
                audit.require(filename not in hashes or hashes[filename] == expected, "Input source identity differs")
                hashes[filename] = expected
        for name, instruction in parent["instructions"].items():
            if name in NAMES:
                audit.require(name not in instructions, "Repeated mean instruction")
                instructions[name] = instruction
        for line in files[directory / "scores.jsonl"].splitlines():
            entry = _json(line)
            audit.require(isinstance(entry, list) and len(entry) == 5, "Malformed raw score row")
            instruction, budget, query, document, score = entry
            audit.require(instruction in parent["instructions"].values() and type(budget) is int and budget == 2048
                          and isinstance(query, str) and isinstance(document, str) and _number(score) and 0 <= score <= 1,
                          "Invalid raw instruction score")
            key = (instruction, budget, query, document)
            audit.require(key not in scores, "Duplicate exact score input")
            scores[key] = score
    audit.require(set(instructions) == set(NAMES) and len(set(instructions.values())) == 3, "Missing distinct mean instructions")
    for case in inputs["cases"]:
        audit.require(_replay(case, instructions, scores) == rows[case["case_id"]]["mean_english3"]["suggestions"],
                      "Mean output differs from exact score replay")
    metrics = {dataset: audit.variant_metrics([row for row in report["rows"] if row["dataset"] == dataset], "mean_english3")
               for dataset in ("main", "additional")}
    # Historical archives predate the two additive micro-coverage counters.
    # Every stored field remains checked; no earlier required field is optional.
    new_fields = {"historical_norm_targets", "matched_historical_norm_targets"}
    audit.require(set(report["metrics"]) == set(metrics), "Stored mean metric datasets differ")
    for dataset, actual in metrics.items():
        stored = report["metrics"][dataset]
        audit.require(set(stored) in (set(actual), set(actual) - new_fields)
                      and stored == {key: actual[key] for key in stored},
                      "Stored mean metrics differ from actual outputs")
    for filename, expected in hashes.items():
        audit.check_hash(filename, expected)
    result = {"format": "quality_instruction_mean_aggregate/1", "decision": "rejected_for_runtime",
              "report_sha256": hashes[str(path)], "protocol_sha256": hashes[str(protocol_path)],
              "source_sha256": hashes[str(source_path)], "rule": {**RULE, "max_length": 2048},
              "datasets": metrics, "audit": {"exact_case_outputs": len(rows), "bound_files": len(hashes), "mismatches": 0},
              "notes": ["Single post-hoc arithmetic mean of the three existing English instruction runs; no new model inference.",
                        "All 162 cases are previously inspected development data, not independent evaluation examples.",
                        "The three prompts use the same model and are not independent validation evidence.",
                        "No threshold/weight selection, learned labels or runtime activation; positive non-target norms remain unadjudicated."]}
    audit.public_payload_check(result)
    return result
