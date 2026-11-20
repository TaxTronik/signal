"""Independent stdlib recount of the frozen 16-case public Signal baseline.

No productive evaluator/parser, model import, model inference, or source mutation.
This is an output/count and archive-binding audit, not a legal adjudication or a
second replay of the producer's complete GPU/source-evidence audit.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, cast


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


class BoundFiles:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        with path.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        require(expected is None or sha == expected, f"File hash differs: {path.name}")
        require(path not in self.files or self.files[path] == sha, "File changed during audit")
        self.files[path] = sha
        return sha

    def read(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        sha = digest(data)
        require(path not in self.files or self.files[path] == sha, "File changed before reading")
        self.files[path] = sha
        return data

    def json(self, path):
        return strict_json(self.read(path))

    def inventory(self, directory, expected, excluded):
        directory = Path(directory).resolve()
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
        require(actual - set(excluded) == set(expected), "Archive inventory differs")
        for relative, sha in expected.items():
            target = (directory / relative).resolve()
            require(target.is_relative_to(directory), "Archive path escapes directory")
            self.bind(target, sha)

    def unchanged(self):
        for path, sha in list(self.files.items()):
            self.bind(path, sha)


def norm_parent(value):
    require(isinstance(value, str) and re.fullmatch(r"norm:[A-Za-z][A-Za-z0-9_-]*:(?:art)?[0-9]+[a-z]*(?::[A-Za-z0-9]+)*", value),
            "Invalid canonical norm ID")
    return ":".join(value.split(":")[:3])


def citation_parent(value):
    """Strict bounded grammar for the frozen catalog's single-parent anchors.

    No permissive free-text scan or guessed alias conversion. Unsupported lists,
    ranges, law names or connectors abort the audit instead of losing anchors.
    """
    require(isinstance(value, str), "Anchor is not text")
    match = re.fullmatch(
        r"\s*(?P<kind>§|Art\.)\s*(?P<number>[0-9]+[a-z]*)"
        r"(?:\s+(?:Abs\.?|Absatz|Satz|S\.|Nr\.?|Nummer|Buchst\.?|Buchstabe|lit\.|Halbsatz|Alt\.|Var\.)"
        r"\s*(?:[0-9]+[a-z]?|[a-z]))*\s+(?P<law>[A-Z][A-Za-z0-9_-]*)\s*", value)
    if match is None:
        raise ValueError("Unsupported hard-anchor citation; independent audit must be extended explicitly")
    number = ("art" if match["kind"] == "Art." else "") + match["number"]
    return norm_parent(f"norm:{match['law']}:{number}")


def output_sets(output):
    semantic = set()
    for focus in output["suggestions"]:
        for norm in focus["normen"]:
            norm_parent(norm["id"])
            semantic.add(norm["id"])
    hard = {citation_parent(citation) for card in output["cards"] for citation in card["norm_anker"]}
    hard.update(norm_parent(norm["id"]) for risk in output["risk_marks"] for norm in risk["norm_vorschlag"])
    require(canonical(sorted(hard)) == canonical(output["hard_norm_ids"]), "Stored hard-norm list differs from original cards/risks")
    return {"semantic_suggestions": semantic, "hard_norm_anchors": hard, "all_visible_norms": semantic | hard}


def check_cases(cases):
    require(len(cases) == 16 and len({c["case_id"] for c in cases}) == 16, "Expected exactly 16 distinct planned cases")
    families: dict[str, list[dict[str, Any]]] = {}
    totals: Counter[str] = Counter()
    for case in cases:
        require(case["split"] in ("development", "evaluation"), "Unknown split")
        family = families.setdefault(case["family_id"], [])
        require(not family or family[0]["split"] == case["split"], "Family crosses split")
        family.append(case)
        labels = case["diagnostic_labels"]
        require(type(labels["hint_expected"]) is bool, "Hint label is not Boolean")
        require(labels["annotation_status"] == "assistant_reviewed_synthetic", "Unexpected annotation status")
        groups = [labels[key] for key in ("target_norm_ids", "no_hint_norm_ids", "allowed_hint_norm_ids", "insufficient_context_norm_ids")]
        flat = [n for group in groups for n in group]
        require(len(flat) == len(set(flat)), "Diagnostic norm labels overlap or repeat")
        for norm in flat:
            require(norm == norm_parent(norm), "Diagnostic ID is not a parent norm")
        require(bool(groups[0]) == labels["hint_expected"], "Positive case/target definition differs")
        totals[case["split"]] += 1
        totals["positive"] += labels["hint_expected"]
        totals["targets"] += len(groups[0])
        if not labels["hint_expected"]:
            totals["context" if groups[3] else "clear_negative"] += 1
    require(len(families) == 8 and all(len(rows) == 2 for rows in families.values()), "Expected eight paired families")
    require(dict(totals) == {"development": 8, "evaluation": 8, "positive": 12, "targets": 14, "context": 1, "clear_negative": 3},
            "Planned 16-case diagnostic denominators differ")


def recount(cases, rows):
    require(len({r["case_id"] for r in rows}) == len(rows), "Duplicate result case")
    by_id = {r["case_id"]: r for r in rows}
    require(set(by_id) <= {c["case_id"] for c in cases}, "Unplanned result case")
    records = []
    for case in cases:
        row = by_id.get(case["case_id"], {"status": "not_run"})
        status = row["status"]
        require(status in ("complete", "technical_error", "invalid_output", "not_run"), "Unknown case status")
        if "case_id" in row:
            require(row["family_id"] == case["family_id"] and row["split"] == case["split"], "Result identity differs")
        labels = case["diagnostic_labels"]
        positive = labels["hint_expected"]
        targets, negative = set(labels["target_norm_ids"]), set(labels["no_hint_norm_ids"])
        context = set(labels.get("insufficient_context_norm_ids", []))
        known = targets | negative | context | set(labels["allowed_hint_norm_ids"])
        sets = output_sets(row["output"]) if status == "complete" else {v: set() for v in ("semantic_suggestions", "hard_norm_anchors", "all_visible_norms")}
        records.append({"case": case, "status": status, "positive": positive, "targets": targets, "negative": negative,
                        "known": known, "group": ("context_only" if context else "clear_negative") if not positive else None, "sets": sets})

    def group(selected, view):
        counts = Counter({"planned_cases": len(selected)})
        for r in selected:
            counts[r["status"]] += 1
        complete = [r for r in selected if r["status"] == "complete"]
        unknown = [r for r in selected if r["status"] != "complete"]
        counts.update({
            "positive_cases": sum(r["positive"] for r in selected),
            "non_hint_cases": sum(not r["positive"] for r in selected),
            "positive_case_hits": sum(bool(r["positive"] and r["targets"] & r["sets"][view]) for r in complete),
            "positive_cases_unknown": sum(r["positive"] for r in unknown),
            "known_target_norm_pairs": sum(len(r["targets"]) for r in selected),
            "known_target_norm_hits": sum(len(r["targets"] & r["sets"][view]) for r in complete),
            "known_target_norm_pairs_unknown": sum(len(r["targets"]) for r in unknown),
            "non_hint_cases_with_output": sum(bool(not r["positive"] and r["sets"][view]) for r in complete),
            "complete_non_hint_cases_without_output": sum(not r["positive"] and not r["sets"][view] for r in complete),
            "non_hint_cases_unknown": sum(not r["positive"] for r in unknown),
            "explicit_no_hint_norm_pairs": sum(len(r["negative"]) for r in selected),
            "explicit_no_hint_output_pairs": sum(len(r["negative"] & r["sets"][view]) for r in complete),
            "explicit_no_hint_output_cases": sum(bool(r["negative"] & r["sets"][view]) for r in complete),
            "explicit_no_hint_pairs_unknown": sum(len(r["negative"]) for r in unknown),
            "output_norm_pairs": sum(len(r["sets"][view]) for r in complete),
            "unadjudicated_output_pairs": sum(len(r["sets"][view] - r["known"]) for r in complete),
        })
        for kind in ("clear_negative", "context_only"):
            members = [r for r in selected if r["group"] == kind]
            good = [r for r in members if r["status"] == "complete"]
            counts.update({f"planned_{kind}_cases": len(members),
                           f"{kind}_cases_with_output": sum(bool(r["sets"][view]) for r in good),
                           f"{kind}_output_norm_pairs": sum(len(r["sets"][view]) for r in good),
                           f"{kind}_cases_unknown": len(members) - len(good),
                           f"complete_{kind}_cases_without_output": sum(not r["sets"][view] for r in good)})
        for status in ("complete", "technical_error", "invalid_output", "not_run"):
            counts.setdefault(status, 0)
        result: dict[str, int | float | str | None] = dict(counts)
        denominator = counts["positive_cases"]
        result.update(hint_recall_lower_bound=counts["positive_case_hits"] / denominator if denominator else None,
                      hint_recall_upper_bound=(counts["positive_case_hits"] + counts["positive_cases_unknown"]) / denominator if denominator else None,
                      precision=None, facts_and_conditions="not provided by this baseline")
        return result

    return {view: {split: group([r for r in records if split == "all" or r["case"]["split"] == split], view)
                   for split in ("all", "development", "evaluation")}
            for view in ("semantic_suggestions", "hard_norm_anchors", "all_visible_norms")}


def measured_timing(rows):
    seconds = [row["timing"]["elapsed_seconds"] for row in rows if "elapsed_seconds" in row.get("timing", {})]
    require(all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in seconds), "Invalid measured duration")
    seconds.sort()

    def percentile(p):
        if not seconds:
            return None
        position = (len(seconds) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return seconds[low] + (seconds[high] - seconds[low]) * (position - low)

    return {"recorded_attempts": len(seconds), "total_seconds": sum(seconds), "median_seconds": percentile(.5),
            "p95_seconds": percentile(.95), "max_seconds": max(seconds) if seconds else None,
            "synchronization_unavailable_attempts": sum(row.get("timing", {}).get("synchronized") is False for row in rows),
            "scope": "Synchronized Pipeline.run including read-only trace overhead and failed attempts; excludes model load, warmup, output audit, persistence and final hashing."}


def profile_from_source(data):
    tree = ast.parse(data)
    assignments = [n for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "PROFILE_CONFIG"]
    require(len(assignments) == 1, "Ambiguous frozen profile declaration")
    return ast.literal_eval(cast(ast.expr, assignments[0].value))["treffer"]


def audit(run, prepared):
    run, prepared = Path(run).resolve(), Path(prepared).resolve()
    bound = BoundFiles()
    report = bound.json(run / "report.json")
    require(report["format"] == "grounded_signal_public_baseline/1" and report["status"] in ("complete", "incomplete", "failed"), "Expected terminal public baseline")
    bound.inventory(run, report["artifacts"], ("report.json",))
    manifest = bound.json(prepared / "prepared-manifest.json")
    require(manifest["format"] == "grounded_signal_public_baseline_inputs/1", "Unknown prepared format")
    require(bound.bind(prepared / "prepared-manifest.json") == report["prepared_manifest_sha256"], "Prepared manifest not bound to run")
    bound.inventory(prepared, manifest["artifacts"], ("prepared-manifest.json",))
    protocol = bound.json(prepared / "protocol.json")
    require(canonical(protocol) == canonical(report["protocol"]) == canonical(bound.json(run / "protocol.json")), "Protocol differs")
    require(digest(canonical(protocol).encode()) == report["protocol_sha256"], "Canonical protocol hash differs")
    for name in ("cases.json", "prepared-manifest.json", "probe-source.py"):
        require(bound.bind(run / name) == bound.bind(prepared / name), "Run snapshot differs")
    require(bound.bind(prepared / "probe-source.py") == protocol["probe_source_sha256"], "Producer source is unbound")
    require(bound.bind(prepared / "primary/input-manifest.json") == protocol["primary_manifest_sha256"], "Primary manifest differs")
    primary = bound.json(prepared / "primary/input-manifest.json")
    require(canonical(primary["artifacts"]) == canonical(protocol["primary_artifacts"]), "Primary input inventory differs")
    cases = bound.json(prepared / "cases.json")
    require(canonical(cases) == canonical(bound.json(prepared / "primary/cases.json")), "Cases differ from primary snapshot")
    require(bound.bind(prepared / "primary/cases.json") == primary["artifacts"]["cases.json"], "Case byte binding differs")
    check_cases(cases)
    require(protocol["case_ids"] == [c["case_id"] for c in cases], "Planned case IDs differ")
    for name, sha in protocol["source_hashes"].items():
        target = (prepared / "source" / name).resolve()
        require(target.is_relative_to(prepared / "source"), "Source path escapes archive")
        bound.bind(target, sha)
    profile = profile_from_source(bound.read(prepared / "source/risk_layer/semantic_search.py"))
    require(protocol["profile"] == protocol["options"]["semantik_profil"] == "treffer", "Not the public treffer profile")
    require(canonical(profile) == canonical(protocol["profile_config"]), "Frozen source and requested profile differ")
    earlier = bound.json(prepared / "gpu-reference.json")
    require(bound.bind(prepared / "gpu-reference.json") == protocol["gpu_reference_sha256"], "Earlier GPU route is unbound")
    require(canonical(earlier["profile_config"]) == canonical(profile), "Public configuration differs from measured route")
    meta = bound.json(prepared / "index/meta.json")
    identity = protocol["source_identity"]
    require(meta["graph_fingerprint"] == identity["graph"] and meta["catalog_fingerprint"] == identity["catalog"], "Source/index identity differs")
    require(meta["encoder"] == identity["encoder"] and meta["encoder_revision"] == identity["revision"], "Index encoder differs")
    require(meta["passage_evidence"]["source_fingerprint"] == identity["passage_sources"], "Passage-source identity differs")
    require(set(protocol["models"]) == {"encoder", "bge", "qwen"}, "Unexpected planned model set")
    for label, planned in protocol["models"].items():
        measured = earlier["models"][label]
        require(planned["name"] == measured["name"] and planned["revision"] == measured["revision"]
                and [planned["dtype"]] == measured["dtypes"], "Planned model differs from measured route")
    require(report["end_verification"]["status"] in ("pass", "failed"), "Missing producer end verification")
    rows = report["rows"]
    require([r["case_id"] for r in rows] == protocol["case_ids"][:len(rows)], "Rows are not a unique planned prefix")
    journal = [strict_json(line) for line in bound.read(run / "rows.jsonl").splitlines()] if (run / "rows.jsonl").exists() else []
    require(canonical(journal) == canonical(rows), "Incremental journal differs")
    for number, row in enumerate(rows, 1):
        require(canonical(bound.json(run / f"case-{number:03d}.json")) == canonical(row), "Standalone case differs")
    if report["status"] != "failed":
        require(len(rows) == len(cases), "Terminal nonfailed report omits cases")
        require((report["status"] == "complete") == all(r["status"] == "complete" for r in rows), "Completion status differs")
    for label, actual in report.get("models", {}).items():
        planned = protocol["models"][label]
        require(actual["name"] == planned["name"] and actual["revision"] == planned["revision"], "Model identity differs")
        require(actual["devices"] == ["cuda:0"] and actual["dtypes"] == [planned["dtype"]], "Saved model placement differs")
    if report["status"] != "failed":
        require(set(report["models"]) == {"encoder", "bge", "qwen"}, "Incomplete model identity receipt")
        require(report["end_verification"]["status"] == "pass", "Successful run lacks end binding")
        require(canonical(report["runtime"]["packages"]) == canonical(protocol["runtime_packages"]), "Saved package runtime differs")
        require(report["runtime"]["device"] == earlier["runtime"]["device"], "GPU identity differs from measured route")
        require(report["warmup"]["status"] == "complete" and report["warmup"]["excluded_from_case_timings"] is True,
                "Warmup outcome or timing exclusion differs")
    counts = recount(cases, rows)
    timing = measured_timing(rows)
    expected = {key: counts[key] for key in ("semantic_suggestions", "all_visible_norms")}
    expected["timing"] = timing
    require(canonical(expected) == canonical(report["aggregate"]), "Stored aggregate differs from independent raw-output recount")
    bound.unchanged()
    return {"format": "grounded_signal_public_baseline_audit/1", "status": "pass", "run_status": report["status"],
            "report_sha256": bound.files[run / "report.json"], "prepared_manifest_sha256": report["prepared_manifest_sha256"],
            "protocol_sha256": report["protocol_sha256"], "primary_manifest_sha256": protocol["primary_manifest_sha256"],
            "producer_source_sha256": protocol["probe_source_sha256"], "auditor_source_sha256": digest(Path(__file__).read_bytes()),
            "checked_archive_files": len(bound.files), "producer_end_verification": report["end_verification"]["status"],
            "counts": counts, "timing": timing, "profile": "treffer", "profile_config": profile,
            "source_identity": identity, "notes": [
                "All 16 planned cases remain in denominators, including failed or unattempted cases; technical errors are not correct abstentions.",
                "Hard norms are independently derived from original catalog-card anchors and risk-mark norm proposals; repeated outputs collapse only per case and norm.",
                "Clear negatives count ANY output, including unadjudicated norms; the separate context case is not relabelled as a clear negative.",
                "The 16 assistant-reviewed synthetic cases are not independent legal gold labels; no expert precision or factual truth is inferred.",
                "This checks archived bytes, declared profile/model identities and raw counts; it does not load or rehash external model weights or repeat the producer's complete source/GPU replay audit.",
                "An audit pass denotes a consistent recount, including any explicitly recorded producer failure; it does not turn failed rows into successful results."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Use a fresh audit directory")
    result = audit(args.run, args.prepared)
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "source.py")
    require(digest((args.out / "source.py").read_bytes()) == result["auditor_source_sha256"], "Auditor changed before saving")
    (args.out / "report.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(canonical({"status": result["status"], "run_status": result["run_status"], "planned_cases": 16}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
