"""Independent, query-free recount of a complete R2 grounded-signal matrix.

Stdlib only; never imports the experiment evaluator, runtime, or a model.
Checks attribution and recorded diagnostics, not legal truth or entailment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import median
from typing import Any


ARMS = ("direct_prepared", "structured_prepared", "structured_retrieved")
LABELS = ("target_norm_ids", "allowed_hint_norm_ids", "no_hint_norm_ids", "insufficient_context_norm_ids")
NONTERMINAL = {"running", "pending", "not_run"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def rawsha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def equal(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid duration")
    return float(value)


def integer(value):
    require(type(value) is int and value >= 0, "Invalid integer accounting")
    return value


def ids(values):
    require(isinstance(values, list) and all(isinstance(v, str) and v for v in values)
            and len(values) == len(set(values)), "Invalid identifier inventory")
    return set(values)


class BoundFiles:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(expected is None or actual == expected, "Bound file hash differs")
        require(path not in self.hashes or self.hashes[path] == actual, "File changed between reads")
        self.hashes[path] = actual
        return actual

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        actual = rawsha(raw)
        require(path not in self.hashes or self.hashes[path] == actual, "File changed between reads")
        self.hashes[path] = actual
        return parse(raw)

    def finish(self):
        for path, expected in list(self.hashes.items()):
            self.bind(path, expected)


def local(directory, relative):
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), "Archive path escapes its root")
    return path


def check_cases(cases):
    require(isinstance(cases, list), "Missing cases")
    ids([c["case_id"] for c in cases])
    families: dict[str, str] = {}
    for case in cases:
        require(case["split"] in {"development", "evaluation"}, "Unknown case split")
        require(families.setdefault(case["family_id"], case["split"]) == case["split"], "Family split leakage")
        labels = case["diagnostic_labels"]
        require(labels["annotation_status"] == "assistant_reviewed_synthetic", "Unexpected label provenance")
        groups = [ids(labels.get(key, [])) for key in LABELS]
        require(not any(a & b for i, a in enumerate(groups) for b in groups[:i]), "Conflicting annotations")
        require(set.union(*groups) <= ids(case["prepared_norm_ids"]), "Labels outside prepared pool")
        equal(labels["hint_expected"], bool(groups[0]), "Hint expectation differs")
    expected = {"all": (16, 12, 14, 3, 1), "development": (8, 6, 8, 1, 1), "evaluation": (8, 6, 6, 2, 0)}
    denominators = {}
    for split, counts in expected.items():
        selected = [c for c in cases if split == "all" or c["split"] == split]
        labels = [c["diagnostic_labels"] for c in selected]
        actual = (len(selected), sum(bool(x[LABELS[0]]) for x in labels), sum(len(x[LABELS[0]]) for x in labels),
                  sum(not x[LABELS[0]] and not x.get(LABELS[3], []) for x in labels),
                  sum(bool(x.get(LABELS[3], [])) for x in labels))
        require(actual == counts, "R2 planned diagnostic denominator differs")
        denominators[split] = dict(zip(("cases", "positive_cases", "target_norm_pairs", "clear_negative_cases", "context_cases"), actual))
    require(len(families) == 8 and all(sum(c["family_id"] == f for c in cases) == 2 for f in families), "Incomplete contrast families")
    return denominators


def terminal_guard(run, report, protocol):
    require(not (run / "abandonment.json").exists() and not (run / "interruption.json").exists(), "Abandoned/interrupted run is not a complete R2 base")
    require(report.get("status") in {"complete", "incomplete"}, "Only terminal R2 bases are accepted")
    require(protocol.get("model_interface", {}).get("format") == "fixed_norm_slots/2"
            and protocol["model_interface"].get("tool_choice") == "required", "R1 or unknown adapter protocol")
    equal(protocol.get("arms"), list(ARMS), "Only the full three-arm primary matrix is accepted")
    require(set(report.get("results", {})) == set(ARMS), "Missing primary arm")
    for rows in report["results"].values():
        require(len(rows) == 16 and all(r.get("status") not in NONTERMINAL and isinstance(r.get("status"), str)
                                       for r in rows), "Incomplete case denominator")
    require(report["status"] == ("complete" if all(r["status"] == "complete" for rows in report["results"].values() for r in rows) else "incomplete"), "Terminal status differs from arm statuses")


def check_package(package, table):
    fingerprint = digest({k: v for k, v in package.items() if k not in {"package_id", "fingerprint"}})
    equal(package["fingerprint"], fingerprint, "Package fingerprint differs")
    equal(package["package_id"], "pkg:" + fingerprint, "Package ID differs")
    require(package["status"] in {"ready", "overflow", "missing"}, "Unknown package status")
    equal(package["complete_local_norms"], package["status"] == "ready", "Package completeness differs")
    ids(package["candidate_norm_ids"])
    equal(package["unit_ids"], [u["unit_id"] for u in package["units"]], "Unit order differs")
    for unit in package["units"]:
        equal(unit, table[unit["unit_id"]], "Original source unit differs")
        require(type(unit["start"]) is int and type(unit["end"]) is int and 0 <= unit["start"] < unit["end"]
                and len(unit["text"]) == unit["end"] - unit["start"], "Invalid unit interval")
    if package["status"] != "ready":
        require(not package["units"], "Atomic overflow contains partial sources")
        return
    equal([g["norm_id"] for g in package["required_groups"]], package["norm_ids"], "Whole-norm inventory differs")
    for group in package["required_groups"]:
        parts = [u for u in package["units"] if u["norm_id"] == group["norm_id"]]
        require(group["policy"] == "all_stored_parent_norm_text" and bool(parts), "Whole-norm policy differs")
        equal(group["unit_ids"], [u["unit_id"] for u in parts], "Whole-norm parts differ")
        require(parts[0]["start"] == 0 and all(a["end"] == b["start"] for a, b in zip(parts, parts[1:])), "Source gap/overlap")
        whole = rawsha("".join(u["text"] for u in parts).encode())
        require(all(u["norm_text_sha256"] == whole for u in parts), "Whole source hash differs")


def shown_sources(package):
    aliases = {u["unit_id"]: f"u{i}" for i, u in enumerate(package["units"], 1)}
    return {
        "candidate_norm_ids": package["candidate_norm_ids"],
        "units": [{"unit_id": aliases[u["unit_id"]], **{k: u[k] for k in ("norm_id", "start", "end", "text")}} for u in package["units"]],
        "whole_norm_groups": [{"norm_id": g["norm_id"], "unit_ids": [aliases[u] for u in g["unit_ids"]]} for g in package["required_groups"]],
        "complete_local_norms": package["complete_local_norms"],
        "reference_closure_complete": package["reference_closure_complete"],
        "legal_completeness": "unassessed",
        "references": [{k: r[k] for k in ("source_norm_id", "target_id", "resolved_norm_id", "exact_target_available")} for r in package["references"]],
        "limitations": package["limitations"],
    }


def http_accounting(call):
    """Count actual requests independently of the client's accepted-call counter."""
    count = Counter({k: 0 for k in ("http_requests", "completion_requests", "completion_responses", "completion_transport_errors", "responses_without_usage", "responses_with_invalid_usage", "response_payload_unavailable", "response_prompt_tokens", "response_completion_tokens", "client_model_calls", "completed_stage_calls")})
    completions = []
    for event in call.get("trace", []):
        require(event["kind"] == "http", "Unknown trace kind")
        count["http_requests"] += 1
        equal(event["request_sha256"], digest(event["request"]), "Request hash differs")
        raw = event.get("raw_response")
        response = None
        if raw is not None:
            equal(event["response_sha256"], rawsha(raw.encode()), "Response hash differs")
            try:
                response = parse(raw)
            except ValueError:
                require(call["status"] == "transport_error" and "error" in event, "Malformed response without recorded transport failure")
        else:
            require(call["status"] != "complete" and ("error" in event or
                    (call["status"] == "transport_error" and "response_sha256" in event)), "Missing response without transport failure")
        if event["path"] != "/v1/chat/completions":
            continue
        count["completion_requests"] += 1
        completions.append((event["request"], response))
        if "response_sha256" not in event:
            count["completion_transport_errors"] += 1
            continue
        count["completion_responses"] += 1
        if not isinstance(response, dict):
            require(call["status"] != "complete", "Complete response is not a JSON object")
            count["response_payload_unavailable"] += 1
            count["responses_without_usage"] += 1
        elif isinstance(response.get("usage"), dict):
            usage = response["usage"]
            if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                count["response_prompt_tokens"] += usage["prompt_tokens"]
                count["response_completion_tokens"] += usage["completion_tokens"]
            else:
                require(call["status"] != "complete", "Complete response has invalid usage")
                count["responses_with_invalid_usage"] += 1
        else:
            count["responses_without_usage"] += 1
    usage = call.get("usage", {})
    count["client_model_calls"] = integer(usage.get("model_calls", 0))
    count["completed_stage_calls"] = int(call["status"] == "complete")
    require(count["client_model_calls"] <= count["completion_responses"] <= count["completion_requests"] <= 1, "Unexpected repeated/fictional model call")
    equal(usage.get("http_requests", 0), count["http_requests"], "HTTP request accounting differs")
    if call["status"] == "complete":
        require(count["client_model_calls"] == 1, "Complete result lacks client-counted model call")
    return dict(count), completions


def check_response(call, request, response, config, alias):
    require(response is not None and response.get("model") == alias, "Complete response model differs")
    require(response.get("truncated") is not True, "Truncated response accepted")
    require(len(response["choices"]) == 1 and response["choices"][0]["finish_reason"] == "tool_calls", "Unfinished structured response")
    message = response["choices"][0]["message"]
    equal(call["message"], message, "Accepted message differs from raw response")
    require(len(message["tool_calls"]) == 1, "Wrong tool-call count")
    function = message["tool_calls"][0]["function"]
    equal(function["name"], request["tools"][0]["function"]["name"], "Wrong function")
    equal(parse(function["arguments"]), call["arguments"], "Arguments differ from raw model response")
    usage = response["usage"]
    prompt = integer(usage["prompt_tokens"])
    completion = integer(usage["completion_tokens"])
    equal(usage["prompt_tokens_details"]["cached_tokens"], 0, "Cold request reused previous case cache")
    require(prompt + request["max_tokens"] <= config["context_tokens"] and completion <= request["max_tokens"], "Native token budget violated")
    events = call["trace"]
    equal([e["path"] for e in events], ["/apply-template", "/tokenize", "/v1/chat/completions"], "Native request path differs")
    equal(events[0]["request"], request, "Template request differs")
    rendered = parse(events[0]["raw_response"])["prompt"]
    equal(events[1]["request"], {"content": rendered, "parse_special": True, "add_special": False}, "Tokenizer request differs")
    tokens = parse(events[1]["raw_response"])["tokens"]
    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens), "Invalid native token inventory")
    equal(len(tokens), prompt, "Full prompt accounting differs")
    equal(call["usage"]["prompt_tokens"], prompt, "Accepted prompt accounting differs")
    equal(call["usage"]["completion_tokens"], completion, "Accepted output accounting differs")


def expanded_facts(items, spans, text):
    by_id = {s["id"]: s for s in spans}
    require(len(by_id) == len(spans), "Duplicate original span")
    for span in spans:
        require(type(span["start"]) is int and type(span["end"]) is int and 0 <= span["start"] < span["end"] <= len(text)
                and text[span["start"]:span["end"]] == span["text"], "Fact span differs from original case")
    ids([f["id"] for f in items])
    require(len(items) <= 8, "Too many facts")
    expanded = []
    for fact in items:
        require(set(fact) == {"id", "span_id", "kind", "polarity", "statement"} and fact["id"] in {f"f{i}" for i in range(1, 9)}
                and fact["kind"] in {"actor", "role", "event", "negation", "time", "context"}
                and fact["polarity"] in {"affirmed", "denied", "uncertain"}
                and isinstance(fact["statement"], str) and 0 < len(fact["statement"].strip()) <= 180, "Invalid attributed fact")
        span = by_id[fact["span_id"]]
        expanded.append({k: fact[k] for k in ("id", "kind", "polarity", "statement")} | {k: span[k] for k in ("start", "end", "text")})
    return expanded


def check_call(call, *, case, package, fact_call, arm, config, alias):
    counts, completions = http_accounting(call)
    if not completions:
        require(call["status"] != "complete", "Complete arm without request")
        return counts
    request, response = completions[0]
    require(len(request["tools"]) == 1 and request["tool_choice"] == "required", "Native forced-tool interface differs")
    equal(request["cache_prompt"], False, "Each stage must be cold")
    equal(request["chat_template_kwargs"], {"enable_thinking": True}, "Thinking policy differs")
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "seed", "reasoning_budget_tokens"):
        equal(request[key], config[key], "Generation configuration differs")
    equal(request["max_tokens"], config["max_output_tokens"], "Output budget differs")
    equal(request["model"], alias, "Requested model differs")
    equal(request["parallel_tool_calls"], False, "Parallel call policy differs")
    equal(request["stream"], False, "Streaming policy differs")
    require(len(request["messages"]) == 2 and [m["role"] for m in request["messages"]] == ["system", "user"], "Unexpected history")
    payload = parse(request["messages"][1]["content"])
    equal(payload["original_case"], case["text"], "Original case changed")
    is_fact = arm == "facts"
    direct = arm == "direct_prepared"
    allowed = {"original_case", "case_spans"} | (set() if is_fact else {"original_sources"})
    if not is_fact and not direct:
        allowed |= {"attributed_facts", "unresolved_case_questions"}
        equal(payload["attributed_facts"], fact_call["answer"]["facts"], "Inherited facts changed")
        equal(payload["unresolved_case_questions"], fact_call["answer"]["unresolved"], "Inherited questions changed")
    require(set(payload) == allowed, "Unexpected model payload field (labels or extra state)")
    if not is_fact:
        equal(payload["original_sources"], shown_sources(package), "Full model source payload differs")
    if call["status"] != "complete":
        return counts
    check_response(call, request, response, config, alias)
    arguments = call["arguments"]
    if is_fact or direct:
        facts = expanded_facts(arguments["facts"], payload["case_spans"], case["text"])
    else:
        facts = fact_call["answer"]["facts"]
    expected = {"facts": facts, "unresolved": arguments["unresolved"]}
    if not is_fact:
        candidates = package["candidate_norm_ids"]
        require(isinstance(arguments["hints"], dict) and set(arguments["hints"]) == set(candidates), "Required norm slots differ")
        aliases = {f"u{i}": u for i, u in enumerate(package["units"], 1)}
        fact_ids = ids([f["id"] for f in facts])
        hints = []
        for norm in candidates:
            h = arguments["hints"][norm]
            require(set(h) == {"status", "relation", "evidence_unit_ids", "supported_fact_ids", "open_conditions"}, "Decision fields differ")
            evidence, support = ids(h["evidence_unit_ids"]), ids(h["supported_fact_ids"])
            require(len(evidence) <= 6 and len(support) <= 8 and support <= fact_ids, "Invalid fact binding")
            require(all(u in aliases and aliases[u]["norm_id"] == norm for u in evidence), "Evidence belongs to different norm")
            require(h["status"] in {"hint", "no_hint", "insufficient_context"}, "Unknown decision status")
            require((h["status"] == "hint" and evidence and support and h["relation"] in {"definition", "duty", "procedure", "exception", "scope"})
                    or (h["status"] != "hint" and h["relation"] == "none"), "Decision relation/status differs")
            hints.append({"norm_id": norm, **h, "evidence_unit_ids": [aliases[u]["unit_id"] for u in h["evidence_unit_ids"]]})
        expected["hints"] = hints
    equal(call["answer"], expected, "Host answer differs from original model arguments and source mapping")
    return counts


def observation(case, row, package):
    labels = case["diagnostic_labels"]
    complete = row["status"] == "complete"
    hints = {h["norm_id"] for h in row["answer"]["hints"] if h["status"] == "hint"} if complete else set()
    unknown = ({h["norm_id"] for h in row["answer"]["hints"] if h["status"] == "insufficient_context"} if complete else set(package["candidate_norm_ids"]))
    return {"split": case["split"], "raw_status": row["status"], "complete": complete,
            "package_status": package["status"], "candidates": set(package["candidate_norm_ids"]),
            "hints": hints, "unknown": unknown, "seconds": number(row["timing"]["elapsed_seconds"]),
            **{key: set(labels.get(key, [])) for key in LABELS}}


def summarize(rows, evaluation_format="grounded_signal_evaluation/2"):
    require(evaluation_format in ("grounded_signal_evaluation/1", "grounded_signal_evaluation/2"),
            "Unknown diagnostic evaluation format")
    positive = [r for r in rows if r[LABELS[0]]]
    negative = [r for r in rows if not r[LABELS[0]]]
    clear = [r for r in negative if not r[LABELS[3]]]
    context = [r for r in rows if r[LABELS[3]]]
    counted_labels: tuple[str, ...] = LABELS
    if evaluation_format == "grounded_signal_evaluation/1":
        counted_labels = LABELS[:3]
    else:
        negative = clear
    def hit(row):
        return bool(row["hints"] & row[LABELS[0]])
    targets = sum(len(r[LABELS[0]]) for r in positive)
    target_hits = sum(len(r["hints"] & r[LABELS[0]]) for r in positive)
    target_unknown = sum(len(r["unknown"] & r[LABELS[0]]) for r in positive)
    hit_cases = sum(hit(r) for r in positive)
    count = {
        "planned_cases": len(rows), "positive_cases": len(positive), "negative_cases": len(negative),
        "execution": dict(Counter("complete" if r["complete"] else "technical_error" for r in rows)),
        "source_package_status": dict(Counter(r["package_status"] for r in rows)),
        "positive_cases_with_any_hint": sum(bool(r["hints"]) for r in positive),
        "positive_cases_with_known_target_hint": hit_cases,
        "known_target_case_recall": hit_cases / len(positive) if positive else None,
        "known_target_case_hit_upper_bound": sum(hit(r) or bool(r["unknown"] & r[LABELS[0]]) for r in positive),
        "known_target_norms": targets, "known_target_norms_emitted": target_hits,
        "known_target_norms_unknown": target_unknown, "known_target_norm_recall": target_hits / targets if targets else None,
        "known_target_norm_hit_upper_bound": target_hits + target_unknown,
        "negative_cases_with_hint": sum(bool(r["hints"]) for r in negative),
        "negative_hint_norm_pairs": sum(len(r["hints"]) for r in negative),
        "negative_cases_technically_unavailable": sum(not r["complete"] for r in negative),
        "negative_cases_with_uncertain_norms": sum(bool(r["unknown"]) for r in negative),
        "negative_case_output_upper_bound": sum(bool(r["hints"] or r["unknown"]) for r in negative),
        "negative_norm_output_upper_bound": sum(len(r["hints"]) + len(r["unknown"]) for r in negative),
        "confirmed_no_hint_negative_cases": sum(r["complete"] and not r["hints"] and not r["unknown"] for r in clear),
        "context_annotated_cases": len(context), "context_annotated_norms": sum(len(r[LABELS[3]]) for r in context),
        "context_annotated_norms_answered_insufficient": sum(len(r["unknown"] & r[LABELS[3]]) for r in context if r["complete"]),
        "diagnostic_negative_norms_emitted": sum(len(r["hints"] & r[LABELS[2]]) for r in rows),
        "unadjudicated_hint_norms": sum(len(r["hints"] - set.union(*(r[k] for k in counted_labels))) for r in rows),
        "output_precision": None, "semantic_truth_verified": False,
        "retrieval_missing_known_targets": sum(len(r[LABELS[0]] - r["candidates"]) for r in positive),
    }
    def negative_group(items):
        return {"planned_cases": len(items), "complete_cases": sum(r["complete"] for r in items),
                "technical_cases": sum(not r["complete"] for r in items),
                "cases_with_hint": sum(bool(r["hints"]) for r in items),
                "hint_norm_pairs": sum(len(r["hints"]) for r in items),
                "cases_with_unknown_norms": sum(bool(r["unknown"]) for r in items)}
    elapsed = [r["seconds"] for r in rows]
    completed_elapsed = [r["seconds"] for r in rows if r["complete"]]
    detail = {"raw_statuses": dict(Counter(r["raw_status"] for r in rows)),
              "clear_negatives": negative_group(clear), "annotated_context_cases": negative_group(context),
              "allowed_extra_hint_norm_pairs": sum(len(r["hints"] & r[LABELS[1]]) for r in rows),
              "context_annotated_norms_emitted_as_hint": sum(len(r["hints"] & r[LABELS[3]]) for r in rows),
              "otherwise_unjudged_hint_norm_pairs": sum(len(r["hints"] - set.union(*(r[k] for k in LABELS))) for r in rows),
              "explicit_no_hint_cases_violated": sum(bool(r["hints"] & r[LABELS[2]]) for r in rows),
              "explicit_no_hint_planned_pairs": sum(len(r[LABELS[2]]) for r in rows),
              "standalone_timing": {"attempts": len(elapsed), "total_seconds": sum(elapsed),
                                    "median_seconds": median(elapsed) if elapsed else None,
                                    "completed_median_seconds": median(completed_elapsed) if completed_elapsed else None}}
    return {"evaluation_counts": count, **detail}


def audit(run, inputs):
    run, inputs = run.resolve(), inputs.resolve()
    bound = BoundFiles()
    report, protocol = bound.read(run / "results.json"), bound.read(run / "protocol.json")
    terminal_guard(run, report, protocol)
    manifest = bound.read(inputs / "input-manifest.json")
    require(manifest.get("format") == "grounded_signal_inputs/1", "Unknown input receipt format")
    require({"cases.json", "packages.json", "source-units.json", "protocol.json"} <= set(manifest["artifacts"]), "Missing required bound inputs")
    equal(protocol["input_manifest_sha256"], bound.hashes[inputs / "input-manifest.json"], "Input receipt differs")
    for name, expected in manifest["artifacts"].items():
        bound.bind(local(inputs, name), expected)
    prepared = bound.read(inputs / "protocol.json")
    for key, expected in prepared.items():
        equal(protocol[key], expected, "Full prepared protocol differs: " + key)
    equal(report["source_end_hashes"], protocol["source_hashes"], "Source end receipt differs")
    for name, expected in protocol["source_hashes"].items():
        bound.bind(local(run / "source", name), expected)
    cases, packages, units = (bound.read(inputs / name) for name in ("cases.json", "packages.json", "source-units.json"))
    denominators = check_cases(cases)
    equal(protocol["case_ids"], [c["case_id"] for c in cases], "Planned case order differs")
    equal(sorted(report["fact_results"]), sorted(protocol["case_ids"]), "Incomplete physical facts stage")
    for package in packages.values():
        check_package(package, units)
    model = bound.read(run / "model-manifest.json")
    equal(model, protocol["model_binding"]["model"], "Model manifest differs")
    equal(bound.hashes[run / "model-manifest.json"], protocol["model_binding"]["model_manifest_sha256"], "Model manifest receipt differs")
    alias, config = protocol["model_binding"]["launch"]["alias"], protocol["configuration"]
    stored = bound.read(run / "evaluation.json")["aggregate"]
    bindings = {"protocol_sha256": digest(protocol), "planned_cases_sha256": digest(cases),
                "results_sha256": digest(report["results"]), "source_packages_sha256": digest(packages), "source_units_sha256": digest(units)}
    for key, expected in bindings.items():
        equal(stored[key], expected, "Canonical evaluation binding differs: " + key)
    equal(stored["planned_cases"], 16, "Stored case denominator differs")
    equal(stored["planned_case_arm_rows"], 48, "Stored arm denominator differs")
    physical_facts: Counter[str] = Counter()
    physical_decisions: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    fact_seconds, decision_seconds = 0.0, 0.0
    observations: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    family_packages: dict[str, str] = {}
    for index, case in enumerate(cases, 1):
        cid = case["case_id"]
        fact = bound.read(run / f"facts-{index:03}.json")
        equal(fact, report["fact_results"][cid], "Physical fact ledger differs")
        require(fact["status"] not in NONTERMINAL, "Nonterminal facts stage")
        physical_facts.update(check_call(fact, case=case, package=None, fact_call=None, arm="facts", config=config, alias=alias))
        fact_elapsed = number(fact["timing"]["elapsed_seconds"])
        fact_seconds += fact_elapsed
        for arm in ARMS:
            rows = report["results"][arm]
            equal([r["case_id"] for r in rows], protocol["case_ids"], "Arm case order/count differs")
            row = rows[index - 1]
            artifact = bound.read(run / f"case-{index:03}-{arm}.json")
            equal(artifact["row"], row, "Standalone row differs from final results")
            call = artifact["call"]
            equal(call["status"], row["status"], "Call status differs")
            equal(call.get("answer"), row["answer"], "Call answer differs")
            equal(call.get("usage", {}), row["usage"], "Call accounting differs")
            equal(row["fact_stage_status"], fact["status"], "Fact-stage status differs")
            pid = protocol["case_packages"][arm][cid]
            package = packages[pid]
            equal(row["source_package_id"], pid, "Package selection differs")
            equal(row["candidate_norm_ids"], package["candidate_norm_ids"], "Candidate selection differs")
            if arm != "structured_retrieved":
                require(family_packages.setdefault(case["family_id"], pid) == pid, "Prepared contrast package changed")
                equal(sorted(package["candidate_norm_ids"]), sorted(case["prepared_norm_ids"]), "Prepared candidates differ")
            if row["status"] == "complete":
                require(package["status"] == "ready", "Complete result from atomic overflow")
            else:
                require(row["answer"] is None, "Technical error contains an accepted answer")
            if package["status"] != "ready":
                equal(row["status"], "source_package_" + package["status"], "Atomic source failure differs")
            counts = check_call(call, case=case, package=package, fact_call=fact, arm=arm, config=config, alias=alias)
            physical_decisions[arm].update(counts)
            elapsed = number(call["timing"]["elapsed_seconds"])
            decision_seconds += elapsed
            equal(row["timing"]["decision_seconds"], elapsed, "Decision timing differs")
            expected_fact = 0.0 if arm == "direct_prepared" else fact_elapsed
            require(abs(number(row["timing"]["fact_seconds"]) - expected_fact) < 1e-8, "Fact time double-counted or dropped")
            require(abs(number(row["timing"]["elapsed_seconds"]) - (elapsed + expected_fact)) < 1e-8, "Standalone timing differs")
            observations[arm].append(observation(case, row, package))
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, rows in observations.items():
        arms[arm] = {}
        for split in ("all", "development", "evaluation"):
            selected = [r for r in rows if split == "all" or r["split"] == split]
            summary = summarize(selected, stored["format"])
            equal(summary["evaluation_counts"], stored["arms"][arm][split], "Independent diagnostic counts differ: " + arm + "/" + split)
            arms[arm][split] = summary
        old_timing = stored["arms"][arm]["timing"]
        equal(old_timing["recorded_attempts"], 16, "Recorded timing denominator differs")
        require(abs(old_timing["total_seconds"] - arms[arm]["all"]["standalone_timing"]["total_seconds"]) < 1e-8, "Stored standalone duration differs")
    wall = number(report["elapsed_seconds"])
    require(wall + 1e-6 >= fact_seconds + decision_seconds, "Recorded wall time is smaller than serial physical stages")
    bound.finish()
    return {"format": "grounded_signal_independent_audit/1", "status": "pass", "run_status": report["status"],
            "source_sha256": rawsha(Path(__file__).read_bytes()),
            "input_manifest_sha256": protocol["input_manifest_sha256"], "bindings": bindings,
            "artifact_receipts": {"results": bound.hashes[run / "results.json"], "protocol": bound.hashes[run / "protocol.json"],
                                  "evaluation": bound.hashes[run / "evaluation.json"], "model_manifest": bound.hashes[run / "model-manifest.json"]},
            "audit_counts": {"files_checked_before_and_after": len(bound.hashes), "cases": 16, "arm_rows": 48, "independently_recounted_subsets": 9},
            "denominators": denominators, "arms": arms,
            "physical_execution": {"fact_calls": dict(physical_facts), "decision_calls": {k: dict(v) for k, v in physical_decisions.items()},
                                   "facts_seconds_once": fact_seconds, "decision_seconds": decision_seconds,
                                   "total_recorded_wall_seconds": wall, "remaining_io_verification_seconds": max(0.0, wall - fact_seconds - decision_seconds)},
            "notes": ["Known assistant-reviewed synthetic diagnostics; no independent legal gold or semantic truth check.",
                      "Allowed/unjudged extra hints are separate; no output precision is inferred.",
                      "Technical failures and insufficient_context remain unknown, never successful negative answers.",
                      "Evaluation v1 included the context-only case in negative_cases. Version 2 counts clear negatives separately.",
                      "Standalone structured-arm latency includes the same historical fact stage; physical totals count it only once.",
                      "Completion attempts, responses, the client model_calls counter and completed stages are separate; model_calls increments before validation. Absent/invalid response usage is unavailable.",
                      "Recorded run wall time excludes model loading and pre-run binding; independent audit does not rehash external model weights.",
                      "Whole source payload and original span links are checked, not the correctness of their interpretation."]}


def markdown(report):
    lines = ["# Unabhängige Greenfield-R2-Nachzählung", "", "Terminale, vollständige 16×3-Matrix; bekannte synthetische Entwicklungsdiagnostik. Kein fachlicher Wahrheitsnachweis und keine Schätzung der Ausgabepräzision.", "",
             "| Arm / Teilmenge | Fertig / geplant | Ziel-Fälle | Zielpaare | Explizite no_hint-Verstöße | Klare Negativfälle mit beliebigem Hinweis | Gesamte Hint-Normpaare auf klaren Negativen | Median s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, subsets in report["arms"].items():
        for split in ("all", "evaluation"):
            item = subsets[split]
            m = item["evaluation_counts"]
            n = item["clear_negatives"]
            lines.append(f"| {arm} / {split} | {m['execution'].get('complete', 0)}/{m['planned_cases']} | {m['positive_cases_with_known_target_hint']}/{m['positive_cases']} | {m['known_target_norms_emitted']}/{m['known_target_norms']} | {m['diagnostic_negative_norms_emitted']} | {n['cases_with_hint']}/{n['planned_cases']} | {n['hint_norm_pairs']} | {item['standalone_timing']['median_seconds']:.2f} |")
    p = report["physical_execution"]
    lines += ["", "Die Spalten zu klar negativen Fällen zählen jeden Hinweis, auch auf nicht annotierte Normen. Null Verstöße auf ausdrücklich benannten no_hint-Normen bedeuten deshalb weder null Ausgaben auf Negativfällen noch belegte Präzision. Die technische Fehlerzahl und der gesonderte Fall mit annotiertem Kontextmangel bleiben im JSON vollständig sichtbar. Fehlende Antworten zählen weder als Zurückweisung noch als richtige Enthaltung.", "",
              f"Physisch: {p['fact_calls']['completion_requests']} Fakten-Anfragen plus {sum(x['completion_requests'] for x in p['decision_calls'].values())} Entscheidungs-Anfragen. Aufgezeichnete Gesamtdauer {p['total_recorded_wall_seconds']:.2f} s, davon Fakten einmalig {p['facts_seconds_once']:.2f} s und Entscheidungen {p['decision_seconds']:.2f} s. Die Armmediane enthalten bei strukturierten Armen deren geteilte Faktenstufe; diese Zeiten werden nicht zu einer physischen Gesamtdauer addiert.", "",
              "Die Nachzählung stimmt mit sämtlichen gespeicherten Fall-/Normzählern und dem vollständigen kanonischen Protokollhash überein. Quellenzuordnung bedeutet keine bestätigte Begründung oder Rechtsanwendung.", "",
              f"Protokoll (kanonisch): `{report['bindings']['protocol_sha256']}`. Ergebnisse (Datei): `{report['artifact_receipts']['results']}`. Auditor: `{report['source_sha256']}`.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Use a fresh audit output directory")
    report = audit(args.run, args.inputs)
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(markdown(report), encoding="utf-8")
    shutil.copyfile(__file__, args.out / "auditor-source.py")
    print(json.dumps({"status": report["status"], "cases": 16, "arm_rows": 48, "out": str(args.out.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
