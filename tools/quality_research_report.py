"""Audit frozen agent experiments and write query-free aggregate research reports.

No agent, model, corpus session or server is called. Source availability means
pages actually present in the frozen tool transcript, not legal applicability.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import types


DATASETS = ("main", "additional")
ARMS = ("static", "research")
TOOLS = ("search_sources", "read_source", "follow_citations", "finish")
STATUSES = {"complete", "timeout", "transport_error", "context_limit", "invalid_completion", "output_limit",
            "source_error", "tool_limit", "invalid_finish", "tool_error", "round_limit", "runner_error"}
CASE_FIELDS = ("case_id", "dataset", "group", "query", "targets", "negative", "category", "uncertain")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def read_bound(path, bindings, expected=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = sha(raw)
    require(expected is None or value == expected, "Frozen JSON checksum differs")
    bindings[path] = value
    return parse(raw), value


def safe_path(directory, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
            "Invalid frozen relative path")
    path = directory / relative
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
            and path.resolve().is_relative_to(directory.resolve()), "Frozen path escapes experiment")
    return path


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def quantiles(values):
    require(all(number(value) for value in values), "Invalid measured timing")
    values = sorted(values)
    if not values:
        return {"n": 0, "total": 0., "mean": None, "median": None, "p95": None, "min": None, "max": None}
    offset = .95 * (len(values) - 1)
    low, high = math.floor(offset), math.ceil(offset)
    return {"n": len(values), "total": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": values[low] + (values[high] - values[low]) * (offset - low),
            "min": values[0], "max": values[-1]}


def result_for(row, arm):
    return {"status": "complete", **row["baseline"]} if arm == "baseline" else row["arms"].get(arm, {}).get("result")


def accepted_ids(row, arm):
    value = result_for(row, arm)
    if value is None or value["status"] != "complete":
        return []
    return [norm["id"] for focus in value["suggestions"] for norm in focus["normen"]]


def metric_counts(rows, arm, diagnostic):
    result = {key: 0 for key in ("cases", "complete_cases", "failed_cases", "pending_cases", "positive_cases",
        "negative_cases", "failed_positive_cases", "failed_negative_cases", "pending_positive_cases",
        "pending_negative_cases", "hit5_cases", "historical_norm_targets", "matched_historical_norm_targets",
        "negative_with_suggestion", "negative_entries", "negative_unique_case_norm_pairs", "unresolved_cases")}
    for row in rows:
        result["cases"] += 1
        side = "negative" if row["negative"] else "positive"
        result[side + "_cases"] += 1
        value = result_for(row, arm)
        state = "pending" if value is None else "complete" if value["status"] == "complete" else "failed"
        result[state + "_cases"] += 1
        if state != "complete":
            result[state + "_" + side + "_cases"] += 1
        elif value.get("unresolved"):
            result["unresolved_cases"] += 1
        entries = accepted_ids(row, arm)
        if row["negative"]:
            result["negative_with_suggestion"] += int(bool(entries))
            result["negative_entries"] += len(entries)
            result["negative_unique_case_norm_pairs"] += len(set(entries))
        else:
            matched = set(entries) & set(row["targets"])
            result["hit5_cases"] += int(bool(matched))
            result["historical_norm_targets"] += len(set(row["targets"]))
            result["matched_historical_norm_targets"] += len(matched)
    p, n = result["positive_cases"], result["negative_cases"]
    unknown = result["failed_negative_cases"] + result["pending_negative_cases"]
    result["rates"] = None if diagnostic else {"hit5_lower_bound": result["hit5_cases"] / p if p else None,
        "negative_output_lower_bound": result["negative_with_suggestion"] / n if n else None,
        "negative_output_upper_bound": (result["negative_with_suggestion"] + unknown) / n if n else None}
    result["suggestion_precision"] = {"status": "unknown", "reason": "Historical positive-case target lists are incomplete."}
    return result


def output_load(rows, arm):
    value = dict.fromkeys(("output_cases", "output_entries", "unique_case_norm_pairs", "positive_entries",
                          "negative_entries", "unadjudicated_positive_case_norm_pairs"), 0)
    for row in rows:
        entries = accepted_ids(row, arm)
        value["output_cases"] += int(bool(entries))
        value["output_entries"] += len(entries)
        value["unique_case_norm_pairs"] += len(set(entries))
        value["negative_entries" if row["negative"] else "positive_entries"] += len(entries)
        if not row["negative"]:
            value["unadjudicated_positive_case_norm_pairs"] += len(set(entries) - set(row["targets"]))
    return value


def transition(rows, before, after):
    totals = dict.fromkeys(("cases", "positive_cases", "gained_hit_cases", "lost_hit_cases", "gained_target_pairs",
                           "lost_target_pairs", "added_negative_norm_pairs", "removed_negative_norm_pairs"), 0)
    paired = dict(totals)
    unavailable = {"origin_cases": 0, "destination_cases": 0}
    for row in rows:
        left, right = result_for(row, before), result_for(row, after)
        known_left = left is not None and left["status"] == "complete"
        known_right = right is not None and right["status"] == "complete"
        unavailable["origin_cases"] += not known_left
        unavailable["destination_cases"] += not known_right
        a, b = set(accepted_ids(row, before)), set(accepted_ids(row, after))
        for value in (totals, paired) if known_left and known_right else (totals,):
            value["cases"] += 1
            if row["negative"]:
                value["added_negative_norm_pairs"] += len(b - a)
                value["removed_negative_norm_pairs"] += len(a - b)
            else:
                value["positive_cases"] += 1
                a_targets, b_targets = a & set(row["targets"]), b & set(row["targets"])
                value["gained_hit_cases"] += bool(b_targets) and not a_targets
                value["lost_hit_cases"] += bool(a_targets) and not b_targets
                value["gained_target_pairs"] += len(b_targets - a_targets)
                value["lost_target_pairs"] += len(a_targets - b_targets)
    return {"all_selected_output_changes": totals, "both_complete_changes": paired,
            "unavailable": unavailable,
            "note": "All-selected changes treat unavailable output as absent; only both-complete changes compare two completed decisions."}


def audit_quote_references(result, ledger):
    """Bind model-facing quote IDs to lossless intervals of supplied pages."""
    if "quote_references" not in result:
        return None  # Earlier frozen protocols used directly copied quotations.
    quotes, aliases = result["quote_references"], result.get("evidence_aliases")
    require(isinstance(quotes, dict) and isinstance(aliases, dict), "Malformed quote reference ledger")
    require(all(isinstance(alias, str) and re.fullmatch(r"e[1-9][0-9]*", alias)
                and isinstance(identity, str) and identity in ledger for alias, identity in aliases.items())
            and len(set(aliases.values())) == len(aliases), "Invalid source alias binding")
    intervals = {alias: [] for alias in aliases}
    for quote_id, quote in quotes.items():
        require(isinstance(quote_id, str), "Invalid quote identifier type")
        match = re.fullmatch(r"(e[1-9][0-9]*)q([1-9][0-9]*)", quote_id)
        require(match is not None and match[1] in aliases and isinstance(quote, dict), "Invalid quote alias")
        alias = match[1]
        page = ledger[aliases[alias]]
        start, end, absolute = quote.get("page_start"), quote.get("page_end"), quote.get("source_start")
        require(type(start) is int and type(end) is int and type(absolute) is int
                and 0 <= start < end <= len(page["text"]) and end - start <= 240
                and quote.get("evidence_id") == aliases[alias] and quote.get("norm_id") == page["norm_id"]
                and quote.get("text") == page["text"][start:end] and absolute == page["source_start"] + start,
                "Quote reference differs from the supplied page interval")
        intervals[alias].append((int(match[2]), start, end))
    for alias, spans in intervals.items():
        cursor = 0
        for number, (ordinal, start, end) in enumerate(sorted(spans), 1):
            require(ordinal == number and start == cursor, "Quote references do not partition the original page")
            cursor = end
        require(cursor == len(ledger[aliases[alias]]["text"]), "Quote references omit original page text")
    return quotes


def audit_case_quote_references(result, text):
    if "case_quote_references" not in result:
        return None
    quotes = result["case_quote_references"]
    require(isinstance(quotes, dict), "Malformed case quote ledger")
    focuses = {focus["id"]: focus for focus in result["focuses"]}
    require(len(focuses) == len(result["focuses"]), "Duplicate original focus identifier")
    intervals = {identity: [] for identity in focuses}
    for quote_id, quote in quotes.items():
        require(isinstance(quote_id, str) and isinstance(quote, dict), "Invalid case quote record")
        match = re.fullmatch(r"(f[0-9]+)q([1-9][0-9]*)", quote_id)
        require(match is not None and match[1] in focuses, "Unknown case quote focus")
        focus = focuses[match[1]]
        start, end = quote.get("start"), quote.get("end")
        require(type(start) is int and type(end) is int and focus["start"] <= start < end <= focus["end"]
                and end - start <= 240 and quote.get("focus_id") == match[1] and quote.get("text") == text[start:end],
                "Case quote differs from its original document interval")
        intervals[match[1]].append((int(match[2]), start, end))
    for identity, spans in intervals.items():
        cursor = focuses[identity]["start"]
        for number, (ordinal, start, end) in enumerate(sorted(spans), 1):
            require(ordinal == number and start == cursor, "Case quotes do not partition the original focus")
            cursor = end
        require(cursor == focuses[identity]["end"], "Case quote ledger omits original focus text")
    return quotes


def audit_quote_selection(result, finish, quotes, case_quotes):
    """Check host expansion and the exact selected occurrence in final output."""
    arguments, expanded = finish.get("arguments"), finish.get("validated_arguments")
    require(isinstance(arguments, dict) and set(arguments) == {"findings", "unresolved"}
            and isinstance(arguments["findings"], list) and isinstance(arguments["unresolved"], list),
            "Missing model quote selection arguments")
    expected = {"findings": [], "unresolved": arguments["unresolved"]}
    focuses = {focus["id"]: (focus["start"], focus["end"]) for focus in result["focuses"]}
    require(len(focuses) == len(result["focuses"]), "Duplicate focus identifier")
    outputs = {(focus["start"], focus["end"], norm["id"]): norm["research"]
               for focus in result["suggestions"] for norm in focus["normen"]}
    selected, duplicates = set(), []
    for finding in arguments["findings"]:
        source_field = "source_quote_id" if case_quotes is not None else "source_quote"
        require(isinstance(finding, dict) and set(finding) == {"focus_id", "norm_id", "quote_id", source_field, "reason"}
                and all(isinstance(value, str) for value in finding.values()), "Invalid model quote selection")
        require(finding["quote_id"] in quotes and finding["focus_id"] in focuses, "Unknown selected quote or focus")
        quote = quotes[finding["quote_id"]]
        require(quote["norm_id"] == finding["norm_id"], "Selected quote belongs to another norm")
        case_quote = None
        if case_quotes is not None:
            require(finding["source_quote_id"] in case_quotes, "Unknown selected case quote")
            case_quote = case_quotes[finding["source_quote_id"]]
        key = (*focuses[finding["focus_id"]], finding["norm_id"])
        if case_quotes is not None and key in selected:
            duplicates.append(finding)
            continue  # Both supplied references were checked before host deduplication.
        item = {key: value for key, value in finding.items() if key not in ("quote_id", "source_quote_id")}
        item.update(evidence_id=quote["evidence_id"], norm_quote=quote["text"], norm_quote_start=quote["source_start"])
        if case_quote is not None:
            item.update(source_quote=case_quote["text"], source_quote_start=case_quote["start"])
        expected["findings"].append(item)
        require(key in outputs and key not in selected, "Quote selection and final findings differ")
        selected.add(key)
        output = outputs[key]
        require(output.get("evidence_id") == quote["evidence_id"]
                and output.get("norm_quote") == {"start": quote["source_start"],
                    "end": quote["source_start"] + len(quote["text"]), "text": quote["text"]}
                and output.get("source_quote", {}).get("text") == item["source_quote"]
                and output.get("reason") == finding["reason"], "Final finding differs from the selected quote occurrence")
        if case_quote is not None:
            require(output.get("source_quote") == {"start": case_quote["start"], "end": case_quote["end"], "text": case_quote["text"]},
                    "Final finding differs from the selected case quote occurrence")
    require(selected == set(outputs) and result.get("unresolved") == expected["unresolved"], "Final result differs from quote selections")
    require(canonical(expanded) == canonical(expected), "Validated finish differs from host quote expansion")
    if case_quotes is not None:
        require(finish.get("merged_duplicate_findings") == duplicates, "Dropped duplicate findings differ from first-occurrence rule")


def audit_sources(row, arm, corpus_identity):
    result = result_for(row, arm)
    if result is None or result["status"] == "runner_error":
        return dict.fromkeys(("initial_pages", "initial_norms", "new_pages", "new_norms", "citation_only_norms",
                             "selected_entries_using_new_pages", "known_targets_newly_available"), 0)
    initial, ledger, citation_ids = set(), {}, set()
    def add(page):
        require(isinstance(page, dict), "Malformed returned source page")
        start, end = page.get("source_start"), page.get("source_end")
        require(type(start) is int and type(end) is int and 0 <= start < end
                and type(page.get("source_length")) is int and end <= page["source_length"]
                and isinstance(page.get("text"), str) and len(page["text"]) == end - start,
                "Invalid returned source page interval")
        expected = "ev:" + sha(canonical([corpus_identity, sha(row["query"]), page["norm_id"],
                                          page["text_sha256"], start, end]))
        require(page.get("evidence_id") == expected
                and page.get("source_fingerprint") == corpus_identity["source_fingerprint"]
                and page.get("graph_fingerprint") == corpus_identity["graph_fingerprint"],
                "Returned evidence identity differs from this document/corpus")
        if expected in ledger:
            require(all(ledger[expected][key] == page[key] for key in
                        ("norm_id", "text", "source_start", "source_end", "text_sha256")), "Evidence identity collision")
        ledger.setdefault(expected, page)
        return expected
    pages = result["initial_sources"]
    if result.get("initial_sources_sha256") is not None:
        require(sha(canonical(pages)) == result["initial_sources_sha256"], "Initial source hash differs")
    require(len(pages) <= 6, "Initial source cap differs")
    for page in pages:
        initial.add(add(page))
    initial_norms = {ledger[key]["norm_id"] for key in initial}
    for event in result["trace"]:
        if event.get("kind") != "tool" or event["name"] == "finish":
            continue
        require(arm == "research", "Static arm executed a research tool")
        output = event.get("result", {})
        if event["name"] == "search_sources":
            for page in output.get("results", []):
                add(page)
        elif event["name"] == "read_source" and output.get("source") is not None:
            add(output["source"])
        elif event["name"] == "follow_citations":
            citation_ids.update(item["norm_id"] for item in output.get("citations", []))
    new_pages = set(ledger) - initial
    all_norms = {page["norm_id"] for page in ledger.values()}
    quotes = audit_quote_references(result, ledger)
    case_quotes = audit_case_quote_references(result, row["query"])
    require(case_quotes is None or quotes is not None, "Case quote ledger requires the source quote ledger")
    selected_new = 0
    accepted_finishes = [event for event in result["trace"] if event.get("kind") == "tool"
                         and event.get("name") == "finish" and event.get("result", {}).get("accepted") is True]
    if result["status"] == "complete":
        require(len(accepted_finishes) == 1
                and accepted_finishes[0]["result"].get("suggestions") == result["suggestions"],
                "Completed output differs from the accepted finish trace")
        require(all(type(f.get("start")) is int and type(f.get("end")) is int
                    and 0 <= f["start"] < f["end"] <= len(row["query"])
                    and row["query"][f["start"]:f["end"]] == f.get("text") for f in result["focuses"]),
                "Malformed original focus interval")
        focuses = {(f["start"], f["end"]): f["text"] for f in result["focuses"]}
        require(len(focuses) == len(result["focuses"]), "Duplicate original focus")
        seen = set()
        for focus in result["suggestions"]:
            key = (focus["start"], focus["end"])
            require(type(key[0]) is int and type(key[1]) is int and key in focuses and key not in seen and focus["satz"] == focuses[key]
                    and row["query"][key[0]:key[1]] == focus["satz"] and 1 <= len(focus["normen"]) <= 5,
                    "Invalid completed focus or output cap")
            seen.add(key)
            require(len({norm["id"] for norm in focus["normen"]}) == len(focus["normen"]), "Duplicate norm per focus")
            for norm in focus["normen"]:
                research = norm["research"]
                evidence_id = research["evidence_id"]
                require(evidence_id in ledger and ledger[evidence_id]["norm_id"] == norm["id"],
                        "Selected norm was never supplied as a source page")
                page = ledger[evidence_id]
                evidence = norm["evidenz"]
                require(evidence.get("fingerprint") == page["source_fingerprint"]
                        and evidence.get("anker") == [] and evidence.get("passagen") == [{
                            "id": f"{norm['id']}@{page['source_start']}:{page['source_end']}", "norm_id": norm["id"],
                            "quell_start": page["source_start"], "quell_end": page["source_end"], "text": page["text"]}],
                        "Output passage differs from its actually returned source page")
                for quote, source, offset in ((research["source_quote"], row["query"], 0),
                                              (research["norm_quote"], page["text"], page["source_start"])):
                    start, end = quote["start"], quote["end"]
                    require(type(start) is int and type(end) is int and offset <= start < end <= offset + len(source)
                            and source[start-offset:end-offset] == quote["text"], "Completed quote does not match returned source")
                require(research.get("legal_validity_verified") is False, "Source verification became a legal claim")
                selected_new += evidence_id in new_pages
        if quotes is not None:
            audit_quote_selection(result, accepted_finishes[0], quotes, case_quotes)
    else:
        require(not accepted_finishes, "Technical failure contains an accepted final result")
    return {"initial_pages": len(initial), "initial_norms": len(initial_norms), "new_pages": len(new_pages),
            "new_norms": len(all_norms - initial_norms), "citation_only_norms": len(citation_ids - all_norms),
            "selected_entries_using_new_pages": selected_new,
            "known_targets_newly_available": len((all_norms - initial_norms) & set(row["targets"]))}


def execution_policy(config):
    """Older archives omit both fields; new protocols declare both exactly."""
    require(isinstance(config, dict), "Malformed agent configuration")
    names = {"reuse_prompt_cache", "finish_reserve_seconds"}
    present = names.intersection(config)
    require(not present or present == names, "Incomplete execution policy configuration")
    if not present:
        return False, 0.
    reuse, reserve = config["reuse_prompt_cache"], config["finish_reserve_seconds"]
    require(type(reuse) is bool and number(reserve), "Invalid execution policy configuration")
    return reuse, reserve


def trace_counts(result):
    tools, finishes, errors, tool_errors, recovery = Counter(), Counter(), Counter(), Counter(), Counter()
    budgets = Counter()
    merged_count = 0
    http_count = 0
    completion_count = 0
    pending_time_finish = False
    reuse_cache, finish_reserve = execution_policy(result.get("config", {}))
    cache_counts = Counter()
    completion_positions = [index for index, event in enumerate(result.get("trace", []))
                            if event.get("kind") == "http" and event.get("path") == "/v1/chat/completions"]
    for position, event in enumerate(result.get("trace", [])):
        if event.get("kind") == "budget":
            require(type(event.get("round")) is int and event["round"] >= 0, "Invalid budget round")
            if event.get("reason") == "context_finish_reserve":
                require(type(event.get("prompt_tokens_before_finish")) is int and event["prompt_tokens_before_finish"] >= 0,
                        "Invalid context budget event")
            elif event.get("reason") == "time_finish_reserve":
                require(set(event) <= {"kind", "reason", "round", "remaining_seconds", "finish_reserve_seconds", "elapsed_seconds"}
                        and number(event.get("remaining_seconds")) and number(event.get("finish_reserve_seconds"))
                        and finish_reserve > 0 and event["finish_reserve_seconds"] == finish_reserve
                        and 30 <= event["remaining_seconds"] <= 2 * finish_reserve
                        and result.get("allow_research") is True and not pending_time_finish,
                        "Time reserve event differs from the declared execution policy")
                pending_time_finish = True
            else:
                raise ValueError("Unknown budget event")
            if "elapsed_seconds" in event:
                require(number(event["elapsed_seconds"]), "Invalid measured budget event time")
            budgets[event["reason"]] += 1
            continue  # This decision marker has no implied or fabricated duration.
        require(number(event.get("elapsed_seconds")), "Missing measured trace time")
        if event.get("kind") == "http":
            http_count += 1
            require(event.get("path") in ("/apply-template", "/tokenize", "/v1/chat/completions"), "Unknown model endpoint")
            require(sha(canonical(event["request"])) == event["request_sha256"], "HTTP request hash differs")
            if pending_time_finish and event["path"] in ("/apply-template", "/v1/chat/completions"):
                available = event["request"].get("tools")
                require(isinstance(available, list) and len(available) == 1 and isinstance(available[0], dict)
                        and available[0].get("type") == "function" and available[0].get("function", {}).get("name") == "finish"
                        and event["request"].get("tool_choice") == {"type": "function", "function": {"name": "finish"}},
                        "Time reserve did not force the declared finish-only request")
                if event["path"] == "/v1/chat/completions":
                    pending_time_finish = False
            if event["path"] == "/v1/chat/completions":
                require(event["request"].get("cache_prompt") is (reuse_cache and completion_count > 0),
                        "Completion prompt-cache policy differs from this case/arm protocol")
                cache_counts["reuse_requested_completions" if event["request"]["cache_prompt"] else "cold_completions"] += 1
                completion_count += 1
            if "raw_response" in event:
                require(sha(event["raw_response"]) == event["response_sha256"], "HTTP response hash differs")
                if event["path"] == "/v1/chat/completions":
                    try:
                        value = parse(event["raw_response"])
                    except ValueError:
                        require(result["status"] != "complete", "Successful run includes invalid HTTP JSON")
                        cache_counts["unavailable_accounting_completions"] += 1
                    else:
                        require(isinstance(value, dict) or result["status"] != "complete", "Completed HTTP response is not an object")
                        if reuse_cache and not isinstance(value, dict):
                            cache_counts["unavailable_accounting_completions"] += 1
                        elif reuse_cache:
                            usage = value.get("usage", {}) if isinstance(value, dict) else {}
                            usage = usage if isinstance(usage, dict) else {}
                            details = usage.get("prompt_tokens_details", {})
                            cached = details.get("cached_tokens") if isinstance(details, dict) else None
                            prompt = usage.get("prompt_tokens")
                            valid = (type(cached) is int and type(prompt) is int and 0 <= cached <= prompt
                                     and (event["request"]["cache_prompt"] or cached == 0))
                            if valid:
                                cache_counts["accounted_completions"] += 1
                                cache_counts["cached_prompt_tokens"] += cached
                            else:
                                require(result["status"] == "invalid_completion" and position == completion_positions[-1],
                                        "Prompt-cache accounting is missing or invalid in a completed decision")
                                cache_counts["invalid_accounting_completions"] += 1
                        else:
                            cache_counts["accounting_not_required_completions"] += 1
                        choices = value.get("choices", []) if isinstance(value, dict) else []
                        for choice in choices if isinstance(choices, list) else []:
                            if not isinstance(choice, dict):
                                require(result["status"] != "complete", "Malformed successful completion")
                                continue
                            name = choice.get("finish_reason")
                            finishes[name if name in ("stop", "tool_calls", "length") else "other"] += 1
            elif event["path"] == "/v1/chat/completions":
                require(result["status"] != "complete", "Completed run has no raw completion response")
                cache_counts["unavailable_accounting_completions"] += 1
        elif event.get("kind") == "tool":
            require(event.get("name") in TOOLS, "Unknown tool in completed transcript")
            tools[event["name"]] += 1
            if "merged_duplicate_findings" in event:
                require(event["name"] == "finish" and isinstance(event["merged_duplicate_findings"], list)
                        and all(isinstance(item, dict) for item in event["merged_duplicate_findings"]), "Invalid duplicate normalization trace")
                merged_count += len(event["merged_duplicate_findings"])
            error = event.get("result", {}).get("error")
            if error:
                allowed = {"ResearchBudgetExceeded", "ResearchSourceError", "ValueError", "TypeError", "KeyError", "OSError"}
                tool_errors[error if error in allowed else "other"] += 1
                if event["name"] == "finish":
                    recovery["finish_validation_errors"] += 1
        else:
            raise ValueError("Unknown trace event")
    merge_errors = 0
    for error in result.get("errors", []):
        kind = error.get("kind") if isinstance(error, dict) else "runner_error"
        errors[kind if kind in STATUSES | {"missing_tool_call", "duplicate_norm_merged"} else "other"] += 1
        if kind == "duplicate_norm_merged":
            require(type(error.get("count")) is int and error["count"] > 0 and error.get("recoverable") is True,
                    "Invalid duplicate normalization warning")
            merge_errors += error["count"]
        recovery["recoverable_errors" if isinstance(error, dict) and error.get("recoverable") is True else "fatal_errors"] += 1
    require(merged_count == merge_errors, "Duplicate normalization warning count differs from transcript")
    require(not pending_time_finish or result["status"] != "complete", "Completed time-reserve decision has no completion request")
    if merged_count:
        recovery["merged_duplicate_findings"] = merged_count
    if result["status"] != "runner_error":
        usage = result.get("usage")
        require(isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
            for key in ("prompt_tokens", "completion_tokens", "model_calls", "http_requests")), "Invalid measured token/call usage")
        require(usage["http_requests"] == http_count, "HTTP request denominator differs from transcript")
    return {"tool_calls": dict(tools), "completion_finish_reasons": dict(finishes), "errors": dict(errors),
            "tool_result_errors": dict(tool_errors), "repair_and_failure_counts": dict(recovery), "budget_events": dict(budgets),
            "prompt_cache": dict(cache_counts)}


def validate_journal(report, events):
    rows, arms = report["rows"], report["arms"]
    expected, planned = [], []
    for row in rows:
        require(set(row["arms"]) <= set(arms), "Undeclared arm result")
        for arm in arms:
            planned.append((row["case_id"], arm))
            if arm in row["arms"]:
                expected.append({"case_id": row["case_id"], "arm": arm, **row["arms"][arm]})
    require(events == expected, "Journal and report results differ")
    require([(e["case_id"], e["arm"]) for e in events] == planned[:len(events)], "Journal is not a gap-free case/arm prefix")
    failures = sum(event["result"]["status"] != "complete" for event in events)
    if report["status"] in ("complete", "complete_with_errors"):
        require(len(events) == len(planned) and report.get("execution_complete") is True
                and report.get("technical_failures") == failures
                and (report["status"] == "complete") == (failures == 0), "Completed experiment omits results or technical failures")
    return len(events)


def public_check(value):
    forbidden = {"query", "case_id", "case_ids", "rows", "targets", "suggestions", "trace", "raw_response",
                 "source_quote", "norm_quote", "text", "source_directory", "model_path", "argv", "errors_raw",
                 "catalog_hints", "catalog_metadata", "concepts", "matches", "norm_anchors",
                 "definition_references", "norm_chains", "catalog_query_vector", "query_vector_sha256"}
    if isinstance(value, dict):
        require(not forbidden.intersection(value), "Private payload entered public aggregate")
        for item in value.values():
            public_check(item)
    elif isinstance(value, list):
        for item in value:
            public_check(item)


RETRIEVAL_COUNTS = {"model_load_count", "query_count", "encode_count", "failed_queries"}
RETRIEVAL_TIMES = {"model_validation_seconds", "model_load_seconds", "encode_seconds", "retrieval_seconds"}
QUERY_VECTOR_FORMAT = {"dtype": "float32", "dimension": 1024, "byte_order": "little", "order": "C"}


def checked_retrieval_metrics(value, *, vector_journal=False):
    require(isinstance(value, dict) and set(value) == RETRIEVAL_COUNTS | RETRIEVAL_TIMES | {"queries"},
            "Missing or unknown hybrid metric field")
    require(all(type(value[key]) is int and value[key] >= 0 for key in RETRIEVAL_COUNTS)
            and all(number(value[key]) for key in RETRIEVAL_TIMES) and isinstance(value["queries"], list),
            "Invalid hybrid measurements")
    queries = value["queries"]
    for event in queries:
        fields = {"number", "query_sha256", "characters", "limit", "status",
                  "model_load_seconds", "encode_seconds", "retrieval_seconds", "encode_count"}
        if vector_journal:
            fields.add("query_vector_sha256")
        require(isinstance(event, dict) and set(event) == fields
                and type(event["number"]) is int and event["number"] > 0
                and isinstance(event["query_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", event["query_sha256"])
                and type(event["characters"]) is int and event["characters"] > 0
                and type(event["limit"]) is int and 1 <= event["limit"] <= 6
                and event["status"] in ("complete", "failed")
                and type(event["encode_count"]) is int and event["encode_count"] in (0, 1)
                and all(number(event[key]) for key in ("model_load_seconds", "encode_seconds", "retrieval_seconds")),
                "Malformed hybrid query journal")
        if vector_journal:
            vector_sha = event["query_vector_sha256"]
            require((vector_sha is None and event["status"] == "failed")
                    or (isinstance(vector_sha, str) and re.fullmatch(r"[0-9a-f]{64}", vector_sha)
                        and event["encode_count"] == 1), "Invalid hybrid query-vector journal binding")
    require(value["query_count"] == len(queries)
            and value["failed_queries"] == sum(event["status"] == "failed" for event in queries)
            and value["encode_count"] == sum(event["encode_count"] for event in queries), "Hybrid query denominators differ")
    for key in ("encode_seconds", "retrieval_seconds"):
        require(math.isclose(value[key], sum(event[key] for event in queries), abs_tol=1e-8, rel_tol=1e-9),
                "Hybrid query time differs from journal")
    return {key: value[key] for key in sorted(RETRIEVAL_COUNTS | RETRIEVAL_TIMES)}


def audit_retrieval(report, protocol, directory):
    if "retrieval" not in report:
        require("retrieval" not in protocol, "Missing frozen retrieval setting")
        return {"mode": "bm25", "measurement_scope": "Legacy source-tool timings; no encoder was used."}
    for key in ("retrieval", "encoder_provenance", "corpus"):
        require(report[key] == protocol.get(key), "Frozen retrieval/corpus identity changed")
    retrieval, provenance = report["retrieval"], report["encoder_provenance"]
    mode = retrieval.get("mode")
    require(mode in ("bm25", "hybrid") and provenance.get("mode") == mode, "Invalid retrieval mode")
    entries = [row["arms"][arm] for row in report["rows"] for arm in report["arms"] if arm in row["arms"]]
    if mode == "bm25":
        require(provenance == {"mode": "bm25", "files": {}} and not retrieval.get("model_identity")
                and ("torch_imported" not in report or report["torch_imported"] is False), "BM25 unexpectedly used an encoder")
        require(report.get("retrieval_setup_metrics") == report.get("retrieval_metrics") == {}
                and all(entry.get("retrieval_metrics") == {} for entry in entries), "BM25 has unexplained encoder measurements")
        return {"mode": "bm25", "measurement_scope": "Source-tool timings; no encoder was used."}
    require("risk_layer/research_hybrid.py" in report["source_hashes"], "Unbound hybrid retrieval implementation")
    vector_journal = "format" in retrieval or "query_vector" in retrieval
    if vector_journal:
        require(retrieval.get("format") == "research_cpu_hybrid/2"
                and canonical(retrieval.get("query_vector")) == canonical(QUERY_VECTOR_FORMAT),
                "Unknown hybrid query-vector format")
    model = retrieval.get("model_identity", {})
    files = provenance.get("files")
    require(isinstance(files, dict) and bool(files) and model.get("files") == files
            and model.get("encoder") == "bge-m3" and model.get("device") == "cpu" and model.get("dtype") == "float32"
            and type(model.get("cpu_threads")) is int and model["cpu_threads"] == 2 and model.get("local_files_only") is True
            and isinstance(model.get("revision"), str) and re.fullmatch(r"[0-9a-f]{40}", model["revision"]),
            "Hybrid model identity or CPU policy differs")
    snapshot_name = provenance["snapshot_directory"]
    snapshot_path = safe_path(directory, snapshot_name)
    require(isinstance(retrieval.get("model_path"), str) and Path(retrieval["model_path"]).resolve() == snapshot_path.resolve(),
            "Hybrid model was not loaded from its bound snapshot")
    expected_files = {snapshot_name + "/" + name: expected for name, expected in files.items()}
    actual_files = {name: expected for name, expected in report["snapshot_hashes"].items() if name.startswith(snapshot_name + "/")}
    require(expected_files == actual_files, "Hybrid encoder snapshot inventory differs")
    for name, expected in files.items():
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid encoder file hash")
        safe_path(snapshot_path, name)
    require(retrieval.get("evidence_fingerprint") == report["corpus"]["index_fingerprint"]
            and retrieval.get("source_fingerprint") == report["corpus"]["source_fingerprint"], "Hybrid source/index identity differs")
    setup, total = report.get("retrieval_setup_metrics"), report.get("retrieval_metrics")
    setup_public = checked_retrieval_metrics(setup, vector_journal=vector_journal)
    total_public = checked_retrieval_metrics(total, vector_journal=vector_journal)
    require(setup["model_load_count"] == 1 and setup["query_count"] == setup["encode_count"] == 1
            and setup["failed_queries"] == 0, "Hybrid prepare was not a separate successful load/warmup")
    measured = [setup]
    for entry in entries:
        delta = entry.get("retrieval_metrics")
        checked_retrieval_metrics(delta, vector_journal=vector_journal)
        require(delta["model_load_count"] == delta["model_load_seconds"] == delta["model_validation_seconds"] == 0,
                "Hybrid encoder loading leaked into case timings")
        measured.append(delta)
    require([event for part in measured for event in part["queries"]] == total["queries"]
            and [event["number"] for event in total["queries"]] == list(range(1, total["query_count"] + 1)),
            "Hybrid per-arm query journal is incomplete or reordered")
    for key in RETRIEVAL_COUNTS | RETRIEVAL_TIMES:
        expected = sum(part[key] for part in measured)
        require(total[key] == expected if key in RETRIEVAL_COUNTS else math.isclose(total[key], expected, abs_tol=1e-8, rel_tol=1e-9),
                "Hybrid cumulative measurements differ from setup plus case arms")
    return {"mode": "hybrid", "encoder": "bge-m3", "revision": model["revision"], "device": "cpu", "dtype": "float32",
            "query_vector_journal": vector_journal,
            "cpu_threads": 2, "encoder_files_sha256": sha(canonical(files)), "policy_sha256": sha(canonical(retrieval["policy"])),
            "prepare": setup_public, "including_prepare": total_public,
            "measurement_scope": "One CPU encoder; prepare load/warmup excluded from case-arm retrieval deltas."}


def replay_catalog(directory, source_hashes, catalog_path, fingerprint, texts, query_vectors=None):
    """Replay only frozen, deterministic catalog matching, without a corpus or model."""
    source_root = directory / "source/risk_layer"
    require("risk_layer/research_catalog.py" in source_hashes, "Unbound catalog implementation")
    for path in source_root.glob("*.py"):
        require(source_hashes.get("risk_layer/" + path.name) == digest(path), "Unbound catalog dependency")
    package_name = "_research_catalog_audit_" + sha(str(directory.resolve()))
    require(not any(name == package_name or name.startswith(package_name + ".") for name in sys.modules),
            "Catalog audit namespace is already loaded")
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_root)]
    sys.modules[package_name] = package
    previous_bytecode_policy = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module(package_name + ".research_catalog")
        if query_vectors is None:
            catalog = module.ResearchCatalog(catalog_path, expected_fingerprint=fingerprint)
        else:
            embedding = importlib.import_module(package_name + ".embedding")
            index = embedding.EmbeddingIndex.load(directory / "inputs/index")
            require(index is not None and index.passage_evidence is not None, "Missing frozen catalog anchor index")
            catalog = module.ResearchCatalog(catalog_path, expected_fingerprint=fingerprint, evidence=index.passage_evidence)
        metadata = catalog.metadata()
        hints = ([catalog.hints(text) for text in texts] if query_vectors is None else
                 [None if vector is None else catalog.hints(text, vector)
                  for text, vector in zip(texts, query_vectors, strict=True)])
        return metadata, hints
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]


def audit_catalog_result(result, metadata, hints, original):
    """Catalog cards remain separate from the source ledger, including empty cards."""
    keys = {"catalog_metadata", "catalog_hints"}
    present = keys.intersection(result)
    requests = [event.get("request", {}) for event in result.get("trace", [])
                if event.get("kind") == "http" and event.get("path") in ("/apply-template", "/v1/chat/completions")]
    if metadata is None:
        require(not present, "Disabled catalog unexpectedly produced hints")
    elif not present:
        require(result.get("status") != "complete" and not requests,
                "Initialized catalog arm omitted its hint ledger")
        return {"arm_ledgers": 0, "audited_request_payloads": 0, "unavailable_arm_ledgers": 1}
    else:
        require(present == keys and canonical(result["catalog_metadata"]) == canonical(metadata)
                and canonical(result["catalog_hints"]) == canonical(hints),
                "Catalog ledger differs from frozen YAML and original text")
    for request in requests:
        users = [message for message in request.get("messages", []) if message.get("role") == "user"]
        require(bool(users), "Missing initial user message")
        content = users[0].get("content")
        try:
            user = parse(content)
        except (ValueError, TypeError):
            require(metadata is None, "Enabled catalog has no structured user payload")
            continue
        if not isinstance(user, dict):
            require(metadata is None, "Enabled catalog has no structured user payload")
            continue
        if metadata is None:
            require("KATALOGHINWEISE" not in user, "Disabled catalog leaked into user payload")
        else:
            require(set(user) <= {"SACHVERHALT", "FOKUSSTELLEN", "STARTQUELLEN", "BUDGET", "RECHERCHE", "KATALOGHINWEISE"}
                    and user.get("SACHVERHALT") == original and canonical(user.get("KATALOGHINWEISE")) == canonical(hints),
                    "Catalog user payload differs from bound original-text hints")
    return {"arm_ledgers": int(metadata is not None), "audited_request_payloads": len(requests) if metadata else 0,
            "unavailable_arm_ledgers": 0}


def checked_catalog_vector(row, entry, initial_limit):
    """Bind the private vector to the original document's existing first retrieval."""
    import numpy as np

    vector = entry["result"].get("catalog_query_vector")
    require(isinstance(vector, list) and len(vector) == QUERY_VECTOR_FORMAT["dimension"]
            and all(type(value) is float and math.isfinite(value)
                    and abs(value) <= np.finfo(np.float32).max for value in vector),
            "Invalid private catalog query vector")
    array = np.asarray(vector, dtype="<f4")
    require(np.array_equal(array.astype(np.float64), np.asarray(vector, dtype=np.float64)),
            "Catalog query vector contains values not representable as float32")
    require(abs(float(np.linalg.norm(array)) - 1) < 1e-4, "Catalog query vector is not normalized")
    queries = entry.get("retrieval_metrics", {}).get("queries", [])
    require(bool(queries), "Catalog has no measured initial query")
    initial = queries[0]
    require(initial.get("query_sha256") == sha(row["query"])
            and initial.get("characters") == len(row["query"])
            and initial.get("limit") == initial_limit and initial.get("status") == "complete"
            and initial.get("encode_count") == 1
            and initial.get("query_vector_sha256") == sha(array.tobytes(order="C")),
            "Catalog vector differs from the original initial retrieval")
    return array


def audit_catalog(report, protocol, reference, directory):
    keys = {"catalog_metadata", "catalog_provenance"}
    present = keys.intersection(report)
    require(present == keys.intersection(protocol) and (not present or present == keys),
            "Incomplete frozen catalog settings")
    jobs = [(row, entry) for row in report["rows"] for entry in row["arms"].values()]
    metadata, expected_hints = None, [None] * len(jobs)
    semantic = False
    if present:
        require(all(canonical(report[key]) == canonical(protocol[key]) for key in keys), "Frozen catalog settings changed")
        provenance, metadata = report["catalog_provenance"], report["catalog_metadata"]
        require(isinstance(provenance, dict)
                and set(provenance) == {"snapshot", "sha256", "size_bytes", "reference_sha256"}
                and provenance["snapshot"] == "inputs/catalog.yaml"
                and type(provenance["size_bytes"]) is int and provenance["size_bytes"] > 0,
                "Invalid catalog snapshot provenance")
        path = safe_path(directory, provenance["snapshot"])
        require(report["snapshot_hashes"].get(provenance["snapshot"]) == provenance["sha256"] == digest(path)
                and path.stat().st_size == provenance["size_bytes"], "Catalog snapshot differs")
        reference_hashes = {value for name, value in reference["input_hashes"].items()
                            if Path(name).name in ("begriffe.yaml", "begriffe.yml")}
        require((not reference_hashes or reference_hashes == {provenance["sha256"]})
                and provenance["reference_sha256"] == (provenance["sha256"] if reference_hashes else None),
                "Catalog differs from historical reference")
        index = parse((directory / "inputs/index/meta.json").read_bytes())
        fingerprint = index.get("catalog_fingerprint")
        require(isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint),
                "Missing source-index catalog fingerprint")
        require(isinstance(metadata, dict) and metadata.get("format") in ("research_catalog/1", "research_catalog/2"),
                "Unknown catalog metadata format")
        semantic = metadata["format"] == "research_catalog/2"
        vectors = None
        if semantic:
            retrieval = report.get("retrieval", {})
            details = metadata.get("semantic", {})
            require(retrieval.get("mode") == "hybrid" and retrieval.get("format") == "research_cpu_hybrid/2"
                    and canonical(details.get("query_vector")) == canonical(QUERY_VECTOR_FORMAT)
                    and details.get("encoder") == retrieval.get("model_identity", {}).get("encoder") == "bge-m3"
                    and details.get("encoder_revision") == retrieval.get("model_identity", {}).get("revision")
                    and details.get("evidence_fingerprint") == report["corpus"]["index_fingerprint"],
                    "Semantic catalog differs from source retrieval/index identity")
            vectors = [checked_catalog_vector(row, entry, report["agent_config"]["initial_sources"])
                       if (entry["result"].get("status") == "complete"
                           or {"catalog_hints", "catalog_metadata", "catalog_query_vector"}.intersection(entry["result"]))
                       else None for row, entry in jobs]
        reconstructed, expected_hints = replay_catalog(directory, report["source_hashes"], path, fingerprint,
                                                       [row["query"] for row, _ in jobs], vectors)
        require(canonical(metadata) == canonical(reconstructed), "Catalog metadata differs from frozen YAML/policy")
    stats = Counter()
    for (row, entry), hints in zip(jobs, expected_hints, strict=True):
        if not semantic:
            require("catalog_query_vector" not in entry["result"], "Nonsemantic catalog has a private query vector")
        checked = audit_catalog_result(entry["result"], metadata, hints, row["query"])
        stats.update(checked)
        if checked["arm_ledgers"]:
            stats["arm_ledgers_with_hints"] += bool(hints["concepts"])
            stats["ledger_concept_cards"] += len(hints["concepts"])
            stats["ledger_omitted_concept_cards"] += hints["omitted_concepts"]
    if metadata is None:
        return {"enabled": False}
    return {"enabled": True, "mode": "literal_and_semantic" if semantic else "literal_and_pattern",
            "yaml_sha256": metadata["sha256"], "fingerprint": metadata["fingerprint"],
            "semantic_fingerprint": metadata["semantic_fingerprint"], "version_sha256": sha(canonical(metadata["version"])),
            "policy_sha256": sha(canonical(metadata["policy"])), "counts": dict(stats),
            "scope": "Unverified search hints; neither source reads nor graph-citation evidence."}


def aggregate(path):
    path = Path(path).resolve()
    directory, bindings = path.parent, {}
    report, report_sha = read_bound(path, bindings)
    require(report.get("format") == "quality_research_agent/1"
            and report.get("status") in ("complete", "complete_with_errors", "failed"), "Expected terminal frozen agent report")
    protocol, protocol_sha = read_bound(directory / "protocol.json", bindings, report["protocol_sha256"])
    require(protocol.get("format") == "quality_research_agent_protocol/1" and protocol.get("status") == "frozen_before_runs"
            and protocol.get("output_cap") == 5 and protocol.get("focus") == "original_sentence", "Invalid frozen protocol")
    for key in ("arms", "input_hashes", "source_hashes", "diagnostic_subset", "reference_sha256", "agent_config",
                "model_provenance", "server", "snapshot_hashes"):
        require(report[key] == protocol[key], "Report changed frozen protocol fields")
    require(report["arms"] and len(report["arms"]) == len(set(report["arms"])) and set(report["arms"]) <= set(ARMS),
            "Unknown or duplicate experimental arm")
    required_sources = {"tools/quality_research_agent.py", "risk_layer/research_agent.py", "risk_layer/research_sources.py"}
    require(required_sources <= set(report["source_hashes"]), "Missing executed client/corpus/runner source inventory")
    required_snapshots = {"reference.json", "model-manifest.json", "launch-manifest.json", "inputs/graph.sqlite",
                          "inputs/index/meta.json", "inputs/index/vectors.npy"}
    require(required_snapshots <= set(report["snapshot_hashes"]), "Missing frozen model/source/reference artifacts")
    for relative, expected in report["snapshot_hashes"].items():
        source = safe_path(directory, relative)
        require(digest(source) == expected, "Frozen input/source snapshot changed")
        bindings[source.resolve()] = expected
    for relative, expected in report["source_hashes"].items():
        require(report["snapshot_hashes"].get("source/" + relative) == expected, "Unbound executed implementation")
    manifest, _ = read_bound(directory / "model-manifest.json", bindings, report["model_provenance"]["model_manifest_sha256"])
    launch, _ = read_bound(directory / "launch-manifest.json", bindings, report["model_provenance"]["launch_manifest_sha256"])
    require(manifest == report["model_provenance"]["model"] and launch == report["model_provenance"]["launch"]
            and launch.get("model_manifest_sha256") == report["model_provenance"]["model_manifest_sha256"]
            and launch.get("alias") == protocol["model_alias"], "Model/launch identity differs from frozen manifests")
    reference, _ = read_bound(directory / "reference.json", bindings, report["reference_sha256"])
    require(reference.get("format") == "quality_recall_live/1" and reference.get("status") in ("pass", "mismatch")
            and reference.get("public_profile") == "treffer" and len(reference["rows"]) == 162
            and reference.get("source_audit", {}).get("errors") == [], "Invalid actual baseline reference")
    originals = {row["case_id"]: row for row in reference["rows"]}
    require(len(originals) == 162, "Duplicate baseline case IDs")
    require(Counter((row["dataset"], row["negative"]) for row in reference["rows"]) ==
            Counter({("main", False): 102, ("main", True): 30, ("additional", False): 15, ("additional", True): 15}),
            "Historical dataset denominator differs")
    for relative in ("inputs/graph.sqlite", "inputs/index/meta.json", "inputs/index/vectors.npy"):
        matches = {value for filename, value in reference["input_hashes"].items() if Path(filename).name == Path(relative).name}
        require(matches == {report["snapshot_hashes"][relative]}, "Research corpus differs from original live reference")
    rows = report["rows"]
    ids = [row["case_id"] for row in rows]
    require(ids == protocol["case_order"] and len(ids) == len(set(ids)) and set(ids) <= set(originals), "Case coverage differs from protocol")
    require(ids == [row["case_id"] for row in reference["rows"] if row["case_id"] in set(ids)], "Case order differs from original reference")
    require(type(report["diagnostic_subset"]) is bool
            and (report["diagnostic_subset"] or len(rows) == 162), "Subset falsely claims full benchmark")
    for row in rows:
        source = originals[row["case_id"]]
        require(all(row[key] == source[key] for key in CASE_FIELDS)
                and row["baseline"] == {"suggestions": source["treffer"]["suggestions"]}, "Reference labels/text/actual baseline changed")
    journal_path = directory / "rows.jsonl"
    journal = journal_path.read_bytes() if journal_path.exists() else b""
    if journal_path.exists():
        bindings[journal_path.resolve()] = sha(journal)
    events = [parse(line) for line in journal.splitlines()]
    attempted = validate_journal(report, events)
    retrieval = audit_retrieval(report, protocol, directory)
    catalog = audit_catalog(report, protocol, reference, directory)
    config_keys = {"context_tokens", "max_output_tokens", "total_output_tokens", "reasoning_budget_tokens",
                   "max_rounds", "max_tool_calls", "initial_sources", "timeout_seconds", "temperature", "top_p",
                   "top_k", "min_p", "presence_penalty", "seed", "reuse_prompt_cache", "finish_reserve_seconds"}
    require(set(report["agent_config"]) <= config_keys, "Unknown public configuration field")
    execution_policy(report["agent_config"])
    require(all(number(value) for key, value in report["agent_config"].items() if key != "reuse_prompt_cache"),
            "Agent configuration contains nonnumeric private payload")
    load_seconds = report.get("timing", {}).get("corpus_load_seconds")
    require(load_seconds is None or number(load_seconds), "Invalid corpus loading measurement")
    prepare_seconds = report.get("timing", {}).get("retrieval_prepare_seconds")
    require(prepare_seconds is None or number(prepare_seconds), "Invalid retrieval prepare measurement")
    model = report["model_provenance"]["model"]
    require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repo"]))
            and bool(re.fullmatch(r"[0-9a-f]{40}", model["revision"]))
            and bool(re.fullmatch(r"[A-Za-z0-9_.-]+\.gguf", model["filename"]))
            and bool(re.fullmatch(r"[A-Za-z0-9_]+", model["quantization"])), "Invalid public model identity")
    public = {"format": "quality_research_aggregate/1", "status": report["status"],
        "scope": "diagnostic_subset" if report["diagnostic_subset"] else "known_development_cases",
        "report_sha256": report_sha, "protocol_sha256": protocol_sha, "journal_sha256": sha(journal),
        "reference_sha256": report["reference_sha256"], "generator_sha256": digest(__file__),
        "cases": len(rows), "attempted_case_arms": attempted, "planned_case_arms": len(rows) * len(report["arms"]),
        "configuration": report["agent_config"], "retrieval": retrieval, "catalog": catalog,
        "model": {key: report["model_provenance"]["model"][key]
            for key in ("repo", "revision", "filename", "quantization")},
        "model_manifest_sha256": report["model_provenance"]["model_manifest_sha256"],
        "launch_manifest_sha256": report["model_provenance"]["launch_manifest_sha256"],
        "source_hashes": report["source_hashes"], "datasets": {}, "transitions": {},
        "timing_scope": {"corpus_load_seconds": report.get("timing", {}).get("corpus_load_seconds"),
            "retrieval_prepare_seconds": prepare_seconds,
            "model_loading": "Before benchmark, separately bound setup; excluded from case timings.",
            "quantile_method": "Linear interpolation at p*(n-1); all attempted arms include failed attempts."},
        "notes": ["All reference cases were previously examined; this is not independent validation.",
            "Diagnostic subsets have counts only, no full-benchmark rates.",
            "Technical failures/pending cases remain in all selected denominators, never counted as correct negative decisions.",
            "Unlisted norms in positive cases are unadjudicated. Suggestion precision remains unknown.",
            "New source availability counts original pages returned by actual source tools; citation metadata alone is not a source page.",
            "Transcript quote checks establish returned-source presence, not independent legal correctness."]}
    source_stats, traces = {}, {}
    for row in rows:
        complete = []
        for arm, entry in row["arms"].items():
            result = entry["result"]
            require(result["status"] in STATUSES and number(entry["wall_seconds"]), "Invalid terminal arm result")
            if result["status"] != "runner_error":
                require(result.get("allow_research") == (arm == "research")
                        and result.get("config") == report["agent_config"]
                        and result.get("model") == protocol["model_alias"], "Agent model/config/mode differs")
            traces[(row["case_id"], arm)] = trace_counts(result)
            source_stats[(row["case_id"], arm)] = audit_sources(row, arm, report["corpus"])
            if result["status"] == "complete":
                complete.append(result)
        if len(complete) == 2:
            require(all(complete[0][key] == complete[1][key] for key in ("initial_sources", "initial_sources_sha256", "focuses")),
                    "Arms did not start from identical sources and focuses")
    for dataset in ("all", *DATASETS):
        selected = [row for row in rows if dataset == "all" or row["dataset"] == dataset]
        public["datasets"][dataset] = {}
        for arm in ("baseline", *report["arms"]):
            metrics = metric_counts(selected, arm, report["diagnostic_subset"])
            require(report["summaries"][dataset][arm] == metrics, "Stored metric counts/rates differ from frozen outputs")
            item = {"metrics": metrics, "output_load": output_load(selected, arm)}
            if arm != "baseline":
                entries = [(row, row["arms"][arm]) for row in selected if arm in row["arms"]]
                status = Counter(entry["result"]["status"] for _, entry in entries)
                status["pending"] = len(selected) - len(entries)
                item["completion_statuses"] = dict(status)
                for kind in ("tool_calls", "completion_finish_reasons", "errors", "tool_result_errors", "repair_and_failure_counts", "budget_events", "prompt_cache"):
                    total = Counter()
                    for row, _ in entries:
                        total.update(traces[(row["case_id"], arm)][kind])
                    item[kind] = dict(total)
                total = Counter()
                for row, _ in entries:
                    total.update(source_stats[(row["case_id"], arm)])
                item["source_availability"] = dict(total)
                item["wall_seconds"] = quantiles([entry["wall_seconds"] for _, entry in entries])
                item["completed_wall_seconds"] = quantiles([entry["wall_seconds"] for _, entry in entries if entry["result"]["status"] == "complete"])
                item["generation_request_seconds"] = quantiles([sum(event["elapsed_seconds"] for event in entry["result"].get("trace", [])
                    if event.get("kind") == "http" and event.get("path") == "/v1/chat/completions") for _, entry in entries])
                item["client_measured_usage"] = {key: sum(entry["result"].get("usage", {}).get(key, 0) for _, entry in entries)
                    for key in ("http_requests", "model_calls", "prompt_tokens", "completion_tokens")}
                if retrieval["mode"] == "hybrid":
                    item["cpu_retrieval"] = {key: sum(entry["retrieval_metrics"][key] for _, entry in entries)
                                             for key in sorted(RETRIEVAL_COUNTS | RETRIEVAL_TIMES)}
            public["datasets"][dataset][arm] = item
        pairs = [("baseline", arm) for arm in report["arms"]]
        if set(ARMS) <= set(report["arms"]):
            pairs.append(("static", "research"))
        public["transitions"][dataset] = {a + "_to_" + b: transition(selected, a, b) for a, b in pairs}
    public["audit"] = {"status": "pass", "bound_files": len(bindings), "exact_journal_entries": attempted,
                       "torch_imported": False, "benchmark_torch_imported": report.get("torch_imported"),
                       "independent_metric_recalculation": True}
    public_check(public)
    for path, expected in bindings.items():
        require(digest(path) == expected, "Frozen artifact changed during aggregation")
    return public


def render(report):
    subset = report["scope"] == "diagnostic_subset"
    lines = ["# Rechercheagent: " + ("Mechanikdiagnostik" if subset else "Entwicklungsvergleich"), "",
        f"{report['cases']} bereits bekannte Fälle, {report['attempted_case_arms']}/{report['planned_case_arms']} ausgeführte Arme. "
        + ("Dies ist eine diagnostische Teilmenge; daraus folgt keine vollständige Qualitätsquote." if subset else
           "Dies sind bekannte Entwicklungsdaten, keine unabhängigen Prüffälle."), "",
        "Die historischen Zielnormlisten sind unvollständig. Die Präzision einzelner Vorschläge bleibt unbekannt. "
        "Fehlgeschlagene und noch offene Fälle bleiben in den Nennern und gelten nicht als richtige Negativentscheidungen.", "",
        "| Datensatz / Arm | bekannte Trefferfälle | Negativfälle mit Ausgabe | technische Fehler / offene Fälle | Ausgabeeinträge / eindeutige Fall-Norm-Paare |",
        "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for arm, item in report["datasets"][dataset].items():
            v, load = item["metrics"], item["output_load"]
            lines.append(f"| {dataset}/{arm} | {v['hit5_cases']} von {v['positive_cases']} | "
                f"{v['negative_with_suggestion']} von {v['negative_cases']} | {v['failed_cases']} / {v['pending_cases']} | "
                f"{load['output_entries']} / {load['unique_case_norm_pairs']} |")
    lines += ["", "Quellensuche: " + ("BM25 und dichte BGE-M3-Suche, gemeinsam auf der CPU. "
        "Encoderladen und der feste technische Probelauf sind separat erfasst und aus den Fallzeiten ausgeschlossen."
        if report["retrieval"]["mode"] == "hybrid" else "BM25 ohne Encoder.")]
    if report.get("catalog", {}).get("enabled"):
        lines += ["", "Zusätzliche Katalogkarten sind ungeprüfte Suchhinweise. Ihre Fundstellen und Normketten "
                  "zählen weder als gelesene Quellen noch als gelesene Graphverweise. "
                  "Hinweise und ursprüngliche Textstellen wurden aus der eingefrorenen YAML-Datei nachgerechnet."]
    lines += ["", "Änderungen zwischen zwei vollständig beendeten Armen werden separat gezählt. "
        "Damit werden technische Ausfälle nicht als inhaltlicher Nutzen der Recherche ausgegeben.", "",
        "| Vergleich (beide vollständig) | Fälle | neu erreichte / verlorene bekannte Zielnormpaare | neue / entfernte negative Normpaare |",
        "|---|---:|---:|---:|"]
    for name, change in report["transitions"]["all"].items():
        v = change["both_complete_changes"]
        lines.append(f"| {name} | {v['cases']} | {v['gained_target_pairs']} / {v['lost_target_pairs']} | "
                     f"{v['added_negative_norm_pairs']} / {v['removed_negative_norm_pairs']} |")
    lines += ["", "Laufzeiten umfassen alle tatsächlich begonnenen und protokollierten Arme einschließlich Fehlern. "
        "Modellladen liegt vor diesem Benchmark; Laden der Quellen wird separat erfasst. "
        "P95 verwendet lineare Interpolation, keine zusätzliche Modellmessung.", "",
        "| Arm | Versuche | Median / P95 Sekunden | Suche / Lesen / Verweise / Abschluss | neue Quellenfenster / Normen |",
        "|---|---:|---:|---:|---:|"]
    status_lines = []
    for arm, item in report["datasets"]["all"].items():
        if arm == "baseline":
            continue
        t, tools, sources = item["wall_seconds"], item["tool_calls"], item["source_availability"]
        timing = "—" if not t["n"] else f"{t['median']:.3f} / {t['p95']:.3f}"
        lines.append(f"| {arm} | {t['n']} | {timing} | " + " / ".join(str(tools.get(k, 0)) for k in TOOLS)
                     + f" | {sources.get('new_pages', 0)} / {sources.get('new_norms', 0)} |")
        status_lines.append(f"Abschlussstatus {arm}: " + ", ".join(f"{key}: {value}" for key, value in item["completion_statuses"].items()) + ".")
        for key, label in (("repair_and_failure_counts", "Reparatur-/Fehlerzähler"), ("tool_result_errors", "Werkzeug-Fehlertypen"),
                           ("budget_events", "Budgetentscheidungen ohne eigene Laufzeitmessung"), ("prompt_cache", "Promptcache-Messung")):
            if item[key]:
                status_lines.append(f"{label} {arm}: " + ", ".join(f"{name}: {value}" for name, value in item[key].items()) + ".")
    lines += ["", *status_lines]
    lines += ["", "Neue Quellen zählen ausschließlich tatsächlich zurückgegebene Originaltextfenster. "
        "Eine neue Verweis-ID allein ist keine gelesene Quelle. Zitatprüfungen bestätigen Textvorkommen, keine rechtliche Anwendbarkeit.", "",
        f"Gebundener Lauf: `{report['report_sha256']}`; Protokoll: `{report['protocol_sha256']}`."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh aggregate output directory")
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Choose a fresh aggregate directory")
    value = aggregate(args.report)
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(render(value), encoding="utf-8")
    print(json.dumps({"status": "complete", "scope": value["scope"], "cases": value["cases"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
