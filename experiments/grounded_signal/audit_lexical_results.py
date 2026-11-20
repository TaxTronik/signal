"""Independent stdlib audit of all eight terminal lexical evaluation cases.

Never imports the producer, experiment evaluator, model runtime or a model.
Original facts are inherited and hash-bound; their time is not new execution.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sqlite3


from experiments.grounded_signal import audit_grounded_results as common


HELPER = Path(common.__file__).resolve()
ARM = "structured_lexical"
require, equal, digest = common.require, common.equal, common.digest


def inventory(bound, directory, expected, *, exclude=()):
    actual_names = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts and p.relative_to(directory).as_posix() not in exclude}
    equal(sorted(actual_names), sorted(expected), "Archive file inventory differs")
    for name, sha in expected.items():
        bound.bind(common.local(directory, name), sha)


def terminal(run, report, protocol):
    require(not any((run / name).exists() for name in ("abandonment.json", "interruption.json")), "Interrupted lexical run")
    require(report.get("format") == "grounded_signal_lexical_probe/1"
            and report.get("status") in {"complete", "incomplete"}, "Only terminal full lexical runs accepted")
    equal(report["planned_cases"], 8, "Lexical case denominator differs")
    equal(report["new_fact_calls"], 0, "Lexical run performed new facts inference")
    equal(protocol["arms"], [ARM], "Unexpected lexical arm")
    require(set(report["results"]) == {ARM} and len(report["results"][ARM]) == 8, "Missing lexical cases")
    require(all(r.get("status") not in common.NONTERMINAL for r in report["results"][ARM]), "Unfinished lexical row")
    equal(report["status"], "complete" if all(r["status"] == "complete" for r in report["results"][ARM]) else "incomplete", "Terminal lexical status differs")
    equal(report["end_verification"], "pass", "Producer end verification did not pass")


def model_identity(binding):
    receipt = binding["model"]
    require(receipt["format"] == "quality_research_model/1" and receipt["status"] == "complete", "Incomplete model receipt")
    identity = {k: receipt[k] for k in ("repo", "revision", "filename", "quantization")}
    identity.update({k: receipt["model"][k] for k in ("bytes", "sha256")})
    common.integer(identity["bytes"])
    require(identity["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]), "Missing GGUF identity")
    return identity


def select_cases(all_cases, cases, learning, primary, protocol):
    common.check_cases(all_cases)
    expected = [c for c in all_cases if c["split"] == "evaluation"]
    equal(cases, expected, "Lexical cases are not all eight unchanged evaluation cases")
    equal(protocol["case_ids"], [c["case_id"] for c in expected], "Lexical case order differs")
    equal([r["case_id"] for r in learning], primary["case_ids"], "Frozen ranking case inventory differs")
    require(len(learning) == 16, "Incomplete original rankings")
    lookup = {r["case_id"]: r for r in learning}
    selected = {}
    for case in cases:
        ranking = lookup[case["case_id"]]["unadapted"]
        require(len(ranking) == 3 and len(common.ids([r["norm_id"] for r in ranking])) == 3, "Expected exactly original lexical top three")
        require(all(type(r["score"]) in (float, int) for r in ranking), "Invalid ranking score")
        for row in ranking:
            common.number(row["score"])
        equal(ranking, sorted(ranking, key=lambda r: (-r["score"], r["norm_id"])), "Frozen lexical order differs")
        selected[case["case_id"]] = [r["norm_id"] for r in ranking]
    return selected


def original_sources(graph_path, packages, table):
    counts = Counter(packages=0, norm_groups=0, units=0, source_characters=0)
    con = sqlite3.connect(graph_path.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        for package in packages.values():
            common.check_package(package, table)
            require(package["status"] == "ready" and len(package["candidate_norm_ids"]) == 3, "Lexical package is not a ready three-norm package")
            counts["packages"] += 1
            chars = 0
            for group in package["required_groups"]:
                found = con.execute("SELECT text FROM knoten WHERE id=?", (group["norm_id"],)).fetchone()
                require(found is not None and isinstance(found[0], str), "Original norm missing")
                units = [u for u in package["units"] if u["norm_id"] == group["norm_id"]]
                equal("".join(u["text"] for u in units), found[0], "Whole unit group differs from original norm")
                require(all(found[0][u["start"]:u["end"]] == u["text"] for u in units), "Original norm offset differs")
                counts["norm_groups"] += 1
                counts["units"] += len(units)
                chars += len(found[0])
            require(chars <= 32000, "Whole lexical package exceeds source character cap")
            counts["source_characters"] += chars
    finally:
        con.close()
    return dict(counts)


def prompts_from(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"FACT_PROMPT", "DECISION_PROMPT"}:
                result[name] = ast.literal_eval(node.value)
    require(set(result) == {"FACT_PROMPT", "DECISION_PROMPT"}, "Frozen prompt constants missing")
    return result


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def enum(values):
    return {"type": "string", "enum": list(values)}


def text_schema(limit):
    return {"type": "string", "minLength": 1, "maxLength": limit}


def array(item, maximum):
    return {"type": "array", "items": item, "maxItems": maximum}


def references(values, maximum, minimum=0):
    return {**array(enum(values), maximum), "minItems": minimum} if values else array({"type": "string"}, 0)


def span_plan(text):
    positions = [m.end() for m in re.finditer(r'[.!?](?:["”])?(?=\s+[A-ZÄÖÜ]|\s*$)|\n+', text)] + [len(text)]
    spans: list[dict[str, str | int]] = []
    previous = 0
    for stop in positions:
        start, end = previous, stop
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append({"id": f"s{len(spans) + 1}", "start": start, "end": end, "text": text[start:end]})
        previous = stop
    return spans


def expected_schema(spans, package, fact):
    if package is None:
        return object_schema({"facts": array(object_schema({
            "id": enum([f"f{i}" for i in range(1, 9)]), "span_id": enum([s["id"] for s in spans]),
            "kind": enum(["actor", "role", "event", "negation", "time", "context"]),
            "polarity": enum(["affirmed", "denied", "uncertain"]), "statement": text_schema(180)}), 8),
            "unresolved": array(text_schema(150), 4)})
    sources = common.shown_sources(package)
    fact_ids = [f["id"] for f in fact["answer"]["facts"]]
    slots = {}
    for norm in sources["candidate_norm_ids"]:
        units = [u["unit_id"] for u in sources["units"] if u["norm_id"] == norm]
        choices = []
        if fact_ids:
            choices.append(object_schema({"status": enum(["hint"]), "evidence_unit_ids": references(units, 6, 1),
                                          "supported_fact_ids": references(fact_ids, 8, 1), "open_conditions": array(text_schema(160), 4),
                                          "relation": enum(["definition", "duty", "procedure", "exception", "scope"])}))
        choices.append(object_schema({"status": enum(["no_hint", "insufficient_context"]), "evidence_unit_ids": references(units, 6),
                                      "supported_fact_ids": references(fact_ids, 8), "open_conditions": array(text_schema(160), 4), "relation": enum(["none"])}))
        slots[norm] = {"oneOf": choices}
    return object_schema({"hints": object_schema(slots), "unresolved": array(text_schema(150), 4)})


def check_prompt(call, text, prompts, package=None, fact=None):
    spans = span_plan(text)
    model_requests = [e["request"] for e in call.get("trace", []) if e["path"] in {"/apply-template", "/v1/chat/completions"}]
    if not model_requests:
        require(call["status"] != "complete", "Successful stage has no native request")
    for request in model_requests:
        equal(request["messages"][0], {"role": "system", "content": prompts["FACT_PROMPT" if package is None else "DECISION_PROMPT"]}, "Prompt differs from frozen original runtime")
        payload = common.parse(request["messages"][1]["content"])
        equal(payload["case_spans"], spans, "Original case segmentation differs")
        equal(request["tools"], [{"type": "function", "function": {
            "name": "extract_facts" if package is None else "decide_hints",
            "description": "Vollständige strukturierte Antwort auf die angegebene Teilaufgabe.",
            "parameters": expected_schema(spans, package, fact)}}], "Required tool parameters differ from the fixed R2 interface")


def audit(run):
    run = run.resolve()
    bound = common.BoundFiles()
    report, protocol = bound.read(run / "report.json"), bound.read(run / "protocol.json")
    terminal(run, report, protocol)
    inventory(bound, run, report["artifact_hashes"], exclude=("report.json",))
    prepared, base, primary_dir = (Path(protocol[k]).resolve() for k in ("prepared_directory", "base_run", "primary_inputs"))
    prep_receipt = bound.read(prepared / "prepared-manifest.json")
    equal(prep_receipt["format"], "grounded_signal_lexical_inputs/1", "Unknown lexical input format")
    equal(bound.hashes[prepared / "prepared-manifest.json"], protocol["prepared_manifest_sha256"], "Prepared receipt differs")
    inventory(bound, prepared, prep_receipt["artifacts"], exclude=("prepared-manifest.json",))
    prepared_protocol = bound.read(prepared / "protocol.json")
    for key, expected in prepared_protocol.items():
        if key != "format":
            equal(protocol[key], expected, "Prepared lexical protocol differs: " + key)
    equal(protocol["format"], "grounded_signal_lexical_probe/1", "Unknown run protocol")
    for key, expected in (("top_k", 3), ("max_package_chars", 32000), ("reference_depth", 0), ("post_hoc_exploratory", True)):
        equal(protocol[key], expected, "Changed exploratory policy")
    primary_receipt = bound.read(primary_dir / "input-manifest.json")
    equal(primary_receipt["format"], "grounded_signal_inputs/1", "Unknown original input receipt")
    equal(bound.hashes[primary_dir / "input-manifest.json"], protocol["primary_input_manifest_sha256"], "Primary input receipt differs")
    inventory(bound, primary_dir, primary_receipt["artifacts"], exclude=("input-manifest.json",))
    primary = bound.read(primary_dir / "protocol.json")
    base_report, base_protocol = bound.read(base / "results.json"), bound.read(base / "protocol.json")
    common.terminal_guard(base, base_report, base_protocol)
    inventory(bound, base, protocol["base_artifacts"])
    equal(bound.hashes[base / "results.json"], protocol["base_results_sha256"], "Original results receipt differs")
    equal(bound.hashes[base / "protocol.json"], protocol["base_protocol_sha256"], "Original protocol receipt differs")
    equal(base_protocol["input_manifest_sha256"], protocol["primary_input_manifest_sha256"], "Different original input generation")
    equal(protocol["primary_source_hashes"], primary["source_hashes"], "Primary code generation differs")
    equal(base_report["source_end_hashes"], base_protocol["source_hashes"], "Primary final source generation differs")
    for key, expected in primary.items():
        equal(base_protocol[key], expected, "Original full primary protocol differs")
    for name, expected in protocol["primary_source_hashes"].items():
        bound.bind(common.local(base / "source", name), expected)
    for name in ("protocol.json", "results.json", "model-manifest.json"):
        equal(bound.read(run / "inherited" / name), bound.read(base / name), "Inherited primary snapshot differs")
    model = bound.read(run / "model-manifest.json")
    equal(model, protocol["model_binding"]["model"], "Lexical model receipt differs")
    equal(bound.hashes[run / "model-manifest.json"], protocol["model_binding"]["model_manifest_sha256"], "Model receipt hash differs")
    equal(bound.read(run / "launch-manifest.json"), protocol["model_binding"]["launch"], "Lexical launch receipt differs")
    equal(bound.hashes[run / "launch-manifest.json"], protocol["model_binding"]["launch_manifest_sha256"], "Launch receipt hash differs")
    identity = model_identity(protocol["model_binding"])
    equal(identity, model_identity(base_protocol["model_binding"]), "Facts belong to a different GGUF/model")
    equal(protocol["configuration"], base_protocol["configuration"], "Facts and decisions use different generation settings")
    for key in ("alias", "executable", "engine_files", "limits"):
        equal(protocol["model_binding"]["launch"][key], base_protocol["model_binding"]["launch"][key], "Original and lexical model/engine policy differs")
    all_cases = bound.read(primary_dir / "cases.json")
    cases, packages, units = (bound.read(prepared / name) for name in ("cases.json", "packages.json", "source-units.json"))
    learning = bound.read(primary_dir / "learning-diagnostic.json")
    plans = select_cases(all_cases, cases, learning, primary, protocol)
    for name in ("cases.json", "protocol.json", "learning-diagnostic.json", "input-manifest.json"):
        equal(bound.read(prepared / "primary" / name), bound.read(primary_dir / name), "Lexical preparation changed original inputs")
    bound.bind(prepared / "graph.sqlite", primary["graph_sha256"])
    source_counts = original_sources(prepared / "graph.sqlite", packages, units)
    producer_sha = bound.bind(run / "probe-source.py", protocol["probe_source_sha256"])
    bound.bind(prepared / "probe-source.py", producer_sha)
    runtime_path = base / "source/experiments/grounded_signal/runtime.py"
    prompts = prompts_from(runtime_path.read_text(encoding="utf-8"))
    rows = report["results"][ARM]
    equal([r["case_id"] for r in rows], protocol["case_ids"], "Lexical result case order differs")
    facts_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    facts_seconds, decision_seconds, observed = 0.0, 0.0, []
    alias = protocol["model_binding"]["launch"]["alias"]
    for index, (case, row) in enumerate(zip(cases, rows), 1):
        cid = case["case_id"]
        original_index = primary["case_ids"].index(cid) + 1
        name = f"facts-{original_index:03}.json"
        fact = bound.read(base / name)
        equal(fact, base_report["fact_results"][cid], "Original facts ledger differs")
        equal(bound.read(run / "inherited" / name), fact, "Inherited facts snapshot differs")
        equal(row["inherited_fact_sha256"], digest(fact), "Row inherits different facts")
        require(fact["status"] not in common.NONTERMINAL, "Unfinished original facts")
        facts_counts.update(common.check_call(fact, case=case, package=None, fact_call=None, arm="facts", config=protocol["configuration"], alias=alias))
        check_prompt(fact, case["text"], prompts)
        pid = protocol["case_packages"][ARM][cid]
        package = packages[pid]
        equal(sorted(package["candidate_norm_ids"]), sorted(plans[cid]), "Package is not the unchanged lexical top-three selection")
        equal(row["source_package_id"], pid, "Row changed package")
        equal(row["candidate_norm_ids"], package["candidate_norm_ids"], "Offered norm order differs")
        artifact = bound.read(run / f"case-{index:03}.json")
        equal(artifact["row"], row, "Private case row differs from final result")
        call = artifact["call"]
        equal(call["status"], row["status"], "Call status differs")
        equal(call.get("answer"), row["answer"], "Call answer differs")
        equal(call.get("usage", {}), row["usage"], "Call accounting differs")
        if row["status"] != "complete":
            require(row["answer"] is None, "Failed row contains accepted hints")
        if fact["status"] != "complete":
            equal(row["status"], "upstream_failure", "Failed facts were replaced or ignored")
            require(not call.get("trace"), "Decision executed without accepted inherited facts")
        decision_counts.update(common.check_call(call, case=case, package=package, fact_call=fact, arm=ARM, config=protocol["configuration"], alias=alias))
        check_prompt(call, case["text"], prompts, package, fact)
        old_seconds = common.number(fact["timing"]["elapsed_seconds"])
        new_seconds = common.number(call["timing"]["elapsed_seconds"])
        require(abs(common.number(row["timing"]["decision_seconds"]) - new_seconds) < 1e-8, "Decision time differs")
        require(abs(common.number(row["timing"]["inherited_fact_seconds"]) - old_seconds) < 1e-8, "Inherited time differs")
        require(abs(common.number(row["timing"]["elapsed_seconds"]) - old_seconds - new_seconds) < 1e-8, "Standalone time differs")
        facts_seconds += old_seconds
        decision_seconds += new_seconds
        observed.append(common.observation(case, row, package))
    stored = report["evaluation"]["aggregate"]
    summary = common.summarize(observed, stored["format"])
    for key, expected in {"protocol_sha256": digest(protocol), "planned_cases_sha256": digest(cases), "results_sha256": digest(report["results"]),
                          "source_packages_sha256": digest(packages), "source_units_sha256": digest(units), "planned_cases": 8, "planned_case_arm_rows": 8}.items():
        equal(stored[key], expected, "Canonical evaluation binding differs: " + key)
    equal(stored["arms"][ARM]["all"], summary["evaluation_counts"], "Independent all-case diagnostic counts differ")
    equal(stored["arms"][ARM]["evaluation"], summary["evaluation_counts"], "Independent evaluation diagnostic counts differ")
    equal(stored["arms"][ARM]["development"], common.summarize([], stored["format"])["evaluation_counts"], "Unexpected development cases")
    equal(stored["arms"][ARM]["timing"]["recorded_attempts"], 8, "Timing denominator differs")
    require(abs(stored["arms"][ARM]["timing"]["total_seconds"] - facts_seconds - decision_seconds) < 1e-8, "Stored standalone total differs")
    execution_seconds = common.number(report["execution_seconds"])
    require(execution_seconds + 1e-6 >= decision_seconds, "Physical lexical duration is below actual decision time")
    common.number(report["end_verification_seconds"])
    bound.finish()
    return {"format": "grounded_signal_lexical_independent_audit/1", "status": "pass", "run_status": report["status"],
            "post_hoc_exploratory": True, "model": identity, "summary": summary, "source_audit": source_counts,
            "audit_counts": {"files_checked_before_and_after": len(bound.hashes), "cases": 8, "target_norm_pairs": 6,
                             "clear_negative_cases": 2, "inherited_fact_ledgers": 8},
            "physical_execution": {"new_fact_calls": 0, "new_decisions": dict(decision_counts), "inherited_fact_calls": dict(facts_counts),
                                   "decision_seconds": decision_seconds, "inherited_fact_seconds": facts_seconds,
                                   "recorded_execution_seconds": execution_seconds, "end_verification_seconds": report["end_verification_seconds"],
                                   "execution_io_seconds": max(0.0, execution_seconds - decision_seconds)},
            "bindings": {"report_sha256": bound.hashes[run / "report.json"], "protocol_sha256": digest(protocol),
                         "protocol_file_sha256": bound.hashes[run / "protocol.json"], "base_results_sha256": bound.hashes[base / "results.json"],
                         "primary_manifest_sha256": protocol["primary_input_manifest_sha256"], "prepared_manifest_sha256": protocol["prepared_manifest_sha256"],
                         "producer_sha256": producer_sha, "auditor_sha256": common.rawsha(Path(__file__).read_bytes()),
                         "counting_helper_sha256": common.rawsha(HELPER.read_bytes())},
            "notes": ["All eight known evaluation cases; six target cases/norm pairs and two clear negative cases. No independent legal gold.",
                      "Frozen unadapted top-three rankings are replayed into exact whole original norms; the TFIDF algorithm is not rerun here.",
                      "All inherited fact request/response/spans and decision payloads are checked against original source and frozen R2 prompt/schema.",
                      "Any hint on clear negatives counts, even outside annotated no_hint IDs. Unjudged extra norms do not become false labels.",
                      "Facts are historical shared work, never new lexical model calls. Standalone latency includes their historical time.",
                      "GGUF identity is compared through bound manifests and original producer end verification; external model weights are not rehashed here.",
                      "Source attribution and target-ID matches do not validate paraphrases, relevance or legal application."]}


def markdown(report):
    s = report["summary"]
    c, n, p = s["evaluation_counts"], s["clear_negatives"], report["physical_execution"]
    return "\n".join([
        "# Unabhängige Nachzählung des lexikalischen R2-Folgearms", "",
        "Alle acht bekannten Evaluationsfälle; explorativer Folgeversuch ohne neue Faktenextraktion. Kein fachlicher Wahrheitsnachweis.", "",
        "| Fertig / geplant | Ziel-Fälle | Zielnormpaare | Klare Negative mit beliebigem Hinweis | Sämtliche Hint-Normpaare darauf | Explizite no_hint-Verstöße |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {c['execution'].get('complete', 0)}/8 | {c['positive_cases_with_known_target_hint']}/6 | {c['known_target_norms_emitted']}/6 | {n['cases_with_hint']}/2 | {n['hint_norm_pairs']} | {c['diagnostic_negative_norms_emitted']} |", "",
        f"Neu ausgeführt: {p['new_decisions']['completion_requests']} Entscheidungs-Anfragen, null Fakten-Anfragen. Entscheidungszeit {p['decision_seconds']:.2f} s; aufgezeichnete Ausführungsdauer {p['recorded_execution_seconds']:.2f} s. Die geerbten Fakten kosteten im ursprünglichen Lauf {p['inherited_fact_seconds']:.2f} s. Standalone-Median inklusive dieser historischen Faktenzeit: {s['standalone_timing']['median_seconds']:.2f} s.", "",
        "Technische Fehler bleiben im Nenner. Zusätzliche unannotierte Hinweise sind keine bestätigten Fehlklassifikationen; auf klar negativen Fällen werden dennoch sämtliche Ausgaben gezählt. Keine Präzisionsbehauptung aus bloß null Verstößen auf benannten Negativ-IDs.", "",
        f"Rohbericht SHA256: `{report['bindings']['report_sha256']}`. Vollständiges kanonisches Protokoll: `{report['bindings']['protocol_sha256']}`.", ""])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Use a fresh independent audit directory")
    report = audit(args.run)
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(markdown(report), encoding="utf-8")
    shutil.copyfile(__file__, args.out / "auditor-source.py")
    shutil.copyfile(HELPER, args.out / "audit_grounded_results.py")
    print(json.dumps({"status": report["status"], "cases": 8, "new_fact_calls": 0, "out": str(args.out.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
