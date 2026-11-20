"""Prepare immutable diagnostic inputs, run local models, and aggregate results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from experiments.grounded_signal import VERSION
from experiments.grounded_signal import runtime
from risk_layer.research_agent import ResearchAgent, ResearchConfig
from tools.quality_research_agent import bind_model, snapshot_server


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("direct_prepared", "structured_prepared", "structured_retrieved")


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(raw, encoding="utf-8")
    temporary.replace(path)


def read(path):
    from risk_layer.norm_check import _strict_json
    return _strict_json(Path(path).read_bytes())


def code_paths():
    return sorted({*ROOT.joinpath("experiments").rglob("*.py"),
                   *ROOT.joinpath("risk_layer").glob("*.py"),
                   *ROOT.joinpath("tests").glob("test_grounded_signal_*.py"),
                   ROOT / "tools/quality_research_agent.py", ROOT / "tools/research_local.py",
                   ROOT / "tools/quality_research_model.py", ROOT / "tools/quality_research_moe_serve.py",
                   ROOT / "pyproject.toml"})


def source_hashes():
    return {path.relative_to(ROOT).as_posix(): sha_file(path) for path in code_paths()}


def freeze_code(out):
    hashes = source_hashes()
    for name, digest in hashes.items():
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha_file(target) != digest:
            raise ValueError("Source changed while freezing")
    return hashes


def artifact_hashes(directory):
    return {p.relative_to(directory).as_posix(): sha_file(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != "input-manifest.json" and "__pycache__" not in p.parts}


def prepare(args):
    from experiments.grounded_signal.cases import dataset_manifest, load_cases
    from experiments.grounded_signal.sources import SourceCorpus, render_package
    from experiments.grounded_signal.learning import compare_retrieval, fit_examples, retrieve, select_examples
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prepared_code = freeze_code(out)
    cases = load_cases()
    save(out / "cases.json", cases)
    save(out / "dataset-manifest.json", dataset_manifest())
    packages, families, automatic, search_rows = {}, {}, {}, []
    graph_before = sha_file(args.graph)
    if graph_before != dataset_manifest()["source_graph_sha256"]:
        raise ValueError("Dataset source rationales are bound to a different graph")
    shutil.copyfile(args.graph, out / "graph.sqlite")
    if sha_file(out / "graph.sqlite") != graph_before:
        raise ValueError("Graph changed while copying")
    from tools.quality_research_agent import freeze_encoder
    encoder_inputs: dict[str, str] = {}
    encoder_path, encoder_provenance = freeze_encoder(args.retrieval, args.encoder_model, out, encoder_inputs)
    index_inputs: dict[str, str] = {}
    index_path = args.index
    if args.retrieval == "hybrid":
        from tools.quality_research_agent import snapshot_file
        for path in sorted(args.index.rglob("*")):
            if path.is_file():
                snapshot_file(path, out / "index" / path.relative_to(args.index), index_inputs)
        index_path = out / "index"
    with SourceCorpus(out / "graph.sqlite", index_path, retrieval=args.retrieval,
                      encoder_model_path=encoder_path) as corpus:
        corpus.prepare_lookup()
        for case in cases:
            family = case["family_id"]
            expected = tuple(case["prepared_norm_ids"])
            if family not in families:
                package = render_package(corpus.package(expected, max_chars=args.max_chars))
                families[family] = package["package_id"]
                packages[package["package_id"]] = package
            elif set(packages[families[family]]["candidate_norm_ids"]) != set(expected):
                raise ValueError("Prepared source selection differs inside a contrast family")
            hits = corpus.search(case["text"], limit=args.top_k)
            auto = (render_package(corpus.package([hit.norm_id for hit in hits], max_chars=args.max_chars))
                    if hits else empty_retrieval_package())
            packages[auto["package_id"]] = auto
            automatic[case["case_id"]] = auto["package_id"]
            search_rows.append({"case_id": case["case_id"], "query": case["text"],
                                "hits": [asdict(hit) for hit in hits], "package_id": auto["package_id"]})
        snapshot = corpus.export_snapshot(out / "corpus")
    all_units = read(out / "corpus/units.json")
    documents = defaultdict(list)
    for unit in all_units:
        documents[unit["norm_id"]].append(unit)
    norm_documents = {nid: "".join(unit["text"] for unit in sorted(units, key=lambda u: u["start"]))
                      for nid, units in documents.items()}
    development = sorted({c["family_id"] for c in cases if c["split"] == "development"})
    evaluation = sorted({c["family_id"] for c in cases if c["split"] == "evaluation"})
    prototype = fit_examples(cases, development_families=development, evaluation_families=evaluation,
                             norm_documents=norm_documents)
    save(out / "learning-prototype.json", prototype)
    learned_rows = [{"case_id": c["case_id"], "split": c["split"],
                     "unadapted": retrieve(c["text"], prototype, k=args.top_k),
                     "adapted": retrieve(c["text"], prototype, k=args.top_k, adapted=True),
                     "example_preview": select_examples(c["text"], prototype)} for c in cases]
    save(out / "learning-diagnostic.json", learned_rows)
    save(out / "learning-comparison.json", compare_retrieval(
        [case for case in cases if case["split"] == "evaluation"], prototype, k=args.top_k))
    save(out / "search-rows.json", search_rows)
    save(out / "packages.json", packages)
    selected_units = {unit["unit_id"]: unit for package in packages.values() for unit in package["units"]}
    save(out / "source-units.json", selected_units)
    cases_by_arm = {arm: {c["case_id"]: (automatic[c["case_id"]] if arm == "structured_retrieved"
                                           else families[c["family_id"]]) for c in cases} for arm in ARMS}
    protocol = {"format": VERSION, "arms": list(ARMS), "family_packages": families,
                "case_packages": cases_by_arm,
                "arm_policies": {arm: {"source_mode": "automatic" if arm == "structured_retrieved"
                                                    else "family_prepared"} for arm in ARMS},
                "annotation_status": "assistant_reviewed_synthetic", "case_ids": [c["case_id"] for c in cases],
                "source_retrieval": args.retrieval, "top_k": args.top_k, "max_package_chars": args.max_chars,
                "graph_sha256": graph_before, "source_snapshot": snapshot,
                "encoder_provenance": encoder_provenance, "encoder_original_hashes": encoder_inputs,
                "first_pass_labels": "synthetic diagnostic expectations, not independently adjudicated gold",
                "learning_runtime": "disabled; CPU ranking comparison and train-only examples exported separately",
                "comparison": "Model identities differ in architecture and quantization; not a pure parameter-size effect",
                "thinking": True, "reasoning_budget_tokens": 768, "max_output_tokens": 3072,
                "context_tokens": 16384, "timeout_seconds": 300,
                "model_interface": {"format": "fixed_norm_slots/2", "tool_choice": "required",
                                    "provider": "llama.cpp b9986", "one_tool_per_stage": True,
                                    "schema_constraints": "Required norm slots, same-norm sources, status/relation branches",
                                    "remaining_checks": "Host checks dynamic fact references and duplicates; semantics unassessed"},
                "prior_case_exposure": "First four development cases were used in the aborted R1 interface pilot; cases and labels unchanged",
                "fact_reuse": "One source-free fact extraction per model/case shared by structured arms",
                "retries": 0, "source_hashes": prepared_code, "index_original_hashes": index_inputs}
    if source_hashes() != prepared_code or sha_file(args.graph) != graph_before:
        raise ValueError("Source code or graph changed during input preparation")
    if any(sha_file(path) != expected for path, expected in {**encoder_inputs, **index_inputs}.items()):
        raise ValueError("Original encoder or index changed during preparation")
    save(out / "protocol.json", protocol)
    manifest = {"format": "grounded_signal_inputs/1", "prepared_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": artifact_hashes(out)}
    save(out / "input-manifest.json", manifest)
    print(json.dumps({"status": "prepared", "cases": len(cases), "families": len(families),
                      "packages": len(packages), "package_statuses": dict(Counter(p["status"] for p in packages.values())),
                      "out": str(out)}), flush=True)


def load_inputs(directory):
    manifest = read(directory / "input-manifest.json")
    actual = artifact_hashes(directory)
    if actual != manifest["artifacts"]:
        raise ValueError("Prepared input artifacts changed")
    protocol = read(directory / "protocol.json")
    if source_hashes() != protocol["source_hashes"]:
        raise ValueError("Source code differs from the prepared protocol")
    return (read(directory / "cases.json"), read(directory / "packages.json"),
            read(directory / "source-units.json"), protocol)


def empty_retrieval_package():
    from experiments.grounded_signal.sources import package_fingerprint
    payload = {"status": "missing", "complete_local_norms": False, "candidate_norm_ids": [],
               "norm_ids": [], "unit_ids": [], "units": [], "required_groups": [],
               "limitations": ["Search returned no candidate sources; no model decision is possible."],
               "missing_norm_ids": [], "reference_closure_complete": False, "legal_completeness": "unassessed"}
    fingerprint = package_fingerprint(payload)
    return {**payload, "fingerprint": fingerprint, "package_id": "pkg:" + fingerprint}


def run(args):
    from experiments.grounded_signal.evaluation import evaluate, validate_answer
    inputs = args.inputs.resolve()
    cases, packages, units, protocol = load_inputs(inputs)
    prepared_manifest_sha = sha_file(inputs / "input-manifest.json")
    prepared_artifacts = read(inputs / "input-manifest.json")["artifacts"]
    selected_arms = args.arms.split(",") if args.arms else list(ARMS)
    if len(set(selected_arms)) != len(selected_arms) or not set(selected_arms) <= set(ARMS):
        raise ValueError("Unknown/repeated arm")
    if args.case_id:
        if len(set(args.case_id)) != len(args.case_id) or not set(args.case_id) <= {c["case_id"] for c in cases}:
            raise ValueError("Unknown/repeated case selection")
        cases = [c for c in cases if c["case_id"] in args.case_id]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = ResearchConfig(max_output_tokens=3072, timeout_seconds=300, reuse_prompt_cache=True)
    bound_inputs: dict[str, str] = {}
    provenance = bind_model(args, out, bound_inputs, config)
    snapshot_server(args.url, args.model, out, 30, provenance, config)
    frozen = freeze_code(out)
    run_protocol = {**protocol, "arms": selected_arms, "case_ids": [c["case_id"] for c in cases]}
    run_protocol["arm_policies"] = {arm: protocol["arm_policies"][arm] for arm in selected_arms}
    run_protocol["case_packages"] = {arm: {c["case_id"]: protocol["case_packages"][arm][c["case_id"]]
                                               for c in cases} for arm in selected_arms}
    run_protocol["family_packages"] = {c["family_id"]: protocol["family_packages"][c["family_id"]] for c in cases}
    run_protocol.update({"inputs_directory": str(inputs),
                                 "input_manifest_sha256": prepared_manifest_sha,
                                 "model_binding": provenance, "model_input_hashes": bound_inputs,
                                 "configuration": asdict(config), "source_hashes": frozen})
    save(out / "protocol.json", run_protocol)
    agent = ResearchAgent(args.url, args.model, config)
    results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in selected_arms}
    fact_results: dict[str, Any] = {}
    started = time.monotonic()
    status = "running"
    save(out / "results.json", {"status": status, "results": results, "fact_results": fact_results})
    for index, case in enumerate(cases, 1):
        cid = case["case_id"]
        if any(arm != "direct_prepared" for arm in selected_arms):
            fact_results[cid] = runtime.extract(agent, case["text"])
            save(out / f"facts-{index:03}.json", fact_results[cid])
        for arm in selected_arms:
            pid = protocol["case_packages"][arm][cid]
            package = packages[pid]
            row = {"case_id": cid, "source_package_id": pid,
                   "candidate_norm_ids": list(package["candidate_norm_ids"])}
            if package["status"] != "ready":
                call = {"status": "source_package_" + package["status"], "errors": ["Atomic package unavailable"],
                        "timing": {"elapsed_seconds": 0.0}, "usage": {"model_calls": 0}}
            else:
                call = {"status": "running", "errors": [], "timing": {"elapsed_seconds": 0.0}}
                try:
                    call = runtime.decide(agent, case["text"], package, fact_results.get(cid),
                                          direct=arm == "direct_prepared")
                    if call["status"] == "complete":
                        audit = validate_answer(call["answer"], text=case["text"], candidate_norm_ids=row["candidate_norm_ids"],
                                                source_units=units, available_unit_ids=[u["unit_id"] for u in package["units"]])
                        if not audit["valid"]:
                            raise ValueError("Independent answer attribution check failed: " + str(audit["errors"]))
                        call["attribution_audit"] = audit
                except (ValueError, TypeError, KeyError) as exc:
                    call["status"] = "invalid_input_or_answer"
                    call.setdefault("errors", []).append(str(exc))
                    if "answer" in call:
                        call["rejected_answer"] = call.pop("answer")
            timing = dict(call.get("timing", {}))
            timing["decision_seconds"] = timing.get("elapsed_seconds", 0.0)
            timing["fact_seconds"] = (fact_results[cid]["timing"]["elapsed_seconds"]
                                      if arm != "direct_prepared" else 0.0)
            timing["elapsed_seconds"] = timing["decision_seconds"] + timing["fact_seconds"]
            timing["scope"] = "standalone arm latency; the same fact call is physically shared by structured arms"
            row.update(status=call["status"], answer=call.get("answer"), error=call.get("errors", []),
                       timing=timing, usage=call.get("usage", {}),
                       fact_stage_status=fact_results.get(cid, {}).get("status"))
            results[arm].append(row)
            save(out / f"case-{index:03}-{arm}.json", {"row": row, "call": call})
            save(out / "results.json", {"status": status, "results": results, "fact_results": fact_results})
            print(json.dumps({"case": index, "planned_cases": len(cases), "case_id": cid, "arm": arm,
                              "status": row["status"], "seconds": row["timing"].get("elapsed_seconds")}), flush=True)
    if (source_hashes() != frozen or sha_file(inputs / "input-manifest.json") != prepared_manifest_sha
            or artifact_hashes(inputs) != prepared_artifacts):
        raise ValueError("Source/input changes during model run")
    if any(sha_file(path) != expected for path, expected in bound_inputs.items()):
        raise ValueError("Bound model/engine/launch inputs changed during run")
    status = "complete" if all(row["status"] == "complete" for rows in results.values() for row in rows) else "incomplete"
    evaluation = evaluate(cases, results, protocol=run_protocol, source_units=units, source_packages=packages)
    save(out / "evaluation.json", evaluation)
    save(out / "results.json", {"status": status, "results": results, "fact_results": fact_results,
                                "elapsed_seconds": time.monotonic() - started, "source_end_hashes": source_hashes()})
    print(json.dumps({"status": status, "out": str(out), "cases": len(cases), "arms": selected_arms}), flush=True)
    return 0 if status == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--graph", type=Path, default=ROOT / "out/gpu-ab-20260907-final/artifacts/graph.sqlite")
    prep.add_argument("--index", type=Path, default=ROOT / "out/quality-production-final/index")
    prep.add_argument("--retrieval", choices=("bm25", "hybrid"), default="hybrid")
    prep.add_argument("--encoder-model", type=Path)
    prep.add_argument("--top-k", type=int, default=3)
    prep.add_argument("--max-chars", type=int, default=32000)
    prep.add_argument("--out", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--inputs", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--url", required=True)
    execute.add_argument("--model", required=True)
    execute.add_argument("--model-manifest", type=Path, required=True)
    execute.add_argument("--launch-manifest", type=Path, required=True)
    execute.add_argument("--arms")
    execute.add_argument("--case-id", action="append")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
