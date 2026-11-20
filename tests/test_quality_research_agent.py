"""Model-free benchmark boundary, failure accounting and provenance checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import quality_research_agent as runner


def sentence(norms=("norm:TEST:1",), start=0, end=5):
    return {"start": start, "end": end, "satz": "Alpha" if start == 0 else "Beta",
            "normen": [{"id": norm} for norm in norms]}


def row(case_id="case-1", *, negative=False, query="Alpha", baseline=None):
    return {"case_id": case_id, "dataset": "main", "query": query, "negative": negative,
            "targets": [] if negative else ["norm:TEST:1", "norm:TEST:2"],
            "baseline": {"suggestions": baseline or []}, "arms": {}}


def response(suggestions=(), status="complete"):
    return {"status": status, "suggestions": list(suggestions), "unresolved": [], "errors": [],
            "trace": [], "timing": {}, "usage": {}, "initial_sources": [{"source_id": "private-source"}],
            "initial_sources_sha256": "a" * 64, "focuses": [{"start": 0, "end": 5}]}


def report(rows, arms=("static", "research"), diagnostic=False):
    return {"status": "running", "rows": rows, "arms": list(arms), "diagnostic_subset": diagnostic}


@pytest.fixture
def full_reference():
    rows = []
    for dataset, (positive, negative) in runner.DATASETS.items():
        for i in range(positive + negative):
            is_negative = i >= positive
            rows.append({"case_id": f"{dataset}:{i}", "dataset": dataset, "group": "synthetic",
                         "query": f"Synthetic text {i}.", "negative": is_negative,
                         "targets": [] if is_negative else ["norm:TEST:1"], "category": "test",
                         "uncertain": False, "treffer": {"suggestions": []}})
    return {"format": "quality_recall_live/1", "status": "mismatch", "public_profile": "treffer",
            "rows": rows, "source_audit": {"errors": []}}


def test_actual_mismatch_reference_is_valid_and_selection_never_reorders(full_reference):
    rows = runner.validate_reference(full_reference)
    selected = runner.select_cases(rows, [rows[7]["case_id"], rows[1]["case_id"]])
    assert selected == [rows[1], rows[7]]
    assert runner.select_cases(rows, case_limit=2) == rows[:2]


@pytest.mark.parametrize("change", ["old_format", "incomplete", "subset", "duplicate", "label_type",
                                    "empty_positive", "targeted_negative", "wrong_counts", "source_error"])
def test_reference_requires_complete_actual_outputs_and_fixed_denominators(full_reference, change):
    if change == "old_format":
        full_reference["format"] = "quality_passage_rerank_probe/1"
    elif change == "incomplete":
        full_reference["rows"].pop()
    elif change == "subset":
        full_reference["diagnostic_subset"] = True
    elif change == "duplicate":
        full_reference["rows"][1]["case_id"] = full_reference["rows"][0]["case_id"]
    elif change == "label_type":
        full_reference["rows"][0]["negative"] = 0
    elif change == "empty_positive":
        full_reference["rows"][0]["targets"] = []
    elif change == "targeted_negative":
        full_reference["rows"][-1]["targets"] = ["norm:TEST:1"]
    elif change == "wrong_counts":
        full_reference["rows"][0]["dataset"] = "additional"
    else:
        full_reference["source_audit"]["errors"].append("source differs")
    with pytest.raises(ValueError):
        runner.validate_reference(full_reference)


@pytest.mark.parametrize("ids,limit", [(["absent"], None), (["case-1", "case-1"], None), ([], 0), ([], True)])
def test_invalid_diagnostic_selection_fails(ids, limit):
    with pytest.raises(ValueError):
        runner.select_cases([row()], ids, limit)


@pytest.mark.parametrize("value", ["research,static,research", "unknown", "static,", "", "baseline"])
def test_private_or_duplicate_arm_names_are_rejected(value):
    with pytest.raises(ValueError):
        runner.parse_arms(value)


def test_failure_and_pending_negatives_remain_unknown_with_full_denominators():
    rows = [row("positive-hit"), row("positive-failed"), row("negative-output", negative=True),
            row("negative-failed", negative=True), row("negative-pending", negative=True)]
    rows[0]["arms"]["research"] = {"result": response([sentence()])}
    rows[1]["arms"]["research"] = {"result": response(status="timeout")}
    rows[2]["arms"]["research"] = {"result": response([sentence(), sentence(start=6, end=10)])}
    rows[3]["arms"]["research"] = {"result": response([sentence()], status="transport_error")}
    result = runner.counts(rows, "research", diagnostic_subset=False)
    assert result["cases"] == 5
    assert result["positive_cases"] == 2 and result["negative_cases"] == 3
    assert result["hit5_cases"] == 1 and result["failed_positive_cases"] == 1
    assert result["failed_negative_cases"] == result["pending_negative_cases"] == 1
    assert result["negative_with_suggestion"] == 1
    assert result["negative_entries"] == 2 and result["negative_unique_case_norm_pairs"] == 1
    assert result["historical_norm_targets"] == 4 and result["matched_historical_norm_targets"] == 1
    assert result["rates"] == {"hit5_lower_bound": .5, "negative_output_lower_bound": 1 / 3,
                               "negative_output_upper_bound": 1.}
    assert result["suggestion_precision"]["status"] == "unknown"


def test_pilot_has_only_selected_counts_and_no_full_benchmark_rate():
    rows = [row()]
    rows[0]["arms"]["static"] = {"result": response([sentence()])}
    value = runner.counts(rows, "static", diagnostic_subset=True)
    assert value["hit5_cases"] == value["cases"] == 1
    assert value["rates"] is None


@pytest.mark.parametrize("change", ["six_norms", "duplicate_norm", "duplicate_focus", "bool_offset", "partial_focus", "wrong_text"])
def test_source_sentence_cap_cannot_be_evaded(change):
    suggestions = [sentence()]
    if change == "six_norms":
        suggestions[0] = sentence(tuple(f"norm:TEST:{i}" for i in range(6)))
    elif change == "duplicate_norm":
        suggestions[0] = sentence(("norm:TEST:1", "norm:TEST:1"))
    elif change == "duplicate_focus":
        suggestions.append(sentence(("norm:TEST:2",)))
    elif change == "bool_offset":
        suggestions[0]["start"] = False
    elif change == "partial_focus":
        suggestions[0].update(end=4, satz="Alph")
    else:
        suggestions[0]["satz"] = "Invented text"
    with pytest.raises(ValueError):
        runner.validate_suggestions("Alpha", suggestions, {(0, 5)})


def test_fresh_sessions_receive_only_text_corpus_and_research_permission(tmp_path):
    corpus = object()
    rows = [row("secret-label-id"), row("secret-negative-id", negative=True)]
    value = report(rows)
    calls, instances = [], []
    class Agent:
        def __init__(self):
            instances.append(self)
        def run(self, *args, **kwargs):
            assert args == ("Alpha", corpus)
            assert set(kwargs) == {"allow_research"}
            calls.append(kwargs["allow_research"])
            return response([sentence()])
    runner.execute_cases(value, tmp_path, corpus, Agent,
                         lambda text: [SimpleNamespace(focus_start=0, focus_end=len(text))])
    assert len(instances) == 4 and calls == [False, True, False, True]
    lines = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(lines) == 4
    assert json.loads((tmp_path / "report.json").read_bytes())["rows"] == rows


def test_expected_technical_status_does_not_abort_or_become_success(tmp_path):
    responses = iter([response(status="timeout"), response()])
    class Agent:
        def run(self, *args, **kwargs):
            return next(responses)
    value = report([row(negative=True)])
    runner.execute_cases(value, tmp_path, object(), Agent,
                         lambda text: [SimpleNamespace(focus_start=0, focus_end=len(text))])
    assert value["summaries"]["all"]["static"]["failed_negative_cases"] == 1
    assert value["summaries"]["all"]["research"]["complete_cases"] == 1


def test_unhandled_error_is_journaled_then_aborts_with_future_cases_pending(tmp_path):
    class Agent:
        def run(self, *args, **kwargs):
            raise RuntimeError("unexpected implementation failure")
    value = report([row(), row("later", negative=True)])
    with pytest.raises(RuntimeError, match="unexpected"):
        runner.execute_cases(value, tmp_path, object(), Agent,
                             lambda text: [SimpleNamespace(focus_start=0, focus_end=len(text))])
    saved = json.loads((tmp_path / "report.json").read_bytes())
    assert saved["status"] == "failed"
    assert saved["summaries"]["all"]["static"]["failed_cases"] == 1
    assert saved["summaries"]["all"]["static"]["pending_cases"] == 1
    assert saved["summaries"]["all"]["research"]["pending_cases"] == 2
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 1


def test_different_start_sources_in_two_arms_fail_without_discarding_raw_result(tmp_path):
    first, second = response(), response()
    second["initial_sources"] = [{"source_id": "another source"}]
    responses = iter([first, second])
    class Agent:
        def run(self, *args, **kwargs):
            return next(responses)
    value = report([row()])
    with pytest.raises(ValueError, match="initialization differs"):
        runner.execute_cases(value, tmp_path, object(), Agent,
                             lambda text: [SimpleNamespace(focus_start=0, focus_end=len(text))])
    assert value["rows"][0]["arms"]["research"]["invalid_agent_result"] == second
    assert value["summaries"]["all"]["research"]["failed_cases"] == 1


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_ambiguous_json_is_rejected(text):
    with pytest.raises(ValueError):
        runner.parse_json(text)


@pytest.mark.parametrize("url", ["https://127.0.0.1:8089", "http://example.com:8089", "http://localhost:8089/v1",
                                 "http://user@127.0.0.1:8089", "http://127.0.0.1:8089?x=1", "http://localhost"])
def test_metadata_url_never_targets_a_remote_or_ambiguous_endpoint(url):
    with pytest.raises(ValueError):
        runner.local_base_url(url)


def test_json_snapshot_and_end_checks_bind_exact_bytes(tmp_path):
    source = tmp_path / "original.json"
    raw = b'{ "one": 1 }\n'
    source.write_bytes(raw)
    inputs = {}
    value, sha = runner.freeze_json(source, tmp_path / "frozen.json", inputs)
    assert value == {"one": 1} and sha == hashlib.sha256(raw).hexdigest()
    assert (tmp_path / "frozen.json").read_bytes() == raw
    runner.check_bindings(inputs, {})
    source.write_text('{"one":2}')
    with pytest.raises(ValueError, match="Bound input changed"):
        runner.check_bindings(inputs, {})
    (tmp_path / "frozen.json").write_text('{}')
    with pytest.raises(ValueError, match="Frozen artifact"):
        runner.check_frozen(tmp_path, {"frozen.json": sha})


@pytest.fixture
def model_files(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    model_path, executable, source = (root / name for name in ("model.gguf", "server.exe", "source.py"))
    for path in (model_path, executable, source):
        path.write_bytes(path.name.encode())
    model = {"format": "quality_research_model/1", "status": "complete", "repo": "test/model",
             "revision": "a" * 40, "quantization": "Q6_K", "model": {"path": str(model_path.resolve()),
                "bytes": model_path.stat().st_size, "sha256": runner.digest(model_path)}}
    model_manifest = root / "model-manifest.json"
    runner.save(model_manifest, model)
    launch = {"format": "quality_research_launch/1", "status": "healthy_handoff", "model": model["model"],
              "model_manifest_sha256": runner.digest(model_manifest), "base_url": "http://127.0.0.1:8089",
              "alias": "test-model", "pid": 123, "executable": str(executable.resolve()),
              "source_sha256": runner.digest(source), "engine_files": {"server.exe": {
                  "bytes": executable.stat().st_size, "sha256": runner.digest(executable)}},
              "argv": [str(executable.resolve()), "-m", str(model_path.resolve()), "--alias", "test-model",
                       "--ctx-size", "16384", "--reasoning-budget", "768", "--parallel", "1",
                       "--host", "127.0.0.1", "--port", "8089"]}
    launch_manifest = root / "launch-manifest.json"
    runner.save(launch_manifest, launch)
    props = {"model_alias": "test-model", "model_path": str(model_path.resolve()), "model_ftype": "Q6_K",
             "total_slots": 1, "default_generation_settings": {"n_ctx": 16384},
             "build_info": "b9986-91c631b21", "chat_template": "test-template"}
    runner.save(root / "props.response.json", props)
    runner.save(root / "report.json", {"format": "quality_research_setup/1", "status": "pass",
                "launch_manifest_sha256": runner.digest(launch_manifest),
                "artifact_hashes": {"props.response.json": runner.digest(root / "props.response.json")}})
    args = SimpleNamespace(model_manifest=model_manifest, launch_manifest=launch_manifest,
                           url="http://127.0.0.1:8089", model="test-model")
    out = tmp_path / "out"
    out.mkdir()
    return args, launch, out, SimpleNamespace(context_tokens=16384, reasoning_budget_tokens=768)


def test_model_binary_and_launch_source_are_all_bound_without_loading(model_files):
    args, launch, out, config = model_files
    inputs = {}
    result = runner.bind_model(args, out, inputs, config)
    assert result["launch"]["alias"] == "test-model"
    assert len(inputs) == 7  # Manifests, GGUF, executable, producer, setup report and properties.
    assert (out / "launch-source.py").read_bytes() == args.launch_manifest.with_name("source.py").read_bytes()
    runner.check_bindings(inputs, {})


@pytest.mark.parametrize("tamper", ["alias", "model_manifest", "argv_budget", "duplicate_arg", "binary", "source"])
def test_changed_server_or_model_identity_is_rejected(model_files, tamper):
    args, launch, out, config = model_files
    if tamper == "alias":
        launch["alias"] = "different"
    elif tamper == "model_manifest":
        launch["model_manifest_sha256"] = "0" * 64
    elif tamper == "argv_budget":
        launch["argv"][launch["argv"].index("--ctx-size") + 1] = "32768"
    elif tamper == "duplicate_arg":
        launch["argv"] += ["--alias", "test-model"]
    elif tamper == "binary":
        Path(launch["executable"]).write_bytes(b"changed")
    else:
        args.launch_manifest.with_name("source.py").write_bytes(b"changed")
    runner.save(args.launch_manifest, launch)
    with pytest.raises(ValueError):
        runner.bind_model(args, out, {}, config)


def test_parser_keeps_fixed_default_arms_and_budget():
    args = runner.parser().parse_args(["--model", "test-model", "--model-manifest", "model.json",
        "--launch-manifest", "launch.json", "--out", "fresh"])
    assert args.arms == "static,research" and args.timeout == 180.
    assert args.case_limit is None and args.case_id == []
    assert args.retrieval == "bm25" and args.encoder_model is None
    assert args.reuse_prompt_cache is False and args.finish_reserve == 0.
    assert args.catalog is None


def test_catalog_is_bound_byte_for_byte_against_existing_reference_before_runtime(tmp_path):
    source = tmp_path / "begriffe.yaml"
    raw = b'# original whitespace\r\nversion: "test"\r\nbegriffe: []\r\n'
    source.write_bytes(raw)
    out, inputs = tmp_path / "out", {}
    sha = hashlib.sha256(raw).hexdigest()
    path, provenance = runner.freeze_catalog(source, out, inputs, {"old/source/catalog/begriffe.yaml": sha})
    assert path.read_bytes() == raw and provenance == {"snapshot": "inputs/catalog.yaml", "sha256": sha,
                                                      "size_bytes": len(raw), "reference_sha256": sha}
    assert inputs == {str(source.resolve()): sha}
    assert runner.freeze_catalog(None, out, inputs, {}) == (None, None)
    with pytest.raises(ValueError, match="live reference"):
        runner.freeze_catalog(source, out, inputs, {"old/source/catalog/begriffe.yaml": "0" * 64})


@pytest.mark.parametrize("tamper", [None, "sha256", "size_bytes", "path", "format"])
def test_catalog_runtime_metadata_must_match_frozen_yaml(tmp_path, tamper):
    provenance = {"sha256": "a" * 64, "size_bytes": 123}
    metadata = {"format": "research_catalog/1", **provenance, "path": str(tmp_path)}
    if tamper is None:
        runner.validate_catalog(metadata, provenance, tmp_path)
        runner.validate_catalog(None, None, None)
        with pytest.raises(ValueError, match="Disabled catalog"):
            runner.validate_catalog(metadata, None, None)
        return
    metadata[tamper] = 124 if tamper == "size_bytes" else "different"
    with pytest.raises(ValueError, match="Catalog identity"):
        runner.validate_catalog(metadata, provenance, tmp_path)


def test_explicit_execution_policy_flags_reach_research_config():
    from dataclasses import asdict
    from risk_layer.research_agent import ResearchConfig
    args = runner.parser().parse_args(["--model", "test-model", "--model-manifest", "model.json",
        "--launch-manifest", "launch.json", "--out", "fresh", "--reuse-prompt-cache", "--finish-reserve", "35"])
    config = ResearchConfig(timeout_seconds=args.timeout, reuse_prompt_cache=args.reuse_prompt_cache,
                            finish_reserve_seconds=args.finish_reserve)
    assert asdict(config)["reuse_prompt_cache"] is True
    assert asdict(config)["finish_reserve_seconds"] == 35.
    assert config.timeout_seconds == 180.


def test_encoder_snapshot_preserves_pinned_hf_suffix_and_binds_before_loading(tmp_path):
    source = tmp_path / "cache/models--BAAI--bge-m3/snapshots" / ("a" * 40)
    (source / "1_Pooling").mkdir(parents=True)
    (source / "1_Pooling/config.json").write_bytes(b'{"pooling":true}')
    (source / "pytorch_model.bin").write_bytes(b"synthetic local encoder weights")
    out, inputs = tmp_path / "out", {}
    out.mkdir()
    path, provenance = runner.freeze_encoder("hybrid", source, out, inputs)
    assert path.relative_to(out).as_posix() == "inputs/encoder/models--BAAI--bge-m3/snapshots/" + "a" * 40
    assert provenance["files"] == {name: runner.digest(path / name) for name in ("1_Pooling/config.json", "pytorch_model.bin")}
    assert len(inputs) == 2
    assert (path / "pytorch_model.bin").read_bytes() == (source / "pytorch_model.bin").read_bytes()
    runner.check_bindings(inputs, {})
    (source / "pytorch_model.bin").write_bytes(b"mutated original")
    with pytest.raises(ValueError, match="Bound input changed"):
        runner.check_bindings(inputs, {})


@pytest.mark.parametrize("mode,model", [("hybrid", None), ("bm25", "unused"), ("unknown", None)])
def test_retrieval_does_not_choose_an_unbound_or_unused_encoder(tmp_path, mode, model):
    with pytest.raises(ValueError):
        runner.freeze_encoder(mode, model, tmp_path, {})
    assert runner.freeze_encoder("bm25", None, tmp_path, {}) == (None, {"mode": "bm25", "files": {}})


@pytest.mark.parametrize("tamper", [None, "mode", "files", "device", "dtype", "threads", "local", "path"])
def test_hybrid_identity_matches_exact_snapshot_and_cpu_contract(tmp_path, tamper):
    model = {"files": {"model.bin": "a" * 64}, "encoder": "bge-m3", "device": "cpu", "dtype": "float32",
             "cpu_threads": 2, "local_files_only": True}
    metadata = {"mode": "hybrid", "model_identity": model, "model_path": str(tmp_path)}
    provenance = {"mode": "hybrid", "files": deepcopy(model["files"])}
    if tamper is None:
        runner.validate_retrieval(metadata, provenance, tmp_path)
        return
    if tamper == "mode":
        metadata["mode"] = "bm25"
    elif tamper == "path":
        metadata["model_path"] = str(tmp_path / "elsewhere")
    else:
        key = {"threads": "cpu_threads", "local": "local_files_only"}.get(tamper, tamper)
        model[key] = {"files": {}, "device": "cuda:0", "dtype": "float16", "threads": True, "local": False}[tamper]
    with pytest.raises(ValueError, match="differs"):
        runner.validate_retrieval(metadata, provenance, tmp_path)


def test_retrieval_deltas_exclude_prepare_and_keep_failed_query_records():
    before = {"model_load_count": 1, "model_load_seconds": 2., "query_count": 1, "encode_seconds": .1,
              "queries": [{"number": 1, "status": "complete"}]}
    after = deepcopy(before)
    after.update(query_count=2, encode_seconds=.3)
    after["queries"].append({"number": 2, "status": "failed"})
    delta = runner.retrieval_delta(before, after)
    assert delta["model_load_count"] == 0 and delta["model_load_seconds"] == 0
    assert delta["query_count"] == 1 and delta["encode_seconds"] == pytest.approx(.2)
    assert delta["queries"] == [{"number": 2, "status": "failed"}]
    after["queries"][0]["status"] = "changed"
    with pytest.raises(ValueError, match="journal changed"):
        runner.retrieval_delta(before, after)


def test_failed_agent_still_records_measured_retrieval_work(tmp_path):
    corpus = SimpleNamespace(retrieval_metrics={"query_count": 1, "queries": [{"number": 1}]})
    class Agent:
        def run(self, *args, **kwargs):
            corpus.retrieval_metrics["query_count"] += 1
            corpus.retrieval_metrics["queries"].append({"number": 2, "status": "failed"})
            raise RuntimeError("failure after retrieval")
    value = report([row()])
    value["retrieval"] = {"mode": "hybrid"}
    with pytest.raises(RuntimeError):
        runner.execute_cases(value, tmp_path, corpus, Agent,
                             lambda text: [SimpleNamespace(focus_start=0, focus_end=len(text))])
    entry = value["rows"][0]["arms"]["static"]
    assert entry["retrieval_metrics"] == {"query_count": 1, "queries": [{"number": 2, "status": "failed"}]}
    assert entry["result"]["status"] == "runner_error"
    assert value["retrieval_metrics"]["query_count"] == 2


@pytest.mark.parametrize("tamper", [None, "path", "alias", "quantization", "slots", "context", "build", "template"])
def test_live_server_properties_must_match_bound_setup(model_files, tamper):
    args, launch, out, config = model_files
    provenance = runner.bind_model(args, out, {}, config)
    values = {"models": {"data": [{"id": args.model}]}, "props": deepcopy(provenance["setup_props"])}
    if tamper is None:
        runner.validate_server(values, args.model, provenance, config)
        return
    key = {"path": "model_path", "alias": "model_alias", "quantization": "model_ftype", "slots": "total_slots",
           "build": "build_info", "template": "chat_template"}.get(tamper)
    if tamper == "context":
        values["props"]["default_generation_settings"]["n_ctx"] = 32768
    else:
        values["props"][key] = 2 if tamper == "slots" else "different"
    with pytest.raises(ValueError):
        runner.validate_server(values, args.model, provenance, config)


@pytest.mark.parametrize("canonical,display", [("Q4_K_M", "Q4_K - Medium"), ("Q6_K", "Q6_K")])
def test_versioned_server_quantization_labels_bind_exact_canonical_manifest(model_files, canonical, display):
    args, _, out, config = model_files
    provenance = runner.bind_model(args, out, {}, config)
    provenance["model"]["quantization"] = canonical
    provenance["setup_props"]["model_ftype"] = display
    values = {"models": {"data": [{"id": args.model}]}, "props": deepcopy(provenance["setup_props"])}
    runner.validate_server(values, args.model, provenance, config)
    assert runner.MODEL_FTYPE_MAPPING_VERSION == "llama.cpp/model_ftype/1"


@pytest.mark.parametrize("tamper", ["unknown_build", "unknown_quantization", "canonical_in_props", "case", "whitespace", "different_setup"])
def test_quantization_mapping_does_not_fuzzily_normalize_or_ignore_setup(model_files, tamper):
    args, _, out, config = model_files
    provenance = runner.bind_model(args, out, {}, config)
    provenance["model"]["quantization"] = "Q4_K_M"
    provenance["setup_props"]["model_ftype"] = "Q4_K - Medium"
    values = {"models": {"data": [{"id": args.model}]}, "props": deepcopy(provenance["setup_props"])}
    if tamper == "unknown_build":
        values["props"]["build_info"] = provenance["setup_props"]["build_info"] = "b9987-unknown"
    elif tamper == "unknown_quantization":
        provenance["model"]["quantization"] = "Q4_K_S"
        values["props"]["model_ftype"] = provenance["setup_props"]["model_ftype"] = "Q4_K - Small"
    elif tamper == "different_setup":
        provenance["setup_props"]["model_ftype"] = "Q6_K"
    else:
        values["props"]["model_ftype"] = {"canonical_in_props": "Q4_K_M", "case": "q4_k - medium",
                                            "whitespace": "Q4_K - Medium "}[tamper]
    with pytest.raises(ValueError):
        runner.validate_server(values, args.model, provenance, config)
