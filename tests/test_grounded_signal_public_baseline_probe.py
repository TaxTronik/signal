from copy import deepcopy
from dataclasses import asdict
import hashlib
import subprocess
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.grounded_signal import public_baseline_probe as probe


def cases():
    values = []
    for i in range(16):
        positive = i % 2 == 0
        values.append({"case_id": f"c{i}", "family_id": f"f{i//2}", "split": "development" if i < 8 else "evaluation",
                       "text": f"Ein Vorgang {i}.", "diagnostic_labels": {
                           "annotation_status": "assistant_reviewed_synthetic", "hint_expected": positive,
                           "target_norm_ids": ["norm:AO:1"] if positive else [], "allowed_hint_norm_ids": ["norm:AO:2"],
                           "no_hint_norm_ids": [] if positive else ["norm:AO:1"], "insufficient_context_norm_ids": []}})
    return values


def primary(data):
    return {"case_ids": [c["case_id"] for c in data], "model_interface": {"format": "fixed_norm_slots/2"}}


def output(norms=(), hard=()):
    return {"suggestions": [{"normen": [{"id": n} for n in norms]}] if norms else [], "hard_norm_ids": list(hard)}


def row(case, status="complete", norms=(), hard=()):
    return {"case_id": case["case_id"], "status": status, "output": output(norms, hard) if status == "complete" else None,
            "timing": {"elapsed_seconds": 2.0}}


def sources():
    text = "Ein beschriebener Vorgang"
    source = "Originalquelle mit Fortsetzung."
    parent = SimpleNamespace(passage_id="p1", norm_id="norm:AO:1", start=0, end=len(source), text=source)
    index = SimpleNamespace(passage_evidence=SimpleNamespace(passages=[parent], source_fingerprint="fp", anchors=(("norm:AO:1", "Hinweis"),)))
    graph = SimpleNamespace(node=lambda _: {"text": source, "titel": "Normtitel"})
    f = probe.foci_for(text)[0]
    suggestions = [{"start": f["focus_start"], "end": f["focus_end"], "satz": text,
                    "profil": "treffer", "kontext": {"start": f["context_start"], "end": f["context_end"], "text": f["query_text"]},
                    "normen": [{"id": "norm:AO:1", "titel": "Normtitel", "evidenz": {
                        "fingerprint": "fp", "anker": ["Hinweis"], "passagen": [{"id": "p1", "norm_id": "norm:AO:1",
                            "quell_start": 0, "quell_end": 14, "text": source[:14]}]}}]}]
    return text, suggestions, index, graph


def test_import_and_plan_are_model_free():
    data = cases()
    probe.validate_cases(data, primary(data))
    assert len(probe.foci_for(data[0]["text"])) == 1


@pytest.mark.parametrize("change", [
    lambda c,p: c.pop(), lambda c,p: p["case_ids"].reverse(),
    lambda c,p: p.update(model_interface={"format": "old"}),
    lambda c,p: c[0].update(split="evaluation"),
    lambda c,p: c[0]["diagnostic_labels"].update(no_hint_norm_ids=["norm:AO:1"]),
    lambda c,p: c[0]["diagnostic_labels"].update(hint_expected=1),
    lambda c,p: c[0].update(text=""),
])
def test_plan_rejects_subsets_conflicts_and_old_interface(change):
    data = cases()
    plan = primary(data)
    change(data, plan)
    with pytest.raises(probe.ProbeError):
        probe.validate_cases(data, plan)


def test_every_planned_case_remains_and_failures_are_not_correct_negatives():
    data = cases()
    rows = [row(data[0], norms=["norm:AO:1"]), row(data[1], "technical_error"), row(data[2], "invalid_output"), row(data[3])]
    metrics = probe.summarize(data, rows)["semantic_suggestions"]["all"]
    assert metrics["planned_cases"] == 16 and metrics["not_run"] == 12
    assert metrics["positive_case_hits"] == 1 and metrics["positive_cases_unknown"] == 7
    assert metrics["non_hint_cases_unknown"] == 7 and metrics["complete_non_hint_cases_without_output"] == 1
    assert metrics["hint_recall_lower_bound"] == 1/8 and metrics["hint_recall_upper_bound"] == 1
    assert metrics["precision"] is None


def test_only_explicit_negative_ids_count_and_hard_norms_stay_separate():
    data = cases()
    rows = [row(data[0], norms=["norm:AO:999"], hard=["norm:AO:1"]),
            row(data[1], norms=["norm:AO:2", "norm:AO:999"])]
    aggregate = probe.summarize(data, rows)
    semantic, union = (aggregate[name]["all"] for name in ("semantic_suggestions", "all_visible_norms"))
    assert semantic["positive_case_hits"] == 0 and union["positive_case_hits"] == 1
    assert semantic["explicit_no_hint_output_pairs"] == 0
    assert semantic["unadjudicated_output_pairs"] == 2
    assert semantic["non_hint_cases_with_output"] == 1
    assert semantic["clear_negative_cases_with_output"] == 1
    assert semantic["clear_negative_output_norm_pairs"] == 2


def test_context_only_case_is_not_mixed_with_clear_negative_case_rates():
    data = cases()
    data[1]["diagnostic_labels"].update(allowed_hint_norm_ids=[], insufficient_context_norm_ids=["norm:AO:2"])
    rows = [row(data[1], norms=["norm:AO:999"]), row(data[3], norms=["norm:AO:999"], hard=["norm:AO:2"])]
    aggregate = probe.summarize(data, rows)
    semantic, union = (aggregate[name]["all"] for name in ("semantic_suggestions", "all_visible_norms"))
    assert semantic["planned_context_only_cases"] == 1 and semantic["context_only_cases_with_output"] == 1
    assert semantic["planned_clear_negative_cases"] == 7 and semantic["clear_negative_cases_unknown"] == 6
    assert semantic["clear_negative_output_norm_pairs"] == 1 and union["clear_negative_output_norm_pairs"] == 2
    assert semantic["explicit_no_hint_output_pairs"] == 0  # Unannotated outputs still appear in the full clear-negative burden.


def test_synthetic_full_denominator_has_three_clear_negatives_and_one_context_case():
    data = cases()
    for i, item in enumerate(data):
        positive = i not in (1, 5, 9, 13)
        item["diagnostic_labels"].update(
            hint_expected=positive, target_norm_ids=(["norm:AO:1", "norm:AO:8"] if i in (0, 2) else ["norm:AO:1"]) if positive else [],
            no_hint_norm_ids=["norm:AO:3"], allowed_hint_norm_ids=["norm:AO:2"],
            insufficient_context_norm_ids=["norm:AO:4"] if i == 5 else [])
    probe.validate_cases(data, primary(data))
    metrics = probe.summarize(data, [])["semantic_suggestions"]["all"]
    assert metrics["positive_cases"] == 12 and metrics["known_target_norm_pairs"] == 14
    assert metrics["planned_clear_negative_cases"] == 3 and metrics["clear_negative_cases_unknown"] == 3
    assert metrics["planned_context_only_cases"] == 1 and metrics["context_only_cases_unknown"] == 1


def test_timing_includes_failures_without_fake_not_run_durations():
    assert probe.timing_summary([])["median_seconds"] is None
    rows = [{"timing": {"elapsed_seconds": v}} for v in (1, 3, 5, 7)] + [{"status": "not_run"}]
    t = probe.timing_summary(rows)
    assert t["recorded_attempts"] == 4 and t["median_seconds"] == 4 and t["total_seconds"] == 16
    assert t["p95_seconds"] == pytest.approx(6.7)
    with pytest.raises(probe.ProbeError):
        probe.timing_summary([{"timing": {"elapsed_seconds": float("nan")}}])


def test_clipped_source_parent_is_allowed_and_anchor_remains_separate():
    text, suggestions, index, graph = sources()
    assert probe.source_audit(text, suggestions, index, graph)["passages"] == 1


@pytest.mark.parametrize("change", [
    lambda s: s[0].update(start=False), lambda s: s.append(deepcopy(s[0])),
    lambda s: s[0].update(satz="Anderer Vorgang."), lambda s: s[0]["kontext"].update(text="wrong"),
    lambda s: s[0]["normen"][0].update(titel="other"),
    lambda s: s[0]["normen"][0]["evidenz"].update(fingerprint="wrong"),
    lambda s: s[0]["normen"][0]["evidenz"].update(anker=["invented"]),
    lambda s: s[0]["normen"][0]["evidenz"]["passagen"][0].update(quell_start=True),
    lambda s: s[0]["normen"][0]["evidenz"]["passagen"][0].update(quell_end=999),
    lambda s: s[0]["normen"][0]["evidenz"]["passagen"][0].update(norm_id="norm:AO:2"),
    lambda s: s[0]["normen"][0]["evidenz"]["passagen"][0].update(text="Hinweis"),
])
def test_source_audit_rejects_wrong_spans_fingerprints_and_anchors(change):
    text, suggestions, index, graph = sources()
    change(suggestions)
    with pytest.raises(probe.ProbeError):
        probe.source_audit(text, suggestions, index, graph)


def fake_score(paare, instruction, max_length=2048):
    ids = [[1, 2, 3] for _ in paare]
    assert ids and instruction and max_length
    return [.6 for _ in paare]


@pytest.fixture
def eigener_tracer():
    # Diese vier Tests benötigen denselben exklusiven Trace-Hook wie die Probe.
    # Nach dem Test erhält auch der Coverage-Lauf seinen bisherigen Hook zurück.
    vorher = sys.gettrace()
    sys.settrace(None)
    try:
        yield
    finally:
        sys.settrace(vorher)


def test_readonly_capture_preserves_real_call_results_and_trace_lifecycle(eigener_tracer):
    pairs = [("Ein beschriebener Vorgang", "Normtext")]
    with probe.CallCapture({fake_score.__code__: "qwen"}) as capture:
        result = fake_score(pairs, "instruction", max_length=2048)
    assert result == [.6] and sys.gettrace() is None
    event = capture.events[0]
    assert event["pairs"] == pairs and event["scores"] == [.6] and event["token_ids"] == [[1, 2, 3]]
    assert probe.audit_capture(capture.events, probe.foci_for("Ein beschriebener Vorgang"), {"instruction": "instruction", "shortlist": 20}, complete=True)["qwen_pairs"] == 1


@pytest.mark.parametrize("change", [
    lambda e: e.update(max_length=1024), lambda e: e.update(instruction="changed"),
    lambda e: e.update(token_ids=[]), lambda e: e.update(scores=[float("nan")]),
    lambda e: e.update(scores=[1.1]), lambda e: e.update(pairs_sha256="wrong"),
    lambda e: e.update(status="error"),
])
def test_capture_rejects_changed_budgets_scores_and_missing_tokens(change, eigener_tracer):
    with probe.CallCapture({fake_score.__code__: "qwen"}) as capture:
        fake_score([("Ein beschriebener Vorgang", "Normtext")], "instruction", max_length=2048)
    change(capture.events[0])
    with pytest.raises(probe.ProbeError):
        probe.audit_capture(capture.events, probe.foci_for("Ein beschriebener Vorgang"), {"instruction": "instruction", "shortlist": 20}, complete=True)


def test_capture_records_failure_and_restores_trace(eigener_tracer):
    def fails(paare, instruction, max_length=2048):
        raise RuntimeError("failure")
    with pytest.raises(RuntimeError), probe.CallCapture({fails.__code__: "qwen"}) as capture:
        fails([("Ein beschriebener Vorgang", "Normtext")], "instruction")
    assert capture.events[0]["status"] == "error" and sys.gettrace() is None


def test_execute_receives_only_text_and_journals_all_cases(tmp_path):
    data, calls = cases(), []
    def run(text):
        calls.append(text)
        return {"status": "technical_error", "output": None, "errors": [{"type": "fake"}], "timing": {"elapsed_seconds": .1}}
    report = {"rows": []}
    probe.execute_cases(data, tmp_path, report, run)
    assert calls == [c["text"] for c in data]
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 16
    assert len(list(tmp_path.glob("case-*.json"))) == 16
    assert probe.read(tmp_path / "report.json")["aggregate"]["semantic_suggestions"]["all"]["technical_error"] == 16


def test_snapshot_and_inventory_detect_later_changes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "a.json").write_text('{"data":1}')
    bindings = {}
    probe.snapshot(source / "a.json", target / "a.json", bindings)
    hashes = probe.inventory(target)
    probe.check_inventory(target, hashes)
    probe.check_files(bindings)
    (source / "a.json").write_text('{"data":2}')
    with pytest.raises(probe.ProbeError):
        probe.check_files(bindings)
    (target / "extra").write_text("extra")
    with pytest.raises(probe.ProbeError):
        probe.check_inventory(target, hashes)


def test_type_exact_public_options_binding():
    declared = asdict(probe.options())
    assert declared["semantik_profil"] == "treffer" and declared["rechtsgebiete"] == ()
    changed = {**declared, "detektor": 1}
    assert probe.value_sha(declared) != probe.value_sha(changed)


def synthetic_gpu_reference(code, profile):
    return {"format": "quality_recall_live/1", "status": "mismatch", "rows": [{} for _ in range(162)],
            "public_profile": "treffer", "profile_config": deepcopy(profile), "source_hashes": dict(code),
            "models": {label: {"name": name, "revision": revision, "devices": ["cuda:0"], "dtypes": [dtype]}
                       for label, (_repo, revision, name, dtype) in probe.PINS.items()},
            "runtime": {"torch": probe.PACKAGES["torch"], "device": "AMD Radeon RX 9070 XT",
                        "packages": {k: v for k, v in probe.PACKAGES.items() if k != "torch"},
                        "peak_allocated_bytes": 1000, "peak_reserved_bytes": 1200, "device_total_bytes": 4096}}


def test_gpu_reference_requires_complete_route_and_bound_core():
    from risk_layer.semantic_search import PROFILE_CONFIG
    code = {name: str(i) * 64 for i, name in enumerate(("risk_layer/reranker.py", "risk_layer/semantic_search.py", "risk_layer/embedding.py"), 1)}
    reference = synthetic_gpu_reference(code, PROFILE_CONFIG["treffer"])
    basis = probe.validate_gpu_reference(reference, code, PROFILE_CONFIG["treffer"])
    assert basis["peak_reserved_bytes"] == 1200
    changed = deepcopy(reference)
    changed["models"]["qwen"]["dtypes"] = ["torch.float32"]
    with pytest.raises(probe.ProbeError):
        probe.validate_gpu_reference(changed, code, PROFILE_CONFIG["treffer"])
    for mutation in (lambda r: r["rows"].pop(), lambda r: r.update(status="failed"),
                     lambda r: r["source_hashes"].update({"risk_layer/reranker.py": "changed"})):
        changed = deepcopy(reference)
        mutation(changed)
        with pytest.raises(probe.ProbeError):
            probe.validate_gpu_reference(changed, code, PROFILE_CONFIG["treffer"])


@pytest.mark.parametrize("device,dtype,buffer", [("cpu", "torch.float16", "cuda:0"),
                                              ("cuda:0", "torch.float32", "cuda:0"),
                                              ("cuda:0", "torch.float16", "cpu")])
def test_gpu_placement_requires_parameters_and_buffers(device, dtype, buffer):
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=device, dtype=dtype)]),
                            buffers=lambda: iter([SimpleNamespace(device=buffer)]))
    wrapper = SimpleNamespace(name=probe.PINS["qwen"][2], revision=probe.PINS["qwen"][1])
    with pytest.raises(probe.ProbeError):
        probe.model_info(model, wrapper, "qwen")


def test_publication_metrics_never_emit_cases_queries_or_norm_identifiers():
    data = cases()
    aggregate = probe.summarize(data, [row(data[0], norms=["norm:AO:1"])])
    raw = json.dumps(aggregate)
    assert "Ein Vorgang" not in raw and "norm:AO:" not in raw and '"case_id"' not in raw


def test_strict_json_rejects_duplicate_keys_and_nonfinite(tmp_path):
    path = tmp_path / "bad.json"
    for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
        path.write_text(raw)
        with pytest.raises(ValueError):
            probe.read(path)
    assert probe.value_sha({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()


def fake_prepared(tmp_path):
    from risk_layer.semantic_search import PROFILE_CONFIG
    prepared, original = tmp_path / "prepared", tmp_path / "original"
    prepared.mkdir()
    (prepared / "primary").mkdir()
    original.mkdir()
    data = cases()
    probe.save(original / "cases.json", data)
    artifacts = probe.inventory(original)
    probe.save(original / "input-manifest.json", {"artifacts": artifacts})
    probe.save(prepared / "cases.json", data)
    probe.save(prepared / "primary/cases.json", data)
    (prepared / "probe-source.py").write_bytes(Path(probe.__file__).read_bytes())
    protocol = {"probe_source_sha256": probe.sha(probe.__file__),
                "case_ids": [c["case_id"] for c in data], "profile_config": PROFILE_CONFIG["treffer"],
                "options": asdict(probe.options()), "original_bindings": {}, "models": {},
                "primary_inputs": str(original), "primary_manifest_sha256": probe.sha(original / "input-manifest.json"),
                "primary_artifacts": artifacts, "hf_cache": "unused", "foci": {c["case_id"]: probe.foci_for(c["text"]) for c in data}}
    probe.save(prepared / "protocol.json", protocol)
    probe.save(prepared / "prepared-manifest.json", {"format": probe.PREPARE_FORMAT, "artifacts": probe.inventory(prepared)})
    return prepared, original, data


@pytest.mark.parametrize("failure", [False, True])
def test_run_lifecycle_preserves_all_denominators_and_checks_end_state(tmp_path, monkeypatch, failure):
    prepared, _, data = fake_prepared(tmp_path)
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: probe.PACKAGES[name])
    def run_gpu(cases, directory, out, report):
        assert cases == data and directory == prepared
        if failure:
            raise RuntimeError("Simulated setup failure")
        probe.execute_cases(cases, out, report, lambda text: {
            "status": "complete", "output": output(), "timing": {"elapsed_seconds": 1.0}})
    monkeypatch.setattr(probe, "run_gpu", run_gpu)
    out = tmp_path / "result"
    code = probe.run(SimpleNamespace(prepared=prepared, out=out))
    result = probe.read(out / "report.json")
    assert code == int(failure) and result["end_verification"]["status"] == "pass"
    assert result["aggregate"]["semantic_suggestions"]["all"]["planned_cases"] == 16
    assert len(result["rows"]) == (0 if failure else 16)
    assert result["aggregate"]["semantic_suggestions"]["all"]["non_hint_cases_unknown"] == (8 if failure else 0)


def test_final_original_change_fails_even_after_all_successful_fake_outputs(tmp_path, monkeypatch):
    prepared, original, _ = fake_prepared(tmp_path)
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: probe.PACKAGES[name])
    def run_gpu(cases, directory, out, report):
        probe.execute_cases(cases, out, report, lambda text: {"status": "complete", "output": output()})
        (original / "cases.json").write_text("changed")
    monkeypatch.setattr(probe, "run_gpu", run_gpu)
    out = tmp_path / "result"
    assert probe.run(SimpleNamespace(prepared=prepared, out=out)) == 1
    result = probe.read(out / "report.json")
    assert result["status"] == "failed" and result["end_verification"]["status"] == "failed"
    assert len(result["rows"]) == 16


@pytest.mark.parametrize("change", [lambda m: m.update(start=True), lambda m: m.update(end=999),
                                    lambda m: m.update(matched_text="invented")])
def test_hard_mark_source_binding(change):
    text = "Ein belegter Vorgang"
    mark = {"start": 0, "end": len(text), "matched_text": text}
    assert probe.hard_audit(text, [mark], [])["cards"] == 1
    change(mark)
    with pytest.raises(probe.ProbeError):
        probe.hard_audit(text, [mark], [])


def test_hard_norm_view_does_not_pull_in_invented_semantic_facts():
    cards = [{"norm_anker": ["§ 146 AO"]}]
    risks = [{"norm_vorschlag": [{"id": "norm:AO:158"}]}]
    assert set(probe.hard_norm_ids(cards, risks)) == {"norm:AO:146", "norm:AO:158"}


def make_model_cache(directory):
    for label, (repo, revision, _name, _dtype) in probe.PINS.items():
        folder = directory / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        names = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
        if label == "encoder":
            names |= {"pytorch_model.bin", "modules.json", "1_Pooling/config.json", "sentence_bert_config.json", "sentencepiece.bpe.model"}
        elif label == "bge":
            names |= {"model.safetensors", "sentencepiece.bpe.model"}
        else:
            names |= {"merges.txt", "vocab.json", "shard1.safetensors", "model.safetensors.index.json"}
        for name in names:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        if label == "qwen":
            probe.save(folder / "model.safetensors.index.json", {"weight_map": {f"tensor{i}": "shard1.safetensors" for i in range(398)}})


def test_complete_model_inventory_binds_originals_but_does_not_load_or_copy_weights(tmp_path):
    cache, out = tmp_path / "cache", tmp_path / "snapshot"
    out.mkdir()
    make_model_cache(cache)
    bindings = {}
    models = probe.freeze_models(cache, out, bindings)
    assert set(models) == {"encoder", "bge", "qwen"}
    assert "shard1.safetensors" in models["qwen"]["files"]
    assert not list(out.rglob("*.safetensors")) and not list(out.rglob("*.bin"))
    assert list(out.rglob("tokenizer.json")) and bindings
    (Path(models["qwen"]["snapshot"]) / "shard1.safetensors").unlink()
    with pytest.raises(probe.ProbeError):
        probe.freeze_models(cache, out, {})


def test_preparation_with_cpu_fakes_freezes_the_complete_plan_and_assets(tmp_path, monkeypatch):
    from risk_layer.semantic_search import PROFILE_CONFIG
    inputs, cache, out = tmp_path / "inputs", tmp_path / "cache", tmp_path / "output"
    inputs.mkdir()
    (inputs / "index").mkdir()
    (inputs / "index/meta.json").write_text("{}")
    (inputs / "index/vectors.npy").write_text("fake vectors")
    (inputs / "graph.sqlite").write_text("fake graph")
    (inputs / "trigger.sqlite").write_text("fake trigger")
    (inputs / "catalog.yaml").write_text("fake catalog")
    data = cases()
    plan = primary(data)
    plan.update(source_hashes={name: probe.sha(probe.ROOT / name) for name in (
        "risk_layer/reranker.py", "risk_layer/semantic_search.py", "risk_layer/embedding.py")},
        graph_sha256=probe.sha(inputs / "graph.sqlite"),
        index_original_hashes={str((inputs / "index" / name).resolve()): digest for name, digest in probe.inventory(inputs / "index").items()})
    probe.save(inputs / "cases.json", data)
    probe.save(inputs / "protocol.json", plan)
    probe.save(inputs / "input-manifest.json", {"format": "grounded_signal_inputs/1", "artifacts": probe.inventory(inputs)})
    make_model_cache(cache)
    reference = tmp_path / "synthetic-prior-live.json"
    probe.save(reference, synthetic_gpu_reference(plan["source_hashes"], PROFILE_CONFIG["treffer"]))
    closed = []
    graph = SimpleNamespace(fingerprint=lambda: "graphfp", close=lambda: closed.append("graph"))
    trigger = SimpleNamespace(close=lambda: closed.append("trigger"))
    index = SimpleNamespace(catalog_fingerprint="catfp", encoder_name="bge-m3", encoder_revision=probe.PINS["encoder"][1],
                            passage_evidence=SimpleNamespace(source_fingerprint="sourcefp"))
    monkeypatch.setattr(probe, "open_resources", lambda _directory: (graph, trigger, index, object()))
    probe.prepare(SimpleNamespace(inputs=inputs, out=out, graph=inputs / "graph.sqlite", trigger=inputs / "trigger.sqlite",
        catalog=inputs / "catalog.yaml", index=inputs / "index", hf_cache=cache,
        gpu_reference=reference))
    protocol = probe.read(out / "protocol.json")
    assert protocol["profile_config"] == PROFILE_CONFIG["treffer"] and len(protocol["case_ids"]) == 16
    assert len(protocol["foci"]) == 16 and closed == ["graph", "trigger"]
    probe.check_inventory(out, probe.read(out / "prepared-manifest.json")["artifacts"], ("prepared-manifest.json",))


def test_model_free_import_and_preparation_in_a_fresh_interpreter():
    # Other test modules may legitimately import torch before this module runs.
    # A fresh process tests the promise without relying on global suite order.
    selected = "import_and_plan_are_model_free or complete_model_inventory_binds_originals or preparation_with_cpu_fakes"
    code = (
        "import pytest,sys; "
        f"result=pytest.main([{str(Path(__file__).resolve())!r},'-q','-p','no:cacheprovider','-k',{selected!r}]); "
        "assert result==0; assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_gpu_control_path_with_cpu_objects_preserves_failed_sync_and_all_rows(tmp_path, monkeypatch, eigener_tracer):
    import risk_layer.embedding as embedding
    import risk_layer.pipeline as pipeline_module
    import risk_layer.reranker as reranker_module
    from risk_layer.models import RunReport
    from risk_layer.semantic_search import PROFILE_CONFIG
    data, calls, closes = cases(), [], []
    class CudaFacade:
        def __init__(self):
            self.sync_calls = 0
        def is_available(self):
            return True
        def get_device_name(self, number):
            return "AMD Radeon RX 9070 XT"
        def get_device_properties(self, number):
            return SimpleNamespace(total_memory=17095983104)
        def mem_get_info(self):
            return 17000000000, 18000000000  # Different APIs remain separately recorded.
        def synchronize(self):
            self.sync_calls += 1
            if self.sync_calls == 7:
                raise RuntimeError("simulated failed synchronization")
        def reset_peak_memory_stats(self):
            pass
        def max_memory_allocated(self):
            return 11807131136
        def max_memory_reserved(self):
            return 12335448064
    cuda = CudaFacade()
    facade = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip="fake", cuda=None), set_num_threads=lambda n: None)
    monkeypatch.setitem(sys.modules, "torch", facade)
    class Wrapper:
        def __init__(self, label):
            self.name, self.revision = probe.PINS[label][2], probe.PINS[label][1]
            self.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cuda:0", dtype=probe.PINS[label][3])]),
                                         buffers=lambda: iter([SimpleNamespace(device="cuda:0")]))
        def _laden(self):
            return SimpleNamespace(model=self.model) if self.name == probe.PINS["bge"][2] else self.model
    encoder, cascade = Wrapper("encoder"), Wrapper("qwen")
    cascade.vorauswahl = Wrapper("bge")
    monkeypatch.setattr(embedding, "BgeM3Encoder", lambda **kwargs: encoder)
    monkeypatch.setattr(reranker_module, "CascadedQwenReranker", lambda **kwargs: cascade)
    class PipelineFacade:
        def __init__(self, catalog):
            pass
        def run(self, text, **kwargs):
            calls.append((text, kwargs))
            if len(calls) == 2:
                raise RuntimeError("simulated pipeline failure")
            return RunReport(text_hash=hashlib.sha256(text.encode()).hexdigest(), katalog_version="fixture", erzeugt_am="fixture")
    monkeypatch.setattr(pipeline_module, "Pipeline", PipelineFacade)
    index = SimpleNamespace(encoder_name=encoder.name, encoder_revision=encoder.revision,
                            passage_evidence=SimpleNamespace(passages=[], anchors=[], source_fingerprint="fp"))
    graph = SimpleNamespace(close=lambda: closes.append("graph"))
    trigger = SimpleNamespace(close=lambda: closes.append("trigger"))
    monkeypatch.setattr(probe, "open_resources", lambda _: (graph, trigger, index, object()))
    report = {"rows": [], "protocol": {"earlier_gpu_memory": {"device_total_bytes": 17095983104, "peak_reserved_bytes": 12335448064},
                                        "profile_config": PROFILE_CONFIG["treffer"]}}
    probe.run_gpu(data, tmp_path, tmp_path, report)
    assert len(calls) == 17 and len(report["rows"]) == 16
    assert all(asdict(kwargs["optionen"]) == asdict(probe.options()) for _, kwargs in calls)
    assert report["rows"][0]["status"] == "technical_error"
    assert report["rows"][0]["timing"]["synchronized"] is False
    assert report["aggregate"]["timing"]["synchronization_unavailable_attempts"] == 1
    assert report["aggregate"]["semantic_suggestions"]["all"]["complete"] == 15
    assert report["runtime"]["memory_api_total_bytes"] == 18000000000
    assert closes == ["graph", "trigger"]


def hf_snapshot(tmp_path):
    repo, revision, _, _ = probe.PINS["bge"]
    root = tmp_path / ("models--" + repo.replace("/", "--"))
    directory, blobs = root / "snapshots" / revision, root / "blobs"
    directory.mkdir(parents=True)
    blobs.mkdir()
    data = b'{"fixture":true}'
    target = blobs / hashlib.sha256(data).hexdigest()
    target.write_bytes(data)
    link = directory / "config.json"
    link.symlink_to(os.path.relpath(target, directory))
    return directory, blobs, link, target, repo, revision


def bound_model(directory, repo, revision):
    files, targets = probe.model_inventory(directory, repo, revision)
    return {"snapshot": str(directory), "repo": repo, "revision": revision,
            "inventory_format": probe.MODEL_INVENTORY_FORMAT, "files": files, "file_targets": targets}


def test_legitimate_direct_hf_blob_links_are_bound_and_general_artifacts_stay_strict(tmp_path):
    directory, _, _, target, repo, revision = hf_snapshot(tmp_path)
    model = bound_model(directory, repo, revision)
    assert model["files"]["config.json"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert model["file_targets"]["config.json"]["kind"] == "hf_blob_symlink"
    probe.check_model_inventory(model)
    with pytest.raises(probe.ProbeError, match="Invalid inventory paths"):
        probe.check_inventory(directory, model["files"])


def test_regular_copied_snapshot_and_git_blob_names_are_supported(tmp_path):
    directory, blobs, link, target, repo, revision = hf_snapshot(tmp_path)
    copied = directory / "tokenizer.json"
    copied.write_text("{}")
    git_blob = blobs / ("a" * 40)
    git_blob.write_bytes(target.read_bytes())
    link.unlink()
    link.symlink_to(os.path.relpath(git_blob, directory))
    model = bound_model(directory, repo, revision)
    probe.check_model_inventory(model)
    assert model["file_targets"]["tokenizer.json"]["kind"] == "regular"
    assert model["file_targets"]["config.json"]["resolved_target"] == str(git_blob)


@pytest.mark.parametrize("destination", ["other_repo", "outside", "inside_snapshot", "nonnumeric_blob", "missing_blob"])
def test_only_direct_existing_named_blob_in_own_repository_is_allowed(tmp_path, destination):
    directory, blobs, link, target, repo, revision = hf_snapshot(tmp_path)
    paths = {"other_repo": tmp_path / "models--Other--Model/blobs" / target.name,
             "outside": tmp_path / target.name, "inside_snapshot": directory / "other.json",
             "nonnumeric_blob": blobs / "not-a-blob", "missing_blob": blobs / ("b" * 64)}
    other = paths[destination]
    other.parent.mkdir(parents=True, exist_ok=True)
    if destination != "missing_blob":
        other.write_bytes(target.read_bytes())
    link.unlink()
    link.symlink_to(os.path.relpath(other, directory))
    with pytest.raises(probe.ProbeError):
        probe.model_inventory(directory, repo, revision)


def test_blob_link_chain_is_rejected_even_when_final_target_remains_in_own_blobs(tmp_path):
    directory, blobs, link, target, repo, revision = hf_snapshot(tmp_path)
    middle = blobs / ("c" * 40)
    middle.symlink_to(target.name)
    link.unlink()
    link.symlink_to(os.path.relpath(middle, directory))
    with pytest.raises(probe.ProbeError, match="chained"):
        probe.model_inventory(directory, repo, revision)


def test_directory_links_inside_snapshot_are_rejected(tmp_path):
    directory, _, _, _, repo, revision = hf_snapshot(tmp_path)
    folder = directory / "actual"
    folder.mkdir()
    (directory / "linked").symlink_to(folder, target_is_directory=True)
    with pytest.raises(probe.ProbeError, match="directory link"):
        probe.model_inventory(directory, repo, revision)


def test_snapshot_parent_symlink_is_not_concealed_by_resolution(tmp_path):
    directory, _, _, _, repo, revision = hf_snapshot(tmp_path)
    original = directory.parent
    moved = original.with_name("real-snapshots")
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    with pytest.raises(probe.ProbeError, match="directory link"):
        probe.model_inventory(directory, repo, revision)


def test_parent_junction_is_rejected_before_hashing(tmp_path, monkeypatch):
    directory, _, _, _, repo, revision = hf_snapshot(tmp_path)
    monkeypatch.setattr(Path, "is_junction", lambda p: p == directory.parent, raising=False)
    monkeypatch.setattr(probe, "sha", lambda _: pytest.fail("Hash must not run before parent guard"))
    with pytest.raises(probe.ProbeError, match="directory link"):
        probe.model_inventory(directory, repo, revision)


@pytest.mark.parametrize("relative", ["../escape", "sub/../../escape", "sub//file", "./file", "C:/escape", "sub\\file", "/absolute"])
def test_expected_model_names_never_allow_traversal_or_nonposix_paths(tmp_path, relative):
    directory, _, _, _, repo, revision = hf_snapshot(tmp_path)
    model = bound_model(directory, repo, revision)
    model["files"][relative] = "a" * 64
    model["file_targets"][relative] = {}
    with pytest.raises(probe.ProbeError, match="relative path"):
        probe.check_model_inventory(model)


@pytest.mark.parametrize("change", ["new_blob_same_bytes", "same_target_different_link_spelling", "regular_file", "changed_bytes", "extra_file"])
def test_exact_bound_link_and_bytes_changes_are_rejected(tmp_path, change):
    directory, blobs, link, target, repo, revision = hf_snapshot(tmp_path)
    model = bound_model(directory, repo, revision)
    if change == "new_blob_same_bytes":
        other = blobs / ("d" * 40)
        other.write_bytes(target.read_bytes())
        link.unlink()
        link.symlink_to(os.path.relpath(other, directory))
    elif change == "same_target_different_link_spelling":
        link.unlink()
        link.symlink_to(target)
    elif change == "regular_file":
        data = target.read_bytes()
        link.unlink()
        link.write_bytes(data)
    elif change == "changed_bytes":
        target.write_bytes(b'{"fixture":none}')
    else:
        (directory / "extra.json").write_text("{}")
    with pytest.raises(probe.ProbeError):
        probe.check_model_inventory(model)


def test_link_switch_during_hash_is_detected_even_for_identical_bytes(tmp_path, monkeypatch):
    directory, blobs, link, target, repo, revision = hf_snapshot(tmp_path)
    other = blobs / ("e" * 40)
    other.write_bytes(target.read_bytes())
    original_sha = probe.sha
    def changing_sha(path):
        result = original_sha(path)
        link.unlink()
        link.symlink_to(os.path.relpath(other, directory))
        return result
    monkeypatch.setattr(probe, "sha", changing_sha)
    with pytest.raises(probe.ProbeError, match="changed while hashing"):
        probe.model_inventory(directory, repo, revision)


def test_preparation_rejects_bad_model_link_before_any_model_loading(tmp_path):
    cache, out = tmp_path / "cache", tmp_path / "out"
    out.mkdir()
    make_model_cache(cache)
    repo, revision, _, _ = probe.PINS["encoder"]
    link = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision / "config.json"
    (link.parent.parent.parent / "blobs").mkdir()
    other = tmp_path / "unrelated.json"
    other.write_text("{}")
    link.unlink()
    link.symlink_to(other)
    with pytest.raises(probe.ProbeError, match="own HF repository"):
        probe.freeze_models(cache, out, {})


def test_terminal_run_rechecks_the_bound_model_links(tmp_path, monkeypatch):
    prepared, _, _ = fake_prepared(tmp_path)
    directory, blobs, link, target, repo, revision = hf_snapshot(tmp_path / "model-cache")
    model = bound_model(directory, repo, revision)
    protocol = probe.read(prepared / "protocol.json")
    protocol["models"] = {"bge": model}
    probe.save(prepared / "protocol.json", protocol)
    probe.save(prepared / "prepared-manifest.json", {"format": probe.PREPARE_FORMAT,
               "artifacts": probe.inventory(prepared, ("prepared-manifest.json",))})
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda name: probe.PACKAGES[name])
    def fake_gpu(cases, directory, out, report):
        probe.execute_cases(cases, out, report, lambda text: {"status": "complete", "output": output()})
        changed = blobs / ("f" * 40)
        changed.write_bytes(target.read_bytes())
        link.unlink()
        link.symlink_to(changed)
    monkeypatch.setattr(probe, "run_gpu", fake_gpu)
    out = tmp_path / "run"
    assert probe.run(SimpleNamespace(prepared=prepared, out=out)) == 1
    report = probe.read(out / "report.json")
    assert len(report["rows"]) == 16 and report["end_verification"]["status"] == "failed"


def test_directory_enumeration_error_never_looks_like_complete_inventory(tmp_path, monkeypatch):
    directory, _, _, _, repo, revision = hf_snapshot(tmp_path)
    def inaccessible_walk(*args, **kwargs):
        kwargs["onerror"](PermissionError("unreadable subtree"))
        return iter(())
    monkeypatch.setattr(probe.os, "walk", inaccessible_walk)
    monkeypatch.setattr(probe, "sha", lambda _: pytest.fail("No hash before complete enumeration"))
    with pytest.raises(probe.ProbeError, match="enumerate the complete"):
        probe.model_inventory(directory, repo, revision)
