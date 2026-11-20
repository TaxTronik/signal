"""CPU-only guards for the separately owned larger-model service."""
import copy
import json
from types import SimpleNamespace

import pytest

from tools import quality_research_moe_serve as setup


def valid_log():
    lines = ["Vulkan0: AMD Radeon RX 9070 XT", "architecture qwen35moe",
             "offloaded 41/41 layers to GPU", "CPU_Mapped model buffer size = 7300.00 MiB",
             "Vulkan0 model buffer size = 12500.00 MiB", "Vulkan0 KV buffer size = 320.00 MiB",
             "Vulkan0 RS buffer size = 80.00 MiB", "Vulkan0 compute buffer size = 200.00 MiB"]
    lines.extend(f"tensor blk.{layer}.ffn_{part}_exps.weight (144 MiB Q4_K) buffer type overridden to CPU_REPACK"
                 for layer in range(16) for part in ("gate", "up", "down"))
    return "\n".join(lines)


def test_actual_overrides_and_buffers_are_required_not_only_layer_summary():
    value = setup.gpu_log_audit(valid_log())
    assert value["cpu_expert_tensor_overrides"] == 48
    assert value["gpu_logged_buffer_sum_mib"] == 13100
    assert "not a measured" in value["memory_scope"]


@pytest.mark.parametrize("old,new", [
    ("41/41", "40/41"), ("qwen35moe", "qwen35"), ("AMD Radeon RX 9070 XT", "other GPU"),
    ("blk.15.ffn_up_exps", "blk.16.ffn_up_exps"),
    ("overridden to CPU_REPACK", "overridden to Vulkan0"),
    ("7300.00 MiB", "1000.00 MiB"), ("320.00 MiB", "640.00 MiB"),
    ("12500.00 MiB", "14000.00 MiB"), ("Vulkan0 RS buffer", "Vulkan_Host RS buffer"),
])
def test_incompatible_memory_or_expert_placement_fails(old, new):
    with pytest.raises(ValueError):
        setup.gpu_log_audit(valid_log().replace(old, new))


def test_repeated_reservations_use_largest_per_category_without_double_counting():
    value = setup.gpu_log_audit(valid_log() + "\nVulkan0 compute buffer size = 250.00 MiB\nVulkan0 compute buffer size = 150.00 MiB")
    assert value["gpu_buffer_mib"]["compute"] == 250
    assert value["gpu_logged_buffer_sum_mib"] == 13150


def test_startup_is_explicit_local_and_only_first_sixteen_expert_layers_on_cpu():
    argv = setup.server_argv("engine.exe", "model.gguf")
    for flag, value in (("--host", "127.0.0.1"), ("--port", "8090"), ("--n-cpu-moe", "16"),
                        ("--ctx-size", "16384"), ("--parallel", "1"), ("--reasoning-budget", "768"),
                        ("--device", "Vulkan0"), ("--fit", "off"), ("--cache-type-k", "f16"),
                        ("--verbosity", "5")):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value
    assert "--cpu-moe" not in argv and "--no-mmproj" in argv and "--offline" in argv
    assert "--jinja" in argv and "--chat-template" not in argv


@pytest.mark.parametrize("port", [True, 80, 65536, "8090"])
def test_invalid_port_is_rejected(port):
    with pytest.raises(ValueError):
        setup.server_argv("engine.exe", "model.gguf", port)


def props():
    return {"model_alias": setup.ALIAS, "model_path": "model.gguf", "model_ftype": "Q4_K - Medium",
            "total_slots": 1, "default_generation_settings": {"n_ctx": 16384}}


@pytest.mark.parametrize("key,value", [("model_alias", "wrong"), ("model_path", "wrong.gguf"),
                                     ("model_ftype", "Q6_K"), ("total_slots", 2), ("total_slots", True),
                                     ("default_generation_settings", {"n_ctx": 8192})])
def test_loaded_props_bind_identity_quantization_context_and_slots(key, value):
    setup.validate_props(props(), "model.gguf", setup.ALIAS)
    changed = props()
    changed[key] = value
    with pytest.raises(ValueError):
        setup.validate_props(changed, "model.gguf", setup.ALIAS)


def test_native_tool_semantics_and_hidden_value_are_unchanged():
    payload = setup.smoke_payload()
    assert setup.TEST_CODE not in json.dumps(payload)
    assert payload["model"] == setup.ALIAS
    assert payload["reasoning_budget_tokens"] == 768
    response = {"model": setup.ALIAS, "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "reasoning_content": "Retrieve the value.",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {
            "name": "lookup_test_record", "arguments": json.dumps({"record_id": setup.TEST_RECORD})}}]}}]}
    assert setup.first_tool_call(response)[1] == "call-1"
    changed = copy.deepcopy(response)
    changed["model"] = "different-model"
    with pytest.raises(ValueError):
        setup.first_tool_call(changed)
    changed = copy.deepcopy(response)
    changed["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError):
        setup.first_tool_call(changed)


def test_bad_download_report_fails_before_model_loading(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"format": "quality_research_moe_setup/1", "status": "running"}))
    with pytest.raises(ValueError, match="completed download"):
        setup.bind_download(tmp_path, path)


def test_host_preference_needs_explicit_complete_cpu_fallback():
    log = valid_log().replace("overridden to CPU_REPACK", "overridden to Vulkan_Host")
    with pytest.raises(ValueError, match="CPU fallback"):
        setup.gpu_log_audit(log)
    summary = ("\ndone_getting_tensors: tensor 'token_embd.weight' (q4_K) (and 48 others) "
               "cannot be used with preferred buffer type Vulkan_Host, using CPU instead")
    audit = setup.gpu_log_audit(log + summary)
    assert audit["host_experts_confirmed_by_cpu_fallback"] == 48
    assert audit["preferred_expert_buffer_types"] == ["Vulkan_Host"]
    with pytest.raises(ValueError, match="CPU fallback"):
        setup.gpu_log_audit(log + summary.replace("48 others", "47 others"))
    with pytest.raises(ValueError, match="CPU fallback"):
        setup.gpu_log_audit(log + summary.replace("using CPU instead", "using Vulkan0 instead"))


@pytest.mark.parametrize("missing", ["gguf-inventory.json", "engine-inventory.json"])
def test_physically_present_but_unbound_inventories_are_rejected(tmp_path, missing):
    names = {"gguf-inventory.json", "engine-inventory.json", "model-manifest.json",
             "offload-plan.json", "original-config.json", "remote-model.json", "source.py"}
    for name in names:
        (tmp_path / name).write_text("{}")
    report = {"format": "quality_research_moe_setup/1", "status": "complete", "gpu_used": False,
              "server_started": False, "artifact_hashes": {name: setup.digest(tmp_path / name)
                                                         for name in names - {missing}}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Missing bound download inventory"):
        setup.bind_download(tmp_path, path)


def test_long_context_is_sized_with_tokenizer_without_generation(monkeypatch, tmp_path):
    calls = []
    def count(base, out, name, payload, *, reserve):
        assert reserve is False
        assert payload["tool_choice"] == "none" and payload["tools"] == []
        assert payload["max_tokens"] == 256 and payload["reasoning_budget_tokens"] == 64
        calls.append(name)
        return 50 + payload["messages"][-1]["content"].count("Record ") * 14
    monkeypatch.setattr(setup, "token_count", count)
    payload, n = setup.long_smoke_payload("http://127.0.0.1:8090", tmp_path, setup.ALIAS)
    assert 12000 <= n <= 14000 and n + payload["max_tokens"] <= 16384
    assert len(calls) <= 13


def test_long_context_sizing_cannot_silently_accept_small_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "token_count", lambda *args, **kwargs: 1)
    with pytest.raises(ValueError, match="size"):
        setup.long_smoke_payload("local", tmp_path, setup.ALIAS)


def long_response(content="LONG-CONTEXT-OK"):
    return {"model": setup.ALIAS, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content, "reasoning_content": "Measured context."}}],
        "usage": {"prompt_tokens": 13487, "completion_tokens": 73}}


def test_long_memory_and_exact_marker_are_reported_separately_without_rewriting():
    response = long_response("</think>\n\nLONG-CONTEXT-OK")
    original = copy.deepcopy(response)
    checks = setup.long_context_check(response, setup.ALIAS, 13487)
    assert checks["memory_request_complete"] is True and checks["marker_exact"] is False
    assert checks["content_diagnostic"] == "leading_closing_think_tag_in_content"
    assert response == original
    assert setup.long_context_check(long_response(), setup.ALIAS, 13487)["marker_exact"] is True
    other = setup.long_context_check(long_response("OTHER"), setup.ALIAS, 13487)
    assert other["marker_exact"] is False and other["content_diagnostic"] == "unexpected_final_content"


@pytest.mark.parametrize("change", ["model", "finish", "prompt", "completion", "truncated", "empty"])
def test_long_memory_transport_accounting_and_completion_remain_strict(change):
    response = long_response()
    if change == "model":
        response["model"] = "different-model"
    elif change == "finish":
        response["choices"][0]["finish_reason"] = "length"
    elif change == "prompt":
        response["usage"]["prompt_tokens"] -= 1
    elif change == "completion":
        response["usage"]["completion_tokens"] = 257
    elif change == "truncated":
        response["truncated"] = True
    else:
        response["choices"][0]["message"]["content"] = ""
    with pytest.raises(ValueError):
        setup.long_context_check(response, setup.ALIAS, 13487)


@pytest.mark.skipif(setup.os.name != "nt", reason="Windows counter path")
def test_windows_gpu_samples_are_own_pid_only_and_explicitly_not_peak(monkeypatch, tmp_path):
    def run(argv, **kwargs):
        assert "pid_123_" in argv[-1] and "Get-CimInstance" in argv[-1]
        return SimpleNamespace(returncode=0, stdout=json.dumps([
            {"Name": "pid_123_luid_0_phys_0", "DedicatedUsage": 12 * 1024**3,
             "SharedUsage": 123, "TotalCommitted": 12 * 1024**3 + 123}]))
    monkeypatch.setattr(setup.subprocess, "run", run)
    value = setup.process_memory(123, tmp_path, "sample")
    assert value["status"] == "observed" and value["shared_usage_bytes"] == 123
    assert "not a continuous peak" in value["scope"]


@pytest.mark.skipif(setup.os.name != "nt", reason="Windows counter path")
def test_unavailable_counters_do_not_become_zero_vram_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="[]"))
    value = setup.process_memory(123, tmp_path, "sample")
    assert value["status"] == "unavailable" and "dedicated_usage_bytes" not in value
