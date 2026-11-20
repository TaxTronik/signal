"""Pinned CPU-expert MoE server handoff and an isolated native-tool smoke test.

No production options are changed. The download target is separate, server
startup is explicit, and only a process created by this invocation is stopped.
The successful server remains alive for the calling researcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPO = "ggml-org/Qwen3.6-35B-A3B-GGUF"
REVISION = "baec3ebee244827cda0f4557eafa8b28f7545fa6"
FILENAME = "Qwen3.6-35B-A3B-Q4_K_M.gguf"
MODEL_BYTES = 20_419_565_568
MODEL_SHA = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7"
DEFAULT_MODEL_DIR = ROOT / "models/research-qwen36-35b-a3b"
DEFAULT_ENGINE = ROOT / "engine/b9986/llama-b9986-bin-win-vulkan-x64/llama-server.exe"
ALIAS = "signal-research-qwen36-35b-a3b-q4"
TEST_RECORD = "cedar-17"
TEST_CODE = "LIME-4826"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_file(path, expected_bytes=MODEL_BYTES, expected_sha=MODEL_SHA):
    path = Path(path)
    require(path.is_file(), f"Missing model: {path}")
    size = path.stat().st_size
    require(size == expected_bytes, f"Model size differs: {size} != {expected_bytes}")
    sha = digest(path)
    require(sha == expected_sha, "Model SHA256 differs from the fixed revision")
    return {"path": str(path.resolve()), "bytes": size, "sha256": sha}


def bind_download(model_dir, download_report):
    """Validate the completed CPU-only download without changing its archive."""
    model_dir = Path(model_dir).resolve()
    report_path = Path(download_report).resolve()
    raw = report_path.read_bytes()
    report = json.loads(raw)
    require(report.get("format") == "quality_research_moe_setup/1" and report.get("status") == "complete"
            and report.get("gpu_used") is False and report.get("server_started") is False,
            "Expected the completed download-only MoE report")
    required_artifacts = {"gguf-inventory.json", "engine-inventory.json", "model-manifest.json",
                          "offload-plan.json", "original-config.json", "remote-model.json", "source.py"}
    require(required_artifacts <= set(report.get("artifact_hashes", {})), "Missing bound download inventory")
    bindings = {str(report_path): hashlib.sha256(raw).hexdigest()}
    for name, expected in report["artifact_hashes"].items():
        require(name == Path(name).name and "/" not in name and "\\" not in name, "Invalid download artifact path")
        path = report_path.parent / name
        require(digest(path) == expected, "Download artifact changed")
        bindings[str(path)] = expected
    manifest_path = model_dir / "model-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    require(digest(manifest_path) == report["model_manifest_sha256"], "Model manifest differs from download")
    require(all(manifest.get(key) == value for key, value in {
        "format": "quality_research_model/1", "status": "complete", "repo": REPO,
        "revision": REVISION, "filename": FILENAME, "quantization": "Q4_K_M"}.items()), "Unexpected model provenance")
    model = model_dir / FILENAME
    verified = verify_file(model)
    require(verified == manifest["model"] == report["model"], "Downloaded model identity differs")
    inventory = json.loads((report_path.parent / "gguf-inventory.json").read_bytes())
    metadata = inventory["metadata"]
    expected = {"general.architecture": "qwen35moe", "qwen35moe.block_count": 40,
        "general.file_type": 15,
        "qwen35moe.embedding_length": 2048, "qwen35moe.full_attention_interval": 4,
        "qwen35moe.attention.head_count_kv": 2, "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256, "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8}
    require(all(metadata.get(key) == value for key, value in expected.items()), "GGUF architecture/KV assumptions differ")
    expert_names = {f"blk.{layer}.ffn_{part}_exps.weight" for layer in range(16) for part in ("gate", "up", "down")}
    tensors = {item["name"]: item for item in inventory["tensors"]}
    require(expert_names <= tensors.keys(), "Missing CPU routed-expert tensors")
    expected_cpu_bytes = sum(tensors[name]["storage_bytes_including_alignment"] for name in expert_names)
    require(expected_cpu_bytes == 7_247_757_312, "First sixteen expert layers have unexpected storage")
    bindings[str(manifest_path)] = digest(manifest_path)
    bindings[str(model)] = verified["sha256"]
    engine = json.loads((report_path.parent / "engine-inventory.json").read_bytes())
    require(all(digest(name) == value for name, value in bindings.items()), "Download inputs changed while binding")
    return model, manifest_path, {"report_sha256": bindings[str(report_path)], "input_hashes": bindings,
        "engine_inventory": engine,
        "cpu_expert_layers": 16, "cpu_expert_tensor_names": sorted(expert_names),
        "cpu_expert_tensor_storage_bytes": expected_cpu_bytes,
        "attention_kv_bytes_at_16k_f16": 335_544_320, "architecture_metadata": expected}


def server_argv(executable, model, port=8090, alias=ALIAS):
    require(isinstance(port, int) and not isinstance(port, bool) and 1024 <= port <= 65535,
            "Port must be an unprivileged TCP port")
    require(isinstance(alias, str) and re.fullmatch(r"[a-zA-Z0-9._-]{1,100}", alias), "Invalid model alias")
    return [str(Path(executable).resolve()), "-m", str(Path(model).resolve()),
            "--host", "127.0.0.1", "--port", str(port), "--alias", alias,
            "--ctx-size", "16384", "--parallel", "1", "--gpu-layers", "all",
            "--device", "Vulkan0", "--fit", "off", "--n-cpu-moe", "16", "--no-mmproj", "--offline",
            "--jinja", "--reasoning", "on", "--reasoning-format", "deepseek",
            "--reasoning-budget", "768", "--flash-attn", "on",
            "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--slots", "--metrics", "--verbosity", "5"]


def check_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", port))


def request_json(base, route, out, name, payload=None, timeout=180):
    raw_request = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if payload is not None:
        (out / f"{name}.request.json").write_bytes(raw_request)
    req = Request(base + route, data=raw_request,
                  headers={"Content-Type": "application/json"} if payload is not None else {})
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except HTTPError as error:
        raw, status = error.read(), error.code
    (out / f"{name}.response.json").write_bytes(raw)
    save(out / f"{name}.http.json", {"status": status, "elapsed_seconds": time.perf_counter() - started,
                                   "response_sha256": hashlib.sha256(raw).hexdigest()})
    require(status == 200, f"{name}: HTTP {status}; raw response preserved")
    return json.loads(raw)


def response_message(response, alias):
    require(response.get("model") == alias, "Completion model differs from requested alias")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "Expected exactly one completion")
    require(isinstance(choices[0].get("message"), dict), "Missing assistant message")
    message = choices[0]["message"]
    require(message.get("role") == "assistant", "Response role is not assistant")
    require(choices[0].get("finish_reason") != "length", "Completion was truncated")
    return message, choices[0].get("finish_reason")


def first_tool_call(response, alias=ALIAS):
    message, reason = response_message(response, alias)
    calls = message.get("tool_calls")
    require(reason == "tool_calls" and isinstance(calls, list) and len(calls) == 1,
            "Native auto tool call did not produce one complete tool_calls response")
    call = calls[0]
    require(call.get("type") == "function" and isinstance(call.get("id"), str) and call["id"],
            "Tool call has no function type or call id")
    function = call.get("function", {})
    require(function.get("name") == "lookup_test_record", "Unexpected tool name")
    arguments = json.loads(function.get("arguments", ""))
    require(arguments == {"record_id": TEST_RECORD}, "Tool arguments differ from the requested record")
    return message, call["id"], arguments


def final_answer(response, alias=ALIAS, expected=TEST_CODE):
    message, reason = response_message(response, alias)
    require(reason == "stop" and not message.get("tool_calls"), "Expected a complete final answer")
    require(isinstance(message.get("content"), str) and message["content"].strip() == expected,
            "Final answer does not exactly match the value returned only by the tool")
    return {"exact_tool_result": True, "finish_reason": reason,
            "reasoning_characters": len(message.get("reasoning_content") or "")}


def smoke_payload(alias=ALIAS):
    return {
        "model": alias, "stream": False, "temperature": 0.6, "top_p": 0.95,
        "top_k": 20, "seed": 1709, "max_tokens": 1536,
        "reasoning_budget_tokens": 768,
        "chat_template_kwargs": {"enable_thinking": True},
        "parallel_tool_calls": False, "tool_choice": "auto",
        "messages": [
            {"role": "system", "content": "You may use the supplied lookup tool. Use its returned value and do not invent record contents."},
            {"role": "user", "content": f"Use lookup_test_record to look up the current verification code for record {TEST_RECORD}. After receiving the tool result, answer with that code alone."},
        ],
        "tools": [{"type": "function", "function": {
            "name": "lookup_test_record", "description": "Look up a verification record in the local test store.",
            "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}},
                           "required": ["record_id"], "additionalProperties": False},
        }}],
    }


def token_count(base, out, name, payload, *, reserve=True):
    """Count the exact native chat prompt, including tool schemas and tags."""
    template_payload = {key: payload[key] for key in
                        ("model", "messages", "tools", "tool_choice", "parallel_tool_calls", "chat_template_kwargs")}
    formatted = request_json(base, "/apply-template", out, f"{name}-template", template_payload)
    require(isinstance(formatted.get("prompt"), str) and formatted["prompt"], "No formatted chat prompt")
    encoded = request_json(base, "/tokenize", out, f"{name}-tokens",
                           {"content": formatted["prompt"], "parse_special": True, "add_special": False})
    tokens = encoded.get("tokens")
    require(isinstance(tokens, list) and tokens and all(type(token) is int and token >= 0 for token in tokens),
            "Tokenizer did not return token IDs")
    require(not reserve or len(tokens) + payload["max_tokens"] <= 16384, "Smoke request exceeds context reservation")
    return len(tokens)


def long_smoke_payload(base, out, alias):
    """Mechanically size synthetic input; no model-quality selection is involved."""
    payload = smoke_payload(alias)
    payload.update(tools=[], tool_choice="none", max_tokens=256, reasoning_budget_tokens=64)
    low, high = 1, 4096
    for attempt in range(13):
        count = (low + high) // 2
        filler = "\n".join(f"Record {i:05d}: copper cedar violet river stone." for i in range(count))
        payload["messages"] = [
            {"role": "system", "content": "This is a context-memory smoke test with inert synthetic records. Reply with LONG-CONTEXT-OK alone."},
            {"role": "user", "content": filler + "\nReturn the exact marker LONG-CONTEXT-OK now."}]
        tokens = token_count(base, out, f"long-sizing-{attempt}", payload, reserve=False)
        if 12000 <= tokens <= 14000:
            require(tokens + payload["max_tokens"] <= 16384, "Long smoke exceeds context reservation")
            return payload, tokens
        if tokens < 12000:
            low = count + 1
        else:
            high = count - 1
        require(low <= high, "Could not mechanically size the long context")
    raise ValueError("Could not size a 12k-14k synthetic prompt in thirteen tokenizations")


def long_context_check(response, alias, expected_tokens):
    """A memory workload is distinct from exact tool-result instruction following."""
    message, reason = response_message(response, alias)
    require(reason == "stop" and not message.get("tool_calls"), "Long memory request did not complete normally")
    usage = response.get("usage", {})
    require(type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] == expected_tokens
            and type(usage.get("completion_tokens")) is int and 0 < usage["completion_tokens"] <= 256
            and response.get("truncated") is not True, "Long prompt/output-token accounting differs")
    content = message.get("content")
    require(isinstance(content, str) and content.strip(), "Long memory request returned no final content")
    exact = content.strip() == "LONG-CONTEXT-OK"
    diagnostic = None if exact else (
        "leading_closing_think_tag_in_content" if content.strip() == "</think>\n\nLONG-CONTEXT-OK"
        else "unexpected_final_content")
    return {"memory_request_complete": True, "marker_exact": exact, "content_diagnostic": diagnostic,
        "finish_reason": reason, "reasoning_characters": len(message.get("reasoning_content") or ""),
        "scope": "Synthetic memory workload completed; marker formatting is recorded separately, never normalized silently."}


def process_memory(pid, out, name):
    """Read only this owned process's Windows GPU counters, when available."""
    require(type(pid) is int and pid > 0, "Invalid owned process id")
    if os.name != "nt":
        value = {"status": "unavailable", "reason": "Windows GPU counters require Windows", "pid": pid}
    else:
        command = ("$ErrorActionPreference='Stop'; "
            "@(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
            f"-Filter \"Name LIKE 'pid_{pid}_%'\" | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted) "
            "| ConvertTo-Json -Compress -Depth 4")
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, timeout=15, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
            require(result.returncode == 0, "GPU counter query was unavailable")
            rows = json.loads(result.stdout or "[]")
            rows = rows if isinstance(rows, list) else [rows]
            require(rows and all(re.match(rf"^pid_{pid}_", row.get("Name", "")) and all(
                type(row.get(key)) is int and row[key] >= 0 for key in ("DedicatedUsage", "SharedUsage", "TotalCommitted"))
                for row in rows), "GPU process counters are absent or malformed")
            value = {"status": "observed", "pid": pid, "at": now(), "rows": rows,
                "dedicated_usage_bytes": sum(row["DedicatedUsage"] for row in rows),
                "shared_usage_bytes": sum(row["SharedUsage"] for row in rows),
                "scope": "Point sample for this process across reported adapters; not a continuous peak measurement."}
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            value = {"status": "unavailable", "pid": pid, "at": now(), "reason": str(error)}
    save(out / f"{name}.json", value)
    if value["status"] == "observed":
        require(value["dedicated_usage_bytes"] <= 14 * 1024 ** 3, "Observed owned-process GPU dedication exceeds 14 GiB")
    return value


def gpu_log_audit(log):
    """Audit actual backend placement; logged buffer totals are not driver VRAM."""
    require("AMD Radeon RX 9070 XT" in log and "Vulkan0" in log, "Expected Vulkan device is not confirmed")
    require("qwen35moe" in log.lower(), "Loaded architecture is not Qwen3.5 MoE")
    offloads = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    require(offloads and tuple(map(int, offloads[-1])) == (41, 41), "Expected 41 GPU layer placements, with explicit CPU tensor overrides")
    overrides = re.findall(r"tensor (blk\.\d+\.ffn_(?:gate|up|down)_exps\.weight) .*?buffer type overridden to (\S+)", log)
    expected = {f"blk.{layer}.ffn_{part}_exps.weight" for layer in range(16) for part in ("gate", "up", "down")}
    actual = {name for name, _ in overrides}
    require(actual == expected and len(overrides) == len(expected)
            and all(device in ("CPU", "CPU_REPACK", "CPU_Mapped", "Vulkan_Host") for _, device in overrides),
            "Actual routed-expert CPU overrides differ from the first sixteen layers")
    # b9986 first chooses a host-visible preferred buffer for CPU overrides.
    # Its subsequent compatibility check moves token_embd plus those exact
    # expert tensors to CPU. A host-buffer name alone cannot prove this move.
    host_count = sum(device == "Vulkan_Host" for _, device in overrides)
    fallback = re.findall(r"done_getting_tensors: tensor 'token_embd\.weight' \(q4_K\) \(and (\d+) others\) "
                          r"cannot be used with preferred buffer type Vulkan_Host, using CPU instead", log)
    require(not host_count or len(fallback) == 1 and int(fallback[0]) == host_count,
            "Host-visible expert buffers lack the exact token-embedding plus expert CPU fallback")
    gpu = {}
    for kind, value in re.findall(r"Vulkan0\s+(model|KV|RS|compute) buffer size\s*=\s*([0-9.]+) MiB", log):
        gpu[kind] = max(gpu.get(kind, 0.0), float(value))
    require(set(gpu) == {"model", "KV", "RS", "compute"}, "Incomplete actual GPU buffer inventory")
    require(abs(gpu["KV"] - 320.0) <= .05, "Actual attention KV allocation differs from 16k/f16 plan")
    total = sum(gpu.values())
    require(0 < total <= 14 * 1024, "Logged GPU buffers exceed the fixed 14 GiB handoff limit")
    cpu = [float(value) for value in re.findall(r"CPU(?:_Mapped|_REPACK)?\s+model buffer size\s*=\s*([0-9.]+) MiB", log)]
    require(cpu and sum(cpu) >= 6912.0 - .1, "CPU model buffers do not contain the routed expert storage")
    patterns = ("Vulkan0", "offloaded", "buffer size", "n_ctx", "flash_attn", "type_k", "type_v", "build:", "buffer type overridden")
    return {"offloaded_layers": 41, "total_layers": 41, "cpu_expert_layers": 16,
        "cpu_expert_tensor_overrides": len(actual), "device": "AMD Radeon RX 9070 XT", "backend": "Vulkan0",
        "preferred_expert_buffer_types": sorted({device for _, device in overrides}),
        "host_experts_confirmed_by_cpu_fallback": host_count,
        "cpu_fallback_summary": "token_embd plus exactly the host-preferred expert tensors" if host_count else None,
        "gpu_buffer_mib": gpu, "gpu_logged_buffer_sum_mib": total, "cpu_model_buffer_mib": cpu,
        "memory_scope": "Largest logged model/KV/recurrent/compute buffers, not a measured process or driver VRAM peak.",
        "memory_limit_gib": 14, "memory_and_configuration_log_lines": [line for line in log.splitlines() if any(p in line for p in patterns)]}


def validate_props(props, model, alias):
    require(props.get("model_alias") == alias and Path(props.get("model_path", "")).resolve() == Path(model).resolve()
            and props.get("model_ftype") == "Q4_K - Medium" and type(props.get("total_slots")) is int
            and props["total_slots"] == 1
            and type(props.get("default_generation_settings", {}).get("n_ctx")) is int
            and props["default_generation_settings"]["n_ctx"] == 16384,
            "Loaded server model/quantization/context/slots differ")


def launch_and_smoke(model, model_manifest, executable, out, download_binding, port=8090, alias=ALIAS):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    executable = Path(executable).resolve()
    require(executable.is_file(), "Missing local llama-server executable")
    argv = server_argv(executable, model, port, alias)
    source_sha = digest(__file__)
    shutil.copy2(__file__, out / "source.py")
    shutil.copy2(model_manifest, out / "model-manifest.json")
    save(out / "download-binding.json", download_binding)
    model_verified = verify_file(model)
    engine_files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                    for p in sorted(executable.parent.iterdir()) if p.suffix.lower() in (".dll", ".exe")}
    require(Path(download_binding["engine_inventory"]["path"]).resolve() == executable
            and download_binding["engine_inventory"]["files"] == engine_files,
            "Engine differs from the pinned download-time inventory")
    base = f"http://127.0.0.1:{port}"
    manifest = {
        "format": "quality_research_launch/1", "status": "starting", "started_at": now(),
        "argv": argv, "working_directory": str(ROOT), "base_url": base, "alias": alias,
        "executable": str(executable), "model": model_verified,
        "model_manifest_sha256": digest(out / "model-manifest.json"),
        "engine_files": engine_files, "source_sha256": source_sha,
        "log_path": str(out / "server.log"), "pid": None,
        "download_binding_sha256": digest(out / "download-binding.json"),
        "quantization_identity": {"gguf_file_type": 15, "canonical": "Q4_K_M", "server_display": "Q4_K - Medium"},
        "limits": {"context_tokens": 16384, "slots": 1, "reasoning_budget": 768,
                   "smoke_request_max_tokens": 1536, "kv_dtype": "f16", "text_only": True,
                   "cpu_expert_layers": 16, "gpu_logged_buffer_limit_gib": 14},
        "note": "Own hidden local server. The successful process is handed to the researcher. No existing service is reused or stopped.",
    }
    save(out / "launch-manifest.json", manifest)
    process = None
    report = {"format": "quality_research_setup/1", "status": "running", "base_url": base,
              "alias": alias, "started_at": now()}
    try:
        check_port_free(port)
        with (out / "server.log").open("wb", buffering=0) as log_stream:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log_stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        manifest.update(pid=process.pid, status="launched", process_started_at=now())
        save(out / "launch-manifest.json", manifest)
        print(f"Own hidden llama-server started, PID {process.pid}, waiting for health.", flush=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            require(process.poll() is None, f"Owned server exited with code {process.returncode}")
            try:
                with urlopen(base + "/health", timeout=3) as response:
                    health = json.loads(response.read())
                if health.get("status") == "ok":
                    break
            except (HTTPError, URLError, TimeoutError, OSError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError("Owned server did not become healthy within 300 seconds")
        models = request_json(base, "/v1/models", out, "models")
        require(alias in [entry.get("id") for entry in models.get("data", [])], "Loaded API model alias differs")
        props = request_json(base, "/props", out, "props")
        validate_props(props, model, alias)
        gpu_log_audit((out / "server.log").read_text(encoding="utf-8", errors="replace"))
        memory_samples = [process_memory(process.pid, out, "memory-before-short-smoke")]
        payload = smoke_payload(alias)
        initial_tokens = token_count(base, out, "initial", payload)
        initial = request_json(base, "/v1/chat/completions", out, "tool-call", payload)
        message, call_id, arguments = first_tool_call(initial, alias)
        require(initial.get("usage", {}).get("prompt_tokens") == initial_tokens, "Initial prompt-token count differs")
        tool_result = {"record_id": arguments["record_id"], "verification_code": TEST_CODE}
        save(out / "tool-result.json", tool_result)
        payload["messages"].extend([message, {"role": "tool", "tool_call_id": call_id,
                                               "content": json.dumps(tool_result)}])
        payload["tool_choice"] = "none"
        final_tokens = token_count(base, out, "final", payload)
        result = request_json(base, "/v1/chat/completions", out, "final", payload)
        checks = final_answer(result, alias)
        require(result.get("usage", {}).get("prompt_tokens") == final_tokens, "Final prompt-token count differs")
        checks.update(initial_prompt_tokens=initial_tokens, final_prompt_tokens=final_tokens,
                      response_format="omitted_native_tool_parser", tool_choice_initial="auto", tool_choice_final="none")
        checks["initial_reasoning_characters"] = len(message.get("reasoning_content") or "")
        require(checks["reasoning_characters"] + checks["initial_reasoning_characters"] > 0,
                "Thinking was requested but no parsed reasoning was observed")
        memory_samples.append(process_memory(process.pid, out, "memory-after-short-smoke"))
        long_payload, long_tokens = long_smoke_payload(base, out, alias)
        memory_samples.append(process_memory(process.pid, out, "memory-before-long-smoke"))
        long_response = request_json(base, "/v1/chat/completions", out, "long-context", long_payload, timeout=600)
        memory_samples.append(process_memory(process.pid, out, "memory-after-long-smoke"))
        long_check = long_context_check(long_response, alias, long_tokens)
        checks["long_context"] = {**long_check, "prompt_tokens": long_tokens, "max_output_tokens": 256,
            "reasoning_budget_tokens": 64, "socket_timeout_seconds": 600,
            "scope": "Synthetic memory test only; no model-quality case."}
        health = request_json(base, "/health", out, "health-final")
        require(health.get("status") == "ok" and process.poll() is None, "Server is unhealthy after smoke test")
        log_audit = gpu_log_audit((out / "server.log").read_text(encoding="utf-8", errors="replace"))
        require(digest(__file__) == source_sha, "Setup source changed during the run")
        require(all(digest(name) == value for name, value in download_binding["input_hashes"].items()),
                "Download model/artifact changed during setup")
        require(all(digest(executable.parent / name) == item["sha256"] for name, item in engine_files.items()),
                "Engine file changed during setup")
        report.update(status="pass", pid=process.pid, checks=checks, gpu=log_audit, gpu_memory_samples=memory_samples,
                      handoff="healthy_owned_server_left_running", completed_at=now())
        manifest.update(status="healthy_handoff", handoff_at=now())
        save(out / "launch-manifest.json", manifest)
        report["launch_manifest_sha256"] = digest(out / "launch-manifest.json")
    except Exception as error:
        report.update(status="failed", error=repr(error), completed_at=now())
        if process is not None and process.poll() is None:
            process.terminate()  # Only the Popen object created by this invocation.
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
            report["own_server_stopped"] = True
        manifest.update(status="failed", error=repr(error))
        save(out / "launch-manifest.json", manifest)
    if (out / "server.log").is_file():
        shutil.copyfile(out / "server.log", out / "server-setup.log")
    report["artifact_hashes"] = {p.name: digest(p) for p in out.iterdir()
                                 if p.is_file() and p.name not in ("report.json", "server.log")}
    report["server_log_hash_scope"] = "Immutable server-setup.log snapshot at setup completion; live server.log is excluded because it may grow."
    save(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "setup"))
    parser.add_argument("--download-report", type=Path, default=ROOT / "out/quality-research-moe-model-20260907/report.json")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--alias", default=ALIAS)
    parser.add_argument("--out", type=Path, default=ROOT / "out/quality-research-moe-server-20260907")
    args = parser.parse_args()
    model, manifest, binding = bind_download(args.model_dir, args.download_report)
    print(f"Verified model: {model}; manifest: {manifest}", flush=True)
    if args.command == "preflight":
        args.out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(__file__, args.out / "source.py")
        save(args.out / "report.json", {"format": "quality_research_moe_launch_preflight/1", "status": "pass",
            "source_sha256": digest(__file__), "binding": binding, "gpu_used": False, "server_started": False,
            "planned_argv": server_argv(args.engine, model, args.port, args.alias)})
        return 0
    report = launch_and_smoke(model, manifest, args.engine, args.out, binding, args.port, args.alias)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
