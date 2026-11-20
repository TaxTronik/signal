"""Pinned local research-model setup and an isolated native-tool smoke test.

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
REPO = "unsloth/Qwen3.5-9B-GGUF"
REVISION = "3885219b6810b007914f3a7950a8d1b469d598a5"
FILENAME = "Qwen3.5-9B-Q6_K.gguf"
MODEL_BYTES = 7_458_301_152
MODEL_SHA = "91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59"
DEFAULT_MODEL_DIR = ROOT / "models/research-qwen35-9b"
DEFAULT_ENGINE = ROOT / "engine/b9986/llama-b9986-bin-win-vulkan-x64/llama-server.exe"
ALIAS = "signal-research-qwen35-9b-q6"
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


def download(model_dir):
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / FILENAME
    existed = target.exists()
    started = now()
    if existed:
        verified = verify_file(target)  # Refuse to replace an unexpected existing file.
    else:
        from huggingface_hub import hf_hub_download
        print("Downloading the pinned 7.458 GB GGUF; no model is loaded.", flush=True)
        fetched = Path(hf_hub_download(REPO, FILENAME, revision=REVISION, local_dir=model_dir))
        require(fetched.resolve() == target, "Download returned an unexpected destination")
        verified = verify_file(target)
    manifest = {
        "format": "quality_research_model/1", "status": "complete", "repo": REPO,
        "revision": REVISION, "filename": FILENAME, "quantization": "Q6_K",
        "model": verified, "already_present_and_verified": existed,
        "started_at": started, "verified_at": now(), "gpu_used": False,
        "source_url": f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}",
        "producer": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
    }
    manifest_path = model_dir / "model-manifest.json"
    if manifest_path.exists():
        original = json.loads(manifest_path.read_bytes())
        require(all(original.get(key) == manifest[key] for key in
                    ("format", "status", "repo", "revision", "filename", "quantization", "model")),
                "Existing model provenance differs; it is not overwritten")
    else:
        save(manifest_path, manifest)
    return target, manifest_path


def server_argv(executable, model, port=8089, alias=ALIAS):
    require(isinstance(port, int) and not isinstance(port, bool) and 1024 <= port <= 65535,
            "Port must be an unprivileged TCP port")
    require(isinstance(alias, str) and re.fullmatch(r"[a-zA-Z0-9._-]{1,100}", alias), "Invalid model alias")
    return [str(Path(executable).resolve()), "-m", str(Path(model).resolve()),
            "--host", "127.0.0.1", "--port", str(port), "--alias", alias,
            "--ctx-size", "16384", "--parallel", "1", "--gpu-layers", "all",
            "--device", "Vulkan0", "--fit", "off", "--no-mmproj", "--offline",
            "--jinja", "--reasoning", "on", "--reasoning-format", "deepseek",
            "--reasoning-budget", "768", "--flash-attn", "on",
            "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--slots", "--metrics", "--verbosity", "4"]


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


def token_count(base, out, name, payload):
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
    require(len(tokens) + payload["max_tokens"] <= 16384, "Smoke request exceeds context reservation")
    return len(tokens)


def gpu_log_audit(log):
    offloads = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    require(offloads and offloads[-1][0] == offloads[-1][1], "Full GPU-layer offload is not confirmed")
    require("AMD Radeon RX 9070 XT" in log and "Vulkan0" in log, "Expected Vulkan device is not confirmed")
    require("qwen35" in log.lower(), "Loaded architecture is not Qwen3.5")
    patterns = ("Vulkan0", "offloaded", "buffer size", "n_ctx", "flash_attn", "type_k", "type_v", "build:")
    return {"offloaded_layers": int(offloads[-1][0]), "total_layers": int(offloads[-1][1]),
            "device": "AMD Radeon RX 9070 XT", "backend": "Vulkan0",
            "memory_and_configuration_log_lines": [line for line in log.splitlines() if any(p in line for p in patterns)]}


def launch_and_smoke(model, model_manifest, executable, out, port=8089, alias=ALIAS):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    executable = Path(executable).resolve()
    require(executable.is_file(), "Missing local llama-server executable")
    argv = server_argv(executable, model, port, alias)
    source_sha = digest(__file__)
    shutil.copy2(__file__, out / "source.py")
    shutil.copy2(model_manifest, out / "model-manifest.json")
    model_verified = verify_file(model)
    engine_files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                    for p in sorted(executable.parent.iterdir()) if p.suffix.lower() in (".dll", ".exe")}
    base = f"http://127.0.0.1:{port}"
    manifest = {
        "format": "quality_research_launch/1", "status": "starting", "started_at": now(),
        "argv": argv, "working_directory": str(ROOT), "base_url": base, "alias": alias,
        "executable": str(executable), "model": model_verified,
        "model_manifest_sha256": digest(out / "model-manifest.json"),
        "engine_files": engine_files, "source_sha256": source_sha,
        "log_path": str(out / "server.log"), "pid": None,
        "limits": {"context_tokens": 16384, "slots": 1, "reasoning_budget": 768,
                   "smoke_request_max_tokens": 1536, "kv_dtype": "f16", "text_only": True},
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
        request_json(base, "/props", out, "props")
        payload = smoke_payload(alias)
        initial_tokens = token_count(base, out, "initial", payload)
        initial = request_json(base, "/v1/chat/completions", out, "tool-call", payload)
        message, call_id, arguments = first_tool_call(initial, alias)
        tool_result = {"record_id": arguments["record_id"], "verification_code": TEST_CODE}
        save(out / "tool-result.json", tool_result)
        payload["messages"].extend([message, {"role": "tool", "tool_call_id": call_id,
                                               "content": json.dumps(tool_result)}])
        payload["tool_choice"] = "none"
        final_tokens = token_count(base, out, "final", payload)
        result = request_json(base, "/v1/chat/completions", out, "final", payload)
        checks = final_answer(result, alias)
        checks.update(initial_prompt_tokens=initial_tokens, final_prompt_tokens=final_tokens,
                      response_format="omitted_native_tool_parser", tool_choice_initial="auto", tool_choice_final="none")
        checks["initial_reasoning_characters"] = len(message.get("reasoning_content") or "")
        require(checks["reasoning_characters"] + checks["initial_reasoning_characters"] > 0,
                "Thinking was requested but no parsed reasoning was observed")
        health = request_json(base, "/health", out, "health-final")
        require(health.get("status") == "ok" and process.poll() is None, "Server is unhealthy after smoke test")
        log_audit = gpu_log_audit((out / "server.log").read_text(encoding="utf-8", errors="replace"))
        require(digest(__file__) == source_sha, "Setup source changed during the run")
        require(all(digest(executable.parent / name) == item["sha256"] for name, item in engine_files.items()),
                "Engine file changed during setup")
        report.update(status="pass", pid=process.pid, checks=checks, gpu=log_audit,
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
    parser.add_argument("command", choices=("download", "setup"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--alias", default=ALIAS)
    parser.add_argument("--out", type=Path, default=ROOT / "out/quality-research-model-20260907")
    args = parser.parse_args()
    model, manifest = download(args.model_dir)
    print(f"Verified model: {model}; manifest: {manifest}", flush=True)
    if args.command == "download":
        return 0
    report = launch_and_smoke(model, manifest, args.engine, args.out, args.port, args.alias)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
