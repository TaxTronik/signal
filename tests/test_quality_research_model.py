"""Model-free guards for the explicitly launched research-model service."""
import copy
import hashlib
import json

import pytest

from tools import quality_research_model as setup


def tool_response():
    return {"model": setup.ALIAS, "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "reasoning_content": "I need to retrieve the record.",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {
            "name": "lookup_test_record", "arguments": json.dumps({"record_id": setup.TEST_RECORD})}}]}}]}


def test_model_bytes_and_digest_are_both_required(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-fixed-test")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    assert setup.verify_file(model, 15, expected)["sha256"] == expected
    with pytest.raises(ValueError, match="size"):
        setup.verify_file(model, 14, expected)
    with pytest.raises(ValueError, match="SHA256"):
        setup.verify_file(model, 15, "0" * 64)


def test_startup_is_local_thinking_text_only_one_gpu():
    argv = setup.server_argv("engine.exe", "weights.gguf")
    for flag, value in (("--host", "127.0.0.1"), ("--port", "8089"), ("--parallel", "1"),
                        ("--ctx-size", "16384"), ("--reasoning-budget", "768"),
                        ("--reasoning", "on"), ("--device", "Vulkan0"), ("--gpu-layers", "all"),
                        ("--cache-type-k", "f16"), ("--cache-type-v", "f16")):
        assert argv[argv.index(flag) + 1] == value
    assert "--no-mmproj" in argv and "--offline" in argv
    assert "--chat-template" not in argv


@pytest.mark.parametrize("port", [True, 80, 65536, "8089"])
def test_invalid_port_fails_before_launch(port):
    with pytest.raises(ValueError, match="Port"):
        setup.server_argv("engine.exe", "weights.gguf", port)


def test_native_tool_call_preserves_id_and_reasoning():
    response = tool_response()
    message, call_id, arguments = setup.first_tool_call(response)
    assert call_id == "call-1" and arguments == {"record_id": setup.TEST_RECORD}
    assert message is response["choices"][0]["message"]


@pytest.mark.parametrize("change", ["model", "length", "wrong_tool", "extra_argument", "no_call_id", "multiple_calls"])
def test_unusable_tool_output_is_not_accepted(change):
    response = copy.deepcopy(tool_response())
    choice = response["choices"][0]
    call = choice["message"]["tool_calls"][0]
    if change == "model":
        response["model"] = "some-other-model"
    elif change == "length":
        choice["finish_reason"] = "length"
    elif change == "wrong_tool":
        call["function"]["name"] = "execute_shell"
    elif change == "extra_argument":
        call["function"]["arguments"] = json.dumps({"record_id": setup.TEST_RECORD, "extra": "ignored?"})
    elif change == "no_call_id":
        call["id"] = ""
    else:
        choice["message"]["tool_calls"].append(copy.deepcopy(call))
    with pytest.raises(ValueError):
        setup.first_tool_call(response)


def test_tool_result_is_not_leaked_into_initial_prompt():
    payload = setup.smoke_payload()
    assert setup.TEST_CODE not in json.dumps(payload)
    assert payload["tool_choice"] == "auto"
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "response_format" not in payload


def test_final_value_must_come_from_the_tool_and_be_complete():
    response = {"model": setup.ALIAS, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": setup.TEST_CODE}}]}
    assert setup.final_answer(response)["exact_tool_result"]
    response["choices"][0]["message"]["content"] = "INVENTED-1234"
    with pytest.raises(ValueError, match="returned only by the tool"):
        setup.final_answer(response)


def test_gpu_claim_requires_matching_architecture_device_and_full_offload():
    log = "Vulkan0: AMD Radeon RX 9070 XT\narchitecture qwen35\noffloaded 33/33 layers to GPU\n"
    assert setup.gpu_log_audit(log)["offloaded_layers"] == 33
    with pytest.raises(ValueError):
        setup.gpu_log_audit(log.replace("33/33", "12/33"))
    with pytest.raises(ValueError):
        setup.gpu_log_audit(log.replace("qwen35", "granite"))
