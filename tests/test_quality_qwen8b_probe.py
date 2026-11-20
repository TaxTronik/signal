"""Tiny original CPU Qwen3 models and synthetic scores; no GPU or weight download."""
from copy import deepcopy

import numpy as np
import pytest

from tools.quality_qwen8b_probe import (
    build_outputs, checkpoint_shapes, load_streamed_model, materialize_rope, numerical_comparison,
    reduce_linear_head, retry_batches, tokenize_pairs, validate_architecture,
)


def architecture():
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "tie_word_embeddings": False,
            "attention_bias": False, "hidden_size": 4096, "intermediate_size": 12288,
            "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
            "head_dim": 128, "vocab_size": 151669, "rope_theta": 1000000,
            "rope_scaling": None, "use_sliding_window": False}


@pytest.mark.parametrize("change", [{"tie_word_embeddings": True}, {"attention_bias": True},
                                   {"num_hidden_layers": 32}, {"vocab_size": 151936},
                                   {"quantization_config": {"bits": 4}}])
def test_architecture_gate_rejects_modified_or_quantized_checkpoint(change):
    validate_architecture(architecture())
    with pytest.raises(ValueError):
        validate_architecture({**architecture(), **change})


def tiny_config():
    from risk_layer.torch_compat import patch_rocm_dist
    patch_rocm_dist()
    Qwen3Config = pytest.importorskip("transformers").Qwen3Config
    return Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
                       rope_parameters={"rope_type": "default", "rope_theta": 1000000.},
                       attention_bias=False, attention_dropout=0., tie_word_embeddings=False,
                       pad_token_id=0, use_cache=False, attn_implementation="eager")


@pytest.mark.parametrize("bias", [False, True])
def test_reduced_head_copies_exact_original_rows_bias_and_order(bias):
    torch = pytest.importorskip("torch")
    torch.manual_seed(718)
    original = torch.nn.Linear(16, 41, bias=bias, device="cpu", dtype=torch.float32)
    before = original.weight.detach().clone()
    reduced = reduce_linear_head(original, [19, 3])
    assert torch.equal(reduced.weight, original.weight[[19, 3]])
    assert torch.equal(original.weight, before)
    assert reduced.bias is None if not bias else torch.equal(reduced.bias, original.bias[[19, 3]])
    assert not any(p.requires_grad for p in reduced.parameters())
    hidden = torch.randn(2, 1, 16, device="cpu")
    assert torch.allclose(reduced(hidden), original(hidden)[..., [19, 3]], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("ids", [[3, 3], [-1, 3], [3, 41], [3], [3., 19]])
def test_reduced_head_rejects_invalid_row_selection(ids):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        reduce_linear_head(torch.nn.Linear(16, 41, device="cpu"), ids)


def test_parameter_inventory_matches_real_original_small_qwen_and_exact_large_total():
    Qwen3ForCausalLM = pytest.importorskip("transformers").Qwen3ForCausalLM
    torch = pytest.importorskip("torch")
    config = tiny_config()
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    assert checkpoint_shapes(config.to_dict()) == {name: tuple(p.shape) for name, p in model.named_parameters()}
    shapes = checkpoint_shapes(architecture())
    assert len(shapes) == 399
    assert sum(int(np.prod(shape)) * 2 for shape in shapes.values()) == 16_377_096_192


def test_meta_rope_is_recreated_in_cpu_float32_with_original_values():
    torch = pytest.importorskip("torch")
    config = tiny_config()
    Qwen3ForCausalLM = pytest.importorskip("transformers").Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    assert model.model.rotary_emb.inv_freq.is_meta
    materialize_rope(model, config, "cpu")
    expected = Qwen3RotaryEmbedding(config)
    for name in ("inv_freq", "original_inv_freq"):
        actual = getattr(model.model.rotary_emb, name)
        assert actual.device.type == "cpu" and actual.dtype == torch.float32 and not actual.is_meta
        assert torch.equal(actual, getattr(expected, name))
        assert "model.rotary_emb." + name not in model.state_dict()


def test_streamed_original_bf16_checkpoint_preserves_every_parameter_and_source_head(tmp_path):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    config = tiny_config()
    Qwen3ForCausalLM = pytest.importorskip("transformers").Qwen3ForCausalLM
    torch.manual_seed(982)
    original = Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.bfloat16)
    weights = {name: value.detach().contiguous() for name, value in original.state_dict().items()}
    save_file(weights, str(tmp_path / "tiny.safetensors"))
    model, full_head = load_streamed_model(tmp_path, config.to_dict(),
                                           dict.fromkeys(weights, "tiny.safetensors"), [3, 19], "cpu")
    for name, parameter in model.named_parameters():
        expected = weights[name].half()
        if name == "lm_head.weight":
            expected = expected[[3, 19]]
        assert parameter.device.type == "cpu" and not parameter.requires_grad
        assert torch.equal(parameter, expected), name
    assert torch.equal(full_head, weights["lm_head.weight"].half())
    assert all(not buffer.is_meta for buffer in model.buffers())
    assert all(buffer.dtype == torch.float32 for buffer in model.model.rotary_emb.buffers())


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_original_cpu_qwen_inputs_embeds_and_two_row_head_preserve_decoder_result(dtype_name):
    torch = pytest.importorskip("torch")
    config = tiny_config()
    Qwen3ForCausalLM = pytest.importorskip("transformers").Qwen3ForCausalLM
    torch.manual_seed(422)
    original = Qwen3ForCausalLM(config).to(device="cpu", dtype=getattr(torch, dtype_name)).eval()
    materialize_rope(original, config, "cpu")
    reduced = deepcopy(original)
    reduced.lm_head = reduce_linear_head(original.lm_head, [3, 19])
    ids = torch.tensor([[0, 0, 7, 5, 11], [2, 4, 6, 8, 10]], device="cpu")
    mask = ids.ne(0).long()
    with torch.inference_mode():
        expected = original(input_ids=ids, attention_mask=mask, position_ids=None,
                            past_key_values=None, use_cache=False, logits_to_keep=1)
        embedded = reduced.model.embed_tokens(ids)
        actual = reduced(inputs_embeds=embedded, attention_mask=mask, position_ids=None,
                         past_key_values=None, use_cache=False, logits_to_keep=1)
        source_hidden = original.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        split_hidden = reduced.model(inputs_embeds=embedded, attention_mask=mask, use_cache=False).last_hidden_state
    assert torch.equal(source_hidden, split_hidden)
    assert actual.logits.shape == (2, 1, 2) and actual.past_key_values is None
    atol = 1e-6 if dtype_name == "float32" else 2e-3
    assert torch.allclose(actual.logits, expected.logits[..., [3, 19]], atol=atol, rtol=atol)


def test_declared_oom_retry_covers_every_pair_once_and_does_not_skip_failed_position():
    class OOM(Exception):
        pass
    attempts, errors = [], []
    def score(start, size):
        attempts.append((start, size))
        if size == 2:
            raise OOM("synthetic memory limit")
        return start
    outputs = list(retry_batches(5, 2, score, OOM, lambda *error: errors.append(error)))
    assert attempts == [(0, 2), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    assert outputs == [(i, 1, i) for i in range(5)] and len(errors) == 1
    with pytest.raises(OOM):
        list(retry_batches(1, 1, lambda *args: (_ for _ in ()).throw(OOM()), OOM, lambda *args: None))


def test_build_outputs_keeps_raw_threshold_and_both_declared_shortlists():
    candidates = [{"id": f"norm:TEST:{i}", "document": f"source{i}", "sim": .99} for i in range(21)]
    row = {"case_id": "synthetic", "negative": True, "targets": [], "query": "Original query",
           "groups": [{"query": "Original query", "start": 0, "end": 14, "candidates": candidates}]}
    pairs = {(row["query"], c["document"]): i for i, c in enumerate(candidates)}
    scores = np.array([.4999] * 20 + [.5], dtype=np.float32)
    assert build_outputs([row], pairs, scores, 20)[0]["suggestions"] == []
    result = build_outputs([row], pairs, scores, 50)[0]
    assert result["negative"] is True and result["targets"] == []
    assert [n["id"] for n in result["suggestions"][0]["normen"]] == ["norm:TEST:20"]
    assert result["suggestions"][0]["normen"][0]["rerank"] == .5


def test_numerical_report_counts_crossings_without_omitting_the_worse_arm():
    a = np.array([[0., 0.], [0., 1.]], dtype=np.float32)
    b = np.array([[0., -.001], [0., 1.]], dtype=np.float32)
    result = numerical_comparison(a, b, np.array([.5, .8], np.float32), np.array([.4999, .8], np.float32))
    assert result["pairs"] == 2 and result["different_logit_pairs"] == result["different_scores"] == 1
    assert result["threshold_crossings"] == 1 and not result["exact_scores"]


def test_tokenization_preserves_original_binary_tokens_and_refuses_truncation():
    class Tokenizer:
        unk_token_id = 0
        pad_token_id = 0
        padding_side = "left"
        def convert_tokens_to_ids(self, word):
            return {"no": 3, "yes": 19}[word]
        def encode(self, text, **kwargs):
            if text in ("no", "yes"):
                return [self.convert_tokens_to_ids(text)]
            assert kwargs.get("add_special_tokens") is False
            assert kwargs.get("truncation") is not True
            return [1] * len(text)
    encoded, ids, info = tokenize_pairs(Tokenizer(), [{"query": "synthetic", "document": "original"}])
    assert ids == [3, 19] and len(encoded) == 1 and info["max_observed_tokens"] == len(encoded[0])
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_pairs(Tokenizer(), [{"query": "synthetic", "document": "x" * 2200}])
