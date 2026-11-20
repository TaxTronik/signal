"""Decoder ownership checks require no CUDA allocator assumptions or GPU."""
import gc

import pytest

from tools.quality_qwen8b_probe import confirm_decoder_released, watch_decoder_lifecycle


class TensorObject:
    pass


class Model:
    def __init__(self):
        self.embedding = TensorObject()
        self.weight = TensorObject()
        self.head = TensorObject()
        self.rope = TensorObject()

    def named_parameters(self):
        return [("model.embed_tokens.weight", self.embedding), ("model.layer.weight", self.weight),
                ("lm_head.weight", self.head)]

    def named_buffers(self):
        return [("model.rotary_emb.inv_freq", self.rope)]


def test_observation_does_not_retain_decoder_or_consult_allocator_bytes():
    model = Model()
    witness = watch_decoder_lifecycle(model)
    kept_cpu_embedding = model.embedding
    del model
    gc.collect()
    result = confirm_decoder_released(witness)
    assert kept_cpu_embedding is not None
    assert result["guard"] == "original_decoder_object_weakrefs/1"
    assert result["parameter_objects_checked"] == 2
    assert result["buffer_objects_checked"] == 1
    assert result["module_released"] and result["live_parameter_objects"] == 0


def test_live_decoder_module_is_rejected():
    model = Model()
    witness = watch_decoder_lifecycle(model)
    with pytest.raises(ValueError, match="remain alive"):
        confirm_decoder_released(witness)


@pytest.mark.parametrize("attribute", ["weight", "head", "rope"])
def test_retained_decoder_parameter_or_buffer_is_rejected_then_released(attribute):
    model = Model()
    witness = watch_decoder_lifecycle(model)
    retained = getattr(model, attribute)
    del model
    gc.collect()
    with pytest.raises(ValueError, match="remain alive"):
        confirm_decoder_released(witness)
    del retained
    gc.collect()
    assert confirm_decoder_released(witness)["module_released"]


def test_real_cpu_module_parameter_and_buffer_lifetimes():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 2, device="cpu")
    model.register_buffer("rotary", torch.ones(2, device="cpu"), persistent=False)
    witness = watch_decoder_lifecycle(model)
    retained = model.weight
    del model
    gc.collect()
    with pytest.raises(ValueError, match="parameters=.*weight"):
        confirm_decoder_released(witness)
    del retained
    gc.collect()
    result = confirm_decoder_released(witness)
    assert result["parameter_objects_checked"] == 2 and result["buffer_objects_checked"] == 1
