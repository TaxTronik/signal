"""Die Messadapter müssen denselben Modellvertrag wie echte Encoder erfüllen."""
import numpy as np
import pytest

from risk_layer.embedding import BGE_M3_REVISION, EmbeddingIndex
from tools.gpu_pipeline_ab import CachedEncoder, CachedReranker


def test_gecachte_gpu_vektoren_erfuellen_den_index_revisionsvertrag():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = EmbeddingIndex(vectors, ["norm:AO:146", "norm:AO:162"],
                           ["§ 146 AO", "§ 162 AO"], ["Aufzeichnungen", "Schätzung"],
                           "bge-m3", encoder_revision=BGE_M3_REVISION)
    cache = CachedEncoder(["Belege fehlen"], vectors[:1], revision=BGE_M3_REVISION)
    assert index.suche("Belege fehlen", cache, k=1)[0]["norm_id"] == "norm:AO:146"


def test_encoder_cache_verbirgt_keine_abweichenden_eingabetexte():
    cache = CachedEncoder([" Text "], np.ones((1, 2)), revision=BGE_M3_REVISION)
    with pytest.raises(KeyError):
        cache.encode(["Text"])


def test_reranker_cache_bindet_das_vollstaendige_normdokument():
    cache = CachedReranker([("Sachverhalt", "Normtext A")], [0.1234567])
    assert cache.score([("Sachverhalt", "Normtext A")]) == [0.1234567]
    with pytest.raises(KeyError):
        cache.score([("Sachverhalt", "Normtext B")])
