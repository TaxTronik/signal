"""Model-free integrity/lifecycle tests for local CPU hybrid research."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from risk_layer.embedding import HashEncoder
from risk_layer.embedding_model import BGE_M3_REQUIRED_FILES, BGE_M3_REVISION, write_model_manifest
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.passages import Passage, PassageEvidence
from risk_layer import research_hybrid as hybrid


def passage(text, nid, start=0):
    return Passage(f"{nid}@{start}:{start+len(text)}", nid, nid.split(":")[1], nid.split(":")[2],
                   "", start, start + len(text), text)


def simple_evidence(passages, scores):
    return SimpleNamespace(passages=tuple(passages), vectors=np.array([[s, 0.] for s in scores], dtype=np.float32), dim=2)


def test_rrf_retains_lexical_and_dense_sources_with_original_late_offsets():
    first = passage("Eine allgemeine Regel ohne Suchbegriff.", "norm:AO:1")
    tail = passage("Späte Aufzeichnungspflicht für die Buchführung.", "norm:AO:1", start=4500)
    semantic = passage("Eine wirtschaftliche Dokumentation ist erforderlich.", "norm:HGB:2")
    unrelated = passage("Eine andere allgemeine Bestimmung.", "norm:AO:3")
    evidence = simple_evidence([first, tail, semantic, unrelated], [.2, .1, .9, .5])
    evidence.anchors = (("norm:AO:3", "Buchführung Buchführung Buchführung"),)
    evidence.anchor_vectors = np.array([[100., 0.]], dtype=np.float32)
    ranked = hybrid.rank_passages(evidence, "Buchführung", np.array([1., 0.]), 3)
    assert ranked[0] == tail and semantic in ranked
    assert len({p.norm_id for p in ranked}) == 3
    assert ranked[0].start == 4500 and ranked[0].text == tail.text
    evidence.anchors = ()
    assert hybrid.rank_passages(evidence, "Buchführung", np.array([1., 0.]), 3) == ranked


def test_dense_ties_are_norm_id_order_and_overlapping_passages_do_not_duplicate_norms():
    rows = [passage("Unbekannte Quelle.", "norm:AO:2", 100),
            passage("Andere Quelle.", "norm:AO:1", 40), passage("Erste Quelle.", "norm:AO:1", 0)]
    evidence = simple_evidence(rows, [.3, .3, .3])
    ranked = hybrid.rank_passages(evidence, "Semantische Hypothese", np.array([1., 0.]), 6)
    assert ranked == [rows[2], rows[0]]


def test_dense_has_no_similarity_cutoff_and_empty_corpus_is_empty():
    item = passage("Originalnorm ohne Worttreffer.", "norm:AO:1")
    assert hybrid.rank_passages(simple_evidence([item], [-.4]), "Andere Suchidee", np.array([1., 0.])) == [item]
    evidence = SimpleNamespace(passages=(), vectors=np.zeros((0, 2), dtype=np.float32), dim=2)
    assert hybrid.rank_passages(evidence, "Suchidee", np.array([1., 0.])) == []


@pytest.mark.parametrize("query,limit", [("", 6), (" \n", 6), (None, 6), ("x", 0), ("x", 7), ("x", True)])
def test_invalid_query_or_limit_is_rejected(query, limit):
    with pytest.raises(ValueError):
        hybrid.rank_passages(simple_evidence([], []), query, np.array([1., 0.]), limit)


@pytest.mark.parametrize("vector", [np.array([float('nan'), 0.]), np.array([1., 0., 0.])])
def test_invalid_vector_is_rejected(vector):
    with pytest.raises(ValueError, match="query vector"):
        hybrid.rank_passages(simple_evidence([], []), "Suchidee", vector)


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    expected = {name: hashlib.sha256(name.encode()).hexdigest() for name in hybrid.DENSE_FILE_HASHES}
    monkeypatch.setattr(hybrid, "DENSE_FILE_HASHES", expected)
    root = tmp_path / "models--BAAI--bge-m3" / "snapshots" / BGE_M3_REVISION
    for name in expected:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    return root


def test_default_cache_only_resolves_existing_exact_revision_without_hub_calls(model_files, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(model_files.parents[2]))
    path, identity = hybrid.validate_model_path()
    assert path == model_files.resolve() and identity['revision'] == BGE_M3_REVISION
    assert identity['weights_sha256'] == hybrid.DENSE_FILE_HASHES['pytorch_model.bin']
    assert identity['local_files_only'] and identity['device'] == 'cpu'


def test_changed_weights_and_alternative_weights_fail_closed(model_files):
    weight = model_files / 'pytorch_model.bin'
    weight.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='differ'):
        hybrid.validate_model_path(model_files)
    weight.write_bytes(b'pytorch_model.bin')
    (model_files / 'model.safetensors').write_bytes(b'alternative weights with loader precedence')
    with pytest.raises(ValueError, match='inventory'):
        hybrid.validate_model_path(model_files)


def test_explicit_release_uses_existing_full_manifest_validation(tmp_path, model_files, monkeypatch):
    root = tmp_path / 'release'
    for name in BGE_M3_REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    write_model_manifest(root)
    called, real = [], hybrid.validate_local_bge_m3

    def check(path, **kwargs):
        called.append((path, kwargs))
        return real(path, **kwargs)

    monkeypatch.setattr(hybrid, 'validate_local_bge_m3', check)
    assert hybrid.validate_model_path(root)[1]['release_manifest'] is True
    assert called == [(root, {'verify_hashes': True})]
    (root / 'sparse_linear.pt').write_bytes(b'broken unused but manifest-bound head')
    with pytest.raises(ValueError):
        hybrid.validate_model_path(root)


@pytest.fixture
def evidence(tmp_path):
    nodes = [NormKnoten(id='norm:AO:1', law='AO', para='1', titel='',
                       text='Eine Aufzeichnungspflicht betrifft die Buchführung. ' * 70),
             NormKnoten(id='norm:HGB:2', law='HGB', para='2', titel='', text='Die kaufmännische Dokumentation ist erforderlich.'),
             NormKnoten(id='norm:AO:3', law='AO', para='3', titel='', text='(weggefallen)')]
    graph = NormGraph.build(nodes, tmp_path / 'graph.sqlite')
    encoder = HashEncoder(dim=1024)
    encoder.name, encoder.revision = 'bge-m3', BGE_M3_REVISION
    try:
        return PassageEvidence.build(graph, encoder)
    finally:
        graph.close()


@pytest.fixture
def fake_loader(monkeypatch):
    state = SimpleNamespace(loads=0, encodes=[], max_tokens=8192, fail=False)
    model = SimpleNamespace(tokenizer=lambda text, **kwargs: {'input_ids': list(range(len(text)))}, max_seq_length=8192)

    class Encoder:
        dim = 1024

        def _laden(self):
            return model

        def encode(self, texts):
            state.encodes.append(texts)
            if state.fail:
                raise RuntimeError('encoding failed')
            vector = np.zeros((len(texts), 1024), dtype=np.float32)
            vector[:, 0] = 1
            state.last_matrix = vector
            return vector

    def load(path):
        state.loads += 1
        return Encoder()

    state.model = model
    monkeypatch.setattr(hybrid, '_load_cpu_encoder', load)
    return state


def test_persistent_encoder_and_separate_timings_do_not_mutate_evidence(evidence, model_files, fake_loader):
    retriever = hybrid.HybridResearchRetriever(evidence, model_files)
    assert retriever.model_path == model_files.resolve()
    with pytest.raises(AttributeError):
        retriever.model_path = model_files
    original = evidence.passages
    assert fake_loader.loads == 0
    first = retriever.search('Buchführung', 2)
    assert retriever.search('Buchführung', 2) == first
    assert fake_loader.loads == 1 and len(fake_loader.encodes) == 2
    assert evidence.passages == original and all(p in original for p in first)
    assert all(p.norm_id != 'norm:AO:3' for p in first)
    metrics = retriever.metrics
    assert metrics['query_count'] == metrics['encode_count'] == 2 and metrics['model_load_count'] == 1
    assert metrics['model_load_seconds'] > 0 and metrics['encode_seconds'] > 0 and metrics['failed_queries'] == 0
    assert [q['encode_count'] for q in metrics['queries']] == [1, 1]
    assert metrics['queries'][0]['model_load_seconds'] > 0 and metrics['queries'][1]['model_load_seconds'] == 0
    assert all(q['status'] == 'complete' and q['encode_seconds'] > 0 for q in metrics['queries'])
    metrics['queries'].clear()
    assert len(retriever.metrics['queries']) == 2
    changed = retriever.identity_metadata()
    changed['source_fingerprint'] = 'fake'
    assert retriever.identity_metadata()['source_fingerprint'] == evidence.source_fingerprint
    evidence.passages = (replace(original[0], text='caller mutation'), *original[1:])
    assert retriever.search('Buchführung', 2) == first
    retriever.close()
    with pytest.raises(ValueError, match='closed'):
        retriever.search('Buchführung')


def test_vector_api_returns_exact_ranking_input_with_one_encode_and_bound_journal(
        evidence, model_files, fake_loader, monkeypatch):
    retriever = hybrid.HybridResearchRetriever(evidence, model_files)
    observed, original_rank = [], hybrid.rank_passages

    def rank(bound_evidence, query, vector, limit):
        observed.append(vector)
        return original_rank(bound_evidence, query, vector, limit)

    monkeypatch.setattr(hybrid, 'rank_passages', rank)
    selected, vector = retriever.search_with_vector('Buchführung', 2)
    assert len(fake_loader.encodes) == 1 and fake_loader.encodes == [['Buchführung']]
    assert observed[0] is vector
    assert selected == original_rank(retriever._evidence, 'Buchführung', vector, 2)
    assert vector.shape == (1024,) and vector.dtype == np.dtype('<f4')
    assert vector.flags.c_contiguous and not vector.flags.writeable
    assert not np.shares_memory(vector, fake_loader.last_matrix)
    identity = retriever.identity_metadata()
    assert identity['format'] == 'research_cpu_hybrid/2'
    assert identity['query_vector'] == {'dtype': 'float32', 'dimension': 1024, 'byte_order': 'little', 'order': 'C'}
    assert identity['policy']['anchors'] is False and identity['policy']['rrf_k'] == 60
    metrics = retriever.metrics
    assert metrics['query_count'] == metrics['encode_count'] == 1
    event = metrics['queries'][0]
    assert event['query_vector_sha256'] == hashlib.sha256(vector.tobytes(order='C')).hexdigest()
    assert event['query_sha256'] == hashlib.sha256('Buchführung'.encode()).hexdigest()
    assert event['status'] == 'complete' and event['encode_count'] == 1 and event['limit'] == 2
    assert event['retrieval_seconds'] > 0 and event['encode_seconds'] > 0
    fake_loader.last_matrix[:] = 0
    assert vector[0] == 1 and np.linalg.norm(vector) == 1
    with pytest.raises(ValueError):
        vector[0] = 0


def test_ordinary_search_delegates_without_a_second_query_or_encode(evidence, model_files, fake_loader, monkeypatch):
    retriever = hybrid.HybridResearchRetriever(evidence, model_files)
    calls, original = [], retriever.search_with_vector

    def with_vector(query, limit=6):
        result = original(query, limit)
        calls.append((query, limit, result))
        return result

    monkeypatch.setattr(retriever, 'search_with_vector', with_vector)
    selected = retriever.search('Buchführung', 3)
    assert len(calls) == 1 and calls[0][:2] == ('Buchführung', 3)
    assert selected is calls[0][2][0]
    assert retriever.metrics['query_count'] == retriever.metrics['encode_count'] == 1
    assert len(fake_loader.encodes) == 1


@pytest.mark.parametrize('kind', ['nan', 'float64', 'shape', 'unnormalized'])
def test_invalid_encoded_vectors_have_no_fabricated_digest(evidence, model_files, fake_loader, monkeypatch, kind):
    retriever = hybrid.HybridResearchRetriever(evidence, model_files)
    retriever.load()
    vector = np.zeros((1, 1024), dtype=np.float32)
    vector[0, 0] = 1
    if kind == 'nan':
        vector[0, 0] = np.nan
    elif kind == 'float64':
        vector = vector.astype(np.float64)
    elif kind == 'shape':
        vector = vector[:, :1023]
    else:
        vector[0, 0] = 2
    monkeypatch.setattr(retriever._encoder, 'encode', lambda texts: vector)
    with pytest.raises(ValueError, match='invalid normalized query vector'):
        retriever.search_with_vector('Buchführung')
    metrics = retriever.metrics
    assert metrics['query_count'] == metrics['encode_count'] == metrics['failed_queries'] == 1
    assert metrics['queries'][0]['query_vector_sha256'] is None


def test_rank_failure_preserves_digest_of_the_already_encoded_vector(evidence, model_files, fake_loader, monkeypatch):
    retriever = hybrid.HybridResearchRetriever(evidence, model_files)

    def broken_rank(*args):
        raise RuntimeError('source ranking failed')

    monkeypatch.setattr(hybrid, 'rank_passages', broken_rank)
    with pytest.raises(RuntimeError, match='source ranking failed'):
        retriever.search_with_vector('Buchführung')
    event = retriever.metrics['queries'][0]
    assert event['status'] == 'failed' and event['encode_count'] == 1
    assert event['query_vector_sha256'] == hashlib.sha256(fake_loader.last_matrix[0].tobytes()).hexdigest()


def test_token_guard_failure_does_not_claim_an_encoded_vector(evidence, model_files, fake_loader):
    retriever = hybrid.HybridResearchRetriever(evidence, model_files)
    fake_loader.model.max_seq_length = 1
    with pytest.raises(ValueError, match='truncation'):
        retriever.search_with_vector('Buchführung')
    event = retriever.metrics['queries'][0]
    assert event['status'] == 'failed' and event['encode_count'] == 0
    assert event['query_vector_sha256'] is None and fake_loader.encodes == []


def test_wrong_index_revision_fails_before_model_validation(evidence, monkeypatch):
    evidence._metadata['encoder_revision'] = 'wrong revision'
    monkeypatch.setattr(hybrid, 'validate_model_path', lambda path: pytest.fail('must not inspect model'))
    with pytest.raises(ValueError, match='different encoder'):
        hybrid.HybridResearchRetriever(evidence)


def test_model_changes_between_binding_and_loading_are_rejected(evidence, model_files, fake_loader):
    retriever = hybrid.HybridResearchRetriever(evidence, model_files)
    (model_files / 'config.json').write_bytes(b'changed configuration')
    with pytest.raises(ValueError, match='changed'):
        retriever.search('Buchführung')
    assert fake_loader.loads == 0 and retriever.metrics['failed_queries'] == 1


def test_no_silent_token_truncation_or_successful_count_on_encoder_failure(evidence, model_files, fake_loader):
    retriever = hybrid.HybridResearchRetriever(evidence, model_files)
    fake_loader.model.max_seq_length = 5
    with pytest.raises(ValueError, match='truncation'):
        retriever.search('Längere Buchführungsfrage')
    assert retriever.metrics['encode_count'] == 0
    fake_loader.model.max_seq_length = 100
    fake_loader.fail = True
    with pytest.raises(RuntimeError, match='encoding failed'):
        retriever.search('Buchführung')
    assert retriever.metrics['encode_count'] == 1 and retriever.metrics['failed_queries'] == 2


def test_cpu_loader_allows_gpu_capable_runtime_but_passes_only_local_cpu_arguments(monkeypatch, tmp_path):
    calls = []
    class ForbiddenGPU:
        def __getattr__(self, name):
            pytest.fail(f'No GPU API or availability probe is allowed: {name}')
    fake_torch = SimpleNamespace(version=SimpleNamespace(cuda='12', hip='7.2'), cuda=ForbiddenGPU(), float32='float32',
                                 set_num_threads=lambda count: calls.append(('threads', count)))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    class Encoder:
        dim = 1024

        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

        def _laden(self):
            return SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type='cpu'), dtype='float32')],
                                   buffers=lambda: [])

    monkeypatch.setattr(hybrid, '_CpuBgeEncoder', Encoder)
    hybrid._load_cpu_encoder(tmp_path)
    assert calls == [('threads', 2), (str(tmp_path), {'device': 'cpu', 'revision': BGE_M3_REVISION, 'local_files_only': True})]


def test_every_encode_explicitly_requests_cpu_and_preserves_integer_buffers(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32'))
    buffer = SimpleNamespace(device=SimpleNamespace(type='cpu'), dtype='int64', is_floating_point=lambda: False)
    calls = []

    def encode(texts, **kwargs):
        calls.append((texts, kwargs))
        return np.ones((len(texts), 1024), dtype=np.float32)

    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type='cpu'), dtype='float32')],
                            buffers=lambda: [buffer], encode=encode)
    encoder = hybrid._CpuBgeEncoder('local', device='cpu', revision=BGE_M3_REVISION, local_files_only=True)
    monkeypatch.setattr(encoder, '_laden', lambda: model)
    assert encoder.encode(['Suchidee']).shape == (1, 1024)
    assert encoder.encode(['Andere Suchidee']).shape == (1, 1024)
    assert all(kwargs['device'] == 'cpu' and kwargs['normalize_embeddings'] for _, kwargs in calls)
    assert buffer.dtype == 'int64'


@pytest.mark.parametrize('target', ['parameter', 'buffer'])
def test_non_cpu_tensor_is_rejected_before_encoding_without_device_api(monkeypatch, target):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32'))
    parameter = SimpleNamespace(device=SimpleNamespace(type='cpu'), dtype='float32')
    buffer = SimpleNamespace(device=SimpleNamespace(type='cpu'), dtype='float32', is_floating_point=lambda: True)
    (parameter if target == 'parameter' else buffer).device.type = 'cuda'
    model = SimpleNamespace(parameters=lambda: [parameter], buffers=lambda: [buffer],
                            encode=lambda *args, **kwargs: pytest.fail('must fail before inference'))
    encoder = hybrid._CpuBgeEncoder('local', device='cpu')
    monkeypatch.setattr(encoder, '_laden', lambda: model)
    with pytest.raises(ValueError, match='CPU'):
        encoder.encode(['Suchidee'])


def test_changed_float_buffer_after_encode_cannot_escape_cpu_contract(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32'))
    buffer = SimpleNamespace(device=SimpleNamespace(type='cpu'), dtype='float32', is_floating_point=lambda: True)

    def encode(texts, **kwargs):
        buffer.dtype = 'float16'
        return np.ones((1, 1024), dtype=np.float32)

    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type='cpu'), dtype='float32')],
                            buffers=lambda: [buffer], encode=encode)
    encoder = hybrid._CpuBgeEncoder('local', device='cpu')
    monkeypatch.setattr(encoder, '_laden', lambda: model)
    with pytest.raises(ValueError, match='buffers'):
        encoder.encode(['Suchidee'])
