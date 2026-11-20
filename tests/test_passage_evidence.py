"""Optional passage evidence stays source-bound and inside the two-file index."""

import hashlib
import json

import numpy as np
import pytest

from risk_layer.catalog import Katalog
from risk_layer.embedding import EmbeddingBuildCancelled, EmbeddingIndex, pruefe_aktualitaet
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.models import Katalogeintrag
from risk_layer.passages import PassageEvidence


class Encoder:
    name = "passage-test"
    revision = None
    dim = 2

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.array([[0., 1.] if "TAIL" in t else [1., 0.] for t in texts],
                        dtype=np.float32)


@pytest.fixture
def graph(tmp_path):
    nodes = [NormKnoten(id="norm:EStG:10", law="EStG", para="10", titel="Realer Titel",
                       text="  (1) " + "Dies ist der vordere Abschnitt mit einer Regelung. " * 70
                            + "\n\n(2) TAIL Nur am Ende steht die entscheidende Rechtsfolge.  \n"),
             NormKnoten(id="norm:EStG:3", law="EStG", para="3", titel="",
                       text="\nTAIL Eine Norm ohne amtliche Überschrift bleibt suchbar.  "),
             NormKnoten(id="norm:EStG:5", law="EStG", para="5", titel="Weggefallen",
                       text="(weggefallen)")]
    graph = NormGraph.build(nodes, tmp_path / "graph.sqlite")
    try:
        yield graph
    finally:
        graph.close()


@pytest.fixture
def catalog():
    return Katalog("test1", [Katalogeintrag(
        id="tail", begriff_kanonisch="Endregelung", norm_anker=("§ 10 EStG",),
        paraphrasen=("Eine ergänzende Aussage.",))])


def test_truncated_ranked_package_cannot_attribute_source_words_to_an_omitted_anchor(graph):
    catalog = Katalog("test", [Katalogeintrag(
        id="clipping", begriff_kanonisch="Beleg", norm_anker=("§ 10 EStG",),
        paraphrasen=("A" * 2200, "vordere Abschnitt"))])
    index = EmbeddingIndex.build(graph, Encoder(), katalog=catalog, passagen=True)
    package = index.norm_package("norm:EStG:10", np.array([1., 0.]))
    assert package.truncated
    assert "vordere Abschnitt" in package.text
    assert package.anchors == ()


def test_optional_build_preserves_base_and_reuses_query_vector(graph, catalog):
    encoder = Encoder()
    old = EmbeddingIndex.build(graph, encoder, katalog=catalog)
    index = EmbeddingIndex.build(graph, encoder, katalog=catalog, passagen=True)
    assert old.passage_evidence is None
    assert old.norm_package("norm:EStG:10", np.array([0., 1.])) is None
    assert index.ids == old.ids
    assert np.array_equal(index.vectors, old.vectors)
    assert index.suche("TAIL", encoder, min_sim=.35) == []
    calls = encoder.calls
    vector = index.query_vector("TAIL", encoder)
    hits = index.hybrid_topk(vector, k=2)
    assert {h["norm_id"] for h in hits} == {"norm:EStG:10", "norm:EStG:3"}
    assert len(hits) == len({h["norm_id"] for h in hits})
    package = index.norm_package("norm:EStG:10", vector)
    assert package is not None and "TAIL" in package.text
    assert package.evidence[0].start > 1500
    for p in package.evidence:
        assert graph.node(p.norm_id)["text"][p.start:p.end] == p.text
    assert encoder.calls == calls + 1
    assert old.input_fingerprint != index.input_fingerprint
    assert pruefe_aktualitaet(index, graph, catalog)


def test_evidence_roundtrip_in_exactly_two_files(graph, catalog, tmp_path):
    index = EmbeddingIndex.build(graph, Encoder(), katalog=catalog, passagen=True)
    target = tmp_path / "index"
    index.save(target)
    assert {p.name for p in target.iterdir()} == {"vectors.npy", "meta.json"}
    assert np.load(target / "vectors.npy").shape[0] > index.vectors.shape[0]
    loaded = EmbeddingIndex.load(target)
    assert loaded is not None
    assert np.array_equal(loaded.vectors, index.vectors)
    assert loaded.ids == index.ids
    assert loaded.passage_evidence.fingerprint == index.passage_evidence.fingerprint
    assert loaded.passage_evidence.passages == index.passage_evidence.passages
    assert np.array_equal(loaded.passage_evidence.vectors, index.passage_evidence.vectors)
    assert not loaded.passage_evidence.vectors.flags.writeable
    assert loaded.norm_package("norm:EStG:10", np.array([0., 1.])) == index.norm_package(
        "norm:EStG:10", np.array([0., 1.]))
    assert pruefe_aktualitaet(loaded, graph, catalog)
    second = tmp_path / "second"
    loaded.save(second)
    assert (target / "meta.json").read_bytes() == (second / "meta.json").read_bytes()
    assert (target / "vectors.npy").read_bytes() == (second / "vectors.npy").read_bytes()


def test_titleless_only_graph_roundtrip_keeps_empty_base(tmp_path):
    graph = NormGraph.build([NormKnoten(id="norm:EStG:3", law="EStG", para="3",
        text="TAIL Diese Norm hat keine amtliche Überschrift.")], tmp_path / "empty-base.sqlite")
    try:
        index = EmbeddingIndex.build(graph, Encoder(), passagen=True)
        assert index.vectors.size == 0 and index.dim == 0
        index.save(tmp_path / "index")
        loaded = EmbeddingIndex.load(tmp_path / "index")
        assert loaded is not None and loaded.vectors.size == 0
        assert loaded.hybrid_suche("TAIL", Encoder())[0]["norm_id"] == "norm:EStG:3"
    finally:
        graph.close()


def test_existing_passage_index_without_base_checksum_is_readable(graph, catalog, tmp_path):
    index = EmbeddingIndex.build(graph, Encoder(), katalog=catalog, passagen=True)
    index.save(tmp_path)
    path = tmp_path / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    del meta["base_sha256"]
    path.write_text(json.dumps(meta), encoding="utf-8")
    loaded = EmbeddingIndex.load(tmp_path)
    assert loaded is not None
    assert np.array_equal(loaded.vectors, index.vectors)


def test_full_original_text_change_invalidates_evidence_even_beyond_old_prefix(graph, catalog):
    index = EmbeddingIndex.build(graph, Encoder(), katalog=catalog, passagen=True)
    source = graph.node("norm:EStG:10")["text"]
    graph.con.execute("UPDATE knoten SET text=? WHERE id=?",
                      (source + " Neue Endvorschrift.", "norm:EStG:10"))
    graph.con.commit()
    assert not index.passage_evidence.matches_graph(graph)
    assert not pruefe_aktualitaet(index, graph, catalog)
    rebuilt = EmbeddingIndex.build(graph, Encoder(), katalog=catalog, passagen=True)
    assert rebuilt.input_fingerprint != index.input_fingerprint


@pytest.mark.parametrize("damage", ["count", "dimension", "nan", "vector", "base_vector",
                                    "base_labels", "missing", "identity", "null"])
def test_declared_damaged_evidence_never_silently_loads(graph, catalog, tmp_path, damage):
    index = EmbeddingIndex.build(graph, Encoder(), katalog=catalog, passagen=True)
    index.save(tmp_path / "index")
    path = tmp_path / "index"
    metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    vectors = np.load(path / "vectors.npy")
    if damage == "count":
        metadata["base_rows"] += 1
    elif damage == "dimension":
        vectors = vectors[:, :1]
    elif damage == "nan":
        vectors[-1, 0] = np.nan
    elif damage == "vector":
        vectors[-1, 0] += .1
    elif damage == "base_vector":
        vectors[0, 0] += .1
    elif damage == "base_labels":
        metadata["ids"][0] = "norm:EStG:999"
    elif damage == "missing":
        del metadata["passage_evidence"]
    elif damage == "identity":
        metadata["encoder"] = "other-space"
    else:
        metadata["passage_evidence"] = None
    (path / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(path / "vectors.npy", vectors)
    with pytest.raises(ValueError, match="[Pp]assage"):
        EmbeddingIndex.load(path)


def _rehash(metadata):
    metadata["fingerprint"] = hashlib.sha256(json.dumps(
        {k: v for k, v in metadata.items() if k != "fingerprint"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("damage", ["offset", "excerpt", "source_hash", "schema", "config"])
def test_load_checks_source_contract_beyond_metadata_checksum(graph, catalog, damage):
    evidence = EmbeddingIndex.build(graph, Encoder(), katalog=catalog, passagen=True).passage_evidence
    metadata, vectors = evidence.storage()
    if damage == "offset":
        metadata["passages"][0]["start"] += 1
    elif damage == "excerpt":
        metadata["passages"][0]["text"] = "Erfundene Rechtsfolge"
    elif damage == "source_hash":
        nid = metadata["sources"][0]["id"]
        metadata["source_text_hashes"][nid] = "0" * 64
    elif damage == "schema":
        metadata["schema"] = "unknown-schema"
    else:
        metadata["config"]["max_chars"] = 1500
    _rehash(metadata)
    with pytest.raises(ValueError, match="Invalid passage evidence"):
        PassageEvidence(metadata, vectors)


def test_build_is_cancellable_before_publishing_evidence(graph, catalog):
    encoder = Encoder()
    with pytest.raises(EmbeddingBuildCancelled):
        EmbeddingIndex.build(graph, encoder, katalog=catalog, passagen=True,
                             cancelled=lambda: encoder.calls >= 2)


def test_query_rejects_wrong_evidence_dimension(graph):
    index = EmbeddingIndex.build(graph, Encoder(), passagen=True)
    with pytest.raises(ValueError, match="vector dimension"):
        index.norm_package("norm:EStG:10", np.array([1., 0., 0.]))


def test_declared_evidence_without_vector_file_fails_loudly(graph, tmp_path):
    index = EmbeddingIndex.build(graph, Encoder(), passagen=True)
    index.save(tmp_path / "index")
    (tmp_path / "index" / "vectors.npy").unlink()
    with pytest.raises(ValueError, match="passage index is missing vectors"):
        EmbeddingIndex.load(tmp_path / "index")


def test_graph_hash_cache_only_applies_to_frozen_app_graphs(graph, monkeypatch):
    import risk_layer.passages as passages_module

    evidence = EmbeddingIndex.build(graph, Encoder(), passagen=True).passage_evidence
    original = passages_module._sources
    calls = []

    def counted(current):
        calls.append(current)
        return original(current)

    monkeypatch.setattr(passages_module, "_sources", counted)
    graph._embedding_graph_fingerprint_memo = "app-frozen-graph"
    assert evidence.matches_graph(graph)
    assert evidence.matches_graph(graph)
    assert calls == [graph]
    del graph._embedding_graph_fingerprint_memo
    graph.con.execute("UPDATE knoten SET titel=? WHERE id=?", ("Neue Überschrift", "norm:EStG:10"))
    graph.con.commit()
    assert not evidence.matches_graph(graph)
    assert calls == [graph, graph]
