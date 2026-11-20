"""Catalog hints are deterministic CPU data, not source evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import numpy as np

from risk_layer.catalog import lade_katalog
from risk_layer.embedding import katalog_fingerprint_berechnen
from risk_layer.research_catalog import ResearchCatalog


def write_catalog(path, entries):
    path.write_text(json.dumps({"version": "test-1", "begriffe": entries}, ensure_ascii=False), encoding="utf-8")
    return katalog_fingerprint_berechnen(lade_katalog(path))


def entry(identifier="concept", **changes):
    return {"id": identifier, "begriff_kanonisch": "formelle Ordnung", "norm_anker": ["§ 1 AO"],
            "definitionsquelle": [{"typ": "Gesetz", "fundstelle": "§ 1 AO"}],
            "normketten": [{"glieder": ["§ 1 AO", "§ 2 AO"], "verknuepfung": "iVm", "hinweis": "Prüfhinweis"}],
            **changes}


def test_catalog_original_offsets_survive_ingest_and_pattern_matching(tmp_path):
    path = tmp_path / "catalog.yaml"
    fingerprint = write_catalog(path, [entry(ausloeser=[[["Belege"], ["fehlen"]]])])
    catalog = ResearchCatalog(path, expected_fingerprint=fingerprint)
    text = "Die formel-\nle\u00a0Ordnung wird geprüft. Belege    fehlen."
    hints = catalog.for_text(text)
    assert hints == catalog.hints(text)
    concept = hints["concepts"][0]
    assert len(concept["matches"]) == 2
    assert {method for match in concept["matches"] for method in match["methods"]} == {"literal", "pattern"}
    for match in concept["matches"]:
        assert match["text"] == text[match["start"]:match["end"]]
    assert concept["matches"][0]["text"] == "formel-\nle\u00a0Ordnung"
    assert concept["matches"][1]["text"] == "Belege    fehlen"
    assert concept["norm_chains"][0]["glieder"] == ["§ 1 AO", "§ 2 AO"]
    assert hints["status"] == concept["status"] == "unverified_hint"
    assert "keine Belege" in hints["purpose"]
    serialized = json.dumps(hints)
    assert "evidence_id" not in serialized and "quote_id" not in serialized


def test_catalog_concepts_are_deduplicated_position_then_id_with_explicit_overflow(tmp_path):
    path = tmp_path / "catalog.yaml"
    entries = [entry(name, begriff_kanonisch="Vorgang", ausloeser=[[["Vorgang"]]])
               for name in ["f", "e", "d", "c", "b", "a"]]
    fingerprint = write_catalog(path, entries)
    hints = ResearchCatalog(path, expected_fingerprint=fingerprint).hints("Vorgang. " * 6)
    assert [c["concept_id"] for c in hints["concepts"]] == ["a", "b", "c", "d"]
    assert [c["hint_id"] for c in hints["concepts"]] == ["k1", "k2", "k3", "k4"]
    assert hints["total_concepts"] == 6 and hints["omitted_concepts"] == 2
    for concept in hints["concepts"]:
        assert len(concept["matches"]) == 4 and concept["omitted"]["matches"] == 2
        assert all(match["methods"] == ["literal", "pattern"] for match in concept["matches"])


def test_catalog_oversized_references_and_whole_chains_are_omitted_counted(tmp_path):
    path = tmp_path / "catalog.yaml"
    data = entry(norm_anker=["x" * 401, *[f"§ {i} AO" for i in range(1, 6)]],
                 definitionsquelle=[{"typ": "Gesetz", "fundstelle": "x" * 401},
                                    *[{"typ": "Gesetz", "fundstelle": f"§ {i} AO"} for i in range(1, 6)]],
                 normketten=[{"glieder": ["§ 1 AO"] * 9},
                             {"glieder": ["§ 1 AO", "§ 2 AO"], "hinweis": "x" * 401},
                             {"glieder": ["§ 3 AO", "§ 4 AO"], "verknuepfung": "iVm"}])
    fingerprint = write_catalog(path, [data])
    concept = ResearchCatalog(path, expected_fingerprint=fingerprint).hints("Formelle Ordnung.")["concepts"][0]
    assert concept["norm_anchors"] == [f"§ {i} AO" for i in range(1, 5)]
    assert len(concept["definition_references"]) == 4
    assert concept["norm_chains"] == [{"glieder": ["§ 3 AO", "§ 4 AO"], "verknuepfung": "iVm", "hinweis": ""}]
    assert concept["omitted"] == {"matches": 0, "norm_anchors": 2, "definition_references": 2, "norm_chains": 2}


def test_catalog_card_size_never_creates_partial_quotes_or_references(tmp_path):
    path = tmp_path / "catalog.yaml"
    anchors = [str(i) * 400 for i in range(4)]
    references = [{"typ": "Literatur", "fundstelle": anchor} for anchor in anchors]
    data = entry(norm_anker=anchors, definitionsquelle=references,
                 normketten=[{"glieder": anchors, "hinweis": "y" * 400}])
    fingerprint = write_catalog(path, [data])
    catalog = ResearchCatalog(path, expected_fingerprint=fingerprint)
    concept = catalog.hints("Formelle Ordnung.")["concepts"][0]
    assert len(json.dumps(concept, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) <= 1600
    assert all(anchor in anchors for anchor in concept["norm_anchors"])
    assert all(reference in references for reference in concept["definition_references"])
    assert concept["matches"][0]["text"] == "Formelle Ordnung"
    assert concept["omitted"]["norm_anchors"] == len(anchors) - len(concept["norm_anchors"])
    assert concept["omitted"]["definition_references"] == len(references) - len(concept["definition_references"])
    assert concept["norm_chains"] == [] and concept["omitted"]["norm_chains"] == 1


def test_catalog_enabled_empty_and_unrenderable_matches_are_explicit(tmp_path):
    path = tmp_path / "catalog.yaml"
    fingerprint = write_catalog(path, [entry(ausloeser=[[["Anfang"], ["Ende"]]])])
    catalog = ResearchCatalog(path, expected_fingerprint=fingerprint)
    empty = catalog.hints("Nur ein neutraler Text.")
    assert empty["concepts"] == [] and empty["total_concepts"] == empty["omitted_concepts"] == 0
    long_match = catalog.hints("Anfang " + "mittig " * 100 + "Ende.")
    assert long_match["concepts"] == [] and long_match["total_concepts"] == long_match["omitted_concepts"] == 1


def test_catalog_full_yaml_identity_binds_fields_outside_embedding_fingerprint(tmp_path):
    path = tmp_path / "catalog.yaml"
    original = entry()
    fingerprint = write_catalog(path, [original])
    old = ResearchCatalog(path, expected_fingerprint=fingerprint)
    changed = deepcopy(original)
    changed["definitionsquelle"] = [{"typ": "Literatur", "fundstelle": "Andere Fundstelle"}]
    assert write_catalog(path, [changed]) == fingerprint
    new = ResearchCatalog(path, expected_fingerprint=fingerprint)
    assert old.metadata()["semantic_fingerprint"] == new.metadata()["semantic_fingerprint"]
    assert old.metadata()["sha256"] != new.metadata()["sha256"]
    assert old.metadata()["fingerprint"] != new.metadata()["fingerprint"]
    assert old.hints("Formelle Ordnung.")["concepts"][0]["definition_references"] == original["definitionsquelle"]
    assert new.metadata()["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    copy_path = tmp_path / "copy.yaml"
    copy_path.write_bytes(path.read_bytes())
    copied = ResearchCatalog(copy_path, expected_fingerprint=fingerprint)
    assert copied.metadata()["fingerprint"] == new.metadata()["fingerprint"]
    assert copied.metadata()["path"] != new.metadata()["path"]
    metadata = new.metadata()
    metadata["policy"]["max_concepts"] = 100
    assert new.metadata()["policy"]["max_concepts"] == 4


def test_catalog_rejects_wrong_index_generation(tmp_path):
    path = tmp_path / "catalog.yaml"
    write_catalog(path, [entry()])
    with pytest.raises(ValueError, match="different catalog generations"):
        ResearchCatalog(path, expected_fingerprint="wrong")


@pytest.mark.parametrize("start,end", [(0, 0), (-1, 2), (0, 1000), (False, 1)])
def test_catalog_rejects_invalid_pattern_spans_before_offset_lookup(tmp_path, monkeypatch, start, end):
    path = tmp_path / "catalog.yaml"
    fingerprint = write_catalog(path, [entry()])
    catalog = ResearchCatalog(path, expected_fingerprint=fingerprint)
    monkeypatch.setattr(catalog._pattern, "match", lambda text: [
        SimpleNamespace(begriff_id="concept", start=start, end=end)])
    with pytest.raises(ValueError, match="invalid normalized offsets"):
        catalog.hints("Formelle Ordnung.")


def test_catalog_rejects_file_change_during_load(tmp_path, monkeypatch):
    from risk_layer import research_catalog as module
    path = tmp_path / "catalog.yaml"
    fingerprint = write_catalog(path, [entry()])
    loader = module.lade_katalog

    def changing_loader(filename):
        catalog = loader(filename)
        path.write_bytes(path.read_bytes() + b"\n")
        return catalog

    monkeypatch.setattr(module, "lade_katalog", changing_loader)
    with pytest.raises(ValueError, match="changed while loading"):
        ResearchCatalog(path, expected_fingerprint=fingerprint)


def test_catalog_load_and_match_never_import_torch_or_spacy(tmp_path):
    from risk_layer import research_catalog as module
    path = tmp_path / "catalog.yaml"
    fingerprint = write_catalog(path, [entry(ausloeser=[[["Belege"], ["fehlen"]]])])
    code = """
import importlib.abc
import json
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'spacy'}:
            raise RuntimeError('Unexpected model import: ' + fullname)
sys.meta_path.insert(0, NoModels())
sys.path.insert(0, sys.argv[1])
from risk_layer.research_catalog import ResearchCatalog
catalog = ResearchCatalog(sys.argv[2], expected_fingerprint=sys.argv[3])
print(json.dumps(catalog.hints('Formelle Ordnung. Belege fehlen.')))
"""
    result = subprocess.run([sys.executable, "-c", code, str(Path(module.__file__).resolve().parents[1]),
                             str(path), fingerprint], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["concepts"][0]["matches"]) == 2


@pytest.fixture
def semantic_catalog(tmp_path):
    from risk_layer.gii import NormKnoten
    from risk_layer.graph import NormGraph
    from risk_layer.passages import PassageEvidence
    entries = [entry("literal", begriff_kanonisch="Faktenbegriff", paraphrasen=["Originale Startparaphrase"]),
               entry("a", begriff_kanonisch="Erstbegriff", norm_anker=["§ 2 AO"], normketten=[],
                     paraphrasen=["Originale Paraphrase alpha"]),
               entry("b", begriff_kanonisch="Zweitbegriff", norm_anker=["§ 2 AO"], normketten=[],
                     paraphrasen=["Originale Paraphrase beta"]),
               entry("c", begriff_kanonisch="Drittbegriff", norm_anker=["§ 3 AO"], normketten=[],
                     paraphrasen=["Originale Paraphrase gamma"]),
               entry("d", begriff_kanonisch="Viertbegriff", norm_anker=["§ 4 AO"], normketten=[],
                     paraphrasen=["Originale Paraphrase delta"])]
    path = tmp_path / "semantic-catalog.yaml"
    fingerprint = write_catalog(path, entries)
    values = dict(zip([e["paraphrasen"][0] for e in entries], [.95, .9, .89, .8, .7]))

    class Encoder:
        name, revision, dim = "test", "fixed", 3
        def encode(self, texts):
            return np.array([[score, np.sqrt(1 - score * score), 0.] if (score := values.get(text)) else [0., 0., 1.]
                             for text in texts], dtype=np.float32)

    graph = NormGraph.build([NormKnoten(id=f"norm:AO:{i}", law="AO", para=str(i), titel=f"Regel {i}",
                                      text=f"Originalregel Nummer {i} enthält die Voraussetzungen dieses Vorgangs.")
                             for i in range(1, 5)], tmp_path / "semantic.sqlite")
    try:
        evidence = PassageEvidence.build(graph, Encoder(), lade_katalog(path), catalog_fingerprint=fingerprint)
    finally:
        graph.close()
    return path, fingerprint, evidence


def test_catalog_semantic_slots_preserve_literal_first_and_prefer_distinct_norms(semantic_catalog):
    path, fingerprint, evidence = semantic_catalog
    catalog = ResearchCatalog(path, expected_fingerprint=fingerprint, evidence=evidence)
    vector = np.array([1., 0., 0.], dtype=np.float32)
    hints = catalog.for_text("Faktenbegriff bleibt offen.", query_vector=vector)
    assert hints["format"] == "research_catalog_hints/2"
    assert [c["concept_id"] for c in hints["concepts"]] == ["literal", "a", "c", "d"]
    assert hints["literal_concepts"] == 1 and hints["semantic_concepts"] == 3
    assert hints["total_concepts"] == 5 and hints["omitted_concepts"] == 1
    assert hints["concepts"][0]["matches"][0]["text"] == "Faktenbegriff"
    for concept, nid, score in zip(hints["concepts"][1:], [2, 3, 4], [.9, .8, .7]):
        assert concept["matches"] == [] and concept["matching"] == "semantic_anchor"
        match = concept["semantic_matches"][0]
        assert match["norm_id"] == f"norm:AO:{nid}" and match["score"] == float(np.float32(score))
        assert (match["norm_id"], match["anchor_text"]) in evidence.anchors
        assert "start" not in match and "end" not in match
    assert "query_vector" not in hints and "evidence_id" not in json.dumps(hints)
    assert catalog.hints("Faktenbegriff bleibt offen.", vector.tolist()) == hints
    assert catalog.metadata()["semantic"]["evidence_fingerprint"] == evidence.fingerprint
    assert catalog.metadata()["semantic"]["anchor_vectors_sha256"] == hashlib.sha256(evidence.anchor_vectors.tobytes()).hexdigest()


def test_catalog_semantic_has_fixed_ties_and_respects_full_literal_slots(semantic_catalog):
    path, fingerprint, evidence = semantic_catalog
    from risk_layer.passages import PassageEvidence, _digest
    metadata, vectors = evidence.storage()
    for row in range(len(evidence.passages), len(vectors)):
        vectors[row] = [1., 0., 0.]
    metadata["vectors_sha256"] = hashlib.sha256(vectors.tobytes()).hexdigest()
    metadata.pop("fingerprint")
    metadata["fingerprint"] = _digest(metadata)
    tied = PassageEvidence(metadata, vectors)
    catalog = ResearchCatalog(path, expected_fingerprint=fingerprint, evidence=tied)
    hints = catalog.hints("Neutraler Vorgang.", np.array([1., 0., 0.], dtype=np.float32))
    assert [c["concept_id"] for c in hints["concepts"]] == ["literal", "a", "c", "d"]
    # With four literal slots already occupied, semantic candidates add no extra cards.
    literal_text = "Faktenbegriff Erstbegriff Zweitbegriff Drittbegriff."
    full = catalog.hints(literal_text, np.array([1., 0., 0.], dtype=np.float32))
    assert full["literal_concepts"] == 4 and full["semantic_concepts"] == 0
    assert all(c["matches"] for c in full["concepts"])


@pytest.mark.parametrize("vector", [None, np.array([1., 0.], dtype=np.float32),
                                  np.array([1., 0., 0.], dtype=np.float64),
                                  np.array([0., 0., 0.], dtype=np.float32),
                                  np.array([np.nan, 0., 0.], dtype=np.float32), [1, 0, 0], [.1, .2, .3]])
def test_catalog_semantic_rejects_missing_changed_or_invalid_query_vectors(semantic_catalog, vector):
    path, fingerprint, evidence = semantic_catalog
    catalog = ResearchCatalog(path, expected_fingerprint=fingerprint, evidence=evidence)
    with pytest.raises(ValueError, match="query"):
        catalog.hints("Neutraler Vorgang.", vector)


@pytest.mark.parametrize("field,value", [("text", "Nicht im Katalog gespeicherte Paraphrase"),
                                      ("norm_id", "norm:AO:4")])
def test_catalog_semantic_requires_exact_norm_and_paraphrase_ownership(semantic_catalog, field, value):
    from risk_layer.passages import PassageEvidence, _digest
    path, fingerprint, evidence = semantic_catalog
    metadata, vectors = evidence.storage()
    metadata["anchors"][0][field] = value
    metadata.pop("fingerprint")
    metadata["fingerprint"] = _digest(metadata)
    changed = PassageEvidence(metadata, vectors)
    with pytest.raises(ValueError, match="anchors differ"):
        ResearchCatalog(path, expected_fingerprint=fingerprint, evidence=changed)


def test_catalog_semantic_snapshot_does_not_follow_mutated_input_vectors(semantic_catalog):
    path, fingerprint, evidence = semantic_catalog
    catalog = ResearchCatalog(path, expected_fingerprint=fingerprint, evidence=evidence)
    vector = np.array([1., 0., 0.], dtype=np.float32)
    before = catalog.hints("Neutraler Vorgang.", vector)
    evidence.anchor_vectors.flags.writeable = True
    evidence.anchor_vectors[:] = [0., 0., 1.]
    assert catalog.hints("Neutraler Vorgang.", vector) == before
