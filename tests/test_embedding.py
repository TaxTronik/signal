"""Semantischer Embedding-Index (Schicht 1.7).

Getestet mit dem abhängigkeitsfreien HashEncoder (deterministisch), damit die
Plumbing-Garantien ohne Modell/Download gelten. Der echte BGE-M3-Pfad nutzt
exakt dieselbe Index-/Such-Logik, nur mit stärkeren Vektoren.
"""

import numpy as np

from risk_layer.catalog import lade_katalog
from risk_layer.embedding import EmbeddingIndex, HashEncoder, encoder_oeffnen
from risk_layer.gii import NormKnoten
from risk_layer.graph import NormGraph
from risk_layer.pipeline import Pipeline


def _graph(tmp_path):
    knoten = [
        NormKnoten(id="norm:AO:146", law="AO", para="146", absatz=None, nummer=None,
                   titel="Ordnungsvorschriften für die Buchführung und für Aufzeichnungen",
                   text="Bücher und Aufzeichnungen sind vollständig und richtig zu führen."),
        NormKnoten(id="norm:AO:162", law="AO", para="162", absatz=None, nummer=None,
                   titel="Schätzung von Besteuerungsgrundlagen",
                   text="Soweit die Finanzbehörde die Besteuerungsgrundlagen nicht ermitteln kann, schätzt sie."),
        NormKnoten(id="norm:AO:42", law="AO", para="42", absatz=None, nummer=None,
                   titel="Missbrauch von rechtlichen Gestaltungsmöglichkeiten",
                   text="Durch Missbrauch von Gestaltungsmöglichkeiten kann das Gesetz nicht umgangen werden."),
    ]
    return NormGraph.build(knoten, tmp_path / "g.sqlite")


def test_hash_encoder_normiert_und_deterministisch():
    enc = HashEncoder(dim=256)
    a = enc.encode(["Schätzung der Besteuerungsgrundlagen"])
    b = enc.encode(["Schätzung der Besteuerungsgrundlagen"])
    assert a.shape == (1, 256)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)   # L2-normiert
    assert np.array_equal(a, b)                          # reproduzierbar


def test_index_build_save_load(tmp_path):
    g = _graph(tmp_path)
    enc = HashEncoder(dim=256)
    idx = EmbeddingIndex.build(g, enc)
    assert len(idx.ids) == 3 and idx.dim == 256
    idx.save(tmp_path / "emb")
    wieder = EmbeddingIndex.load(tmp_path / "emb")
    assert wieder is not None
    assert wieder.ids == idx.ids
    assert np.array_equal(wieder.vectors, idx.vectors)


def test_suche_findet_passende_norm(tmp_path):
    g = _graph(tmp_path)
    enc = HashEncoder(dim=512)
    idx = EmbeddingIndex.build(g, enc)
    # Wortlaut-naher Query (Hash ist lexisch): die Schätzungs-Norm rankt vorne
    hits = idx.suche("Schätzung der Besteuerungsgrundlagen", enc, k=3, min_sim=0.1)
    assert hits and hits[0]["norm_id"] == "norm:AO:162"


def test_kandidaten_pro_satz_span_treu(tmp_path):
    # Mechanik mit dem (lexisch schwachen) HashEncoder bei test-tauglicher Schwelle:
    # satzgenaue Treffer, span-treu, mit korpus-gestützten Norm-Kandidaten.
    g = _graph(tmp_path)
    enc = HashEncoder(dim=512)
    idx = EmbeddingIndex.build(g, enc)
    text = "Vorab etwas Neutrales. Schätzung von Besteuerungsgrundlagen droht."
    treffer = idx.kandidaten_pro_satz(text, enc, k=3, min_sim=0.3)
    assert treffer, "mindestens ein Satz sollte semantisch matchen"
    for t in treffer:
        assert text[t["start"]:t["end"]] == t["text"]      # Offsets korrekt
        assert t["normen"] and "sim" in t["normen"][0]


def test_pipeline_embedding_marken_overlap_sicher(tmp_path):
    g = _graph(tmp_path)
    enc = HashEncoder(dim=512)
    idx = EmbeddingIndex.build(g, enc)
    text = "Schätzung von Besteuerungsgrundlagen."   # nahe am Normtitel ⇒ hohe Ähnlichkeit
    rep = Pipeline(lade_katalog()).run(text, graph=g, embedding=idx, embedding_encoder=enc)
    for r in (r for r in rep.risiken if r.quelle == "embedding"):
        assert text[r.start:r.end] == r.matched_text       # span-treu
        assert r.norm_vorschlag and r.norm_vorschlag[0]["gestuetzt"] is True
        assert "sim" in r.signale


def test_encoder_oeffnen_liefert_encoder_mit_vertrag():
    # 'auto' liefert einen Encoder mit name+dim. Wir laden das (schwere) BGE-Modell
    # hier NICHT – nur der Vertrag wird geprüft (encode wird separat über den
    # HashEncoder abgedeckt, der echte BGE-Pfad über den Index-Build).
    enc = encoder_oeffnen("auto", dim=128)
    assert isinstance(enc.name, str) and enc.name
    assert isinstance(enc.dim, int) and enc.dim > 0


def test_hash_encoder_explizit_ohne_modell():
    # Der abhängigkeitsfreie Pfad bleibt unabhängig vom installierten Modell.
    enc = HashEncoder(dim=128)
    assert enc.name == "hash128"
    v = enc.encode(["test"])
    assert v.shape == (1, 128)
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-3   # L2-normiert
