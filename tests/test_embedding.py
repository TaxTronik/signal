"""Semantischer Embedding-Index (Schicht 1.7).

Getestet mit dem abhängigkeitsfreien HashEncoder (deterministisch), damit die
Plumbing-Garantien ohne Modell/Download gelten. Der echte BGE-M3-Pfad nutzt
exakt dieselbe Index-/Such-Logik, nur mit stärkeren Vektoren.
"""

import json
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from risk_layer.catalog import lade_katalog
from risk_layer.embedding import (EMBEDDING_BUILDER_VERSION,
                                  HF_MODELL_REVISIONEN, BgeM3Encoder,
                                  EmbeddingBuildCancelled, EmbeddingIndex,
                                  GraniteR2Encoder, HashEncoder,
                                  JinaV5Encoder, Qwen3Encoder, encoder_oeffnen,
                                  pruefe_aktualitaet)
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


def test_operator_build_prueft_abbruch_zwischen_modellbatches(tmp_path):
    knoten = [
        NormKnoten(
            id=f"norm:AO:{nummer}",
            law="AO",
            para=str(nummer),
            absatz=None,
            nummer=None,
            titel=f"Testtitel {nummer}",
            text=f"Ausreichend langer Testwortlaut fuer Paragraph {nummer}.",
        )
        for nummer in range(100, 140)
    ]
    graph = NormGraph.build(knoten, tmp_path / "cancel.sqlite")

    class CountingEncoder:
        name = "test-encoder"
        revision = "test-revision"
        dim = 4

        def __init__(self) -> None:
            self.calls = 0

        def encode(self, texte):
            self.calls += 1
            return np.ones((len(texte), self.dim), dtype=np.float32)

    encoder = CountingEncoder()
    try:
        with pytest.raises(EmbeddingBuildCancelled):
            EmbeddingIndex.build(
                graph,
                encoder,
                cancelled=lambda: encoder.calls >= 1,
            )
        assert encoder.calls == 1
    finally:
        graph.close()


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


def test_kandidaten_pro_satz_kann_leere_pools_fuer_fusion_liefern():
    enc = HashEncoder(dim=64)
    idx = EmbeddingIndex(np.zeros((1, 64), dtype=np.float32),
                         ["norm:AO:42"], ["§ 42 AO"], ["Missbrauch"], enc.name)
    text = "  Hier fehlt ein semantischer Treffer.\n  Auch dieser Satz bleibt offen.  "
    assert idx.kandidaten_pro_satz(text, enc) == []

    saetze = idx.kandidaten_pro_satz(text, enc, auch_ohne_treffer=True)
    assert [s["text"] for s in saetze] == [
        "Hier fehlt ein semantischer Treffer", "Auch dieser Satz bleibt offen"]
    assert all(s["normen"] == [] for s in saetze)
    assert all(text[s["start"]:s["end"]] == s["text"] for s in saetze)


@pytest.mark.parametrize("text", ["", "  ", "Kurz.", "--------------------"])
def test_leere_fusionspools_nur_fuer_inhaltliche_saetze(text):
    class UnbenutzterEncoder:
        def encode(self, _texte):
            raise AssertionError("Leerer oder zu kurzer Text braucht kein Embedding")

    idx = EmbeddingIndex(np.zeros((0, 0), dtype=np.float32), [], [], [], "unused")
    assert idx.kandidaten_pro_satz(
        text, UnbenutzterEncoder(), auch_ohne_treffer=True) == []


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


@pytest.mark.parametrize(("encoder", "modell"), [
    (BgeM3Encoder(), "BAAI/bge-m3"),
    (GraniteR2Encoder(), "ibm-granite/granite-embedding-311m-multilingual-r2"),
    (JinaV5Encoder(), "jinaai/jina-embeddings-v5-text-small-retrieval"),
    (Qwen3Encoder(variante="0.6B"), "Qwen/Qwen3-Embedding-0.6B"),
    (Qwen3Encoder(variante="4B"), "Qwen/Qwen3-Embedding-4B"),
    (Qwen3Encoder(variante="8B"), "Qwen/Qwen3-Embedding-8B"),
])
def test_kuratierte_encoder_laden_exakten_commit(monkeypatch, encoder, modell):
    aufrufe = []

    class FakeModell:
        @staticmethod
        def get_embedding_dimension():
            return 64

    def fake_sentence_transformer(name, **kwargs):
        aufrufe.append((name, kwargs))
        return FakeModell()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )
    encoder._laden()
    assert encoder.revision == HF_MODELL_REVISIONEN[modell]
    assert aufrufe == [(modell, {
        "device": None,
        "trust_remote_code": isinstance(encoder, JinaV5Encoder),
        "revision": HF_MODELL_REVISIONEN[modell],
        "local_files_only": False,
    })]


def test_lokaler_encoder_pfad_bleibt_ohne_hf_revision(monkeypatch, tmp_path):
    modellpfad = tmp_path / "lokales-modell"
    modellpfad.mkdir()
    aufrufe = []

    class FakeModell:
        @staticmethod
        def get_embedding_dimension():
            return 64

    def fake_sentence_transformer(name, **kwargs):
        aufrufe.append((name, kwargs))
        return FakeModell()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )
    encoder = encoder_oeffnen(str(modellpfad))
    encoder._laden()
    assert encoder.revision is None
    assert aufrufe[0][0] == str(modellpfad)
    assert aufrufe[0][1]["revision"] is None
    assert aufrufe[0][1]["local_files_only"] is True


def test_kuratierter_encoder_laesst_pin_nicht_ueberschreiben():
    with pytest.raises(ValueError, match="ausschließlich Revision"):
        BgeM3Encoder(revision="0" * 40)


# --- Staleness-Erkennung (graph_fingerprint / pruefe_aktualitaet) ------------
# Problemfall: corpus/Graph wird neu gebaut, der Embedding-Index vergessen ⇒
# stille, plausible, FALSCHE Semantik-Vorschläge. Der Fingerprint des Quell-
# Graphen wird beim Build eingebrannt und beim Start abgeglichen.

def _graph_geaendert(tmp_path):
    """Wie _graph, aber mit geändertem Wortlaut von § 146 (Korpus-Rebuild)."""
    knoten = [
        NormKnoten(id="norm:AO:146", law="AO", para="146", absatz=None, nummer=None,
                   titel="Ordnungsvorschriften für die Buchführung und für Aufzeichnungen",
                   text="GEÄNDERTER Wortlaut: Aufzeichnungen sind einzeln zu führen."),
        NormKnoten(id="norm:AO:162", law="AO", para="162", absatz=None, nummer=None,
                   titel="Schätzung von Besteuerungsgrundlagen",
                   text="Soweit die Finanzbehörde die Besteuerungsgrundlagen nicht ermitteln kann, schätzt sie."),
        NormKnoten(id="norm:AO:42", law="AO", para="42", absatz=None, nummer=None,
                   titel="Missbrauch von rechtlichen Gestaltungsmöglichkeiten",
                   text="Durch Missbrauch von Gestaltungsmöglichkeiten kann das Gesetz nicht umgangen werden."),
    ]
    return NormGraph.build(knoten, tmp_path / "g2.sqlite")


def test_build_brennt_graph_fingerprint_ein_und_meta_roundtrip(tmp_path):
    g = _graph(tmp_path)
    idx = EmbeddingIndex.build(g, HashEncoder(dim=64))
    assert idx.graph_fingerprint == g.fingerprint()        # beim Build abgegriffen
    assert idx.embedding_graph_fingerprint is not None
    assert idx.input_fingerprint is not None
    assert idx.builder_version == EMBEDDING_BUILDER_VERSION
    idx.save(tmp_path / "emb")
    wieder = EmbeddingIndex.load(tmp_path / "emb")
    # Roundtrip build → save → load: der Fingerprint überlebt die Persistenz.
    assert wieder is not None
    assert wieder.graph_fingerprint == g.fingerprint()
    assert wieder.embedding_graph_fingerprint == idx.embedding_graph_fingerprint
    assert wieder.input_fingerprint == idx.input_fingerprint
    assert wieder.builder_version == EMBEDDING_BUILDER_VERSION


def test_build_persistiert_encoder_revision(tmp_path):
    class FakeEncoder:
        name = "bge-m3"
        revision = HF_MODELL_REVISIONEN["BAAI/bge-m3"]
        dim = 8

        @staticmethod
        def encode(texts):
            return np.zeros((len(texts), 8), dtype=np.float32)

    idx = EmbeddingIndex.build(_graph(tmp_path), FakeEncoder())
    idx.save(tmp_path / "emb")
    meta = json.loads((tmp_path / "emb" / "meta.json").read_text(encoding="utf-8"))
    assert meta["encoder_revision"] == FakeEncoder.revision
    wieder = EmbeddingIndex.load(tmp_path / "emb")
    assert wieder is not None
    assert wieder.encoder_revision == FakeEncoder.revision


def test_build_persistiert_katalogstand_und_erkennt_paraphrasen_drift(tmp_path):
    from risk_layer.catalog import Katalog
    from risk_layer.models import Katalogeintrag

    katalog = Katalog("2026.1", [Katalogeintrag(
        id="kasse",
        begriff_kanonisch="Kasse",
        norm_anker=("§ 146 AO",),
        paraphrasen=("Bareinnahmen fehlen.",),
    )])
    g = _graph(tmp_path)
    idx = EmbeddingIndex.build(g, HashEncoder(dim=64), katalog=katalog)
    idx.save(tmp_path / "emb-katalog")
    wieder = EmbeddingIndex.load(tmp_path / "emb-katalog")
    assert wieder is not None
    assert wieder.catalog_version == "2026.1"
    assert wieder.catalog_fingerprint == idx.catalog_fingerprint
    assert pruefe_aktualitaet(wieder, g, katalog) is True

    geaendert = Katalog("2026.1", [Katalogeintrag(
        id="kasse",
        begriff_kanonisch="Kasse",
        norm_anker=("§ 146 AO",),
        paraphrasen=("Auch Kartenzahlungen fehlen.",),
    )])
    assert pruefe_aktualitaet(wieder, g, geaendert) is False
    neu = EmbeddingIndex.build(g, HashEncoder(dim=64), katalog=geaendert)
    assert neu.input_fingerprint != wieder.input_fingerprint


def test_load_alt_index_ohne_fingerprint_tolerant(tmp_path):
    # Alt-Index simulieren: meta.json OHNE das Feld (vor der Einführung gebaut).
    g = _graph(tmp_path)
    EmbeddingIndex.build(g, HashEncoder(dim=64)).save(tmp_path / "emb")
    mp = tmp_path / "emb" / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    del meta["graph_fingerprint"]
    del meta["embedding_graph_fingerprint"]
    del meta["input_fingerprint"]
    del meta["encoder_revision"]
    del meta["catalog_version"]
    del meta["catalog_fingerprint"]
    del meta["builder_version"]
    mp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    alt = EmbeddingIndex.load(tmp_path / "emb")
    assert alt is not None                                 # lädt weiter (tolerant)
    assert alt.graph_fingerprint is None                   # … nur ohne Fingerprint
    assert alt.embedding_graph_fingerprint is None
    assert alt.input_fingerprint is None
    assert alt.encoder_revision is None                    # … und ohne Modell-SHA
    assert alt.catalog_version is None                     # … sowie Katalogstand
    assert alt.catalog_fingerprint is None
    assert alt.builder_version is None


def test_fehlende_builder_version_markiert_sonst_passenden_index_veraltet(
    tmp_path, caplog
):
    g = _graph(tmp_path)
    ziel = tmp_path / "emb-builder-alt"
    EmbeddingIndex.build(g, HashEncoder(dim=64)).save(ziel)
    meta_pfad = ziel / "meta.json"
    meta = json.loads(meta_pfad.read_text(encoding="utf-8"))
    del meta["builder_version"]
    meta_pfad.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    alt = EmbeddingIndex.load(ziel)
    assert alt is not None

    with caplog.at_level(logging.WARNING, logger="risk_layer.embedding"):
        assert pruefe_aktualitaet(alt, g) is False
    assert "Builder-Version" in caplog.text


def test_pruefe_aktualitaet_match_still_und_true(tmp_path, caplog):
    g = _graph(tmp_path)
    idx = EmbeddingIndex.build(g, HashEncoder(dim=64))
    with caplog.at_level(logging.INFO, logger="risk_layer.embedding"):
        assert pruefe_aktualitaet(idx, g) is True
    # Nachgewiesene Übereinstimmung ⇒ KEIN Log-Lärm.
    assert not caplog.records


def test_pruefe_aktualitaet_erkennt_reine_titelaenderung(tmp_path, caplog):
    g = _graph(tmp_path)
    idx = EmbeddingIndex.build(g, HashEncoder(dim=64))
    text_fingerprint = g.fingerprint()
    g.con.execute(
        "UPDATE knoten SET titel = ? WHERE id = ?",
        ("Neue fachliche Ueberschrift", "norm:AO:146"),
    )
    g.con.commit()

    # Der allgemeine Graphvertrag ist bewusst textbasiert; der neue
    # Embedding-Vertrag muss die Titelabweichung trotzdem erkennen.
    assert g.fingerprint() == text_fingerprint
    with caplog.at_level(logging.WARNING, logger="risk_layer.embedding"):
        assert pruefe_aktualitaet(idx, g) is False
    assert "Titel" in caplog.text


def test_fehlender_embedding_graph_fingerprint_ist_nicht_aktuell(tmp_path):
    g = _graph(tmp_path)
    ziel = tmp_path / "emb-ohne-graph-input"
    EmbeddingIndex.build(g, HashEncoder(dim=64)).save(ziel)
    meta_pfad = ziel / "meta.json"
    meta = json.loads(meta_pfad.read_text(encoding="utf-8"))
    del meta["embedding_graph_fingerprint"]
    meta_pfad.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    alt = EmbeddingIndex.load(ziel)
    assert alt is not None
    assert pruefe_aktualitaet(alt, g) is False


def test_fehlender_composite_input_fingerprint_ist_nicht_aktuell(tmp_path):
    g = _graph(tmp_path)
    ziel = tmp_path / "emb-ohne-composite-input"
    EmbeddingIndex.build(g, HashEncoder(dim=64)).save(ziel)
    meta_pfad = ziel / "meta.json"
    meta = json.loads(meta_pfad.read_text(encoding="utf-8"))
    del meta["input_fingerprint"]
    meta_pfad.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    alt = EmbeddingIndex.load(ziel)
    assert alt is not None
    assert pruefe_aktualitaet(alt, g) is False


def test_pruefe_aktualitaet_mismatch_warnt_mit_abhilfe(tmp_path, caplog):
    g_alt = _graph(tmp_path)
    idx = EmbeddingIndex.build(g_alt, HashEncoder(dim=64))
    g_neu = _graph_geaendert(tmp_path)                     # Korpus-Rebuild danach
    with caplog.at_level(logging.WARNING, logger="risk_layer.embedding"):
        assert pruefe_aktualitaet(idx, g_neu) is False     # degradierend: kein Raise
    warnungen = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnungen) == 1
    # Die Warnung muss die HANDLUNGSANWEISUNG nennen, nicht nur das Problem.
    assert "risk-layer embedding build" in warnungen[0].getMessage()


def test_pruefe_aktualitaet_alt_index_info_einmalig(tmp_path, caplog):
    g = _graph(tmp_path)
    # Alt-Index: direkt ohne Fingerprint instanziieren (wie nach load() einer
    # alten meta.json) – Aktualität ist nicht prüfbar, also False, aber nur INFO.
    idx = EmbeddingIndex(np.zeros((0, 0), np.float32), [], [], [], "hash64")
    with caplog.at_level(logging.INFO, logger="risk_layer.embedding"):
        assert pruefe_aktualitaet(idx, g) is False
        assert pruefe_aktualitaet(idx, g) is False         # zweiter Aufruf …
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1                                 # … loggt NICHT erneut
    assert "risk-layer embedding build" in infos[0].getMessage()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_alt_index_ohne_encoder_revision_ist_klar_veraltet(tmp_path, caplog):
    g = _graph(tmp_path)
    idx = EmbeddingIndex(
        np.zeros((0, 0), np.float32),
        [],
        [],
        [],
        "bge-m3",
        graph_fingerprint=g.fingerprint(),
    )
    with caplog.at_level(logging.WARNING, logger="risk_layer.embedding"):
        assert pruefe_aktualitaet(idx, g) is False
        assert pruefe_aktualitaet(idx, g) is False
    warnungen = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnungen) == 1
    assert "encoder_revision fehlt" in warnungen[0].getMessage()
    assert "risk-layer embedding build" in warnungen[0].getMessage()


def test_alt_index_ohne_revision_wird_nicht_still_abgefragt():
    class FakeEncoder:
        name = "bge-m3"
        revision = HF_MODELL_REVISIONEN["BAAI/bge-m3"]
        dim = 2

        @staticmethod
        def encode(_texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    idx = EmbeddingIndex(
        np.array([[1.0, 0.0]], dtype=np.float32),
        ["norm:AO:1"],
        ["§ 1 AO"],
        ["Test"],
        "bge-m3",
    )
    with pytest.raises(ValueError, match="keine encoder_revision"):
        idx.suche("test", FakeEncoder())


def test_pruefe_aktualitaet_ohne_index_oder_graph(tmp_path):
    # Degradierender Vertrag: None-Eingaben sind kein Fehler, nur "nicht aktuell".
    g = _graph(tmp_path)
    idx = EmbeddingIndex.build(g, HashEncoder(dim=64))
    assert pruefe_aktualitaet(None, g) is False
    assert pruefe_aktualitaet(idx, None) is False
