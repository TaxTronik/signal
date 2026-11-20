"""Explicit semantic profiles and resources through the public CLI, without models."""

from types import SimpleNamespace

import pytest

from risk_layer import cli, embedding, reranker, triggerindex


@pytest.mark.parametrize("profile", ["standard", "normpaket", "kontext", "kaskade", "praezision", "dokument", "treffer"])
def test_parser_accepts_declared_profiles(profile):
    args = cli.build_parser().parse_args(["analyse", "case.txt", "--semantik-profil", profile])
    assert args.semantik_profil == profile


def test_parser_preserves_default_profile_and_resources():
    args = cli.build_parser().parse_args(["analyse", "case.txt"])
    assert args.semantik_profil == "standard"
    assert args.rechtsgebiet == []
    assert args.embedding_dir is None
    assert args.reranker is None


@pytest.mark.parametrize("extra", [
    ["--semantik-profil", "unknown"],
    ["--semantik-profil", "TREFFER"],
    ["--rechtsgebiet", "norm:AO:42"],
    ["--rechtsgebiet", "AO 1977"],
    ["--rechtsgebiet", ""],
    ["--rechtsgebiet", "A" * 65],
])
def test_parser_rejects_invalid_profile_and_law_ids(extra):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["analyse", "case.txt", *extra])
    assert error.value.code == 2


@pytest.mark.parametrize("profile", ["package20", "package50", "coherent_package50"])
def test_cli_rejects_internal_research_profiles_before_loading(monkeypatch, profile):
    from risk_layer.semantic_search import PROFILE_CONFIG

    monkeypatch.setitem(PROFILE_CONFIG, profile, PROFILE_CONFIG["treffer"])

    def forbidden(*args, **kwargs):
        pytest.fail("Research profile names must fail before resource loading")

    monkeypatch.setattr(cli, "lade_katalog", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["analyse", "not-read.txt", "--semantik-profil", profile])
    assert error.value.code == 2


def _analysis_stubs(monkeypatch, tmp_path):
    calls = {}
    index = SimpleNamespace(encoder_name="fixture-encoder")
    encoder = object()
    ranker = object()
    monkeypatch.setenv("RISK_LAYER_EMB_DEVICE", "cuda")
    monkeypatch.setattr(cli, "lade_katalog", lambda _: object())
    monkeypatch.setattr(cli, "graph_oeffnen", lambda *a: object())
    monkeypatch.setattr(triggerindex, "index_oeffnen", lambda: None)
    monkeypatch.setattr(embedding, "pruefe_aktualitaet", lambda *a: True)

    def open_index(*args):
        calls["index_args"] = args
        return index

    def open_encoder(name, **kwargs):
        calls["encoder"] = (name, kwargs)
        return encoder

    def open_reranker(*args, **kwargs):
        calls["reranker"] = (args, kwargs)
        return ranker

    def run(self, text, **kwargs):
        calls["text"] = text
        calls["run"] = kwargs
        return SimpleNamespace(luecken=[])

    monkeypatch.setattr(embedding, "index_oeffnen", open_index)
    monkeypatch.setattr(embedding, "encoder_oeffnen", open_encoder)
    monkeypatch.setattr(reranker, "reranker_oeffnen", open_reranker)
    monkeypatch.setattr(cli.Pipeline, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(cli.Pipeline, "run", run)
    monkeypatch.setattr(cli, "report_text", lambda _: "analysis complete")
    source = tmp_path / "case.txt"
    source.write_text("Die Buchfuehrung ist vorhanden.", encoding="utf-8")
    return source, calls, index, encoder, ranker


@pytest.mark.parametrize("profile", ["kaskade", "treffer"])
def test_analysis_forwards_explicit_profile_scope_index_and_reranker(monkeypatch, tmp_path, profile):
    source, calls, index, encoder, ranker = _analysis_stubs(monkeypatch, tmp_path)
    directory = str(tmp_path / "separate-passage-index")
    assert cli.main([
        "analyse", str(source), "--no-schicht2", "--semantik-profil", profile,
        "--rechtsgebiet", "EStG", "--rechtsgebiet", "AO", "--embedding-dir", directory,
        "--reranker", "bge-qwen3-reranker-4b",
    ]) == 0
    options = calls["run"]["optionen"]
    assert options.semantik_profil == profile
    assert options.rechtsgebiete == ("EStG", "AO")
    assert calls["index_args"] == (directory,)
    assert calls["reranker"] == (("bge-qwen3-reranker-4b",), {"device": "cuda"})
    assert calls["run"]["embedding"] is index
    assert calls["run"]["embedding_encoder"] is encoder
    assert calls["run"]["reranker"] is ranker


def test_analysis_defaults_keep_legacy_factory_calls(monkeypatch, tmp_path):
    source, calls, *_ = _analysis_stubs(monkeypatch, tmp_path)
    assert cli.main(["analyse", str(source), "--no-schicht2"]) == 0
    options = calls["run"]["optionen"]
    assert options.semantik_profil == "standard" and options.rechtsgebiete == ()
    assert calls["index_args"] == ()
    assert calls["reranker"] == ((), {"device": "cuda"})


@pytest.mark.parametrize("extra", [
    ["--rechtsgebiet", "AO"],
    ["--semantik-profil", "normpaket", *(["--rechtsgebiet", "AO"] * 33)],
])
def test_invalid_scope_combination_fails_before_loading_resources(monkeypatch, capsys, extra):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid options must not load models or the catalog")

    monkeypatch.setattr(cli, "lade_katalog", forbidden)
    assert cli.main(["analyse", "not-read.txt", *extra]) == 2
    assert "Optionsfehler" in capsys.readouterr().err


@pytest.mark.parametrize("profile", ["standard", "treffer"])
def test_explicit_reranker_load_failure_is_reported(monkeypatch, tmp_path, capsys, profile):
    source, calls, *_ = _analysis_stubs(monkeypatch, tmp_path)

    def unavailable(*args, **kwargs):
        raise RuntimeError("fixture model unavailable")

    monkeypatch.setattr(reranker, "reranker_oeffnen", unavailable)
    assert cli.main([
        "analyse", str(source), "--semantik-profil", profile,
        "--reranker", "bge-qwen3-reranker-4b",
    ]) == 2
    assert "run" not in calls
    assert "fixture model unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("profile", ["kontext", "treffer"])
def test_profile_resource_failure_has_clean_cli_error(monkeypatch, tmp_path, capsys, profile):
    from risk_layer.semantic_search import SemanticResourcesUnavailable

    source, _, *_ = _analysis_stubs(monkeypatch, tmp_path)

    def unavailable(*args, **kwargs):
        raise SemanticResourcesUnavailable("fixture passage index unavailable")

    monkeypatch.setattr(cli.Pipeline, "run", unavailable)
    assert cli.main([
        "analyse", str(source), "--no-schicht2", "--semantik-profil", profile,
    ]) == 2
    assert "fixture passage index unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("corrupt", [False, True])
def test_explicit_unreadable_index_stops_default_analysis(monkeypatch, tmp_path, capsys, corrupt):
    source, calls, *_ = _analysis_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(embedding, "index_oeffnen", embedding.EmbeddingIndex.load)
    if corrupt:
        (tmp_path / "meta.json").write_text("{", encoding="utf-8")
    assert cli.main(["analyse", str(source), "--embedding-dir", str(tmp_path)]) == 2
    assert "run" not in calls
    assert "Embedding-Index" in capsys.readouterr().err


def test_explicit_index_requires_its_encoder(monkeypatch, tmp_path, capsys):
    source, calls, *_ = _analysis_stubs(monkeypatch, tmp_path)

    def unavailable(*args, **kwargs):
        raise ImportError("fixture encoder unavailable")

    monkeypatch.setattr(embedding, "encoder_oeffnen", unavailable)
    assert cli.main(["analyse", str(source), "--embedding-dir", str(tmp_path)]) == 2
    assert "run" not in calls
    assert "fixture encoder unavailable" in capsys.readouterr().err


def test_explicit_reranker_requires_gpu_but_can_be_disabled_on_cpu(monkeypatch, tmp_path, capsys):
    source, calls, *_ = _analysis_stubs(monkeypatch, tmp_path)
    monkeypatch.setenv("RISK_LAYER_EMB_DEVICE", "cpu")
    assert cli.main(["analyse", str(source), "--reranker", "bge-reranker-v2-m3"]) == 2
    assert "run" not in calls
    assert "Reranker" in capsys.readouterr().err
    assert cli.main(["analyse", str(source), "--reranker", "aus"]) == 0
    assert calls["run"]["reranker"] is None


@pytest.mark.parametrize("passages", [False, True])
def test_embedding_build_enables_passages_only_when_requested(monkeypatch, tmp_path, passages):
    from risk_layer import graph

    calls = {}
    fake_graph, fake_encoder, catalog = object(), SimpleNamespace(name="hash"), object()
    monkeypatch.setattr(graph, "graph_oeffnen", lambda _: fake_graph)
    monkeypatch.setattr(embedding, "encoder_oeffnen", lambda *a, **kw: fake_encoder)
    monkeypatch.setattr(cli, "lade_katalog", lambda _: catalog)

    def save(path):
        calls["save"] = path

    def build(g, enc, *, katalog, **kwargs):
        assert g is fake_graph and enc is fake_encoder and katalog is catalog
        calls["options"] = kwargs
        return SimpleNamespace(save=save, ids=["norm:AO:1"], dim=8)

    monkeypatch.setattr(embedding.EmbeddingIndex, "build", staticmethod(build))
    target = str(tmp_path / "new-index")
    args = ["embedding", "build", "--encoder", "hash", "--dir", target]
    if passages:
        args += ["--passagen"]
    assert cli.main(args) == 0
    assert calls["options"] == ({"passagen": True} if passages else {})
    assert calls["save"] == target
