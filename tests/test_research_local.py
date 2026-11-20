"""The local entry point never starts a model; all service I/O here is fake."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import research_local as cli


PROPS = {"model_alias": "local-model", "total_slots": 1,
         "default_generation_settings": {"n_ctx": 16384}}
RESULT = {"status": "complete", "model": "local-model", "allow_research": True,
          "suggestions": [{"start": 0, "end": 24, "satz": "Ein rechtlicher Vorgang.", "normen": [{
              "id": "norm:AO:42", "zitat": "§ 42 AO", "titel": "Missbrauch",
              "research": {"reason": "Die Norm erklärt den Vorgang.",
                           "source_quote": {"start": 4, "end": 15, "text": "rechtlicher"},
                           "norm_quote": {"start": 31, "end": 42, "text": "Originalnorm"}}}]}],
          "unresolved": ["Ein Umstand bleibt offen."], "errors": [],
          "timing": {"elapsed_seconds": 1.25}, "usage": {},
          "trace": [{"reasoning_content": "PRIVATE_THINKING", "request": "PRIVATE_HTTP"}],
          "evidence_aliases": {"e1": "FULL_PRIVATE_EVIDENCE_ID"}}


class Opener:
    def __init__(self, props=None):
        self.raw = json.dumps(PROPS if props is None else props).encode()
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return io.BytesIO(self.raw)


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(corpora=[], agents=[], result=deepcopy(RESULT), props=deepcopy(PROPS), events=[])

    class Corpus:
        def __init__(self, graph, index, *, retrieval='bm25', encoder_model_path=None, catalog_path=None):
            self.paths, self.closed = (graph, index), False
            self.retrieval, self.encoder_model_path, self.prepared = retrieval, encoder_model_path, False
            self.catalog_path = catalog_path
            state.corpora.append(self)

        def identity_metadata(self):
            return {"graph_fingerprint": "original-graph"}

        def close(self):
            self.closed = True

        def prepare(self):
            state.events.append('prepare')
            self.prepared = True
            if getattr(state, 'prepare_output', None):
                print(state.prepare_output)

        def retrieval_metadata(self):
            return {'mode': self.retrieval, 'model_path': str(self.encoder_model_path) if self.encoder_model_path else None}

        @property
        def retrieval_metrics(self):
            return {'encode_count': int(self.prepared) + int(hasattr(state, 'invocation'))} if self.retrieval == 'hybrid' else {}

    class Agent:
        def __init__(self, url, model, config):
            self.base_url, self.model, self.config = url, model, config
            self._opener = Opener(state.props)
            state.agents.append(self)

        def run(self, text, corpus, allow_research):
            assert corpus.prepared
            state.events.append('run')
            state.invocation = (text, corpus, allow_research)
            return state.result

    monkeypatch.setattr(cli, "_runtime", lambda: (Agent, lambda **kw: SimpleNamespace(context_tokens=16384, **kw),
                                                 Corpus, json.loads))
    return state


def argv(tmp_path, *extra):
    text = tmp_path / "fall.txt"
    text.write_text("Ein rechtlicher Vorgang.\n", encoding="utf-8")
    return [str(text), "--graph", str(tmp_path / "graph.sqlite"), "--index", str(tmp_path / "index"),
            "--url", "http://127.0.0.1:8089", "--model", "local-model", *extra]


def test_text_output_contains_proposals_reasons_original_quotes_without_private_dialog(tmp_path, capsys, runtime):
    assert cli.main(argv(tmp_path)) == 0
    output = capsys.readouterr().out
    assert "Vorschlag: § 42 AO" in output and "Fokus [0:24]" in output
    assert "Die Norm erklärt den Vorgang." in output and "„Originalnorm“" in output
    assert "Sachverhaltszitat [4:15]" in output and "Offen: Ein Umstand" in output
    assert "Dauer: 1.25 s" in output and "PRIVATE" not in output
    assert runtime.corpora[0].closed and runtime.invocation[2] is True
    request, timeout = runtime.agents[0]._opener.requests[0]
    assert request.full_url.endswith("/props") and request.method == "GET" and timeout == 15


def test_json_output_is_single_object_and_static_has_no_research(tmp_path, capsys, runtime):
    assert cli.main(argv(tmp_path, "--json", "--static", "--timeout", "75")) == 0
    raw = capsys.readouterr().out
    output = json.loads(raw)
    assert output["suggestions"] == RESULT["suggestions"]
    assert "trace" not in output and "evidence_aliases" not in output and "PRIVATE" not in raw
    assert runtime.invocation[2] is False and runtime.agents[0].config.timeout_seconds == 75


def test_catalog_defaults_to_disabled_without_additional_output_metadata(tmp_path, capsys, runtime):
    assert cli.main(argv(tmp_path, "--json")) == 0
    output = json.loads(capsys.readouterr().out)
    assert runtime.corpora[0].catalog_path is None
    assert "catalog_metadata" not in output and "catalog_hints" not in output


def test_catalog_path_is_forwarded_and_metadata_visible_but_hints_stay_in_private_trace(tmp_path, capsys, runtime):
    catalog, trace = tmp_path / "catalog.yaml", tmp_path / "catalog-trace.json"
    runtime.result["catalog_metadata"] = {"format": "research_catalog/1", "path": str(catalog),
                                          "sha256": "a" * 64, "fingerprint": "b" * 64}
    runtime.result["catalog_hints"] = {"status": "unverified_hint", "concepts": ["PRIVATE_CATALOG_HINT"]}
    runtime.result["catalog_query_vector"] = [1., 0.]
    assert cli.main(argv(tmp_path, "--catalog", str(catalog), "--json", "--trace", str(trace))) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert runtime.corpora[0].catalog_path == catalog
    assert output["catalog_metadata"] == runtime.result["catalog_metadata"]
    assert "catalog_hints" not in output and "catalog_query_vector" not in output and "PRIVATE" not in captured.out
    private = json.loads(trace.read_text(encoding="utf-8"))["result"]
    assert private["catalog_hints"] == runtime.result["catalog_hints"]
    assert private["catalog_metadata"] == output["catalog_metadata"]
    assert private["catalog_query_vector"] == [1., 0.]


def test_catalog_text_output_marks_hints_as_unverified(tmp_path, capsys, runtime):
    runtime.result["catalog_metadata"] = {"format": "research_catalog/1", "fingerprint": "b" * 64}
    assert cli.main(argv(tmp_path, "--catalog", str(tmp_path / "catalog.yaml"))) == 0
    assert "Kataloghinweise: aktiviert (ungeprüfte Suchhinweise; keine Belege)" in capsys.readouterr().out


def test_execution_policy_defaults_preserve_existing_time_control(tmp_path, capsys, runtime):
    assert cli.main(argv(tmp_path, "--json")) == 0
    output = json.loads(capsys.readouterr().out)
    config = runtime.agents[0].config
    assert config.reuse_prompt_cache is False and config.finish_reserve_seconds == 0.
    assert output["execution_policy"] == {"reuse_prompt_cache": False, "finish_reserve_seconds": 0.}
    text = cli.render_text(output)
    assert "bisherige Zeitsteuerung (Abschlussmodus unter 30 s Restzeit)" in text
    assert "Abschlussreserve 0 s" not in text


def test_execution_flags_reach_config_and_json_and_private_trace(tmp_path, capsys, runtime):
    trace = tmp_path / "execution.json"
    assert cli.main(argv(tmp_path, "--reuse-prompt-cache", "--finish-reserve", "37.5", "--timeout", "180",
                         "--json", "--trace", str(trace))) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    config = runtime.agents[0].config
    assert config.reuse_prompt_cache is True and config.finish_reserve_seconds == 37.5
    assert config.timeout_seconds == 180.
    expected = {"reuse_prompt_cache": True, "finish_reserve_seconds": 37.5}
    assert output["execution_policy"] == expected
    private = json.loads(trace.read_text(encoding="utf-8"))["result"]
    assert private["execution_policy"] == expected and private["trace"] == RESULT["trace"]
    assert "PRIVATE" not in captured.out


def test_execution_flags_are_visible_in_text_without_claiming_a_cache_hit(tmp_path, capsys, runtime):
    assert cli.main(argv(tmp_path, "--reuse-prompt-cache", "--finish-reserve", "45")) == 0
    output = capsys.readouterr().out
    assert "Promptcache-Wiederverwendung aktiviert" in output
    assert "Reserve je weiterer Runde und Abschluss 45 s" in output
    assert "Zulassung prüft mehr als 90 s Restzeit" in output
    assert "PRIVATE" not in output


def test_utf8_stdin_and_explicit_private_trace(tmp_path, capsys, runtime, monkeypatch):
    args = argv(tmp_path, "--json", "--trace", str(tmp_path / "private.json"))
    args[0] = "-"
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO("Äußerung\n".encode())))
    assert cli.main(args) == 0
    assert runtime.invocation[0] == "Äußerung\n"
    output = json.loads((tmp_path / "private.json").read_text(encoding="utf-8"))
    assert {k: output['result'][k] for k in RESULT} == RESULT and output["input"] == "Äußerung\n"
    assert output["server"]["props"] == PROPS and output["corpus"]["graph_fingerprint"] == "original-graph"
    assert "PRIVATE" not in capsys.readouterr().out
    assert not list(tmp_path.glob(".research-*.tmp"))


def test_failed_agent_returns_one_and_saves_full_diagnostic(tmp_path, capsys, runtime):
    runtime.result.update(status="context_limit", suggestions=[], errors=[{"kind": "context_limit", "message": "Zu lang"}])
    path = tmp_path / "failed.json"
    assert cli.main(argv(tmp_path, "--json", "--trace", str(path))) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "context_limit"
    assert json.loads(path.read_text(encoding="utf-8"))["result"]["trace"] == RESULT["trace"]
    assert runtime.corpora[0].closed


def test_existing_trace_rejected_before_runtime_and_never_overwritten(tmp_path, capsys, runtime):
    path = tmp_path / "private.json"
    path.write_text("existing", encoding="utf-8")
    assert cli.main(argv(tmp_path, "--json", "--trace", str(path))) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert path.read_text() == "existing" and runtime.agents == []


def test_concurrent_trace_creation_is_not_overwritten_and_temp_removed(tmp_path, monkeypatch):
    path = tmp_path / "trace.json"
    real_link = cli.os.link

    def race(source, target):
        Path(target).write_text("other writer", encoding="utf-8")
        return real_link(source, target)

    monkeypatch.setattr(cli.os, "link", race)
    with pytest.raises(FileExistsError):
        cli.write_fresh_json(path, {"new": "data"})
    assert path.read_text() == "other writer"
    assert not list(tmp_path.glob(".research-*.tmp"))


@pytest.mark.parametrize("change", [{"model_alias": "wrong"}, {"total_slots": 2}, {"total_slots": True},
                                  {"default_generation_settings": {"n_ctx": 8192}},
                                  {"default_generation_settings": {"n_ctx": True}}])
def test_incompatible_server_never_runs_agent_and_closes_corpus(tmp_path, capsys, runtime, change):
    runtime.props.update(change)
    assert cli.main(argv(tmp_path, "--json")) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert not hasattr(runtime, "invocation") and runtime.corpora[0].closed


def test_larger_server_context_is_accepted(tmp_path, capsys, runtime):
    runtime.props["default_generation_settings"]["n_ctx"] = 32768
    assert cli.main(argv(tmp_path, "--json")) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_hybrid_prepares_before_run_and_reports_setup_and_final_metrics(tmp_path, capsys, runtime):
    model = tmp_path / 'local-model'
    trace = tmp_path / 'hybrid-trace.json'
    runtime.prepare_output = 'Encoder loader diagnostic'
    assert cli.main(argv(tmp_path, '--retrieval', 'hybrid', '--encoder-model', str(model), '--json', '--trace', str(trace))) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert runtime.events == ['prepare', 'run']
    assert runtime.corpora[0].encoder_model_path == model and runtime.corpora[0].retrieval == 'hybrid'
    assert result['retrieval_metadata'] == {'mode': 'hybrid', 'model_path': str(model)}
    assert result['retrieval_metrics_before_run']['encode_count'] == 1
    assert result['retrieval_metrics']['encode_count'] == 2
    assert result['source_prepare_seconds'] > 0 and result['timing']['elapsed_seconds'] == 1.25
    assert 'Encoder loader diagnostic' in captured.err and 'Encoder loader diagnostic' not in captured.out
    private = json.loads(trace.read_text(encoding='utf-8'))['result']
    assert private['retrieval_metrics'] == result['retrieval_metrics']


def test_hybrid_default_uses_existing_local_cache_and_text_reports_preparation(tmp_path, capsys, runtime):
    assert cli.main(argv(tmp_path, '--retrieval', 'hybrid')) == 0
    output = capsys.readouterr().out
    assert 'Quellenzugang: hybrid' in output and 'Quellenvorbereitung zusätzlich:' in output
    assert runtime.corpora[0].encoder_model_path is None


def test_bm25_encoder_argument_is_rejected_before_runtime(tmp_path, capsys, runtime):
    assert cli.main(argv(tmp_path, '--encoder-model', str(tmp_path), '--json')) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'error' and not runtime.agents


@pytest.mark.parametrize("data", [b"\xff", b" \n"])
def test_invalid_or_empty_utf8_fails_before_runtime(tmp_path, capsys, runtime, data):
    args = argv(tmp_path, "--json")
    Path(args[0]).write_bytes(data)
    assert cli.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error" and not runtime.agents


def test_direct_cli_help_works_outside_repository_without_loading_runtime(tmp_path):
    script = Path(cli.__file__).resolve()
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0 and "--trace" in result.stdout and "--static" in result.stdout
    assert "--reuse-prompt-cache" in result.stdout and "--finish-reserve" in result.stdout
    assert "--catalog" in result.stdout and "ungeprüfte Suchhinweise" in result.stdout
    help_text = " ".join(result.stdout.split())
    assert "0 = bisherige Zeitsteuerung" in help_text and "mehr als 2 * Wert Restzeit" in help_text


def test_direct_cli_rejects_nonlocal_url_without_attempting_server_or_model(tmp_path):
    args = argv(tmp_path, "--json")
    args[args.index("--url") + 1] = "https://example.com"
    result = subprocess.run([sys.executable, str(Path(cli.__file__).resolve()), *args], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 1 and json.loads(result.stdout)["status"] == "error"
    assert not result.stderr
