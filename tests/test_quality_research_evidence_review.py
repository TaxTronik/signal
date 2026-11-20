"""No models, tokenizer, encoder or HTTP; exercise the real transport guard."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest


TEST_SOURCE = Path(__file__).resolve()
if TEST_SOURCE.name == "test_review_probe.py":
    SCRIPT = TEST_SOURCE.with_name("review_probe.py")
elif TEST_SOURCE.name == "test_quality_research_evidence_review.py":
    SCRIPT = TEST_SOURCE.parents[1] / "tools/quality_research_evidence_review.py"
else:
    raise ValueError("Unexpected review test location")
spec = importlib.util.spec_from_file_location("evidence_review_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def sample():
    text = "Es gibt einen Sachverhalt. Weiterer ausdrücklich benannter Kontext."
    focuses = [
        {"id": "f0", "start": 0, "end": 24, "text": text[:24]},
        {"id": "f1", "start": 26, "end": len(text), "text": text[26:]},
    ]
    identity = {
        "source_fingerprint": "source",
        "graph_fingerprint": "graph",
        "index_fingerprint": "index",
        "citation_fingerprint": "citations",
    }
    norms = {
        "norm:TEST:1": {
            "titel": "Originaltitel",
            "text": "Allgemeine erste Originalregel. " * 24,
        },
        "norm:TEST:2": {
            "titel": "Zweite Quelle",
            "text": "Diese andere Originalnorm darf nicht untergeschoben werden.",
        },
    }
    pages = []
    for nid, norm in norms.items():
        start, end = 0, len(norm["text"])
        eid = "ev:" + probe.sha(
            probe._json(
                [identity, probe.sha(text), nid, probe.sha(norm["text"]), start, end]
            )
        )
        pages.append(
            {
                "evidence_id": eid,
                "norm_id": nid,
                "requested_id": nid,
                "fallback": False,
                "titel": norm["titel"],
                "text": norm["text"],
                "source_start": start,
                "source_end": end,
                "source_length": end,
                "text_sha256": probe.sha(norm["text"]),
                "source_fingerprint": "source",
                "graph_fingerprint": "graph",
                "next_cursor": None,
                "complete": True,
            }
        )
    suggestions = []
    for i, page in enumerate(pages):
        focus = focuses[i]
        norm = {
            "id": page["norm_id"],
            "research": {
                "evidence_id": page["evidence_id"],
                "reason": "Bisherige Endbegründung, kein Thinking.",
                "source_quote": {
                    "start": focus["start"],
                    "end": focus["end"],
                    "text": focus["text"],
                },
                "norm_quote": {"start": 0, "end": 15, "text": page["text"][:15]},
            },
        }
        suggestions.append(
            {
                "start": focus["start"],
                "end": focus["end"],
                "satz": focus["text"],
                "normen": [norm],
            }
        )
    result = {
        "status": "complete",
        "initial_sources": [pages[0]],
        "initial_sources_sha256": probe.sha(probe._json([pages[0]])),
        "focuses": focuses,
        "suggestions": suggestions,
        "catalog_hints": {"secret": "CATALOG_SENTINEL"},
        "trace": [
            {"kind": "http", "reasoning_content": "THINKING_SENTINEL"},
            {"kind": "tool", "name": "read_source", "result": {"source": pages[1]}},
            {
                "kind": "tool",
                "name": "finish",
                "result": {"accepted": True, "suggestions": deepcopy(suggestions)},
            },
        ],
    }
    row = {
        "case_id": "test:1",
        "query": text,
        "targets": ["GOLD_SENTINEL"],
        "negative": False,
        "baseline": {"secret": "BASELINE_SENTINEL"},
        "arms": {"research": {"result": result}},
    }
    return row, identity, norms


def prepared(sample):
    row, identity, norms = sample
    return probe.prepare_case(row, "research", identity, norms.get)


def answers(data):
    return {
        "decisions": [
            {
                "candidate_id": c["candidate_id"],
                "action": "keep",
                "quote_ids": [
                    next(
                        qid
                        for qid, q in data["quotes"].items()
                        if q["norm_id"] == c["norm_id"]
                    )
                ],
                "source_quote_id": next(
                    qid
                    for qid, q in data["case_quotes"].items()
                    if q["focus_id"] == c["focus_id"]
                ),
                "reason": "Der Originaltext begründet Prüfbedarf; weitere Voraussetzungen bleiben offen.",
            }
            for c in data["candidates"]
        ]
    }


def test_payload_only_allowlisted_inputs(sample):
    data = prepared(sample)
    serialized = json.dumps(data["payload"])
    for sentinel in (
        "GOLD_SENTINEL",
        "BASELINE_SENTINEL",
        "CATALOG_SENTINEL",
        "THINKING_SENTINEL",
    ):
        assert sentinel not in serialized
    assert set(data["payload"]) == {
        "SACHVERHALT",
        "FOKUSSTELLEN",
        "KANDIDATEN",
        "ORIGINALQUELLEN",
    }
    assert len(data["payload"]["ORIGINALQUELLEN"]) == 2
    assert "evidence_id" not in data["payload"]["KANDIDATEN"][0]


@pytest.mark.parametrize(
    "fault",
    [
        "offset",
        "quote",
        "foreign",
        "page_identity",
        "source_hash",
        "duplicate",
        "finish",
        "initial_hash",
    ],
)
def test_corrupted_originals_fail_closed(sample, fault):
    row, identity, norms = sample
    result = row["arms"]["research"]["result"]
    if fault == "offset":
        result["focuses"][0]["start"] = -1
    if fault == "quote":
        result["suggestions"][0]["normen"][0]["research"]["norm_quote"]["text"] = (
            "Invented passage"
        )
    if fault == "foreign":
        result["suggestions"][0]["normen"][0]["id"] = "norm:TEST:999"
    if fault == "page_identity":
        result["trace"][1]["result"]["source"]["evidence_id"] = "ev:unread"
    if fault == "source_hash":
        norms["norm:TEST:2"]["text"] += "changed"
    if fault == "duplicate":
        result["suggestions"][0]["normen"] *= 2
    if fault == "finish":
        result["trace"][-1]["result"]["accepted"] = False
    if fault == "initial_hash":
        result["initial_sources_sha256"] = "wrong"
    # Keep final output binding intact except for the explicit finish fault.
    result["trace"][-1]["result"]["suggestions"] = deepcopy(result["suggestions"])
    with pytest.raises(ValueError):
        probe.prepare_case(row, "research", identity, norms.get)


def test_keep_adjacent_original_segments_and_context_flag(sample):
    data = prepared(sample)
    args = answers(data)
    qids = [qid for qid, q in data["quotes"].items() if q["norm_id"] == "norm:TEST:1"]
    args["decisions"][0]["quote_ids"] = qids[:2]
    args["decisions"][0]["source_quote_id"] = "f1q1"
    result = probe.validate_review(args, data)
    first = result[0]
    quote = first["norm_quote"]
    assert (
        sample[2]["norm:TEST:1"]["text"][quote["start"] : quote["end"]] == quote["text"]
    )
    assert first["context_quote"] is True and first["focus_id"] == "f0"
    assert first["source_quote_focus_id"] == "f1"
    assert first["legal_validity_verified"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "new_candidate",
        "new_norm",
        "unread_quote",
        "foreign_quote",
        "unordered",
        "empty_keep",
        "bad_reject",
        "long_reason",
    ],
)
def test_decisions_are_complete_and_cannot_change_identity(sample, fault):
    data = prepared(sample)
    args = answers(data)
    d = args["decisions"][0]
    if fault == "missing":
        args["decisions"].pop()
    if fault == "duplicate":
        args["decisions"][1] = deepcopy(d)
    if fault == "new_candidate":
        d["candidate_id"] = "c999"
    if fault == "new_norm":
        d["norm_id"] = "norm:TEST:999"
    if fault == "unread_quote":
        d["quote_ids"] = ["e999q1"]
    if fault == "foreign_quote":
        d["quote_ids"] = args["decisions"][1]["quote_ids"]
    if fault == "unordered":
        d["quote_ids"] = ["e1q2", "e1q1"]
    if fault == "empty_keep":
        d["quote_ids"] = []
    if fault == "bad_reject":
        d["action"] = "reject"
    if fault == "long_reason":
        d["reason"] = "x" * 601
    with pytest.raises(ValueError):
        probe.validate_review(args, data)


def test_reject_has_no_fake_evidence(sample):
    data = prepared(sample)
    args = answers(data)
    args["decisions"][0].update(
        action="reject",
        quote_ids=[],
        source_quote_id="",
        reason="Gelesene Quelle trägt diese Zuordnung nicht.",
    )
    result = probe.validate_review(args, data)
    assert "evidence_id" not in result[0]
    assert len(result) == 2


class StubAgent(probe.ResearchAgent):
    def __init__(self, reply, mutate=None, tokens=100):
        super().__init__(
            "http://127.0.0.1:8999",
            "stub",
            probe.ResearchConfig(reuse_prompt_cache=True),
        )
        self.reply, self.mutate, self.tokens, self.requests = reply, mutate, tokens, []

    def _post(self, path, payload, deadline, result):
        self.requests.append((path, deepcopy(payload)))
        if path == "/apply-template":
            return {"prompt": "native rendered prompt"}
        if path == "/tokenize":
            return {"tokens": [1] * self.tokens}
        assert path == "/v1/chat/completions"
        response = {
            "model": "stub",
            "usage": {
                "prompt_tokens": self.tokens,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call1",
                                "type": "function",
                                "function": {
                                    "name": "review",
                                    "arguments": json.dumps(self.reply),
                                },
                            }
                        ],
                    },
                }
            ],
        }
        if self.mutate:
            self.mutate(response)
        return response


def test_real_completion_guard_and_fresh_context(sample):
    data = prepared(sample)
    agent = StubAgent(answers(data))
    for _ in range(2):
        result = probe.run_review(agent, data)
        assert result["status"] == "complete"
        assert result["usage"]["model_calls"] == 1
    for path, payload in agent.requests:
        if path != "/tokenize":
            assert len(payload["messages"]) == 2
            assert payload["cache_prompt"] is False
            assert payload["tool_choice"]["function"]["name"] == "review"


@pytest.mark.parametrize(
    "fault,status",
    [
        ("model", "invalid_completion"),
        ("count", "invalid_completion"),
        ("length", "output_limit"),
        ("excess", "invalid_completion"),
        ("cache", "invalid_completion"),
        ("missing_cache", "invalid_completion"),
        ("truncated", "invalid_completion"),
        ("partial", "invalid_review"),
    ],
)
def test_runtime_failures_leave_no_partial_review(sample, fault, status):
    data = prepared(sample)

    def mutate(r):
        if fault == "model":
            r["model"] = "wrong"
        if fault == "count":
            r["usage"]["prompt_tokens"] += 1
        if fault == "length":
            r["choices"][0]["finish_reason"] = "length"
        if fault == "excess":
            r["usage"]["completion_tokens"] = 9999
        if fault == "cache":
            r["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
        if fault == "missing_cache":
            r["usage"].pop("prompt_tokens_details")
        if fault == "truncated":
            r["truncated"] = True
        if fault == "partial":
            r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
                '{"decisions":[]}'
            )

    result = probe.run_review(StubAgent(answers(data), mutate), data)
    assert result["status"] == status
    assert result["decisions"] == []


def test_context_limit_precedes_generation(sample):
    data = prepared(sample)
    agent = StubAgent(answers(data), tokens=16000)
    result = probe.run_review(agent, data)
    assert result["status"] == "context_limit"
    assert [x[0] for x in agent.requests] == ["/apply-template", "/tokenize"]


def test_late_answer_is_failure(sample, monkeypatch):
    data = prepared(sample)
    ticks = iter([0.0, 181.0, 181.0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    result = probe.run_review(StubAgent(answers(data)), data)
    assert result["status"] == "timeout" and result["decisions"] == []


def test_deadline_after_validation_discards_whole_review(sample, monkeypatch):
    data = prepared(sample)
    ticks = iter([0.0, 1.0, 181.0, 181.0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    result = probe.run_review(StubAgent(answers(data)), data)
    assert result["status"] == "timeout" and result["decisions"] == []


def test_current_case_cannot_borrow_other_case_evidence(sample):
    row, identity, norms = sample
    row["query"] += " Weiterer Satz."
    with pytest.raises(ValueError, match="Evidence identity"):
        probe.prepare_case(row, "research", identity, norms.get)


def test_non_candidate_norm_sources_are_excluded(sample):
    row, identity, norms = sample
    result = row["arms"]["research"]["result"]
    result["suggestions"].pop()
    result["trace"][-1]["result"]["suggestions"] = deepcopy(result["suggestions"])
    data = probe.prepare_case(row, "research", identity, norms.get)
    assert len(data["payload"]["ORIGINALQUELLEN"]) == 1
    assert all(q["norm_id"] == "norm:TEST:1" for q in data["quotes"].values())


def test_no_candidates_never_calls_model(sample):
    data = prepared(sample)
    data["candidates"] = []
    agent = StubAgent({"decisions": []})
    assert probe.run_review(agent, data)["status"] == "no_candidates"
    assert agent.requests == []


def test_cli_prepare_only_is_atomic_and_model_free(sample, tmp_path, monkeypatch):
    row, identity, norms = sample
    inputs = tmp_path / "input" / "inputs"
    inputs.mkdir(parents=True)
    graph = inputs / "graph.sqlite"
    con = sqlite3.connect(graph)
    con.execute("CREATE TABLE knoten (id TEXT, titel TEXT, text TEXT)")
    con.executemany(
        "INSERT INTO knoten VALUES (?,?,?)",
        [(nid, n["titel"], n["text"]) for nid, n in norms.items()],
    )
    con.commit()
    con.close()
    source = {
        "status": "complete",
        "rows": [row],
        "corpus": identity,
        "snapshot_hashes": {"inputs/graph.sqlite": probe.sha(graph.read_bytes())},
    }
    report = inputs.parent / "report.json"
    report.write_text(json.dumps(source), encoding="utf-8")
    raw = report.read_bytes()
    monkeypatch.setattr(
        probe,
        "ResearchAgent",
        lambda *a, **k: pytest.fail("Must not construct model client"),
    )
    out = tmp_path / "new"
    assert probe.main(["--report", str(report), "--out", str(out)]) == 0
    assert (
        json.loads((out / "report.json").read_text(encoding="utf-8"))["status"]
        == "prepared"
    )
    assert report.read_bytes() == raw
    with pytest.raises(ValueError):
        probe.main(["--report", str(report), "--out", str(out)])
    with pytest.raises(ValueError):
        probe.write_trace(out / "report.json", {"changed": True})


def test_direct_cli_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "--run" in result.stdout


def test_canonical_copy_runs_and_freezes_the_actual_canonical_tests(tmp_path):
    staging = tmp_path / "frozen"
    staging.mkdir()
    probe.freeze_sources(staging)
    root = staging / "source"
    script = root / "tools/quality_research_evidence_review.py"
    test = root / "tests/test_quality_research_evidence_review.py"
    test.parent.mkdir(exist_ok=True)
    shutil.copyfile(SCRIPT, script)
    shutil.copyfile(TEST_SOURCE, test)
    assert probe.repository_root(script) == root
    assert probe.test_source_path(script, root) == test
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "canonical-snapshot"
    code = (
        "import json,runpy,sys; from pathlib import Path; "
        "m=runpy.run_path(sys.argv[1],run_name='review_import'); "
        "print(json.dumps(m['freeze_sources'](Path(sys.argv[2]))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(script), str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    hashes = json.loads(result.stdout)
    assert hashes["tests/test_quality_research_evidence_review.py"] == probe.file_sha(
        test
    )
    assert hashes["tools/quality_research_evidence_review.py"] == probe.file_sha(script)
    assert "out/research-evidence-review-draft/test_review_probe.py" not in hashes
    assert probe.file_sha(
        output / "source/tests/test_quality_research_evidence_review.py"
    ) == probe.file_sha(test)


def test_root_validation_does_not_accept_a_nearby_unrelated_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="other"\n', encoding="utf-8"
    )
    script = tmp_path / "tools/quality_research_evidence_review.py"
    with pytest.raises(ValueError, match="not a complete Signal"):
        probe.repository_root(script)
    with pytest.raises(ValueError, match="actual review probe tests"):
        probe.test_source_path(script, tmp_path)


@pytest.fixture
def model_setup(tmp_path):
    base = tmp_path / "setup"
    base.mkdir()
    model = base / "model.gguf"
    model.write_bytes(b"tiny fake weights; never loaded")
    exe = base / "llama-server.exe"
    exe.write_bytes(b"tiny fake executable; never executed")
    model_meta = {
        "path": str(model),
        "bytes": model.stat().st_size,
        "sha256": probe.file_sha(model),
    }
    manifest = {"status": "complete", "model": model_meta}
    raw_model = json.dumps(manifest).encode()
    launch = {
        "status": "healthy_handoff",
        "model": model_meta,
        "model_manifest_sha256": probe.sha(raw_model),
        "alias": "stub",
        "base_url": "http://127.0.0.1:8999",
        "executable": str(exe),
        "engine_files": {
            exe.name: {"bytes": exe.stat().st_size, "sha256": probe.file_sha(exe)}
        },
    }
    raw_launch = json.dumps(launch).encode()
    props = {
        "model_path": str(model),
        "build_info": "stub-build",
        "chat_template": "stub-template",
        "model_ftype": "stub-quantization",
        "total_slots": 1,
        "default_generation_settings": {"n_ctx": 16384},
    }
    props_raw = json.dumps(props).encode()
    setup = {
        "status": "pass",
        "launch_manifest_sha256": probe.sha(raw_launch),
        "alias": "stub",
        "base_url": launch["base_url"],
        "artifact_hashes": {"props.response.json": probe.sha(props_raw)},
    }
    (base / "model-manifest.json").write_bytes(raw_model)
    (base / "launch-manifest.json").write_bytes(raw_launch)
    (base / "model-setup-report.json").write_text(json.dumps(setup), encoding="utf-8")
    (base / "setup-props.json").write_bytes(props_raw)
    return base, model, exe


def test_model_binding_checks_real_local_bytes_and_archives(model_setup, tmp_path):
    base, model, _ = model_setup
    out = tmp_path / "binding"
    out.mkdir()
    checked = probe.bind_model(
        base,
        StubAgent({}),
        {"props": json.loads((base / "setup-props.json").read_bytes())},
        out,
    )
    assert len(checked["files"]) == 2
    for name, digest in checked["manifest_hashes"].items():
        assert probe.file_sha(out / name) == digest


@pytest.mark.parametrize(
    "fault",
    [
        "weights",
        "binary",
        "chain",
        "server_path",
        "alias",
        "inventory",
        "build_info",
        "chat_template",
        "model_ftype",
        "context",
        "props_hash",
    ],
)
def test_bad_model_binding_cannot_run(model_setup, tmp_path, fault):
    base, model, exe = model_setup
    server = {"props": json.loads((base / "setup-props.json").read_bytes())}
    agent = StubAgent({})
    if fault == "weights":
        model.write_bytes(b"changed")
    if fault == "binary":
        exe.write_bytes(b"changed")
    if fault == "chain":
        (base / "launch-manifest.json").write_text("{}", encoding="utf-8")
    if fault == "server_path":
        server["props"]["model_path"] = str(base / "other.gguf")
    if fault == "alias":
        agent.model = "other"
    if fault in ("build_info", "chat_template", "model_ftype"):
        server["props"][fault] = "changed"
    if fault == "context":
        server["props"]["default_generation_settings"]["n_ctx"] = 32768
    if fault == "props_hash":
        (base / "setup-props.json").write_bytes(b"{}")
    if fault == "inventory":
        launch = json.loads((base / "launch-manifest.json").read_text(encoding="utf-8"))
        launch["engine_files"] = {}
        raw = json.dumps(launch).encode()
        (base / "launch-manifest.json").write_bytes(raw)
        setup = json.loads(
            (base / "model-setup-report.json").read_text(encoding="utf-8")
        )
        setup["launch_manifest_sha256"] = probe.sha(raw)
        (base / "model-setup-report.json").write_text(
            json.dumps(setup), encoding="utf-8"
        )
    out = tmp_path / "binding"
    out.mkdir()
    with pytest.raises(ValueError):
        probe.bind_model(base, agent, server, out)


def test_global_preflight_failure_retains_all_candidate_denominators(
    sample, tmp_path, monkeypatch
):
    data = prepared(sample)
    cases = [
        {
            "case_id": "one",
            "status": "prepared",
            "prepared": data,
            "original_candidate_count": 2,
        },
        {
            "case_id": "two",
            "status": "prepared",
            "prepared": {**data, "candidates": []},
            "original_candidate_count": 0,
        },
    ]
    report = tmp_path / "original.json"
    report.write_text("{}", encoding="utf-8")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "graph.sqlite").write_bytes(b"fake graph")
    monkeypatch.setattr(
        probe,
        "prepare_report",
        lambda *a: {
            "cases": cases,
            "original_report_sha256": probe.file_sha(report),
            "graph_sha256": probe.file_sha(inputs / "graph.sqlite"),
        },
    )
    monkeypatch.setattr(
        probe,
        "check_server",
        lambda *a: (_ for _ in ()).throw(ValueError("server mismatch")),
    )
    out = tmp_path / "failed"
    assert (
        probe.main(
            [
                "--report",
                str(report),
                "--out",
                str(out),
                "--run",
                "--url",
                "http://127.0.0.1:8999",
                "--model",
                "stub",
            ]
        )
        == 1
    )
    result = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert (
        result["status"] == "failed"
        and result["total_cases"] == 2
        and result["total_candidates"] == 2
    )
    assert len(result["cases"]) == 2 and result["reviewed_candidates"] == 0


def test_real_search_results_preserve_selected_and_alternative_returned_pages(tmp_path):
    from risk_layer.embedding import EmbeddingIndex, HashEncoder
    from risk_layer.gii import NormKnoten
    from risk_layer.graph import NormGraph
    from risk_layer.research_sources import ResearchCorpus

    graph_path, index_path = tmp_path / "graph.sqlite", tmp_path / "index"
    norm_text = "Eine besondere Belegpflicht bestimmt die Aufzeichnung. " * 100
    norm_text += "Die Geschäftsführerhaftung betrifft eine andere Pflicht. " * 100
    graph = NormGraph.build(
        [
            NormKnoten(
                id="norm:TEST:1", law="TEST", para="1", titel="Regel", text=norm_text
            )
        ],
        graph_path,
    )
    try:
        EmbeddingIndex.build(graph, HashEncoder(dim=16), passagen=True).save(index_path)
    finally:
        graph.close()
    corpus = ResearchCorpus(graph_path, index_path)
    try:
        text = "Eine Belegpflicht ist zu prüfen."
        session = corpus.new_session(text)
        initial = session.initial_sources(limit=1)
        searched = session.execute(
            "search_sources", {"query": "Geschäftsführerhaftung", "limit": 1}
        )
        page = searched["results"][0]
        assert page["evidence_id"] != initial[0]["evidence_id"]
        focus = session.focuses[0]
        suggestions = session.validate_finish(
            {
                "findings": [
                    {
                        "focus_id": focus["id"],
                        "norm_id": page["norm_id"],
                        "evidence_id": page["evidence_id"],
                        "source_quote": focus["text"],
                        "norm_quote": page["text"][:30],
                        "reason": "Nur ein mechanischer Test der Quellenzuordnung.",
                    }
                ],
                "unresolved": [],
            }
        )
        result = {
            "status": "complete",
            "focuses": session.focuses,
            "initial_sources": initial,
            "initial_sources_sha256": probe.sha(probe._json(initial)),
            "suggestions": suggestions,
            "trace": [
                {"kind": "tool", "name": "search_sources", "result": searched},
                {
                    "kind": "tool",
                    "name": "finish",
                    "result": {"accepted": True, "suggestions": suggestions},
                },
            ],
        }
        row = {"query": text, "arms": {"research": {"result": result}}}
        data = probe.prepare_case(
            row,
            "research",
            corpus.identity_metadata(),
            lambda nid: {"text": norm_text, "titel": "Regel"},
        )
        assert set(data["ledger"]) == {initial[0]["evidence_id"], page["evidence_id"]}
        assert len(data["candidates"]) == 1
    finally:
        corpus.close()


def test_full_additional_archive_has_all_original_candidates_and_all_returned_candidate_pages():
    path = probe.ROOT / "out/quality-research-moe-additional-20260908-r1/report.json"
    if not path.is_file():
        pytest.skip("Private completed Additional archive is not installed")
    original = json.loads(path.read_text(encoding="utf-8"))
    result = probe.prepare_report(path, "research")
    assert len(result["cases"]) == len(original["rows"]) == 30
    total_candidates, total_pages = 0, 0
    for case, row in zip(result["cases"], original["rows"]):
        assert case["case_id"] == row["case_id"] and case["status"] == "prepared"
        source = row["arms"]["research"]["result"]
        expected_count = sum(len(s["normen"]) for s in source["suggestions"])
        assert (
            case["original_candidate_count"]
            == len(case["prepared"]["candidates"])
            == expected_count
        )
        needed = {n["id"] for s in source["suggestions"] for n in s["normen"]}
        # Independent explicit collector: do not reuse the helper under test.
        returned = list(source["initial_sources"])
        for event in source["trace"]:
            if event.get("kind") != "tool":
                continue
            if event.get("name") == "search_sources":
                returned.extend(event.get("result", {}).get("results", []))
            if event.get("name") == "read_source" and isinstance(
                event.get("result", {}).get("source"), dict
            ):
                returned.append(event["result"]["source"])
        expected_pages = {
            p["evidence_id"]: p for p in returned if p["norm_id"] in needed
        }
        assert case["prepared"]["ledger"] == expected_pages
        total_candidates += expected_count
        total_pages += len(expected_pages)
    assert (total_candidates, total_pages) == (22, 28)


def test_invalid_source_does_not_remove_original_candidates_from_denominator(
    sample, tmp_path, monkeypatch
):
    row, identity, norms = sample
    source = tmp_path / "source"
    inputs = source / "inputs"
    inputs.mkdir(parents=True)
    graph = inputs / "graph.sqlite"
    con = sqlite3.connect(graph)
    con.execute("CREATE TABLE knoten(id TEXT,titel TEXT,text TEXT)")
    con.executemany(
        "INSERT INTO knoten VALUES(?,?,?)",
        [(nid, n["titel"], n["text"]) for nid, n in norms.items()],
    )
    con.commit()
    con.close()
    row["arms"]["research"]["result"]["initial_sources_sha256"] = "wrong"
    original = {
        "status": "complete",
        "rows": [row],
        "corpus": identity,
        "snapshot_hashes": {"inputs/graph.sqlite": probe.file_sha(graph)},
    }
    path = source / "report.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    prepared_report = probe.prepare_report(path, "research")
    assert prepared_report["cases"][0]["status"] == "invalid_input"
    assert prepared_report["cases"][0]["original_candidate_count"] == 2
    out = tmp_path / "failure"
    assert probe.main(["--report", str(path), "--out", str(out)]) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["total_candidates"] == 2 and report["known_original_candidates"] == 2
    assert (
        report["cases"][0]["candidate_count"] == 2
        and report["reviewed_candidates"] == 0
    )


def test_malformed_original_candidate_denominator_is_unavailable_not_zero():
    with pytest.raises(ValueError, match="denominator unavailable"):
        probe.original_candidate_count({"suggestions": [{"normen": None}]})


def test_native_quote_enums_are_separate_and_exhaustive(sample):
    data = prepared(sample)
    schema = probe.review_tool(data)["function"]["parameters"]
    fields = schema["properties"]["decisions"]["items"]["properties"]
    norm_ids = fields["quote_ids"]["items"]["enum"]
    case_ids = fields["source_quote_id"]["enum"]
    assert norm_ids == list(data["quotes"])
    assert case_ids == ["", *data["case_quotes"]]
    assert not set(norm_ids) & set(case_ids)
    assert "e1q1" not in case_ids and "f0q1" not in norm_ids
    assert "e1q1-e1q4" not in norm_ids


def test_actual_completion_payload_carries_enums_and_host_still_rejects_noncontiguous_segments(
    sample,
):
    data = prepared(sample)
    args = answers(data)
    args["decisions"][0]["quote_ids"] = ["e1q1", "e1q3"]
    agent = StubAgent(args)
    result = probe.run_review(agent, data)
    request = next(p for path, p in agent.requests if path == "/v1/chat/completions")
    fields = request["tools"][0]["function"]["parameters"]["properties"]["decisions"][
        "items"
    ]["properties"]
    assert fields["source_quote_id"]["enum"] == ["", *data["case_quotes"]]
    assert result["status"] == "invalid_review" and result["decisions"] == []
    assert "contiguous" in result["errors"][0]
