from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest


from experiments.grounded_signal import audit_public_baseline as audit


def case(name, *, positive=True, context=False, split="evaluation", targets=None):
    return {"case_id": name, "family_id": name, "split": split, "text": "Ein Vorgang.", "diagnostic_labels": {
        "hint_expected": positive, "target_norm_ids": (targets or ["norm:AO:1"]) if positive else [],
        "no_hint_norm_ids": ["norm:AO:2"], "allowed_hint_norm_ids": ["norm:AO:3"],
        "insufficient_context_norm_ids": ["norm:AO:4"] if context else []}}


def row(case, *, semantic=(), cards=(), risks=(), hard=(), status="complete", elapsed=None):
    result = {"case_id": case["case_id"], "family_id": case["family_id"], "split": case["split"], "status": status,
              "output": {"suggestions": [{"normen": [{"id": n} for n in semantic]}] * 2,
                         "cards": [{"norm_anker": list(cards)}],
                         "risk_marks": [{"norm_vorschlag": [{"id": n} for n in risks]}], "hard_norm_ids": sorted(hard)}}
    if elapsed is not None:
        result["timing"] = {"elapsed_seconds": elapsed, "synchronized": True}
    return result


def test_independent_hard_sources_and_union_deduplicate_per_case():
    value = case("positive", targets=["norm:AO:1", "norm:AO:8"])
    result = row(value, semantic=["norm:AO:1", "norm:AO:9"], cards=["§ 8 Abs. 1 AO", "§ 8 AO"],
                 risks=["norm:AO:1:abs2", "norm:AO:7"], hard=["norm:AO:1", "norm:AO:7", "norm:AO:8"])
    got = audit.recount([value], [result])
    assert got["semantic_suggestions"]["all"]["output_norm_pairs"] == 2
    assert got["hard_norm_anchors"]["all"]["output_norm_pairs"] == 3
    assert got["all_visible_norms"]["all"]["output_norm_pairs"] == 4
    assert got["semantic_suggestions"]["all"]["known_target_norm_hits"] == 1
    assert got["all_visible_norms"]["all"]["known_target_norm_hits"] == 2


def test_unknown_norm_on_clear_negative_is_any_output_without_fabricated_negative_label():
    neg, context = case("negative", positive=False), case("context", positive=False, context=True)
    rows = [row(neg, cards=["§ 99 AO"], hard=["norm:AO:99"]), row(context, semantic=["norm:AO:4"])]
    result = audit.recount([neg, context], rows)
    visible = result["all_visible_norms"]["all"]
    assert visible["explicit_no_hint_output_pairs"] == 0
    assert visible["clear_negative_cases_with_output"] == 1
    assert visible["clear_negative_output_norm_pairs"] == 1
    assert visible["context_only_cases_with_output"] == 1
    assert visible["unadjudicated_output_pairs"] == 1
    assert visible["precision"] is None
    assert result["semantic_suggestions"]["all"]["clear_negative_cases_with_output"] == 0


def test_failed_and_pending_cases_keep_targets_and_negatives_unknown():
    pos = case("p", targets=["norm:AO:1", "norm:AO:8"], split="development")
    neg = case("n", positive=False)
    context = case("c", positive=False, context=True)
    got = audit.recount([pos, neg, context], [row(pos, status="technical_error"), row(neg, status="invalid_output", semantic=["norm:AO:2"])])
    values = got["all_visible_norms"]["all"]
    assert values["planned_cases"] == 3
    assert values["known_target_norm_pairs_unknown"] == 2
    assert values["hint_recall_upper_bound"] == 1
    assert values["output_norm_pairs"] == 0
    assert values["clear_negative_cases_unknown"] == 1
    assert values["context_only_cases_unknown"] == 1
    assert values["complete_non_hint_cases_without_output"] == 0
    assert values["not_run"] == values["invalid_output"] == values["technical_error"] == 1
    assert got["all_visible_norms"]["development"]["known_target_norm_pairs"] == 2
    assert got["all_visible_norms"]["evaluation"]["planned_cases"] == 2


@pytest.mark.parametrize("citation,expected", [("§ 8c Abs. 1 KStG", "norm:KStG:8c"),
                                               ("§\u00a02 Abs. 2 Nr. 2 UStG", "norm:UStG:2"),
                                               ("Art. 3 Abs. 1 GG", "norm:GG:art3")])
def test_bounded_citation_grammar(citation, expected):
    assert audit.citation_parent(citation) == expected


@pytest.mark.parametrize("citation", ["§§ 1, 2 AO", "§ 1 bis 3 AO", "§ 1 AO iVm § 2 AO", "§ 1 Abgabenordnung extra", "Freitext 42 AO"])
def test_unsupported_citation_never_silently_loses_norms(citation):
    with pytest.raises(ValueError, match="Unsupported"):
        audit.citation_parent(citation)


def test_saved_hard_list_is_not_trusted():
    c = case("one")
    with pytest.raises(ValueError, match="hard-norm"):
        audit.recount([c], [row(c, cards=["§ 1 AO"], hard=[])])


def test_repeated_or_unplanned_rows_rejected():
    c = case("one")
    r = row(c)
    with pytest.raises(ValueError, match="Duplicate"):
        audit.recount([c], [r, r])
    with pytest.raises(ValueError, match="Unplanned"):
        audit.recount([], [r])
    wrong = deepcopy(r)
    wrong["split"] = "development"
    with pytest.raises(ValueError, match="identity"):
        audit.recount([c], [wrong])


def test_timing_measures_failures_and_excludes_unmeasured_pending():
    c = case("x")
    rows = [row(c, elapsed=2), row(c, status="technical_error", elapsed=10), row(c)]
    rows[1]["timing"]["synchronized"] = False
    got = audit.measured_timing(rows)
    assert got["recorded_attempts"] == 2
    assert got["total_seconds"] == 12
    assert got["median_seconds"] == 6
    assert got["p95_seconds"] == 9.6
    assert got["synchronization_unavailable_attempts"] == 1


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf")])
def test_bad_duration_rejected(bad):
    with pytest.raises(ValueError, match="duration"):
        audit.measured_timing([row(case("x"), elapsed=bad)])


@pytest.mark.parametrize("bad", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json(bad):
    with pytest.raises(ValueError):
        audit.strict_json(bad)


def test_file_change_detected_and_inventory_rejects_escape(tmp_path):
    file = tmp_path / "a.json"
    file.write_text("{}")
    bound = audit.BoundFiles()
    bound.json(file)
    file.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        bound.unchanged()
    with pytest.raises(ValueError):
        audit.BoundFiles().inventory(tmp_path, {"../a.json": "a" * 64}, ())


def test_profile_read_is_ast_only():
    source = 'raise RuntimeError("must never execute")\nPROFILE_CONFIG: dict = {"treffer": {"context": "sentence"}}'
    assert audit.profile_from_source(source) == {"context": "sentence"}


def test_import_is_stdlib_only_in_fresh_interpreter():
    code = f"import runpy,sys;runpy.run_path({str(Path(audit.__file__))!r},run_name='audit_import');print([m for m in sys.modules if m.split('.')[0] in ('torch','risk_layer','experiments','numpy','transformers')])"
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_exact_json_comparison_distinguishes_boolean():
    assert audit.canonical({"hits": True}) != audit.canonical({"hits": 1})


def fake_archive(tmp_path, *, failure=False):
    """Small, wholly synthetic archive; no producer imports or local model data."""
    prepared, run = tmp_path / "prepared", tmp_path / "run"
    prepared.mkdir()
    run.mkdir()

    def save(root, name, value):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(audit.canonical(value), encoding="utf8")

    def inventory(root, exclude=()):
        return {p.relative_to(root).as_posix(): audit.digest(p.read_bytes()) for p in root.rglob("*")
                if p.is_file() and p.relative_to(root).as_posix() not in exclude}

    cases = []
    for i in range(16):
        c = case(f"c{i}", positive=i not in (1, 5, 9, 13), context=i == 5,
                 split="development" if i < 8 else "evaluation", targets=["norm:AO:1", "norm:AO:8"] if i in (0, 2) else None)
        c["family_id"] = f"f{i//2}"
        c["diagnostic_labels"]["annotation_status"] = "assistant_reviewed_synthetic"
        cases.append(c)
    save(prepared, "cases.json", cases)
    save(prepared, "primary/cases.json", cases)
    primary_inventory = {"cases.json": audit.digest((prepared / "cases.json").read_bytes())}
    save(prepared, "primary/input-manifest.json", {"artifacts": primary_inventory})
    profile = {"context": "sentence", "shortlist": 20}
    source = prepared / "source/risk_layer/semantic_search.py"
    source.parent.mkdir(parents=True)
    source.write_text(f"PROFILE_CONFIG: dict = {{'treffer': {profile!r}}}")
    (prepared / "probe-source.py").write_text("# No runtime executed.\n")
    models = {k: {"name": k, "revision": "pinned-" + k, "dtype": "torch.float32" if k == "encoder" else "torch.float16"}
              for k in ("encoder", "bge", "qwen")}
    receipts = {k: {"name": m["name"], "revision": m["revision"], "dtypes": [m["dtype"]], "devices": ["cuda:0"]}
                for k, m in models.items()}
    save(prepared, "gpu-reference.json", {"profile_config": profile, "models": receipts, "runtime": {"device": "synthetic"}})
    identity = {"graph": "a" * 64, "catalog": "b" * 64, "encoder": "encoder", "revision": "pin", "passage_sources": "c" * 64}
    save(prepared, "index/meta.json", {"graph_fingerprint": identity["graph"], "catalog_fingerprint": identity["catalog"],
                                       "encoder": "encoder", "encoder_revision": "pin", "passage_evidence": {"source_fingerprint": identity["passage_sources"]}})
    protocol = {"profile": "treffer", "profile_config": profile, "options": {"semantik_profil": "treffer"},
                "primary_manifest_sha256": audit.digest((prepared / "primary/input-manifest.json").read_bytes()),
                "primary_artifacts": primary_inventory, "case_ids": [c["case_id"] for c in cases],
                "source_hashes": {"risk_layer/semantic_search.py": audit.digest(source.read_bytes())},
                "probe_source_sha256": audit.digest((prepared / "probe-source.py").read_bytes()),
                "gpu_reference_sha256": audit.digest((prepared / "gpu-reference.json").read_bytes()),
                "source_identity": identity, "models": models, "runtime_packages": {"test": "1"}}
    save(prepared, "protocol.json", protocol)
    manifest = {"format": "grounded_signal_public_baseline_inputs/1", "artifacts": inventory(prepared)}
    save(prepared, "prepared-manifest.json", manifest)
    for name in ("protocol.json", "cases.json", "prepared-manifest.json", "probe-source.py"):
        (run / name).write_bytes((prepared / name).read_bytes())
    rows = [] if failure else [row(c, semantic=["norm:AO:1"] if c["diagnostic_labels"]["hint_expected"] else (), elapsed=1) for c in cases]
    for n, r in enumerate(rows, 1):
        save(run, f"case-{n:03d}.json", r)
    (run / "rows.jsonl").write_text("".join(audit.canonical(r) + "\n" for r in rows))
    counts = audit.recount(cases, rows)
    report = {"format": "grounded_signal_public_baseline/1", "status": "failed" if failure else "complete",
              "rows": rows, "protocol": protocol, "protocol_sha256": audit.digest(audit.canonical(protocol).encode()),
              "prepared_manifest_sha256": audit.digest((prepared / "prepared-manifest.json").read_bytes()),
              "models": {} if failure else receipts, "runtime": {"packages": {"test": "1"}, "device": "synthetic"},
              "warmup": {"status": "complete", "excluded_from_case_timings": True},
              "end_verification": {"status": "pass"},
              "aggregate": {k: v for k, v in counts.items() if k != "hard_norm_anchors"}, "artifacts": inventory(run)}
    report["aggregate"]["timing"] = audit.measured_timing(rows)
    save(run, "report.json", report)
    return run, prepared


def test_complete_archive_audit_and_queryfree_export(tmp_path):
    run, prepared = fake_archive(tmp_path)
    result = audit.audit(run, prepared)
    assert result["status"] == "pass"
    assert result["counts"]["all_visible_norms"]["all"]["positive_case_hits"] == 12
    assert result["counts"]["all_visible_norms"]["all"]["known_target_norm_hits"] == 12
    rendered = audit.canonical(result)
    assert "Ein Vorgang" not in rendered and "case_id" not in rendered and "c15" not in rendered


def test_setup_failure_has_full_unknown_denominators(tmp_path):
    run, prepared = fake_archive(tmp_path, failure=True)
    result = audit.audit(run, prepared)
    assert result["run_status"] == "failed"
    counts = result["counts"]["all_visible_norms"]["all"]
    assert counts["not_run"] == 16 and counts["known_target_norm_pairs_unknown"] == 14
    assert counts["clear_negative_cases_unknown"] == 3 and counts["context_only_cases_unknown"] == 1
    assert counts["complete_non_hint_cases_without_output"] == 0


def test_aggregate_tampering_is_detected_after_valid_archive_bindings(tmp_path):
    run, prepared = fake_archive(tmp_path)
    report = json.loads((run / "report.json").read_bytes())
    report["aggregate"]["all_visible_norms"]["all"]["positive_case_hits"] = 11
    (run / "report.json").write_text(audit.canonical(report), encoding="utf8")
    with pytest.raises(ValueError, match="independent raw-output recount"):
        audit.audit(run, prepared)
