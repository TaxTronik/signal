"""Four fixed BGE/Qwen agreement rules on exact archived cascade inputs.

CPU-only post-hoc exploration, without threshold search or label-aware filtering.
The development and historical additional datasets are reported separately.
No production setting, model, GPU, or activated index is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools._gpu_eval_metrics import compare_rows, summarize  # noqa: E402

CONFIG = "passage_package_200"
CASE_FIELDS = ("case_id", "group", "query", "targets", "negative", "category", "uncertain")
AGREEMENT_RULES = {
    "qwen_half_bge_floor": {"qwen_min": .5, "bge_min": .01},
    "qwen_half_bge_weak_rescue": {"qwen_min": .5, "bge_weak_rescue": True},
    "qwen_half_bge_tenth": {"qwen_min": .5, "bge_min": .1},
    "qwen_tenth_bge_tenth": {"qwen_min": .1, "bge_min": .1},
}
BASELINES = ("standard", "bge_pool200_weak_rescue", "qwen_half")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pair_digest(pair):
    if len(pair) != 2 or not all(isinstance(part, str) for part in pair):
        raise ValueError("Expected an exact query/document string pair")
    # JSON framing distinguishes all query/document boundaries; no normalization.
    return hashlib.sha256(json.dumps(pair, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("A finite raw model score is required")
    return float(value)


def load_scores(paths):
    """Stream large JSONL caches; later inputs override exactly as the GPU probe."""
    scores, overrides = {}, []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream) if path.suffix == ".json" else (
                json.loads(line) for line in stream if line.strip())
            for record in records:
                query, document, value = ((record["query"], record["document"], record["score"])
                                          if isinstance(record, dict) else record)
                key, value = pair_digest((query, document)), finite(value)
                if key in scores and scores[key] != value:
                    overrides.append({"path": str(path), "pair_sha256": key,
                                      "previous": scores[key], "replacement": value})
                scores[key] = value
    return scores, overrides


def plan_cases(path, config=CONFIG):
    """Stream one case at a time from the hash-bound, indent=2 plan artifact.

    The producer's explicit layout is checked instead of loading its 300 MB
    multi-configuration JSON document while a separate GPU job is running.
    Unsupported/minified layouts fail loudly; they are never partially replayed.
    """
    active, buffer = False, []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.rstrip("\r\n")
            if not active:
                active = stripped == f'  "{config}": ['
                continue
            if stripped in ("  ]", "  ],"):
                if buffer:
                    raise ValueError("Truncated case in plan artifact")
                return
            if not buffer and stripped != "    {":
                raise ValueError("Unsupported plan case layout")
            buffer.append(line)
            if stripped in ("    }", "    },"):
                yield json.loads("".join(buffer).rstrip().removesuffix(","))
                buffer.clear()
    raise ValueError(f"Missing or incomplete configuration: {config}")


def shortlist(sentence, bge_scores):
    candidates = sentence["candidates"]
    pairs = sentence["pairs"]
    if len(candidates) != len(pairs):
        raise ValueError("Candidate/pair count mismatch")
    norm_ids = [candidate["norm_id"] for candidate in candidates]
    if len(norm_ids) != len(set(norm_ids)):
        raise ValueError("Duplicate candidate norm ID")
    scored = [{**candidate, "pair_sha256": pair_digest(pair),
               "bge": bge_scores[pair_digest(pair)]}
              for candidate, pair in zip(candidates, pairs, strict=True)]
    return sorted(scored, key=lambda n: (-n["bge"], n["norm_id"]))[:20]


def select(candidates, rule):
    """Model-score-only conjunction, followed by Qwen/sim/ID order and Top5."""
    kept = []
    for candidate in candidates:
        qwen, bge = finite(candidate["qwen"]), finite(candidate["bge"])
        sim = finite(candidate["sim"]) if candidate.get("sim") is not None else None
        accepted = qwen >= rule["qwen_min"] and bge >= rule.get("bge_min", float("-inf"))
        if rule.get("bge_weak_rescue"):
            accepted = accepted and (bge >= .1 or (bge >= .01 and sim is not None and sim >= .55))
        if accepted:
            kept.append(candidate)
    return sorted(kept, key=lambda n: (-n["qwen"], -(n["sim"] or 0), n["norm_id"]))[:5]


def suggestion(context, candidates):
    return {"start": context["focus_start"], "end": context["focus_end"], "satz": context["focus_text"],
            "normen": [{"id": n["norm_id"], "sim": n["sim"], "rerank": n["qwen"]} for n in candidates]}


def evaluate_dataset(name, source, qwen_dir, baseline_dir):
    bge, qwen = read(source / "report.json"), read(qwen_dir / "report.json")
    manifest = qwen["manifest"]
    hashes = {}

    def verify(path, expected):
        actual = digest(path)
        hashes[str(path)] = actual
        if not isinstance(expected, str) or actual != expected:
            raise ValueError(f"Archived source checksum mismatch: {path}")

    if read(qwen_dir / "manifest.json") != manifest:
        raise ValueError("Qwen manifest differs from completed report")
    if manifest["bge_shortlist"] != 20 or CONFIG not in manifest["configs"]:
        raise ValueError("Requires the original BGE Top20 shortlist of passage_package_200")
    for filename in ("report.json", "plans.json", "new-scores.jsonl"):
        verify(source / filename, manifest["inputs"][filename])
    for filename in ("plans.json", "new-scores.jsonl"):
        if hashes[str(source / filename)] != bge["artifact_hashes"][filename]:
            raise ValueError("BGE artifact checksum differs from the completed report")
    verify(baseline_dir / "report.json", bge["manifest"]["baseline_report_sha256"])
    verify(baseline_dir / "reranker-scores.json", bge["manifest"]["input_hashes"]["reranker-scores.json"])
    verify(qwen_dir / "scores.jsonl", qwen["score_sha256"])
    for path in (qwen_dir / "report.json", qwen_dir / "manifest.json"):
        hashes[str(path)] = digest(path)
    baseline_runtime = read(baseline_dir / "report.json")["runtime"]
    if any(bge["runtime"][key] != baseline_runtime[key] for key in ("encoder_revision", "reranker_revision")):
        raise ValueError("BGE or encoder revision differs from the baseline score source")
    model_revisions = {"bge": bge["runtime"]["reranker_revision"],
                       "qwen": manifest["model_revision"], "encoder": bge["runtime"]["encoder_revision"]}
    bge_scores, bge_overrides = load_scores([baseline_dir / "reranker-scores.json", source / "new-scores.jsonl"])
    qwen_scores, qwen_overrides = load_scores([qwen_dir / "scores.jsonl"])
    bge_rows = {row["case_id"]: row for row in bge["rows"]}
    qwen_rows = {row["case_id"]: row for row in qwen["rows"]}
    if len(bge_rows) != len(bge["rows"]) or len(qwen_rows) != len(qwen["rows"]) or bge_rows.keys() != qwen_rows.keys():
        raise ValueError("Duplicate or differing BGE/Qwen case IDs")
    rows, audit, seen = [], [], set()
    rules = {"qwen_half": {"qwen_min": .5}, **AGREEMENT_RULES}
    for plan in plan_cases(source / "plans.json"):
        case_id = plan["case_id"]
        if case_id in seen or case_id not in bge_rows:
            raise ValueError("Duplicate or unknown plan case")
        seen.add(case_id)
        bge_row, qwen_row = bge_rows[case_id], qwen_rows[case_id]
        if any(bge_row.get(key) != qwen_row.get(key) for key in CASE_FIELDS):
            raise ValueError("BGE/Qwen case content or labels differ")
        row = {key: bge_row.get(key) for key in CASE_FIELDS}
        row["standard"] = bge_row["baseline/baseline_or"]
        row["bge_pool200_weak_rescue"] = bge_row[f"{CONFIG}/weak_rescue"]
        row.update({rule: {"suggestions": []} for rule in rules})
        for sentence in plan["sentences"]:
            candidates = shortlist(sentence, bge_scores)
            # Lookup happens only after the BGE shortlist, exactly like the GPU
            # probe; every required Qwen pair must exist. No fallback scores.
            candidates = [{**n, "qwen": qwen_scores[n["pair_sha256"]]} for n in candidates]
            audit.append({"case_id": case_id, "start": sentence["context"]["focus_start"],
                          "pool_count": len(sentence["candidates"]), "shortlist": candidates})
            for rule_name, rule in rules.items():
                kept = select(candidates, rule)
                if kept:
                    row[rule_name]["suggestions"].append(suggestion(sentence["context"], kept))
        if row["qwen_half"] != qwen_row[f"{CONFIG}/qwen_half"]:
            raise ValueError(f"Reconstructed Qwen baseline differs: {case_id}")
        rows.append(row)
    if seen != bge_rows.keys():
        raise ValueError("Missing plan cases")
    variants = [*BASELINES, *AGREEMENT_RULES]
    return {"dataset": name, "model_revisions": model_revisions, "source_hashes": hashes,
            "qwen_baseline_exact": True, "rows": rows, "shortlist_audit": audit,
            "cache_overrides": {"bge": bge_overrides, "qwen": qwen_overrides},
            "summaries": {variant: summarize(rows, variant) for variant in variants},
            "changes": {baseline: {variant: compare_rows([
                {**row, "before": row[baseline], "after": row[variant]} for row in rows])
                for variant in AGREEMENT_RULES} for baseline in ("standard", "qwen_half")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "out/gpu-ab-20260907-final")
    parser.add_argument("--main-bge", type=Path, default=ROOT / "out/quality-experiments-20260907-r2")
    parser.add_argument("--main-qwen", type=Path, default=ROOT / "out/quality-qwen-20260907-r2")
    parser.add_argument("--additional-bge", type=Path, default=ROOT / "out/quality-additional-validation/cascade-bge")
    parser.add_argument("--additional-qwen", type=Path, default=ROOT / "out/quality-additional-validation/qwen")
    parser.add_argument("--out", type=Path, default=ROOT / "out/quality-agreement")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output directory exists; choose a fresh path to preserve evidence")
    result = {"format": "quality_agreement_probe/1", "agreement_rules": AGREEMENT_RULES,
              "baselines": BASELINES, "source_hashes": {
                  str(path.relative_to(ROOT)): digest(path) for path in
                  (Path(__file__).resolve(), ROOT / "tools/_gpu_eval_metrics.py")},
              "notes": [
                  "Four explicitly requested fixed rules, reported equally; no threshold sweep or selected winner.",
                  "Post-hoc exploration after seeing baseline metrics; neither dataset is fresh blind gold.",
                  "Main development and historical additional validation are separate, without a pooled headline.",
                  "Same exact query/document pairs and recorded model revisions; BGE Top20 from original pool200.",
                  "All agreement rules rank by raw Qwen score, similarity, norm ID, then Top5 per sentence.",
                  "Models can share errors; agreement is not a legal correctness probability.",
                  "Only immutable caches are read; no model, GPU, production change, or activation."],
              "datasets": {}}
    for name, source, qwen_dir in (("main", args.main_bge, args.main_qwen),
                                    ("additional", args.additional_bge, args.additional_qwen)):
        dataset = evaluate_dataset(name, source, qwen_dir, args.baseline)
        result["datasets"][name] = dataset
        if name != "main" and dataset["model_revisions"] != result["datasets"]["main"]["model_revisions"]:
            raise ValueError("Main/additional model revisions differ")
        for variant, metrics in dataset["summaries"].items():
            p, n = metrics["positive"], metrics["negative"]
            print(f"{name:10} {variant:30} hits={p['hit@5'] * p['n']:.0f}/{p['n']} "
                  f"FP={n['false_suggestion_rate'] * n['n']:.0f}/{n['n']} "
                  f"MRR={p['mrr_best_sentence']:.3f}", flush=True)
    result["torch_imported"] = "torch" in sys.modules
    if result["torch_imported"]:
        raise RuntimeError("Unexpected model runtime imported during CPU-only probe")
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
