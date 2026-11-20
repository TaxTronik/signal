"""Reproducible local GPU ablations of retrieval, evidence and abstention.

Research runner: no index activation or change of application defaults. The
baseline replays current production primitives and is checked against the
archived real Pipeline report. All configurations are fixed before inference.
Gold targets are used only by metrics, never for retrieval or document choice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# One-factor probes followed by declared combinations; no threshold search.
EXPERIMENTS = [
    dict(name="baseline", retrieval="base", context="sentence", doc="prefix300", pool=20),
    dict(name="prefix1500", retrieval="base", context="sentence", doc="prefix1500", pool=20),
    dict(name="norm_package", retrieval="base", context="sentence", doc="package", pool=20),
    dict(name="passage_index", retrieval="hybrid", context="sentence", doc="prefix300", pool=20),
    dict(name="neighbors", retrieval="base", context="neighbors", doc="prefix300", pool=20),
    dict(name="paragraph", retrieval="base", context="paragraph", doc="prefix300", pool=20),
    dict(name="wide_pool", retrieval="base", context="sentence", doc="prefix300", pool=60),
    dict(name="passage_package", retrieval="hybrid", context="sentence", doc="package", pool=20),
    dict(name="passage_package_wide", retrieval="hybrid", context="sentence", doc="package", pool=60),
    dict(name="passage_package_200", retrieval="hybrid", context="sentence", doc="package", pool=200),
    dict(name="context_package", retrieval="hybrid", context="neighbors", doc="package", pool=60),
    dict(name="context_rerank", retrieval="hybrid", context="neighbors", doc="package", pool=60,
         retrieval_context="sentence"),
    dict(name="paragraph_package", retrieval="hybrid", context="paragraph", doc="package", pool=60),
    dict(name="workflow_scope", retrieval="hybrid", context="neighbors", doc="package", pool=60,
         workflow_scope=True),
]
REUSE_PACKAGES = ("numpy", "torch", "transformers", "sentence-transformers")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class ValidatedReuse:
    """Completed cache provenance plus file checks repeated at the copy boundary."""

    directory: Path
    report_sha256: str
    manifest_sha256: str
    artifact_hashes: dict[str, str]
    copied: dict[str, str] = field(default_factory=dict)

    def verified_path(self, name):
        if Path(name).name != name or name not in self.artifact_hashes:
            raise ValueError(f"Reuse artifact has no recorded hash: {name}")
        path = self.directory / name
        if not path.is_file() or sha256(path) != self.artifact_hashes[name]:
            raise ValueError(f"Reuse artifact checksum mismatch: {path}")
        return path

    def copy_files(self, names, destination):
        # Check every member of a vector/text pair before copying either one.
        sources = [(name, self.verified_path(name)) for name in names]
        temporary = []
        try:
            for name, source in sources:
                target = destination / f".{name}.reuse-tmp"
                temporary.append((name, target))
                shutil.copy2(source, target)
                if sha256(target) != self.artifact_hashes[name]:
                    raise ValueError(f"Reuse artifact changed while copying: {source}")
            # A failed check must never leave a bad NPY at a resumable cache path.
            for name, target in temporary:
                target.replace(destination / name)
                self.copied[name] = self.artifact_hashes[name]
        finally:
            for _, target in temporary:
                target.unlink(missing_ok=True)


def validiere_reuse(directory, *, encoder_revision, reranker_revision, packages):
    """Require matching BGE model revisions and exact inference-package versions.

    Package compatibility deliberately means equality, including Torch's local
    CUDA/ROCm build tag. Tokenization or inference changes must not inherit scores
    merely because input strings match. Old completed /2 reports remain usable;
    newer manifests also record model revisions and must agree with runtime.
    Individual data-file hashes are checked immediately before copying and again
    on the temporary copy. An unfinished run without report.json is not a source.
    """
    directory = Path(directory).resolve()
    try:
        report = read_json(directory / "report.json")
        manifest = read_json(directory / "manifest.json")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Reuse requires a completed report and manifest: {directory}") from exc
    if (not isinstance(report, dict) or not isinstance(manifest, dict)
            or manifest.get("format") != "quality_experiments/2"
            or report.get("manifest") != manifest):
        raise ValueError("Reuse report/manifest mismatch or unsupported experiment format")
    if (report.get("status") not in (None, "pass", "complete") or report.get("error")
            or not isinstance(report.get("rows"), list)
            or not isinstance(report.get("summaries"), dict)):
        raise ValueError("Reuse requires completed experiment results")
    runtime = report.get("runtime", {})
    expected_revisions = {"encoder_revision": encoder_revision, "reranker_revision": reranker_revision}
    if (not isinstance(runtime, dict)
            or any(not isinstance(value, str) or not value or runtime.get(name) != value
                   for name, value in expected_revisions.items())):
        raise ValueError("Reuse BGE encoder/reranker model revision mismatch")
    if "model_revisions" in manifest and manifest["model_revisions"] != expected_revisions:
        raise ValueError("Reuse model revisions disagree between manifest and runtime")
    recorded_packages = manifest.get("packages", {})
    if (not isinstance(recorded_packages, dict)
            or any(not isinstance(packages.get(name), str) or not packages[name]
                   or recorded_packages.get(name) != packages[name] for name in REUSE_PACKAGES)
            or runtime.get("torch") != recorded_packages.get("torch")):
        raise ValueError("Reuse inference package versions are incompatible")
    artifact_hashes = report.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("Reuse report has no artifact hashes")
    reuse = ValidatedReuse(directory, sha256(directory / "report.json"),
                           sha256(directory / "manifest.json"), dict(artifact_hashes))
    reuse.verified_path("manifest.json")
    return reuse


def encode_cached(texts, path, encoder, *, batch_size=16, reuse_from=None):
    import numpy as np
    text_path = path.with_suffix(".texts.json")
    if reuse_from is not None and not path.exists():
        if not isinstance(reuse_from, ValidatedReuse):
            raise ValueError("Reuse must pass validiere_reuse before copying cached inference")
        source_text = reuse_from.verified_path(text_path.name)
        if read_json(source_text) == texts:
            reuse_from.copy_files([path.name, text_path.name], path.parent)
    if path.exists():
        if read_json(text_path) != texts:
            raise ValueError(f"Text cache mismatch: {path}")
        return np.load(path)
    save_json(text_path, texts)
    result = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        result.append(encoder.encode(texts[start:start + batch_size]))
        if start % (batch_size * 25) == 0:
            print(f"{path.stem}: {start}/{len(texts)} ({time.perf_counter()-started:.1f}s)", flush=True)
    vectors = np.concatenate(result) if result else np.empty((0, encoder.dim), dtype=np.float32)
    np.save(path, vectors)
    return vectors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evidence", type=Path, default=ROOT / "out/gpu-ab-20260907-final")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--case-file", type=Path)
    parser.add_argument("--configs", nargs="+")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    experiments = [e for e in EXPERIMENTS if not args.configs or e["name"] in args.configs]
    if (not any(e["name"] == "baseline" for e in experiments)
            or (args.configs and set(args.configs) != {e["name"] for e in experiments})):
        parser.error("configs must include baseline and contain only declared names")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                      RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus")
    import numpy as np
    import torch
    import yaml
    from risk_layer.catalog import lade_katalog
    from risk_layer.embedding import BgeM3Encoder, EmbeddingIndex
    from risk_layer.graph import NormGraph
    from risk_layer.zitate import knoten_zitat
    from risk_layer.pipeline import rrf_fusion
    from risk_layer.reranker import CrossEncoderReranker, baue_norm_doc_fn
    from risk_layer.triggerindex import TriggerIndex
    from tools._gpu_eval_metrics import compare_rows, summarize
    from tools._quality_context import build_context_queries, source_workflow_laws
    from tools._quality_passages import assemble_norm_package, build_passages, catalog_anchors
    from tools._quality_policy import POLICY_SPECS, apply_policy
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the real local GPU")
    # Constructors are lazy: prove cache identity before the first model load.
    encoder = BgeM3Encoder(device="cuda", local_files_only=True)
    reranker = CrossEncoderReranker(device="cuda")
    packages = {name: importlib.metadata.version(name) for name in REUSE_PACKAGES}
    model_revisions = {"encoder_revision": encoder.revision, "reranker_revision": reranker.revision}
    reuse = (validiere_reuse(args.reuse_from, packages=packages, **model_revisions)
             if args.reuse_from else None)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence = args.baseline_evidence.resolve()
    previous = read_json(evidence / "report.json")
    input_hashes = {}
    for name, expected in previous["source_hashes"].items():
        if name.startswith(("artifacts", "gold")) or name == "reranker-scores.json":
            actual = sha256(evidence / name)
            if actual != expected:
                raise ValueError(f"Archived input changed: {name}")
            input_hashes[name] = actual
    source_paths = sorted((ROOT / "risk_layer").glob("*.py")) + [ROOT / "catalog/begriffe.yaml"]
    source_paths += [Path(__file__).resolve()] + sorted((ROOT / "tools").glob("_quality_*.py"))
    source_paths += [ROOT / "tools/_gpu_eval_metrics.py"]
    manifest = {
        "format": "quality_experiments/2", "experiments": experiments,
        "policies": POLICY_SPECS,
        "baseline_report_sha256": sha256(evidence / "report.json"),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "batch_size": args.batch_size,
        "input_hashes": input_hashes,
        "packages": packages,
        "model_revisions": model_revisions,
        "reuse_from": str(args.reuse_from.resolve()) if args.reuse_from else None,
        "reuse_source": ({"report_sha256": reuse.report_sha256,
                           "manifest_sha256": reuse.manifest_sha256} if reuse else None),
        "case_file_sha256": sha256(args.case_file) if args.case_file else None,
        "notes": ["All Gold is development evidence, not a fresh blind test.",
                  "Workflow source is optional external context, not inferred from targets.",
                  "No performance conclusions from cached inference."]}
    if (out / "manifest.json").exists():
        if read_json(out / "manifest.json") != manifest:
            raise ValueError("Changed source/configuration; use a new output directory")
    else:
        save_json(out / "manifest.json", manifest)
        for p in source_paths:
            dest = out / "source" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    cases = [{k: v for k, v in row.items() if k not in ("before", "after")}
             for row in previous["rows"]]
    if args.case_file:
        cases = read_json(args.case_file)
        shutil.copy2(args.case_file, out / "cases.json")
    sources = {}
    for p in (evidence / "gold").glob("*.yaml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        group = p.stem.split("-", 1)[1]
        for i, case in enumerate(data["faelle"]):
            sources[f"{group}:{case.get('fall_id', i + 1)}"] = case.get("quelle", "")
    catalog = lade_katalog(ROOT / "catalog/begriffe.yaml")
    base = EmbeddingIndex.load(evidence / "artifacts/index")
    if base is None:
        raise ValueError(f"Missing archived embedding index in {evidence / 'artifacts/index'}")
    graph = NormGraph(evidence / "artifacts/graph.sqlite")
    trigger = TriggerIndex(evidence / "artifacts/trigger.sqlite")
    if base.graph_fingerprint != graph.fingerprint():
        raise ValueError("Graph/index fingerprint mismatch")
    passages = build_passages(graph)
    anchors = catalog_anchors(catalog)
    anchor_records = [(nid, text) for nid in sorted(anchors) for text in anchors[nid]]
    save_json(out / "passages.json", [asdict(p) for p in passages])
    save_json(out / "anchors.json", anchor_records)
    contexts = {(case["case_id"], mode): build_context_queries(case["query"], mode=mode)
                for case in cases for mode in ("sentence", "neighbors", "paragraph")}
    texts = sorted({c.query_text for cs in contexts.values() for c in cs})
    print(f"{len(cases)} cases; {len(passages)} passages; {len(anchor_records)} anchors; {len(texts)} queries", flush=True)
    query_vectors = encode_cached(texts, out / "queries.npy", encoder, batch_size=args.batch_size,
                                  reuse_from=reuse)
    passage_vectors = encode_cached([p.embedding_text for p in passages], out / "passages.npy",
                                    encoder, batch_size=args.batch_size, reuse_from=reuse)
    anchor_vectors = encode_cached([text for _, text in anchor_records], out / "anchors.npy",
                                   encoder, batch_size=args.batch_size, reuse_from=reuse)
    qmap = dict(zip(texts, query_vectors, strict=True))
    by_norm = defaultdict(list)
    for i, passage in enumerate(passages):
        by_norm[passage.norm_id].append(i)
    anchor_by_norm = defaultdict(list)
    for i, (nid, _) in enumerate(anchor_records):
        anchor_by_norm[nid].append(i)
    all_index = EmbeddingIndex(
        np.concatenate([base.vectors, passage_vectors]),
        base.ids + [p.norm_id for p in passages],
        base.zitate + [knoten_zitat(p.law, p.para) for p in passages],
        base.titel + [p.title for p in passages], base.encoder_name,
        encoder_revision=base.encoder_revision)
    prefix300 = baue_norm_doc_fn(catalog, graph)
    prefix1500 = baue_norm_doc_fn(catalog, graph, max_text=1500)
    retrieval_cache, package_cache, pair_set = {}, {}, set()
    def retrieve(query, representation, pool, scope):
        key = (query, representation, pool, scope)
        if key not in retrieval_cache:
            index = base if representation == "base" else all_index
            # Scope comes only from explicit document metadata in its own probe.
            dense = index._topk(qmap[query], len(set(index.ids)) if scope else pool, .35)
            all_trigs = trigger.kandidaten(query, k=len(by_norm) if scope else max(pool, 20))
            support = {t["norm_id"] for t in all_trigs[:20]}
            trigs = all_trigs[:pool]
            if scope:
                dense = [n for n in dense if n["norm_id"].split(":")[1] in scope][:pool]
                trigs = [n for n in all_trigs if n["norm_id"].split(":")[1] in scope][:pool]
            retrieval_cache[key] = (rrf_fusion(dense, trigs)[:pool], support)
        return retrieval_cache[key]
    def document(query, nid, mode):
        if mode == "prefix300":
            return prefix300(nid)
        if mode == "prefix1500":
            return prefix1500(nid)
        key = (query, nid)
        if key not in package_cache:
            pids, aids = by_norm[nid], anchor_by_norm[nid]
            ps = passage_vectors[pids] @ qmap[query] if pids else []
            aa = anchor_vectors[aids] @ qmap[query] if aids else []
            ranked_p = [passages[pids[i]] for i in sorted(range(len(pids)), key=lambda i: (-float(ps[i]), pids[i]))]
            ranked_a = [anchor_records[aids[i]][1] for i in sorted(range(len(aids)), key=lambda i: (-float(aa[i]), aids[i]))]
            package_cache[key] = assemble_norm_package(nid, ranked_p, ranked_anchors=ranked_a) or prefix300(nid)
        return package_cache[key]
    plans = {}
    for config in experiments:
        entries = []
        for case in cases:
            scope = source_workflow_laws(sources.get(case["case_id"], "")) if config.get("workflow_scope") else None
            rows = []
            for context in contexts[case["case_id"], config["context"]]:
                retrieval_query = context.query_text
                if config.get("retrieval_context") == "sentence":
                    retrieval_query = next(c.query_text for c in contexts[case["case_id"], "sentence"]
                                           if c.focus_start == context.focus_start and c.focus_end == context.focus_end)
                pool, trigs = retrieve(retrieval_query, config["retrieval"], config["pool"], scope)
                # Production embeds raw sentence but reranks the trimmed focus.
                query = context.focus_text if config["context"] == "sentence" else context.query_text
                pairs = [(query, document(context.query_text, n["norm_id"], config["doc"])) for n in pool]
                pair_set.update(pairs)
                rows.append(dict(context=asdict(context), candidates=pool, trigger_ids=sorted(trigs), pairs=pairs))
            entries.append(dict(case_id=case["case_id"], scope=scope, sentences=rows))
        plans[config["name"]] = entries
        print(f"Prepared {config['name']}: union {len(pair_set)} rerank pairs", flush=True)
    save_json(out / "plans.json", plans)
    pairs = sorted(pair_set, key=lambda pair: (len(pair[0]) + len(pair[1]), pair))
    # Reuse only EXACT old query/document pairs; all new pairs get real GPU passes.
    old_scores = read_json(evidence / "reranker-scores.json")
    print(f"Old raw score format: {type(old_scores).__name__}", flush=True)
    scores = {}
    for item in old_scores:
        scores[(item["query"], item["document"])]=item["score"]
    score_path = out / "new-scores.jsonl"
    if reuse is not None and not score_path.exists():
        reuse.copy_files([score_path.name], score_path.parent)
    if score_path.exists():
        for line in score_path.read_text(encoding="utf-8").splitlines():
            q, d, score = json.loads(line)
            scores[(q, d)] = score
    missing = [pair for pair in pairs if pair not in scores]
    print(f"Scoring {len(missing)} new / {len(pairs)} total pairs", flush=True)
    started = time.perf_counter()
    with score_path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start:start + args.batch_size]
            values = reranker.score(batch)
            for pair, score in zip(batch, values, strict=True):
                scores[pair] = score
                stream.write(json.dumps([*pair, score], ensure_ascii=False) + "\n")
            stream.flush()
            if start % (args.batch_size * 100) == 0:
                print(f"Rerank: {start}/{len(missing)} ({time.perf_counter()-started:.1f}s)", flush=True)
    rows = [dict(case) for case in cases]
    summaries, changes = {}, {}
    for name, entries in plans.items():
        for policy in POLICY_SPECS:
            variant = name + "/" + policy
            for row, entry in zip(rows, entries, strict=True):
                suggestions = []
                for sentence in entry["sentences"]:
                    ranked = [{**n, "rerank": scores[tuple(pair)]}
                              for n, pair in zip(sentence["candidates"], sentence["pairs"], strict=True)]
                    selected = apply_policy(ranked, policy, trigger_ids=set(sentence["trigger_ids"]))
                    c = sentence["context"]
                    if selected:
                        suggestions.append(dict(start=c["focus_start"], end=c["focus_end"], satz=c["focus_text"],
                                                normen=[dict(id=n["norm_id"], sim=n["sim"], rerank=n["rerank"]) for n in selected]))
                row[variant] = {"suggestions": suggestions}
            summaries[variant] = summarize(rows, variant)
            paired = [{**row, "before": row["baseline/baseline_or"], "after": row[variant]} for row in rows]
            changes[variant] = compare_rows(paired)
    baseline = summaries["baseline/baseline_or"]
    if not args.case_file and baseline != previous["summaries"]["after"]:
        save_json(out / "baseline-mismatch.json", dict(expected=previous["summaries"]["after"], actual=baseline))
        raise ValueError("Experimental baseline does not reproduce real Pipeline metrics")
    def norm_order(suggestions):
        return [(s["start"], s["end"], [n["id"] for n in s["normen"]]) for s in suggestions]
    baseline_order_differences = None if args.case_file else [row["case_id"] for row, old in zip(rows, previous["rows"], strict=True)
                                  if norm_order(row["baseline/baseline_or"]["suggestions"])
                                  != norm_order(old["after"]["suggestions"])]
    result = dict(manifest=manifest, runtime=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  encoder_revision=encoder.revision, reranker_revision=reranker.revision),
                  n_passages=len(passages), n_norms=len(by_norm), n_queries=len(texts), n_pairs=len(pairs),
                  summaries=summaries, changes=changes, rows=rows,
                  reuse_copied_artifacts=reuse.copied if reuse else {},
                  baseline_order_differences=baseline_order_differences,
                  artifact_hashes={p.name: sha256(p) for p in out.iterdir()
                                   if p.is_file() and p.name != "report.json"})
    save_json(out / "report.json", result)
    for variant, summary in summaries.items():
        p, n = summary["positive"], summary["negative"]
        print(f"{variant:40} hits={p['hit@5']*p['n']:.0f}/{p['n']} MRR={p['mrr_best_sentence']:.3f} FP={n['false_suggestion_rate']*n['n']:.0f}/{n['n']}")
    graph.close()
    trigger.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
