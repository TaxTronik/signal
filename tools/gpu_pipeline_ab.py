"""Offline-GPU-A/B der echten Pipeline gegen einen Git-Stand.

Unveränderte Korpusartefakte, getrennte Quell-Snapshots und gemeinsame rohe
Modellwerte isolieren Codeänderungen von Modell-/Indexänderungen. Der Lauf
aktiviert keine Generation und ändert keine Betriebseinstellungen.

Beispiel (ROCm unter Windows):
  .venv-amd/Scripts/python.exe tools/gpu_pipeline_ab.py --index PFAD
      --extra-gold C:/dev/taxtronikexctract/discovery/klausuren_v2.yaml

Gold und Rohwerte werden im Berichtverzeichnis archiviert. Cache-Auswertung
ist KEINE Latenzmessung; warme echte Pipeline-Aufrufe werden separat gemessen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def snapshot_sources(ref, out):
    before, after = out / "source-before", out / "source-after"
    archive = git("archive", "--format=tar", ref, "risk_layer", "catalog/begriffe.yaml")
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for entry in bundle.getmembers():
            if not entry.isfile():
                continue
            dest = (before / entry.name).resolve()
            if not dest.is_relative_to(before.resolve()):
                raise ValueError("Unerwarteter Pfad im Git-Archiv")
            source = bundle.extractfile(entry)
            if source is None:
                raise ValueError(entry.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read())
    for source in sorted((ROOT / "risk_layer").glob("*.py")):
        dest = after / "risk_layer" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    (after / "catalog").mkdir()
    shutil.copy2(ROOT / "catalog/begriffe.yaml", after / "catalog/begriffe.yaml")
    for source in (Path(__file__), ROOT / "tools/_gpu_eval_metrics.py"):
        shutil.copy2(source, out / source.name)
    return before, after


def open_variant(name, source, artifacts):
    alias = "_signal_gpu_" + name
    spec = importlib.util.spec_from_file_location(
        alias, source / "risk_layer/__init__.py",
        submodule_search_locations=[str(source / "risk_layer")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    modules = {part: importlib.import_module(alias + "." + part)
               for part in ("pipeline", "catalog", "embedding", "triggerindex", "muster")}
    catalog = modules["catalog"].lade_katalog(source / "catalog/begriffe.yaml")
    return {
        "modules": modules,
        "catalog": catalog,
        "pipe": modules["pipeline"].Pipeline(catalog),
        "index": modules["embedding"].EmbeddingIndex.load(artifacts / "index"),
        "trigger": modules["triggerindex"].TriggerIndex(artifacts / "trigger.sqlite"),
    }


def load_cases(paths, out):
    import yaml
    cases, provenance = [], []
    (out / "gold").mkdir()
    for number, path in enumerate(paths):
        path = path.resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        destination = out / "gold" / f"{number:02d}-{path.name}"
        shutil.copy2(path, destination)
        group = path.stem
        provenance.append({"path": str(path), "sha256": sha256(path),
                           "metadata": {k: v for k, v in data.items() if k != "faelle"}})
        for i, case in enumerate(data["faelle"]):
            targets = case.get("ziele", [])
            negative = data.get("zweck") == "praezision" or bool(case.get("negativ"))
            if not negative and not targets:
                raise ValueError(f"Gold ohne Ziele/Negativkennzeichnung: {path}:{i}")
            cases.append({"case_id": f"{group}:{case.get('fall_id', i + 1)}",
                          "group": group, "query": case["query"], "targets": targets,
                          "negative": negative, "category": case.get("kategorie"),
                          "uncertain": case.get("unsicher", False)})
    return cases, provenance


class CachedEncoder:
    name = "bge-m3"
    dim = 1024

    def __init__(self, texts, vectors, *, revision):
        self.revision = revision
        self.vectors = dict(zip(texts, vectors, strict=True))

    def encode(self, texts):
        import numpy as np
        return np.stack([self.vectors[text] for text in texts])


class PairCollector:
    name = "collect-only"

    def __init__(self):
        self.pairs = set()

    def score(self, pairs):
        self.pairs.update(pairs)
        return [0.0] * len(pairs)


class CachedReranker:
    name = "bge-reranker-v2-m3"

    def __init__(self, pairs, scores):
        self.scores = dict(zip(pairs, scores, strict=True))

    def score(self, pairs):
        return [self.scores[pair] for pair in pairs]


def run_case(variant, case, graph, encoder, reranker):
    options = variant["modules"]["pipeline"].AnalyseOptionen(
        erzeugt_am="gpu-ab", detektor=True, streit=True)
    report = variant["pipe"].run(
        case["query"], optionen=options, graph=graph, trigger=variant["trigger"],
        embedding=variant["index"], embedding_encoder=encoder, reranker=reranker)
    return {
        "suggestions": report.semantik_vorschlaege,
        "cards": [{"id": card.eintrag.id, "via": card.via,
                   "span": [card.treffer.start, card.treffer.end]}
                  for card in report.karten],
        "trigger_marks": [{"span": [risk.start, risk.end], "title": risk.titel,
                           "norm_ids": [n["id"] for n in risk.norm_vorschlag]}
                          for risk in report.risiken if risk.quelle == "trigger"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--extra-gold", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timing-cases", type=int, default=6)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.timing_cases < 0:
        parser.error("batch-size muss positiv und timing-cases nichtnegativ sein")
    # Vor ML-Imports setzen: kein Remote-Fallback oder impliziter NER-Load.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                      RISK_LAYER_NER="aus", RISK_LAYER_LEMMA="aus",
                      RISK_LAYER_FUSION="an", RISK_LAYER_RERANK_SCHWELLE="0.10",
                      RISK_LAYER_SIM_RETTUNG="0.55")
    import numpy as np
    import torch
    from risk_layer.embedding import BgeM3Encoder
    from risk_layer.graph import NormGraph
    from risk_layer.reranker import CrossEncoderReranker
    from tools._gpu_eval_metrics import summarize, evaluate_variant

    if not torch.cuda.is_available():
        raise RuntimeError("GPU nicht verfügbar; kein stiller CPU-/Hash-Fallback")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.out or ROOT / "out" / f"gpu-ab-{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"Berichtverzeichnis: {out}", flush=True)
    baseline_commit = git("rev-parse", args.baseline_ref).decode().strip()
    before, after = snapshot_sources(baseline_commit, out)
    artifacts = out / "artifacts"
    (artifacts / "index").mkdir(parents=True)
    for filename in ("vectors.npy", "meta.json"):
        shutil.copy2(args.index / filename, artifacts / "index" / filename)
    for filename in ("graph.sqlite", "trigger.sqlite"):
        with sqlite3.connect((ROOT / "corpus" / filename).as_uri() + "?mode=ro", uri=True) as src:
            with sqlite3.connect(artifacts / filename) as dest:
                src.backup(dest)
    paths = [ROOT / "fixtures/gold" / name for name in
             ("ao_v1.yaml", "organschaft_v1.yaml", "negativ_v1.yaml")]
    cases, gold = load_cases(paths + args.extra_gold, out)
    variants = {name: open_variant(name, source, artifacts)
                for name, source in (("before", before), ("after", after))}
    graph = NormGraph(artifacts / "graph.sqlite")
    meta = json.loads((artifacts / "index/meta.json").read_text(encoding="utf-8"))
    if meta.get("graph_fingerprint") != graph.fingerprint():
        raise RuntimeError("Index und Graph passen nicht zusammen; A/B abgebrochen")
    texts = sorted({sentence.text for variant in variants.values() for case in cases
                    for sentence in variant["modules"]["muster"].saetze_mit_offset(case["query"])
                    if len(sentence.text.strip()) >= 12}, key=lambda text: (len(text), text))
    encoder = BgeM3Encoder(device="cuda", local_files_only=True)
    reranker = CrossEncoderReranker(device="cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}; {len(cases)} Fälle, {len(texts)} Satztexte", flush=True)
    load_start = time.perf_counter()
    encoder._laden()
    reranker._laden()
    load_seconds = time.perf_counter() - load_start
    if any(next(model.parameters()).device.type != "cuda"
           for model in (encoder._modell, reranker._modell.model)):
        raise RuntimeError("Modell wurde nicht auf der GPU geladen")
    if meta.get("encoder_revision") != encoder.revision:
        raise RuntimeError("Encoder-Revision passt nicht zum Index")
    batches = []
    inference_start = time.perf_counter()
    for start in range(0, len(texts), args.batch_size):
        batches.append(encoder.encode(texts[start:start + args.batch_size]))
        if start % (args.batch_size * 5) == 0:
            print(f"Embedding: {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
    vectors = np.vstack(batches)
    encoding_seconds = time.perf_counter() - inference_start
    np.save(out / "query-vectors.npy", vectors)
    save_json(out / "query-texts.json", texts)
    cached_encoder = CachedEncoder(texts, vectors, revision=encoder.revision)
    collector = PairCollector()
    for variant in variants.values():
        for case in cases:
            run_case(variant, case, graph, cached_encoder, collector)
    pairs = sorted(collector.pairs, key=lambda pair: (len(pair[0]) + len(pair[1]), pair))
    print(f"Reranker: {len(pairs)} unterschiedliche Satz/Norm-Paare", flush=True)
    scores = []
    inference_start = time.perf_counter()
    for start in range(0, len(pairs), args.batch_size):
        scores.extend(reranker.score(pairs[start:start + args.batch_size]))
        if start % (args.batch_size * 20) == 0:
            print(f"Reranker: {min(start + args.batch_size, len(pairs))}/{len(pairs)}", flush=True)
    scoring_seconds = time.perf_counter() - inference_start
    save_json(out / "reranker-scores.json", [
        {"query": q, "document": d, "score": score}
        for (q, d), score in zip(pairs, scores, strict=True)])
    cached_reranker = CachedReranker(pairs, scores)
    rows = []
    for case in cases:
        row = dict(case)
        for name, variant in variants.items():
            row[name] = run_case(variant, case, graph, cached_encoder, cached_reranker)
            row[name]["metrics"] = evaluate_variant(row[name]["suggestions"], case["targets"])
        rows.append(row)
    summaries = {name: summarize(rows, name) for name in variants}
    # Reale, warme Aufrufe getrennt vom Cache-A/B. Keine Geschwindigkeitsbehauptung
    # aus den Cachezeiten; Messfälle stehen mit IDs im Bericht.
    timing, direct_rows = [], []
    if args.timing_cases:
        positions = np.linspace(0, len(cases) - 1, min(args.timing_cases, len(cases)), dtype=int)
        for i, position in enumerate(positions):
            case = cases[position]
            direct_row = dict(case)
            # Reihenfolge wechseln, damit kein Stand systematisch den ersten
            # Aufruf bekommt. Es werden echte GPU-Forward-Pässe ausgeführt.
            order = ("before", "after") if i % 2 == 0 else ("after", "before")
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                actual = run_case(variants[name], case, graph, encoder, reranker)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                direct_row[name] = actual
                cached = rows[position][name]["suggestions"]
                def norm_lists(suggestions):
                    return [[n["id"] for n in s["normen"]] for s in suggestions]
                timing.append({"case_id": case["case_id"], "variant": name,
                               "seconds": elapsed,
                               "same_norm_order_as_cached": norm_lists(actual["suggestions"]) == norm_lists(cached),
                               "same_suggestions_as_cached": actual["suggestions"] == cached})
            direct_rows.append(direct_row)
            if i % 20 == 0:
                print(f"Direkte GPU-Analysen: {i + 1}/{len(positions)} Fälle je Stand", flush=True)
    source_hashes = {str(path.relative_to(out)): sha256(path)
                     for root in (before, after, artifacts, out / "gold")
                     for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    for filename in ("gpu_pipeline_ab.py", "_gpu_eval_metrics.py", "query-texts.json",
                     "query-vectors.npy", "reranker-scores.json"):
        source_hashes[filename] = sha256(out / filename)
    result = {
        "harness": "gpu_pipeline_ab/1", "created_utc": stamp,
        "baseline_commit": baseline_commit,
        "source_hashes": source_hashes, "gold": gold,
        "runtime": {"python": sys.version, "torch": torch.__version__, "hip": torch.version.hip,
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("numpy", "sentence-transformers", "transformers")},
                    "gpu": torch.cuda.get_device_name(0), "encoder_revision": encoder.revision,
                    "reranker_revision": reranker.revision,
                    "encoder_dtype": str(next(encoder._modell.parameters()).dtype),
                    "reranker_dtype": str(next(reranker._modell.model.parameters()).dtype)},
        "index": {key: meta.get(key) for key in ("encoder", "dim", "encoder_revision",
                  "graph_fingerprint", "catalog_version", "catalog_fingerprint", "builder_version")},
        "index_vectors": len(meta["ids"]),
        "configuration": {"scope": "actual Pipeline (no oracle scope)", "fusion": True,
                          "inference_batch_size": args.batch_size,
                          "dense_floor": 0.35, "topn": 20, "topk_per_sentence": 5,
                          "rerank_threshold": 0.10, "sim_rescue": 0.55, "llm": False},
        "notes": ["Same index for both revisions: paraphrases and norm targets unchanged; catalog version is formally stale after code change.",
                  "Hit/MRR use the best sentence rank per original query; no artificial global Top5.",
                  "Historical Gold, not a newly blind test; catalog marks and semantic suggestions are separate.",
                  "Identical raw query vectors and pair scores reused in both revisions; warm timing measured separately."],
        "model_work": {"load_seconds": load_seconds, "encoding_seconds": encoding_seconds,
                       "scoring_seconds": scoring_seconds, "unique_texts": len(texts),
                       "unique_pairs": len(pairs)},
        "warm_timing": {"cases": timing, "medians_seconds": {
            name: statistics.median(t["seconds"] for t in timing if t["variant"] == name)
            if timing else None for name in variants}},
        "direct_summaries": {name: summarize(direct_rows, name) for name in variants},
        "direct_rows": direct_rows,
        "summaries": summaries, "rows": rows,
    }
    save_json(out / "report.json", result)
    for variant in variants.values():
        variant["trigger"].close()
    graph.close()
    print(json.dumps({"report": str(out / "report.json"),
                      "quality": {name: {key: summaries[name][key]
                                         for key in ("positive", "negative")}
                                  for name in variants},
                      "warm_medians_seconds": result["warm_timing"]["medians_seconds"]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
