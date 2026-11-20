"""Explicit passage-based semantic profiles, separate from hard markings.

The legacy profile remains the default. New profiles require a coherent index
generation containing source-bound passage evidence. Gold labels and guessed
law classifications never participate in runtime retrieval.
"""
from __future__ import annotations

import re
import math
from typing import Any

import numpy as np

from .reranker import filtere_mit_rettung
from .semantic_context import build_context_queries
from .zitate import knoten_zitat

PROFILE_CONFIG: dict[str, dict[str, Any]] = {
    "normpaket": {"context": "sentence", "pool": 20, "hybrid": False},
    "kontext": {"context": "neighbors", "pool": 60, "hybrid": True},
    "kaskade": {"context": "sentence", "pool": 200, "hybrid": True,
                "rerank_threshold": .5, "sim_rescue": 0., "cascade": True},
    "praezision": {"context": "sentence", "pool": 200, "hybrid": True,
                   "rerank_threshold": .5, "sim_rescue": 0., "cascade": True,
                   "shortlist": 20, "agreement": True},
    "dokument": {"context": "neighbors", "pool": 200, "hybrid": True,
                 "rerank_threshold": .5, "sim_rescue": 0., "cascade": True,
                 "package": "source_first", "shortlist": 50, "agreement": True,
                 "neighbor_radius": 2},
    "treffer": {"context": "sentence", "pool": 200, "hybrid": True,
                "rerank_threshold": .5, "sim_rescue": 0., "cascade": True,
                "instruction": (
                    "Judge whether the legal provision (Document) matches the legal concept "
                    "or factual situation described in the Query."
                ),
                "passage_rerank": True, "passage_shortlist": "package", "shortlist": 20},
}


class SemanticResourcesUnavailable(RuntimeError):
    """An explicitly requested profile cannot run with the current resources."""


def validate_profile(profile: str, laws: tuple[str, ...]) -> None:
    if profile not in ("standard", *PROFILE_CONFIG):
        raise ValueError("Unbekanntes Semantik-Profil")
    if (not isinstance(laws, tuple) or len(laws) > 32
            or any(not isinstance(law, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", law)
                   for law in laws)):
        raise ValueError("Rechtsgebiete müssen gültige Gesetz-IDs sein")
    if laws and profile == "standard":
        raise ValueError("Rechtsgebiete erfordern ein Passagen-Profil")


def passage_suggestions(text: str, *, profile: str, laws: tuple[str, ...],
                        index, encoder, reranker, trigger, graph,
                        chains: dict[str, list], catalog=None) -> list[dict[str, Any]]:
    """Run one declared profile with exact input/source offsets and raw scores."""
    from .pipeline import rrf_fusion  # lazy: pipeline calls this optional layer

    validate_profile(profile, laws)
    if (index is None or encoder is None or reranker is None or graph is None
            or getattr(index, "passage_evidence", None) is None):
        raise SemanticResourcesUnavailable("Passagen-Profil benötigt Passagen-Index, Encoder, Reranker und Normgraph")
    if index.graph_fingerprint != graph.fingerprint():
        raise SemanticResourcesUnavailable("Passagen-Index und Normgraph gehören nicht zum selben Stand")
    from .embedding import katalog_fingerprint_berechnen
    catalog_fingerprint = (katalog_fingerprint_berechnen(catalog) if catalog is not None
                           else index.catalog_fingerprint)
    evidence = index.passage_evidence
    if (not evidence.matches_graph(graph)
            or not evidence.matches_identity(index.encoder_name, index.encoder_revision, catalog_fingerprint)):
        raise SemanticResourcesUnavailable("Passagen-Evidenz, Normgraph und Katalog gehören nicht zum selben Stand")
    preselector = getattr(reranker, "vorauswahl", None)
    config = PROFILE_CONFIG[profile]
    if config.get("cascade") and preselector is None:
        raise SemanticResourcesUnavailable("Kaskade benötigt den konfigurierten bge-qwen3-reranker-4b")
    checks = bool(config.get("checks"))
    if (checks or config.get("instruction")) and not callable(getattr(reranker, "score_with_instruction", None)):
        raise SemanticResourcesUnavailable("Explizite Modellinstruktion benötigt einen Reranker mit Instruktionsunterstützung")
    contexts = build_context_queries(text, mode=config["context"],
                                     neighbor_radius=config.get("neighbor_radius", 1),
                                     max_chars=config.get("context_chars", 900))
    if not contexts:
        return []
    texts = list(dict.fromkeys(c.query_text for c in contexts))
    vectors = np.asarray(encoder.encode(texts))
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise RuntimeError("Ungültige Embedding-Vektoren für das Passagen-Profil")
    index._encoder_pruefen(encoder, int(vectors.shape[1]))
    by_text = dict(zip(texts, vectors, strict=True))
    result = []
    # Query-local caches: neither user text nor mutable scores persist globally.
    completed: dict[Any, list[dict[str, Any]]] = {}
    for context in contexts:
        query = context.focus_text if config["context"] == "sentence" else context.query_text
        cache_key: Any = context.query_text
        if checks:
            from .semantic_checks import format_focus_context
            # Neighbors provide facts; they must not turn every focus into the
            # same proposition. An incomplete focus cannot be confirmed.
            if context.focus_truncated:
                continue
            query = format_focus_context(context)
            cache_key = (context.focus_start, context.focus_end, context.query_text)
        if cache_key not in completed:
            vector = by_text[context.query_text]
            pool = int(config["pool"])
            limit = max(len(index.ids), len(index.passage_evidence.passages)) if laws else pool
            candidates = (index.hybrid_topk(vector, k=limit, min_sim=.35) if config["hybrid"]
                          else index._topk(vector, limit, .35))
            # A trigger index can contain more rows than the live vector index.
            # With explicit scope, apply the law filter before any trigger cap.
            trigs = trigger.kandidaten(context.query_text, k=None if laws else limit) if trigger is not None else []
            if laws:
                candidates = [c for c in candidates if c["norm_id"].split(":")[1] in laws][:pool]
                trigs = [c for c in trigs if c["norm_id"].split(":")[1] in laws][:pool]
            if trigger is not None:
                candidates = rrf_fusion(candidates, trigs)[:pool]
            ranking_query = context.focus_text if config["context"] == "sentence" else context.query_text
            if config.get("citation_pool") and candidates:
                from .citation_candidates import citation_candidates
                if preselector is None:
                    raise SemanticResourcesUnavailable("Zitatsuche benötigt eine Reranker-Vorauswahl")
                # Seeds are ranked on the original anchor-aware package before
                # either expansion route. New neighbors do not seed a second hop.
                originals = [(c, index.norm_package(c["norm_id"], vector)) for c in candidates]
                originals = [(c, p) for c, p in originals if p is not None and p.evidence]
                seed_pairs = [(ranking_query, p.text) for _, p in originals]
                seed_scores = (_validated_scores(preselector, seed_pairs, "Zitat-Vorauswahl")
                               if seed_pairs else [])
                seeds = [c["norm_id"] for (c, _), _score in sorted(
                    zip(originals, seed_scores, strict=True),
                    key=lambda item: (-item[1], item[0][0]["norm_id"]))[:10]]
                candidates.extend(citation_candidates(
                    graph, evidence, seeds, {c["norm_id"] for c in candidates},
                    k=config["citation_pool"], laws=laws))
            if config.get("lexical_pool"):
                from .passage_selection import lexical_candidates
                # A bounded independent source-text route may recover a norm
                # absent from the dense/trigger pool. Every added norm still
                # undergoes the same passage ranking, shortlist and Qwen gate.
                sparse = lexical_candidates(evidence, context.query_text,
                                            k=config["lexical_pool"], laws=laws)
                present = {c["norm_id"] for c in candidates}
                candidates.extend(c for c in sparse if c["norm_id"] not in present
                                  and (not laws or c["norm_id"].split(":")[1] in laws))
            package_options = {"strategy": config["package"]} if config.get("package") else {}
            packages = {c["norm_id"]: index.norm_package(c["norm_id"], vector, **package_options)
                        for c in candidates}
            passage_scores = None
            if config.get("passage_rerank") and candidates:
                if preselector is None:
                    raise SemanticResourcesUnavailable("Passagen-Ranking benötigt eine Reranker-Vorauswahl")
                packages, passage_scores = _rank_passage_packages(
                    candidates, packages, evidence, vector, ranking_query, preselector,
                    limit=config.get("passage_pool", 8))
                if config.get("passage_shortlist") == "package":
                    # Window scores choose evidence only. Re-score the complete
                    # package with its catalog paraphrases to retain the bridge
                    # from everyday facts to the statutory wording. This also
                    # avoids ranking norms by maxima over unequal window counts.
                    passage_scores = None
            # A declared evidence generation may not silently fall back to a
            # different text representation for an unexplained missing norm.
            candidates = [c for c in candidates
                          if packages[c["norm_id"]] is not None and packages[c["norm_id"]].evidence]
            # Relevance ranking uses the existing search context; only the
            # factual questions bind this context to one explicitly named focus.
            pairs = [(ranking_query, packages[c["norm_id"]].text) for c in candidates]
            if config.get("cascade") and candidates:
                assert preselector is not None
                first_scores = ([passage_scores[c["norm_id"]] for c in candidates]
                                if passage_scores is not None else
                                _validated_scores(preselector, pairs, "Reranker-Vorauswahl"))
                order = sorted(range(len(candidates)),
                               key=lambda i: (-first_scores[i], candidates[i]["norm_id"]))[:config.get("shortlist", 20)]
                candidates = [{**candidate, "vorauswahl": float(score)}
                              for candidate, score in zip(candidates, first_scores, strict=True)]
                candidates = [candidates[i] for i in order]
                pairs = [pairs[i] for i in order]
            if config.get("agreement"):
                # This gate depends only on scores already available from BGE
                # and retrieval. Apply it AFTER the declared shortlist cap, but
                # BEFORE the expensive Qwen call; no output-eligible candidate
                # is lost and excluded slots are not refilled from outside it.
                eligible = [(c, pair) for c, pair in zip(candidates, pairs, strict=True)
                            if c["vorauswahl"] >= .1 or (
                                c["vorauswahl"] >= .01 and c["sim"] is not None and c["sim"] >= .55)]
                candidates = [c for c, _ in eligible]
                pairs = [pair for _, pair in eligible]
            if pairs and config.get("instruction"):
                scores = _checked_scores(reranker, pairs, config["instruction"])
            else:
                scores = _validated_scores(reranker, pairs, "Reranker-Bewertung") if pairs else []
            scored = [{**c, "rerank": float(score)} for c, score in zip(candidates, scores, strict=True)]
            selected = filtere_mit_rettung(
                scored, config.get("rerank_threshold", .10), config.get("sim_rescue", .55),
                rettung_min_rerank=.01)
            assessments = {}
            if checks and selected:
                from .semantic_checks import (
                    SUPPORT_INSTRUCTION, CONTRADICTION_INSTRUCTION,
                    CHECK_INSTRUCTION_VERSION, classify,
                )
                check_pairs = [(query, packages[c["norm_id"]].text) for c in selected]
                support = _checked_scores(reranker, check_pairs, SUPPORT_INSTRUCTION)
                contradiction = _checked_scores(reranker, check_pairs, CONTRADICTION_INSTRUCTION)
                for c, positive, negative in zip(selected, support, contradiction, strict=True):
                    assessments[c["norm_id"]] = {
                        "status": classify(positive, negative), "version": CHECK_INSTRUCTION_VERSION,
                        "unterstuetzung_score": positive, "widerspruch_score": negative,
                        "kalibriert": False,
                    }
                selected = [c for c in selected
                            if assessments[c["norm_id"]]["status"] == "modell_gestuetzt"]
            selected = selected[:5]
            norms = []
            for n in selected:
                package = packages[n["norm_id"]]
                source = package.evidence[0]
                norms.append({
                    "id": source.norm_id, "zitat": knoten_zitat(source.law, source.para),
                    "titel": source.title,
                    "sim": round(n["sim"], 3) if n["sim"] is not None else None,
                    "rerank": round(n["rerank"], 3), "gestuetzt": True, "im_korpus": True,
                    "ketten": chains.get(n["norm_id"], []),
                    "evidenz": {
                        "fingerprint": package.source_fingerprint,
                        "gekuerzt": package.truncated,
                        "passagen": [{"id": p.passage_id, "norm_id": p.norm_id,
                                      "quell_start": p.start, "quell_end": p.end, "text": p.text}
                                     for p in package.evidence],
                        "anker": list(package.anchors),
                    },
                })
                if checks:
                    norms[-1]["pruefung"] = assessments[n["norm_id"]]
            completed[cache_key] = norms
        if completed[cache_key]:
            result.append({"start": context.focus_start, "end": context.focus_end,
                           "satz": context.focus_text, "normen": completed[cache_key],
                           "profil": profile,
                           "kontext": {"start": context.context_start, "end": context.context_end,
                                       "text": context.query_text}})
    return result


def _rank_passage_packages(candidates, packages, evidence, vector, query,
                          preselector, *, limit: int):
    """Score source windows before selecting the norm's evidence.

    Also return per-norm window maxima for diagnostic configurations. The
    public ``treffer`` profile instead ranks the complete selected packages,
    including their existing catalog anchors, to choose the norm shortlist.
    """
    from .passage_selection import select_passage_candidates, package_ranked_passages

    views = {c["norm_id"]: select_passage_candidates(
        evidence, c["norm_id"], vector, query, limit=limit) for c in candidates}
    flat = [(nid, passage) for nid, passages in views.items() for passage in passages]
    pairs = [(query, f"{nid}\n\n{p.embedding_text}") for nid, p in flat]
    scores = _validated_scores(preselector, pairs, "Passagen-Vorauswahl") if pairs else []
    by_norm: dict[str, list[float]] = {nid: [] for nid in views}
    for (nid, _), score in zip(flat, scores, strict=True):
        by_norm[nid].append(score)
    ranked_packages, norm_scores = {}, {}
    for nid, passages in views.items():
        if not passages:
            continue
        previous = packages[nid]
        anchors = previous.anchors if previous is not None else ()
        ranked_packages[nid] = package_ranked_passages(
            passages, by_norm[nid], anchors, evidence.source_fingerprint)
        norm_scores[nid] = max(by_norm[nid])
    # Missing source remains missing; do not fabricate a low score or fall back
    # to a different, unrecorded representation.
    return {c["norm_id"]: ranked_packages.get(c["norm_id"]) for c in candidates}, norm_scores


def _checked_scores(reranker, pairs, instruction: str) -> list[float]:
    from numbers import Real
    raw = reranker.score_with_instruction(pairs, instruction=instruction, max_length=2048)
    if len(raw) != len(pairs):
        raise RuntimeError("Unvollständige fachliche Reranker-Prüfung")
    if any(isinstance(score, bool) or not isinstance(score, Real)
           or not math.isfinite(score) or not 0 <= float(score) <= 1 for score in raw):
        raise RuntimeError("Ungültige fachliche Reranker-Prüfung")
    return [float(score) for score in raw]


def _validated_scores(reranker, pairs, stage: str) -> list[float]:
    raw = reranker.score(pairs)
    if len(raw) != len(pairs):
        raise RuntimeError(f"Unvollständige {stage}")
    try:
        scores = [float(score) for score in raw]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Ungültige {stage}") from exc
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError(f"Nicht-endliche {stage}")
    return scores
