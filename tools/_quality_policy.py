"""Four predeclared abstention comparisons, evaluated without model inference.

Apply to the complete scored sentence pool BEFORE Top-5. Inputs must contain
raw, unrounded model scores. Trigger support means membership in the independently
retrieved Trigger-Top20 of this same sentence; it must never come from Gold labels.
These experiment policies do not change production settings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any, TypedDict


class PolicySpec(TypedDict):
    rerank_threshold: float
    sim_rescue: float | None
    rescue_rerank_floor: float | None
    require_trigger_support: bool


POLICY_SPECS: dict[str, PolicySpec] = {
    "baseline_or": {"rerank_threshold": 0.10, "sim_rescue": 0.55,
                    "rescue_rerank_floor": None, "require_trigger_support": False},
    "rerank_only": {"rerank_threshold": 0.10, "sim_rescue": None,
                    "rescue_rerank_floor": None, "require_trigger_support": False},
    "weak_rescue": {"rerank_threshold": 0.10, "sim_rescue": 0.55,
                    "rescue_rerank_floor": 0.01, "require_trigger_support": False},
    "supported_rescue": {"rerank_threshold": 0.10, "sim_rescue": 0.55,
                         "rescue_rerank_floor": None, "require_trigger_support": True},
}


def _score(value: Any, field: str, norm_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ValueError(f"{norm_id}: {field} requires a finite raw score, got {value!r}")
    return float(value)


def apply_policy(candidates: Sequence[Mapping[str, Any]], policy_name: str, *,
                 trigger_ids: Iterable[str] | None = None,
                 topk: int = 5) -> list[dict[str, Any]]:
    """Filter and order the complete scored pool without mutating its members.

    All reranker scores are required: missing cached inference is an evaluation
    error, never an abstention. ``sim=None`` is valid for trigger-only candidates.
    Missing trigger provenance fails for the support policy instead of silently
    treating all candidates as unsupported. No rounded display rows as inputs.
    """
    if policy_name not in POLICY_SPECS:
        raise ValueError(f"Unknown abstention policy: {policy_name}")
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
        raise ValueError("topk must be a positive integer")
    policy = POLICY_SPECS[policy_name]
    if policy["require_trigger_support"] and trigger_ids is None:
        raise ValueError("supported_rescue requires the sentence's actual trigger_ids")
    support = set(trigger_ids or ())
    scored = []
    seen = set()
    for candidate in candidates:
        norm_id = candidate.get("norm_id")
        if not isinstance(norm_id, str) or not norm_id or norm_id in seen:
            raise ValueError(f"Candidate norm_id must be nonempty and unique: {norm_id!r}")
        seen.add(norm_id)
        rerank = _score(candidate.get("rerank"), "rerank", norm_id)
        sim = (_score(candidate["sim"], "sim", norm_id)
               if candidate.get("sim") is not None else None)
        rescued = policy["sim_rescue"] is not None and sim is not None \
            and sim >= policy["sim_rescue"]
        if policy["rescue_rerank_floor"] is not None:
            rescued = rescued and rerank >= policy["rescue_rerank_floor"]
        if policy["require_trigger_support"]:
            rescued = rescued and norm_id in support
        if rerank >= policy["rerank_threshold"] or rescued:
            scored.append((rerank, sim if sim is not None else float("-inf"),
                           norm_id, dict(candidate)))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in scored[:topk]]
