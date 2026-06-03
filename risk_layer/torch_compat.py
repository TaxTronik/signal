"""Kleine torch/ROCm-Kompatibilitätshelfer (geteilt von Encoder + Reranker)."""

from __future__ import annotations


def patch_rocm_dist() -> None:
    """ROCm-Workaround: sentence-transformers' ``get_device_name()`` ruft
    ``torch.distributed.is_initialized()`` auf, das in manchen torch-ROCm-Builds
    FEHLT (AttributeError beim Modell-Load). Wir patchen die Ursache an
    ``torch.distributed`` – aber NUR, wenn das Attribut wirklich fehlt, damit
    echtes Multi-GPU (wo es existiert) unberührt bleibt."""
    try:
        import torch.distributed as _dist
        if not hasattr(_dist, "is_initialized"):
            _dist.is_initialized = lambda: False
    except ImportError:
        pass
