#!/usr/bin/env python3
"""Snapshotte das gepinnte BGE-M3-PyTorch-Modell als lokales Release-Artefakt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from risk_layer.embedding_model import (
    BGE_M3_REPO_ID,
    BGE_M3_REQUIRED_FILES,
    BGE_M3_REVISION,
    validate_local_bge_m3,
    write_model_manifest,
)


EXCLUDED_ALTERNATIVES = (
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.onnx_data",
)


def snapshot(output: Path) -> Path:
    if output.exists():
        raise SystemExit(f"Modell-Ziel existiert bereits: {output}")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "BGE-M3-Snapshot braucht huggingface-hub aus dem Embedding-Lock"
        ) from exc

    info = HfApi().model_info(BGE_M3_REPO_ID, revision=BGE_M3_REVISION)
    if info.sha != BGE_M3_REVISION:
        raise SystemExit(
            f"Hugging Face löste Revision unerwartet auf: {info.sha!r}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    download = Path(tempfile.mkdtemp(prefix="risk-layer-hf-download-"))
    try:
        snapshot_download(
            repo_id=BGE_M3_REPO_ID,
            revision=BGE_M3_REVISION,
            local_dir=download,
            allow_patterns=list(BGE_M3_REQUIRED_FILES),
            ignore_patterns=list(EXCLUDED_ALTERNATIVES),
            # Der Hub-Client kann bei parallelen Xet/LFS-Futures nach bereits
            # vollstaendig geschriebenen Dateien auf einen nie finalisierten
            # Worker warten. Ein Worker ist fuer den einmaligen Release-Build
            # schnell genug und macht Abschluss/Cleanup deterministisch.
            max_workers=1,
        )
        for relative in BGE_M3_REQUIRED_FILES:
            source = download / Path(*PurePosixPath(relative).parts)
            if not source.is_file():
                raise SystemExit(f"Pflichtdatei fehlt im HF-Snapshot: {relative}")
            target = stage / Path(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            # copyfile dereferenziert eventuelle Hub-Cachelinks; das Release
            # enthält ausschließlich eigenständige reguläre Dateien.
            shutil.copyfile(source, target)
            os.chmod(target, 0o644)
        write_model_manifest(stage)
        validate_local_bge_m3(stage, verify_hashes=True)
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(download, ignore_errors=True)
    print(f"BGE-M3 offline gesnapshottet: {output} @ {BGE_M3_REVISION}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
