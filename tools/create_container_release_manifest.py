#!/usr/bin/env python3
"""Binde OCI-Image und Festwissen-Artefakt an ein Container-Deployment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from risk_layer.release_manifest import ReleaseManifest, validate_artifact_path


SCHEMA = "taxtronik-release/v1"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
BOOTSTRAP_KEYS = frozenset(
    {
        "release_tools",
        "verify_container",
        "extract_native",
        "deploy_container",
        "install_release_service",
        "release_launcher",
        "windows_install_service",
        "windows_restart_task",
        "windows_install_backup_task",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _artifact_block(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"kein reguläres Release-Artefakt: {path}")
    validate_artifact_path(path.name)
    return {"artifact": path.name, "sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--engine-repository", required=True)
    parser.add_argument("--engine-digest", required=True)
    parser.add_argument("--festwissen", type=Path, required=True)
    parser.add_argument("--festwissen-manifest", type=Path, required=True)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-release-manifest", type=Path, required=True)
    parser.add_argument("--compose", type=Path, default=Path("docker-compose.yml"))
    parser.add_argument("--release-tools", type=Path, required=True)
    parser.add_argument("--verify-container-tool", type=Path, required=True)
    parser.add_argument("--extract-native-tool", type=Path, required=True)
    parser.add_argument("--deploy-container-tool", type=Path, required=True)
    parser.add_argument("--install-release-service", type=Path, required=True)
    parser.add_argument("--release-launcher", type=Path, required=True)
    parser.add_argument("--windows-install-service", type=Path, required=True)
    parser.add_argument("--windows-restart-task", type=Path, required=True)
    parser.add_argument("--windows-install-backup-task", type=Path, required=True)
    parser.add_argument("--llm-model", type=Path)
    parser.add_argument("--llm-revision")
    parser.add_argument("--llm-provenance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not GIT_SHA_RE.fullmatch(args.git_sha):
        raise SystemExit("ungültiger Git-SHA")
    if not DIGEST_RE.fullmatch(args.engine_digest):
        raise SystemExit("ungültiger OCI-Digest")
    llm_values = (args.llm_model, args.llm_revision, args.llm_provenance)
    if any(value is not None for value in llm_values) and not all(
        value is not None for value in llm_values
    ):
        raise SystemExit(
            "--llm-model, --llm-revision und --llm-provenance nur gemeinsam angeben"
        )
    if args.llm_model is not None:
        if not args.llm_model.is_file() or args.llm_model.is_symlink():
            raise SystemExit(f"kein reguläres LLM-Modell: {args.llm_model}")
        validate_artifact_path(args.llm_model.name)
        for label, value in (
            ("Revision", args.llm_revision),
            ("Provenance", args.llm_provenance),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 2048
                or value.strip() != value
                or any(ord(character) < 32 for character in value)
            ):
                raise SystemExit(f"ungültige LLM-{label}")
    inner = json.loads(args.festwissen_manifest.read_text(encoding="utf-8"))
    if inner.get("schema") != "taxtronik.festwissen-release/v1":
        raise SystemExit("unerwartetes Festwissen-Manifest-Schema")
    if inner.get("release_id") != args.release_id or inner.get("git_sha") != args.git_sha:
        raise SystemExit("Festwissen und Container haben verschiedene Release-ID/Git-SHA")
    native_manifest = ReleaseManifest.load(args.native_release_manifest)
    if native_manifest.release_id != args.release_id or native_manifest.git_sha != args.git_sha:
        raise SystemExit("natives Release-Manifest widerspricht Release-ID/Git-SHA")

    bootstrap = {
        "release_tools": _artifact_block(args.release_tools),
        "verify_container": _artifact_block(args.verify_container_tool),
        "extract_native": _artifact_block(args.extract_native_tool),
        "deploy_container": _artifact_block(args.deploy_container_tool),
        "install_release_service": _artifact_block(args.install_release_service),
        "release_launcher": _artifact_block(args.release_launcher),
        "windows_install_service": _artifact_block(args.windows_install_service),
        "windows_restart_task": _artifact_block(args.windows_restart_task),
        "windows_install_backup_task": _artifact_block(
            args.windows_install_backup_task
        ),
    }
    if set(bootstrap) != BOOTSTRAP_KEYS:
        raise AssertionError("interner Bootstrap-Vertrag ist inkonsistent")

    document = {
        "schema": SCHEMA,
        "release_id": args.release_id,
        "git_sha": args.git_sha,
        "created_at": datetime.fromtimestamp(args.source_date_epoch, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "engine": {
            "repository": args.engine_repository,
            "digest": args.engine_digest,
            "image": f"{args.engine_repository}@{args.engine_digest}",
        },
        "festwissen": {
            "artifact": args.festwissen.name,
            "sha256": _sha256(args.festwissen),
            "manifest": args.festwissen_manifest.name,
            "manifest_sha256": _sha256(args.festwissen_manifest),
            "embedding": inner.get("embedding") is True,
        },
        "native_bundle": {
            "artifact": args.native_bundle.name,
            "sha256": _sha256(args.native_bundle),
            "manifest_sha256": native_manifest.fingerprint(),
        },
        "compose": {
            "artifact": args.compose.name,
            "sha256": _sha256(args.compose),
        },
        "bootstrap": bootstrap,
    }
    if args.llm_model is not None:
        document["llm_model"] = {
            "artifact": args.llm_model.name,
            "sha256": _sha256(args.llm_model),
            "revision": args.llm_revision,
            "provenance": args.llm_provenance,
        }
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    _write_atomic(args.output, data)
    _write_atomic(
        Path(f"{args.output}.sha256"),
        f"{hashlib.sha256(data).hexdigest()}  {args.output.name}\n".encode("ascii"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
