#!/usr/bin/env python3
"""Baue das checkout-freie, stdlib-only Release-Bootstrap-Zipapp deterministisch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "risk_layer/dateisperre.py": ROOT / "risk_layer/dateisperre.py",
    "risk_layer/release_manifest.py": ROOT / "risk_layer/release_manifest.py",
    "risk_layer/release_lifecycle.py": ROOT / "risk_layer/release_lifecycle.py",
    "risk_layer/embedding_model.py": ROOT / "risk_layer/embedding_model.py",
    "release_tools/verify_container.py": (
        ROOT / "tools/verify_container_release_manifest.py"
    ),
    "release_tools/extract_native.py": ROOT / "tools/extract_release_bundle.py",
    "release_tools/deploy_container.py": ROOT / "tools/deploy_container_release.py",
    "risk_layer/operator_cli.py": ROOT / "risk_layer/operator_cli.py",
}
MAIN = '''\
from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {
        "verify-container", "extract-native", "deploy-container", "ops"
    }:
        print(
            "Aufruf: risk-layer-release-tools.pyz "
            "verify-container MANIFEST [--artifact-dir DIR] | "
            "extract-native ARCHIV ZIEL | "
            "deploy-container {deploy,rollback,status} ... | "
            "ops [--config DATEI] BEFEHL ...",
            file=sys.stderr,
        )
        return 2
    command = sys.argv.pop(1)
    if command == "verify-container":
        from release_tools.verify_container import main as command_main
        try:
            return command_main()
        except ValueError as exc:
            raise SystemExit(f"Deployment-Manifest abgelehnt: {exc}") from exc
    if command == "extract-native":
        from release_tools.extract_native import main as command_main
        return command_main()
    if command == "ops":
        from risk_layer.operator_cli import main as command_main
        return command_main()
    from release_tools.deploy_container import main as command_main
    return command_main()


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path, epoch: int) -> None:
    if epoch < 0:
        raise SystemExit("--source-date-epoch muss nichtnegativ sein")
    timestamp = datetime.fromtimestamp(epoch, timezone.utc)
    if timestamp.year < 1980:
        timestamp = datetime(1980, 1, 1, tzinfo=timezone.utc)
    zip_time = (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second - timestamp.second % 2,
    )
    entries = {
        "__main__.py": MAIN.encode("utf-8"),
        "risk_layer/__init__.py": b'"""Minimal package for release bootstrap."""\n',
        "release_tools/__init__.py": b"",
        **{name: path.read_bytes() for name, path in SOURCES.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary_name,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(entries, key=lambda value: value.encode("utf-8")):
                info = zipfile.ZipInfo(name, date_time=zip_time)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, entries[name], compresslevel=9)
        Path(temporary_name).replace(output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    Path(f"{output}.sha256").write_text(
        f"{_sha256(output)}  {output.name}\n",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    build(args.output, args.source_date_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
