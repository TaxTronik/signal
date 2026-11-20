"""Audit every locked PyPI version, including upstream versions of vendor wheels."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def audit_pins(lock: Path) -> set[str]:
    pins: set[str] = set()
    # pip-compile writes one logical requirement including its hash options.
    content = lock.read_text(encoding="utf-8").replace("\\\n", " ")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--extra-index-url ", "--index-url ")):
            continue
        spec, *hashes = re.split(r"\s+--hash=", line)
        requirement = Requirement(spec.strip())
        name = canonicalize_name(requirement.name)
        if not hashes or not all(re.fullmatch(r"sha256:[a-fA-F0-9]{64}", digest.strip()) for digest in hashes):
            raise ValueError(f"{lock}: SHA-256 pin required for {name}")
        if requirement.url:
            url = urlsplit(requirement.url)
            filename = unquote(url.path.rsplit("/", 1)[-1])
            if filename.endswith(".whl"):
                artifact_name, version, _, _ = parse_wheel_filename(filename)
            else:
                artifact_name, version = parse_sdist_filename(filename)
            if artifact_name != name:
                raise ValueError(f"{lock}: artifact name differs from {name}")
            amd_component = name == "rocm" or name.startswith(
                ("rocm-", "amd-torch-device-", "amd-torchvision-device-")
            )
            if ((name == "de-core-news-lg" and url.hostname == "github.com")
                    or (amd_component and url.hostname in ("repo.radeon.com", "stable.repo.amd.com"))):
                print(f"{lock.name}: {name} has a vendor artifact hash, no upstream PyPI audit", flush=True)
                continue
        else:
            specs = list(requirement.specifier)
            if len(specs) != 1 or specs[0].operator != "==":
                raise ValueError(f"{lock}: exact pin required for {name}")
            version = Version(specs[0].version)
        # PyPI publishes advisories against e.g. torch 2.13.0, while official
        # CPU and ROCm builds carry local version suffixes such as +cpu.
        pins.add(f"{name}=={version.public}")
    if not pins:
        raise ValueError(f"{lock}: no auditable packages")
    return pins


def main() -> int:
    audited: set[str] = set()
    failed = False
    with tempfile.TemporaryDirectory() as temp:
        requirements = Path(temp) / "pins.txt"
        for lock in sorted(Path(__file__).resolve().parents[1].glob("requirements*lock.txt")):
            pins = audit_pins(lock) - audited
            if not pins:
                continue
            print(f"{lock.name}: checking {len(pins)} additional upstream versions", flush=True)
            requirements.write_text("\n".join(sorted(pins)) + "\n", encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "pip_audit", "--strict", "--disable-pip",
                                     "--no-deps", "-r", str(requirements)])
            failed |= result.returncode != 0
            audited.update(pins)
    if not audited:
        raise ValueError("No locked versions found")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
