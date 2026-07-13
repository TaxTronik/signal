#!/usr/bin/env python3
"""Rendert und installiert die gehärtete systemd-User-Unit für Releases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from urllib.parse import urlparse


TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:@/+-]+$")


def _real_path(value: str, description: str, *, directory: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{description} muss absolut sein: {path}")
    info: os.stat_result | None = None
    for component in reversed((path, *path.parents)):
        try:
            component_info = os.lstat(component)
        except OSError as exc:
            raise SystemExit(f"{description} fehlt: {component}: {exc}") from exc
        if stat.S_ISLNK(component_info.st_mode):
            raise SystemExit(
                f"{description} enthält einen Symlink: {component}"
            )
        if component == path:
            info = component_info
    assert info is not None
    if directory and not stat.S_ISDIR(info.st_mode):
        raise SystemExit(f"{description} ist kein Verzeichnis: {path}")
    if not directory and not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"{description} ist keine reguläre Datei: {path}")
    return path


def _systemd_quote(value: str) -> str:
    if not value or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SystemExit("Unit-Werte dürfen nicht leer sein oder Steuerzeichen enthalten")
    if "$" in value:
        raise SystemExit("Unit-Werte dürfen kein Dollarzeichen enthalten")
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
    )


def render(
    *,
    template: Path,
    output: Path,
    python: Path,
    launcher: Path,
    release_home: Path,
    berater: Path,
    env_file: Path,
    host: str,
    port: int,
    llm_url: str,
    emb_device: str,
) -> None:
    if not 1 <= port <= 65535:
        raise SystemExit("Port muss zwischen 1 und 65535 liegen")
    if not TOKEN_RE.fullmatch(host) or not TOKEN_RE.fullmatch(emb_device):
        raise SystemExit("Host/Embedding-Device enthält unzulässige Zeichen")
    parsed = urlparse(llm_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit("LLM-URL muss eine vollständige HTTP(S)-URL sein")
    if parsed.username or parsed.password or parsed.fragment:
        raise SystemExit("LLM-URL darf keine Credentials oder Fragmente enthalten")

    text = template.read_text(encoding="utf-8")
    replacements = {
        "@PYTHON@": str(python),
        "@LAUNCHER@": str(launcher),
        "@RELEASE_HOME@": str(release_home),
        "@BERATER@": str(berater),
        "@ENVFILE@": str(env_file),
        "@HOST@": host,
        "@PORT@": str(port),
        "@LLM_URL@": llm_url,
        "@EMB_DEVICE@": emb_device,
    }
    for placeholder, value in replacements.items():
        if text.count(placeholder) == 0:
            raise SystemExit(f"Unit-Vorlage enthält {placeholder} nicht")
        text = text.replace(placeholder, _systemd_quote(value))
    if re.search(r"@[A-Z_]+@", text):
        raise SystemExit("Unit-Vorlage enthält nicht ersetzte Platzhalter")

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _real_path(str(output.parent), "Unit-Ausgabeverzeichnis", directory=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, output)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument(
        "--launcher",
        default=str(Path(__file__).with_name("release-launcher.py")),
    )
    parser.add_argument("--release-home", required=True)
    parser.add_argument("--berater", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080")
    parser.add_argument("--emb-device", default="cpu")
    parser.add_argument(
        "--output",
        default=str(Path.home() / ".config/systemd/user/risk-layer-release.service"),
    )
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    python = _real_path(args.python, "System-Python", directory=False)
    if not os.access(python, os.X_OK):
        raise SystemExit(f"System-Python ist nicht ausführbar: {python}")
    launcher = _real_path(args.launcher, "Release-Launcher", directory=False)
    release_home = _real_path(args.release_home, "Release-Home", directory=True)
    berater = _real_path(args.berater, "Berater-Verzeichnis", directory=True)
    env_file = _real_path(args.env_file, "Environment-Datei", directory=False)
    if os.name != "nt" and stat.S_IMODE(os.lstat(env_file).st_mode) & 0o077:
        raise SystemExit(f"Environment-Datei muss Modus 0600 haben: {env_file}")
    output = Path(args.output).expanduser().absolute()
    template = Path(__file__).with_name("risk-layer-release.service.in")
    render(
        template=template,
        output=output,
        python=python,
        launcher=launcher,
        release_home=release_home,
        berater=berater,
        env_file=env_file,
        host=args.host,
        port=args.port,
        llm_url=args.llm_url,
        emb_device=args.emb_device,
    )

    subprocess.run(["systemd-analyze", "--user", "verify", str(output)], check=True)
    if not args.render_only:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", output.name], check=True
        )
    print(f"installiert: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
