"""Launch one already-downloaded research model with the verified smoke helper."""

import argparse
from pathlib import Path

from tools import quality_research_model as small
from tools import quality_research_moe_serve as large


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("9b", "35b"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.model == "9b":
        directory = ROOT / "models/research-qwen35-9b"
        manifest = directory / "model-manifest.json"
        model = directory / small.FILENAME
        small.verify_file(model)
        # Reuse the frozen existing manifest; never rewrite a download receipt.
        report = small.launch_and_smoke(model, manifest, small.DEFAULT_ENGINE, args.out)
    else:
        model, manifest, binding = large.bind_download(
            large.DEFAULT_MODEL_DIR, ROOT / "out/quality-research-moe-model-20260907/report.json")
        report = large.launch_and_smoke(model, manifest, large.DEFAULT_ENGINE, args.out, binding)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
