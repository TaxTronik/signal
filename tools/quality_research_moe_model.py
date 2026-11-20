"""Download and inspect the pinned larger research candidate, without inference.

Only CPU metadata/tensor-directory inspection is performed. The proposed Vulkan
arguments are recorded for later review; this tool never starts a process.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REPO = "ggml-org/Qwen3.6-35B-A3B-GGUF"
REVISION = "baec3ebee244827cda0f4557eafa8b28f7545fa6"
FILENAME = "Qwen3.6-35B-A3B-Q4_K_M.gguf"
MODEL_BYTES = 20_419_565_568
MODEL_SHA = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7"
ORIGINAL_REPO = "Qwen/Qwen3.6-35B-A3B"
ORIGINAL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
ENGINE = ROOT / "engine/b9986/llama-b9986-bin-win-vulkan-x64/llama-server.exe"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def verified_model(path, size=MODEL_BYTES, sha=MODEL_SHA):
    path = Path(path)
    require(path.is_file() and path.stat().st_size == size, "Model file size differs from the pinned artifact")
    require(digest(path) == sha, "Model SHA256 differs from the pinned artifact")
    return {"path": str(path.resolve()), "bytes": size, "sha256": sha}


def validate_remote(metadata):
    require(metadata.get("id") == REPO and metadata.get("sha") == REVISION, "Remote model revision differs")
    files = [item for item in metadata.get("siblings", []) if item.get("rfilename") == FILENAME]
    require(len(files) == 1 and files[0].get("size") == MODEL_BYTES
            and files[0].get("lfs", {}).get("size") == MODEL_BYTES
            and files[0].get("lfs", {}).get("sha256") == MODEL_SHA, "Remote file metadata differs")


def fetch_json(url, target):
    with urlopen(url, timeout=60) as response:
        raw = response.read(4_194_305)
    require(len(raw) <= 4_194_304, "Oversized metadata")
    value = json.loads(raw)
    Path(target).write_bytes(raw)
    return value


class HeaderReader:
    """Read GGUF metadata/directory only, never tensor payloads or GPU libraries."""

    _types = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def __init__(self, stream):
        self.stream = stream

    def take(self, fmt):
        width = struct.calcsize("<" + fmt)
        raw = self.stream.read(width)
        require(len(raw) == width, "Truncated GGUF header")
        values = struct.unpack("<" + fmt, raw)
        return values[0] if len(values) == 1 else values

    def string(self, keep=True):
        length = self.take("Q")
        require(length <= 16_777_216, "Oversized GGUF string")
        raw = self.stream.read(length)
        require(len(raw) == length, "Truncated GGUF string")
        return raw.decode("utf-8") if keep else None

    def value(self, kind, keep=True):
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            element, count = self.take("IQ")
            require(count <= 10_000_000, "Oversized GGUF metadata array")
            if element in self._types and not keep:
                size = struct.calcsize("<" + self._types[element]) * count
                require(len(self.stream.read(size)) == size, "Truncated GGUF metadata array")
                return None
            values = [self.value(element, keep) for _ in range(count)]
            return values if keep else None
        require(kind in self._types, f"Unknown GGUF metadata type {kind}")
        value = self.take(self._types[kind])
        return value if keep else None


def inspect_gguf(path):
    """Return exact directory spans, with alignment padding explicitly included."""
    path = Path(path)
    with path.open("rb") as stream:
        reader = HeaderReader(stream)
        magic, version, n_tensors, n_meta = reader.take("4sIQQ")
        require(magic == b"GGUF" and version == 3, "Expected a little-endian GGUF v3 model")
        require(0 < n_tensors <= 100_000 and 0 < n_meta <= 10_000, "Unexpected GGUF inventory size")
        metadata = {}
        keys = set()
        for _ in range(n_meta):
            key = reader.string()
            require(key not in keys, "Duplicate GGUF metadata key")
            keys.add(key)
            keep = not key.startswith("tokenizer.")
            value = reader.value(reader.take("I"), keep)
            if keep:
                metadata[key] = value
        tensors = []
        names = set()
        for _ in range(n_tensors):
            name, dimensions = reader.string(), reader.take("I")
            require(name not in names and 1 <= dimensions <= 4, "Invalid or duplicate GGUF tensor")
            names.add(name)
            shape = [reader.take("Q") for _ in range(dimensions)]
            require(all(0 < dim <= 10_000_000 for dim in shape), "Invalid tensor dimensions")
            kind, offset = reader.take("IQ")
            tensors.append({"name": name, "shape_ggml": shape, "ggml_type": kind, "relative_offset": offset})
        alignment = metadata.get("general.alignment", 32)
        require(type(alignment) is int and 1 <= alignment <= 4096 and alignment & (alignment - 1) == 0,
                "Invalid GGUF data alignment")
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
    ordered = sorted(tensors, key=lambda item: item["relative_offset"])
    require(ordered[0]["relative_offset"] == 0, "Unexpected gap before tensor data")
    data_size = path.stat().st_size - data_start
    for index, tensor in enumerate(ordered):
        end = ordered[index + 1]["relative_offset"] if index + 1 < len(ordered) else data_size
        require(end > tensor["relative_offset"] and end <= data_size, "Invalid tensor storage range")
        tensor["storage_bytes_including_alignment"] = end - tensor["relative_offset"]
    return {"version": version, "tensor_count": n_tensors, "metadata_count": n_meta,
            "metadata": metadata, "metadata_keys": sorted(keys), "data_start": data_start,
            "tensor_storage_bytes": data_size, "tensors": ordered,
            "tensor_ggml_types": dict(Counter(str(t["ggml_type"]) for t in ordered)),
            "storage_scope": "Actual GGUF directory spans including inter-tensor alignment; no runtime allocation claim."}


def offload_plan(inventory, cpu_expert_layers=16):
    meta = inventory["metadata"]
    require(meta.get("general.architecture") == "qwen35moe", "GGUF architecture is not Qwen3.5 MoE")
    require(meta.get("qwen35moe.block_count") == 40 and meta.get("qwen35moe.embedding_length") == 2048,
            "GGUF layer count or hidden width differs")
    require(type(cpu_expert_layers) is int and 1 <= cpu_expert_layers <= 40, "Invalid CPU expert layer count")
    routed = [t for t in inventory["tensors"] if re.fullmatch(r"blk\.\d+\.ffn_(?:gate|up|down|gate_up)_exps\.weight", t["name"])]
    layers = {int(t["name"].split(".")[1]) for t in routed}
    require(layers == set(range(40)), "Routed-expert tensor inventory is incomplete")
    cpu_bytes = sum(t["storage_bytes_including_alignment"] for t in routed if int(t["name"].split(".")[1]) < cpu_expert_layers)
    total = inventory["tensor_storage_bytes"]
    return {"status": "planned_not_run", "cpu_expert_layers": cpu_expert_layers,
            "candidate_flags": ["--gpu-layers", "all", "--n-cpu-moe", str(cpu_expert_layers),
                                "--device", "Vulkan0", "--parallel", "1", "--ctx-size", "16384",
                                "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
                                "--jinja", "--reasoning", "on", "--reasoning-format", "deepseek",
                                "--reasoning-budget", "768", "--no-mmproj", "--offline"],
            "routed_expert_storage_bytes": sum(t["storage_bytes_including_alignment"] for t in routed),
            "proposed_cpu_expert_storage_bytes": cpu_bytes,
            "remaining_tensor_storage_upper_bound_bytes": total - cpu_bytes,
            "attention_kv_bytes_at_16k_f16": 2 * 10 * 2 * 256 * 16384 * 2,
            "notes": ["A later owned-server launch requires separate authorization; no launch is performed here.",
                      "Stop/release the existing 9B server before a larger-model GPU trial; no simultaneous-residency claim.",
                      "The remaining-weight figure excludes KV, recurrent state, scratch buffers, backend repacking and driver allocations.",
                      "Exact backend tensor placement and actual peak memory must be checked in the eventual server log.",
                      "Use identical source tools, cases and generation budgets for model comparisons. No legal-quality claim follows from parameter count.",
                      "CPU expert execution may limit prompt and decode throughput; no token-rate estimate is treated as measured."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/research-qwen36-35b-a3b")
    parser.add_argument("--out", type=Path, default=ROOT / "out/quality-research-moe-model-20260907")
    args = parser.parse_args()
    out, directory = args.out.resolve(), args.model_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    directory.mkdir(parents=True, exist_ok=True)
    source_sha = digest(__file__)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    report = {"format": "quality_research_moe_setup/1", "status": "preflight", "gpu_used": False,
              "model_loaded": False, "server_started": False, "started_at": utcnow(), "source_sha256": source_sha}
    try:
        remote = fetch_json(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", out / "remote-model.json")
        validate_remote(remote)
        config = fetch_json(f"https://huggingface.co/{ORIGINAL_REPO}/resolve/{ORIGINAL_REVISION}/config.json", out / "original-config.json")
        text_config = config.get("text_config", {})
        require(config.get("model_type") == "qwen3_5_moe" and text_config.get("num_hidden_layers") == 40
                and text_config.get("num_experts") == 256 and text_config.get("num_experts_per_tok") == 8,
                "Original architecture metadata differs")
        report["original_config"] = {"repo": ORIGINAL_REPO, "revision": ORIGINAL_REVISION,
                                     "sha256": digest(out / "original-config.json"),
                                     "scope": "Pinned current upstream architectural reference, not a claim about the converter's exact source revision."}
        save(out / "report.json", report)
        target = directory / FILENAME
        existed = target.exists()
        if not existed:
            from huggingface_hub import hf_hub_download
            print("Downloading pinned 20.420 GB MoE GGUF; no model or GPU runtime loaded.", flush=True)
            result = Path(hf_hub_download(REPO, FILENAME, revision=REVISION, local_dir=directory))
            require(result.resolve() == target, "Downloader returned an unexpected destination")
        verified = verified_model(target)
        print("Complete file SHA256 verified; reading only the GGUF header and tensor directory.", flush=True)
        inventory = inspect_gguf(target)
        plan = offload_plan(inventory)
        save(out / "gguf-inventory.json", inventory)
        save(out / "offload-plan.json", plan)
        manifest = {"format": "quality_research_model/1", "status": "complete", "repo": REPO,
                    "revision": REVISION, "filename": FILENAME, "quantization": "Q4_K_M", "model": verified,
                    "already_present_and_verified": existed, "started_at": report["started_at"], "verified_at": utcnow(),
                    "gpu_used": False, "source_url": f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}",
                    "producer": {"path": str(Path(__file__).resolve()), "sha256": source_sha},
                    "transfer": {"library": "huggingface_hub", "disable_xet": os.environ.get("HF_HUB_DISABLE_XET") == "1"}}
        path = directory / "model-manifest.json"
        if path.exists():
            old = json.loads(path.read_bytes())
            require(all(old.get(key) == manifest[key] for key in ("repo", "revision", "filename", "quantization", "model")),
                    "Existing manifest differs; it is not overwritten")
        else:
            save(path, manifest)
        (out / "model-manifest.json").write_bytes(path.read_bytes())
        engine_files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                        for p in sorted(ENGINE.parent.iterdir()) if p.suffix.lower() in (".dll", ".exe")}
        save(out / "engine-inventory.json", {"path": str(ENGINE), "files": engine_files, "runtime_started": False})
        require(digest(__file__) == source_sha, "Setup source changed during download")
        report.update(status="complete", model=verified, model_manifest_sha256=digest(path), offload_plan=plan)
    except Exception as error:
        report.update(status="failed", error=repr(error))
    report["completed_at"] = utcnow()
    report["artifact_hashes"] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "report.json"}
    save(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
