"""Run one frozen TIGER transfer feature case on Chiamaka; never scores references."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root, args.output
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((root / "features/tiger_15/PROTOCOL.json").read_text())
    plan_path = root / "inputs/tiger_15/PLAN.json"
    assert protocol["cases"] == ["tiger_15"]
    assert digest(plan_path) == protocol["source_plans"][str(plan_path.parent)]
    assert protocol["multiscale_factors"] == [1, 2, 4]
    assert protocol["gpu_memory_fraction"] == 0.4
    for relative in (
        "experiments/virchow2_multi/run.py",
        "experiments/virchow2_inverse/gpu_ideas.py",
        "experiments/virchow2_inverse/common.py",
    ):
        assert digest(args.source / relative) == protocol["code_sha256"][relative]
    assert digest(args.model / "config.json") == protocol["encoder_config_sha256"]
    assert digest(args.model / "model.safetensors") == protocol["encoder_checkpoint_sha256"]
    plan = json.loads(plan_path.read_text())
    sample = plan["samples"][0]
    case_dir = plan_path.parent / "cases/tiger_15"
    assert sample["case"] == "tiger_15" and sample["target_reference_read"] is False
    assert digest(case_dir / "coarse.tif") == sample["coarse_sha256"]
    assert digest(case_dir / "original.png") == sample["original_sha256"]
    assert sample["working_mpp_xy"] == [0.5, 0.5]
    image = np.asarray(Image.open(case_dir / "original.png").convert("RGB"))
    assert list(image.shape[:2]) == sample["working_shape_yx"]
    sys.path.insert(0, str(args.source))
    import torch
    from experiments.virchow2_inverse.gpu_ideas import load_model
    from experiments.virchow2_multi.run import encode

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.4)
    output.mkdir(parents=True)
    atomic_json(output / "START.json", {
        "pid": os.getpid(), "pgid": os.getpgid(0), "sid": os.getsid(0),
        "proc_startticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
        "start_unix": time.time(), "case": "tiger_15", "source_plan_sha256": digest(plan_path),
        "checkpoint_sha256": protocol["encoder_checkpoint_sha256"],
        "source_sha256": {r: digest(args.source / r) for r in (
            "experiments/virchow2_multi/run.py", "experiments/virchow2_inverse/gpu_ideas.py")},
        "gpu_name": torch.cuda.get_device_name(0), "reference_read": False,
    })
    model = load_model(args.model)
    entries = {}
    start = time.monotonic()
    for name, factor in (("base", 1), ("scale_1", 2), ("scale_2", 4)):
        if time.monotonic() - start > 240:
            raise TimeoutError("Application deadline before next encoder scale")
        resized = image if factor == 1 else np.asarray(Image.fromarray(image).resize(
            (round(image.shape[1] / factor), round(image.shape[0] / factor)), Image.Resampling.LANCZOS))
        values, metadata = encode(model, resized)
        expected = tuple((np.asarray(resized.shape[:2]) + 13) // 14)
        assert values.shape == (*expected, 1280) and np.isfinite(values).all()
        tmp = output / f"{name}.tmp.npy"
        final = output / f"{name}.npy"
        np.save(tmp, values)
        os.replace(tmp, final)
        entries[name] = {"shape": list(values.shape), "image_shape_yx": list(resized.shape[:2]),
                         "sha256": digest(final), **metadata}
        atomic_json(output / f"{name}.json", entries[name])
        print(json.dumps({"checkpoint": name, "seconds": round(time.monotonic() - start, 3)}), flush=True)
    atomic_json(output / "FEATURES_COMPLETE.json", {
        "case": "tiger_15", "files": entries, "source_plan_sha256": digest(plan_path),
        "checkpoint_sha256": protocol["encoder_checkpoint_sha256"],
        "elapsed_seconds": time.monotonic() - start, "reference_read": False,
    })
    print(json.dumps({"complete": True, "elapsed_seconds": round(time.monotonic() - start, 3)}), flush=True)


if __name__ == "__main__":
    main()
