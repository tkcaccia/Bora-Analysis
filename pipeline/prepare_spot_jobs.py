#!/usr/bin/env python3
"""Freeze the three Visium spot-centered Virchow2 feature queues."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root: Path, output: Path, unit_tiles: int) -> dict:
    if output.exists() or not 1 <= unit_tiles <= 1024:
        raise ValueError("Use a new queue path and 1..1024 tiles per unit")
    root = root.resolve()
    status = json.loads((root / "visium-spot-benchmark/data/download_status.json").read_text())
    cases = []
    for name in ("breast_giotto", "colorectal_seurat", "tonsil_spatialexperiment"):
        folder = root / "preprocessing/virchow2-visium-spot-centered-plans-v1" / name
        plan = json.loads((folder / "PLAN_COMPLETE.json").read_text())
        image = Path(plan["image"])
        if (plan["format"] != "wsiTools-virchow2-spot-plan-v1" or
                not image.is_file() or not Path(plan["spots"]).is_file() or
                plan["planned_spots"] < 16):
            raise ValueError("Incomplete spot plan: " + name)
        source_sha = status[name][image.name]["sha256"]
        units = [dict(first_tile=i, last_tile=min(i + unit_tiles, plan["planned_spots"]))
                 for i in range(0, plan["planned_spots"], unit_tiles)]
        cases.append(dict(case=name, image=str(image), image_sha256=source_sha,
                          plan_dir=str(folder),
                          plan_sha256=digest(folder / "PLAN_COMPLETE.json"),
                          tile_table_sha256=digest(folder / plan["tile_table"]),
                          spots_sha256=digest(Path(plan["spots"])),
                          planned_spots=plan["planned_spots"],
                          source_mpp_xy=plan["source_mpp_xy"],
                          source_upsampled=plan["source_upsampled"],
                          units=units, tile_images_saved=False))
    data = dict(format="bora-virchow2-spot-queue-v1", root=str(root),
                model_revision="3158645804b69e3f3bc4439d4116edddf0840a72",
                model_sha256="8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b",
                target_mpp=.5, tile_pixels=224, unit_tiles=unit_tiles,
                cases=cases, total_units=sum(len(c["units"]) for c in cases),
                gene_expression_used_for_encoding=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(output)
    return dict(cases=len(cases), total_units=data["total_units"],
                total_spots=sum(c["planned_spots"] for c in cases),
                output=str(output), sha256=digest(output))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unit-tiles", type=int, default=1024)
    args = p.parse_args()
    print(json.dumps(prepare(args.root, args.output, args.unit_tiles), indent=2))


if __name__ == "__main__":
    main()
