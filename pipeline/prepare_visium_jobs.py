#!/usr/bin/env python3
"""Freeze the three H&E Visium source-image Virchow2 queues."""
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
        raise ValueError("New output file and 1..1024 tiles per unit required")
    root = root.resolve()
    data_root = root / "visium-spot-benchmark/data"
    downloads = json.loads((data_root / "download_status.json").read_text())
    cases = []
    for name in ("breast_giotto", "colorectal_seurat", "tonsil_spatialexperiment"):
        images = list((data_root / name).glob("*tissue_image.*"))
        if len(images) != 1:
            raise ValueError(f"Expected one original tissue image for {name}")
        image = images[0]
        expected = downloads[name][image.name]["sha256"]
        plan_dir = root / "preprocessing/virchow2-visium-spot-plans-v1" / name
        plan = json.loads((plan_dir / "PLAN_COMPLETE.json").read_text())
        if Path(plan["image"]).resolve() != image.resolve():
            raise ValueError(f"Tile plan belongs to a different image: {name}")
        if name == "breast_giotto" and not plan["source_upsampled"]:
            raise ValueError("Breast source's explicit slight upsampling was not recorded")
        if name != "breast_giotto" and plan["source_upsampled"]:
            raise ValueError("Unexpected source upsampling")
        units = [dict(first_tile=i, last_tile=min(i + unit_tiles, plan["tiles"]))
                 for i in range(0, plan["tiles"], unit_tiles)]
        cases.append(dict(case=name, image=str(image), image_sha256=expected,
                          plan_dir=str(plan_dir),
                          plan_sha256=digest(plan_dir / "PLAN_COMPLETE.json"),
                          tile_table_sha256=digest(plan_dir / plan["tile_table"]),
                          mpp_xy=plan["source_mpp_xy"], native_tile_wh=plan["native_tile_wh"],
                          source_upsampled=plan["source_upsampled"],
                          grid_tiles=plan["tiles"], units=units,
                          gene_expression_used_for_encoding=False))
    result = dict(format="bora-virchow2-visium-queue-v1", root=str(root),
                  model_revision="3158645804b69e3f3bc4439d4116edddf0840a72",
                  model_sha256="8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b",
                  target_mpp=.5, tile_pixels=224, unit_tiles=unit_tiles,
                  cases=cases, total_units=sum(len(c["units"]) for c in cases),
                  image_tile_files_retained=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(output)
    return dict(cases=len(cases), total_units=result["total_units"],
                total_grid_tiles=sum(c["grid_tiles"] for c in cases),
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
