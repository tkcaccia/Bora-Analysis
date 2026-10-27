#!/usr/bin/env python3
"""Freeze calibrated WSI Virchow2 work units without starting a GPU job."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root: Path, output: Path, unit_tiles: int) -> dict:
    if output.exists() or not 1 <= unit_tiles <= 1024:
        raise ValueError("Use a new output file and 1..1024 tiles per GPU unit")
    root = root.resolve()
    with (root / "wsi_benchmark/cases.tsv").open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 5 or len({r["case"] for r in rows}) != 5:
        raise ValueError("The frozen WSI index must contain five unique cases")
    cases = []
    for row in rows:
        case = row["case"]
        image = root / row["source_image"]
        annotation = root / row["annotation_geojson"]
        if not image.is_file() or not annotation.is_file():
            raise FileNotFoundError(case)
        plan_dir = root / "preprocessing/virchow2-wsi-plans-v3" / case
        plan = json.loads((plan_dir / "PLAN_COMPLETE.json").read_text())
        if Path(plan["image"]).resolve() != image.resolve():
            raise ValueError("WSI plan image changed: " + case)
        if any(abs(float(row[key]) - plan["source_mpp_xy"][i]) > 1e-12
               for i, key in enumerate(("mpp_x", "mpp_y"))):
            raise ValueError("Exact OME microscopy spacing differs: " + case)
        if case == "human_colon_cancer":
            receipt = json.loads((image.parent.parent / "dataset_manifest.json").read_text())
            expected = receipt["source_image"]["sha256"]
        else:
            receipt = json.loads((image.parent.parent / "TRANSFER_COMPLETE.json").read_text())
            expected = receipt["image"]["sha256"]
        if image.stat().st_size < 1_000_000 or not expected:
            raise ValueError("Missing verified WSI source bytes: " + case)
        units = [dict(first_tile=first, last_tile=min(first + unit_tiles, plan["tiles"]))
                 for first in range(0, plan["tiles"], unit_tiles)]
        cases.append(dict(case=case, image=str(image), image_sha256=expected,
                          annotation=str(annotation), annotation_used_for_encoding=False,
                          plan_dir=str(plan_dir),
                          plan_sha256=sha256(plan_dir / "PLAN_COMPLETE.json"),
                          tile_table_sha256=sha256(plan_dir / plan["tile_table"]),
                          mpp_xy=plan["source_mpp_xy"], native_tile_wh=plan["native_tile_wh"],
                          grid_tiles=plan["tiles"], units=units,
                          scoring_status=row["scoring_status"]))
    frozen = dict(format="bora-virchow2-wsi-queue-v1", root=str(root),
                  model_revision="3158645804b69e3f3bc4439d4116edddf0840a72",
                  model_sha256="8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b",
                  target_mpp=0.5, tile_pixels=224, unit_tiles=unit_tiles,
                  cases=cases, total_units=sum(len(c["units"]) for c in cases),
                  expert_geojson_used_for_inference=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text(json.dumps(frozen, indent=2) + "\n")
    temp.replace(output)
    return dict(cases=len(cases), total_units=frozen["total_units"],
                total_grid_tiles=sum(c["grid_tiles"] for c in cases),
                output=str(output), sha256=sha256(output))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unit-tiles", type=int, default=1024)
    args = p.parse_args()
    print(json.dumps(prepare(args.root, args.output, args.unit_tiles), indent=2))


if __name__ == "__main__":
    main()
