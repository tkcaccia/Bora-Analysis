#!/usr/bin/env python3
"""Run the released Bora prepare/train/refine CLI on registered cases.

The manifest points to already calibrated H&E, anonymous tile labels,
Virchow2 spatial tokens and tissue support. Expert masks/GeoJSON are read
only for the training images. This script contains no model-selection code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(base: Path, value: str) -> Path:
    path = (base / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def command(parts: list[str]) -> None:
    print(" ".join(parts), flush=True)
    subprocess.run(parts, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bora", default="bora", help="installed Bora 0.1 executable")
    args = parser.parse_args()
    spec_path = args.workflow.resolve()
    base = spec_path.parent
    spec = json.loads(spec_path.read_text())
    if spec.get("format") != "bora-analysis-workflow-v1":
        raise ValueError("Expected bora-analysis-workflow-v1")
    if not spec.get("classes") or not spec.get("training") or not spec.get("targets"):
        raise ValueError("At least one training case and one target are required")
    training_groups = [row["source_group"] for row in spec["training"]]
    target_groups = [row["source_group"] for row in spec["targets"]]
    if set(training_groups) & set(target_groups):
        raise ValueError("Training patients/slides must be absent from targets")
    if args.output.exists():
        raise FileExistsError("Use a new output directory; no result is overwritten")
    output = args.output.resolve()
    output.mkdir(parents=True)
    prepared = output / "prepared"
    prepared.mkdir()
    rows = []
    for index, row in enumerate(spec["training"], 1):
        case = resolve(base, row["case"])
        annotation = resolve(base, row["annotation"])
        folder = prepared / f"train-{index:03d}"
        command([args.bora, "prepare", str(case), "--output", str(folder)])
        item = dict(descriptors=str(folder / "descriptors.npz"),
                    groups=str(folder / "groups.ome.tif"),
                    annotation=str(annotation),
                    annotation_values=row["annotation_values"],
                    source_group=row["source_group"], split="train",
                    min_coverage=float(row.get("min_coverage", .5)),
                    min_purity=float(row.get("min_purity", .8)))
        if annotation.suffix.lower() == ".geojson":
            item["annotation_coordinate_space"] = "native_image_pixel_edges"
            item["annotation_field"] = row.get("annotation_field", "value")
        if "ignore_values" in row:
            item["ignore_values"] = row["ignore_values"]
        rows.append(item)
    training = output / "training.json"
    training.write_text(json.dumps(dict(format="bora-training-v1",
                                        classes=spec["classes"], cases=rows), indent=2) + "\n")
    model = output / "neighbor-context-model.npz"
    command([args.bora, "train", str(training), "--output", str(model)])
    targets = output / "targets"
    targets.mkdir()
    for index, row in enumerate(spec["targets"], 1):
        case = resolve(base, row["case"])
        command([args.bora, "refine", str(case), "--model", str(model),
                 "--output", str(targets / f"target-{index:03d}")])
    result = dict(format="bora-analysis-run-v1", workflow=str(spec_path),
                  workflow_sha256=sha256(spec_path), model=str(model),
                  training_cases=len(rows), target_cases=len(spec["targets"]),
                  classifier="Neighbor context", no_algorithm_selection=True)
    marker = output / "WORKFLOW_COMPLETE.json"
    temp = output / "WORKFLOW_COMPLETE.json.tmp"
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, marker)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
