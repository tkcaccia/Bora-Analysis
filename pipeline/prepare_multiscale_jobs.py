#!/usr/bin/env python3
"""Freeze the two extra Virchow2 context scales required by fixed RGB9+19."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
WEIGHTS_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root: Path, split_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    root = root.resolve()
    split = json.loads(split_path.read_text())
    if split.get("format") != "bora-full-benchmark-split-v1":
        raise ValueError("Expected frozen full benchmark split")
    entries = []
    seen = set()
    for row in split["assignments"]:
        if row["split"] not in ("train", "test", "external_tumor_only"):
            continue
        if row["case_id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(row["case_id"])
        image = root / row["source_image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        mpp = row["mpp_xy"]
        if len(mpp) != 2 or not all(0.1 <= x <= 1.0 for x in mpp):
            raise ValueError("Unverified source spacing")
        entries.append(dict(case_id=row["case_id"], dataset=row["dataset"],
                            split=row["split"], source_image=str(image),
                            source_mpp_xy=mpp, target_scales_mpp=[1.0, 2.0],
                            base_feature_case=row["case_id"], tile_images_saved=False))
    entries.sort(key=lambda x: (x["split"], x["dataset"], x["case_id"]))
    if not entries or len(entries) != 1060:
        raise ValueError("Frozen candidate count changed; inspect split before launch")
    data = dict(format="bora-virchow2-multiscale-queue-v1",
                split_sha256=sha256(split_path), catalog_sha256=split["catalog_sha256"],
                encoder_revision=REVISION, encoder_sha256=WEIGHTS_SHA256,
                base_target_mpp=0.5, context_scales_mpp=[1.0, 2.0],
                tile_pixels=224, cases=entries, total_cases=len(entries),
                downstream="Bora 0.1 fixed RGB9+19; no conversion changed",
                full_image_tile_files_saved=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    with temp.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, output)
    return dict(output=str(output), sha256=sha256(output), total_cases=len(entries))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.root, a.split, a.output), indent=2))


if __name__ == "__main__":
    main()
