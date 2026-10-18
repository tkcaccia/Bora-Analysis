"""List the exact annotated source fields used for the manuscript fastPLS model.

This is a read-only manifest browser. It does not download images or start an
encoder. Acquisition instructions are in docs/MANUSCRIPT_MODEL_DATA.md.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

MANIFEST = Path(__file__).resolve().parents[1] / "docs/manuscript_model_training_sources.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", help="show only one dataset")
    p.add_argument("--csv", type=Path, help="write a flat per-image CSV instead of printing")
    args = p.parse_args(argv)
    document = json.loads(MANIFEST.read_text())
    if document.get("format") != "bora-manuscript-training-sources-v1" or len(document["cases"]) != 349:
        raise ValueError("Unexpected manuscript training-source manifest")
    if args.dataset and args.dataset not in document["dataset_counts"]:
        p.error("unknown dataset: " + args.dataset)
    rows = [r for r in document["cases"] if not args.dataset or r["dataset"] == args.dataset]
    flat = [dict(dataset=r["dataset"], case=r["case"], source_id=r["source_id"],
                 source_group=r["source_group"], source_image=r["source_files"].get("image", ""),
                 source_mask=r["source_files"].get("mask", r["source_files"].get("annotation", "")),
                 image_zip_member=r["source_files"].get("image_zip_member", r["source_files"].get("source_zip_member", "")),
                 native_crop_xywh=json.dumps(r["native_crop_xywh"]),
                 native_mpp_xy=json.dumps(r["native_mpp_xy"]),
                 prepared_image_sha256=r["prepared_image_sha256"],
                 prepared_reference_sha256=r["prepared_reference_sha256"])
            for r in rows]
    fields = list(flat[0])
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
        print(f"wrote {len(flat)} rows to {args.csv}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    main()
