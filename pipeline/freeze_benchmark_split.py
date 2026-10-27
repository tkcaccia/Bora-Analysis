#!/usr/bin/env python3
"""Freeze slide-disjoint Bora field splits before fitting or scoring.

The OCELOT release is kept as a fully held-out, tumor-only secondary endpoint:
its non-cancer code does not identify expert stroma. Previously inspected
fields cannot be represented as untouched confirmatory observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

SOURCES = ("bcss", "tiger", "ignite", "ocelot")
TCGA = re.compile(r"TCGA-[A-Z0-9]+-[A-Z0-9]+", re.I)
SALT = "bora-full-rerun-2026-10-27-fixed-split-v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_group(dataset: str, raw: str) -> str:
    match = TCGA.search(raw)
    if match:
        return match.group(0).upper()
    if dataset == "ignite":
        return "IGNITE:" + raw.casefold()
    return dataset.upper() + ":" + raw.casefold()


def ordered(groups: set[str], dataset: str) -> list[str]:
    return sorted(groups, key=lambda g: hashlib.sha256(
        f"{SALT}|{dataset}|{g}".encode()).hexdigest())


def freeze(root: Path, output: Path, n_test: int = 20) -> dict:
    if output.exists():
        raise FileExistsError("The frozen split must be written to a new path")
    if n_test != 20:
        raise ValueError("This prespecified field protocol uses 20 images per training dataset")
    root = root.resolve()
    catalog = root / "benchmark_catalog"
    overview = catalog / "overview.json"
    if not overview.is_file():
        raise FileNotFoundError(overview)
    rows: dict[str, list[dict]] = {}
    tsv_sha = {}
    for dataset in SOURCES:
        tsv = catalog / "datasets" / dataset / "cases.tsv"
        tsv_sha[dataset] = digest(tsv)
        with tsv.open(newline="") as f:
            items = list(csv.DictReader(f, delimiter="\t"))
        if not items or len({r["case_id"] for r in items}) != len(items):
            raise ValueError(f"Missing or duplicate case IDs in {dataset}")
        for row in items:
            row["global_source_group"] = source_group(dataset, row["patient_or_slide"])
            if not (root / row["source_image"]).is_file() or not (root / row["expert_tissue_mask"]).is_file():
                raise FileNotFoundError("Missing paired image or tissue mask: " + row["case_id"])
        rows[dataset] = items

    # BCSS/TIGER share TCGA patients. Their selected test groups are distinct,
    # and any cross-release copy of a selected test patient is quarantined.
    bcss_groups = {r["global_source_group"] for r in rows["bcss"]}
    tiger_groups = {r["global_source_group"] for r in rows["tiger"]}
    bcss_test = set(ordered(bcss_groups, "bcss")[:n_test])
    tiger_test = set(ordered(tiger_groups - bcss_test, "tiger")[:n_test])
    ignite_groups = {r["global_source_group"] for r in rows["ignite"]}
    ignite_test = set(ordered(ignite_groups, "ignite")[:n_test])
    if any(len(s) != n_test for s in (bcss_test, tiger_test, ignite_test)):
        raise ValueError("Insufficient independent source groups for a 20-image test")
    selected = {"bcss": bcss_test, "tiger": tiger_test, "ignite": ignite_test}
    ocelot_external = {r["global_source_group"] for r in rows["ocelot"]}
    reserved_groups = bcss_test | tiger_test | ignite_test | ocelot_external
    assignments = []
    counts: dict[str, dict[str, int]] = {}
    for dataset in SOURCES:
        own_test = selected.get(dataset, set())
        by_group: dict[str, list[dict]] = {}
        for row in rows[dataset]:
            by_group.setdefault(row["global_source_group"], []).append(row)
        chosen = {g: ordered({r["case_id"] for r in by_group[g]}, dataset)[0]
                  for g in own_test}
        for row in rows[dataset]:
            group = row["global_source_group"]
            if dataset == "ocelot":
                split = "external_tumor_only"
            elif group in own_test and chosen[group] == row["case_id"]:
                split = "test"
            elif group in reserved_groups:
                split = "reserved_related_source"
            else:
                split = "train"
            assignments.append(dict(dataset=dataset, case_id=row["case_id"],
                                    global_source_group=group, split=split,
                                    source_image=row["source_image"],
                                    expert_tissue_mask=row["expert_tissue_mask"],
                                    mpp_xy=[float(row["mpp_x"]), float(row["mpp_y"])],
                                    feature_dir=row.get("virchow2_features") or None))
        counts[dataset] = {name: sum(x["dataset"] == dataset and x["split"] == name
                                     for x in assignments)
                           for name in ("train", "test", "reserved_related_source",
                                        "external_tumor_only")}
    train_groups = {r["global_source_group"] for r in assignments if r["split"] == "train"}
    test_groups = {r["global_source_group"] for r in assignments if r["split"] in
                   ("test", "external_tumor_only")}
    if train_groups & test_groups:
        raise AssertionError("Training and test source groups overlap")
    for dataset in selected:
        if counts[dataset]["test"] != n_test:
            raise AssertionError("A training-source dataset lacks 20 independent test images")
    data = dict(format="bora-full-benchmark-split-v1", salt=SALT,
                catalog_sha256=digest(overview), case_table_sha256=tsv_sha,
                source_overlap=dict(bcss_tiger_tcga_groups=len(bcss_groups & tiger_groups)),
                counts=counts, source_grouping="TCGA patient where present; otherwise provider patient/slide ID",
                primary_datasets=["bcss", "tiger", "ignite"],
                external_secondary_dataset="ocelot",
                external_secondary_endpoint="expert cancer-area pixels only; non-cancer is not expert stroma",
                analysis_status="development rerun; prior image inspection prevents untouched confirmation",
                assignments=assignments)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temp, output)
    return dict(output=str(output), sha256=digest(output), counts=counts,
                bcss_tiger_overlap=data["source_overlap"]["bcss_tiger_tcga_groups"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(freeze(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
