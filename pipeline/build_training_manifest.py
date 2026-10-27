#!/usr/bin/env python3
"""Freeze whole-group expert targets for the released Bora train command."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import tifffile

from build_bora_case import sha256


def build(root: Path, cases_root: Path, prepared_root: Path) -> dict:
    from bora.annotations import group_targets, read_annotation

    split_path = root / "preprocessing/full-rerun-2026-10-27/split-v1.json"
    split = json.loads(split_path.read_text())
    exclusion_path = root / "preprocessing/full-rerun-2026-10-27/preflight-exclusions-v1.json"
    exclusions = json.loads(exclusion_path.read_text())
    if exclusions["split_sha256"] != sha256(split_path):
        raise ValueError("Frozen training exclusions belong to a different split")
    excluded_ids = {row["case_id"] for row in exclusions["excluded"]}
    if len(excluded_ids) != 12:
        raise ValueError("Expected the 12 frozen single-group training exclusions")
    train = [r for r in split["assignments"] if r["split"] == "train"
             and r["case_id"] not in excluded_ids]
    if len(train) != 476:
        raise ValueError("Eligible training cohort changed")
    held = {r["global_source_group"] for r in split["assignments"] if r["split"] == "test"}
    if any(r["global_source_group"] in held for r in train):
        raise ValueError("Training shares a patient/slide with frozen test set")
    prepared = []
    all_roles = None
    represented = Counter()
    for row in train:
        case = cases_root / row["case_id"]
        output = prepared_root / row["case_id"]
        audit = json.loads((case / "CASE_COMPLETE.json").read_text())
        if (audit["split"] != "train" or audit["source_group"] != row["global_source_group"]
                or sha256(case / "case.json") != audit["case_sha256"]):
            raise ValueError("Training case split, source or checksum differs")
        if all_roles is None:
            all_roles = audit["canonical_role_ids"]
        elif all_roles != audit["canonical_role_ids"]:
            raise ValueError("Canonical tissue role IDs differ between cases")
        groups = tifffile.imread(output / "groups.ome.tif")
        target = tifffile.imread(case / "expert-target-private.tif")
        with np.load(output / "descriptors.npz", allow_pickle=False) as descriptors:
            ids = descriptors["group_ids"].copy()
            if descriptors["x"].shape != (len(ids), 2560):
                raise ValueError("Prepared descriptor shape differs")
        valid, labels, excluded = group_targets(groups, ids, target,
                                                 min_coverage=.5, min_purity=.8)
        represented.update(map(int, labels))
        prepared.append(dict(case=case, output=output, audit=audit, groups=groups,
                             target=target, ids=ids, valid=valid, labels=labels,
                             excluded=excluded))
    if len(represented) < 2:
        raise ValueError("At least two represented tissue classes required")
    names = {str(i): name for name, i in all_roles.items() if i in represented}
    rows = []
    for item in prepared:
        unique = sorted(int(v) for v in np.unique(item["target"]) if v)
        mapping = {str(i): i for i in unique if i in represented}
        ignored = ["0"] + [str(i) for i in unique if i not in represented]
        annotation = read_annotation(item["case"] / "expert-target-private.tif",
                                     item["groups"].shape, mapping,
                                     ignore_values=ignored)
        again, again_labels, again_excluded = group_targets(item["groups"], item["ids"],
                                                            annotation,
                                                            min_coverage=.5, min_purity=.8)
        if not np.array_equal(again, item["valid"]) or not np.array_equal(again_labels, item["labels"]):
            raise ValueError("Omitting unrepresented roles changed training targets")
        rows.append(dict(descriptors=str(item["output"] / "descriptors.npz"),
                         groups=str(item["output"] / "groups.ome.tif"),
                         annotation=str(item["case"] / "expert-target-private.tif"),
                         annotation_values=mapping, ignore_values=ignored,
                         source_group=item["audit"]["source_group"], split="train",
                         min_coverage=.5, min_purity=.8,
                         sha256=sha256(item["output"] / "descriptors.npz"),
                         groups_sha256=sha256(item["output"] / "groups.ome.tif"),
                         annotation_sha256=sha256(item["case"] / "expert-target-private.tif")))
    return dict(format="bora-training-v1", classes=names, cases=rows,
                benchmark_audit=dict(split_sha256=sha256(split_path),
                                     exclusion_sha256=sha256(exclusion_path),
                                     training_images=len(rows),
                                     training_source_groups=len({r["source_group"] for r in rows}),
                                     represented_training_groups=sum(represented.values()),
                                     class_support=dict(represented),
                                     evaluated_test_images=sum(r["split"] == "test"
                                                               for r in split["assignments"]),
                                     frozen_before_final_scoring=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = build(a.root.resolve(), a.cases.resolve(), a.prepared.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["benchmark_audit"], indent=2))


if __name__ == "__main__":
    main()
