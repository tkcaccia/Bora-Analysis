#!/usr/bin/env python3
"""Aggregate the complete frozen 60-field development evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from build_bora_case import sha256


def summarize(scores: list[dict]) -> dict:
    metrics = ("macro_iou", "macro_dice", "pixel_accuracy")
    result = {}
    for name in metrics:
        current = np.array([r["final"][name] for r in scores], float)
        coarse = np.array([r["oracle_expert_decoded_coarse"][name] for r in scores], float)
        result[name] = dict(mean=float(current.mean()), median=float(np.median(current)),
                            oracle_expert_decoded_coarse_mean=float(coarse.mean()),
                            paired_mean_difference=float(np.mean(current - coarse)))
    interfaces = [r["tumor_stroma_interface"]["f1"] for r in scores
                  if r["tumor_stroma_interface"]["f1"] is not None]
    groups = [g for r in scores for g in r["groups"] if g["eligible_for_accuracy"]]
    result["whole_group_accuracy"] = float(np.mean([g["correct"] for g in groups])) if groups else None
    result["whole_group_count"] = len(groups)
    recalls = []
    for code in sorted({g["expert_class"] for g in groups}):
        selected = [g for g in groups if g["expert_class"] == code]
        recalls.append(float(np.mean([g["correct"] for g in selected])))
    result["whole_group_balanced_accuracy"] = float(np.mean(recalls)) if recalls else None
    result["tumor_stroma_boundary_f1_at_8um"] = dict(images=len(interfaces),
        mean=float(np.mean(interfaces)) if interfaces else None)
    result["tile_count_bins"] = {
        name: dict(groups=len(selected), accuracy=float(np.mean([g["correct"] for g in selected]))
                   if selected else None)
        for name, selected in (("1-4", [g for g in groups if g["input_tiles"] < 5]),
                               ("5-9", [g for g in groups if 5 <= g["input_tiles"] < 10]),
                               ("10-19", [g for g in groups if 10 <= g["input_tiles"] < 20]),
                               ("20+", [g for g in groups if g["input_tiles"] >= 20]))}
    return result


def bootstrap(scores: list[dict], repeats=2000) -> dict:
    # Resample globally linked patients/slides, retaining all their images.
    by_source = defaultdict(list)
    for row in scores:
        by_source[row["source_group"]].append(row)
    groups = list(sorted(by_source))
    rng = np.random.default_rng(20261027)
    values = []
    for _ in range(repeats):
        sampled = [r for g in rng.choice(groups, size=len(groups), replace=True)
                   for r in by_source[g]]
        values.append(float(np.mean([r["final"]["macro_iou"] for r in sampled])))
    return dict(repeats=repeats, source_groups=len(groups),
                mean_macro_iou_95pct=[float(x) for x in np.quantile(values, [.025, .975])],
                describes_reused_development_cohort=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if not (a.run / "TEST_COMPLETE.json").is_file():
        raise ValueError("Frozen primary evaluation has not completed")
    split_path = a.root / "preprocessing/full-rerun-2026-10-27/split-v1.json"
    split = json.loads(split_path.read_text())
    expected = {r["case_id"]: r for r in split["assignments"] if r["split"] == "test"}
    if len(expected) != 60:
        raise ValueError("Frozen test count changed")
    scores = []
    score_hashes = {}
    for case_id, row in sorted(expected.items()):
        file = a.run / "test-results" / case_id / "score.json"
        item = json.loads(file.read_text())
        if (item["case_id"] != case_id or item["dataset"] != row["dataset"]
                or item["source_group"] != row["global_source_group"]
                or item["split"] != "test"
                or item["endpoint"] != "expert_multiclass_annotated_pixels"):
            raise ValueError("Score case, source or endpoint changed")
        scores.append(item)
        score_hashes[case_id] = sha256(file)
    dataset = {name: summarize([r for r in scores if r["dataset"] == name])
               for name in ("bcss", "tiger", "ignite")}
    if any(sum(r["dataset"] == name for r in scores) != 20
           for name in dataset):
        raise ValueError("Expected twenty scored fields per dataset")
    result = dict(format="bora-full-rerun-primary-summary-v1", fields=60,
                  datasets={name: 20 for name in dataset},
                  split_sha256=sha256(split_path),
                  model_sha256=sha256(a.run / "model.npz"),
                  score_sha256=score_hashes,
                  overall=summarize(scores), per_dataset=dataset,
                  source_group_bootstrap=bootstrap(scores),
                  expert_derived_anonymous_coarse_input=True,
                  independent_final_test_claim=False,
                  no_algorithm_selection_on_this_cohort=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(fields=result["fields"],
                          mean_macro_iou=result["overall"]["macro_iou"]["mean"],
                          group_accuracy=result["overall"]["whole_group_accuracy"]),indent=2))


if __name__ == "__main__":
    main()
