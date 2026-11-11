#!/usr/bin/env python3
"""Choose a per-image class-presence floor using training images only.

The frozen eight-class head is refit on each original grouped training fold.
No development validation image or whole-slide expert target is consulted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

from importlib.util import module_from_spec, spec_from_file_location

ROOT = Path("/mnt/sata_ssd/ChatGPT/Bora")
HERE = Path(__file__).resolve().parent
TRAIN = ROOT / "migration/benchmark-label-reduction-2026-11-11-v1/train_supervised.py"
SAMPLE = ROOT / "migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SAMPLE_COMPLETE.json"
SPLITS = ROOT / "migration/benchmark1-supervised-strategy-search-2026-11-10-v1/SEARCH_COMPLETE.json"
ONTOLOGY = ROOT / "migration/benchmark-label-reduction-2026-11-11-v1/ONTOLOGY_FROZEN.json"
MODEL = ROOT / "migration/benchmark-label-reduction-2026-11-11-v1/SUPERVISED_MODEL_FROZEN.json"
THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gated(scores: np.ndarray, minimum_fraction: float) -> np.ndarray:
    """Remove unsupported classes, then reassign each token by its next score."""
    if not 0 <= minimum_fraction < 1 or scores.ndim != 2:
        raise ValueError("Invalid class-presence threshold or scores")
    winning = scores.argmax(axis=1)
    fractions = np.bincount(winning, minlength=scores.shape[1]) / len(winning)
    keep = fractions >= minimum_fraction
    if not keep.any():
        keep[np.argmax(fractions)] = True
    return np.where(keep[None, :], scores, -np.inf).argmax(axis=1) + 1


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([
        np.count_nonzero((pred == cls) & (truth == cls)) /
        np.count_nonzero((pred == cls) | (truth == cls))
        for cls in np.unique(truth)
    ]))


def main() -> None:
    out = HERE / "TRAINING_PRESENCE_SEARCH.json"
    if out.exists():
        raise FileExistsError(out)
    spec = spec_from_file_location("frozen_train", TRAIN)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sample, splits = json.loads(SAMPLE.read_text()), json.loads(SPLITS.read_text())
    ontology, model_info = json.loads(ONTOLOGY.read_text()), json.loads(MODEL.read_text())
    if sample["training_cases"] != 397 or sample["validation_cases_opened"] or \
       model_info["validation_masks_opened"] or splits["validation_cases_opened"]:
        raise ValueError("The selection input is not training-only")
    if sha(SAMPLE) != model_info["sample_sha256"] or sha(SPLITS) != model_info["splits_sha256"]:
        raise ValueError("Frozen training input hash differs")
    x = np.load(sample["feature_path"], mmap_mode="r")
    idx = np.load(sample["index_path"])
    old_y, cases, datasets = (idx[k] for k in ("label", "case_index", "dataset_id"))
    lut = np.zeros(256, np.uint8)
    for old, new in ontology["old_to_new"].items():
        if new is not None:
            lut[int(old)] = int(new)
    truth = lut[old_y]
    valid = truth > 0
    reports = []
    for fold_no, fold in enumerate(splits["folds"], 1):
        train = valid & np.isin(cases, fold["training_case_indices"])
        held = valid & np.isin(cases, fold["heldout_case_indices"])
        model = module.fit(np.asarray(x[train], np.float32), truth[train],
                           model_info["selected"]["channels"], model_info["selected"]["C"])
        scores = module.predict_scores(model, np.asarray(x[held], np.float32))
        held_cases, held_truth, held_datasets = cases[held], truth[held], datasets[held]
        for did, dataset in enumerate(sample["dataset_names"]):
            rows = held_datasets == did
            if rows.any():
                allowed = np.isin(np.arange(1, 9), model_info["eligible_classes_by_dataset"][dataset])
                scores[np.ix_(rows, ~allowed)] = -np.inf
        for case in np.unique(held_cases):
            rows = held_cases == case
            if not rows.any():
                continue
            dataset = sample["dataset_names"][int(held_datasets[rows][0])]
            s, t = scores[rows], held_truth[rows]
            variants = {}
            for threshold in THRESHOLDS:
                p = gated(s, threshold)
                variants[str(threshold)] = {
                    "macro_iou": iou(p, t),
                    "pixel_accuracy": float(np.mean(p == t)),
                    "predicted_classes": int(len(np.unique(p))),
                    "false_positive_classes": int(len(set(np.unique(p)) - set(np.unique(t)))),
                }
            reports.append({"fold": fold_no, "case_index": int(case), "dataset": dataset,
                            "sampled_tokens": int(rows.sum()), "expert_classes": int(len(np.unique(t))),
                            "variants": variants})
        print("FOLD", fold_no, "CASES", len(reports), flush=True)
    summary = []
    for threshold in THRESHOLDS:
        key = str(threshold)
        summary.append({"minimum_fraction": threshold,
                        "mean_case_macro_iou": float(np.mean([r["variants"][key]["macro_iou"] for r in reports])),
                        "mean_case_accuracy": float(np.mean([r["variants"][key]["pixel_accuracy"] for r in reports])),
                        "mean_predicted_classes": float(np.mean([r["variants"][key]["predicted_classes"] for r in reports])),
                        "mean_false_positive_classes": float(np.mean([r["variants"][key]["false_positive_classes"] for r in reports]))})
    result = {"status": "TRAINING_ONLY_CLASS_PRESENCE_SEARCH", "training_cases": 397,
              "validation_cases_opened": 0, "whole_slide_expert_masks_opened": 0,
              "selection": "pending review", "summary": summary, "cases": reports,
              "sample_sha256": sha(SAMPLE), "splits_sha256": sha(SPLITS),
              "ontology_sha256": sha(ONTOLOGY), "model_receipt_sha256": sha(MODEL),
              "source_sha256": sha(Path(__file__))}
    temp = out.with_suffix(".partial.json")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, out)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
