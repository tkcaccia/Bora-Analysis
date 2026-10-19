"""CPU-only WSI tissue readout of bounded anonymous group segments.

Virchow2 and Bora geometry are frozen before this step. A global KMeans ID is
only an anonymous appearance ID; disconnected WSI-wide pieces must not be
forced to share one tissue identity. Each non-overlapping native core therefore
gets its own group means and physical shared-boundary context. The fixed
Neighbor context fastPLS model is loaded once. No expert reference is read.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import os

import numpy as np
from bora.classifier import NeighborContextClassifier
from finalize_wsi import geometry, group_adjacency, atomic_json


def classify(source, output, model_path):
    source, output = Path(source), Path(output)
    info = geometry(source)
    h, w = info["native_shape_xy"][1], info["native_shape_xy"][0]
    shape = (h, w)
    groups = np.memmap(source / "groups.native.dat", mode="r", dtype=np.uint8,
                       shape=shape)
    model = NeighborContextClassifier.load(model_path)
    output.mkdir(parents=True, exist_ok=False)
    raw = np.memmap(output / "classified.raw.dat", mode="w+", dtype=np.uint8,
                    shape=shape)
    raw[:] = 0
    raw.flush()
    counts = Counter()
    records = []
    core = int(info["core_pixels"])
    for index in range(int(info["total_windows"])):
        row = json.loads((source / f"window-{index:03d}.json").read_text())
        if row["status"] == "background":
            continue
        x, y, cw, ch = row["native_core_xywh"]
        native = np.asarray(groups[y:y+ch, x:x+cw])
        graph, pixels = group_adjacency(native, info["native_mpp_xy"])
        mass = np.asarray(row["mass"], np.float64)
        if not np.array_equal(pixels[1:], mass.astype(np.int64)):
            raise ValueError(f"Window {index} checkpoint mass disagrees with native raster")
        present = mass > 0
        sums = np.asarray(row["sums"], np.float64)
        means = np.zeros((6, 1280), np.float64)
        means[present] = sums[present] / mass[present, None]
        neighbours = means.copy()
        total = graph.sum(1)
        has = total > 0
        neighbours[has] = (graph @ means)[has] / total[has, None]
        descriptors = np.concatenate([means, means-neighbours], axis=1)
        prediction = np.zeros(6, np.uint8)
        prediction[present] = model.predict(descriptors[present])
        lookup = np.r_[np.uint8(0), prediction]
        rendered = lookup[native]
        raw[y:y+ch, x:x+cw] = rendered
        raw.flush()
        for cls, n in zip(*np.unique(rendered, return_counts=True)):
            if cls:
                counts[int(cls)] += int(n)
        result = {"index": index, "core_xywh": [x, y, cw, ch],
                  "anonymous_group_class_ids": prediction.tolist(),
                  "present_groups": np.flatnonzero(present).astype(int).tolist(),
                  "predicted_names": {str(i+1): model.names_.get(int(prediction[i]))
                                      for i in range(6) if present[i]},
                  "expert_annotation_read": False}
        atomic_json(output / f"window-{index:03d}.json", result)
        records.append(result)
        print(json.dumps({"classified_window": index,
                          "total_windows": info["total_windows"],
                          "classes": result["predicted_names"]}), flush=True)
    report = {"native_shape_yx": [h, w],
              "native_mpp_xy": info["native_mpp_xy"],
              "classification_unit": "anonymous group within one native WSI core",
              "window_core_pixels": core,
              "class_names": {str(k): v for k, v in model.names_.items()},
              "predicted_class_pixels": dict(counts),
              "classified_windows": len(records),
              "classifier": "Neighbor context, native fastPLS",
              "classifier_fit_seconds": model.fit_seconds_,
              "expert_annotation_read": False,
              "source_geometry": str(source)}
    atomic_json(output / "CLASSIFIED_COMPLETE.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    report = classify(args.source, args.output, args.model)
    print(json.dumps({"stage": "classify", "class_pixels": report["predicted_class_pixels"]}))


if __name__ == "__main__":
    main()
