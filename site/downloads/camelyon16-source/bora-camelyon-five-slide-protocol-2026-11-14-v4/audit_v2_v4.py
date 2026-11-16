"""Source-free semantic regression of v4 maximum matching against v2."""

from pathlib import Path
import importlib.util
import json

import numpy as np


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


root = Path(__file__).parent
old = module(root.parent / "bora-camelyon-five-slide-protocol-2026-11-14-v2/binary_score.py",
             "cam_binary_old")
new = module(root / "binary_score.py", "cam_binary_new")
cases = 0
for seed in range(20):
    rng = np.random.default_rng(seed)
    shape = (48, 52)
    reference = (rng.random(shape) < .035).astype(np.uint8)
    native = (rng.random(shape) < .04).astype(np.uint8)
    support = np.ones(shape, np.uint8)
    annotated = np.ones(shape, np.uint8)
    excluded = np.zeros(shape, np.uint8)
    excluded[:2] = 1
    support[-2:] = 0
    reference[excluded != 0] = 0
    shared = native.copy()
    shared[support == 0] = 0
    previous = old.score_pair(reference, native, shared, support, annotated, excluded)
    current = new.score_pair(reference, native, shared, support, annotated, excluded)
    for mode in ("native", "shared_support"):
        a, b = previous[mode], current[mode]
        for key in ("tp", "fp", "fn", "tn", "precision", "recall", "dice", "iou",
                    "false_positive_tissue_area_mm2", "positive_outside_reviewed_support_pixels"):
            assert a[key] == b[key], (seed, mode, key)
        assert a["boundary"] == b["boundary"], (seed, mode, "boundary")
        assert b["lesions"]["matched_components"] >= a["lesions"]["matched_components"]
    cases += 1

result = {"status": "SYNTHETIC_REGRESSION_VERIFIED", "cases": cases,
          "semantic_and_boundary_metrics_identical": True,
          "new_lesion_match_count_never_below_greedy": True,
          "camelyon16_data_opened": False}
(root / "V2_V4_SYNTHETIC_REGRESSION_VERIFIED.json").write_text(
    json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
