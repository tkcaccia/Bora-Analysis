"""Source-free v4 scorer profile on the same 9.4M-pixel v2 synthetic grid."""

from pathlib import Path
import json
import resource
from time import perf_counter

import numpy as np

from binary_score import score_pair


shape = (3072, 3072)
reference = np.zeros(shape, np.uint8)
reference[600:1400, 700:1800] = 1
native = reference.copy()
native[1200:1440, 1600:1940] = 1
support = np.ones(shape, np.uint8)
annotated = np.ones(shape, np.uint8)
excluded = np.zeros(shape, np.uint8)
started = perf_counter()
result = score_pair(reference, native, native.copy(), support, annotated, excluded)
record = {"status": "SYNTHETIC_ONLY", "shape_yx": shape,
          "pixels": shape[0] * shape[1],
          "wall_seconds": perf_counter() - started,
          "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "native_iou": result["native"]["iou"],
          "native_boundary_f1": result["native"]["boundary"]["f1"],
          "native_matched_lesions": result["native"]["lesions"]["matched_components"]}
Path(__file__).with_name("SYNTHETIC_RESOURCE_PROFILE_9M.json").write_text(
    json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
