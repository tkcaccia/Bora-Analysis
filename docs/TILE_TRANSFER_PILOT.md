# Source-to-target tile transfer pilot

This is a **partial, exploratory ROI experiment**, separate from the article's
reported Bora benchmark. One annotated TIGER training ROI (`tiger_42`) supplied
69 high-purity, physically calibrated tile examples. Native fastPLS 0.3.0
predicted anonymous 45-µm tile maps on five distinct TIGER evaluation source
groups without reading their annotations. The frozen maps were then scored:
two-role coarse macro IoU was 0.603, 0.685, 0.734, 0.680 and 0.603 (mean
0.661). The two-role subset was selected using registry knowledge, and these
development images are not an independent patient test.

The CPU-only preparation and scoring scripts are:

- [`prepare_cmpb_transfer_tiles.py`](../benchmark/prepare_cmpb_transfer_tiles.py)
- [`prepare_cmpb_transfer_feature_plan.py`](../benchmark/prepare_cmpb_transfer_feature_plan.py)
- [`score_cmpb_transfer_tiles.py`](../benchmark/score_cmpb_transfer_tiles.py)

The full Bora path is **not complete**. The isolated, single-case multiscale
feature extraction stopped before model loading because the selected Python
environment lacked `timm`. No feature checkpoint was written. The owned
process and compute context were absent on bounded post-exit checks. The
project's `hpc-gpu-policy.md`
requires failing the whole bundle on an abnormal exit; no GPU retry or next
case was launched. A separately approved Chiamaka scope with a validated
environment is required before generating RGB evidence from predicted groups
and running Bora. UCT HPC GPU work remains paused.

The pilot must not be cited as a WSI experiment, a final segmentation score or
proof of independent generalization.
