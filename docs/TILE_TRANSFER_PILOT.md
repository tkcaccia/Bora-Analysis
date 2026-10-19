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

After separate user approval, **one** corrected, bounded Chiamaka encoder case
completed for `tiger_15` in an isolated Python environment. The input tile
SHA256 was `7471377ae10a33f745afda75bd0ed2a88ac590191994819943b62ac400873e9a`;
the Virchow2 checkpoint SHA256 was
`8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b`.
The 0.5-µm/pixel RGB ROI measured 1135 × 1236 pixels. Three feature scales
were checkpointed, and their complete marker recorded 2.33 seconds of encoding
after model loading. The process group exited cleanly; a bounded post-exit
check found neither an owned descendant nor an active GPU compute context.
No second target was encoded.

CPU-only Bora refinement then used the fixed RGB conversion and appearance
field on this **predicted** tile map. Each resulting whole tissue group received
the area-majority source-trained fastPLS tile class within its refined pixels.
This is an exploratory tile-to-segment readout, **not** the main Neighbor
context group classifier. Bora inference did not read the target annotation,
but frozen tile scores on this target had already been inspected before the
Bora run. On 1,402,860 annotated target pixels,
the frozen coarse map had macro IoU **0.602691**, while the final vector-derived
mask had macro IoU **0.770969** (difference **0.168277**). Tumor and stroma
final IoUs were 0.791065 and 0.750872. Independent rasterization of both
GeoJSON features exactly matched the final native mask. Its 26 polygon parts
and 28 enclosed holes had no continuous area below 20 native pixels. The
six-panel figure and machine-readable scores are linked below.
The shared exporter initially wrote a Neighbor context identity-source string;
the pilot GeoJSON now correctly records its area-majority tile readout.
The two geometries and scored raster were unchanged, and the original vector
file is preserved in the local study record.

- [Frozen one-case score](../benchmark/cmpb_evidence/transfer_roi_tiger_15.json)
- [Vector/raster visual audit](../benchmark/cmpb_evidence/transfer_roi_tiger_15_visual_audit.json)
- [Bounded encoder script](../benchmark/pilots/one_case_transfer_encoder.py)
- [CPU Bora/refinement script](../benchmark/pilots/transfer_refine_cpu.py)
- [Post-prediction scoring script](../benchmark/pilots/score_transfer_refinement.py)

This selected, previously inspected, two-role **ROI** is not a source-WSI to
target-WSI experiment or a patient-independent performance estimate. Source
identifiers differ, but patient linkage is not verified. The prior failed
environment attempt and its logs remain preserved; this approval did not
authorize any further GPU target. UCT HPC GPU work remains paused.
