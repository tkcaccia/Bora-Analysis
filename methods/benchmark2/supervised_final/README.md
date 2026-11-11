# Final supervised Bora whole-slide mask

These are byte-identical copies of the executed Chiamaka research sources for
the final public Benchmark 2 supervised Bora masks. They are organized as
functions for input verification, class selection, score projection, physical
area repair, whole-slide contour smoothing, vector conversion, coordinate
mapping, and independent output audit. The local source paths and frozen input
receipts are project-specific; a fresh checkout needs authorized images,
features, model files, support geometry, and those receipts before it can run.
No provider images or model weights are redistributed here.

| Source | Role |
| --- | --- |
| `select_class_presence.py` | Search a global class-presence threshold using two source-group-disjoint splits of the 397 training images only. |
| `supervised_wsi_pipeline.py` | Define the reusable CPU functions for verified inputs, full-slide score projection, class selection, support, physical cleanup, 6 µm contour smoothing, global vectorization and audit. |
| `repair_unoccupied_gaps.py` | Repair tiny uncovered vector gaps without overwriting an existing positive polygon. |
| `finalize_public_wsi.py` | Compose those functions for the 12 remaining public slides and produce one calibrated TIFF and native-coordinate GeoJSON per slide. |
| `render_public_views.py` | Render an overview and nine aligned zooms from that single audited full-slide mask. |
| `audit_code.py` | Check training-only model selection, exact 400 µm² boundary, 6 µm seams, support preservation, and production output identities. |
| `audit_campaign.py` | Recheck the final 13-slide receipts, file hashes, 130 rendered views, geometry minima and public publication receipts. |

The class rule retains a class when its winner share on the complete tissue
score grid is at least 2%; omitted-class pixels take their next-highest
supervised score. The threshold was chosen with held-out training cases, not
whole-slide or Benchmark 1 validation expert masks. The 6 µm contour rule is
applied globally at calibrated physical spacing. Positive components and
enclosed holes strictly below 2,000 µm² are repaired after smoothing and
vectorization. The native GeoJSON must rasterize exactly back to the final
mask. Bora's final reviewed tissue support remains the hard upper bound.

The core smoothing and global vector functions are in the
[Bora repository](https://github.com/tkcaccia/Bora) as
`src/bora/contour_polish.py` and `src/bora/wsi_global_vector.py`. The public
pages are exploratory: they have no independent whole-slide expert masks, so
these files do not establish class accuracy or a method ranking. PM016 and
the withdrawn BEETLE data are outside this campaign.
