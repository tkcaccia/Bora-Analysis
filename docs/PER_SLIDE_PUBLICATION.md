# Publish each qualified slide result

Each newly completed slide is released as its own small site change. A batch
need not finish before its first qualified slide appears. The local site and
GitHub Pages must show the same result.

The 17 November one-slide Benchmark 3 index also lists the 20 source-selected
test slide IDs before analysis is complete. A status-only row may be published
after source identity audit, and an independently audited source-only preview
may be shown with its score explicitly pending. This is a progress record, not
a completed method result. Never display a pending method's score, infer one
from an old model version, or count the status row in a completed denominator.
The primary Benchmark 3 fit uses tumor_093 only. Older pair-trained CAM
comparators remain historical and are excluded from one-slide tables.

For each slide:

1. Wait for the method controller to finish and verify owned-process and GPU
   cleanup. Keep a failed or incomplete bundle out of the site.
2. Verify the source, model, calibration, tissue support, mask, GeoJSON,
   figure, scorer and reference identities. Record the native and shared-
   support score domains separately. Inspect the figure and tissue boundary.
3. Add that slide's page, scores, figure and downloads to the local site. Link
   it from the correct benchmark index with its training and reference scope.
   Label historical and new model versions separately; never replace old
   numbers without a new versioned receipt. For Benchmark 3, run
   `python3 tools/refresh_cam_method_tables.py` after staging the independently
   audited `<slide>-scores.json`. This updates the positive and benign method
   tables from current tumor_093-only scores. Run
   `python3 tools/refresh_cam_method_tables.py --check` as a release gate. A
   pending method remains pending; do not fill it from an older comparator fit.
   For every completed Benchmark 3 Bora result, render both black-and-white
   class-response maps from that slide's audited, smoothed tumor_093-only score
   grid and reviewed H&E tissue support using
   `python tools/render_cam_bw_projections.py --manifest <frozen-local-manifest.json> --site site --receipt <versioned-receipt.json>`.
   Add the two panels to the slide page and a gallery link. These are relative
   softmax displays, not calibrated probabilities or the final thresholded
   mask. Confirm the same tissue crop as the H&E overview, exact source/model/
   support hashes, and no visible exterior glass. Run
   `python tools/verify_cam_bw_projections.py site`; the Pages workflow enforces
   this for every currently scored one-slide test page, including future pages.
4. Run site link and asset checks, then publish the same files from `main`.
   Wait for successful Pages deployment and stream-hash the live page and
   assets against the local release. Save a per-slide publication receipt.

Benchmark 3 uses official known pixels for binary metastasis scoring. Benign
IoU is undefined; show false-positive tissue area instead. Do not publish a
population or independent-test claim from selected or previously inspected
slides. No full CAMELYON16 source slide is redistributed.

The one-slide CAM study uses tumor_093 as its sole CAM fitting slide. The
current index tables include every independently audited method score and
distinguish source-selected slides from previously inspected ones. The six
earlier inspected score slides remain descriptive. Future newly verified
CAMELYON16 method results must update the table in the same per-slide release.
