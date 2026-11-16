# Publish each qualified slide result

Each newly completed slide is released as its own small site change. A batch
need not finish before its first qualified slide appears. The local site and
GitHub Pages must show the same result.

For each slide:

1. Wait for the method controller to finish and verify owned-process and GPU
   cleanup. Keep a failed or incomplete bundle out of the site.
2. Verify the source, model, calibration, tissue support, mask, GeoJSON,
   figure, scorer and reference identities. Record the native and shared-
   support score domains separately. Inspect the figure and tissue boundary.
3. Add that slide's page, scores, figure and downloads to the local site. Link
   it from the correct benchmark index with its training and reference scope.
   Label historical and new model versions separately; never replace old
   numbers without a new versioned receipt.
4. Run site link and asset checks, then publish the same files from `main`.
   Wait for successful Pages deployment and stream-hash the live page and
   assets against the local release. Save a per-slide publication receipt.

Benchmark 3 uses official known pixels for binary metastasis scoring. Benign
IoU is undefined; show false-positive tissue area instead. Do not publish a
population or independent-test claim from selected or previously inspected
slides. No full CAMELYON16 source slide is redistributed.

The one-slide Bora reanalysis added on 16 November 2026 has one frozen
source-only fit and six already inspected score slides. Its per-slide score
blocks and JSON are separate from the historical two-slide five-method gallery.
Future newly verified CAMELYON16 slides should be added individually by this
same process, using a new versioned result and receipt.
