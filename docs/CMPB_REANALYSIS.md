# Reproducing the manuscript's descriptive reanalysis

The article reports a **conditional, exploratory** evaluation of Bora on 116
previously inspected, calibrated multiclass fields. Its coarse tile assignments
come from expert annotations. The original `test` and `validation` names are
retained to describe the source registry; neither split is an untouched test.

The repository records sanitized numerical inputs and outputs in
[`benchmark/cmpb_evidence`](../benchmark/cmpb_evidence). `registry_groups.json`
contains case IDs, dataset/split and known source-group identifiers.
`neighbor_summary.json` contains per-case IoU, whole-group correctness and cached timings
for Bora's Neighbor context classifier. `boundaries.json` gives native-resolution
tumor–stroma interface scores for each image. `vector_fidelity.json` gives
polygon-fitting diagnostics relative to the **preceding predicted geometry**.
`duplicate_screen.json` records a cross-split exact-image and global
perceptual-hash screen of 40,484 image pairs.
None of these JSON files contains raw WSI pixels, spatial-token tensors or
gated model weights. Obtain original archives under each provider's terms as
described in [the acquisition guide](MANUSCRIPT_MODEL_DATA.md).

The aggregate can be recomputed without model inference:

```bash
python benchmark/analyze_cmpb_cohort.py \
  --viewer site/data.json \
  --registry benchmark/cmpb_evidence/registry_groups.json \
  --neighbor benchmark/cmpb_evidence/neighbor_summary.json \
  --output /tmp/bora-reanalysis.json
```

The scripts `benchmark/analyze_cmpb_boundaries.py` and
`benchmark/analyze_cmpb_vector_fidelity.py` reproduce their respective JSONs
from the frozen group descriptors, provider masks and saved output geometry.
Those larger inputs are retained outside Git because of provider and checkpoint
terms. These scripts read saved outputs and
perform no encoder or segmentation inference. The public summaries make the
reported arithmetic independently inspectable even without those inputs.
`benchmark/analyze_cmpb_duplicates.py` verifies original-image SHA256s against
the registry and screens whole-image perceptual hashes at Hamming distance ≤4.
This global screen cannot find every overlapping crop or shared slide.

Tumor–stroma F1 uses a direct four-neighbor interface, physical 8 µm matching
radius and annotated pixels only. A missing predicted interface receives F1
zero. The expert-decoded coarse comparison uses the identity key of the
expert-derived tile mask and is a privileged geometry diagnostic, not a
deployable upstream predictor. The 95% paired interval resamples known
source-group identifiers 10,000 times with seed 19017. These identifiers are
not verified patients in every provider cohort.

The article and supplement are built from these summaries with
[`manuscript/build_cmpb_manuscript.py`](../manuscript/build_cmpb_manuscript.py)
and [`manuscript/build_cmpb_supplement.py`](../manuscript/build_cmpb_supplement.py).
See [`manuscript/CURRENT_DRAFT.md`](../manuscript/CURRENT_DRAFT.md) for exact
arguments. Draft author names and affiliations are saved in
[`manuscript/author_block.json`](../manuscript/author_block.json), pending
confirmation by the named people.
The generated DOCX and private figure source masks are kept outside Git.

A separate, single-slide scale analysis is documented in the
[Visium HD colon study](../studies/visium-hd-colon-wsi-2026-10-19/README.md).
`benchmark/analyze_cmpb_wsi_scale.py` checks the saved final score, export
record and all 228 raster/vector windows and generates
[`wsi_scale.json`](../benchmark/cmpb_evidence/wsi_scale.json). It performs no
inference. That annotation covers 8.14% of the slide, and the selected
window-local group readout was chosen on the same reference. Its result is a
descriptive scale and failure analysis, not a new independent test cohort.

These results cannot establish patient-independent performance on a genuinely
new institution, because the development images were inspected. A separate
frozen donor-to-target evaluation using predicted or clustered input tiles is
needed to test the intended application beyond expert-derived coarse maps.
