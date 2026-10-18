# Reproducing the manuscript's descriptive reanalysis

The article reports a **conditional, exploratory** evaluation of Bora on 116
previously inspected, calibrated multiclass fields. Its coarse tile assignments
come from expert annotations. The original `test` and `validation` names are
retained to describe the source registry; neither split is an untouched test.

The repository records sanitized numerical inputs and outputs in
[`benchmark/cmpb_evidence`](../benchmark/cmpb_evidence). `registry_groups.json`
contains case IDs, dataset/split and known source-group identifiers. The three
method summaries contain per-case IoU, group correctness and cached timings.
`classifier_controls.json` gives matched whole-group controls trained from the
same frozen descriptor matrix. `boundaries.json` gives native-resolution
tumor–stroma interface scores for each image. `vector_fidelity.json` gives
polygon-fitting diagnostics relative to the **preceding predicted geometry**.
None of these JSON files contains raw WSI pixels, spatial-token tensors or
gated model weights. Obtain original archives under each provider's terms as
described in [the acquisition guide](MANUSCRIPT_MODEL_DATA.md).

The aggregate can be recomputed without model inference:

```bash
python benchmark/analyze_cmpb_cohort.py \
  --viewer site/data.json \
  --registry benchmark/cmpb_evidence/registry_groups.json \
  --neighbor benchmark/cmpb_evidence/neighbor_summary.json \
  --mean benchmark/cmpb_evidence/mean_summary.json \
  --spatial benchmark/cmpb_evidence/spatial_summary.json \
  --output /tmp/bora-reanalysis.json
```

The scripts `benchmark/analyze_cmpb_classifier_controls.py`,
`benchmark/analyze_cmpb_boundaries.py` and
`benchmark/analyze_cmpb_vector_fidelity.py` reproduce their respective JSONs
from the frozen group descriptors, provider masks and saved output geometry.
Those larger inputs are retained outside Git because of provider and checkpoint
terms. The first script uses only the CPU; the other two read saved outputs and
perform no encoder or segmentation inference. The public summaries make the
reported arithmetic independently inspectable even without those inputs.

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
arguments. Author names and affiliations come from the retained author draft.
The generated DOCX and private figure source masks are kept outside Git.

These results cannot establish patient-independent performance on a genuinely
new institution, because the development images were inspected. A separate
frozen donor-to-target evaluation using predicted or clustered input tiles is
needed to test the intended application beyond expert-derived coarse maps.
