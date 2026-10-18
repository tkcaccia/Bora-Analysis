# Identify tissue segments after Bora

**Superseded primary workflow:** the user clarified that input groups must come
from the previous expert-derived tile mask, with IDs deidentified only. Use
`experiments/bora_group` for the current benchmark. The fixed-four-cluster and
connected-fragment experiment below is preserved as exploratory history.

The prediction unit is a **connected native-resolution Bora tissue segment**.
Physical tiles supply coarse grouping and evidence. A tile label is never the
final output of this study. Each positive region ID receives one tissue
identity and one ranked shortlist of up to five roles. Boundaries remain fixed
while semantic classifiers are compared.

## Geometry before identity

1. Use H&E calibrated to 0.5 µm per pixel and its full unselected 1280-channel
   Virchow2 spatial tokens. Fourteen-pixel token spacing is 7 µm; a 224-pixel
   encoder window covers 112 µm. Neither number is optical resolving power.
2. Pool features into 45 µm physical fields. Fit within-image PCA and four
   KMeans groups using fields with at least 50% H&E tissue and at least 36 µm
   extent in **each** axis. Predict all fields, including clipped edges.
3. Run the existing algorithm 19 RGB conversion with these anonymous groups.
   No fine reference or reference-derived selected-channel cache is supplied.
4. Apply Bora-AF. Project directly to the native grid and remove exterior
   background using the H&E-derived support. Never use the expert annotation
   as the tissue support.
5. Label eight-connected components within each anonymous group. Preserve
   both the anatomical region ID and its original anonymous group ID. Pool
   full token means, standard deviations, and exact native-pixel tile overlaps.
6. Freeze and hash **all** geometry and evidence before attaching training or
   validation labels. Test targets remain unprepared.

The four-group setting is a fixed proof-of-concept choice, not an estimate of
the true number of tissue types. Regions can be fragmented or cross several
expert tissue classes. Report their annotation purity and the pixel-accuracy
ceiling of a perfect majority-label region classifier. Semantic assignment
cannot repair an incorrect frozen partition.

Version 1 stopped at `ignite_20`: a clipped 11 by 1 pixel corner formed a
singleton grouping observation and disappeared on the analysis grid. Version
2 corrects physical field admission before clustering and regenerates every
region. The incomplete version and its log are retained. The correction was
chosen from image geometry, without consulting semantic labels.

## Five segment identification families

| Family | Evidence used for one whole segment | Native CPU head |
|---|---|---|
| R01 | Area-weighted Borda vote of complete tile rankings inside its refined boundary | Existing trained tile PLS-LDA, then regional voting |
| R02 | Full-feature region mean | fastPLS PLS-LDA |
| R03 | Full-feature region mean and standard deviation | fastPLS PLS-LDA |
| R04 | Full-feature region mean with training tissue fractions | fastPLS multivariate response regression |
| R05 | Region mean and contrast with shared-boundary-weighted neighboring region means | fastPLS PLS-LDA |

R02 through R05 compare 16, 24 and 48 latent components with autoscaling.
These are hyperparameters within four families. All heads use the pinned
fastPLS C++ CPU backend from the companion supervised study and two numerical
threads. A stable capped reservoir selects at most 1000 training segments per
dominant tissue role; a segment must have mapped annotation on at least half
its area. R04 learns the composition within the **annotated portion**, without
assigning ignored pixels to a tissue role.

Select configurations using region balanced accuracy on the 39 eligible
validation ROIs. Single-role ROIs remain valid training examples; they are
excluded from multiclass benchmarking. Report top-one, top-five, per-role
recall, and native pixel IoU. The pixel metric uses every mapped annotated
pixel, including low-coverage and mixed regions and annotations wrongly
removed by the foreground heuristic. No mask value zero is taught as stroma.

The registry retains its 315 training, 69 validation and 85 reserved test ROIs,
with 62 test ROIs eligible for multiclass scoring. These images were already
used during refinement development, and the test was already scored for the
earlier tile/superpixel controls. Follow-up results preserve training-group
separation but are exploratory reused-test results, not an untouched final
evaluation. Provider ontologies are mapped to 27 stable broad roles and still
require pathology review. See `experiments/bora_supervised/README.md` for
grouping caveats and append-only registry extension.

## Execution

Use a new output directory for every immutable version. These modules do not
download or extract a new encoder; they consume the calibrated source registry
and full feature shards already prepared by `bora_supervised`.

```bash
python -m experiments.bora_segment.geometry --registry-root REGISTRY --output GEOMETRY
python -m experiments.bora_segment.targets --registry-root REGISTRY --geometry GEOMETRY --output TARGETS
python -m experiments.bora_segment.train all --registry-root REGISTRY --geometry GEOMETRY --targets TARGETS --root VALIDATION --tile-bundle STUDY/models/S06-c24 --tile-registry REGISTRY
python -m experiments.bora_segment.viewer --registry-root REGISTRY --geometry GEOMETRY --cycle VALIDATION --output VIEWER
python -m experiments.bora_segment.final --registry-root REGISTRY --geometry GEOMETRY --validation VALIDATION --targets TARGETS --tile-bundle STUDY/models/S06-c24 --tile-registry REGISTRY --output FOLLOWUP_TEST
```

For an already frozen unannotated case, the direct region predictor accepts
only its geometry/evidence and a region model bundle:

```bash
python -m experiments.bora_segment.predict --model VALIDATION/R02-c24 --geometry-case GEOMETRY/cases/CASE --native-mpp-xy 0.25 0.25 --output PREDICTION
```

`top1.native.tif` holds semantic tissue IDs. `ranked_regions.npz` and
`tissue_segments.json` hold one ranking per positive native region ID. Keep
`regions.native.tif` alongside the semantic mask: adjacent regions assigned
the same tissue identity remain separate regions. Ranked options are not
calibrated probabilities.

Native fastPLS objects have no public serialization in the pinned release.
Bundles preserve training arrays and settings and reconstruct the head once.
Report startup refit, feature/context preparation, and warm head prediction
separately. Cached-feature grouping, RGB construction, Bora refinement and
region pooling are also timed separately. Millisecond classifier latency is
not the runtime of the complete Virchow2 pipeline.
