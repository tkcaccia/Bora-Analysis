# Bora supervised tissue experiments

**Task correction:** the user clarified that the primary prediction unit is
the tissue segment **after Bora**, with one top-1 identity and a top-5 shortlist
per connected region. The tile and superpixel studies below are preserved as
controls. The corrected study lives in `experiments/bora_segment`; see
`SEGMENT_TASK_CORRECTION.json`. Manuscript claims must use that distinction.

This is a new supervised study, separate from the stopped refinement cycles.
It develops an extendable labelled tissue registry, native fastPLS CPU heads,
ranked tissue options, and matched image-based refinement comparisons.

## Methods and evidence

The manuscript names J01 **Bora-AF**, H&E-guided Appearance Fields. It fits a
spatial appearance model for each supplied class using H&E edge conductance,
then compares evidence RGB with those fitted appearances. N08 is **Bora-VAF**,
Variance-Adaptive Appearance Fields. It uses a bending regularizer and fitted
log-variance fields; its spatial regularizer differs from J01.

Neither a graph Laplacian nor local Gaussian appearance modelling is new.
Related work includes the Fast Bilateral Solver, random walkers, local Gaussian
distribution fitting, FeatUp, Upsample Anything, and trained pathology dense
predictors. See the revised manuscript for scope and citations. The controlled
reference-derived RGB experiment and the supervised tissue study have different
inputs and label spaces. Their numerical scores must not be pooled.

Three supervised development cycles were defined and run:

| Cycle | Formulations | Purpose |
|---|---|---|
| 1 | S01 centre token; S02 physical tile mean; S03 pure-tissue training; S04 tissue-fraction regression | Test local representation and mixed-tile target handling |
| 2 | S05 spatial context PLS-DA; S06 native PLS-LDA; S07 contextual native PLS-LDA; ridge and centroid controls | Test architectural context and the classification head against matched linear baselines |
| 3 | S08 H&E superpixel PLS-DA; S09 superpixel tissue-fraction regression | Classify features pooled inside an image-derived segmentation |

Component counts are hyperparameters, not additional algorithms. The first
exploratory reports include all69 validation ROIs. `ELIGIBLE_RESULTS.json`
retains only the39 ROIs with at least two **distinct mapped expert tissue
roles**. Thus several Gleason patterns collapsed to tumour do not count as
several classes, and an automatic non-cancer complement does not count as
expert stroma. The complete training set retains valid single-role examples.

The immutable source registry contains315 training,69 validation and85 test
ROIs, comprising102638 physical45um units. Known TCGA case identities and exact
source-image duplicates are linked before partitioning. Exact provider split
tokens are reserved by the parser; development-tagged releases are repartitioned.
In particular, PESO's provider test release was explicitly tagged as development:
24 fields enter internal training, 9 validation and 7 internal test. The original
registry's blanket statement that provider test groups never enter training is
incorrect. Its immutable bytes are preserved with a separate source-split audit.
These are internal partitions, not claims of preserving every provider test set.
Other source groups are not all verified patients.
Images were previously viewed during Bora development, so this is held-out
evaluation of a new tissue head, not independent validation of the selected
Bora architecture. Final multiclass test eligibility is62 ROIs.

The ontology has27 stable broad-role IDs. Original provider labels remain in
each registry row. Heterogeneous residual categories and automatic negatives
are excluded. Gleason subtypes are retained as metadata, not predicted grades.
This mapping needs pathology review before claims of semantic equivalence.

## Reproducible dependencies

The implementation uses `tkcaccia/fastPLS-py` commit
`fc0c011969b754ee196aacc220a217783afeb05d`, version0.3.0, with the canonical
fastPLS C++ core pinned by its `UPSTREAM_CORE.json` to
`82bbc48a0d69e4bd0d7c261fdcc8e636694133b1`. `backend="cpu"` is explicit.
This release's bounded `simpls` route is not described as an unqualified
implementation of classical de Jong SIMPLS. Numerical libraries use two threads.

`predict_scores()` returns latent PLS scores. `predict_responses()` returns
response predictions. Native LDA ranks come from `predict(X, top=5)`; response
argmax is not substituted for the LDA decision rule.

Linear response exports are qualified against native validation predictions.
They are mathematical equivalents measured separately from native head speed.
The public API does not serialize fitted native objects. Deployment bundles
therefore contain frozen training arrays and model settings to reconstruct
the native head once. Startup refitting and warm prediction latency are
reported separately. No test annotations enter those bundles.

The Fast Bilateral Solver comparator uses the author's MIT-licensed reference
notebook at commit `4b6ef7e4c0d151aee2d3480f26a876b41ce07435`. Definitions are
executed unchanged; Python `xrange` and SciPy `cg(tol)` receive compatibility
adapters. Convergence status is recorded. This notebook is not the optimized
implementation behind the paper's speed claims. A500-iteration follow-up
checks the default25-iteration truncation.

## Data contract and extension

The raw predictor is the full unselected1280-channel Virchow2 spatial-token
grid. CLS and register tokens are excluded. At0.5um/px the14-pixel token grid
has7um spacing, while each224-pixel inference window has112um context. The
grid spacing is not the optical resolving power. Unknown optical resolution
and objective magnification remain unknown. Old per-image selected-channel
multiscale caches and reference-conditioned RGB are not used by tissue heads.

Each immutable version contains `REGISTRY.json`, a fixed ontology and per-case
feature/target shards. `PREPARED.json` hashes every shard. Test targets are
not prepared. Fitters use train targets only; validation predictions are
frozen before scoring. All final test predictions must be frozen before the
test evaluator opens any annotation.

To add data, create a new calibrated source plan and run `registry` with
`--previous OLD_VERSION --root NEW_VERSION`. Existing case identities, source
groups, splits and hashes must remain unchanged. A revised `--ontology` may
append role IDs and source-label mappings but cannot alter existing meanings.
New duplicate links that would change an old partition require an explicit
version migration; they are never silently split. Then prepare new shards and
retrain a new model version. A retired test cohort must not silently become a
training cohort under the same evaluation protocol.

Example commands from the repository, with environment paths supplied by the
operator:

```bash
python -m experiments.bora_supervised.registry --source CALIBRATED_INPUTS --results RESULTS --root REGISTRY_V1
python -m experiments.bora_supervised.prepare --root REGISTRY_V1
python -m experiments.bora_supervised.train --root REGISTRY_V1 --output CYCLE_01 --dependency FASTPLS_SOURCE
python -m experiments.bora_supervised.cycle_two --root REGISTRY_V1 --output CYCLE_02 --previous CYCLE_01
python -m experiments.bora_supervised.regions cycle --root REGISTRY_V1 --output STUDY_ROOT
python -m experiments.bora_supervised.eligibility --root REGISTRY_V1 --cycle CYCLE_01
python -m experiments.bora_supervised.eligibility --root REGISTRY_V1 --cycle CYCLE_02
python -m experiments.bora_supervised.eligibility --root STUDY_ROOT/regions-v1 --cycle STUDY_ROOT/cycle-03
python -m experiments.bora_supervised.evaluate --root REGISTRY_V1 --regions STUDY_ROOT/regions-v1 --cycles CYCLE_01 CYCLE_02 STUDY_ROOT/cycle-03 --output STUDY_ROOT/validation-native-v1
python -m experiments.bora_supervised.final predict --base STUDY_ROOT --output STUDY_ROOT/test-v1
python -m experiments.bora_supervised.final evaluate --base STUDY_ROOT --output STUDY_ROOT/test-v1
```

`final.py` contains this study's preselected model IDs and refuses changed
selection evidence. A new study must version that selection and its protocol.

For a new unannotated ROI, `predict.py` accepts an H&E image, its calibrated
pixel spacing and full raw token cache. Its feature sidecar must identify the
input image hash, feature hash, exact encoder revision, window and patch size.
No tissue annotation or per-image label vocabulary is accepted. Resample the
image and extract features again when the encoder's physical sampling changes;
editing a cache's MPP metadata cannot fix a different receptive field.

```bash
python -m experiments.bora_supervised.predict --model STUDY_ROOT/models/S07-c24 --features NEW/base.npy --image NEW/original.png --feature-metadata NEW/feature-provenance.json --mpp-xy 0.5 0.5 --output NEW/prediction-v1
```

Outputs include `top1.tif`, `ranked_units.npz` and named options in
`tissue_options.json`. Ranks are not calibrated probabilities. The exterior
white heuristic preserves enclosed lumina but can mistake border adipose for
glass; its errors remain errors on expert annotation pixels.

## Chiamaka locations

Study data and model bundles:
`/home/chiamaka/Bora-benchmark-compute/tissue-supervised-2026-10-18-v1`.
Downloaded source datasets remain in `/mnt/sata_ssd/Bora-benchmark`.
Versioned JSON reports are mirrored under the local
`benchmark/results/bora-supervised-2026-10-18-v1` directory.

The viewer is served through the existing local SSH tunnel at
<http://127.0.0.1:64889/bora-supervised-2026-10-18-v1/>. It preserves each
completed cycle and displays expert annotations, H&E and model overlays, with
ranked tissue choices available for every unit.
