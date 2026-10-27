# Reproduce the Bora benchmark pipeline

This directory contains the released data-handling route. It downloads
provider images, organizes a user-selected benchmark store, plans physically
calibrated Virchow2 tiles, encodes them, and runs Bora's single supported
Neighbor context classifier. It contains no algorithm-search code.

## Data and storage

Choose a data root outside Git and pass it through `--root` or the
`BORA_DATA_ROOT` environment variable. For example:

```bash
export BORA_DATA_ROOT=/path/to/bora-data
python pipeline/organize_data.py --root "$BORA_DATA_ROOT"
python -m benchmark.download --list-datasets
```

`organize_data.py` creates a `BENCHMARKS/`
index of relative links without copying images. Its optional `--layout` JSON
lets each installation map category names to its own directory names. It does
not move or rewrite source files or frozen manifests.

Public tissue datasets are acquired with `python -m benchmark.download`,
`download_catalog.py`, `download_ocelot.py`, and the pinned source catalogues in
`benchmark/` and `docs/`. The three H&E Visium gene-expression samples are
acquired with `download_visium.py`, using `visium_datasets.json`. Provider
terms and source checksums must be reviewed before redistribution. The four
additional Visium HD image/GeoJSON pairs in the WSI source index were imported from
a separate annotation collection; their annotations must not be treated as
independent expert ground truth until provenance and tissue-label semantics
are checked.

## Physical calibration and tile handling

Virchow2 expects 224 × 224 RGB pixels representing approximately 112 × 112
µm, or 0.5 µm per output pixel. The wsiTools additions staged under
`../integrations/wsiTools/` provide:

- `wsi_virchow2_geometry()` to verify microscopy spacing and determine native
  crop width and height separately for the two axes;
- `wsi_virchow2_tile_plan()` to cover a WSI without overlap, including
  partially filled right and bottom edges;
- `wsi_virchow2_spot_tile_plan()` to center the same physical crop on each
  Visium spot while retaining native spot coordinates;
- `wsi_virchow2_tile()` to return one padded, resized RGB tile in memory.

The functions reject an unknown scale and default to rejecting upsampling
from pixels coarser than 0.5 µm. Scanner objective magnification and TIFF
display DPI are not used as substitutes for verified µm/pixel. For the five
10x Visium HD WSIs, the provider's full-precision OME PhysicalSizeX/Y values
were passed explicitly because the current installed wsiTools version rounds
its extracted values. Their native Virchow2 crop is 409 × 409 pixels.

`plan_virchow2_tiles.R` saves a geometry table, not image tiles. For a source
with full-precision OME spacing:

```bash
Rscript pipeline/plan_virchow2_tiles.R IMAGE.btf PLAN_DIR \
  /path/to/virchow2-tiles.R MPP_X MPP_Y
```

When these functions are exported by an updated wsiTools installation,
omit the source file. The source file argument is a temporary integration
route while publication of the wsiTools update is pending.

`encode_virchow2.py` consumes this plan on an authorized GPU host. It crops one
native tile, pads an edge if needed, resizes, encodes Virchow2's full spatial
tokens, then releases the tile in memory. It saves only feature/position
chunks and SHA256 receipts under the selected data root. No tile images need deletion.
`prepare_wsi_jobs.py` freezes 1024-tile WSI units, and `run_wsi_queue.py`
waits for the public-field encoder to finish before processing those units.
Every unit has a timeout and GPU cleanup check; failure stops the queue.
Neither launcher submits a scheduler job; operators must follow their site's compute policy.

The five frozen WSI plans and queue are under the selected data root at
`preprocessing/virchow2-wsi-plans-v3/` and
`preprocessing/virchow2-wsi-v1/QUEUE.json`. The public-field queue is under
`preprocessing/virchow2-v5/`. Three additional Visium section plans and their
queue are under `preprocessing/virchow2-visium-spot-plans-v1/` and
`preprocessing/virchow2-visium-v1/QUEUE.json`; the breast source is slightly
coarser than 0.5 µm/pixel and explicitly records a 205-to-224-pixel
interpolation. The queued full-image grids and spot-centered crops are
distinct. `plan_visium_spots.R` uses `wsi_virchow2_spot_tile_plan()` to center
one physically calibrated tile on each registered expression spot. The frozen
spot plans and queue are in `preprocessing/virchow2-visium-spot-centered-plans-v1/`
and `preprocessing/virchow2-spot-v1/QUEUE.json`. `run_visium_queue.py` waits
for WSI encoding, and `run_spot_queue.py` waits for the Visium full-image
queue. `encode_spot_virchow2.py` retains spot barcodes alongside token chunks.
All three queues stop on the first failed unit. These are feature extraction
checkpoints, not trained models or benchmark scores.

## Frozen full-field evaluation

The 27 October 2026 split is fixed under the selected data root in
`preprocessing/full-rerun-2026-10-27/split-v1.json`. BCSS and TIGER
TCGA patient identifiers were reconciled before assignment. Twenty distinct
test fields each were reserved from BCSS, TIGER and IGNITE; training has 109,
111 and 268 catalog fields, respectively. A physical 45-µm coarse-grid audit
excluded 12 single-group IGNITE training fields before fitting, leaving 476
eligible training fields. No frozen test field was excluded. The exclusion
list and exact reasons are in `preflight-exclusions-v1.json`. OCELOT's 512
fields are a separate tumor-only external screen: code 1 is a broad automatic
non-cancer complement, not expert stroma. Its outcome must not be combined
with multiclass IoU. Other acquired datasets remain outside scoring until
spacing, pairing and tissue-code checks pass.

BEETLE contributes a separate capped breast-cancer development subset. Its
source images and masks are under `capped_tissue/beetle/` within the selected data root; twenty
calibrated, nonoverlapping 3072-pixel fields from twelve patients are in
`preprocessing/beetle-rois-v1/`. The provider's `data_overview.csv` confirms
that the selected patients came from three clinical centers and not from its
TCGA/TIGER subset. The SHA256-checked source audit is saved beside the ROI
manifest. Field selection used expert masks, and the official BEETLE evaluation
masks are sequestered. Source code 2, “non-invasive epithelium,” is not
equivalent to the frozen training class “in-situ tumor”; no semantic accuracy
for that code may be reported. This subset is not part of the primary 60-field
mean, and its new score is pending inference.

`prepare_multiscale_jobs.py` fixed the additional 1 and 2 µm/pixel Virchow2
views required by Bora's unchanged RGB conversion. The GPU queue runs
after the existing public-field, WSI, full-Visium and spot-centered jobs. Its
immutable `multiscale-queue-v2.json` contains the same 1,060 fields as v1,
ordered training, test, then OCELOT to allow primary validation first. The v1
waiting controller was stopped before it launched a GPU case, and its evidence
is preserved. `encode_multiscale_virchow2.py` stores only spatial tokens;
`run_multiscale_queue.py` requires a per-case cleanup receipt and stops the
entire queue after a failure. All large source images and output features are
stored under the selected data root; the compute host may be separate. The
queue runner does not submit scheduler jobs.

`build_bora_case.py` assembles checksum-verified base tokens and both context
scales, an aligned 0.5 µm/pixel H&E image, and independent image-derived tissue
support. It coarsens the expert mask to 90 working pixels (about 45 µm), then
permutes its class numbers **separately for each image**. The expert mask is
kept outside the predictor `case.json`. This controlled benchmark measures
refinement and whole-region classification conditional on expert-derived
coarse geometry; it is not autonomous tissue discovery.

`build_training_manifest.py` joins prepared anonymous groups to expert tissue
roles only after `bora prepare` has completed. Canonical roles and ignored
codes are in `benchmark/tissue_label_harmonization.json`. The model uses the
single Neighbor context fastPLS classifier. `score_bora_case.py` scores final
masks on annotated pixels, records patient/source identity and group accuracy
by the number of input tiles, and labels the expert-decoded coarse comparison
as an oracle diagnostic. It reports OCELOT tumor recall and a weak broad
negative control separately. Figures and the public viewer must be regenerated
from complete, verified score receipts whenever the method or cohort changes.

`run_primary_after_multiscale.py` is a CPU-only, fail-fast handoff. It verifies
the existing multiscale queue's immutable hash, PID/start time, ordered
checkpoints and per-case GPU cleanup receipts. Once the 488 training feature
cases are complete, it invokes `run_primary_cpu.py` to build 476 eligible
cases, prepare descriptors and fit the released Bora 0.2.1 model. After the
60 designated evaluation feature cases are complete, it refines and scores
them and runs `aggregate_primary_scores.py`. Every phase writes an atomic
completion receipt; a failure stops the controller without a retry. This
controller never starts a GPU worker or scheduler job.

The separate `run_ocelot_external_cpu.py` may be run only after all 512
OCELOT context features and the primary fit are complete. It uses the
already fitted model and reports expert tumor recall and the tumor fraction
on the provider's broad automatic non-cancer region. It does not assign a
multiclass IoU to that dataset. `build_full_rerun_viewer.py` requires the
complete 60-field primary summary and each score/output checksum before
constructing public previews and compressed native GeoJSON. It writes to a
new versioned directory; it cannot replace an existing viewer or publish
partial results.

The planned 60-field test is a development evaluation because some source
material and methodology have been inspected in prior work. It must not be
called a previously untouched independent test.

## Bora classification and refinement

The released [Bora 0.2.1 workflow](https://github.com/tkcaccia/Bora/blob/main/docs/WORKFLOW.md)
accepts a registered working H&E image, anonymous coarse tile labels, full
Virchow2 spatial-token field, tissue support, per-axis spacing and the pinned
encoder identity. Token chunks must first be assembled into each Bora case's
registered working grid; the source image and feature coordinate systems must
match. A WSI feature queue alone is not a Bora case.

Prepare each annotated training image with `bora prepare`. Supply its expert
tissue mask or native-coordinate GeoJSON only in a `bora train` manifest.
`run_bora.py` performs these fixed commands and then calls `bora refine` on
separate target patients or slides. Its workflow JSON has this form:

```json
{
  "format": "bora-analysis-workflow-v1",
  "classes": {"1": "tumor", "2": "stroma"},
  "training": [{
    "case": "cases/train-A.json",
    "annotation": "annotations/train-A.tif",
    "annotation_values": {"10": 1, "20": 2},
    "source_group": "patient-A"
  }],
  "targets": [{"case": "cases/test-B.json", "source_group": "patient-B"}]
}
```

```bash
python pipeline/run_bora.py workflow.json --output new-run
```

The target case contains no expert mask. Bora writes the final native tissue
mask and GeoJSON with the predicted tissue class for each refined group. The
workflow checks that training source groups are absent from targets. A held-out
dataset and patient-disjoint fields must be fixed before reporting accuracy.

An [annotation-free coarse-map example](NONSPATIAL_WSI_KODAMA.md) clusters
Virchow2 tile features with KODAMA without slide coordinates, projects the
anonymous groups to a colorectal whole-slide image, and tests one registered
window with Bora. It is an exploratory same-slide pilot; its tissue-name model
was not selected independently of the target slide.
