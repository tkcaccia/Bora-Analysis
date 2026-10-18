# Benchmark organization and reproduction

The application is maintained in https://github.com/tkcaccia/Bora . This
repository, formerly Bora-extra, preserves acquisition, research and viewer
code. Renaming retained the existing Git history.

## Published classifier study

The public site packages completed cycle-03 of the revised classifier study,
`bora-classifier-evolution-2026-10-18-v2`. It includes all 116 eligible multiclass
evaluation images: BCSS 42, TIGER 26, GCSS 24 and IGNITE 24. PUMA is excluded from
both evaluation splits. The original split names (78 test and 38 validation)
are metadata, not a claim of untouched testing: both were reused adaptively.

The main page selects P04 Neighbor context; the archive preserves the previous
five-method comparison on the exact same images. No scientific results are
recomputed when the website is built. Images and geometry are copied exactly,
and source hashes are recorded in `site/BUILD_PROVENANCE.json`.

The fixed study inputs are expert-derived anonymous coarse groups, calibrated
Virchow2 features, RGB9+19 conversion and Bora-AF geometry. The classifier
identifies whole refined tissue groups. P04 combines each group's 1280-dimensional
feature mean with its 1280-dimensional contrast to physical shared-boundary-
weighted neighboring groups, and fits one fastPLS SIMPLS-LDA head. The benchmark
uses 25 represented training classes and 24 components. The independent
training partition contains 967 groups; it is not part of
the 116-image evaluation viewer. There are 28 mapped ontology roles, and some
evaluation roles have no represented training examples.

All tissue can contribute to image processing, while unannotated/unmapped
reference pixels are excluded from scores. Zero source codes cannot be assumed
to mean background: read each dataset's label mapping. The displayed expert
legend retains source annotation names and flags excluded labels.

Final vectors preserve ROI-local native pixel coordinates and per-axis µm/px.
Browser JPEGs are display previews, not calibrated model inputs. Polygon parts
and enclosed holes strictly below 20 native pixels of area are removed or
filled, while exactly 20 is retained. The continuous-area and native-raster
audits are preserved with each export. Shared-boundary straight-line fitting
uses at most 3 µm tolerance and the study's predicted-shape agreement and area
guards. It does not fit against expert target boundaries.

`scores.csv` and `data.json` are downloadable from each viewer. The mean is
computed over all 116 images. Cached deployment times include the recorded
geometry, cleanup and fitting costs, but exclude encoder extraction, cold
loading and training. They are not whole-slide end-to-end times.

## Research source locations

* `benchmark/`: original acquisition and paired-image preparation; source URLs,
  hashes, annotation semantics, microscopy calibration, image-type records and
  earlier controlled-refinement scoring.
* `history/bora-development/experiments/bora_evolution/`: classifier candidates,
  group descriptors, training, evaluation, evolution, revised-cohort preparation,
  geometry postprocessing and original viewer generator.
* Other `history/bora-development/experiments/` directories: Virchow2 RGB
  conversions, calibrated eligibility, tissue acquisition and prior Bora
  strategies. The historical tests and legacy application runtime are preserved
  alongside them. These are not installed into the current Bora application.
* `manuscript/`: earlier figure/document builders. Historical manuscripts and
  their cohort-specific results are not descriptions of the selected release.
* `tools/build_public_viewer.py`: packages the completed results without training.
  `tools/verify_public_viewer.py` checks the complete publication artifact.

Old experimental modules import the original `experiments` and `bora`
namespaces. For historical reproduction, use the frozen source/environment from
that study or its matching original Bora revision; do not silently run them
against the cleaned P04 application. Files under `history/` are provenance and
contain no instruction or authorization to launch more work.

## Data retained outside Git

On Chiamaka, downloaded datasets are organized below
`/mnt/sata_ssd/Bora-benchmark`; compute studies and feature caches are below
`/home/chiamaka/Bora-benchmark-compute`. On the development Mac, mirrored viewer
results are below `Bora/benchmark/results`. Acquisition inventories, raw image
and mask files, large features and gated checkpoint weights remain in those
local data stores, outside this repository. The website publishes attributed
previews and predicted vectors only. Follow the original provider's access and
reuse terms when obtaining full data.

The evolution study was explicitly stopped by the user. Do not restart any
archived controller. UCT HPC GPU execution remains paused under the project's
coordination policy; this website deployment performs no GPU work.
