# Benchmark organization and reproduction

[Bora](https://github.com/tkcaccia/Bora) contains the released tissue
refinement and Neighbor context classifier. This repository contains public
dataset acquisition, preprocessing, scoring, figures and the viewer. The
[pipeline guide](../pipeline/README.md) gives the runnable sequence.

## Evaluation data

The public viewer has 116 calibrated, multiclass tissue images: BCSS 42,
TIGER 26, GCSS 24 and IGNITE 24. Its 78 test and 38 validation labels were
reused during development, so its scores are exploratory rather than a new
independent confirmation. PUMA is excluded from those two evaluation splits.
The tissue classifier was trained on 967 refined whole-group examples from
349 annotated source fields; the exact source IDs are in
[the model source manifest](manuscript_model_training_sources.json).

The coarse tile mask in the public viewer is derived from the expert tissue
annotation and anonymized before refinement. That benchmark measures
geometry and tissue-group identity under a controlled coarse input; it does
not establish that the pipeline can discover a correct tile clustering from
H&E alone. Score only pixels covered by a valid expert annotation. Source
code zero has dataset-specific meaning and must be mapped explicitly.

The whole-slide section uses five Visium HD image/GeoJSON pairs. The original
colorectal annotation is user supplied; the other four GeoJSON files have
unresolved provenance or tissue-label semantics and are not independent
accuracy references. The three Visium gene-expression datasets are breast,
colorectal and tonsil H&E sections with spot-cluster inputs, a separate use
case from the annotated histopathology endpoint.

## Data layout

After running `python pipeline/organize_data.py --root "$BORA_DATA_ROOT"`,
start at `$BORA_DATA_ROOT/BENCHMARKS/README.md`:

1. `01_public_tissue_fields/` indexes calibrated image/tissue-mask pairs.
2. `02_whole_slide_images/` indexes full-slide images and GeoJSONs.
3. `03_visium_gene_clustering/` holds the three expression datasets.
4. `04_private_development/` is excluded from the public manuscript.

The numbered folders are links to original source directories; source paths
in download and preprocessing receipts remain valid. Feature checkpoints are
under `preprocessing/`. Virchow2 crops are calibrated by the source's
µm/pixel and resized to 224 × 224 RGB at about 0.5 µm/pixel. No tile image
files are retained by the current encoder.

## Published outputs

The website packages the completed raster and GeoJSON outputs. Its preview
JPEGs are not calibrated model inputs. Final vectors use ROI-local native
pixel coordinates and preserve the per-axis source spacing. Small polygon
parts and enclosed holes under 20 native pixels are removed or filled; exact
20-pixel features remain. Reported Bora deployment times exclude upstream
Virchow2 extraction and model training. Source-image model training and
held-out evaluation remain separate operations.

The checked-in archive is checked by `tools/verify_public_viewer.py`. A new
viewer is generated from complete full-rerun receipts by
`pipeline/build_full_rerun_viewer.py`, as described in the pipeline guide.
Large source images, model weights and token caches are outside Git. Respect
each provider's data-use terms.
