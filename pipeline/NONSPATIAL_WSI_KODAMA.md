# Nonspatial KODAMA coarse maps for Bora

This route tests Bora with tile groups generated without expert assignments.
Virchow2 features and an H&E image are required. KODAMA and Louvain receive
only image features; tile coordinates are used afterward to place the frozen
groups back on the whole slide. An expert mask is opened only for evaluation.

This is distinct from the Visium expression workflow, which passes spot
coordinates to KODAMA. The example here uses H&E-derived Virchow2 features.
The scripts require Python with NumPy, pandas, scikit-learn, threadpoolctl,
Pillow and tifffile, and R with KODAMA, RANN, igraph and jsonlite.

## Inputs and calibration

Use the complete, checkpointed Virchow2 whole-slide feature set described in
the [preprocessing guide](README.md). Every nonoverlapping native tile must be
resampled to 224 × 224 pixels at approximately 0.5 µm/pixel. Keep the source
image checksum, per-axis pixel spacing, encoder revision and the tile table.
The `collect_wsi_pooled.py` reader verifies every completed feature chunk
before extracting 1,280-channel pooled tile descriptors. It does not read
expert labels. Its source feature format is the `unit-*` output of
`encode_wsi_virchow2.py` in this directory.

```bash
python pipeline/collect_wsi_pooled.py \
  --features-root data/preprocessing/colon-features \
  --output work/pooled

python pipeline/prepare_wsi_kodama.py \
  --pooled work/pooled --sample-count 2000 --seed 20261027 \
  --output work/kodama-input

Rscript pipeline/run_wsi_kodama.R \
  work/kodama-input/sample_pca.csv work/kodama-clusters

python pipeline/project_wsi_kodama.py \
  --pooled work/pooled --kodama-input work/kodama-input \
  --clusters work/kodama-clusters --output work/projected
```

The 30-component principal-component analysis is fitted on the sampled
standardized tile features. KODAMA uses `spatial = NULL`, `M = 100` and seed
1234. Louvain is applied to a 20-neighbor graph on the two-dimensional KODAMA
plot. The selected resolution is sought from 0.10 to 1.50 against a six-group
display target fixed before annotation inspection. It may yield a different
number of groups: the colorectal example yielded 13, and these were retained.
A seven-neighbor inverse-distance vote in the same feature space assigns
sample groups to the remaining slide tiles. Cross-validation measures only
agreement with the sample's anonymous clusters, **not tissue accuracy**.

`projected/tile_groups.npy` is a complete row-major anonymous tile grid.
For a bounded ROI, use the native origin and dimensions to register it to a
calibrated Bora working image:

```bash
python pipeline/crop_wsi_groups.py \
  --grid work/projected/tile_groups.npy \
  --working-image cases/roi/original.tif \
  --origin-xy X Y --native-shape-yx HEIGHT WIDTH \
  --native-tile-wh TILE_WIDTH TILE_HEIGHT \
  --output cases/roi/kodama-coarse.tif
```

That ROI can be supplied as `coarse` in a normal Bora 0.2.1 case with the
registered 0.5 µm/pixel H&E image, full Virchow2 spatial tokens, 1 and
2 µm/pixel context features, native tissue support and a predeclared tissue
classifier. `bora refine` outputs a native mask and GeoJSON. For independent
semantic testing, fit and select the tissue classifier only on other patients
and slides, then score held-out expert pixels after all predictions are frozen.

The public colorectal example used all 22,080 tile features for clustering
and checked one 2,048 × 2,048-native-pixel tumor–stroma field with cached
multiscale features. The field and classifier had previously been inspected
on that same slide. Its score is therefore an exploratory demonstration, not
a patient-independent validation or a full-slide KODAMA–Bora result. The
previous full-slide run used a different K-means coarse map. Keep those
experiments separate.
