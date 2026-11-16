# Benchmark 4: one R entry script per Visium section

The three entry scripts run the full **new-analysis** path: read the original
10x count matrix and Space Ranger spot positions, normalize and cluster the
expression data with the indicated R front end, pass the clustered object to
`Bora::Bora.spatial()`, and save a mask, native-image-coordinate GeoJSON,
preview and receipt. The source image must be the original registered H&E.
Supply a reviewed tissue mask/GeoJSON in that image's full-resolution pixel
coordinates, so Virchow2 does not encode exterior glass.

| Section | RDS-only clustering script | One-command Bora script | Expression front end |
| --- | --- | --- | --- |
| Breast cancer | [`cluster_breast.R`](cluster_breast.R) | [`breast.R`](breast.R) | Giotto + KODAMA/Louvain |
| Colorectal cancer | [`cluster_colorectal.R`](cluster_colorectal.R) | [`colorectal.R`](colorectal.R) | Seurat + KODAMA/Louvain |
| Tonsil | [`cluster_tonsil.R`](cluster_tonsil.R) | [`tonsil.R`](tonsil.R) | SpatialExperiment + KODAMA/Louvain |

To create the R object used by the tutorial's `readRDS()` call, run the
matching `cluster_*.R` script after downloading the 10x data and generating
`image_metadata.json` with `inspect_ome.py`. The second argument is a **new**
output directory:

```bash
Rscript methods/benchmark4/cluster_breast.R \
  /data/visium/breast_giotto /data/clustered/breast-v1
Rscript methods/benchmark4/cluster_colorectal.R \
  /data/visium/colorectal_seurat /data/clustered/colorectal-v1
Rscript methods/benchmark4/cluster_tonsil.R \
  /data/visium/tonsil_spatialexperiment /data/clustered/tonsil-v1
```

Each command writes `NEW_OUTPUT_DIR/clustered-object.rds` and its
`expression-groups/` companion files. The RDS contains the clustered Giotto,
Seurat or SpatialExperiment object with full-resolution H&E spot coordinates
and `KODAMA_Louvain` labels. The shared
[`driver.R`](driver.R) attaches and checks Giotto labels against Space Ranger
positions before saving. Keep the original 10x input and companion files at
their paths; a SpatialExperiment may retain HDF5-backed assays. Clustering
uses R/CPU. The subsequent `Bora.spatial()` call uses the Virchow2 model to
create the mask and GeoJSON. The one-command scripts also save this RDS in
their own result directory before Bora inference.

From the Bora-Analysis repository root, for one section:

```bash
python3 methods/benchmark3/download_visium.py \
  --root /data/visium --dataset breast_giotto
python3 methods/benchmark4/inspect_ome.py \
  /data/visium/breast_giotto/CytAssist_FFPE_Human_Breast_Cancer_tissue_image.tif
export RETICULATE_PYTHON=/path/to/bora-python/bin/python
Rscript methods/benchmark4/breast.R \
  /data/visium/breast_giotto /data/support/breast-tissue.geojson \
  /models/virchow2 /data/results/breast-new-run-v1
```

Replace `breast_giotto`/`breast.R` with `colorectal_seurat`/`colorectal.R`
or `tonsil_spatialexperiment`/`tonsil.R`, and supply the corresponding tissue
support. `inspect_ome.py` reads OME PhysicalSizeX/Y and writes the
`image_metadata.json` required by the historical front ends. Inspect the
measured spacing and 10x registration before running. The output directory
must be new. The licensed Virchow2 weights must be available separately;
`Bora.download_virchow2()` can help fetch them after access is granted.

Install the [MIT Bora R package at tag `bora-r-v0.1.7`](https://github.com/tkcaccia/Bora/tree/bora-r-v0.1.7)
and its matching Python backend. The front ends additionally need Giotto,
Seurat or SpatialExperiment, KODAMA, `wsiTools` and the R dependencies imported
by their source files in
[`methods/benchmark3/legacy_clustering`](../benchmark3/legacy_clustering/).
The example's measured OME spacing, Space Ranger spot diameter and source
image must agree; the front ends stop on a failed registration check.

`Bora.spatial()` uses anonymous expression groups as sparse labels on this
same section. Its predictions away from spots are image-guided extrapolations,
not expert tissue or cell annotations. These scripts produce a **new** result:
they are not bit-for-bit reconstructions of the three published GeoJSONs,
whose complete historical feature/model inputs are not in the repository.
The published outputs can be integrity-checked with
[`verify_published.py`](../benchmark3/verify_published.py).

## Checks performed

`Rscript methods/benchmark4/tests/test_driver.R` passes synthetic Visium
front-end handoffs for all three routes, reloads each saved RDS and checks its
cluster labels and spot coordinates, then checks the `Bora.spatial()` call and
GeoJSON result. The Bora R package's `test-spatial.R` passes its 13 assertions
for registered data frame, Seurat, Giotto and SpatialExperiment objects.
`inspect_ome.py` was checked against a generated OME TIFF. A full real-image
rerun was not performed here because the three raw Visium count datasets were
not present in this workspace. Do not compare a new output hash to
the published hashes unless all historical inputs and software are recovered.
