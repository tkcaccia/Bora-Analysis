# Bora-Analysis

Benchmark acquisition, scoring, research workflows, figures and manuscript
builders for [Bora](https://github.com/tkcaccia/Bora).

**[Open the three-section benchmark viewer](https://tkcaccia.github.io/Bora-Analysis/)**

The viewer separates three distinct evaluations:

1. [Public histology datasets with expert masks](https://tkcaccia.github.io/Bora-Analysis/) — BCSS, TIGER, GCSS and IGNITE.
2. [Whole Slide Images](https://tkcaccia.github.io/Bora-Analysis/visium-hd/) — colorectal cancer, pancreas, lymph node and tonsil. Each tissue page shows the full-slide mask, its GeoJSON, slide-wide group responses, and nine exact crops of the saved result. The groups are anonymous and do not establish tissue-class accuracy.
3. [Visium gene-expression clustering](https://tkcaccia.github.io/Bora-Analysis/visium/) — breast/Giotto, colorectal/Seurat and tonsil/SpatialExperiment spot clusters with selected Bora refinement fields. These examples have no expert pixel-level tissue mask.

The 27 October 2026 full rerun is in progress. Its frozen protocol and
reproduction code are documented in [the pipeline guide](pipeline/README.md)
and [the manuscript integration note](manuscript/FULL_RERUN_INTEGRATION_2026-10-27.md).
The public viewer and Word article below still report the earlier inspected
development cohort. New scores and figures will replace them only after all
designated cases and their output checksums are complete.

The main viewer shows **Neighbor context**, the selected tissue-group
classifier, on **116 calibrated multiclass images**: BCSS 42, TIGER 26,
GCSS 24 and IGNITE 24. Each image includes the original H&E, expert mask and
legend, anonymous expert-derived input tile mask, fixed RGB9+19 conversion,
Bora-AF groups, final classified mask, scores, per-axis pixel resolution and
interactive simplified GeoJSON. PUMA is excluded from evaluation.

The original split labels are preserved (78 test, 38 validation). These are
**exploratory benchmark results**, not independent final validation. The coarse input is derived from expert annotations;
this experiment does not establish annotation-free segmentation.

The benchmark and Bora application use Neighbor context as the sole tissue
classifier. Upstream clustering or tile prediction supplies anonymous groups;
Neighbor context assigns tissue identities after refinement.

The earlier [colon WSI study](studies/visium-hd-colon-wsi-2026-10-19/)
is retained for provenance. The current Visium HD viewer above presents the
four whole-slide examples separately from the public-mask cohort.

The [standard Visium workflows](pipeline/visium_clustering/) provide Seurat,
Giotto and SpatialExperiment clustering for colorectal, breast and tonsil
sections, respectively. Their spot coordinates are passed to calibrated
Virchow2 tile plans and then Bora. The local exploratory study directory is
excluded from this repository; it contains research code and large private
checkpoints. The public sections do not include independent pixel-level tissue
annotations, so these results are descriptive.

To train a tissue classifier for new images, use the
[Bora training and inference guide](https://github.com/tkcaccia/Bora/blob/main/docs/WORKFLOW.md)
and the [dataset-to-Bora pipeline guide](pipeline/README.md).
The [manuscript model data guide](docs/MANUSCRIPT_MODEL_DATA.md) lists the
exact 349 annotated source fields, provider download routes, native crop
coordinates and checksums used for the reported 967 training groups.
The [manuscript reanalysis note](docs/CMPB_REANALYSIS.md) provides the
descriptive cohort, boundary and vector-fidelity evidence
with commands and limitations.
The [pipeline resource pilot](docs/PIPELINE_RESOURCE_BENCHMARK.md) reports
stage times and whole-process peak memory on four TIGER regions with cached
Virchow2 features. The [GeoJSON resource pilot](docs/GEOJSON_RESOURCE_BENCHMARK.md)
compares three exporters on identical predicted masks.

## Repository organization

| Location | Contents |
| --- | --- |
| `benchmark/` | Dataset acquisition, paired-image preparation, calibration/annotation metadata, scoring and audit code |
| `pipeline/` | Download, organize, physically calibrate, encode Virchow2 and run the fixed Bora workflow |
| `integrations/wsiTools/` | Proposed calibrated tile-plan and in-memory tile functions for the wsiTools R package |
| `manuscript/` | Figure and document generation code |
| `web/` | Human-editable static viewer HTML, JavaScript, CSS and data attribution |
| `tools/` | Viewer-asset assembly, portability and published-site checks |
| `site/` | Published viewer assets, per-image scores, metadata, GeoJSON and asset hashes |
| `docs/` | Benchmark navigation, model-source manifest, provenance and reproduction notes |
| `studies/visium-hd-colon-wsi-2026-10-19/` | Full-slide source and interpretation notes |
| `pipeline/visium_clustering/` | Three 10x Visium expression-clustering workflows and calibrated wsiTools spot plans |

This repository was renamed from **Bora-extra**; its Git history is preserved.
The segmentation application remains in **tkcaccia/Bora**. Large raw archives,
full-resolution source images, feature caches and model checkpoints stay outside
Git. The website includes only the attributed research previews and derived
vectors needed to inspect these completed results. Dataset rights are retained;
see [attribution](https://tkcaccia.github.io/Bora-Analysis/attribution.html) and
[the data reuse notice](web/DATA_LICENSES.md).

## Validate the published website

The checked-in `site/` is a frozen archive of the earlier completed cohort.
The current full-rerun pipeline builds a new viewer only after all case and
checksum receipts are complete; see [the pipeline guide](pipeline/README.md).
To validate and inspect the existing published archive locally:

```bash
python tools/check_portability.py
python tools/verify_public_viewer.py --site site
python -m http.server 8000 --directory site
```

The site verifier checks every image, vector, internal HTML link, case count,
aggregate IoU, copied-asset hash and continuous polygon/hole area threshold.
The new whole-slide and Visium display images are tied to their verified source
figures in `site/BENCHMARK_SECTIONS_PROVENANCE.json`. Large images are resized
for the browser; whole-slide GeoJSON coordinates are unchanged.
GitHub Actions validates and deploys `site/` to GitHub Pages on changes to main.

For data acquisition and the current workflow, see
[the pipeline guide](pipeline/README.md), [benchmark/README.md](benchmark/README.md),
and [the organization guide](docs/benchmark_organization.md).
