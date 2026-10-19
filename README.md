# Bora-Analysis

Benchmark acquisition, scoring, research workflows, figures and manuscript
builders for [Bora](https://github.com/tkcaccia/Bora).

**[Open the public benchmark viewer](https://tkcaccia.github.io/Bora-Analysis/)**

The main viewer shows **Neighbor context**, the selected tissue-group
classifier, on **116 calibrated multiclass images**: BCSS 42, TIGER 26,
GCSS 24 and IGNITE 24. Each image includes the original H&E, expert mask and
legend, anonymous expert-derived input tile mask, fixed RGB9+19 conversion,
Bora-AF groups, final classified mask, scores, per-axis pixel resolution and
interactive simplified GeoJSON. PUMA is excluded from evaluation.

The original split labels are preserved (78 test, 38 validation). These are
**exploratory benchmark results**, not independent final validation. The coarse input is derived from expert annotations;
this experiment does not establish annotation-free segmentation.

The [classifier comparison](https://tkcaccia.github.io/Bora-Analysis/comparison/)
shows named methods on the same images. The current Bora application uses
Neighbor context.

The [full-slide Visium HD colon viewer](https://tkcaccia.github.io/Bora-Analysis/visium-hd-colon/)
shows Bora on a 75,250 × 48,740-pixel H&E image with the supplied partial
expert annotation. The [study code and audits](studies/visium-hd-colon-wsi-2026-10-19/)
document physical calibration, 228 bounded processing windows, native mask and
GeoJSON export, and the limits of the externally trained tissue classifier.

To train a tissue classifier for new images, use the
[Bora training and inference guide](https://github.com/tkcaccia/Bora/blob/codex/virchow2-twenty-ideas/docs/WORKFLOW.md).
The [manuscript model data guide](docs/MANUSCRIPT_MODEL_DATA.md) lists the
exact 349 annotated source fields, provider download routes, native crop
coordinates and checksums used for the reported 967 training groups.
The [manuscript reanalysis note](docs/CMPB_REANALYSIS.md) provides the
descriptive cohort, classifier-control, boundary and vector-fidelity evidence
with commands and limitations.

## Repository organization

| Location | Contents |
| --- | --- |
| `benchmark/` | Dataset acquisition, paired-image preparation, calibration/annotation metadata, scoring and audit code |
| `history/bora-development/` | Preserved experimental algorithms, research runners, tests, legacy runtime and historical reports |
| `manuscript/` | Figure and document generation code |
| `web/` | Human-editable static viewer HTML, JavaScript, CSS and data attribution |
| `tools/` | CPU-only static-site packaging and exhaustive publication checks |
| `site/` | Published viewer assets, per-image scores, metadata, GeoJSON and asset hashes |
| `docs/` | Benchmark navigation, model-source manifest, provenance and reproduction notes |
| `studies/visium-hd-colon-wsi-2026-10-19/` | Full-slide calibrated processing code, reference-restricted scores and all-window geometry audit |

This repository was renamed from **Bora-extra**; its Git history is preserved.
The segmentation application remains in **tkcaccia/Bora**. Large raw archives,
full-resolution source images, feature caches and model checkpoints stay outside
Git. The website includes only the attributed research previews and derived
vectors needed to inspect these completed results. Dataset rights are retained;
see [attribution](https://tkcaccia.github.io/Bora-Analysis/attribution.html) and
[the data reuse notice](web/DATA_LICENSES.md).

## Rebuild the website from completed results

Python 3.10+ and the standard library are sufficient; no GPU, scientific package,
training or model inference is used by these commands.

```bash
python tools/build_public_viewer.py --source /path/to/completed/viewer
python tools/verify_public_viewer.py --site site
python -m http.server 8000 --directory site
```

The source viewer's relative shared assets and vector cache must be present.
The builder copies exact images and GeoJSONs, preserves native coordinates,
deduplicates vectors, and records source SHA256s in `site/BUILD_PROVENANCE.json`.
The verifier checks every image, vector, internal HTML link, case count,
aggregate IoU, copied-asset hash and continuous polygon/hole area threshold.
GitHub Actions validates and deploys `site/` to GitHub Pages on changes to main.

For data acquisition and earlier experiments, see [benchmark/README.md](benchmark/README.md)
and [the organization guide](docs/benchmark_organization.md). Archived runner
commands require their original Bora implementation and environment. They do
not authorize restarting experiments or paused HPC work.
