# Size-aware tissue segmentation resource benchmark

This protocol compares complete pixel-level prediction on the same H&E inputs.
The primary endpoint is **cold process-start to verified native-resolution mask**.
For methods that also export GeoJSON, mask and vector completion are timed
separately and the complete export time is reported. Model training is a
separate cost; pretrained models and Bora's small fastPLS startup fit are not
treated as equivalent training regimes.

## Methods and input contracts

| Method | Primary input | Analysis |
| --- | --- | --- |
| Bora 0.2.1 | RGB H&E, fixed coarse tile labels, calibrated Virchow2 features, frozen fastPLS training set | Cold Virchow2 extraction, evidence construction, Bora segmentation, region classification and export are timed together. |
| VISTA-PATH | RGB H&E and prespecified tissue-class names | Run its released WSI checkpoint **without boxes** first. A second, explicitly conditional experiment may provide boxes derived only from the same coarse tile map supplied to Bora. |
| TIAToolbox BCSS semantic segmenter | RGB H&E | Use its released BCSS checkpoint on compatible breast classes; report other organs as out of scope. |
| 2D nnU-Net | RGB H&E and fitted dataset-specific weights | Train only on permitted training slides, report training resources separately, and use the same WSI tiler contract at inference. |
| Lv et al. distilled student | RGB H&E | Run only after the exact public checkpoint and inference recipe are verified. Do not substitute a differently trained model. |

No model receives expert test labels, boxes or masks at inference. Annotations
score only covered pixels. Predicted masks cover the whole tissue area. If a
checkpoint's training set overlaps the tested patient/slide/dataset, mark the
comparison as *in-domain pretrained* and do not present it as independent.

## Image-size design

Freeze a manifest before any method is run. Include at least 20 eligible,
different-slide ROIs per compatible dataset where available; prespecify
representative fields near 1, 4, 16 and 64 native megapixels, plus at least
one complete WSI from each feasible tissue type. Crop size and selected tissue
area must be frozen without reviewing method results. Report native width,
height, micrometres per pixel in both axes, full image pixels, tissue pixels,
tissue area in mm², and tile count. Distinct crops of one slide remain one
statistical unit for confidence intervals. Full-slide runs are not inferred
from crops. Report methods' supported class sets and abstentions explicitly.

## Measurement

Run methods sequentially on the same configured GPU workstation after the existing GPU
queue has fully completed and cleanup/admission are verified. UCT HPC GPU is
paused. Fix software revisions, model hashes, image hashes, physical spacing,
output format and thread counts. Use a fresh process and empty output path for
each cold run. Time image decode/resampling, model/checkpoint load, feature
inference, any startup fitting, dense prediction, stitching, native mask write
and vector conversion. Preserve stage times as well as total. Run three repeats
per case and method only when each prior case completed cleanly. The primary
time is the median complete-run wall clock; report maximum peak memory across
repeats. If a run fails or cleanup is uncertain, stop the bundle and do not
substitute a shorter crop.

The runner `size_resource_benchmark.py` samples **owned process-tree** host
resident memory and owned GPU memory at no more than 1 Hz during GPU work. It
records size and timing per case; `seconds_per_tissue_megapixel` is descriptive
and does not replace absolute time. Its sampled peak is a lower bound on any
shorter memory spike. Retain device model, GPU capacity, CPU, RAM, storage and
I/O path. Exclude viewer rendering from model runtime, but include native mask
and GeoJSON export if that output is compared. Record GeoJSON bytes and contour
segments separately; a compact label raster may be a more appropriate WSI
exchange artifact.

## Manuscript display

Place a **log-scale scatterplot** of cold total seconds versus native
megapixels, with symbols for ROI versus WSI and colors for methods. A second
panel shows peak GPU and host memory versus native megapixels; annotate tissue
area and micrometres per pixel in the data table. A compact manuscript table
reports each method's median/p95 ROI seconds, full-WSI seconds, maximum host
RSS, maximum owned GPU memory and successful completion count. Report
per-stage times and every case in the repository. Do not fit or extrapolate a
scaling law from fewer than four distinct sizes for a method. Preserve all
failures and unsupported label mappings in denominators.

## Current evidence

The existing Bora four-ROI pilot used **cached** Virchow2 features: median
post-import wall time was 1.666 s and median peak process RSS was 491 MB. The
116-image 2.071-s mean is another cached-feature measure. Neither is a cold
end-to-end observation. The prior colorectal WSI vector was 203.0 MB with
11,107,958 segments; whole-slide peak memory was not recorded. These values
must not be placed in a cross-method cold-runtime ranking.
