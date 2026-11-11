# Bora public benchmark viewer

This repository contains the verified, public-data-only static website.
It includes a historical controlled Benchmark 1 gallery, an 87-image
development validation gallery, 13 exploratory public-source Bora H&E
whole-slide pages in Benchmark 2, and historical Visium gene-cluster
examples in Benchmark 3. The Benchmark 1 validation images were inspected
during development and do not form an untouched confirmatory test.
Tissue names on whole slides are provisional.

The [Benchmark 1 development gallery](https://tkcaccia.github.io/Bora-Analysis/benchmark1-comparators/)
shows the canonical full-channel KODAMA–Bora + fastPLS route and one
training-selected supervised Virchow2 route for all 87 retained images. The
supervised route blends a 444-channel joint classifier with a classwise PLS
head, then restricts its final decision to classes scorable in the known
source dataset. Two source-group-disjoint training splits selected this gate:
four-multiclass-dataset mean case macro IoU was 0.590, versus 0.580 without
the gate. The final ROI mask receives 1.5 µm contour smoothing and 400 µm²
small-region repair. On the 87 development images, mean expert-present case
macro IoU is 0.604, versus 0.593 for the preceding rule. Across the 68 images
from four multiclass datasets, it is 0.494 versus 0.486. Nineteen PEsO images
have one scorable class and should not be interpreted as a multiclass test.
Case 050 still has poor muscle recognition. The six colored method panels and
all 24 linked black-and-white response maps use one Bora H&E tissue/background
support per image; the response maps show scores before the final ontology
gate. The other methods' predictions and scores are unchanged. Benchmark 2
whole-slide pages retain the earlier frozen PLS-only model; those overlays
were not rerun with the revised Benchmark 1 rule.

The [13 public Benchmark 2 whole-slide pages](https://tkcaccia.github.io/Bora-Analysis/visium-hd/)
show the same H&E frame with KODAMA–Bora, supervised Virchow2–PLS Bora,
U-Net, DeepLabV3+, MedSAM and VISTA-PATH at full overview and nine matched
zoom views. The supervised model was fitted on the frozen 397 Benchmark 1
training images and expert masks. All six displayed overlays use the final
Bora tissue-group support to handle background consistently. Supervised Bora
web views additionally receive 6 µm physical contour smoothing with a halo;
the frozen score grids and other methods' predictions remain unchanged.
These exploratory views have no
independent whole-slide expert masks or accuracy ranking.

The [benchmark method implementations](methods/README.md) provide the
Bora-specific training, prompting, Virchow2–PLS projection, and full-slide
inference code and document the exact settings and external model sources.
This repository was previously named Bora-extra.

The [Benchmark 2 full-slide comparator gallery](https://tkcaccia.github.io/Bora-Analysis/visium-hd/comparators/)
shows 11 verified public-source H&E slides with class-aware
U-Net, DeepLabV3+, MedSAM and VISTA-PATH outputs. Those predictions cover
the complete calibrated slides; the website shows downsampled previews with
one Bora-derived tissue/background support shared by all four methods. Eight
slides use final Bora tissue-group geometry; three TCGA slides without final
Bora geometry use the same H&E-derived Bora support rule. Raw full-resolution
predictions are unchanged.
No independent whole-slide expert masks are available, so the gallery
does not report accuracy scores or a method ranking. All 12 slides in the
separate local comparator campaign have checksum-verified full-resolution
outputs; the user-supplied PM016 slide remains local and unpublished. See the
[comparator methods and publication record](docs/benchmark2-full-slide-comparators-2026-11-10.md).

No user-supplied Brendon, PM, F-series or Scanned Image slide is included.
The publication workflow verifies exact file hashes, internal links, the
public slide allowlist, and forbidden-content markers before uploading.

Original dataset providers retain their image and annotation rights; see
`site/attribution.html` and `site/DATA_LICENSES.md`.
