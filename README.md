# Bora public benchmark viewer

This repository contains the verified, public-data-only static website.
It includes a historical controlled Benchmark 1 gallery, an 87-image
development validation gallery, 13 exploratory public-source Bora H&E
whole-slide pages in Benchmark 2, and historical Visium gene-cluster
examples in Benchmark 3. The Benchmark 1 validation images were inspected
during development and do not form an untouched confirmatory test.
Tissue names on whole slides are provisional.

The [Benchmark 1 development gallery](https://tkcaccia.github.io/Bora-Analysis/benchmark1-comparators/)
now displays one training-selected supervised Virchow2 model for all 87
retained images: a 444-channel joint classifier blended with the earlier
classwise PLS head. Mean expert-present case macro IoU is 0.592. This is
development evidence, and case 050 still has poor muscle recognition. The
Benchmark 2 whole-slide pages retain the earlier frozen PLS-only model;
those slide overlays were not rerun with the new Benchmark 1 blend.

The [13 public Benchmark 2 whole-slide pages](https://tkcaccia.github.io/Bora-Analysis/visium-hd/)
show the same H&E frame with KODAMA–Bora, supervised Virchow2–PLS Bora,
U-Net, DeepLabV3+, MedSAM and VISTA-PATH at full overview and nine matched
zoom views. The supervised model was fitted on the frozen 397 Benchmark 1
training images and expert masks. All six displayed overlays use the final
Bora tissue-group support to handle background consistently; this display
step does not alter raw predictions. These exploratory views have no
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
