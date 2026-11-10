# Bora public benchmark viewer

This repository contains the verified, public-data-only static website.
It includes a historical controlled Benchmark 1 gallery, an 87-image
development validation gallery, 13 exploratory public-source Bora H&E
whole-slide pages in Benchmark 2, and historical Visium gene-cluster
examples in Benchmark 3. The Benchmark 1 validation images were inspected
during development and do not form an untouched confirmatory test.
Tissue names on whole slides are provisional.

The [Benchmark 2 full-slide comparator gallery](https://tkcaccia.github.io/Bora-Analysis/visium-hd/comparators/)
currently shows nine verified public-source H&E slides with class-aware
U-Net, DeepLabV3+, MedSAM and VISTA-PATH outputs. Those predictions cover
the complete calibrated slides; the website shows downsampled previews.
No independent whole-slide expert masks are available, so the gallery
does not report accuracy scores or a method ranking. The separate 12-slide
local comparator campaign is still in progress. See the
[comparator methods and publication record](docs/benchmark2-full-slide-comparators-2026-11-10.md).

No user-supplied Brendon, PM, F-series or Scanned Image slide is included.
The publication workflow verifies exact file hashes, internal links, the
public slide allowlist, and forbidden-content markers before uploading.

Original dataset providers retain their image and annotation rights; see
`site/attribution.html` and `site/DATA_LICENSES.md`.
