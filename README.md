# Bora public benchmark viewer

This repository contains the verified, public-data-only static website.
It includes a historical controlled Benchmark 1 gallery, an 87-image
development validation gallery, 13 exploratory public-source Bora H&E
whole-slide pages in Benchmark 2, and historical Visium gene-cluster
examples in Benchmark 3. The Benchmark 1 validation images were inspected
during development and do not form an untouched confirmatory test.
Tissue names on whole slides are provisional.

The [Benchmark 1 development gallery](https://tkcaccia.github.io/Bora-Analysis/benchmark1-comparators/)
uses eight broad tissue labels selected from the 397 retained training images.
An eight-class supervised Virchow2 head was fitted on training only, then
applied to the 87 development validation images. Its 256-channel regularized
multinomial head and ROI contour rule yielded mean expert-present case macro
IoU 0.694; the mean across the four multiclass datasets was 0.603. The older
24-class head mapped to the same broad labels was essentially tied (0.692
overall; 0.604 across the four multiclass datasets). The score task changed,
so these values must not be compared directly with prior 24-class scores.
The gallery shows eight new black-and-white supervised response maps per case.
The former KODAMA–Bora + fastPLS output and four external predictions were
mapped and rescored under the broad ontology without refitting those models.
The original Benchmark 1 KODAMA spatial group masks are absent from the
curated local cache; the gallery leaves that spatial panel unfilled rather
than reconstructing it from a lossy WebP. The exact frozen pixel confusion
counts still support its broad-label score. These are development results,
and the 19 single-class PEsO cases are not a multiclass test.

The [13 public Benchmark 2 whole-slide pages](https://tkcaccia.github.io/Bora-Analysis/visium-hd/)
show the same H&E frame with KODAMA–Bora, supervised Virchow2 Bora,
U-Net, DeepLabV3+, MedSAM and VISTA-PATH at full overview and nine matched
zoom views. The current supervised head was fitted on the frozen 397 Benchmark 1
training images and expert masks after a training-only reduction from 24 labels
to eight broad tissue classes. Five sparsely represented tissue categories are
unscored. KODAMA–Bora geometry and the four external model predictions were
mapped into these broader labels; those models were not retrained. All six
displayed overlays use the final
Bora tissue-group support to handle background consistently. The supervised
head also uses a training-selected 2% whole-slide class-presence rule to
remove trace predictions. Its final class mask is smoothed over 6 µm,
repaired to remove regions and enclosed holes below 2,000 µm², and exported
as a full-resolution TIFF and native-coordinate GeoJSON. Each page's overview
and nine zooms are rendered from that one final mask. The full-resolution
source features and external predictions remain unchanged.
These exploratory views have no
independent whole-slide expert masks or accuracy ranking.

The [supervised full-slide method and audit](docs/supervised-bora-fullslide-final-2026-11-12.md)
describe the training-only class rule, physical contour and geometry cleanup,
and native GeoJSON output.

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
