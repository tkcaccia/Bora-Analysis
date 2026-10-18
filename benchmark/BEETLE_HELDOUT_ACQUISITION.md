# Held-out tissue images for a future external validation

The supplied *Segment Anything in Pathology Images with Natural Language*
(arXiv:2506.20988v2), Appendix A, Table A1, was checked against the Bora
inventory. The tissue-relevant entries BCSS, WSSS4LUAD and RINGS were already
acquired; the CoCaHis release in the inventory is a processed training
derivative with unresolved calibration and label semantics. SICAPv2's distributed
JPEG masks have unresolved pixel encoding, while DigestPath2019 and PAIP2019
require provider-controlled access. The remaining entries are principally
nuclear, cell or gland targets. Merely downloading more samples from these
sources would not resolve the manuscript reviewer's concern that the 116-image
cohort was inspected during method development and its coarse inputs came from
expert masks. See [the existing source-by-source audit](ADDITIONAL_DATASETS.md).

For a patient-disjoint multiclass tissue evaluation, the official
[BEETLE development and evaluation release](https://zenodo.org/records/16812932)
is a better fit. The provider's evaluation set comprises 170 ROI PNGs from 54
patients at three clinical centres, each absent from the provider development
set. The provider reports approximately 0.5 µm/pixel for the release. The
pixel-level evaluation masks are held by the
[BEETLE Grand Challenge](https://beetle.grand-challenge.org/); no local quality
score is possible from the downloaded images alone. The original BEETLE
development set is represented in Bora's training sources by seven fields,
but its patient IDs (e.g. patient163 through patient196) are disjoint from the
evaluation patient IDs in the official overview. This does **not** establish
that Virchow2's pretraining data exclude those slides.

The reproducible downloader is `python -m benchmark.download_beetle_evaluation
/mnt/sata_ssd/Bora-benchmark/raw/beetle-evaluation-v1` from the repository
root. It obtains only the 170 original evaluation ROI members by HTTP byte
ranges from the pinned 147-GB image ZIP, without downloading the development or
evaluation WSIs. It verifies the publisher's overview MD5, archive size, every
extracted member's ZIP CRC and size, PNG header, and individual SHA256. Its
`acquisition.json` records the exact scope, patient and WSI IDs, centre, scanner,
dimensions and hashes. The full archive MD5 is identified but cannot be
verified from selected members. Original ROI bytes are kept unchanged; no
annotations, predictions, resized images or scores are created here.

**Completed acquisition, 19 October 2026:** all 170 ROI files were verified on
Chiamaka at `/mnt/sata_ssd/Bora-benchmark/raw/beetle-evaluation-v1/rois`.
They total 1,103,405,377 bytes, measure 1536–2560 pixels along each axis, and
map to 54 distinct patients: 18 each from `biopticka`, `scdc` and
`uwmedicine`. None of the seven patient IDs in the exact Bora BEETLE training
source manifest occurs in the evaluation set. The complete file-by-file
[acquisition inventory](evidence/beetle_evaluation_acquisition.json) is
committed without redistributing image pixels. A separate post-download pass
recomputed SHA256 for all 170 files and matched every manifest entry.

The suitable confirmatory experiment must freeze Bora and its ontology before
using the held-out pixels to select settings. Using development patients only,
fit both a BEETLE-specific fastPLS tile predictor and a Neighbor context
whole-group classifier with fixed architecture and the four provider classes.
Generate coarse
predictions on each evaluation ROI without target annotations; run the fixed
Bora refinement and classify final tissue segments. A separate
Virchow2-clustering arm can test the unsupervised coarse-input use case. Report
`other` as the provider's heterogeneous class, not as a new stroma annotation.
Report the coarse map and final result separately with identical case inclusion,
plus cold end-to-end runtime and hardware. Submit outputs through the official
challenge to obtain the sequestered-mask metrics; summarize by patient and
centre, not by treating 170 correlated ROIs as 170 patients. The images may be
smaller or single-class under their hidden annotations; do not filter them by
unavailable ground truth after scoring. Preserve the published 116-image
results as exploratory development evidence. No BEETLE
evaluation result belongs in the manuscript until this independent execution
and official scoring are complete.
