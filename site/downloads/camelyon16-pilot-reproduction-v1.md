# Reproducing the initial CAMELYON16 technical pilot

**Release state:** the initial five-slide pilot was conducted locally on 15
November 2026. Its compact derived previews and selected technical scores
are now available in [Benchmark 3](../camelyon16/index.html), alongside a
separate three-slide extension. This guide describes the original pilot;
it is not a population estimate or a claim of eight-class whole-slide
accuracy. The original slides and model weights are not distributed with Bora.

## Source, cohort and target

The exact source was the [CAMELYON AWS Open Data release](https://registry.opendata.aws/camelyon/),
listed there as CC0. The older [challenge rules](https://camelyon16.grand-challenge.org/Rules/)
use narrower language. The study used the AWS CC0 source.
The public viewer displays compact derived masks and H&E crops and links to
the original full-resolution slides at AWS; it does not host those slides.
Cite the [CAMELYON16 JAMA challenge
report](https://jamanetwork.com/journals/jama/article-abstract/2665774).

Only `tumor_093` and `normal_093` were used for CAMELYON16-specific fitting.
The three test IDs were selected and frozen before looking at their pixels or
reference masks. Original scanner metadata and the published scanner–centre mapping place
`tumor_093` (fit), `tumor_084` and `tumor_091` (test) at UMCU, and
`normal_093` (fit) and `normal_019` (test) at RUMC. Both centers therefore
occur on both sides of this selected split; it is not center-disjoint.
The five source filenames are distinct, but patient identities remain
unverified, so no patient-disjoint claim is made. See the
[original CAMELYON data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6007545/)
and local source-hash/metadata audit `v169/CENTER_PROVENANCE_AUDIT_VERIFIED.json`.
All five slides use a precomputed H&E-only tissue support. The official
reference codes are 1 (nonmetastatic tissue), 2 (metastasis), and 0 (unscored).
No additional semantic class was introduced on the test slides. Score only
official code 1/2 pixels, including known pixels outside the display support.

| Role | Slide | Original AWS TIFF SHA256 |
| --- | --- | --- |
| CAM-specific fit | `tumor_093` | `aba3ae5a6739e0fb20de5b870ae160a7b32f523c6ffd7b3eafbda3a56cabc041` |
| CAM-specific fit | `normal_093` | `51273349947c2554065d4d606b6f040c98088b674135bff9cf3923d18df2ea05` |
| Frozen test | `tumor_084` | `73c9766616c40aff2bd6315eb44cd39e7b3d6bb7292cb52e25d3872256baee30` |
| Frozen test | `tumor_091` | `29f5ca4caf206a2197a134fd61fd79032cf9f8364495aba5840a842a6967d4a4` |
| Frozen test | `normal_019` | `2d477f93014b9934094d349993c8e3ee2519a968951a408a38dda73d2b5e349c` |

The source checks were recorded before inference. The frozen test-ID selection
is `Bora/migration/bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json`;
the three test source-byte audit is v22 `SOURCE_BYTES_AUDIT_VERIFIED.json`.
Training-slide source hashes above were independently rechecked from retained
AWS TIFF bytes. These paths identify local evidence, not files bundled here.

## Software and inputs

Install the MIT [Bora R release `bora-r-v0.1.4`](https://github.com/tkcaccia/Bora/releases/tag/bora-r-v0.1.4)
with its matching Python backend 0.3.4. The package's [one-result guide](https://github.com/tkcaccia/Bora/blob/bora-r-v0.1.4/r/REPRODUCE_ONE_RESULT.md)
gives exact installation and CPU/GPU requirements. Virchow2, U-Net,
DeepLabV3+, MedSAM and VISTA-PATH checkpoints are separately obtained under
their upstream terms. The Virchow2 checkpoint SHA256 was
`8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b`;
its config SHA256 was
`7db445b996bb165e88fe70e826c2ebb530539a2b1d136aa16eeb847df5f1e3db`.

Calibrate each source to the recorded 0.5 µm/pixel RGB grid. Every slide had a
precomputed, independently reviewed H&E-only tissue mask; the
[preprocessing guide](../preprocessing.html) explains its source-bound TIFF
and GeoJSON procedure. The **two training-slide caches** were historically
encoded over the full field, including glass; their reviewed support was then
used to restrict training tokens and predictions. The **three test-slide
caches** used sparse, support-admitted extraction at factors 1/2/4; the v41
`TEST_FEATURES_FROZEN.json` binds each source, support, tile count and
1,280-channel scale receipt. Do not describe all five caches as masked or all
five as full-field. A new extraction using
`Bora.preparation(tissue_masks = ...)` can skip glass tiles in the training
pair, but it would have different cache provenance and must not be called
byte-identical to this pilot. The study froze **no additional stain normalization**:
all slides used the same calibrated RGB path, and this choice was made before
opening the three test references. Retain all 1,280 float16 Virchow2 channels
at factors 1, 2 and 4 with source-to-token registration and SHA256 receipts.
The training feature contract is in local v61
`source-proofs-v1/PRECOMPUTED_FEATURES_FROZEN.json`; the test feature freeze
precedes scoring. Run one Chiamaka GPU bundle at a time, preserve a 100 GiB
free-space reserve and a 200 GiB new-campaign peak cap, and verify terminal
owned-process/GPU cleanup before the next bundle.

## Code, model roles and order of operations

The [code-only ZIP](camelyon16-pilot-source-v1.zip), SHA256
`898f1be66ca7a808a00025c511a5d5f057d4426a2a968a15d44610e562151f48`,
contains 115 Python source snapshots and `MANIFEST.json`. Each snapshot is
byte-identical to the versioned local source. It contains no images, masks,
features, checkpoints or scores. These scripts preserve Chiamaka-specific
paths, so adapt paths and verify each frozen SHA before any independent rerun;
do not treat the archive as a one-command installer. The general Bora API and
tissue-support rasterizer live in the released package.
The same 115 files are [browsable individually](camelyon16-source/index.html)
with their SHA256 values in a local source index.

1. Freeze slide identities, original-source checksums, calibrated H&E grids,
   reviewed support, complete Virchow2 caches and the official binary target.
   The snapshot includes the acquisition, calibration, support and feature
   scripts in v19–v66. Use their source and exact-pixel receipts as acceptance
   checks; do not silently relax the strict less-than-10% KODAMA tile rule.
2. Freeze the training-only settings in v67 `METHOD_SETTINGS_FROZEN.json`.
   The CAM-pair Bora ridge selected **32** channels from the full 1,280-channel
   cache; its independent fit audit is v70. The earlier eight-class Bora
   checkpoint has 203 channels and is a different model.
3. Train the binary U-Net and DeepLabV3+ heads on the same two CAM slides using
   v81 `train_network.py`. Their backbones inherit weights fitted earlier on
   397 non-CAM ROIs; the binary heads were initialized for this study. The
   independent completed-fit audits are v88 (U-Net) and v90 (DeepLabV3+).
4. Generate source-only test predictions before opening expert masks. Bora
   projection uses v91/v97/v102; bounded network bands use v103/v106 and
   v114 assembly. MedSAM uses boxes from the CAM-adapted U-Net, frozen in v120,
   then v122/v125/v129. VISTA-PATH uses its released generic “tumor” prompt,
   frozen in v132, then v136/v141. MedSAM and VISTA-PATH are prompted
   descriptive comparators, not identically trained metastasis classifiers.
5. Freeze v144 `SCORING_FROZEN.json` and `score_test_case.py` **before** opening
   the three test references. The scorer source SHA256 is
   `02c7b3639cfeda13fdc11fcc48a6ad1480b4c84aa04a7ed5fec0d8e4a60b7972`.
   Align the official references with v143, run the frozen score cases with
   v151, and audit exact full-pixel arithmetic with v152. Score native and
   common-support-clipped predictions separately at 4 µm/pixel; a 5 µm
   boundary tolerance and lesion-match IoU 0.1 were frozen in v144.

The anonymous KODAMA route admitted 40,411 five-slide tiles, above the
package's 10,000-tile cap. It was recorded as capacity noncompletion, not
silently subsampled. It has no validated CAM metastasis naming head, so no
semantic tumor IoU was fabricated.

## Expected result and verification

The [historical pilot score aggregate](../camelyon16/aggregate.json) has SHA256
`41aae3a5fcb4904a47ce75a106f0b8f9b513e52a8af461f16ae6289ef2e67b27`.
Its `online_publication: false` field records the local state when this pilot
receipt was frozen; the separate [six-slide public aggregate](../camelyon16/aggregate-six.json)
is the current Benchmark 3 summary.
It corrects one descriptive typo in an earlier aggregate: the CAM Bora model
has 32 selected channels. The underlying numerical scores are unchanged.
All five rendered methods completed all three selected test slides (15/15
method-slide outputs). Shared-support mean metastasis IoU across the two
positive slides was 0.328 Bora, 0.072 U-Net, 0.100 DeepLabV3+, 0.123
U-Net-box-prompted MedSAM and 0.294 generic-tumor-prompt VISTA-PATH. On the
benign slide, positive-class IoU is undefined; false-positive tissue area was
0.0205, 0.0017, 0.0078, 0.0022 and 0.5744 mm² in that order. Per-slide
precision, recall, Dice, boundary and lesion results are in the aggregate and
local slide pages. These numbers are a technical pilot and do not rank model
architectures.

Verify downloaded code and numeric metadata before use:

```bash
sha256sum camelyon16-pilot-source-v1.zip ../camelyon16/aggregate.json
python3 -m zipfile -t camelyon16-pilot-source-v1.zip
```

The two expected hashes are listed above. The independent local v162 audit
checked every ZIP member against its historical source and all site links.
The historical manuscript and portrait supplement were rendered and visually
checked as local v58 drafts. The public Benchmark 3 viewer includes the
later six-test-slide aggregate, which is separate from this pilot guide.
