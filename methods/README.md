# Bora benchmark method implementations

This directory contains the Bora-specific training, prompt, projection, and
whole-slide inference code behind the public comparisons. The repository was
previously named **Bora-extra**. The core Bora implementation is in
[tkcaccia/Bora](https://github.com/tkcaccia/Bora). These scripts are research
sources, not a packaged one-command reproduction: the provider images,
expert masks, Virchow2 weights, comparator checkpoints, and local SHA256
manifests are supplied separately under their respective access terms.

| Displayed method | Input and training | Public implementation |
| --- | --- | --- |
| KODAMA–Bora | Calibrated H&E, reviewed tissue support, Virchow2 features; anonymous image-only tile groups | [Bora](https://github.com/tkcaccia/Bora), [KODAMA](https://github.com/tkcaccia/KODAMA), and the [whole-slide method record](../docs/benchmark2-full-slide-comparators-2026-11-10.md) |
| Eight broad-label reanalysis | Training-only ontology; new 256-channel supervised Virchow2 head; exact remapping of existing Bora and comparator predictions | [`reduced-labels/`](reduced-labels/) contains the executed training, validation, support-audit and rescore sources plus frozen aggregate ontology receipts |
| Selected eight-class ridge head | Full 1,280-channel Virchow2 input; 203 training-selected channels; 397 expert-masked training images | [`benchmark1/ridge/`](benchmark1/ridge/) contains the executed search, fit, 87-image development evaluation and response-map sources |
| Supervised Virchow2 (Benchmark 1) | 1,280-channel Virchow2 token features; 397 expert-masked training images | [`classwise_joint_search.py`](benchmark1/classwise_joint_search.py), [`classwise_joint_fit.py`](benchmark1/classwise_joint_fit.py), [`blend_search.py`](benchmark1/blend_search.py), [`ontology_gate.py`](benchmark1/ontology_gate.py), [`blend_validate.py`](benchmark1/blend_validate.py), [`supervised_contour.py`](benchmark1/supervised_contour.py) |
| Supervised Bora (Benchmark 2 ridge head) | Full 1,280-channel input cache; 203 training-selected channels; 397 expert-masked training images | [`supervised_ridge/`](benchmark2/supervised_ridge/) contains the executed training-only presence selection, whole-slide projection, contour, geometry, vector and rendering code; [`supervised_final/`](benchmark2/supervised_final/) preserves the earlier logistic route |
| Historical supervised Virchow2–PLS | The same 1,280-channel input and 397 training images; earlier frozen head | [`search.py`](benchmark1/search.py), [`final_fit.py`](benchmark1/final_fit.py), [`project_supervised.py`](benchmark2/project_supervised.py) |
| Class-aware U-Net | RGB H&E and expert masks on the same 397 training images | [`train_unet_balanced.py`](benchmark1/train_unet_balanced.py), [`run_four_methods.py`](benchmark2/run_four_methods.py) |
| DeepLabV3+ | RGB H&E and expert masks on the same 397 training images | [`train_deeplabv3plus.py`](benchmark1/train_deeplabv3plus.py), [`run_four_methods.py`](benchmark2/run_four_methods.py) |
| U-Net-guided MedSAM | RGB H&E and U-Net-derived boxes; external pretrained MedSAM ViT-B weights | [`prepare_medsam_prompts.py`](benchmark1/prepare_medsam_prompts.py), [`run_medsam_guided.py`](benchmark1/run_medsam_guided.py), [`run_four_methods.py`](benchmark2/run_four_methods.py); [upstream MedSAM](https://github.com/bowang-lab/MedSAM) |
| VISTA-PATH | RGB H&E and 13 text class names; external pretrained model | [`run_four_methods.py`](benchmark2/run_four_methods.py) and the settings below; [upstream VISTA-PATH](https://github.com/zhihuanglab/VISTA-PATH) |

The common visualization step is implemented in
[`shared_display.py`](benchmark2/shared_display.py). It takes an already
aligned final Bora support raster and the existing 24-role palette.
The later eight-role release maps the original class outputs to the frozen
broad ontology before display. Its newly fitted supervised head is distinct
from the older Benchmark 1 blend and Benchmark 2 PLS projection below.
KODAMA geometry and external models were not refitted under the eight roles.

## Benchmark 1 model fitting

U-Net is the four-down-block, base-width-24 model in the source, with 24
logits and random initialization. DeepLabV3+ uses a randomly initialized
ResNet-50 encoder at output stride 16, ASPP rates 6/12/18, a 48-channel
low-level projection, and 24 logits. Both use 512-pixel patches, four sampled
patches per training image per epoch, training-only class-aware sampling,
training-pixel-frequency square-root weighted cross-entropy, AdamW at
2e-4 with 1e-4 weight decay, and source-group-disjoint inner validation for
epoch choice. They then refit on all 397 training images. Batch size is four
for U-Net and two for DeepLabV3+. Neither uses Virchow2 or black-and-white
response maps as input. The retained development gallery has 87 images;
those expert masks are not opened by the training scripts.

The earlier 24-class Benchmark 1 supervised mask uses the classwise union of 32
standardized contrast channels per class (444 unique original channels), a
joint 24-class multinomial logistic head, and a 25% blend with the older
64-channel-per-class, eight-component PLS logistic head. The 75%/25% blend
was chosen on two source-group-disjoint splits of the 397 training images;
both heads were fitted on all 397 images before the 87 development images
were scored. The final ROI mask uses Bora's 1.5 µm one-hot contour smoothing
and 400 µm² small-region repair. The selected mask has mean expert-present case macro IoU 0.593
on that gallery. Those images were inspected during development, so this is
not an untouched test estimate. The 19 one-class PEsO images raise the
all-image mean; the other 68 images have mean 0.486. Case 050 tumor IoU rises
from 0.307 to 0.822, but muscle IoU remains 0. The code and versioned
receipts preserve this limitation.

The Benchmark 1 eight-class development gallery and Benchmark 2 public whole-slide pages use the training-selected 203-channel classwise ridge head, fitted on the same 397 training images.
KODAMA–Bora's anonymous groups and earlier fastPLS tissue names remain a
separate route. A source-group-disjoint training search selected a 2%
whole-slide class-presence rule, so trace predicted classes are reassigned
using the remaining supervised scores. The final supervised mask is projected
over the complete calibrated slide, smoothed over 6 µm, repaired at the
2,000 µm² physical-area cutoff, and exported as a full-resolution raster and
native-coordinate GeoJSON. The ten public views are crops of that one final
mask. There are no independent whole-slide expert masks or accuracy scores.

The 24-role harmonization used during fitting is recorded in the project
[`tissue_label_harmonization.json`](benchmark1/tissue_label_harmonization.json).
The retained source mappings are unchanged from the fitted model's mapping;
the public file omits a withdrawn source entry and therefore has a different
whole-file SHA256 from the historical training receipt. A new run needs a
fresh manifest and receipt bound to this current file.
Provider images, annotations, weight tensors, and withdrawn datasets
are not redistributed here. The source scripts retain project-specific
manifest and output paths, which must be bound to the user's own authorized
inputs before a new run. No run is launched by this publication.

## Benchmark 1 MedSAM protocol

The retained development comparison used an **earlier** U-Net prediction to
derive boxes for tumor, stroma, and necrosis. Eight-connected components of
at least 100 pixels were sorted by area; up to six per role received an 8%
box pad relative to the larger component side. MedSAM ViT-B inferred each
box and merged its binary outputs per role at probability greater than 0.5.
These are prompted boundary masks, not a 24-class MedSAM classifier. The
displayed class-aware U-Net is a later balanced revision, so its panel is not
the exact prompt source for this Benchmark 1 run.

## Benchmark 2 whole-slide protocol

[`stage_slide.py`](benchmark2/stage_slide.py) divides a calibrated slide into
1,024-pixel H&E fields with a 64-pixel context margin around 896-pixel cores.
The frozen whole-slide runner processes each field as follows:

* U-Net and DeepLabV3+ average 24-class logits from nine overlapping
  512-pixel crops (starts 0, 384, 512 along each axis), then take argmax.
* MedSAM uses the **same field's class-aware U-Net prediction** for up to
  three largest components per tumor, stroma, and necrosis role, each at
  least 256 pixels. Its output is limited to these three roles. This differs
  from the Benchmark 1 prompt rule above.
* VISTA-PATH uses its released model without box prompts. The invocation
  sets `bbx_random=1`, `num_queries=20`, model image size 512, crop size
  1024, overlap 128, foreground threshold 0.05, batch size one, and the
  exact ordered class names below. Its foreground gate is an inference
  setting; the final displayed background is handled separately.

The later five-public-slide recovery used the same four inference methods
with a locally staged complete PLIP tokenizer/model bundle. Its exact
[runner](benchmark2/public_five_recovery/run_gpu.py) and
[full-slide stitch audit](benchmark2/public_five_recovery/audit_stitch.py)
are included. The licensed VISTA adapter and PLIP model files must be
obtained and staged locally; only their paths and SHA256 identities changed
in that recovery, not its class vocabulary or model architecture.

The ordered VISTA names are `tumor`, `stroma`,
`lymphocytic_infiltrate`, `necrosis_or_debris`, `blood`,
`metaplasia_NOS`, `fat`, `plasma_cells`, `other_immune_infiltrate`,
`lymphatics`, `blood_vessel`, `normal_acinus_or_duct`, `dcis`.
Their output IDs 1–13 map to the shared role IDs
`1, 3, 12, 5, 8, 15, 11, 14, 13, 10, 9, 6, 2` respectively; zero remains
unclassified. This map is a display/score role correspondence, not a claim
that the underlying ontologies are identical. The four complete calibrated
slide label TIFFs are assembled from the 896-pixel cores and SHA256-audited.

On the 13 integrated public whole-slide pages, the union of the final Bora
tissue-group polygons clips all six **displayed overlays** to the same tissue
area. This does not alter saved raw predictions. The pages show one overview
and nine aligned zoom views. They have no independent whole-slide expert
masks, so no whole-slide accuracy or method ranking is reported.

## Source and model provenance

The public U-Net and DeepLab scripts reproduce the fitted model and training
logic, with only the obsolete evaluation-membership check changed from 106
to the retained 87 images. Their original source SHA256 values are
`9ccf84a28152a3151ce4bc64ad150943543c02a29b9448d316df2d1eb9012dce`
and `ba20a83c49b0034c361bbdf5eeb2cddc53d32932e0be6971ac6054666c871bfc`.
The fitted checkpoint SHA256 values are
`b3c304486ec2cb1c4d7b3aba49001029d2df961aa30159abd9ce1ce894691b9e`
and `3d57ae554643354e2b856127191623260df0e4d82a2d68100b129bbd66b2510c`.
The earlier Benchmark 2 supervised PLS model SHA256 is
`52557adeecf45508de760c067d62cc20ce0b2e9cbeac0a5001ef84a05b8e8dc0`.
The Benchmark 1 joint model SHA256 is
`11797d16043952a859931855c451b345acd988d2407c936681c5ee9335126335`;
its PLS component uses the earlier model SHA256 above.
These hashes identify local research outputs; the checkpoint bytes are not
included in this repository.

The VISTA-PATH source and checkpoint remain with their upstream provider.
Its academic license restricts redistribution, so this repository provides
our invocation settings and class map, and links to the upstream
implementation rather than copying its code or weights. MedSAM code and
weights are likewise obtained from its upstream project; our prompt
construction and inference calls are provided here. The whole-slide runner
refers to the local VISTA adapter path used in the run. Reproduction requires
a locally authorized upstream checkout, model files, the recorded offline
PLIP/SAM/Mask2Former dependencies, and remapping those project paths.

The implementations here explain the reported results; a fresh checkout
alone cannot reconstruct provider data or restricted weights. For the exact
public result identities and limitations, see the
[whole-slide publication record](../docs/benchmark2-full-slide-comparators-2026-11-10.md).
