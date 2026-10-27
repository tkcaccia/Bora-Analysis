# Prospective external tissue evaluation

This protocol describes a future BEETLE experiment. It contains no hidden-mask
scores and does not authorize GPU/HPC execution. The source is the official
[BEETLE v1 release](https://zenodo.org/records/16812932); the 170 evaluation
ROIs from 54 patients at three centres have been acquired and byte-verified as
described in [the acquisition record](../benchmark/BEETLE_HELDOUT_ACQUISITION.md).
Their reference masks are sequestered by the challenge.

The separate 27 October 2026 capped BEETLE **development** subset contains
20 annotated fields from 12 patients at three clinical centers. Its prepared
ROIs and source metadata are described in the
[full-rerun protocol](../pipeline/full_benchmark_protocol.json). It is a
secondary development analysis and is not the confirmatory 54-patient
evaluation proposed here. No development-subset result may be presented as a
Grand Challenge held-out score.

## Freeze before evaluation

1. Use only the official BEETLE development patients for model fitting and
   setting selection. Check patient and WSI identifiers against the evaluation
   manifest. Record image, annotation, Virchow2 checkpoint, Bora source,
   fastPLS core, ontology and model hashes before generating evaluation masks.
2. Train a four-role tissue model corresponding to the provider's invasive
   epithelium, non-invasive epithelium, necrosis and heterogeneous `other`.
   Preserve `other` as the provider category; do not relabel it as stroma.
3. Define two annotation-free evaluation input routes: Virchow2 tile-feature
   clustering and a fastPLS tile predictor trained on development patients.
   Freeze each route's tile size in micrometres, feature revision, calibration,
   cluster count or predictor settings, and treatment of tissue background.
   Generate both coarse maps before any challenge feedback is inspected.
4. Apply the same fixed Bora refinement, whole-group Neighbor context model,
   final vector cleanup and ontology to both routes. Save coarse, pre-vector
   and final native masks, GeoJSON, timing, peak memory and provenance for every
   ROI. Do not exclude an ROI after seeing an unavailable reference. Record
   technical failures and single-class predictions as outcomes.

## Endpoints and statistical unit

The primary endpoint is patient-weighted macro IoU of the final four-role
native mask, with all 54 evaluation patients included. Compute each patient's
mean of its available ROI scores first, then average patients. The principal
paired comparison is final Bora minus its own upstream coarse map for each
route on identical valid annotated pixels. Report 95% patient-bootstrap
intervals and centre-specific estimates without treating 170 ROIs as
independent patients. Secondary endpoints are per-role Dice/IoU, missing-class
frequency, native boundary F1 at a fixed physical tolerance, topology failures,
and native raster/GeoJSON agreement.

To assess the method against a contemporary direct predictor, use the official
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 2D configuration, following its
[published self-configuration method](https://doi.org/10.1038/s41592-020-01008-z),
with the same development-patient annotations, four-role ontology and input H&E
images. Pin the software revision, training budget, configuration and model
selection before comparing evaluation predictions.
Its task differs from conditional refinement, so present both the common-input
coarse-versus-Bora comparison and the common-supervision dense comparison rather
than conflating them. Record cold end-to-end time including image I/O,
Virchow2 extraction where applicable, model startup, refinement, classification
and vector export, as well as peak CPU/GPU memory on specified hardware.

Challenge feedback can be used only once for this confirmatory analysis.
Changing thresholds, choosing between readouts or excluding cases after seeing
the score converts subsequent comparisons to exploratory development results.
No BEETLE evaluation metric belongs in the manuscript until the frozen masks
have been submitted and officially scored.
