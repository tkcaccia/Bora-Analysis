# Identify the original tissue groups after Bora

This is the corrected primary supervised experiment. The prior fixed-four-cluster
experiment in `bora_segment/geometry.py` is superseded and retained as an audited
exploratory history. It does not supply inputs to this benchmark.

The input is the **exact previous expert-derived categorical tile mask**, with
only its IDs deidentified. The old 9 + 19 evidence RGB is copied unchanged. No
KMeans, new partition, new class, new RGB fitting or new boundary restriction is
introduced. One anonymous group remains one classification unit even when its
mask has several disconnected pieces.

## Fixed inputs and proof

The source registry contains 469 calibrated ROIs or WSI windows. `geometry.py`
checks the image, original tile mask, frozen RGB and full Virchow2 feature hashes.
It replaces positive source IDs with a per-image anonymous permutation, retaining
zero exactly. Inverse mapping must reproduce every input tile assignment.

Bora-AF receives the same H&E, RGB, support and physical sampling as the previous
benchmark. A separate original-ID solve must match the anonymized result exactly
after ID restoration. This parity solve is an audit and is excluded from runtime.
The present run preserves all input groups: 1,359 whole groups over 469 images.
It replaces the earlier 67,008-fragment interpretation.

All geometry freezes before semantic targets attach. The reversible numeric
answer key lives in `evaluation_keys`, separately from head input arrays. The
head consumes full 1,280-channel Virchow2 evidence pooled inside each group,
or tile ranks aggregated by overlap. It receives neither the anonymous ID,
source ID, source tissue name, conversion palette nor per-image class vocabulary.
RGB construction and the coarse partition remain reference-derived: this is a
controlled benchmark of refinement and tissue identification, not autonomous
discovery of tissue regions.

## Supervision and models

Training hard labels are the original expert tissue identities behind the
deidentified groups. They are not arbitrary cluster IDs or post-refinement
fragment majorities. A group needs a mapped identity and at least one mapped
expert pixel; unannotated pixels never contribute to pixel quality. Tissue
composition targets describe the annotated part only. The current training
set has 838 labelled whole groups from 315 internal training ROIs and 25 broad
tissue roles. Source labels and stable ontology IDs are retained. The ontology
still needs pathology review; generic epithelium and organ-specific categories
are not asserted mutually exclusive clinical entities.

Five formulations share the exact same anonymous Bora output:

|Family|Whole-group evidence|Native head|
|---|---|---|
|R01|Area-weighted voting of training-fitted tile ranks|Existing native PLS-LDA plus group voting|
|R02|Full-feature group mean|New group PLS-LDA|
|R03|Group mean and standard deviation|New group PLS-LDA|
|R04|Group mean and annotated tissue-composition responses|New PLS regression|
|R05|Group mean and shared-boundary neighbor contrast|New group PLS-LDA|

R02–R05 compare 16, 24 and 48 latent components. Components are hyperparameters,
not separate method families. Model selection uses validation group balanced
accuracy. All reserved predictions freeze before test identities or fine test
masks are used for this cycle's scoring. The previously inspected cohort is an
exploratory follow-up; it cannot become an untouched test through anonymization.

The internal partition remains 315 training, 69 validation and 85 reserved test
ROIs. All 85 receive predictions. The multiclass viewer and aggregate benchmark
retain 62 with at least two mapped expert tissue roles; the other 23 have an
explicit exclusion list. PESO's provider test release is development data in
these internal partitions; see the source-split audit in `bora_supervised`.

## GeoJSON and viewer

The final export uses **`bora.multicpu_geojson.convert`**, the published Bora
converter, unchanged from commit `08b63a6595df687bd20a516f43f65dc531e664dd`.
The primary `accurate` profile polygonizes at native resolution, filters source
components below 500 native pixels, performs coverage-aware simplification at
tolerance 2 native pixels, then dissolves to one feature per predicted tissue
label. Shared borders are simplified together. One low-priority worker avoids
unnecessary parallel memory use on these ROIs. No new converter is substituted.

These are the existing profile's native-pixel settings. Each image's metadata
also records the corresponding tolerance in micrometres and area in µm²; these
physical equivalents vary with the native pixel spacing. The encoder calibration
and input tile assignments remain unchanged.

Coordinates are native **ROI-local image pixel edges**, with a top-left origin;
source WSI crop offsets are not applied to these ROI-local coordinates.
No geographic-coordinate claim is made. Final polygons must be valid and have
zero positive-area interclass overlap. Native rasterization measures geometric
agreement and final expert-reference IoU; pixel identity is **not** required.
The original pixel-edge exports in `geojson-test` are a superseded audit, retained
for before/after complexity comparisons. Their exactness is not a quality goal.

The reorganized viewer shows original H&E, expert reference, anonymous input
tiles, unchanged RGB, anonymous Bora output and identified tissue. Five head
outputs can be compared side by side. Final tissue previews come from rasterizing
the actual simplified GeoJSON. Expert source names have their own matching-color
legend directly below the reference image, including explicit ignored labels.
The interactive GeoJSON panel supports
opacity, outlines, zoom, pan and download. The top-five column is removed from
the viewer; ranked predictions remain research outputs. Reference tissue names
are comparison labels revealed after prediction, not classifier inputs.

The validation-selected primary is R02-c24 (whole-group mean). On the 62 eligible
internal test ROIs it identifies 174/199 scored groups correctly (87.4%), with
66.5% group balanced accuracy and 0.554 case-mean semantic pixel IoU. R05-c48
(neighbor context) has the highest descriptive test scores: 180/199 groups
(90.5%), 71.3% balanced accuracy and 0.578 case-mean semantic IoU. This test result
does not replace the validation-selected primary. These semantic scores include
both boundary error and tissue identification error.

After the published accurate-profile conversion, primary R02-c24 has mean
GeoJSON IoU 0.554914 across the same 62 scored ROIs (raster IoU 0.554429).
Across all 85 reserved exports its polygon parts fall from 7,252 to 2,956 and
vertices from 878,107 to 497,244. The mean converter call takes 0.108 s/ROI with
one worker. These are cached-mask export timings, separate from inference,
feature extraction, startup and evaluation. The whole-group classifications
and original raster masks are unchanged.

## Execution

```bash
python -m experiments.bora_group.geometry --registry-root REGISTRY --output GROUP_STUDY
python -m experiments.bora_group.workflow --study-root SUPERVISED_STUDY --geometry GROUP_STUDY
```

The existing study root supplies `registry-v1`, full features and the tile-vote
baseline bundle. The workflow trains new whole-group models, selects on validation,
predicts every reserved image, scores eligible masks, exports GeoJSON for every
reserved image and builds `viewer-bora-geojson-v2/index.html`. Each run uses fresh immutable
output paths. `source-*` snapshots and launch receipts preserve executed code.

fastPLS uses the pinned C++ CPU backend from the earlier study, with two numerical
threads and one low-priority numerical worker. Rasterio 1.4.4 and Shapely 2.1.2
provide polygonization, validity checks and rasterization. Startup refitting,
pooling, warm prediction, Bora refinement and vector export are distinct costs.

To regenerate vectors from already frozen predictions without rerunning Bora or
training, use:

```bash
python -m experiments.bora_group.bora_geojson --registry-root REGISTRY \
  --geometry GROUP_STUDY --cycle GROUP_STUDY/classifiers-test-v1 \
  --output GROUP_STUDY/geojson-bora-accurate-v2 \
  --viewer-output GROUP_STUDY/viewer-bora-geojson-v2
```

To extend training, use the append-only registry versioning described in
`experiments/bora_supervised/README.md`, generate expert-derived coarse masks using
the existing calibration/tiling rules, and produce a new anonymized group study.
Preserve old case hashes, group partitions, label meanings and test-use history.
