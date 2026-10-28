# Optional WSI color-drift correction

Run `pipeline/wsi_color_preprocess.py` **before** physical resampling and
Virchow2 extraction. It writes a separate lossless pyramidal OME-TIFF with
the original dimensions and verified micrometres per pixel. Annotations keep
their original coordinates. Source images and completed benchmarks are retained.

```bash
python pipeline/wsi_color_preprocess.py \
  --image /path/to/original.tif --mpp 0.2737752 0.2737744 \
  --source-sha256 VERIFIED_SOURCE_SHA256 \
  --output /path/to/new-color-preprocessing --apply
```

Use the full-precision spacing from the source metadata in place of these
example values. Dependencies are NumPy, SciPy, scikit-learn, Pillow, pyvips
and a libvips installation with TIFF support. The step is CPU-only. Its
default storage reserve is 80 GiB. Omitting `--apply` produces a diagnostic
and model without writing a corrected WSI.

## Method and decision

A thumbnail is used to identify bright, low-chroma blank glass connected to
the slide edge and separated from stained tissue. Median RGB values are
sampled in a 24 × 24 grid. A robust plane is fitted separately to each
channel in global slide coordinates. The local gain is the median glass
intensity divided by the predicted plane. Channels without supporting
evidence are left unchanged.

The fixed gates require at least 48 distributed cells, 75% coverage of each
coordinate axis, eight cells in each spatial quadrant and no more than 5%
clipped glass. A channel must show a predicted spatial span of at least 1.5%,
reduce held-out spatial-quadrant median absolute prediction error by at least
25% relative to a constant model, and require no gain outside ±5%. These are
conservative technical gates; they are not clinically validated cutoffs.

One smooth field is applied everywhere. Tissue brightness and histograms are
not equalized, and no expert annotations are used. No dark-field calibration
or ICC conversion is inferred. This is an empirical correction of decoded
RGB values, not BaSiC, Macenko normalization or a reconstruction of scanner
illumination. Slides with insufficient glass are skipped for review. A skip
does not prove that a slide is artifact-free.

The output includes `COLOR_MODEL.json`, the glass samples, before/after
previews and `PREPROCESS_COMPLETE.json`. The lossless export is checked
against an independent NumPy implementation at 15 fixed locations. Original
and output hashes, physical spacing, decisions and runtime are recorded.

## Use in the benchmark

Read `PREPROCESS_COMPLETE.json` and use its `output_image`, `output_sha256`
and `mpp_xy` to make a **new** wsiTools tile plan. If the decision is `SKIP`,
`output_image` points to the unchanged source. The existing
`plan_virchow2_tiles.R` and encoder accept this new plan and image. A staged
copy can be passed to `encode_virchow2.py --image STAGED_IMAGE
--source-image PLANNED_IMAGE --image-sha256 OUTPUT_SHA256
--preprocessing-receipt /path/to/new-color-preprocessing/PREPROCESS_COMPLETE.json`.
The encoder checks the receipt, original planned-image path, output checksum,
dimensions and physical spacing before loading the model, and records the
preprocessing identity alongside the new features.

For an applied correction, new feature outputs and all subsequent KODAMA
and Bora fits are required. Never combine corrected H&E with Virchow2
features extracted from the original image, reuse a completed queue, or
overwrite its source/plan. Freeze a new preprocessing/feature manifest.
Any comparison should use the same tissue fields, model checkpoint, physical
scale and evaluation protocol. Improved glass uniformity alone does not
establish improved segmentation, classification or embedding quality.

The pancreas pilot is supported by spatial glass validation. The current
colorectal and brain controls are skipped by this policy. This optional step
has not been applied retrospectively to the frozen published benchmark.

Method context: [Peng et al., 2017](https://doi.org/10.1038/ncomms14836)
addresses microscopy shading correction using BaSiC;
[Macenko et al., 2009](https://cseweb.ucsd.edu/~mniethammer/publication/macenko-nmbwgst-09/)
addresses histology appearance normalization. Neither method is claimed as
the implementation used here.
