# Visium whole-section outputs

The Visium viewer provides the final whole-section GeoJSON for the breast
(Giotto), colorectal (Seurat), and tonsil (SpatialExperiment) examples.
Each tissue page shows one saved global output and nine exact crops of it.
The expression-group IDs are anonymous; these files do not provide expert
tissue names or establish pixel-level accuracy.

The download button decompresses the verified gzip package in the browser and
saves a `.geojson` file. A direct `.geojson.gz` link is also provided. Both
routes preserve the exact exported bytes. Coordinates refer to the original
full-resolution 10x H&E image, rather than the approximately 0.5 µm/pixel
working raster. `site/visium/FULLSECTION_PROVENANCE.json` records the hashes,
physical calibration, group counts, and image identities.

The exports include smoothing and global removal/filling of polygon parts and
enclosed holes smaller than 400 µm². Refinement was run before cropping; no
field-specific model or cleanup was applied. Uncolored pale regions in the
breast example expose a tissue-support failure, which is disclosed on the page.
Predictions outside the measured spot footprint are image-based extrapolations.

To rebuild from verified render receipts and exported vectors:

```bash
python tools/build_visium_fullsection.py \
  /path/to/verified/renders \
  /path/to/verified/study \
  site
python tools/verify_public_viewer.py --site site
```

The render root contains `breast_giotto`, `colorectal_seurat`, and
`tonsil_spatialexperiment`, each with a `RENDER_COMPLETE.json` receipt and the
referenced images. The study root contains the same case directories, each
with `full-slide-01/BORA_FULL_SLIDE.wsi.geojson` or its exact gzip mirror.
The builder rejects incomplete receipts, altered render assets, and mismatched
decoded vector hashes. The static-site verifier additionally checks published
links, all image hashes, vector hashes, and physical polygon/hole areas.
