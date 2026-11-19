# Benchmark 3 tissue viewer

The completed CAMELYON16 test pages embed `camelyon16/tissue-viewer.html` in
the “Inspect the display-filtered Bora GeoJSON” section. It adapts the
[`tkcaccia/wsiTools` Tissue visualizer skill](https://github.com/tkcaccia/wsiTools/tree/d423c7b0d0a3d4a7ff8565d5bd56a0cdddf7ce08/skills/wsi-tissue-visualizer)
and its OpenSeadragon controls. The source skill was inspected at commit
`d423c7b0d0a3d4a7ff8565d5bd56a0cdddf7ce08` with wsiTools 0.1.28.

The skill normally generates a complete Deep Zoom image bundle. Benchmark 3
instead uses its reviewed tissue-only overview at fit zoom and reads visible
source TIFF tiles on demand from the public official CAMELYON16 AWS object at
close zoom. This keeps full source-slide pixels out of GitHub Pages. The
viewer masks each detailed tile to the reviewed H&E support and registers the
saved Bora contours through the calibrated-to-source coordinate transform.
The original review GeoJSON is downloadable. A separate display-only GeoJSON
omits metastasis-like polygons smaller than 20,000 µm², matching the existing
site display rule. The underlying model output, exact scored raster, saved
vector and metrics are not recalculated or modified.

The official TIFF detail needs a network connection to AWS and browser range
requests. If unavailable, the tissue-only overview remains usable and the
viewer shows a warning. The viewer uses OpenSeadragon 5.0.1 from jsDelivr and
the site's already vendored geotiff.js 3.0.5.

After rendering a new completed test page, run
`python3 tools/install_cam_tissue_viewer.py` and
`python3 tools/verify_cam_tissue_viewer.py`. Pending pages without a saved
Bora review vector are intentionally excluded. Check fit, medium and close
zoom on at least one positive and one cropped normal slide before publishing.
