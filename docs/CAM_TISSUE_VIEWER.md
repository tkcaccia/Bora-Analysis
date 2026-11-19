# Benchmark 3 tissue viewer

The completed CAMELYON16 test pages embed `camelyon16/tissue-viewer.html` in
the “Inspect the display-filtered Bora GeoJSON” section. It adapts the
[`tkcaccia/wsiTools` Tissue visualizer skill](https://github.com/tkcaccia/wsiTools/tree/d423c7b0d0a3d4a7ff8565d5bd56a0cdddf7ce08/skills/wsi-tissue-visualizer)
and its OpenSeadragon controls. The source skill was inspected at commit
`d423c7b0d0a3d4a7ff8565d5bd56a0cdddf7ce08` with wsiTools 0.1.28.

The skill's builder generates a complete Deep Zoom bundle. Benchmark 3 cannot
publish that bundle because it would redistribute full source-slide pixels.
This site therefore uses the skill's read-only OpenSeadragon interaction and
fidelity checks with a public official TIFF range source at detail zoom. A
separate source-verified 1,600-pixel H&E overview supplies fit zoom and an
offline fallback. The previous overview contained visible green and gray
blocks; it is no longer used by this viewer.

`tools/build_cam_tissue_viewer_preview.py` verifies the local source TIFF
SHA256 against the frozen manifest, reads only its appropriate pyramid level,
and writes the small tissue overview, an alpha mask, and a vector boundary for
close zoom. The display mask is
derived from H&E chroma and intersected with the reviewed H&E support. It is
for removing obvious exterior glass in the viewer only. It may omit pale
peripheral tissue. Neither it nor the two-pass visual contour fit enters
prediction or scoring. The exact saved review GeoJSON remains downloadable;
the separate displayed GeoJSON records the visual fit and its display-only
small-region filter. The model output, reviewed analysis support, exact scored
raster, and metrics are not recalculated or modified. The embedded viewer sizes
itself to its controls and image so its own scrollbar does not hide the toolbar.

The official TIFF detail needs a network connection to AWS and browser range
requests. If unavailable, the tissue-only overview remains usable and the
viewer shows a warning. The viewer uses OpenSeadragon 5.0.1 from jsDelivr and
the site's already vendored geotiff.js 3.0.5.

After rendering a new completed test page, first run
`python3 tools/build_cam_tissue_viewer_preview.py --site site --slide SLIDE --source /absolute/path/to/SLIDE.tif`,
then `python3 tools/install_cam_tissue_viewer.py` and
`python3 tools/verify_cam_tissue_viewer.py`. Pending pages without a saved
Bora review vector are intentionally excluded. Visually inspect **each** new
display mask against source H&E, then check fit, medium, close zoom, at least
two landmarks, narrow viewport, controls and browser console before publishing.
If the H&E color refinement omits tissue or includes glass, do not publish the
new viewer until its display mask is reviewed and corrected. The wsiTools skill
itself remains suitable for locally built or publishable full Deep Zoom bundles;
this source-only tile mode is a Benchmark 3 publication constraint.
