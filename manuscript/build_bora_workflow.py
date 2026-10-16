"""Draw the landscape Bora workflow as an editable publication SVG."""
from pathlib import Path
from xml.sax.saxutils import escape
from benchmark.runtime import BORA_ROOT

HERE = BORA_ROOT / 'docs'
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

parts = ["""<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1190" viewBox="0 0 1800 1190">
<defs>
  <marker id="arrow" viewBox="0 0 28 28" markerUnits="userSpaceOnUse" markerWidth="28" markerHeight="28" refX="25" refY="14" orient="auto">
    <path d="M2 3 L25 14 L2 25" fill="none" stroke="#111" stroke-width="3.7" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
<rect width="1800" height="1190" fill="white"/>
<style>
text {font-family:DejaVu Sans,Arial,sans-serif;fill:#111}
.phase {font-size:25px;font-weight:700;letter-spacing:1px}
.lane {font-size:27px;font-weight:700}
.note {font-size:21px;fill:#3b4652}
.head {font-size:29px;font-weight:700}
.body {font-size:24px}
</style>
<text class="phase" x="90" y="49">A  INPUT LABEL ROUTES</text>
<line x1="460" y1="40" x2="1710" y2="40" stroke="#c4cbd2" stroke-width="2"/>
<text class="lane" x="470" y="93" text-anchor="middle">Prospective feature-clustering route</text>
<text class="note" x="470" y="119" text-anchor="middle">integration context; not evaluated here</text>
<text class="lane" x="1330" y="93" text-anchor="middle">Controlled benchmark route</text>
<text class="note" x="1330" y="119" text-anchor="middle">reference-derived inputs; evaluated here</text>
"""]


def box(x, y, w, h, head, body, fill="#f7f9fa"):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" fill="{fill}" stroke="#1c2733" stroke-width="2.6"/>')
    parts.append(f'<text class="head" x="{x+w/2}" y="{y+34}" text-anchor="middle">{escape(head)}</text>')
    parts.append(f'<text class="body" x="{x+w/2}" y="{y+63}" text-anchor="middle">{escape(body)}</text>')


def arrow(points, dashed=False):
    coords = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    parts.append(f'<polyline points="{coords}" fill="none" stroke="#111" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"{dash} marker-end="url(#arrow)"/>')


box(90, 143, 760, 80, "Original H&E image", "RGB tissue image at known pixel coordinates", "#eef5f8")
box(950, 143, 760, 80, "Dense reference annotation", "benchmark only; dataset-specific tissue labels", "#faf5ed")
box(90, 247, 760, 80, "Tile-feature extraction", "UNI2 or Virchow2 embeddings", "#eef5f8")
box(950, 247, 760, 80, "tiling-mask", "majority vote at 56, 90 or 216 pixel width", "#faf5ed")
box(90, 351, 760, 80, "Cluster or classify tiles", "map feature groups to categorical IDs", "#eef5f8")
box(950, 351, 760, 80, "Quantize the reference", "0%, 1% or 5% requested tile-border error", "#faf5ed")
for x in (470, 1330):
    arrow([(x, 223), (x, 247)])
    arrow([(x, 327), (x, 351)])

parts.append('<text class="phase" x="90" y="477">B  BORA REFINEMENT</text>')
parts.append('<line x1="432" y1="468" x2="1710" y2="468" stroke="#c4cbd2" stroke-width="2"/>')
box(500, 496, 800, 80, "Align image and categorical mask", "expand labels to exact RGB dimensions", "#f3f6f8")
arrow([(470, 431), (470, 451), (700, 451), (700, 496)])
arrow([(1330, 431), (1330, 451), (1100, 451), (1100, 496)])
# The same source image supports both the proposed encoder and Bora itself.
arrow([(90, 183), (51, 183), (51, 536), (500, 536)])
box(500, 609, 800, 80, "Validate and partition", "class map, ignored pixels and haloed blocks", "#f3f6f8")
arrow([(900, 576), (900, 609)])
box(90, 722, 760, 80, "Internal tissue interfaces", "editable class-pair bands; protected interiors", "#f2f7f4")
box(950, 722, 760, 80, "Tissue–background interface", "foreground and background seeds", "#f2f7f4")
arrow([(900, 689), (900, 705), (470, 705), (470, 722)])
arrow([(900, 689), (900, 705), (1330, 705), (1330, 722)])
box(90, 826, 760, 80, "Balanced local colour competition", "spatial CIELAB costs for adjacent classes", "#f2f7f4")
box(950, 826, 760, 80, "Seeded GrabCut", "image-guided external foreground contour", "#f2f7f4")
arrow([(470, 802), (470, 826)])
arrow([(1330, 802), (1330, 826)])
box(500, 939, 800, 80, "Cleanup and class retention", "categorical smoothing and seed connectivity", "#f3f6f8")
arrow([(470, 906), (470, 923), (700, 923), (700, 939)])
arrow([(1330, 906), (1330, 923), (1100, 923), (1100, 939)])

parts.append('<text class="phase" x="90" y="1061">C  OUTPUT AND ASSESSMENT</text>')
parts.append('<line x1="523" y1="1052" x2="1710" y2="1052" stroke="#c4cbd2" stroke-width="2"/>')
box(120, 1081, 480, 80, "OME-TIFF raster", "native-size categorical mask", "#f7f8fa")
box(660, 1081, 480, 80, "GeoJSON regions", "level-zero class polygons", "#f7f8fa")
box(1200, 1081, 480, 80, "Evaluation", "borders, overlap, fidelity, time", "#faf5ed")
arrow([(600, 1019), (600, 1039), (580, 1039), (580, 1081)])
arrow([(600, 1121), (660, 1121)])
arrow([(1140, 1121), (1200, 1121)])
# A separate dashed reference path ends at evaluation, not at Bora refinement.
arrow([(1710, 183), (1754, 183), (1754, 1121), (1680, 1121)], dashed=True)
parts.append('</svg>')

path = OUT / "bora_workflow.svg"
path.write_text("\n".join(parts))
print(path)
