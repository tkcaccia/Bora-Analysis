#!/usr/bin/env python3
"""Build the Visium expression viewer only from verified whole-section renders."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
from shutil import copy2, copyfileobj


CASES = (
    ("breast_giotto", "breast", "Breast cancer · Giotto", "https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_web_summary.html"),
    ("colorectal_seurat", "colorectal", "Colorectal cancer · Seurat", "https://www.10xgenomics.com/datasets/human-colorectal-cancer-11-mm-capture-area-ffpe-2-standard"),
    ("tonsil_spatialexperiment", "tonsil", "Tonsil · SpatialExperiment", "https://www.10xgenomics.com/datasets/visium-cytassist-gene-and-protein-expression-library-of-human-tonsil-with-add-on-antibodies-h-e-6-5-mm-ffpe-2-standard"),
)
GLOBAL = (
    ("he-overview.jpg", "Original H&E"),
    ("tile-groups-over-he.jpg", "Gene-expression groups assigned to tiles"),
    ("global-rgb-overview.jpg", "Whole-section Virchow2 group responses"),
    ("full-output-mask-with-geojson.jpg", "Bora output mask and GeoJSON boundaries"),
)
CROPS = (("wide", "1.683 mm"), ("large", "842 µm"), ("field", "561 µm"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image(src: str, label: str) -> str:
    return (f'<figure><a href="{html.escape(src)}"><img loading="lazy" src="{html.escape(src)}" '
            f'alt="{html.escape(label)}"></a><figcaption>{html.escape(label)}</figcaption></figure>')


def package_vector(vector: Path, archive: Path, expected: str) -> int:
    """Accept a raw or gzip mirror, verifying the exact decoded final bytes."""
    if vector.is_file():
        if sha(vector) != expected:
            raise ValueError(f"Native geometry differs: {vector}")
        with vector.open('rb') as source, archive.open('wb') as destination:
            with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as packed:
                copyfileobj(source, packed)
    else:
        copy2(Path(str(vector) + '.gz'), archive)
    digest = hashlib.sha256()
    count = 0
    with gzip.open(archive, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
            count += len(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Compressed download differs: {vector}")
    return count


def main(render_root: Path, study_root: Path, site: Path) -> None:
    target = site / "visium"
    target.mkdir(parents=True, exist_ok=True)
    blocks = []
    published = {}
    for slug, anchor, title, sample_url in CASES:
        root = render_root / slug
        receipt = json.loads((root / "RENDER_COMPLETE.json").read_text())
        if receipt.get("status") != "COMPLETE" or receipt.get("case") != slug:
            raise ValueError(f"Incomplete render: {slug}")
        if not receipt.get("all_nine_zoom_masks_exact_global_crops") or receipt.get("no_field_specific_model_or_cleanup") is not True:
            raise ValueError(f"Non-global crop: {slug}")
        files = receipt["files_sha256"]
        output = target / "fullsection" / anchor
        output.mkdir(parents=True, exist_ok=True)
        vector = study_root / slug / 'full-slide-01/BORA_FULL_SLIDE.wsi.geojson'
        download_name = f'{anchor}-Visium-Bora-final.geojson'
        compressed_name = download_name + '.gz'
        vector_bytes = package_vector(vector, output / compressed_name,
                                      receipt['global_geojson_sha256'])
        selected = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".png")]
        asset_hashes = {}
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            selected.extend(p for p in folder.iterdir() if p.name in ("he.jpg", "output-mask-with-geojson.jpg", "coarse-over-he.jpg", "global-rgb.jpg", "group-01.jpg"))
        for path in selected:
            rel = path.relative_to(root)
            if files.get(str(rel)) != sha(path):
                raise ValueError(f"Render checksum differs: {slug}/{rel}")
            dst = output / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy2(path, dst)
            if sha(dst) != sha(path):
                raise ValueError(f"Copy checksum differs: {dst}")
            asset_hashes[str(dst.relative_to(target))] = sha(dst)
        published[slug] = {"render_receipt_sha256": sha(root / "RENDER_COMPLETE.json"),
                           "global_mask_sha256": receipt["global_mask_sha256"],
                           "global_geojson_sha256": receipt["global_geojson_sha256"],
                           "group_count": receipt["groups"], "crop_count": len(receipt["fields"]),
                           "download": f"fullsection/{anchor}/{compressed_name}",
                           "uncompressed_filename": download_name,
                           "geojson_bytes": vector_bytes,
                           "compressed_bytes": (output / compressed_name).stat().st_size,
                           "compressed_sha256": sha(output / compressed_name),
                           "coordinates": "original full-resolution 10x H&E pixel coordinates",
                           "identities": "anonymous gene-expression groups",
                           "native_mpp_xy": receipt["native_mpp_xy"],
                           "minimum_polygon_and_hole_um2": 400,
                           "assets_sha256": asset_hashes}
        prefix = f"fullsection/{anchor}/"
        height, width = receipt['shape_yx']
        mppx, mppy = receipt['working_mpp_xy']
        span = f"{width*mppx/1000:.2f} × {height*mppy/1000:.2f} mm"
        published[slug]['physical_span_mm_xy'] = [width*mppx/1000, height*mppy/1000]
        overview = "".join(image(prefix + filename, caption + ' · ' + span) for filename, caption in GLOBAL)
        gray = "".join(image(prefix + f"group-{g:02d}.jpg", f"Group {g:02d} versus strongest competing group")
                       for g in range(1, receipt["groups"] + 1))
        zooms = []
        for kind, length in CROPS:
            heading = 'Small' if kind == 'field' else kind.title()
            parts = [f"<h3>{heading} fields · {length} across</h3>"]
            for index in range(1, 4):
                name = f"{kind}-{index:02d}"
                folder = prefix + name + "/"
                parts.append(f'<h4>{name.replace("-", " ").title()}</h4><div class="overview">'
                             + image(folder + "he.jpg", "H&E used for inference · " + length + " across")
                             + image(folder + "coarse-over-he.jpg", "Input expression-derived tile groups over H&E · " + length + " across")
                             + image(folder + "global-rgb.jpg", "Slide-wide Virchow2 group responses · " + length + " across")
                             + image(folder + "output-mask-with-geojson.jpg", "Exact crop of whole-section Bora mask with geometry · " + length + " across")
                             + "</div>")
            zooms.append("".join(parts))
        coverage_note = (
            '<p class="note">Coverage limitation: the tissue-support rule excludes part of the pale breast tissue, '
            'most clearly in Wide 02 and Large 03. Those uncolored regions are an output failure, '
            'not evidence of background or absence of tissue.</p>' if anchor == 'breast' else ''
        )
        blocks.append(f'''<section id="{anchor}" class="tissue">
<h2>{html.escape(title)}</h2>
<p>The measured spot clusters were frozen before image refinement. Bora used one whole-section fit and all {receipt['groups']} response maps. Output outside the measured spot footprint is image-only extrapolation.</p>
<p><a href="{html.escape(sample_url)}">10x Genomics source sample</a></p>
<p><button type="button" class="download geojson-download" data-src="{prefix}{compressed_name}" data-filename="{download_name}" aria-describedby="download-status-{anchor}">Download final whole-section GeoJSON ({vector_bytes/1e6:.1f} MB)</button> · <a href="{prefix}{compressed_name}" download="{compressed_name}">Compressed copy ({(output / compressed_name).stat().st_size/1e6:.1f} MB)</a> <span id="download-status-{anchor}" aria-live="polite"></span></p>
<p>The GeoJSON uses the original full-resolution H&amp;E image pixel coordinates and retains the anonymous expression-group identifiers shown here.</p>
{coverage_note}
<div class="overview">{overview}</div>
<h3>KODAMA embedding of expression spots</h3>
<div class="embedding">{image(prefix + 'kodama-groups.png', 'Expression spots colored by the same groups as the mask')}</div>
<details><summary>Inspect all {receipt['groups']} grayscale group responses</summary><p>These are display contrasts against the strongest competing group. The mask uses all response channels together.</p><div class="gray">{gray}</div></details>
{''.join(zooms)}
</section>''')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bora · Visium gene-expression clustering</title><link rel="stylesheet" href="../viewer.css"><style>
main{{max-width:1900px}}.tissue{{padding:24px 0;border-top:2px solid #cbd8dd}}.overview{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.overview figure,.gray figure{{min-width:0}}.embedding{{max-width:1150px}}.gray{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.note{{border-left:4px solid #006d80;background:white;padding:12px 18px;max-width:none}}.jump{{display:flex;gap:18px;flex-wrap:wrap}}.download{{display:inline-block;background:#006d80;color:white;padding:10px 14px;border:0;border-radius:6px;text-decoration:none;font:inherit;font-weight:700;cursor:pointer}}.download:hover,.download:focus{{background:#004b5a}}.download:disabled{{opacity:.65;cursor:wait}}h4{{margin:24px 0 8px}}@media(max-width:900px){{.overview,.gray{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:600px){{.overview,.gray{{grid-template-columns:1fr}}}}
</style></head><body><header><p class="eyebrow">BORA · BENCHMARK 3</p><h1>Visium gene-expression clustering</h1><p>Whole-section Bora refinement of spot clusters from three expression-analysis workflows.</p><nav class="benchmark-sections" aria-label="Benchmark collections"><a href="../index.html"><strong>1 · Public datasets and masks</strong><span>Expert-annotated histology images</span></a><a href="../visium-hd/"><strong>2 · Visium HD whole slides</strong><span>Whole-slide tissue examples</span></a><a href="index.html" aria-current="page"><strong>3 · Visium gene clusters</strong><span>Breast, colorectal, and tonsil examples</span></a></nav></header><main>
<p class="note">Each mask was generated over the complete H&E section with one fixed fit. The nine fields per tissue are exact crops of that output. No expert pixel-level mask is available for these examples, so they demonstrate registered refinement, not segmentation accuracy.</p>
<nav class="jump"><a href="index.html">All samples</a><a href="breast.html">Breast · Giotto</a><a href="colorectal.html">Colorectal · Seurat</a><a href="tonsil.html">Tonsil · SpatialExperiment</a></nav>
{''.join(blocks)}
<p><a href="https://github.com/tkcaccia/Bora-Analysis/tree/main/pipeline/visium_clustering">Reproduce the expression clustering</a> · <a href="FULLSECTION_PROVENANCE.json">Output provenance</a></p>
</main><script src="geojson-download.js" defer></script></body></html>'''
    all_blocks = ''.join(blocks)
    cards = []
    for block, (slug, anchor, title, _) in zip(blocks, CASES):
        detail = page.replace(all_blocks, block).replace(
            '<h1>Visium gene-expression clustering</h1>', f'<h1>{html.escape(title)}</h1>')
        (target / f'{anchor}.html').write_text(detail)
        intro = block.split('<div class="overview">', 1)[0]
        sx, sy = published[slug]['physical_span_mm_xy']
        cards.append(intro + image(f'fullsection/{anchor}/full-output-mask-with-geojson.jpg',
            f'Whole-section output and GeoJSON boundaries · {sx:.2f} × {sy:.2f} mm') +
            f'<p><a href="{anchor}.html">View the complete result and all nine zooms →</a></p></section>')
    landing = page.replace(all_blocks, '<div class="sample-grid">' + ''.join(cards) + '</div>')
    landing = landing.replace('</style>', '.sample-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}.sample-grid .tissue{min-width:0}.sample-grid .download{max-width:100%}@media(max-width:1000px){.sample-grid{grid-template-columns:1fr}}</style>')
    (target / "index.html").write_text(landing)
    (target / 'geojson-download.js').write_text('''document.querySelectorAll('.geojson-download').forEach(button => {
  button.addEventListener('click', async () => {
    const label = button.textContent;
    const status = document.getElementById(button.getAttribute('aria-describedby'));
    button.disabled = true;
    button.setAttribute('aria-busy', 'true');
    status.textContent = 'Preparing download…';
    try {
      const response = await fetch(button.dataset.src);
      if (!response.ok) throw new Error('Download unavailable');
      const data = await response.arrayBuffer();
      const bytes = new Uint8Array(data);
      let blob = new Blob([data], {type: 'application/geo+json'});
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        if (!('DecompressionStream' in window)) throw new Error('Use compressed copy');
        blob = await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).blob();
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = button.dataset.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      status.textContent = 'Download ready.';
    } catch (error) {
      status.textContent = 'Use the compressed copy beside this button.';
    } finally {
      button.disabled = false;
      button.removeAttribute('aria-busy');
      button.textContent = label;
    }
  });
});
''')
    (target / "FULLSECTION_PROVENANCE.json").write_text(json.dumps(published, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("render_root", type=Path)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("site_dir", type=Path)
    arguments = parser.parse_args()
    main(arguments.render_root, arguments.study_root, arguments.site_dir)
