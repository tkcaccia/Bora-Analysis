#!/usr/bin/env python3
"""Attach the read-only tissue viewer to completed Benchmark 3 Bora pages.

This is intentionally idempotent so the per-slide publisher can run it after
rendering a new page. Pending slides have no Bora vector and are not changed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
CAM = SITE / "camelyon16"
MANIFEST = SITE / "geojson-inspectors.json"
OLD = re.compile(r'<section class="geojson-inspector"[^>]* id="inspect-geojson">.*?</section>', re.S)


def main() -> None:
    records = json.loads(MANIFEST.read_text())["records"]
    completed = sorted(
        record["id"].removeprefix("benchmark3-")
        for record in records
        if record["id"].startswith("benchmark3-")
        and record.get("geometry", "").endswith("-bora.geojson.gz")
        and record.get("high_resolution")
        and record.get("tissue_support")
    )
    for slide in completed:
        page = CAM / f"{slide}.html"
        if not page.is_file():
            raise SystemExit(f"Missing completed slide page: {page}")
        html = page.read_text()
        block = (
            '<section class="cam-tissue-viewer" id="inspect-geojson">'
            '<h2>Inspect the display-filtered Bora GeoJSON</h2>'
            '<p>Pan and zoom the reviewed H&amp;E tissue with Bora contours. '
            'The viewer hides metastasis-like polygons smaller than 0.02 mm² for display only. '
            'The saved review vector and exact scored raster are unchanged.</p>'
            f'<iframe title="Bora tissue and GeoJSON viewer for {slide}" '
            f'src="tissue-viewer.html?slide={slide}" loading="lazy" '
            'style="display:block;width:100%;height:clamp(620px,calc(70vw + 160px),1020px);border:0" '
            'referrerpolicy="no-referrer"></iframe>'
            f'<p><a href="tissue-viewer.html?slide={slide}">Open the tissue viewer in a full window</a> · '
            f'<a download href="inspector-vectors/{slide}-bora.geojson.gz">Download the saved Bora GeoJSON</a>.</p>'
            '</section>'
        )
        if OLD.search(html):
            html, count = OLD.subn(block, html)
            assert count == 1, (slide, count)
        elif re.search(r'<section class="cam-tissue-viewer" id="inspect-geojson">.*?</section>', html, re.S):
            html = re.sub(r'<section class="cam-tissue-viewer" id="inspect-geojson">.*?</section>', block, html, count=1, flags=re.S)
        else:
            raise SystemExit(f"Missing inspection section: {page}")
        html = re.sub(r'<link href="\.\./geojson-inspector\.css[^\"]*" rel="stylesheet"\s*/>', '', html)
        html = re.sub(r'<script defer="" src="\.\./vendor/geotiffjs-3\.0\.5/geotiff\.js"></script>', '', html)
        html = re.sub(r'<script defer="" src="\.\./geojson-inspector\.js[^\"]*"></script>', '', html)
        page.write_text(html)
    print(f"Installed tissue viewer on {len(completed)} completed Benchmark 3 pages")


if __name__ == "__main__":
    main()
