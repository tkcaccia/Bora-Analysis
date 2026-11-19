#!/usr/bin/env python3
"""Release gate for the Benchmark 3 tissue viewer and completed slide links."""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from PIL import Image, ImageChops

SITE = Path(__file__).resolve().parents[1] / "site"
CAM = SITE / "camelyon16"


def read_json(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def main() -> None:
    manifest = read_json(SITE / "geojson-inspectors.json")
    completed = [
        record for record in manifest["records"]
        if record["id"].startswith("benchmark3-")
        and record.get("geometry", "").endswith("-bora.geojson.gz")
        and record.get("high_resolution")
    ]
    assert completed, "No completed Benchmark 3 records"
    for asset in ("tissue-viewer.html", "tissue-viewer.css", "tissue-viewer.js"):
        assert (CAM / asset).is_file(), asset
    checked = []
    for record in completed:
        slide = record["id"].removeprefix("benchmark3-")
        assert re.fullmatch(r"(?:tumor|normal)_\d{3}", slide), slide
        page = (CAM / f"{slide}.html").read_text()
        url = f"tissue-viewer.html?slide={slide}"
        assert page.count(f'<iframe title="Bora tissue and GeoJSON viewer for {slide}"') == 1, slide
        assert page.count(f'src="{url}"') == 1, slide
        assert page.count('id="inspect-geojson"') == 1, slide
        assert 'class="geojson-inspector"' not in page, slide
        assert f'href="inspector-vectors/{slide}-bora.geojson.gz"' in page, slide
        for key in ("image", "tissue_support", "geometry"):
            assert (SITE / record[key]).is_file(), (slide, key)
        support = read_json(SITE / record["tissue_support"])
        vectors = read_json(SITE / record["geometry"])
        assert support["coordinate_system"] == "calibrated_image_pixel_xy", slide
        assert support["features"], slide
        assert vectors["type"] == "FeatureCollection", slide
        assert len(vectors["features"]) == record["source_feature_count"], slide
        assert record["high_resolution"]["url"].startswith(
            "https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/images/"
        ), slide
        assert record["native_mpp_xy"][0] > 0 and record["native_mpp_xy"][1] > 0, slide
        preview = CAM / "tissue-viewer-previews" / f"{slide}-tissue.webp"
        mask = CAM / "tissue-viewer-previews" / f"{slide}-mask.png"
        display_vector = CAM / "tissue-viewer-previews" / f"{slide}-display-support.geojson.gz"
        assert preview.is_file() and mask.is_file() and display_vector.is_file(), slide
        with Image.open(preview) as picture, Image.open(mask) as alpha_image:
            assert picture.mode == "RGBA" and alpha_image.mode == "RGBA", slide
            assert picture.size == alpha_image.size, slide
            assert picture.width >= 500 and picture.height >= 100, slide
            assert ImageChops.difference(picture.getchannel("A"), alpha_image.getchannel("A")).getbbox() is None, slide
            assert picture.getchannel("A").getextrema() == (0, 255), slide
            visual = read_json(display_vector)
            assert visual["coordinate_system"] == "viewer_preview_pixel_xy", slide
            assert visual["source_sha256"] == record["high_resolution"]["source_sha256"], slide
            assert (visual["width"], visual["height"]) == picture.size, slide
            assert visual["features"] and visual["display_only"] is True, slide
        checked.append(slide)
    print(f"CAM tissue viewer verified: {len(checked)} completed pages: {', '.join(sorted(checked))}")


if __name__ == "__main__":
    main()
