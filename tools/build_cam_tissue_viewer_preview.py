#!/usr/bin/env python3
"""Build a display-only, tissue-masked H&E overview from the audited source TIFF.

This never reads a prediction or reference and does not change the analysis mask.
The output is a small overview; close zoom reads the official source TIFF.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import pyvips
from PIL import Image, ImageChops, ImageDraw
import numpy as np
from scipy import ndimage as ndi
from rasterio.features import shapes

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--slide", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2400)
    args = parser.parse_args()
    if not 500 <= args.width <= 2400:
        parser.error("width must be 500–2400 pixels")
    site = args.site.resolve()
    manifest = json.loads((site / "geojson-inspectors.json").read_text())
    record = next(r for r in manifest["records"] if r["id"] == f"benchmark3-{args.slide}")
    source = args.source.resolve(strict=True)
    expected = record["high_resolution"]
    original = pyvips.Image.new_from_file(str(source), level=0)
    if (original.width, original.height) != (expected["width"], expected["height"]):
        raise ValueError("source dimensions differ from the frozen manifest")
    digest = sha256(source)
    if digest != expected["source_sha256"]:
        raise ValueError("source SHA256 differs from the frozen manifest")

    bx, by, bw, bh = record["image_bounds_xywh"]
    sx = original.width / expected["calibrated_width"]
    sy = original.height / expected["calibrated_height"]
    native_box = (bx * sx, by * sy, bw * sx, bh * sy)
    out_w = args.width
    out_h = round(out_w * native_box[3] / native_box[2])
    if out_h < 1 or out_h > 6000:
        raise ValueError("invalid cropped overview aspect ratio")
    level_count = int(original.get("openslide.level-count"))
    target_downsample = native_box[2] / out_w
    level = min(level_count - 1, max(0, math.floor(math.log2(max(1, target_downsample)))))
    level_image = pyvips.Image.new_from_file(str(source), level=level)
    rx = original.width / level_image.width
    ry = original.height / level_image.height
    x0 = max(0, math.floor(native_box[0] / rx))
    y0 = max(0, math.floor(native_box[1] / ry))
    x1 = min(level_image.width, math.ceil((native_box[0] + native_box[2]) / rx))
    y1 = min(level_image.height, math.ceil((native_box[1] + native_box[3]) / ry))
    cropped = level_image.crop(x0, y0, x1 - x0, y1 - y0)
    if cropped.hasalpha():
        cropped = cropped.flatten(background=[255, 255, 255])
    if cropped.bands != 3:
        raise ValueError("source H&E preview is not RGB")
    raw = cropped.write_to_buffer(".png")
    rgb = Image.open(io.BytesIO(raw)).convert("RGB").resize((out_w, out_h), LANCZOS)

    support_path = site / record["tissue_support"]
    with gzip.open(support_path, "rt") as stream:
        support = json.load(stream)
    if support["coordinate_system"] != "calibrated_image_pixel_xy":
        raise ValueError("unexpected tissue coordinate system")
    # Supersampling antialiases the *display edge* while preserving the frozen
    # reviewed support geometry. It does not smooth or expand the support.
    aa = 2
    mask = Image.new("L", (out_w * aa, out_h * aa), 0)
    def points(ring):
        return [((x - bx) / bw * out_w * aa, (y - by) / bh * out_h * aa) for x, y in ring]
    for feature in support["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            piece = Image.new("L", mask.size, 0)
            draw = ImageDraw.Draw(piece)
            draw.polygon(points(polygon[0]), fill=255)
            for hole in polygon[1:]:
                draw.polygon(points(hole), fill=0)
            mask = ImageChops.lighter(mask, piece)
    mask = mask.resize((out_w, out_h), LANCZOS)
    # The reviewed analysis support is intentionally permissive around pale
    # tissue. For display, exclude obvious exterior glass within that support.
    # This does not feed a predictor, alter the review support, or change scores.
    pixels = np.asarray(rgb, dtype=np.int16)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    seeds = (chroma > 9) & (pixels.min(axis=2) < 246)
    seeds = ndi.binary_closing(seeds, structure=np.ones((9, 9), dtype=bool))
    labels, _ = ndi.label(seeds)
    counts = np.bincount(labels.ravel())
    seeds = (labels > 0) & (counts[labels] >= 60)
    seeds = ndi.binary_dilation(seeds, iterations=4)
    holes = ~seeds
    hole_labels, _ = ndi.label(holes)
    hole_counts = np.bincount(hole_labels.ravel())
    border_labels = set(np.unique(np.concatenate((hole_labels[0], hole_labels[-1],
                                                   hole_labels[:, 0], hole_labels[:, -1]))))
    fill = (hole_labels > 0) & (hole_counts[hole_labels] <= 3000)
    for label in border_labels:
        fill &= hole_labels != label
    seeds |= fill
    reviewed = np.asarray(mask) >= 128
    display = seeds & reviewed
    if display.sum() < 0.25 * reviewed.sum():
        raise ValueError("display tissue detection removed more than 75% of reviewed support")
    alpha = np.uint8(np.clip(ndi.gaussian_filter(display.astype(np.float32), 1.0) * 255, 0, 255))
    mask = Image.fromarray(alpha, "L")
    mask_output = site / "camelyon16" / "tissue-viewer-previews" / f"{args.slide}-mask.png"
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    alpha_image = Image.new("RGBA", mask.size, (255, 255, 255, 0))
    alpha_image.putalpha(mask)
    alpha_image.save(mask_output, "PNG", optimize=True)
    contours = [geometry for geometry, value in shapes(display.astype("uint8"),
                mask=display, connectivity=4) if value == 1]
    if not contours:
        raise ValueError("No vectorized display support")
    support_vector = {
        "type": "FeatureCollection", "coordinate_system": "viewer_preview_pixel_xy",
        "source_sha256": digest, "reviewed_support_sha256": sha256(support_path),
        "width": out_w, "height": out_h,
        "display_only": True,
        "features": [{"type": "Feature", "properties": {}, "geometry": geometry} for geometry in contours],
    }
    vector_output = site / "camelyon16" / "tissue-viewer-previews" / f"{args.slide}-display-support.geojson.gz"
    with gzip.open(vector_output, "wt", compresslevel=9) as stream:
        json.dump(support_vector, stream, separators=(",", ":"))
    image = rgb.convert("RGBA")
    image.putalpha(mask)
    output = site / "camelyon16" / "tissue-viewer-previews" / f"{args.slide}-tissue.webp"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "WEBP", lossless=True, method=5)
    check = Image.open(output).convert("RGBA")
    if check.size != (out_w, out_h) or check.getextrema()[3][1] != 255:
        raise ValueError("preview verification failed")
    print(json.dumps({"slide": args.slide, "source_sha256": digest, "support_sha256": sha256(support_path),
                      "level": level, "size": [out_w, out_h], "reviewed_fraction": round(float(reviewed.mean()), 4),
                      "display_fraction": round(float(display.mean()), 4), "output": str(output),
                      "output_sha256": sha256(output), "mask_sha256": sha256(mask_output),
                      "display_vector_sha256": sha256(vector_output), "display_components": len(contours)}))


if __name__ == "__main__":
    main()
