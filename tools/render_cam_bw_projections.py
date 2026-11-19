#!/usr/bin/env python3
"""Render tissue-only Bora class-response maps from audited score receipts.

Usage: python tools/render_cam_bw_projections.py --manifest /path/to/manifest.json
The manifest supplies local, checksum-bound score receipts and reviewed H&E
support GeoJSON. It may include any newly completed Benchmark 3 slide.
This is a display operation; it never edits masks, vectors, or scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import map_coordinates
from scipy.special import expit


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def overview_image(page: Path) -> Path:
    body = page.read_text()
    section = re.search(r'<section\b[^>]*\bid="overview"[^>]*>.*?</section>', body, re.S)
    if section is None:
        raise ValueError(f'No current overview in {page}')
    he = re.search(r'<figure\b[^>]*data-method="he"[^>]*>\s*<img\b[^>]*\bsrc="([^"]+)"',
                   section.group(), re.S)
    if he is None:
        raise ValueError(f'No overview H&E in {page}')
    return page.parent / he.group(1).split('?', 1)[0]


def render_one(entry: dict, site: Path) -> dict:
    slide = entry['slide']
    page = site / 'camelyon16' / f'{slide}.html'
    score_receipt_path = Path(entry['score_receipt'])
    support_path = Path(entry['reviewed_support'])
    if sha(score_receipt_path) != entry['score_receipt_sha256'] or \
            sha(support_path) != entry['reviewed_support_sha256']:
        raise ValueError(f'{slide}: frozen receipt/support changed')
    receipt = json.loads(score_receipt_path.read_text())
    support_geojson = json.loads(support_path.read_text())
    if receipt['status'] != 'SPARSE_BORA_SCORES_COMPLETE' or \
            receipt['slide_id'] != slide or receipt['classes'] != [1, 2] or \
            receipt['model_sha256'] != entry['model_sha256'] or \
            receipt['support_sha256'] != entry['reviewed_support_sha256'] or \
            support_geojson['source_image_sha256'] != receipt['image_sha256'] or \
            support_geojson['coordinate_system'] != 'calibrated_image_pixel_xy':
        raise ValueError(f'{slide}: source/model/support contract differs')
    score_path = Path(receipt['smoothed_scores_path'])
    if sha(score_path) != receipt['smoothed_scores_sha256']:
        raise ValueError(f'{slide}: smoothed score bytes changed')
    scores = np.load(score_path, mmap_mode='r', allow_pickle=False)
    source_h, source_w = map(int, support_geojson['image_shape_yx'])
    if tuple(scores.shape) != tuple(receipt['score_shape_class_yx']) or \
            scores.shape != (2, source_h // 14, source_w // 14) or \
            scores.dtype != np.float32:
        raise ValueError(f'{slide}: score registration differs')
    output_shape = (source_h // 8, source_w // 8)
    support = rasterize(((feature['geometry'], 1)
                         for feature in support_geojson['features']),
                        out_shape=output_shape, transform=Affine.scale(8, 8),
                        fill=0, dtype='uint8', all_touched=False)
    yy, xx = np.nonzero(support)
    if len(yy) == 0:
        raise ValueError(f'{slide}: empty reviewed tissue')
    top = max(0, int(yy.min()) - 100)
    left = max(0, int(xx.min()) - 100)
    bottom = min(output_shape[0], int(yy.max()) + 101)
    right = min(output_shape[1], int(xx.max()) + 101)
    crop = (top, left, bottom - top, right - left)
    he_path = overview_image(page)
    with Image.open(he_path) as he:
        display_w, display_h = he.size
    if abs((crop[3] / crop[2]) - (display_w / display_h)) > .006:
        raise ValueError(f'{slide}: score crop does not match displayed H&E frame')
    tissue = np.asarray(Image.fromarray(support[top:bottom, left:right] * 255, 'L')
                        .resize((display_w, display_h), Image.Resampling.NEAREST)) > 0
    # Pixel centres follow the released 14-calibrated-pixel score spacing and
    # 8-calibrated-pixel 4 µm output grid. Only a small display frame is sampled.
    y4 = top + (np.arange(display_h, dtype=np.float32) + .5) * crop[2] / display_h
    x4 = left + (np.arange(display_w, dtype=np.float32) + .5) * crop[3] / display_w
    ys = (y4 + .5) * 8 / 14 - .5
    xs = (x4 + .5) * 8 / 14 - .5
    coordinates = np.broadcast_arrays(ys[:, None], xs[None, :])
    one = map_coordinates(scores[0], coordinates, order=3, mode='nearest', prefilter=True)
    two = map_coordinates(scores[1], coordinates, order=3, mode='nearest', prefilter=True)
    if not np.isfinite(one).all() or not np.isfinite(two).all():
        raise ValueError(f'{slide}: nonfinite response')
    positive = expit((two - one).astype(np.float64))
    display = [np.rint((1 - positive) * 255).astype(np.uint8),
               np.rint(positive * 255).astype(np.uint8)]
    if not np.all((display[0][tissue].astype(np.int16) +
                   display[1][tissue].astype(np.int16)) == 255):
        raise ValueError(f'{slide}: relative responses do not sum to one')
    folder = site / 'camelyon16' / 'bw-projections'
    folder.mkdir(exist_ok=True)
    assets = []
    for cls, gray in enumerate(display, 1):
        rgba = np.stack((gray, gray, gray, tissue.astype(np.uint8) * 255), axis=2)
        destination = folder / f'{slide}-class-{cls:02d}.webp'
        if destination.exists():
            raise FileExistsError(destination)
        Image.fromarray(rgba, 'RGBA').save(destination, 'WEBP', lossless=True, method=6)
        with Image.open(destination) as check:
            decoded = np.asarray(check.convert('RGBA'))
        # WebP canonicalizes RGB under fully transparent pixels. Their color is
        # invisible; require exact alpha and exact visible grayscale instead.
        if not np.array_equal(decoded[:, :, 3], rgba[:, :, 3]) or \
                not np.array_equal(decoded[tissue, :3], rgba[tissue, :3]):
            raise ValueError(f'{slide}: WebP decoded pixels differ')
        assets.append({'path': str(destination.relative_to(site)),
                       'sha256': sha(destination), 'bytes': destination.stat().st_size,
                       'class': cls, 'display_size_wh': [display_w, display_h]})
    body = page.read_text()
    if 'id="bora-bw-projections"' in body:
        raise ValueError(f'{slide}: BW section already installed')
    section = (f'<section id="bora-bw-projections" class="bw-projections">'
               '<h2>Bora black-and-white class responses</h2>'
               '<p>White indicates a stronger relative response from the tumor_093-only Bora head; '
               'black indicates a weaker response. The two displays are a softmax of the smoothed '
               'class scores, not calibrated probabilities, the thresholded BW masks, or independent '
               'Virchow2 measurements at each pixel. Transparent areas are outside reviewed H&amp;E '
               'tissue. The training-selected threshold, 6 µm contour smoothing and 2,000 µm² '
               'cleanup apply to the separate final mask; these displays do not alter scores.</p>'
               '<div class="bw-projection-grid">')
    for cls, label in ((1, 'Non-metastatic tissue'), (2, 'Metastasis-like tissue')):
        section += (f'<figure><img alt="Bora grayscale response for {label.lower()} on reviewed '
                    f'tissue in {slide}" loading="lazy" width="{display_w}" height="{display_h}" '
                    f'src="bw-projections/{slide}-class-{cls:02d}.webp"/>'
                    f'<figcaption>Class {cls} · {label}</figcaption></figure>')
    section += '</div></section>'
    match = re.search(r'<section\b[^>]*\bid="overview"[^>]*>', body)
    if match is None:
        raise ValueError(f'{slide}: overview anchor disappeared')
    page.write_text(body[:match.start()] + section + body[match.start():])
    return {'slide': slide, 'score_receipt': str(score_receipt_path),
            'score_receipt_sha256': entry['score_receipt_sha256'],
            'score_sha256': receipt['smoothed_scores_sha256'],
            'model_sha256': receipt['model_sha256'],
            'reviewed_support': str(support_path),
            'reviewed_support_sha256': entry['reviewed_support_sha256'],
            'crop_4um_yxhw': list(crop), 'tissue_display_pixels': int(tissue.sum()),
            'assets': assets, 'page_sha256': sha(page)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['status'] != 'CAM_BW_SOURCE_FROZEN' or args.receipt.exists():
        raise ValueError('Manifest status differs or receipt already exists')
    rows = [render_one(entry, args.site) for entry in manifest['slides']]
    result = {'status': 'CAM_BW_RESPONSES_RENDERED', 'definition':
              'two-class softmax of smoothed Bora scores sampled to the 4 µm tissue crop; '
              'transparent outside reviewed H&E tissue; not calibrated probabilities or final masks',
              'slide_count': len(rows), 'map_count': 2 * len(rows), 'slides': rows}
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Rendered {len(rows)} slides and {2 * len(rows)} verified BW maps')


if __name__ == '__main__':
    main()
