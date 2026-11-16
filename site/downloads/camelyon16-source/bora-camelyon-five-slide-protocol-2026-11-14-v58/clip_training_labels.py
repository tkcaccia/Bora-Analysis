"""Intersect the two official CAM training masks with frozen H&E-only support.

Code 0 remains unscored. This is a training input, not a replacement for the
official reference. The test references are never read by this program.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SOURCE = DATA / 'external-camelyon16-five-slide-2026-11-14-v51/training-class-masks'
SUPPORT = DATA / 'external-camelyon16-pair-2026-11-13-v54/he-support'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v58/training-labels'
DECISION = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v57/TRAINING_DOMAIN_DECISION_FROZEN.json'
MAX_SECONDS = 900


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main(slide):
    if slide not in ('tumor_093', 'normal_093'):
        raise ValueError('Only the two frozen CAM training slides are admitted')
    decision = json.loads(DECISION.read_text())
    if decision['status'] != 'FROZEN_WITH_COVERAGE_LIMITATION':
        raise ValueError('Training-domain decision differs')
    state_path = SOURCE / slide / 'STATE.json'
    state = json.loads(state_path.read_text())
    audit = ROOT / f'bora-camelyon-five-slide-protocol-2026-11-14-v54/{slide.upper()}_MASK_ALIGNMENT_AUDIT_VERIFIED.json'
    if state['status'] != 'COMPLETE' or json.loads(audit.read_text())['status'] != 'VERIFIED':
        raise ValueError('Training mask is not independently audited')
    geo_path = SUPPORT / slide / 'tissue-area.geojson'
    geo = json.loads(geo_path.read_text())
    if geo['source_image_sha256'] != state['calibrated_image_sha256'] or geo['image_shape_yx'] != state['padded_shape_yx']:
        raise ValueError('H&E-only support registration differs')
    shapes = [(f['geometry'], 1) for f in geo['features']]
    ph, pw = state['padded_shape_yx']
    case = OUT / slide
    if case.exists():
        raise FileExistsError(case)
    case.mkdir(parents=True)
    inputs = {'slide': slide, 'decision_sha256': digest(DECISION),
              'official_aligned_mask_sha256': state['class_mask_sha256'],
              'official_alignment_audit_sha256': digest(audit),
              'source_only_support_sha256': digest(geo_path),
              'calibrated_image_sha256': state['calibrated_image_sha256'],
              'test_reference_opened': False, 'online_publication': False}
    atomic_json(case / 'INPUTS_FROZEN.json', inputs)
    partial = case / 'training-classes-in-support.partial.tif'
    final = case / 'training-classes-in-support.tif'
    began = time.monotonic()
    counts = np.zeros(3, np.int64)
    outside = np.zeros(3, np.int64)
    try:
        profile = {'driver': 'GTiff', 'height': ph, 'width': pw, 'count': 1,
                   'dtype': 'uint8', 'tiled': True, 'blockxsize': 512,
                   'blockysize': 512, 'compress': 'ZSTD', 'zstd_level': 7,
                   'BIGTIFF': 'YES', 'nodata': 0}
        with rasterio.Env(GDAL_CACHEMAX=64), rasterio.open(state['class_mask_path']) as source, rasterio.open(partial, 'w', **profile) as target:
            if (source.height, source.width) != (ph, pw):
                raise ValueError('Aligned mask dimensions differ')
            for y0 in range(0, ph, 512):
                if time.monotonic() - began > MAX_SECONDS:
                    raise TimeoutError('Bounded training-label clipping deadline reached')
                height = min(512, ph - y0)
                window = Window(0, y0, pw, height)
                labels = source.read(1, window=window)
                if labels.max(initial=0) > 2:
                    raise ValueError('Unknown official class code')
                support = rasterize(shapes, out_shape=(height, pw),
                                    transform=Affine.translation(0, y0),
                                    fill=0, dtype=np.uint8).astype(bool)
                counts += np.bincount(labels[support], minlength=3)[:3]
                outside += np.bincount(labels[~support], minlength=3)[:3]
                labels[~support] = 0
                target.write(labels, 1, window=window)
        os.replace(partial, final)
        result = {'format': 'bora-cam-training-support-clipped-mask-v1',
                  'status': 'COMPLETE', 'slide_id': slide,
                  'training_mask_path': str(final), 'training_mask_sha256': digest(final),
                  'training_mask_bytes': final.stat().st_size,
                  'class_counts_inside_support_0_1_2': counts.tolist(),
                  'official_class_counts_outside_support_0_1_2': outside.tolist(),
                  'input_sha256': digest(case / 'INPUTS_FROZEN.json'),
                  'wall_seconds': time.monotonic() - began,
                  'peak_host_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'test_reference_opened': False, 'online_publication': False}
        atomic_json(case / 'STATE.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        atomic_json(case / 'FAILED.json', {'status': 'FAILED', 'slide': slide,
                    'error': repr(error), 'partial_path': str(partial),
                    'partial_exists': partial.exists(),
                    'test_reference_opened': False, 'online_publication': False})
        raise


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: clip_training_labels.py tumor_093|normal_093')
    main(sys.argv[1])
