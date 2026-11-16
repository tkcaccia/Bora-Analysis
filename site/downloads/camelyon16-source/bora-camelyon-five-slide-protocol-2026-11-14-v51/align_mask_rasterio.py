#!/usr/bin/env python3
"""Bounded rasterio recovery for the two official CAM training masks only.

The official tiled-LZW mask has sparse zero-byte tiles that libvips cannot
decode on this host. GDAL/rasterio reads them. Output pixel centres are mapped
to the audited calibrated RGB dimensions with nearest-neighbor sampling.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
V19 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
V31 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v31/TRAINING_REFS_STATE.json'
V46 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v46/TRAINING_MASK_SOURCES_AUDIT_VERIFIED.json'
V49 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v49/V42_FAILURE_AND_CLEANUP_VERIFIED.json'
V52 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v52/V48_FAILED_BEFORE_COMPUTE.json'
GATE = ROOT / 'bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v51/training-class-masks'
CAL = {
    'normal_093': DATA / 'external-camelyon16-pair-2026-11-13-v48/calibrated/normal_093/STATE.json',
    'tumor_093': DATA / 'external-camelyon16-pair-2026-11-13-v49/calibrated/tumor_093/STATE.json',
}
BLOCK = 512
MAX_SECONDS = 6500

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()

def atomic(path, value):
    temp = path.with_name(path.name + '.partial')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

def source_indices(start, stop, native_length, target_length):
    """Nearest source centres for target centre pixels on the common frame."""
    dest = np.arange(start, stop, dtype=np.float64)
    selected = np.floor((dest + .5) * native_length / target_length).astype(np.int64)
    return np.clip(selected, 0, native_length - 1)

def run(slide):
    if slide not in CAL:
        raise ValueError('Only frozen CAM training slides are admitted')
    gate = json.loads(GATE.read_text())
    ids = json.loads(V19.read_text())
    refs = json.loads(V31.read_text())
    source_audit = json.loads(V46.read_text())
    failure = json.loads(V49.read_text())
    edge_failure = json.loads(V52.read_text())
    if (gate['status'] != 'LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD' or
        not gate['local_cam_specific_training_allowed'] or
        gate['new_cam_online_publication_allowed'] or
        ids['training_ids'] != ['tumor_093', 'normal_093'] or
        ids['test_ids'] != ['tumor_084', 'tumor_091', 'normal_019'] or
        refs['status'] != 'COMPLETE_SOURCE_BYTES_VERIFIED_NOT_OPENED' or
        source_audit['status'] != 'VERIFIED' or
        failure['status'] != 'FAILED_LATCHED_CLEANUP_VERIFIED' or
        edge_failure['status'] != 'FAILED_BEFORE_COMPUTE_CLEANUP_VERIFIED'):
        raise PermissionError('Frozen local-only CAM source/recovery scope differs')
    reference = next(r for r in refs['completed'] if r['slide_id'] == slide)
    official = Path(reference['mask_path'])
    if sha(official) != reference['sha256']:
        raise ValueError('Official training mask source SHA256 differs')
    cal = json.loads(CAL[slide].read_text())
    if cal['status'] != 'COMPLETE' or sha(cal['target_image']) != cal['target_sha256']:
        raise ValueError('Audited calibrated RGB source differs')
    native_h, native_w = cal['native_shape_yx']
    h, w = cal['unpadded_shape_yx']
    ph, pw, _ = cal['target_shape_yxc']
    assert ph >= h and pw >= w
    case = OUT / slide
    if case.exists():
        raise FileExistsError('Preserve previous case bytes and use a new version')
    case.mkdir(parents=True)
    state = {'format': 'bora-cam-training-mask-0p5-rasterio-v1',
             'status': 'RUNNING', 'slide_id': slide,
             'official_mask_path': str(official),
             'official_mask_sha256': reference['sha256'],
             'official_mask_md5': reference['official_md5'],
             'calibrated_state_path': str(CAL[slide]),
             'calibrated_state_sha256': sha(CAL[slide]),
             'calibrated_image_sha256': cal['target_sha256'],
             'native_shape_yx': [native_h, native_w],
             'unpadded_shape_yx': [h, w], 'padded_shape_yx': [ph, pw],
             'started_unix': time.time(), 'pid': os.getpid(),
             'test_reference_opened': False, 'online_publication': False}
    atomic(case / 'STATE.json', state)
    counts = np.zeros(3, dtype=np.int64)
    blocks_done = 0
    partial = case / 'official-class-0p5.partial.tif'
    final = case / 'official-class-0p5.tif'
    try:
        with rasterio.Env(GDAL_CACHEMAX=64), rasterio.open(official) as source:
            if (source.height, source.width, source.count, source.dtypes[0]) != (native_h, native_w, 1, 'uint8'):
                raise ValueError('Official mask native dimensions or dtype differ')
            profile = {'driver': 'GTiff', 'height': ph, 'width': pw, 'count': 1,
                       'dtype': 'uint8', 'tiled': True, 'blockxsize': BLOCK,
                       'blockysize': BLOCK, 'compress': 'ZSTD', 'zstd_level': 7,
                       'BIGTIFF': 'YES', 'nodata': 0}
            with rasterio.open(partial, 'w', **profile) as output:
                for y0 in range(0, ph, BLOCK):
                    if time.time() - state['started_unix'] >= MAX_SECONDS:
                        raise TimeoutError('Bounded CAM mask alignment deadline reached')
                    block_h = min(BLOCK, ph - y0)
                    y1 = min(y0 + block_h, h)
                    ny = source_indices(y0, y1, native_h, h) if y0 < h else None
                    for x0 in range(0, pw, BLOCK):
                        block_w = min(BLOCK, pw - x0)
                        x1 = min(x0 + block_w, w)
                        block = np.zeros((block_h, block_w), np.uint8)
                        if ny is not None and x0 < w:
                            nx = source_indices(x0, x1, native_w, w)
                            ymin, ymax = int(ny.min()), int(ny.max())
                            xmin, xmax = int(nx.min()), int(nx.max())
                            window = Window(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)
                            raw = source.read(1, window=window)
                            sampled = raw[(ny - ymin)[:, None], (nx - xmin)[None, :]]
                            if sampled.max(initial=0) > 2:
                                raise ValueError('Official mask has unexpected class code')
                            block[:y1 - y0, :x1 - x0] = sampled
                        counts += np.bincount(block.ravel(), minlength=3)[:3]
                        output.write(block, 1, window=Window(x0, y0, block_w, block_h))
                        blocks_done += 1
        with partial.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(partial, final)
        with rasterio.open(final) as check:
            if (check.height, check.width, check.count, check.dtypes[0]) != (ph, pw, 1, 'uint8'):
                raise ValueError('Persisted training mask dimensions differ')
        if counts.sum() != ph * pw or not counts[1]:
            raise ValueError('Class counts or nonmetastatic training tissue differ')
        if slide.startswith('tumor') and not counts[2]:
            raise ValueError('Positive training slide has no annotated metastasis')
        if slide.startswith('normal') and counts[2]:
            raise ValueError('Negative training slide has annotated metastasis')
        state.update(status='COMPLETE', completed_unix=time.time(),
                     class_mask_path=str(final), class_mask_sha256=sha(final),
                     class_mask_bytes=final.stat().st_size,
                     class_counts_0_1_2=counts.tolist(), blocks_done=blocks_done,
                     transform='target centre (x+0.5,y+0.5) mapped to nearest native centre by floor((coord+0.5)*native/target); padded area code 0',
                     peak_host_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic(case / 'STATE.json', state)
        print(json.dumps({'slide_id': slide, 'status': 'COMPLETE',
                          'counts': counts.tolist(), 'bytes': state['class_mask_bytes']}))
    except BaseException as error:
        state.update(status='FAILED_PRESERVE_PARTIAL', failure_type=type(error).__name__,
                     failure=str(error), failed_unix=time.time(),
                     blocks_done=blocks_done)
        atomic(case / 'STATE.json', state)
        raise

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: align_mask_rasterio.py tumor_093|normal_093')
    run(sys.argv[1])
