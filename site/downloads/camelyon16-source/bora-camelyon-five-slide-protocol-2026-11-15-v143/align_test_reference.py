"""Align the locked official CAM test codes to audited 4-um pixel centres.

Runs only after the source-only methods and local-only scorer are frozen. Code
0 is unscored, 1 is nonmetastatic tissue, and 2 is metastasis. The routine
never changes the official semantic codes or uses H&E support as reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
TEST = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
HEADER = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v23/HEADER_RESOURCE_SCREEN.json'
GATE = ROOT / 'migration/bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-15-v143/test-references'
BLOCK = 256


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def centre_indices(first, last, native, unpadded):
    # Comparison pixel d samples the calibrated 0.5-um pixel at 8*d+4.
    d = np.arange(first, last, dtype=np.float64)
    return np.clip(np.floor((8*d+4.5)*native/unpadded).astype(np.int64), 0, native-1)


def main(slide, official):
    gate = json.loads(GATE.read_text())
    assert gate['status'] == 'LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD'
    assert gate['local_three_slide_scoring_allowed'] and not gate['new_cam_online_publication_allowed']
    cohort = json.loads(TEST.read_text())
    assert slide in cohort['test_ids'] and cohort['official_binary_label_contract'].startswith('mask 0 unscored')
    row = next(v for v in cohort['selected_official_metadata'] if v['slide_id'] == slide)
    header = next(v for v in json.loads(HEADER.read_text())['rows'] if v['slide_id'] == slide)
    assert official.stat().st_size == row['mask_object_bytes']
    md5 = hashlib.md5()
    with official.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            md5.update(chunk)
    assert md5.hexdigest() == row['mask_object_md5']
    native_h, native_w = header['native_shape_yx']
    h, w = header['calibrated_unpadded_shape_yx']
    ph, pw = header['calibrated_padded_shape_yx']
    assert ph % 8 == pw % 8 == 0
    shape = (ph//8, pw//8)
    case = OUT / slide
    if case.exists():
        raise FileExistsError('No in-place official test-reference retry')
    case.mkdir(parents=True)
    partial = case / 'official-code-4um.partial.dat'
    final = case / 'official-code-4um.uint8.dat'
    started = time.monotonic()
    state = {'status': 'RUNNING', 'slide': slide, 'official_path': str(official),
             'official_sha256': digest(official), 'comparison_shape_yx': list(shape),
             'comparison_spacing_um_xy': [4.0, 4.0], 'test_reference_opened': True,
             'online_publication': False}
    (case/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    counts = np.zeros(3, dtype=np.int64)
    blocks = 0
    try:
        output = np.memmap(partial, dtype=np.uint8, mode='w+', shape=shape)
        output[:] = 0
        with rasterio.Env(GDAL_CACHEMAX=64), rasterio.open(official) as source:
            assert (source.height, source.width, source.count, source.dtypes[0]) == (native_h, native_w, 1, 'uint8')
            for y0 in range(0, shape[0], BLOCK):
                if time.monotonic()-started > 2400:
                    raise TimeoutError('Bounded official reference alignment deadline')
                y1 = min(y0+BLOCK, shape[0])
                valid_y1 = min(y1, (h-5)//8+1)
                if valid_y1 <= y0:
                    continue
                ys = centre_indices(y0, valid_y1, native_h, h)
                for x0 in range(0, shape[1], BLOCK):
                    x1 = min(x0+BLOCK, shape[1])
                    valid_x1 = min(x1, (w-5)//8+1)
                    if valid_x1 <= x0:
                        continue
                    xs = centre_indices(x0, valid_x1, native_w, w)
                    lowy, highy = int(ys.min()), int(ys.max())
                    lowx, highx = int(xs.min()), int(xs.max())
                    raw = source.read(1, window=Window(lowx,lowy,highx-lowx+1,highy-lowy+1))
                    sampled = raw[(ys-lowy)[:,None],(xs-lowx)[None,:]]
                    if sampled.max(initial=0) > 2:
                        raise ValueError('Official mask has a code outside 0/1/2')
                    output[y0:valid_y1,x0:valid_x1] = sampled
                    counts += np.bincount(sampled.ravel(),minlength=3)[:3]
                    blocks += 1
        output.flush()
        del output
        os.replace(partial, final)
        counts[0] += shape[0]*shape[1]-int(counts.sum())
        assert counts.sum() == shape[0]*shape[1]
        if slide.startswith('tumor') and counts[2] == 0:
            raise ValueError('Positive slide has no official metastasis pixels')
        if slide.startswith('normal') and counts[2] != 0:
            raise ValueError('Benign control has official metastasis pixels')
        state.update(status='COMPLETE', aligned_path=str(final), aligned_sha256=digest(final),
                     class_counts_0_1_2=counts.tolist(), blocks=blocks,
                     source_coordinate_rule='4-um index d samples calibrated 0.5-um index 8*d+4; nearest native centre floor((8*d+4.5)*native/unpadded)',
                     seconds=time.monotonic()-started,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (case/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'slide':slide,'status':'COMPLETE','counts':counts.tolist()}),flush=True)
    except BaseException as error:
        state.update(status='FAILED_LATCHED_PRESERVE_PARTIAL',error=repr(error),blocks=blocks)
        (case/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--slide', choices=('tumor_084','tumor_091','normal_019'), required=True)
    p.add_argument('--official', type=Path, required=True)
    a = p.parse_args()
    main(a.slide,a.official)
