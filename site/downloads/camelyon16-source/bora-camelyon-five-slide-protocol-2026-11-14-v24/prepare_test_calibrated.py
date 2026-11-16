#!/usr/bin/env python3
"""Serial source-only 0.5-µm RGB calibration for frozen CAM test slides."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import pyvips

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
V22 = ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v22/SOURCE_BYTES_AUDIT_VERIFIED.json'
V23 = ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v23/HEADER_RESOURCE_SCREEN.json'
OUT = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16-five-slide-2026-11-14-v24/calibrated')
STATE = Path(__file__).parent/'CALIBRATION_STATE.json'
CAP = 200 * (1 << 30)
RESERVE = 100 * (1 << 30)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix(path.suffix + '.partial')
    with tmp.open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def storage_check(screen, expected_new_bytes):
    # This conservative admission treats compressed TIFFs as uncompressed
    # and permits an additional output-sized transient write.
    if screen['baseline_campaign_bytes_before_selected_download'] + \
       screen['selected_source_bytes'] + 2 * expected_new_bytes > CAP:
        raise OSError('Worst-case calibrated images plus temporary bytes exceed campaign cap')
    stat = os.statvfs('/mnt/sata_ssd/ChatGPT/Bora-Data')
    if stat.f_bavail * stat.f_frsize - 2 * expected_new_bytes < RESERVE:
        raise OSError('Calibrated images could breach 100-GiB free reserve')


def run():
    if STATE.exists() or OUT.exists():
        raise FileExistsError('Versioned calibration output exists')
    audit = json.loads(V22.read_text())
    screen = json.loads(V23.read_text())
    if audit['status'] != 'VERIFIED' or \
       screen['status'] != 'DENSE_FEATURE_STORAGE_BLOCKED_SPARSE_TISSUE_CACHE_REQUIRED' or \
       screen['source_audit_sha256'] != sha(V22):
        raise ValueError('Source or header audit differs')
    source = {r['slide_id']: r for r in audit['sources']}
    rows = screen['rows']
    assert [r['slide_id'] for r in rows] == ['tumor_084','tumor_091','normal_019']
    worst = sum(r['calibrated_padded_shape_yx'][0] * r['calibrated_padded_shape_yx'][1] * 3
                for r in rows)
    storage_check(screen, worst)
    OUT.mkdir(parents=True)
    state = {'format': 'bora-cam-test-calibration-v1', 'status': 'RUNNING',
             'source_audit_sha256': sha(V22), 'header_screen_sha256': sha(V23),
             'pid': os.getpid(), 'started_unix': time.time(),
             'maximum_uncompressed_output_bytes': worst,
             'completed': [], 'expert_reference_opened': False,
             'online_publication': False}
    atomic(STATE, state)
    try:
        for row in rows:
            sid = row['slide_id']
            origin = Path(source[sid]['source_path'])
            if sha(origin) != source[sid]['source_sha256']:
                raise ValueError(f'{sid} source SHA256 differs')
            directory = OUT/sid
            directory.mkdir()
            native = pyvips.Image.tiffload(str(origin), page=0, access='sequential')
            page = pyvips.Image.tiffload(str(origin), page=1, access='sequential')
            if [native.height,native.width] != row['native_shape_yx'] or \
               [page.height,page.width] != row['scanner_level_1_shape_yx'] or \
               page.bands < 3 or page.format != 'uchar':
                raise ValueError(f'{sid} header differs at decode')
            height,width = row['calibrated_unpadded_shape_yx']
            padded_h,padded_w = row['calibrated_padded_shape_yx']
            rgb = page.extract_band(0,n=3)
            calibrated = rgb.resize(width/rgb.width,
                                    vscale=height/rgb.height,kernel='lanczos3')
            if (calibrated.height,calibrated.width) != (height,width):
                raise ValueError(f'{sid} calibrated dimensions differ')
            padded = calibrated.embed(0,0,padded_w,padded_h,extend='white')
            partial = directory/'calibrated-0p5-padded.partial.tif'
            final = directory/'calibrated-0p5-padded.tif'
            padded.tiffsave(str(partial),tile=True,tile_width=256,tile_height=256,
                            compression='zstd',predictor='horizontal',bigtiff=True,
                            pyramid=False,xres=2000,yres=2000)
            os.replace(partial,final)
            decoded = pyvips.Image.new_from_file(str(final),access='random')
            if (decoded.height,decoded.width,decoded.bands,decoded.format) != \
               (padded_h,padded_w,3,'uchar'):
                raise ValueError(f'{sid} persisted RGB geometry differs')
            overview = directory/'he-overview.jpg'
            decoded.crop(0,0,width,height).thumbnail_image(1600).write_to_file(str(overview),Q=88)
            record = {'slide_id': sid, 'source_sha256': source[sid]['source_sha256'],
                      'source_path': str(origin), 'calibrated_path': str(final),
                      'calibrated_sha256': sha(final), 'calibrated_bytes': final.stat().st_size,
                      'overview_path': str(overview), 'overview_sha256': sha(overview),
                      'native_mpp_xy': row['native_mpp_xy'],
                      'unpadded_shape_yx': [height,width],
                      'padded_shape_yx': [padded_h,padded_w],
                      'calibration': 'scanner pyramid level 1, Lanczos3 to 0.5 um/px, white right/bottom padding to 896 pixels',
                      'expert_reference_opened': False}
            state['completed'].append(record)
            atomic(STATE,state)
            print(f'{sid} calibrated {padded_h}x{padded_w}',flush=True)
        state['status'] = 'COMPLETE'
        state['completed_unix'] = time.time()
        atomic(STATE,state)
    except BaseException as exc:
        state['status'] = 'FAILED_PRESERVE_PARTIAL'
        state['error'] = f'{type(exc).__name__}: {exc}'
        state['failed_unix'] = time.time()
        atomic(STATE,state)
        raise


if __name__ == '__main__':
    run()
