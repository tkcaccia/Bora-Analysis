#!/usr/bin/env python3
"""Header-only geometry and exact dense-feature storage screen for CAM tests."""
import hashlib
import json
import math
import os
from pathlib import Path
import re

import tifffile

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
AUDIT = ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v22/SOURCE_BYTES_AUDIT_VERIFIED.json'
HERE = Path(__file__).parent
OUT = HERE/'HEADER_RESOURCE_SCREEN.json'
VOLUME = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
CAM_CAP = 200 * (1 << 30)
FREE_RESERVE = 100 * (1 << 30)
CAM_BASELINE = 139_993_686_016


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    audited = json.loads(AUDIT.read_text())
    assert audited['status'] == 'VERIFIED'
    assert [r['slide_id'] for r in audited['sources']] == [
        'tumor_084', 'tumor_091', 'normal_019']
    rows = []
    for source in audited['sources']:
        path = Path(source['source_path'])
        with tifffile.TiffFile(path) as tif:
            first = tif.pages[0]
            second = tif.pages[1]
            native_y, native_x = first.shape[:2]
            assert len(first.shape) == 3 and first.shape[2] == 3
            assert len(second.shape) == 3 and second.shape[2] == 3
            assert .45 < second.shape[0]/native_y < .55
            assert .45 < second.shape[1]/native_x < .55
            match = re.search(r'DICOM_PIXEL_SPACING[^>]*>&quot;([0-9.]+)&quot; &quot;([0-9.]+)&quot;',
                              first.description or '')
            assert match is not None
            mpp_y, mpp_x = [float(x) * 1000 for x in match.groups()]
            assert .20 < mpp_y < .30 and .20 < mpp_x < .30
            calibrated_y = round(native_y * mpp_y / .5)
            calibrated_x = round(native_x * mpp_x / .5)
            padded_y = math.ceil(calibrated_y / 896) * 896
            padded_x = math.ceil(calibrated_x / 896) * 896
            scales = []
            for factor in (1, 2, 4):
                gy = padded_y // (14 * factor)
                gx = padded_x // (14 * factor)
                scales.append({'factor': factor, 'grid_yx': [gy, gx],
                               'dense_full1280_float16_bytes': gy * gx * 1280 * 2})
            rows.append({'slide_id': source['slide_id'],
                         'source_sha256': source['source_sha256'],
                         'native_shape_yx': [native_y, native_x],
                         'scanner_level_1_shape_yx': list(second.shape[:2]),
                         'native_mpp_xy': [mpp_x, mpp_y],
                         'calibrated_unpadded_shape_yx': [calibrated_y, calibrated_x],
                         'calibrated_padded_shape_yx': [padded_y, padded_x],
                         'scales': scales,
                         'dense_feature_total_bytes': sum(s['dense_full1280_float16_bytes'] for s in scales)})
    dense = sum(r['dense_feature_total_bytes'] for r in rows)
    source_total = sum(r['bytes'] for r in audited['sources'])
    free = os.statvfs(VOLUME)
    free_bytes = free.f_bavail * free.f_frsize
    result = {
        'format': 'bora-camelyon16-three-source-header-space-screen-v1',
        'status': 'DENSE_FEATURE_STORAGE_BLOCKED_SPARSE_TISSUE_CACHE_REQUIRED',
        'source_audit_sha256': hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
        'rows': rows,
        'selected_source_bytes': source_total,
        'dense_feature_total_bytes': dense,
        'baseline_campaign_bytes_before_selected_download': CAM_BASELINE,
        'campaign_cap_bytes': CAM_CAP,
        'maximum_additional_campaign_bytes_after_selected_sources': CAM_CAP - CAM_BASELINE - source_total,
        'projected_dense_campaign_bytes': CAM_BASELINE + source_total + dense,
        'dense_exceeds_campaign_cap': CAM_BASELINE + source_total + dense > CAM_CAP,
        'free_bytes_now': free_bytes,
        'free_reserve_bytes': FREE_RESERVE,
        'input_image_pixels_opened': False,
        'expert_reference_opened': False,
        'admission_conclusion': 'Do not allocate dense test feature grids. Prepare source-only H&E tissue support and use sparse, full-1280 tissue-token cache with exact registration and peak-space audit.',
    }
    assert result['dense_exceeds_campaign_cap']
    temp = OUT.with_suffix('.json.partial')
    with temp.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, OUT)
    print(json.dumps({'status': result['status'],
                      'dense_feature_gib': round(dense/(1<<30), 3),
                      'maximum_additional_gib': round(result['maximum_additional_campaign_bytes_after_selected_sources']/(1<<30), 3)}))


if __name__ == '__main__':
    main()
