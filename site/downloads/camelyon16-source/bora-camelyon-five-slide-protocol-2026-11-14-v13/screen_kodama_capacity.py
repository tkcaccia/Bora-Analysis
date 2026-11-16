"""Read existing H&E-only 4-µm supports for a joint-tile capacity proxy.

This is not exact 0.5-µm tile admission or a KODAMA fit. It opens no CAM
source image, feature array, or expert annotation.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
BASE = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16-pair-2026-11-14-v88/common-support-4um')
OUT = ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v13/KODAMA_CAPACITY_SCREEN.json'
EXPECTED = {
    'normal_093': '7a378584b2c96f0a09122501b7fcc5c9595acf2b142d99e1c4aa1c30d3a6ce88',
    'tumor_093': '768501a9882f762dc3f5f71514718c072c55faf7b6fa58e1f5c4d7c0ac245c65',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    assert json.loads((ROOT/'migration/bora-r-camelyon-2026-11-14-v1/PERMISSION_GATE_2026-11-14.json').read_text())['status'] == 'PENDING'
    rows = []
    for slide, expected in EXPECTED.items():
        path = BASE/slide/'common-tissue-4um.tif'
        assert sha(path) == expected
        with rasterio.open(path) as raster:
            image = raster.read(1)
        mask = (image > 0).astype(np.uint8)
        height, width = mask.shape
        assert height % 28 == 0 and width % 28 == 0
        # A 224-pixel tile at 0.5 µm/px spans 112 µm, or 28 × 28 pixels on
        # the already audited 4-µm support raster.
        area = mask.reshape(height // 28, 28, width // 28, 28).sum(axis=(1, 3))
        rows.append({
            'slide': slide, 'support_4um_sha256': expected,
            'support_shape_yx': [height, width],
            'proxy_224px_tile_grid_shape_yx': list(area.shape),
            'proxy_tiles_gt90_percent_tissue': int((area >= 706).sum()),
            'proxy_tiles_100_percent_tissue': int((area == 784).sum()),
            'proxy_tiles_any_tissue': int((area > 0).sum()),
        })
    total = sum(row['proxy_tiles_gt90_percent_tissue'] for row in rows)
    result = {
        'format': 'bora-camelyon16-kodama-capacity-proxy-v1',
        'status': 'SOURCE_ONLY_APPROXIMATE_CAPACITY_SCREEN',
        'support_resolution_um_per_pixel': 4.0,
        'calibrated_rgb_resolution_um_per_pixel': 0.5,
        'kodama_tile_edge_calibrated_px': 224,
        'strict_tissue_rule': 'greater than 90 percent tissue',
        'current_package_default_max_joint_tiles': 10000,
        'rows': rows,
        'joint_proxy_tiles_gt90_percent_tissue': total,
        'default_cap_exceeded_in_proxy': total > 10000,
        'caveat': '4-µm raster block counts approximate the exact 0.5-µm raster admission near support boundaries; this is not a KODAMA job admission or result',
        'cam_images_features_expert_references_opened': False,
        'cam_specific_training_or_scoring_performed': False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(OUT)
    print(json.dumps({'joint_proxy_tiles': total, 'default_cap_exceeded': total > 10000}))


if __name__ == '__main__':
    main()
