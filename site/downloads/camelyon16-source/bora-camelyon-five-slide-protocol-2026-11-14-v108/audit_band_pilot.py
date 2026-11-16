"""Independent source-only audit of the first binary CAM WSI network band."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch import nn

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
INPUT = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v103/INPUTS_FROZEN.json'
FOLDER = DATA / 'external-camelyon16-five-slide-2026-11-14-v103/binary-network-bands/unet/tumor_091/band-0008'
MODEL_CODE = ROOT / 'migration/benchmark1-eight-class-comparator-refit-2026-11-12-v2/models.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def starts(length):
    values = list(range(0, length - 512 + 1, 384))
    if values[-1] != length - 512:
        values.append(length - 512)
    return values


def main():
    frozen = json.loads(INPUT.read_text())
    receipt_path = FOLDER / 'BAND_COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    slide = frozen['slides']['tumor_091']
    model = frozen['models']['unet']
    assert receipt['status'] == 'CAM_BINARY_NETWORK_SOURCE_ONLY_BAND_COMPLETE'
    assert (receipt['method'], receipt['slide'], receipt['band']) == ('unet', 'tumor_091', 8)
    assert receipt['input_freeze_sha256'] == sha(INPUT)
    assert receipt['checkpoint_sha256'] == model['checkpoint_sha256'] == sha(model['checkpoint_path'])
    assert receipt['calibrated_image_sha256'] == slide['calibrated_image_sha256']
    assert receipt['source_sha256'] == frozen['source_sha256']['infer_band']
    assert receipt['grid_source_sha256'] == frozen['source_sha256']['binary_grid']
    assert not receipt['expert_reference_opened']
    for name in ('score', 'count'):
        assert sha(receipt[name + '_path']) == receipt[name + '_sha256']

    height, width = slide['calibrated_shape_yx']
    sy = np.arange(4, height, 8)
    sx = np.arange(4, width, 8)
    patch_rows = starts(height)[32:36]
    patch_cols = starts(width)
    y0 = int(np.searchsorted(sy, patch_rows[0]))
    y1 = int(np.searchsorted(sy, patch_rows[-1] + 512))
    assert receipt['selected_patch_tops'] == patch_rows
    assert receipt['comparison_row_start'] == y0
    assert receipt['comparison_row_stop'] == y1
    assert receipt['patches'] == len(patch_rows) * len(patch_cols)
    scores = np.load(receipt['score_path'], allow_pickle=False)
    counts = np.load(receipt['count_path'], allow_pickle=False)
    assert scores.dtype == np.float32 and counts.dtype == np.uint8
    assert scores.shape == (2, y1-y0, len(sx))
    assert counts.shape == (y1-y0, len(sx))
    assert np.isfinite(scores).all()
    expected = np.zeros_like(counts)
    for top in patch_rows:
        ay = np.searchsorted(sy[y0:y1], top)
        by = np.searchsorted(sy[y0:y1], top+512)
        for left in patch_cols:
            ax = np.searchsorted(sx, left)
            bx = np.searchsorted(sx, left+512)
            expected[ay:by, ax:bx] += 1
    assert np.array_equal(counts, expected)

    # One patch is rerun from the frozen checkpoint on CPU. Mixed-precision GPU
    # and full-precision CPU logits need not agree bitwise; check sampled values.
    import sys
    sys.path.insert(0, str(MODEL_CODE.parent))
    from models import make_model
    torch.set_num_threads(4)
    network = make_model('unet')
    old_head = network.head
    network.head = nn.Conv2d(old_head.in_channels, 2, 1)
    network.load_state_dict(torch.load(model['checkpoint_path'], map_location='cpu', weights_only=True), strict=True)
    network.eval()
    top, left = patch_rows[0], patch_cols[0]
    with rasterio.open(slide['calibrated_image_path']) as image:
        assert (image.height, image.width, image.count) == (height, width, 3)
        patch = image.read(window=Window(left, top, 512, 512))
    tensor = torch.from_numpy(np.ascontiguousarray(patch)).float()[None] / 255
    with torch.no_grad():
        logits = network(tensor)[0].numpy()
    assert logits.shape == (2, 512, 512) and np.isfinite(logits).all()
    # No single patch value can be extracted from the accumulated score when
    # patches overlap, so compare only at the corner sampled pixel with count 1.
    yy = int(np.searchsorted(sy[y0:y1], top))
    xx = int(np.searchsorted(sx, left))
    assert counts[yy, xx] == 1
    local_y, local_x = sy[y0+yy]-top, sx[xx]-left
    cpu = logits[:, local_y, local_x]
    gpu = scores[:, yy, xx]
    assert np.allclose(cpu, gpu, rtol=0.02, atol=0.02), (cpu.tolist(), gpu.tolist())

    output = {'status': 'VERIFIED', 'method': 'unet', 'slide': 'tumor_091',
              'band': 8, 'receipt_sha256': sha(receipt_path),
              'score_sha256': sha(receipt['score_path']),
              'count_sha256': sha(receipt['count_path']),
              'count_min': int(counts.min()), 'count_max': int(counts.max()),
              'exact_coverage_recomputed': True,
              'cpu_single_patch_sample': cpu.tolist(),
              'gpu_band_sample': gpu.tolist(),
              'test_reference_opened': False, 'online_publication': False}
    path = HERE / 'BINARY_BAND_PILOT_AUDIT_VERIFIED.json'
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(output, indent=2) + '\n')
    os.replace(partial, path)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
