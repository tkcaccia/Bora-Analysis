#!/usr/bin/env python3
"""Independently verify one completed full-channel sparse CAM feature scale."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
V28 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v28'
V35 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v35'
WORKER = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v26/sparse_feature_worker.py'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide', required=True,
                        choices=('normal_019', 'tumor_091', 'tumor_084'))
    parser.add_argument('--factor', required=True, type=int, choices=(1, 2, 4))
    parser.add_argument('--service', required=True)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads((V28 / 'SPARSE_INPUTS_FROZEN.json').read_text())
    audit = json.loads((V35 / 'SPARSE_ADMISSION_AUDIT_VERIFIED.json').read_text())
    launch = json.loads(args.freeze.read_text())
    assert frozen['status'] == 'SOURCE_ONLY_INPUTS_FROZEN_STORAGE_ADMITTED'
    assert audit['status'] == 'VERIFIED'
    assert sha(WORKER) == frozen['worker_source_sha256'] == launch['worker_source_sha256']
    assert launch['status'] == 'FROZEN_BEFORE_GPU'
    assert (launch['slide_id'], launch['factor']) == (args.slide, args.factor)
    case = next(c for c in frozen['cases'] if c['slide_id'] == args.slide)
    scale = next(s for s in case['scales'] if s['factor'] == args.factor)
    spec_path = Path(scale['spec_path'])
    spec = json.loads(spec_path.read_text())
    assert sha(spec_path) == scale['spec_sha256'] == launch['spec_sha256']
    assert spec['image_sha256'] == case['image_sha256']
    assert spec['tissue_support_sha256'] == case['support_sha256']
    assert sha(spec['image']) == spec['image_sha256']
    assert sha(spec['tissue_support']) == spec['tissue_support_sha256']
    assert sha(spec['eligible_path']) == case['eligible_sha256']
    assert sha(Path(spec['model_dir']) / 'model.safetensors') == spec['model_checkpoint_sha256']
    assert sha(Path(spec['model_dir']) / 'config.json') == spec['model_config_sha256']
    complete_path = Path(spec['output_dir']) / 'FEATURES_COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    assert complete['status'] == 'COMPLETE'
    assert complete['source_spec_sha256'] == sha(spec_path)
    assert complete['factor'] == args.factor and complete['channels'] == 1280
    assert complete['dtype'] == 'float16'
    base = np.load(case['eligible_path'], allow_pickle=False)
    assert base.dtype == np.bool_ and list(base.shape) == case['base_grid_shape_yx']
    factor = args.factor
    if factor == 1:
        grid = base
    else:
        gy, gx = base.shape
        grid = base.reshape(gy // factor, factor, gx // factor, factor).any(axis=(1, 3))
    assert list(grid.shape) == scale['grid_yx']
    assert complete['shape_yxc'] == [grid.shape[0] * 16, grid.shape[1] * 16, 1280]
    assert complete['rows'] == len(complete['row_receipts']) == grid.shape[0]
    total_tiles = total_bytes = 0
    for row, receipt in enumerate(complete['row_receipts']):
        path = complete_path.parent / f'row-{row:04d}.json'
        assert json.loads(path.read_text()) == receipt
        expected = np.flatnonzero(grid[row]).astype(np.int32)
        assert receipt['row'] == row and receipt['tile_columns'] == len(expected)
        if len(expected):
            columns = Path(receipt['column_path'])
            tiles = Path(receipt['tile_path'])
            assert columns.parent == tiles.parent == complete_path.parent
            assert sha(columns) == receipt['column_sha256']
            assert sha(tiles) == receipt['tile_sha256']
            assert np.array_equal(np.load(columns, allow_pickle=False), expected)
            data = np.load(tiles, mmap_mode='r', allow_pickle=False)
            assert data.dtype == np.float16 and data.shape == (len(expected), 16, 16, 1280)
            assert np.isfinite(data).all()
            total_bytes += columns.stat().st_size + tiles.stat().st_size
        else:
            assert receipt['tile_path'] is None and receipt['column_path'] is None
        total_tiles += len(expected)
    assert total_tiles == complete['eligible_tiles'] == complete['persisted_tiles'] == scale['eligible_tiles']
    assert total_bytes == complete['persisted_bytes'] <= spec['max_feature_bytes']
    status = subprocess.check_output(['systemctl', '--user', 'show', args.service,
                                      '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
                                      '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'ControlGroup'],
                                     text=True)
    fields = dict(line.split('=', 1) for line in status.splitlines())
    assert fields['ActiveState'] == 'inactive' and fields['SubState'] == 'dead'
    assert fields['MainPID'] == '0' and fields['Result'] == 'success'
    assert fields['ExecMainStatus'] == '0' and fields['ControlGroup'] == ''
    gpu = subprocess.check_output(['timeout', '8', 'nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    assert not gpu.strip()
    result = {'format': 'bora-cam-sparse-production-scale-audit-v1', 'status': 'VERIFIED',
              'slide_id': args.slide, 'factor': factor,
              'source_only_frozen_sha256': sha(V28 / 'SPARSE_INPUTS_FROZEN.json'),
              'admission_audit_sha256': sha(V35 / 'SPARSE_ADMISSION_AUDIT_VERIFIED.json'),
              'launch_freeze_sha256': sha(args.freeze), 'source_spec_sha256': sha(spec_path),
              'worker_source_sha256': sha(WORKER),
              'feature_receipt_sha256': sha(complete_path), 'rows': grid.shape[0],
              'persisted_tiles': total_tiles, 'persisted_bytes': total_bytes,
              'all_tile_rows_columns_hashes_and_finite_tokens_verified': True,
              'terminal_service': fields, 'assigned_gpu_contexts_after': [],
              'peak_host_rss_kb': complete['peak_host_rss_kb'],
              'peak_cuda_reserved_bytes': complete['peak_cuda_reserved_bytes'],
              'wall_seconds': complete['wall_seconds'],
              'expert_reference_opened': False, 'online_publication': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.partial')
    with tmp.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, args.output)
    print(json.dumps({'status': 'VERIFIED', 'slide': args.slide,
                      'factor': factor, 'tiles': total_tiles,
                      'wall_seconds': complete['wall_seconds']}))

if __name__ == '__main__':
    main()
