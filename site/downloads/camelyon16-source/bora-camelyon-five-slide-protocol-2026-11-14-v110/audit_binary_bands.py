"""Independent hash, geometry, and lifecycle audit of one CAM network bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
INPUT = ROOT / 'migration/bora-camelyon-five-slide-protocol-2026-11-14-v103/INPUTS_FROZEN.json'
SOURCE = INPUT.with_name('infer_binary_band.py')
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v103/binary-network-bands'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def starts(length):
    values = list(range(0, length - 512 + 1, 384))
    if values[-1] != length - 512:
        values.append(length - 512)
    return values


def main(run_dir):
    run_dir = Path(run_dir)
    frozen_path = run_dir / 'FULL_RUN_FROZEN.json'
    state_path = run_dir / 'CONTROLLER_STATE.json'
    done_path = run_dir / 'BANDS_COMPLETE.json'
    frozen = json.loads(frozen_path.read_text())
    state = json.loads(state_path.read_text())
    done = json.loads(done_path.read_text())
    inputs = json.loads(INPUT.read_text())
    method, slide = frozen['method'], frozen['slide']
    assert state['status'] == 'COMPLETE' and state['active'] is None
    assert state['frozen_sha256'] == sha(frozen_path)
    assert done['status'] == 'BINARY_NETWORK_SOURCE_ONLY_BANDS_COMPLETE'
    assert done['controller_state_sha256'] == sha(state_path)
    assert (done['method'], done['slide']) == (method, slide)
    assert sha(SOURCE) == frozen['band_source_sha256'] == inputs['source_sha256']['infer_band']
    assert sha(INPUT) == frozen['band_inputs_sha256']
    assert sha(inputs['models'][method]['checkpoint_path']) == frozen['model_checkpoint_sha256']
    assert frozen['campaign_bytes_before'] + frozen['new_output_bytes_bound'] <= 200*(1<<30)
    assert not state['test_reference_opened'] and not state['online_publication']

    unit = frozen['service']
    active = subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'], text=True).strip()
    assert active != 'active', active
    other = subprocess.check_output(['systemctl','--user','list-units','bora-*','--state=running','--no-legend','--no-pager'], text=True).strip()
    assert not other, other
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'], text=True).strip()
    assert not gpu, gpu

    height, width = inputs['slides'][slide]['calibrated_shape_yx']
    sy, sx = np.arange(4, height, 8), np.arange(4, width, 8)
    tops, lefts = starts(height), starts(width)
    groups = (len(tops)+3)//4
    assert len(state['completed']) == done['bands'] == groups
    assert [x['band'] for x in state['completed']] == list(range(groups))
    details = []
    output_bytes = 0
    for band, entry in enumerate(state['completed']):
        folder = OUT / method / slide / f'band-{band:04d}'
        receipt_path = folder / 'BAND_COMPLETE.json'
        receipt = json.loads(receipt_path.read_text())
        assert sha(receipt_path) == entry['receipt_sha256']
        assert (receipt['method'], receipt['slide'], receipt['band']) == (method, slide, band)
        assert receipt['status'] == 'CAM_BINARY_NETWORK_SOURCE_ONLY_BAND_COMPLETE'
        assert receipt['source_sha256'] == sha(SOURCE)
        assert receipt['input_freeze_sha256'] == sha(INPUT)
        assert receipt['checkpoint_sha256'] == frozen['model_checkpoint_sha256']
        assert receipt['calibrated_image_sha256'] == inputs['slides'][slide]['calibrated_image_sha256']
        assert not receipt['expert_reference_opened']
        rows = tops[band*4:(band+1)*4]
        y0 = int(np.searchsorted(sy, rows[0]))
        y1 = int(np.searchsorted(sy, rows[-1]+512))
        assert receipt['comparison_row_start'] == y0 and receipt['comparison_row_stop'] == y1
        assert receipt['selected_patch_tops'] == rows
        assert receipt['patches'] == len(rows)*len(lefts)
        score_path, count_path = Path(receipt['score_path']), Path(receipt['count_path'])
        assert sha(score_path) == receipt['score_sha256']
        assert sha(count_path) == receipt['count_sha256']
        scores = np.load(score_path, mmap_mode='r', allow_pickle=False)
        counts = np.load(count_path, allow_pickle=False)
        assert scores.shape == (2,y1-y0,len(sx)) and scores.dtype == np.float32
        assert counts.shape == (y1-y0,len(sx)) and counts.dtype == np.uint8
        assert np.isfinite(scores).all()
        expected = np.zeros_like(counts)
        for top in rows:
            ay = int(np.searchsorted(sy[y0:y1], top))
            by = int(np.searchsorted(sy[y0:y1], top+512))
            for left in lefts:
                ax = int(np.searchsorted(sx,left))
                bx = int(np.searchsorted(sx,left+512))
                expected[ay:by,ax:bx] += 1
        assert np.array_equal(counts,expected)
        assert counts.min() == receipt['min_patch_coverage_in_band']
        assert counts.max() == receipt['max_patch_coverage_in_band']
        output_bytes += score_path.stat().st_size+count_path.stat().st_size+receipt_path.stat().st_size
        details.append({'band':band,'receipt_sha256':sha(receipt_path),'rows':[y0,y1],
                        'count_min':int(counts.min()),'count_max':int(counts.max())})
    assert output_bytes == state['new_output_bytes']
    assert output_bytes <= frozen['new_output_bytes_bound']
    assert state['latest_cleanup']['status'] == 'VERIFIED'
    record = {'status':'VERIFIED','format':'cam-binary-bands-audit-v1',
              'method':method,'slide':slide,'bands':groups,'details':details,
              'output_bytes':output_bytes,'frozen_sha256':sha(frozen_path),
              'controller_state_sha256':sha(state_path),
              'terminal_service_state':active,'terminal_gpu_contexts':[],
              'exact_patch_coverage_recomputed':True,
              'test_reference_opened':False,'online_publication':False}
    path = run_dir / 'BANDS_AUDIT_VERIFIED.json'
    partial = path.with_name(path.name+'.partial')
    partial.write_text(json.dumps(record,indent=2)+'\n')
    os.replace(partial,path)
    print(json.dumps({'status':'VERIFIED','method':method,'slide':slide,'bands':groups,
                      'output_bytes':output_bytes,'audit_sha256':sha(path)}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    main(args.run_dir)
