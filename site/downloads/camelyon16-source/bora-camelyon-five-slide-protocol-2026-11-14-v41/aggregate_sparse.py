#!/usr/bin/env python3
"""Freeze independently audited full-1280 sparse features for three CAM test slides."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
V28 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v28'
V36 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v36'
V37 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v37'
V39 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v39'
V40 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v40'
OUT = Path(__file__).parent / 'TEST_FEATURES_FROZEN.json'
AUDITS = {
    ('normal_019', 4): V37 / 'SPARSE_PILOT_AUDIT_VERIFIED.json',
    ('normal_019', 2): V39 / 'NORMAL_019_F2_AUDIT_VERIFIED.json',
    ('normal_019', 1): V40 / 'NORMAL_019_F1_AUDIT_VERIFIED.json',
    ('tumor_091', 4): V40 / 'TUMOR_091_F4_AUDIT_VERIFIED.json',
    ('tumor_091', 2): V40 / 'TUMOR_091_F2_AUDIT_VERIFIED.json',
    ('tumor_091', 1): V40 / 'TUMOR_091_F1_AUDIT_VERIFIED.json',
    ('tumor_084', 4): V40 / 'TUMOR_084_F4_AUDIT_VERIFIED.json',
    ('tumor_084', 2): V40 / 'TUMOR_084_F2_AUDIT_VERIFIED.json',
    ('tumor_084', 1): V40 / 'TUMOR_084_F1_AUDIT_VERIFIED.json',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    frozen_path = V28 / 'SPARSE_INPUTS_FROZEN.json'
    frozen = json.loads(frozen_path.read_text())
    assert frozen['status'] == 'SOURCE_ONLY_INPUTS_FROZEN_STORAGE_ADMITTED'
    active = subprocess.check_output(['systemctl', '--user', 'list-units', 'bora*',
                                      '--state=running', '--no-legend', '--no-pager'], text=True)
    if active.strip():
        raise RuntimeError('Bora service still active')
    gpu = subprocess.check_output(['timeout', '8', 'nvidia-smi',
                                   '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    if gpu.strip():
        raise RuntimeError('GPU compute contexts still active')
    result = {'format': 'bora-cam-three-test-full1280-sparse-features-v1',
              'status': 'SOURCE_ONLY_FEATURES_FROZEN',
              'frozen_admission_sha256': sha(frozen_path),
              'channels': 1280, 'factors': [1, 2, 4], 'cases': [],
              'expert_reference_opened': False, 'online_publication': False}
    for case in frozen['cases']:
        sid = case['slide_id']
        row = {'slide_id': sid, 'image_sha256': case['image_sha256'],
               'support_sha256': case['support_sha256'], 'scales': []}
        for scale in case['scales']:
            factor = scale['factor']
            audit_path = AUDITS[(sid, factor)]
            audit = json.loads(audit_path.read_text())
            spec_path = Path(scale['spec_path'])
            spec = json.loads(spec_path.read_text())
            if (sid, factor) == ('normal_019', 4):
                complete_path = DATA / ('external-camelyon16-five-slide-2026-11-14-v36/'
                                        'pilot-normal_019-f4/FEATURES_COMPLETE.json')
                source_spec_path = V36 / 'PILOT_SPEC.json'
            else:
                complete_path = Path(spec['output_dir']) / 'FEATURES_COMPLETE.json'
                source_spec_path = spec_path
            complete = json.loads(complete_path.read_text())
            assert audit['status'] == 'VERIFIED'
            assert audit['source_spec_sha256'] == sha(source_spec_path)
            assert audit['feature_receipt_sha256'] == sha(complete_path)
            assert audit['persisted_tiles'] == complete['persisted_tiles'] == scale['eligible_tiles']
            assert audit['persisted_bytes'] == complete['persisted_bytes']
            assert complete['channels'] == 1280 and complete['factor'] == factor
            assert complete['status'] == 'COMPLETE'
            assert audit['terminal_service']['ActiveState'] == 'inactive'
            assert audit['terminal_service']['MainPID'] == '0'
            assert not audit['assigned_gpu_contexts_after']
            row['scales'].append({'factor': factor,
                                  'features_dir': str(complete_path.parent),
                                  'source_spec_path': str(source_spec_path),
                                  'source_spec_sha256': sha(source_spec_path),
                                  'feature_receipt_path': str(complete_path),
                                  'feature_receipt_sha256': sha(complete_path),
                                  'audit_path': str(audit_path),
                                  'audit_sha256': sha(audit_path),
                                  'eligible_tiles': scale['eligible_tiles'],
                                  'persisted_bytes': complete['persisted_bytes'],
                                  'peak_host_rss_kb': complete['peak_host_rss_kb'],
                                  'peak_cuda_reserved_bytes': complete['peak_cuda_reserved_bytes'],
                                  'wall_seconds': complete['wall_seconds']})
        result['cases'].append(row)
    paths = sorted(DATA.glob('external-camelyon16-*'))
    used = int(subprocess.check_output(['du', '-scB1', *(str(p) for p in paths)],
                                       text=True).splitlines()[-1].split()[0])
    free = shutil.disk_usage(DATA).free
    assert used < 200 * (1 << 30) and free > 100 * (1 << 30)
    result['campaign_bytes_after'] = used
    result['free_bytes_after'] = free
    result['campaign_cap_bytes'] = 200 * (1 << 30)
    result['free_reserve_bytes'] = 100 * (1 << 30)
    temp = OUT.with_suffix('.json.partial')
    with temp.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, OUT)
    print(json.dumps({'status': result['status'],
                      'scales': sum(len(c['scales']) for c in result['cases']),
                      'campaign_bytes': used}))

if __name__ == '__main__':
    main()
