"""Admit exact official source-byte acquisition for only two CAM training masks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
V19 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
V31 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v31/acquire_training_masks.py'
V44 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v44/TEST_FEATURES_AUDIT_VERIFIED.json'
GATE = ROOT / 'bora-r-camelyon-2026-11-14-v2/LOCAL_CC0_USE_GATE.json'
OUT = Path(__file__).parent / 'TRAINING_MASK_SOURCE_ADMISSION.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    ids = json.loads(V19.read_text())
    gate = json.loads(GATE.read_text())
    features = json.loads(V44.read_text())
    assert ids['training_ids'] == ['tumor_093', 'normal_093']
    assert ids['test_ids'] == ['tumor_084', 'tumor_091', 'normal_019']
    assert gate['status'] == 'LOCAL_ANALYSIS_AUTHORIZED_PUBLICATION_HELD'
    assert gate['local_cam_specific_training_allowed'] and not gate['new_cam_online_publication_allowed']
    assert features['status'] == 'VERIFIED' and not features['expert_reference_opened']
    assert not (DATA / 'external-camelyon16-five-slide-2026-11-14-v31/training-reference').exists()
    assert not (ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v31/TRAINING_REFS_STATE.json').exists()
    active = subprocess.check_output(['systemctl', '--user', 'list-units', 'bora*',
                                      '--state=running', '--no-legend', '--no-pager'], text=True)
    if active.strip():
        raise RuntimeError('Bora service still active: ' + active)
    gpu = subprocess.check_output(['timeout', '8', 'nvidia-smi',
                                   '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    if gpu.strip():
        raise RuntimeError('GPU context still active')
    free = shutil.disk_usage(DATA).free
    assert free - 100 * (1 << 30) > 100 * (1 << 30)
    result = {'format': 'bora-cam-two-training-mask-source-admission-v1',
              'status': 'FROZEN_BEFORE_SOURCE_READ',
              'training_ids': ids['training_ids'], 'test_ids_not_downloaded': ids['test_ids'],
              'id_freeze_sha256': sha(V19), 'local_gate_sha256': sha(GATE),
              'test_feature_audit_sha256': sha(V44),
              'acquisition_source_sha256': sha(V31),
              'official_normal_mask_url': 's3://camelyon-dataset/CAMELYON16/masks/normal_093_mask.tif',
              'existing_tumor_mask_path': str(DATA / 'raw/camelyon16-external-2026-11-12/tumor_093_mask.tif'),
              'free_bytes_before': free, 'reserve_bytes': 100 * (1 << 30),
              'expert_reference_pixels_opened': False,
              'test_reference_downloaded': False, 'online_publication': False}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
