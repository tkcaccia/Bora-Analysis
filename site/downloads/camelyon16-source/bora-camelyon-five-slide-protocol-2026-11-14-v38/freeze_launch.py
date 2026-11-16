"""Freeze one bounded CAM sparse-feature production GPU admission."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
V28 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v28'
V35 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v35'
V37 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v37'
WORKER = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v26/sparse_feature_worker.py'
SPEC = V28 / 'normal_019-f2-SPEC.json'
GPU = 'GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    frozen = json.loads((V28 / 'SPARSE_INPUTS_FROZEN.json').read_text())
    audit = json.loads((V35 / 'SPARSE_ADMISSION_AUDIT_VERIFIED.json').read_text())
    pilot = json.loads((V37 / 'SPARSE_PILOT_AUDIT_VERIFIED.json').read_text())
    assert frozen['status'] == 'SOURCE_ONLY_INPUTS_FROZEN_STORAGE_ADMITTED'
    assert audit['status'] == pilot['status'] == 'VERIFIED'
    assert sha(WORKER) == frozen['worker_source_sha256'] == pilot['worker_source_sha256']
    spec = json.loads(SPEC.read_text())
    expected = next(s for c in frozen['cases'] if c['slide_id'] == 'normal_019'
                    for s in c['scales'] if s['factor'] == 2)
    assert sha(SPEC) == expected['spec_sha256']
    assert not Path(spec['output_dir']).exists()
    active = subprocess.check_output(['systemctl', '--user', 'list-units', 'bora*',
                                      '--state=running', '--no-legend', '--no-pager'], text=True)
    if active.strip():
        raise RuntimeError('Bora service is running: ' + active)
    contexts = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    if contexts:
        raise RuntimeError('GPU compute contexts exist: ' + contexts)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    if GPU not in gpu:
        raise RuntimeError('Assigned GPU missing')
    paths = sorted(DATA.glob('external-camelyon16-*'))
    used = int(subprocess.check_output(['du', '-scB1', *(str(p) for p in paths)],
                                       text=True).splitlines()[-1].split()[0])
    free = shutil.disk_usage(DATA).free
    max_bytes = spec['max_feature_bytes']
    assert used + max_bytes + (1 << 30) < 200 * (1 << 30)
    assert free - max_bytes - (1 << 30) > 100 * (1 << 30)
    result = dict(format='bora-cam-sparse-production-f2-admission-v1',
                  status='FROZEN_BEFORE_GPU', slide_id='normal_019', factor=2,
                  spec_sha256=sha(SPEC), spec_path=str(SPEC),
                  source_only_frozen_sha256=sha(V28 / 'SPARSE_INPUTS_FROZEN.json'),
                  admission_audit_sha256=sha(V35 / 'SPARSE_ADMISSION_AUDIT_VERIFIED.json'),
                  pilot_audit_sha256=sha(V37 / 'SPARSE_PILOT_AUDIT_VERIFIED.json'),
                  worker_source_sha256=sha(WORKER), assigned_gpu_uuid=GPU,
                  gpu_before=gpu, gpu_compute_contexts_before=[],
                  campaign_bytes_before=used, free_bytes_before=free,
                  max_new_bytes=max_bytes, cap_bytes=200 * (1 << 30),
                  reserve_bytes=100 * (1 << 30), expert_reference_opened=False,
                  online_publication=False)
    path = HERE / 'LAUNCH_FROZEN.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
