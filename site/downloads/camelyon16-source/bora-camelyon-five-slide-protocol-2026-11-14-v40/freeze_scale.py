"""Freeze admission for the next serial CAM test-slide sparse feature scale."""
import argparse
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
GPU = 'GPU-59b2b884-a1c9-f7a3-9f35-e8380f4609d5'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide', required=True,
                        choices=('normal_019', 'tumor_091', 'tumor_084'))
    parser.add_argument('--factor', type=int, required=True, choices=(1, 2, 4))
    parser.add_argument('--prior-audit', type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((V28 / 'SPARSE_INPUTS_FROZEN.json').read_text())
    audit = json.loads((V35 / 'SPARSE_ADMISSION_AUDIT_VERIFIED.json').read_text())
    pilot = json.loads((V37 / 'SPARSE_PILOT_AUDIT_VERIFIED.json').read_text())
    prior = json.loads(args.prior_audit.read_text())
    assert frozen['status'] == 'SOURCE_ONLY_INPUTS_FROZEN_STORAGE_ADMITTED'
    assert audit['status'] == pilot['status'] == prior['status'] == 'VERIFIED'
    assert prior['terminal_service']['ActiveState'] == 'inactive'
    assert not prior['assigned_gpu_contexts_after']
    assert sha(WORKER) == frozen['worker_source_sha256'] == pilot['worker_source_sha256']
    case = next(c for c in frozen['cases'] if c['slide_id'] == args.slide)
    scale = next(s for s in case['scales'] if s['factor'] == args.factor)
    spec_path = Path(scale['spec_path'])
    spec = json.loads(spec_path.read_text())
    assert sha(spec_path) == scale['spec_sha256']
    assert spec['slide_id'] == args.slide and spec['factor'] == args.factor
    assert not Path(spec['output_dir']).exists()
    active = subprocess.check_output(['systemctl', '--user', 'list-units', 'bora*',
                                      '--state=running', '--no-legend', '--no-pager'], text=True)
    if active.strip():
        raise RuntimeError('Bora service is running: ' + active)
    contexts = subprocess.check_output(['timeout', '8', 'nvidia-smi',
                                        '--query-compute-apps=pid,gpu_uuid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    if contexts:
        raise RuntimeError('GPU compute contexts exist: ' + contexts)
    gpu = subprocess.check_output(['timeout', '8', 'nvidia-smi',
                                   '--query-gpu=uuid,memory.used,memory.total',
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
    result = dict(format='bora-cam-sparse-production-scale-admission-v1',
                  status='FROZEN_BEFORE_GPU', slide_id=args.slide, factor=args.factor,
                  spec_sha256=sha(spec_path), spec_path=str(spec_path),
                  prior_audit_path=str(args.prior_audit),
                  prior_audit_sha256=sha(args.prior_audit),
                  source_only_frozen_sha256=sha(V28 / 'SPARSE_INPUTS_FROZEN.json'),
                  admission_audit_sha256=sha(V35 / 'SPARSE_ADMISSION_AUDIT_VERIFIED.json'),
                  pilot_audit_sha256=sha(V37 / 'SPARSE_PILOT_AUDIT_VERIFIED.json'),
                  worker_source_sha256=sha(WORKER), assigned_gpu_uuid=GPU,
                  gpu_before=gpu, gpu_compute_contexts_before=[],
                  campaign_bytes_before=used, free_bytes_before=free,
                  max_new_bytes=max_bytes, cap_bytes=200 * (1 << 30),
                  reserve_bytes=100 * (1 << 30), expert_reference_opened=False,
                  online_publication=False)
    path = HERE / f'LAUNCH_{args.slide}_f{args.factor}.json'
    with path.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
