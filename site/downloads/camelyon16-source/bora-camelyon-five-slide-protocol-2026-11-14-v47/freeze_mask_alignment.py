"""Admit local-only nearest-neighbor training-mask alignment."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
V42 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v42/align_training_masks.py'
V46 = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v46/TRAINING_MASK_SOURCES_AUDIT_VERIFIED.json'
OUT = Path(__file__).parent / 'MASK_ALIGNMENT_FROZEN.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = json.loads(V46.read_text())
    assert audit['status'] == 'VERIFIED'
    assert not audit['reference_pixels_opened']
    active = subprocess.check_output(['systemctl', '--user', 'list-units', 'bora*',
                                      '--state=running', '--no-legend', '--no-pager'], text=True)
    assert not active.strip()
    gpu = subprocess.check_output(['timeout', '8', 'nvidia-smi',
                                   '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    assert not gpu.strip()
    free = shutil.disk_usage(DATA).free
    paths = sorted(DATA.glob('external-camelyon16-*'))
    campaign = int(subprocess.check_output(['du', '-scB1', *(str(p) for p in paths)],
                                           text=True).splitlines()[-1].split()[0])
    max_new = 12 * (1 << 30)
    assert campaign + max_new < 200 * (1 << 30)
    assert free - max_new > 100 * (1 << 30)
    result = {'format': 'bora-cam-pair-mask-alignment-freeze-v1',
              'status': 'FROZEN_BEFORE_TRAINING_REFERENCE_PIXELS',
              'training_slides': ['tumor_093', 'normal_093'],
              'test_slides_not_opened': ['tumor_084', 'tumor_091', 'normal_019'],
              'source_audit_sha256': sha(V46), 'alignment_script_sha256': sha(V42),
              'normalization': 'nearest-neighbor native annotation to 0.5-micrometer calibrated image; zero-class padding',
              'campaign_bytes_before': campaign, 'max_new_bytes': max_new,
              'campaign_cap_bytes': 200 * (1 << 30),
              'free_bytes_before': free, 'free_reserve_bytes': 100 * (1 << 30),
              'gpu_contexts_before': [], 'online_publication': False}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
