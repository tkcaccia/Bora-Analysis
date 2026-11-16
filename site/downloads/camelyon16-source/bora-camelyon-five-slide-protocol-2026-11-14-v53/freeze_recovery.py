"""Fresh serial CPU admission after latched libvips mask failure."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SCRIPT = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v51/align_mask_rasterio.py'
FAILURE = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v52/V48_FAILED_BEFORE_COMPUTE.json'
SOURCE = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v46/TRAINING_MASK_SOURCES_AUDIT_VERIFIED.json'
OUT = Path(__file__).parent / 'RECOVERY_FROZEN.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    failure = json.loads(FAILURE.read_text())
    source = json.loads(SOURCE.read_text())
    assert failure['status'] == 'FAILED_BEFORE_COMPUTE_CLEANUP_VERIFIED'
    assert source['status'] == 'VERIFIED'
    assert not (DATA / 'external-camelyon16-five-slide-2026-11-14-v51/training-class-masks').exists()
    active = subprocess.check_output(['systemctl', '--user', 'list-units', 'bora*',
                                      '--state=running', '--no-legend', '--no-pager'], text=True)
    assert not active.strip()
    gpu = subprocess.check_output(['timeout', '8', 'nvidia-smi',
                                   '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    assert not gpu.strip()
    free = shutil.disk_usage(DATA).free
    paths = sorted(DATA.glob('external-camelyon16-*'))
    used = int(subprocess.check_output(['du', '-scB1', *(str(p) for p in paths)],
                                       text=True).splitlines()[-1].split()[0])
    max_new = 12 * (1 << 30)
    assert used + max_new < 200 * (1 << 30)
    assert free - max_new > 100 * (1 << 30)
    result = {'format': 'bora-cam-rasterio-mask-recovery-admission-v1',
              'status': 'FROZEN_BEFORE_RECOVERY',
              'training_slides': ['tumor_093', 'normal_093'],
              'source_audit_sha256': sha(SOURCE),
              'latched_failure_sha256': sha(FAILURE),
              'new_script_sha256': sha(SCRIPT),
              'diagnosis': 'v48 used a false 512-divisibility assumption for a grid padded to multiples of 896; v51 admits partial TIFF edge windows',
              'windowed_nearest_0p5_pixel_mapping': True,
              'campaign_bytes_before': used, 'max_new_bytes': max_new,
              'campaign_cap_bytes': 200 * (1 << 30),
              'free_bytes_before': free, 'free_reserve_bytes': 100 * (1 << 30),
              'owned_processes_and_gpu_contexts_clean': True,
              'test_reference_opened': False, 'online_publication': False}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
