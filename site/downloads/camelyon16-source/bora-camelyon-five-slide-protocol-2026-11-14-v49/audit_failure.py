"""Latch failed v42 mask alignment before any isolated recovery."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
STATE = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/external-camelyon16-five-slide-2026-11-14-v42/training-class-masks/tumor_093/STATE.json')
JOURNAL = Path(__file__).parent / 'V42_JOURNAL.log'
OUT = Path(__file__).parent / 'V42_FAILURE_AND_CLEANUP_VERIFIED.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    state = json.loads(STATE.read_text())
    assert state['status'] == 'FAILED_PRESERVE_PARTIAL'
    assert state['failure_type'] == 'Error' and 'TIFFFillTile' in state['failure']
    assert not state['test_reference_opened'] and not state['online_publication']
    assert 'TIFFFillTile' in JOURNAL.read_text()
    service = subprocess.check_output(['systemctl', '--user', 'show',
                                       'bora-cam16-align-tumor093-v42.service',
                                       '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result',
                                       '-p', 'ExecMainStatus', '-p', 'MainPID', '-p', 'ControlGroup'], text=True)
    fields = dict(line.split('=', 1) for line in service.splitlines())
    assert fields['ActiveState'] == 'failed' and fields['SubState'] == 'failed'
    assert fields['Result'] == 'exit-code' and fields['ExecMainStatus'] == '1'
    assert fields['MainPID'] == '0' and fields['ControlGroup'] == ''
    ps = subprocess.run(['ps', '-p', str(state['pid']), '-o', 'pid='], capture_output=True, text=True)
    assert not ps.stdout.strip()
    gpu = subprocess.check_output(['timeout', '8', 'nvidia-smi',
                                   '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    assert not gpu.strip()
    partials = sorted(str(path) for path in STATE.parent.iterdir())
    assert partials == [str(STATE)]
    result = {'format': 'bora-cam-pair-mask-alignment-v42-failure-v1',
              'status': 'FAILED_LATCHED_CLEANUP_VERIFIED',
              'cause': 'libvips/libtiff cannot decode some sparse zero-byte LZW tiles in the official tumor_093 mask; rasterio/GDAL reads them correctly',
              'failed_service': fields, 'state_path': str(STATE),
              'state_sha256': sha(STATE), 'journal_path': str(JOURNAL),
              'journal_sha256': sha(JOURNAL), 'preserved_paths': partials,
              'owned_pid_absent': True, 'assigned_gpu_contexts_after': [],
              'test_reference_opened': False, 'online_publication': False,
              'recovery': 'new isolated rasterio windowed nearest-neighbor implementation only; do not edit or reuse v42'}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'cause': result['cause']}))

if __name__ == '__main__':
    main()
