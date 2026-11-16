"""Serial, failure-latched source-only Bora masks for remaining CAM test slides."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
ADMISSION = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v102/ADMISSION_FROZEN.json'
SCRIPT = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v97/project_bora_mask.py'
OUT = DATA / 'external-camelyon16-five-slide-2026-11-14-v102/bora-test-masks'


def atomic(path, data):
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main():
    admission = json.loads(ADMISSION.read_text())
    if admission['status'] != 'SOURCE_ONLY_SERIAL_MASK_ADMISSION_FROZEN' or OUT.exists():
        raise ValueError('Frozen source-only serial mask admission differs')
    OUT.mkdir(parents=True)
    state = {'status': 'RUNNING', 'completed': [], 'test_reference_opened': False,
             'online_publication': False}
    atomic(OUT / 'CONTROLLER_STATE.json', state)
    for slide in ('tumor_084', 'normal_019'):
        path = OUT / slide
        try:
            subprocess.run([sys.executable, str(SCRIPT), '--slide', slide,
                            '--output', str(path)], check=True,
                           env={**os.environ,
                                'PYTHONPATH': '/mnt/sata_ssd/ChatGPT/Bora-Data/worktrees/bora-r-package-2026-11-14-v1/src'})
            receipt = path / 'MASK_COMPLETE.json'
            if not receipt.is_file():
                raise RuntimeError('Child returned success without mask receipt')
            state['completed'].append(slide)
            atomic(OUT / f'{slide}-done.json', json.loads(receipt.read_text()))
        except BaseException as error:
            state['status'] = 'FAILED_LATCHED'
            state['error'] = repr(error)
            atomic(OUT / 'CONTROLLER_FAILED.json', state)
            raise
    state['status'] = 'COMPLETE'
    atomic(OUT / 'CONTROLLER_COMPLETE.json', state)


if __name__ == '__main__':
    main()
