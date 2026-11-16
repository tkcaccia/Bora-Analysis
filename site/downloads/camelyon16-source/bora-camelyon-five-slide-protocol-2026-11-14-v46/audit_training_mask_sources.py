"""Independently verify only the two official CAM training mask source files."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
IDS = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
SOURCE = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v31/TRAINING_REFS_STATE.json'
ADMISSION = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v45/TRAINING_MASK_SOURCE_ADMISSION.json'
OUT = Path(__file__).parent / 'TRAINING_MASK_SOURCES_AUDIT_VERIFIED.json'
SERVICE = 'bora-cam16-training-mask-source-v31.service'

def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            md5.update(part)
            sha.update(part)
    return md5.hexdigest(), sha.hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    frozen = json.loads(IDS.read_text())
    source = json.loads(SOURCE.read_text())
    admission = json.loads(ADMISSION.read_text())
    assert source['status'] == 'COMPLETE_SOURCE_BYTES_VERIFIED_NOT_OPENED'
    assert admission['status'] == 'FROZEN_BEFORE_SOURCE_READ'
    assert not source['reference_pixels_opened']
    assert not source['test_reference_downloaded'] and not source['online_publication']
    assert {r['slide_id'] for r in source['completed']} == set(frozen['training_ids'])
    assert set(frozen['test_ids']) == {'tumor_084', 'tumor_091', 'normal_019'}
    verified = []
    for row in source['completed']:
        official = next(r for r in frozen['selected_official_metadata'] if r['slide_id'] == row['slide_id'])
        path = Path(row['mask_path'])
        assert path.stat().st_size == row['bytes'] == official['mask_object_bytes']
        md5, sha = hashes(path)
        assert md5 == row['official_md5'] == official['mask_object_md5']
        assert sha == row['sha256']
        verified.append({'slide_id': row['slide_id'], 'path': str(path),
                         'bytes': path.stat().st_size, 'official_md5': md5,
                         'sha256': sha})
    status = subprocess.check_output(['systemctl', '--user', 'show', SERVICE,
                                      '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result',
                                      '-p', 'ExecMainStatus', '-p', 'MainPID', '-p', 'ControlGroup'], text=True)
    fields = dict(line.split('=', 1) for line in status.splitlines())
    assert fields['ActiveState'] == 'inactive' and fields['SubState'] == 'dead'
    assert fields['Result'] == 'success' and fields['ExecMainStatus'] == '0'
    assert fields['MainPID'] == '0' and fields['ControlGroup'] == ''
    result = {'format': 'bora-cam-two-training-mask-source-audit-v1', 'status': 'VERIFIED',
              'training_masks': verified, 'official_metadata_sha256': hashes(IDS)[1],
              'source_receipt_sha256': hashes(SOURCE)[1],
              'admission_sha256': hashes(ADMISSION)[1],
              'terminal_service': fields, 'reference_pixels_opened': False,
              'test_reference_downloaded': False, 'online_publication': False}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'VERIFIED', 'training_masks': len(verified)}))

if __name__ == '__main__':
    main()
