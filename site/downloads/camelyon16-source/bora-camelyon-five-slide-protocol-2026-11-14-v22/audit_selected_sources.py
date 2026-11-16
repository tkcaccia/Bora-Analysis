#!/usr/bin/env python3
"""Stream independent hashes for three frozen CAM test image sources only."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
FROZEN = ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v19/TEST_IDS_FROZEN.json'
STATE = ROOT/'bora-camelyon-five-slide-protocol-2026-11-14-v20/DOWNLOAD_STATE.json'
OUT = Path(__file__).parent/'SOURCE_BYTES_AUDIT_VERIFIED.json'


def hashes(path):
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    frozen = json.loads(FROZEN.read_text())
    state = json.loads(STATE.read_text())
    assert state['status'] == 'COMPLETE'
    assert state['frozen_ids_sha256'] == hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert state['no_test_reference_opened'] is True
    assert state['online_publication'] is False
    ids = frozen['test_ids']
    assert [x['slide_id'] for x in state['completed']] == ids
    by_slide = {x['slide_id']: x for x in frozen['selected_official_metadata']}
    rows = []
    for completion in state['completed']:
        slide = completion['slide_id']
        spec = by_slide[slide]
        path = Path(completion['source_path'])
        assert path.is_file() and path.name == f'{slide}.tif'
        assert path.stat().st_size == spec['source_object_bytes'] == completion['source_bytes']
        md5, sha = hashes(path)
        assert md5 == spec['source_object_md5'] == completion['official_md5']
        assert sha == completion['local_sha256']
        assert completion['source_url'] == spec['source_url']
        rows.append({'slide_id': slide, 'source_path': str(path),
                     'bytes': path.stat().st_size, 'official_md5': md5,
                     'source_sha256': sha})
    assert sum(x['bytes'] for x in rows) == frozen['selected_test_source_object_bytes']
    assert not any(p.name.endswith('.part') for p in Path(rows[0]['source_path']).parent.iterdir())
    result = {
        'format': 'bora-camelyon16-three-source-byte-audit-v1',
        'status': 'VERIFIED',
        'frozen_id_manifest_sha256': hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        'download_state_sha256': hashlib.sha256(STATE.read_bytes()).hexdigest(),
        'sources': rows,
        'total_bytes': sum(x['bytes'] for x in rows),
        'all_official_md5_and_local_sha256_match': True,
        'source_hashing_opened_image_pixels': False,
        'expert_reference_opened': False,
        'new_online_publication': False,
    }
    temporary = OUT.with_suffix('.json.partial')
    with temporary.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, OUT)
    print(json.dumps({'status': result['status'], 'sources': len(rows),
                      'bytes': result['total_bytes']}))


if __name__ == '__main__':
    main()
