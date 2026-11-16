"""Independent nine-scale handoff audit without opening test reference masks."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
FROZEN = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v41/TEST_FEATURES_FROZEN.json'
ADMISSION = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v28/SPARSE_INPUTS_FROZEN.json'
OUT = Path(__file__).parent / 'TEST_FEATURES_AUDIT_VERIFIED.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    handoff = json.loads(FROZEN.read_text())
    admission = json.loads(ADMISSION.read_text())
    assert handoff['status'] == 'SOURCE_ONLY_FEATURES_FROZEN'
    assert handoff['frozen_admission_sha256'] == sha(ADMISSION)
    assert handoff['channels'] == 1280 and handoff['factors'] == [1, 2, 4]
    assert not handoff['expert_reference_opened'] and not handoff['online_publication']
    assert {r['slide_id'] for r in handoff['cases']} == {'tumor_084', 'tumor_091', 'normal_019'}
    assert len(handoff['cases']) == 3
    tiles = bytes_total = 0
    seen = set()
    for case in handoff['cases']:
        source = next(r for r in admission['cases'] if r['slide_id'] == case['slide_id'])
        assert case['image_sha256'] == source['image_sha256']
        assert case['support_sha256'] == source['support_sha256']
        assert len(case['scales']) == 3
        for item in case['scales']:
            key = case['slide_id'], item['factor']
            assert key not in seen
            seen.add(key)
            expected = next(s for s in source['scales'] if s['factor'] == item['factor'])
            receipt_path = Path(item['feature_receipt_path'])
            audit_path = Path(item['audit_path'])
            spec_path = Path(item['source_spec_path'])
            receipt = json.loads(receipt_path.read_text())
            audit = json.loads(audit_path.read_text())
            assert sha(receipt_path) == item['feature_receipt_sha256'] == audit['feature_receipt_sha256']
            assert sha(audit_path) == item['audit_sha256']
            assert sha(spec_path) == item['source_spec_sha256'] == audit['source_spec_sha256']
            assert audit['status'] == 'VERIFIED' and receipt['status'] == 'COMPLETE'
            assert receipt['channels'] == 1280 and receipt['dtype'] == 'float16'
            assert receipt['factor'] == item['factor']
            assert receipt['persisted_tiles'] == item['eligible_tiles'] == expected['eligible_tiles']
            assert receipt['persisted_bytes'] == item['persisted_bytes']
            assert audit['all_tile_rows_columns_hashes_and_finite_tokens_verified']
            assert audit['terminal_service']['ActiveState'] == 'inactive'
            assert audit['terminal_service']['MainPID'] == '0'
            assert not audit['assigned_gpu_contexts_after']
            assert Path(item['features_dir']) == receipt_path.parent
            tiles += item['eligible_tiles']
            bytes_total += item['persisted_bytes']
    assert seen == {(slide, factor) for slide in ('normal_019', 'tumor_091', 'tumor_084')
                    for factor in (1, 2, 4)}
    assert tiles == sum(s['eligible_tiles'] for c in admission['cases'] for s in c['scales'])
    assert handoff['campaign_bytes_after'] < handoff['campaign_cap_bytes']
    assert handoff['free_bytes_after'] > handoff['free_reserve_bytes']
    result = {'format': 'bora-cam-three-test-sparse-feature-handoff-audit-v1',
              'status': 'VERIFIED', 'frozen_handoff_sha256': sha(FROZEN),
              'admission_sha256': sha(ADMISSION), 'scales': len(seen),
              'full_channel_tiles': tiles, 'persisted_feature_bytes': bytes_total,
              'all_scale_audits_rechecked': True, 'expert_reference_opened': False,
              'online_publication': False}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
