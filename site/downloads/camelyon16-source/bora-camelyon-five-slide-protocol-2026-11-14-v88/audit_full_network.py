"""Independent, read-only receipt audit for a two-slide CAM binary fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch


ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
MIG = ROOT / 'migration'
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def service_state(unit: str) -> dict[str, str]:
    stdout = subprocess.check_output(
        ['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p',
         'MainPID', '-p', 'Result', '-p', 'ExecMainStatus', '-p', 'ControlGroup'],
        text=True)
    return dict(line.split('=', 1) for line in stdout.strip().splitlines())


def main(method: str, version: int, out_version: int) -> None:
    directory = DATA / f'external-camelyon16-five-slide-2026-11-14-v{version}' / f'{method}-full'
    receipt_path = directory / 'MODEL_COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    admission_path = MIG / f'bora-camelyon-five-slide-protocol-2026-11-14-v{version}' / 'ADMISSION_FROZEN.json'
    admission = json.loads(admission_path.read_text())
    assert receipt['status'] == 'CAM_PAIR_BINARY_MODEL_COMPLETE'
    assert receipt['method'] == method and receipt['mode'] == 'full'
    assert receipt['training_slides'] == ['tumor_093', 'normal_093']
    assert receipt['test_slides_unopened'] == ['tumor_084', 'tumor_091', 'normal_019']
    assert receipt['test_reference_opened'] is False and receipt['online_publication'] is False
    assert receipt['stain_normalization'] == 'none'
    assert receipt['class_mapping'] == {'official 1': 0, 'official 2': 1, 'official 0': 255}
    source = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v81/train_network.py'
    index = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v72/PATCH_INDEX_FROZEN.json'
    patch_audit = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v73/PATCH_INDEX_AUDIT_VERIFIED.json'
    assert sha(source) == admission['script_sha256']
    assert sha(index) == receipt['index_sha256'] == admission['patch_index_sha256']
    assert sha(patch_audit) == receipt['patch_audit_sha256'] == admission['patch_audit_sha256']
    source_index = json.loads(index.read_text())
    assert len(source_index['patches']) == 960
    fit = [row for row in source_index['patches'] if row['split'] == 'inner_fit']
    valid = [row for row in source_index['patches'] if row['split'] == 'inner_validation']
    assert len(fit) == receipt['inner_fit_patches'] == 768
    assert len(valid) == receipt['inner_validation_patches'] == 192
    histories = []
    for number in range(1, receipt['maximum_inner_epochs'] + 1):
        item = json.loads((directory / f'inner-history-epoch-{number:02d}.json').read_text())
        assert item['epoch'] == number
        assert item['fit']['patches'] == 768 and item['inner_validation']['patches'] == 192
        assert item['fit']['metastasis_pixels_seen'] > 0 and item['inner_validation']['metastasis_pixels_seen'] > 0
        histories.append(item)
    selected = min(histories, key=lambda x: x['inner_validation']['loss'])['epoch']
    assert selected == receipt['selected_training_only_epoch']
    assert sha(directory / 'inner-best.pt') == receipt['inner_best_sha256']
    final_epochs = max(2, selected)
    for number in range(1, final_epochs + 1):
        item = json.loads((directory / f'final-history-epoch-{number:02d}.json').read_text())
        assert item['epoch'] == number and item['patches'] == 960
        assert item['metastasis_pixels_seen'] > 0
    final_path = directory / 'pair-final.pt'
    assert sha(final_path) == receipt['final_model_sha256']
    weights = torch.load(final_path, map_location='cpu', weights_only=True)
    classifier_key = 'head.weight' if method == 'unet' else 'decoder.6.weight'
    assert weights[classifier_key].shape[0] == 2
    unit = f'bora-cam16-binary-{method}-full-v{version}.service'
    state = service_state(unit)
    assert state == {'MainPID': '0', 'Result': 'success', 'ExecMainStatus': '0',
                     'ControlGroup': '', 'ActiveState': 'inactive'}, state
    contexts = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
         '--format=csv,noheader'], text=True).strip()
    assert not contexts, f'GPU context ownership requires inspection: {contexts}'
    audit = {
        'status': 'VERIFIED', 'method': method, 'format': 'bora-cam-binary-pair-fit-audit-v1',
        'admission_sha256': sha(admission_path), 'model_receipt_sha256': sha(receipt_path),
        'checkpoint_sha256': sha(final_path), 'checkpoint_tensors': len(weights),
        'selected_training_only_epoch': selected, 'inner_epochs': len(histories),
        'final_training_epochs': final_epochs, 'training_patches': 960,
        'terminal_service': state, 'owned_gpu_contexts_after': [],
        'test_reference_opened': False, 'online_publication': False,
    }
    out = MIG / f'bora-camelyon-five-slide-protocol-2026-11-14-v{out_version}'
    out.mkdir(exist_ok=True)
    path = out / f'{method.upper()}_PAIR_FIT_AUDIT_VERIFIED.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(audit, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['unet', 'deeplabv3plus'], required=True)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--out-version', type=int, required=True)
    args = parser.parse_args()
    main(args.method, args.version, args.out_version)
