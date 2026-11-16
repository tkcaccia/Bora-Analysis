"""Bounded two-slide CAM binary U-Net/DeepLab fitting with spatial inner control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MIG = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
PILOT_SOURCE = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v75'
sys.path.insert(0, str(PILOT_SOURCE))
from pilot_network import (TwoSlideWindows, make_model, sha, atomic, INDEX,
                           CHECKPOINT, MODEL_SOURCE, SEED)  # noqa: E402

PILOT_AUDITS = {
    'unet': MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v77/PILOT_UNET_AUDIT_VERIFIED.json',
    'deeplabv3plus': MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v80/PILOT_DEEPLAB_AUDIT_VERIFIED.json',
}
PATCH_AUDIT = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v73/PATCH_INDEX_AUDIT_VERIFIED.json'
MAX_EPOCHS = 12
INTERNAL_LIMIT_SECONDS = 11 * 3600


def initialized_model(method):
    ckpt = CHECKPOINT / method / 'all397-final.pt'
    proof = json.loads((CHECKPOINT / method / 'MODEL_COMPLETE.json').read_text())
    if proof['status'] != 'EIGHT_CLASS_COMPARATOR_MODEL_COMPLETE' or sha(ckpt) != proof['model_sha256']:
        raise ValueError('Non-CAM checkpoint identity differs')
    model = make_model(method)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
    if method == 'unet':
        old = model.head
        model.head = nn.Conv2d(old.in_channels, 2, 1)
        head = model.head
    else:
        old = model.decoder[-1]
        model.decoder[-1] = nn.Conv2d(old.in_channels, 2, 1)
        head = model.decoder[-1]
    nn.init.kaiming_normal_(head.weight)
    nn.init.zeros_(head.bias)
    return model.cuda(), ckpt, proof


def epoch(model, windows, rows, optimizer, scaler, weights, batch_size,
          epoch_id, phase, start):
    training = optimizer is not None
    model.train(training)
    rng = np.random.default_rng(SEED + epoch_id * 257 + (1 if training else 0))
    order = rng.permutation(len(rows)) if training else np.arange(len(rows))
    total_loss, total_items, known_pixels, positive_pixels = 0., 0, 0, 0
    for start_i in range(0, len(order), batch_size):
        if time.monotonic() - start > INTERNAL_LIMIT_SECONDS:
            raise TimeoutError('Internal bounded training deadline reached before next batch')
        batch_rows = [rows[int(i)] for i in order[start_i:start_i + batch_size]]
        x, y, stats = windows.batch(batch_rows)
        known_pixels += sum(s['known_pixels'] for s in stats)
        positive_pixels += sum(s['tumor_pixels'] for s in stats)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        if training:
            k = int(rng.integers(0, 4))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
            if rng.random() < .5:
                x, y = torch.flip(x, (-1,)), torch.flip(y, (-1,))
            x = torch.clamp(x * float(rng.uniform(.9, 1.1)), 0, 1)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=weights, ignore_index=255)
            if logits.shape != (len(batch_rows), 2, 512, 512) or not torch.isfinite(loss):
                raise ValueError('Invalid binary network batch or loss')
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * len(batch_rows)
        total_items += len(batch_rows)
    torch.cuda.synchronize()
    return {'phase': phase, 'epoch': epoch_id, 'loss': total_loss / total_items,
            'patches': total_items, 'known_pixels_seen': known_pixels,
            'metastasis_pixels_seen': positive_pixels,
            'seconds_since_start': time.monotonic() - start}


def checkpoint(model, path):
    tmp = path.with_name(path.name + '.partial')
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def main(method, mode, output):
    if mode not in ('smoke', 'full') or method not in ('unet', 'deeplabv3plus'):
        raise ValueError('Only frozen CAM network modes/methods are admitted')
    if output.exists():
        raise FileExistsError(output)
    index = json.loads(INDEX.read_text())
    patch_audit = json.loads(PATCH_AUDIT.read_text())
    pilot_audit = json.loads(PILOT_AUDITS[method].read_text())
    if (patch_audit['status'] != 'VERIFIED' or pilot_audit['status'] != 'VERIFIED' or
            patch_audit['patch_index_sha256'] != sha(INDEX) or
            index['test_reference_opened'] or pilot_audit['test_reference_opened']):
        raise ValueError('Frozen training-only pilot/patch input differs')
    if not torch.cuda.is_available():
        raise RuntimeError('Chiamaka single GPU unavailable')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    begun = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    try:
        rows = index['patches']
        fit_rows = [x for x in rows if x['split'] == 'inner_fit']
        valid_rows = [x for x in rows if x['split'] == 'inner_validation']
        if mode == 'smoke':
            def pick(src, cls, split):
                return next(x for x in rows if x['slide'] == src and
                            x['official_class_id'] == cls and x['split'] == split)
            fit_rows = [pick('tumor_093', 2, 'inner_fit'), pick('normal_093', 1, 'inner_fit')]
            valid_rows = [pick('tumor_093', 2, 'inner_validation'),
                          pick('normal_093', 1, 'inner_validation')]
        windows = TwoSlideWindows(rows)
        model, old_ckpt, _ = initialized_model(method)
        weights = torch.tensor([1., 4.], device='cuda')
        batch = 4 if method == 'unet' and mode == 'full' else 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        history = []
        best_loss, best_epoch = float('inf'), 0
        for number in range(1, (1 if mode == 'smoke' else MAX_EPOCHS) + 1):
            fitted = epoch(model, windows, fit_rows, optimizer, scaler, weights, batch,
                           number, 'inner_fit', begun)
            checked = epoch(model, windows, valid_rows, None, scaler, weights, batch,
                            number, 'inner_validation', begun)
            history.append({'epoch': number, 'fit': fitted, 'inner_validation': checked})
            atomic(output / f'inner-history-epoch-{number:02d}.json', history[-1])
            print(json.dumps(history[-1]), flush=True)
            if checked['loss'] < best_loss:
                best_loss, best_epoch = checked['loss'], number
                checkpoint(model, output / 'inner-best.pt')
        if mode == 'full':
            del model
            torch.cuda.empty_cache()
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            model, _, _ = initialized_model(method)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda')
            for number in range(1, max(2, best_epoch) + 1):
                fitted = epoch(model, windows, rows, optimizer, scaler, weights, batch,
                               number, 'both_training_slides', begun)
                atomic(output / f'final-history-epoch-{number:02d}.json', fitted)
                print(json.dumps(fitted), flush=True)
            checkpoint(model, output / 'pair-final.pt')
        result = {'status': 'TRAINING_ONLY_SMOKE_COMPLETE' if mode == 'smoke' else 'CAM_PAIR_BINARY_MODEL_COMPLETE',
                  'method': method, 'mode': mode, 'training_slides': ['tumor_093', 'normal_093'],
                  'test_slides_unopened': ['tumor_084', 'tumor_091', 'normal_019'],
                  'index_sha256': sha(INDEX), 'patch_audit_sha256': sha(PATCH_AUDIT),
                  'pilot_audit_sha256': sha(PILOT_AUDITS[method]),
                  'model_source_sha256': sha(MODEL_SOURCE),
                  'pretrained_checkpoint_sha256': sha(old_ckpt),
                  'non_cam_pretraining': 'eight-output Benchmark 1 model trained on 397 non-CAM ROIs; output head replaced',
                  'class_mapping': {'official 1': 0, 'official 2': 1, 'official 0': 255},
                  'stain_normalization': 'none', 'patch_px': 512,
                  'batch_size': batch, 'inner_fit_patches': len(fit_rows),
                  'inner_validation_patches': len(valid_rows),
                  'selected_training_only_epoch': best_epoch,
                  'maximum_inner_epochs': 1 if mode == 'smoke' else MAX_EPOCHS,
                  'inner_best_sha256': sha(output / 'inner-best.pt'),
                  'final_model_sha256': sha(output / 'pair-final.pt') if mode == 'full' else None,
                  'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                  'peak_cuda_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                  'peak_host_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'wall_seconds': time.monotonic() - begun,
                  'test_reference_opened': False, 'online_publication': False}
        atomic(output / ('SMOKE_COMPLETE.json' if mode == 'smoke' else 'MODEL_COMPLETE.json'), result)
    except BaseException as error:
        atomic(output / 'FAILED.json', {'status': 'FAILED_LATCHED', 'method': method,
               'mode': mode, 'error': repr(error), 'partial_output_preserved': True,
               'test_reference_opened': False, 'online_publication': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['unet', 'deeplabv3plus'], required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    arg = parser.parse_args()
    main(arg.method, arg.mode, arg.output)
