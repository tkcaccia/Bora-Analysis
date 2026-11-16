"""One bounded, training-only CAM binary-head GPU pilot per RGB architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pyvips
import rasterio
from rasterio.windows import Window
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
MIG = ROOT / 'migration'
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
INDEX = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v72/PATCH_INDEX_FROZEN.json'
INDEX_AUDIT = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v73/PATCH_INDEX_AUDIT_VERIFIED.json'
VISUAL_AUDIT = MIG / 'bora-camelyon-five-slide-protocol-2026-11-14-v74/PATCH_VISUAL_QA_VERIFIED.json'
MODEL_SOURCE = MIG / 'benchmark1-eight-class-comparator-refit-2026-11-12-v2/models.py'
sys.path.insert(0, str(MODEL_SOURCE.parent))
from models import make_model  # noqa: E402

CAL = {'tumor_093': DATA / 'external-camelyon16-pair-2026-11-13-v49/calibrated/tumor_093/calibrated-0p5-padded.tif',
       'normal_093': DATA / 'external-camelyon16-pair-2026-11-13-v48/calibrated/normal_093/calibrated-0p5-padded.tif'}
MASK = DATA / 'external-camelyon16-five-slide-2026-11-14-v58/training-labels'
CHECKPOINT = DATA / 'benchmark1-eight-class-comparator-refit-2026-11-12-v4'
SEED = 20261114


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class TwoSlideWindows:
    def __init__(self, rows):
        self.rows = rows
        self.images = {k: pyvips.Image.new_from_file(str(v), access='random') for k, v in CAL.items()}
        self.masks = {k: rasterio.open(MASK / k / 'training-classes-in-support.tif') for k in CAL}

    def batch(self, rows):
        images, targets, stats = [], [], []
        for row in rows:
            slide, x, y = row['slide'], row['x_center_0p5'], row['y_center_0p5']
            rgb = self.images[slide].crop(x - 256, y - 256, 512, 512)
            arr = np.frombuffer(rgb.write_to_memory(), np.uint8).reshape(512, 512, 3).copy()
            raw = self.masks[slide].read(1, window=Window(x - 256, y - 256, 512, 512))
            if raw.shape != (512, 512) or np.max(raw) > 2 or raw[256, 256] != row['official_class_id']:
                raise ValueError('Training patch/label registration differs')
            target = np.full(raw.shape, 255, np.uint8)
            target[raw == 1] = 0
            target[raw == 2] = 1
            images.append(torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float() / 255)
            targets.append(torch.from_numpy(target.astype(np.int64)))
            stats.append({'slide': slide, 'center_class': row['official_class_id'],
                          'known_pixels': int(np.count_nonzero(target != 255)),
                          'tumor_pixels': int(np.count_nonzero(target == 1))})
        return torch.stack(images), torch.stack(targets), stats


def main(method, output):
    if method not in ('unet', 'deeplabv3plus'):
        raise ValueError('Only the two frozen RGB comparator architectures are admitted')
    if output.exists():
        raise FileExistsError(output)
    patches = json.loads(INDEX.read_text())
    index_audit = json.loads(INDEX_AUDIT.read_text())
    visual = json.loads(VISUAL_AUDIT.read_text())
    if (patches['status'] != 'TRAINING_ONLY_PATCH_INDEX_FROZEN' or
            index_audit['status'] != 'VERIFIED' or
            index_audit['patch_index_sha256'] != sha(INDEX) or
            visual['status'] != 'VISUALLY_VERIFIED_FOR_TRAINING_PILOT' or
            visual['patch_index_sha256'] != sha(INDEX)):
        raise ValueError('Training patch audit differs')
    ckpt = CHECKPOINT / method / 'all397-final.pt'
    receipt = json.loads((CHECKPOINT / method / 'MODEL_COMPLETE.json').read_text())
    if receipt['status'] != 'EIGHT_CLASS_COMPARATOR_MODEL_COMPLETE' or receipt['model_sha256'] != sha(ckpt):
        raise ValueError('Non-CAM pretraining checkpoint differs')
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned Chiamaka GPU unavailable')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        model = make_model(method)
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
        if method == 'unet':
            old = model.head
            model.head = nn.Conv2d(old.in_channels, 2, 1)
        else:
            old = model.decoder[-1]
            model.decoder[-1] = nn.Conv2d(old.in_channels, 2, 1)
        nn.init.kaiming_normal_(model.head.weight if method == 'unet' else model.decoder[-1].weight)
        nn.init.zeros_(model.head.bias if method == 'unet' else model.decoder[-1].bias)
        model = model.cuda()
        windows = TwoSlideWindows(patches['patches'])
        def pick(slide, cls, split):
            return next(p for p in patches['patches'] if p['slide'] == slide and
                        p['official_class_id'] == cls and p['split'] == split)
        train_rows = [pick('tumor_093', 2, 'inner_fit'), pick('normal_093', 1, 'inner_fit')]
        valid_rows = [pick('tumor_093', 2, 'inner_validation'), pick('normal_093', 1, 'inner_validation')]
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        weights = torch.tensor([1., 4.], device='cuda')
        model.train()
        x, y, train_stats = windows.batch(train_rows)
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, ignore_index=255)
        if logits.shape != (2, 2, 512, 512) or not torch.isfinite(loss):
            raise ValueError('Nonfinite or misregistered binary-head pilot logits')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        train_loss = float(loss.detach().cpu())
        model.eval()
        with torch.no_grad():
            x, y, valid_stats = windows.batch(valid_rows)
            x, y = x.cuda(), y.cuda()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                valid_loss = F.cross_entropy(logits, y, weight=weights, ignore_index=255)
        if not torch.isfinite(valid_loss):
            raise ValueError('Nonfinite spatial-validation pilot loss')
        result = {'status': 'TRAINING_ONLY_GPU_PILOT_COMPLETE', 'method': method,
                  'patch_index_sha256': sha(INDEX), 'patch_audit_sha256': sha(INDEX_AUDIT),
                  'model_source_sha256': sha(MODEL_SOURCE), 'pretrained_checkpoint_sha256': sha(ckpt),
                  'pretraining': '397 non-CAM Benchmark 1 images; eight-output head replaced by a new two-output head',
                  'train_patch_stats': train_stats, 'inner_validation_patch_stats': valid_stats,
                  'train_loss': train_loss, 'inner_validation_loss': float(valid_loss.detach().cpu()),
                  'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                  'peak_cuda_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                  'peak_host_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'wall_seconds': time.monotonic() - started,
                  'test_reference_opened': False, 'online_publication': False}
        atomic(output / 'PILOT_COMPLETE.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        atomic(output / 'FAILED.json', {'status': 'FAILED_LATCHED', 'method': method,
               'error': repr(error), 'test_reference_opened': False, 'online_publication': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['unet', 'deeplabv3plus'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.method, args.output)
