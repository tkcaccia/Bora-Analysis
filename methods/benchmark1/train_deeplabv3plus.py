"""Same-split DeepLabV3+ comparator for frozen Benchmark 1.

Runs on Chiamaka only. Evaluation masks are never opened. The first phase
chooses an epoch using only training source groups; the second phase refits on
all 397 training images. A separate inference phase writes predictions for the
87 retained development validation images. No prediction is scored in this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class DeepLabV3Plus(nn.Module):
    def __init__(self, classes: int = 24, base: int = 24):
        super().__init__()
        self.encoder = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        self.aspp = ASPP(2048, [6, 12, 18], out_channels=256)
        self.low_projection = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, classes, 1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        encoder = self.encoder
        high = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(x))))
        low = encoder.layer1(high)
        high = encoder.layer4(encoder.layer3(encoder.layer2(low)))
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        decoded = self.decoder(torch.cat((high, self.low_projection(low)), dim=1))
        return F.interpolate(decoded, size=size, mode='bilinear', align_corners=False)


class Patches(Dataset):
    def __init__(self, rows: list[dict], root: Path, mapping: dict,
                 class_ids: dict, patch: int, repeats: int, seed: int,
                 augment: bool):
        self.rows, self.root, self.patch = rows, root, patch
        self.repeats, self.seed, self.augment = repeats, seed, augment
        self.epoch = 0
        self.luts = {}
        for dataset, source_map in mapping.items():
            lut = np.full(65536, 255, dtype=np.uint8)
            for code, info in source_map.items():
                if info['use'] == 'score' and info['canonical_role'] in class_ids:
                    lut[int(code)] = class_ids[info['canonical_role']] - 1
            self.luts[dataset] = lut

    def __len__(self):
        return len(self.rows) * self.repeats

    def __getitem__(self, index):
        row = self.rows[index // self.repeats]
        seed = int(hashlib.sha256(f'{self.seed}:{self.epoch}:{index}'.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        image_path = self.root / row['staged_image']
        mask_path = self.root / row['staged_expert_mask']
        with Image.open(image_path) as src:
            rgb = np.asarray(src.convert('RGB'))
        with Image.open(mask_path) as src:
            raw = np.asarray(src)
        w, h = row['true_working_size_wh']
        rgb, raw = rgb[:h, :w], raw[:h, :w]
        if raw.ndim != 2 or rgb.shape[:2] != raw.shape:
            raise ValueError(f'Image/mask frame mismatch: {row["case"]}')
        if int(raw.max()) >= 65536:
            raise ValueError(f'Expert code exceeds lookup table: {row["case"]}')
        target = self.luts[row['dataset']][raw.astype(np.int64)]
        pad_y, pad_x = max(0, self.patch - h), max(0, self.patch - w)
        if pad_y or pad_x:
            rgb = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), constant_values=255)
            target = np.pad(target, ((0, pad_y), (0, pad_x)), constant_values=255)
        h, w = target.shape
        y = int(rng.integers(0, h - self.patch + 1))
        x = int(rng.integers(0, w - self.patch + 1))
        # Annotation-aware class sampling uses training labels only. The inner
        # validation crop policy is unchanged from the original reference.
        if self.augment and rng.random() < 0.8:
            labels, pixels = np.unique(target, return_counts=True)
            eligible = (labels != 255) & (pixels >= 100)
            if np.any(eligible):
                labels, pixels = labels[eligible], pixels[eligible]
                likelihood = 1 / np.sqrt(pixels.astype(np.float64))
                likelihood /= likelihood.sum()
                label = int(rng.choice(labels, p=likelihood))
                locations = np.flatnonzero(target == label)
                cy, cx = np.unravel_index(int(rng.choice(locations)), target.shape)
                y = min(max(0, cy - self.patch // 2), h - self.patch)
                x = min(max(0, cx - self.patch // 2), w - self.patch)
        elif rng.random() < 0.7:
            for _ in range(20):
                cy = int(rng.integers(0, h)); cx = int(rng.integers(0, w))
                if target[cy, cx] != 255:
                    y = min(max(0, cy - self.patch // 2), h - self.patch)
                    x = min(max(0, cx - self.patch // 2), w - self.patch)
                    break
        if not np.any(target[y:y+self.patch, x:x+self.patch] != 255):
            valid = np.flatnonzero(target != 255)
            if not len(valid):
                raise ValueError(f'No compatible annotated pixel: {row["case"]}')
            cy, cx = np.unravel_index(int(valid[int(rng.integers(len(valid)))]), target.shape)
            y = min(max(0, cy - self.patch // 2), h - self.patch)
            x = min(max(0, cx - self.patch // 2), w - self.patch)
        rgb = rgb[y:y+self.patch, x:x+self.patch].copy()
        target = target[y:y+self.patch, x:x+self.patch].copy()
        if self.augment:
            k = int(rng.integers(0, 4))
            rgb, target = np.rot90(rgb, k), np.rot90(target, k)
            if rng.random() < .5:
                rgb, target = np.flip(rgb, 1), np.flip(target, 1)
            gain = float(rng.uniform(.9, 1.1))
            rgb = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return (torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255,
                torch.from_numpy(np.ascontiguousarray(target)).long())


def loss_batch(model, batch, device, scaler, weights, optimizer=None):
    x, y = (part.to(device, non_blocking=True) for part in batch)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=weights, ignore_index=255)
    if optimizer is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return float(loss.detach().cpu())


def run_phase(name, train_rows, valid_rows, root, mapping, class_ids,
              epochs, out, seed, weights, patch=512, batch_size=2):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = DeepLabV3Plus(classes=len(class_ids)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    ds = Patches(train_rows, root, mapping, class_ids, patch, 4, seed, True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                        pin_memory=True, persistent_workers=False)
    validation = None
    if valid_rows:
        validation = DataLoader(Patches(valid_rows, root, mapping, class_ids,
                                        patch, 1, seed + 109, False),
                                batch_size=batch_size, shuffle=False, num_workers=2,
                                pin_memory=True, persistent_workers=False)
    best_loss, best_epoch = float('inf'), None
    history = []
    for epoch in range(1, epochs + 1):
        ds.epoch = epoch
        model.train()
        start = time.monotonic()
        train_loss = np.mean([loss_batch(model, b, 'cuda', scaler, weights, optimizer) for b in loader])
        val_loss = None
        if validation is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(np.mean([loss_batch(model, b, 'cuda', scaler, weights) for b in validation]))
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                tmp = out / f'{name}-best.pt.tmp'
                torch.save(model.state_dict(), tmp)
                tmp.replace(out / f'{name}-best.pt')
        rec = {'phase': name, 'epoch': epoch, 'train_loss': float(train_loss),
               'validation_loss': val_loss, 'seconds': time.monotonic() - start}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        (out / f'{name}-history.json').write_text(json.dumps(history, indent=2) + '\n')
    if validation is None:
        tmp = out / f'{name}-final.pt.tmp'
        torch.save(model.state_dict(), tmp)
        tmp.replace(out / f'{name}-final.pt')
    return best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--harmonization', type=Path, required=True)
    ap.add_argument('--max-epochs', type=int, default=60)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--phase', choices=['fit', 'infer'], required=True)
    args = ap.parse_args()
    manifest = json.loads((args.root / 'INPUTS_FROZEN.json').read_text())
    assert manifest['status'] == 'COMPARATOR_INPUTS_FROZEN_NO_PREDICTIONS'
    rows = manifest['rows']
    train = [r for r in rows if r['split'] == 'train']
    evaluation = [r for r in rows if r['split'] == 'evaluation']
    assert len(train) == 397 and len(evaluation) == 87
    class_ids = {name: int(i) for i, name in manifest['class_names'].items()}
    assert set(class_ids.values()) == set(range(1, 25))
    mapping = json.loads(args.harmonization.read_text())['mapping']
    weight_receipt = json.loads(args.weights.read_text())
    if (weight_receipt['status'] != 'TRAINING_ONLY_CLASS_WEIGHTS_FROZEN' or
            weight_receipt['training_images'] != 397 or
            weight_receipt['evaluation_masks_opened'] != 0 or
            weight_receipt['source_manifest_sha256'] != sha(args.root / 'INPUTS_FROZEN.json') or
            weight_receipt['harmonization_sha256'] != sha(args.harmonization)):
        raise ValueError('Training-only weight identity mismatch')
    weights = torch.tensor(weight_receipt['class_weights'], dtype=torch.float32, device='cuda')
    if weights.shape != (24,) or not torch.all(torch.isfinite(weights)):
        raise ValueError('Class weights invalid')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required on Chiamaka')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'fit':
        inner_fit = [r for r in train if r['inner_role'] == 'fit']
        inner_valid = [r for r in train if r['inner_role'] == 'validation']
        assert inner_fit and inner_valid and not ({r['source_group'] for r in inner_fit} &
                                                  {r['source_group'] for r in inner_valid})
        best_epoch = run_phase('inner', inner_fit, inner_valid, args.root,
                               mapping, class_ids, args.max_epochs, args.out, 1234, weights)
        assert best_epoch is not None
        best_epoch = max(12, best_epoch)
        run_phase('all397', train, [], args.root, mapping, class_ids,
                  best_epoch, args.out, 5678, weights)
        model_file = args.out / 'all397-final.pt'
        rec = {'status': 'SAME_SPLIT_DEEPLABV3PLUS_MODEL_COMPLETE',
               'training_images': len(train), 'evaluation_images_used_for_training': 0,
               'inner_fit_images': len(inner_fit), 'inner_validation_images': len(inner_valid),
               'best_inner_epoch': best_epoch, 'max_inner_epochs': args.max_epochs,
               'model_sha256': sha(model_file), 'input_manifest_sha256': sha(args.root/'INPUTS_FROZEN.json'),
               'harmonization_sha256': sha(args.harmonization),
               'architecture': 'DeepLabV3+ ResNet-50 output stride 16, ASPP 6/12/18, low-level 48-channel decoder, 24 class logits, random initialization',
               'training_only_weights_sha256': sha(args.weights),
               'loss': 'training-pixel-frequency square-root weighted cross-entropy',
               'sampling': '80% class-aware within training image, inverse sqrt of image class pixels',
               'patch_px': 512, 'batch_size': 2, 'training_patches_per_image_per_epoch': 4,
               'seed_inner': 1234, 'seed_all397': 5678}
        (args.out / 'MODEL_COMPLETE.json').write_text(json.dumps(rec, indent=2) + '\n')
        print(json.dumps(rec), flush=True)
    else:
        rec = json.loads((args.out / 'MODEL_COMPLETE.json').read_text())
        if rec['status'] != 'SAME_SPLIT_DEEPLABV3PLUS_MODEL_COMPLETE' or rec['input_manifest_sha256'] != sha(args.root/'INPUTS_FROZEN.json'):
            raise ValueError('Model/input identity mismatch')
        model = DeepLabV3Plus(classes=24).cuda().eval()
        model.load_state_dict(torch.load(args.out/'all397-final.pt', map_location='cpu', weights_only=True))
        infer_dir = args.out / 'predictions'
        infer_dir.mkdir(exist_ok=True)
        for row in evaluation:
            p = infer_dir / f'{row["case"]}.npz'
            if p.exists():
                raise FileExistsError(p)
            with Image.open(args.root/row['staged_image']) as src:
                rgb = np.asarray(src.convert('RGB'))
            w, h = row['true_working_size_wh']
            rgb = rgb[:h, :w]
            patch, stride = 512, 384
            ys = list(range(0, max(1, h-patch+1), stride)); xs = list(range(0, max(1, w-patch+1), stride))
            ys.append(max(0, h-patch)); xs.append(max(0, w-patch))
            scores = np.zeros((24, h, w), np.float32)
            count = np.zeros((h, w), np.uint16)
            start = time.monotonic()
            with torch.no_grad():
                for y in sorted(set(ys)):
                    for x in sorted(set(xs)):
                        tile = rgb[y:y+patch, x:x+patch]
                        th, tw = tile.shape[:2]
                        if th < patch or tw < patch:
                            tile = np.pad(tile, ((0, patch-th), (0, patch-tw), (0, 0)), constant_values=255)
                        tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float().unsqueeze(0).cuda()/255
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits = model(tensor)[0, :, :th, :tw].float().cpu().numpy()
                        scores[:, y:y+th, x:x+tw] += logits
                        count[y:y+th, x:x+tw] += 1
            if not np.all(count):
                raise ValueError('Uncovered inference pixel: '+row['case'])
            pred = np.argmax(scores/count[None], axis=0).astype(np.uint8) + 1
            np.savez_compressed(p, prediction=pred)
            item = {'case': row['case'], 'split': row['split'], 'status': 'PREDICTION_NO_SCORE',
                    'prediction_sha256': sha(p), 'image_sha256': row['image_sha256'],
                    'model_sha256': rec['model_sha256'], 'shape_yx': list(pred.shape),
                    'seconds': time.monotonic()-start}
            (infer_dir/f'{row["case"]}.json').write_text(json.dumps(item, indent=2)+'\n')
            print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
