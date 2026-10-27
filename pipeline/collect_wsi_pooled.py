#!/usr/bin/env python3
"""Verify completed Virchow2 WSI units and collect pooled tile features.

This reads only the feature receipts and .npz chunks, never the expert annotation.
The original spatial tokens stay in their source chunks; only the 1280-wide
pooled descriptors and tile coordinates are materialized for clustering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def collect(root: Path, output: Path) -> dict:
    root = root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    units = sorted(root.glob('unit-*/FEATURES_COMPLETE.json'))
    if not units:
        raise FileNotFoundError('No completed Virchow2 units')
    receipts = [json.loads(path.read_text()) for path in units]
    keys = ('image_sha256', 'plan_sha256', 'tile_table_sha256',
            'encoder_revision', 'encoder_sha256', 'grid_tiles', 'native_tile_wh')
    exemplar = receipts[0]
    if any(any(r.get(k) != exemplar.get(k) for k in keys) for r in receipts):
        raise ValueError('Unit image, plan, grid or encoder identity differs')
    if any(r.get('expert_annotation_used') is not False for r in receipts):
        raise ValueError('Feature receipts must exclude the expert annotation')
    if exemplar.get('target_mpp') != 0.5 or exemplar.get('tile_pixels') != 224:
        raise ValueError('Expected calibrated 224-pixel Virchow2 tiles at 0.5 um/pixel')
    total = int(exemplar['grid_tiles'])
    ordered = sorted(zip(units, receipts), key=lambda pair: int(pair[1]['first_tile']))
    cursor = 0
    for _, receipt in ordered:
        first, last = int(receipt['first_tile']), int(receipt['last_tile'])
        if first != cursor or not first < last <= total:
            raise ValueError('Unit ranges have a gap, overlap or invalid endpoint')
        cursor = last
    if cursor != total:
        raise ValueError('Virchow2 units do not cover the WSI grid')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.wsi-pooled-', dir=output.parent))
    try:
        pooled = np.lib.format.open_memmap(temporary / 'pooled.npy', mode='w+',
                                           dtype=np.float32, shape=(total, 1280))
        table = np.lib.format.open_memmap(temporary / 'tile_table.npy', mode='w+',
                                          dtype=np.int64, shape=(total, 5))
        valid = np.zeros(total, dtype=np.bool_)
        encoded = 0
        chunks = 0
        for receipt_path, receipt in ordered:
            folder = receipt_path.parent
            for number in range(int(receipt['chunks'])):
                chunk = folder / f'chunk-{number:05d}.npz'
                audit = json.loads(chunk.with_suffix('.json').read_text())
                if sha256(chunk) != audit['sha256']:
                    raise ValueError(f'Feature chunk checksum mismatch: {chunk}')
                with np.load(chunk, allow_pickle=False) as data:
                    x = data['pooled']
                    rows = data['tile_table']
                if x.shape != (len(rows), 1280) or rows.shape != (len(rows), 5):
                    raise ValueError('Pooled feature or tile table shape differs')
                if x.dtype.kind != 'f' or not np.isfinite(x).all():
                    raise ValueError('Nonfinite pooled Virchow2 feature')
                ids = rows[:, 0].astype(np.int64)
                if (np.any(ids < int(receipt['first_tile'])) or
                    np.any(ids >= int(receipt['last_tile'])) or
                    np.any(valid[ids]) or len(np.unique(ids)) != len(ids)):
                    raise ValueError('Tile IDs are duplicated or outside their unit')
                pooled[ids] = x
                table[ids] = rows
                valid[ids] = True
                encoded += len(ids)
                chunks += 1
                if int(audit['encoded_tiles']) != len(ids):
                    raise ValueError('Chunk receipt tile count differs')
            if int(receipt['encoded_tissue_tiles']) != int(valid[int(receipt['first_tile']):int(receipt['last_tile'])].sum()):
                raise ValueError('Unit receipt tile count differs')
        if encoded != sum(int(r['encoded_tissue_tiles']) for r in receipts):
            raise ValueError('Total encoded tile count differs')
        pooled.flush(); table.flush()
        np.save(temporary / 'valid.npy', valid)
        record = dict(format='bora-wsi-pooled-virchow2-v1', source_root=str(root),
                      image_sha256=exemplar['image_sha256'],
                      plan_sha256=exemplar['plan_sha256'],
                      tile_table_sha256=exemplar['tile_table_sha256'],
                      encoder_revision=exemplar['encoder_revision'],
                      encoder_sha256=exemplar['encoder_sha256'],
                      grid_tiles=total, encoded_tiles=encoded,
                      units=len(units), chunks=chunks, target_mpp=0.5,
                      tile_pixels=224, native_tile_wh=exemplar['native_tile_wh'],
                      expert_annotation_used=False,
                      files={name: sha256(temporary / name)
                             for name in ('pooled.npy', 'tile_table.npy', 'valid.npy')})
        (temporary / 'POOLED_COMPLETE.json').write_text(json.dumps(record, indent=2) + '\n')
        os.replace(temporary, output)
        return record
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.features_root, args.output), indent=2))


if __name__ == '__main__':
    main()
