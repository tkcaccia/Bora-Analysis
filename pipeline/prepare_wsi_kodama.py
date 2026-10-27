#!/usr/bin/env python3
"""Prepare a fixed annotation-free WSI tile sample for nonspatial KODAMA.

Coordinates are retained only for projecting the eventual cluster IDs back to
slide tiles. Neither spatial coordinates nor expert labels enter the scaler,
PCA or KODAMA input. KODAMA itself is run by run_wsi_kodama.R.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import tempfile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(pooled_dir: Path, output: Path, sample_count: int, seed: int) -> dict:
    pooled_dir = pooled_dir.resolve(); output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((pooled_dir / 'POOLED_COMPLETE.json').read_text())
    if receipt['format'] != 'bora-wsi-pooled-virchow2-v1' or receipt['expert_annotation_used']:
        raise ValueError('Expected annotation-free pooled Virchow2 features')
    for name, expected in receipt['files'].items():
        if sha256(pooled_dir / name) != expected:
            raise ValueError(f'Pooled feature checksum mismatch: {name}')
    x = np.load(pooled_dir / 'pooled.npy', mmap_mode='r', allow_pickle=False)
    valid = np.load(pooled_dir / 'valid.npy', allow_pickle=False)
    if x.shape != (receipt['grid_tiles'], 1280) or valid.shape != (len(x),):
        raise ValueError('Pooled feature grid differs')
    candidates = np.flatnonzero(valid)
    if not 100 <= sample_count <= len(candidates):
        raise ValueError('Require at least 100 sampled tissue tiles')
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(candidates, size=sample_count, replace=False))
    with threadpool_limits(limits=2):
        scaler = StandardScaler().fit(x[candidates])
        scaler.scale_ = np.maximum(scaler.scale_, .05)
        z_sample = scaler.transform(x[sample]).astype(np.float32)
        pca = PCA(n_components=30, svd_solver='randomized', random_state=seed).fit(z_sample)
        all_pca = np.zeros((len(x), 30), dtype=np.float32)
        for first in range(0, len(candidates), 1024):
            ids = candidates[first:first + 1024]
            all_pca[ids] = pca.transform(scaler.transform(x[ids])).astype(np.float32)
    if not np.isfinite(all_pca[candidates]).all():
        raise ValueError('Nonfinite projected tile feature')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.wsi-kodama-', dir=output.parent) as tmp:
        folder = Path(tmp)
        np.save(folder / 'sample_indices.npy', sample)
        np.save(folder / 'all_pca.npy', all_pca)
        np.savez_compressed(folder / 'pca_model.npz', scaler_mean=scaler.mean_,
                            scaler_scale=scaler.scale_, pca_mean=pca.mean_,
                            pca_components=pca.components_)
        np.savetxt(folder / 'sample_pca.csv', all_pca[sample], delimiter=',',
                   header=','.join(f'PC{i}' for i in range(1, 31)), comments='')
        record = dict(format='bora-wsi-kodama-input-v1', pooled_receipt_sha256=sha256(pooled_dir / 'POOLED_COMPLETE.json'),
                      image_sha256=receipt['image_sha256'], encoder_revision=receipt['encoder_revision'],
                      encoder_sha256=receipt['encoder_sha256'], sample_count=sample_count,
                      sample_seed=seed, input_dimensions=1280, kodama_dimensions=30,
                      spatial_coordinates_used_for_embedding=False,
                      expert_annotation_used=False,
                      files={name: sha256(folder / name) for name in
                             ('sample_indices.npy','all_pca.npy','pca_model.npz','sample_pca.csv')})
        (folder / 'INPUT_COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
        os.replace(folder, output)
    return record


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pooled',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-count',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261027)
    a=p.parse_args()
    print(json.dumps(prepare(a.pooled,a.output,a.sample_count,a.seed),indent=2))


if __name__=='__main__':
    main()
