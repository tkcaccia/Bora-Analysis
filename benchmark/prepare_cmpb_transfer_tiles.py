"""CPU-only source-to-target tile prediction from frozen Virchow2 tokens.

Reads an expert annotation for the declared training source only. Target
annotations are neither opened nor used to select a model or threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


def tile_features(tokens: np.ndarray, shape_yx: tuple[int, int], tile_px: int) -> np.ndarray:
    h, w = shape_yx
    ty, tx = (h + tile_px - 1) // tile_px, (w + tile_px - 1) // tile_px
    result = np.empty((ty, tx, tokens.shape[-1]), np.float32)
    for y in range(ty):
        y0, y1 = y * tile_px, min((y + 1) * tile_px, h)
        iy0, iy1 = y0 // 14, (y1 - 1) // 14 + 1
        yw = np.minimum(y1, (np.arange(iy0, iy1) + 1) * 14) - np.maximum(y0, np.arange(iy0, iy1) * 14)
        for x in range(tx):
            x0, x1 = x * tile_px, min((x + 1) * tile_px, w)
            ix0, ix1 = x0 // 14, (x1 - 1) // 14 + 1
            xw = np.minimum(x1, (np.arange(ix0, ix1) + 1) * 14) - np.maximum(x0, np.arange(ix0, ix1) * 14)
            weights = yw[:, None] * xw[None, :]
            result[y, x] = np.einsum('ij,ijc->c', weights, tokens[iy0:iy1, ix0:ix1].astype(np.float32)) / weights.sum()
    return result


def tile_targets(reference: np.ndarray, tissue: np.ndarray, tile_px: int, mapping: dict[int, int], purity: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = reference.shape
    ty, tx = (h + tile_px - 1) // tile_px, (w + tile_px - 1) // tile_px
    labels = np.zeros((ty, tx), np.uint8)
    good = np.zeros((ty, tx), bool)
    for y in range(ty):
        for x in range(tx):
            ys, xs = slice(y*tile_px, min((y+1)*tile_px,h)), slice(x*tile_px,min((x+1)*tile_px,w))
            support = tissue[ys,xs] > 0
            if support.mean() < .5:
                continue
            raw = reference[ys,xs][support]
            count = {role: int(np.sum(raw == value)) for value, role in mapping.items()}
            if not count:
                continue
            label = max(sorted(count), key=count.get)
            if count[label] / max(1,len(raw)) >= purity:
                labels[y,x] = label
                good[y,x] = True
    return labels, good


def read_token_case(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(row['original_path']).convert('RGB'))
    tissue = tifffile.imread(row['tissue_support_path']) > 0
    tokens = np.load(row['feature']['path'], mmap_mode='r', allow_pickle=False)
    assert image.shape[:2] == tissue.shape
    assert tokens.shape == ((image.shape[0]+13)//14, (image.shape[1]+13)//14, 1280)
    return image, tissue, tokens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--targets', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from bora._vendor.fastpls import PLS, __version__
    assert __version__ == '0.3.0'
    registry = {r['case']: r for r in json.loads(a.registry.read_text())['cases']}
    source = registry[a.source]
    assert source['split'] == 'train'
    assert all(registry[c]['split'] != 'train' and registry[c]['group'] != source['group'] for c in a.targets)
    mapping = {int(value): int(role) for value, role in source['ontology_raw_to_role'].items()}
    assert set(mapping.values()) == {1,3}, 'One-source pilot requires tumor and stroma only'
    image, tissue, tokens = read_token_case(source)
    reference = tifffile.imread(source['reference_path'])
    assert reference.shape == image.shape[:2]
    tile_px = int(source['sample']['coarse_tile_px'])
    features = tile_features(tokens, reference.shape, tile_px)
    labels, good = tile_targets(reference, tissue, tile_px, mapping, purity=.8)
    x, y = features[good].astype(np.float64), labels[good]
    assert len(x) >= 16 and set(np.unique(y)) == {1,3}
    model = PLS(n_components=1, method='simpls', classifier='lda', scaling='autoscaling',
                backend='cpu', seed=314159).fit(np.asfortranarray(x), y)
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name in a.targets:
        row = registry[name]
        target_image, target_tissue, target_tokens = read_token_case(row)
        assert abs(float(row['sample']['working_mpp_xy'][0])-float(source['sample']['working_mpp_xy'][0])) < .01
        tile_px_target = int(row['sample']['coarse_tile_px'])
        assert tile_px_target == tile_px
        target_x = tile_features(target_tokens, target_image.shape[:2], tile_px)
        predictions = np.asarray(model.predict(np.asfortranarray(target_x.reshape(-1,1280).astype(np.float64))), np.uint8).reshape(target_x.shape[:2])
        assert set(np.unique(predictions)) <= {1,3}
        anonymous = np.where(predictions == 1, 1, 2).astype(np.uint8)
        # Tile support is derived from H&E; no target reference is consulted.
        for y0 in range(anonymous.shape[0]):
            for x0 in range(anonymous.shape[1]):
                ys, xs = slice(y0*tile_px,min((y0+1)*tile_px,target_tissue.shape[0])),slice(x0*tile_px,min((x0+1)*tile_px,target_tissue.shape[1]))
                if target_tissue[ys,xs].mean() < .5:
                    anonymous[y0,x0] = 0
        folder = a.output/name
        folder.mkdir()
        np.save(folder/'anonymous_tiles.npy', anonymous)
        file_hash = hashlib.sha256((folder/'anonymous_tiles.npy').read_bytes()).hexdigest()
        records.append({'case':name,'source_group':row['group'], 'tile_shape_yx':list(anonymous.shape),
                        'positive_tiles':int(np.count_nonzero(anonymous)),
                        'predicted_groups':sorted(int(v) for v in np.unique(anonymous) if v),
                        'tile_file_sha256':file_hash,'target_reference_read':False})
    report={'source_case':a.source,'source_group':source['group'],'source_split':'train',
            'training_tiles':len(x),'training_class_counts':{str(int(v)):int(np.sum(y==v)) for v in np.unique(y)},
            'tile_purity_threshold':.8,'target_reference_read':False,'tile_px':tile_px,
            'micrometres_per_tile':float(tile_px*source['sample']['working_mpp_xy'][0]),
            'fastpls_version':__version__,'records':records}
    (a.output/'PREDICTIONS_FROZEN.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
