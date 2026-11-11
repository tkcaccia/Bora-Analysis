"""Freeze U-Net-derived MedSAM boxes without opening evaluation masks."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects, label

ROLES = ('tumor', 'stroma', 'necrosis')
MIN_AREA = 100
MAX_COMPONENTS = 6
BOX_PAD_FRACTION = 0.08


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(root: Path) -> None:
    manifest_path = root/'data/INPUTS_FROZEN.json'
    model_path = root/'results/unet-v1/MODEL_COMPLETE.json'
    complete_path = root/'results/unet-v1/BUNDLE_COMPLETE.json'
    target = root/'data/MEDSAM_PROMPTS_FROZEN.json'
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads(manifest_path.read_text())
    model = json.loads(model_path.read_text())
    complete = json.loads(complete_path.read_text())
    if (model['status'] != 'SAME_SPLIT_UNET_MODEL_COMPLETE' or
            complete['status'] != 'GPU_WORK_COMPLETE_PENDING_INDEPENDENT_AUDIT'):
        raise ValueError('U-Net predictions are incomplete')
    class_ids = {name: int(i) for i, name in manifest['class_names'].items()}
    if any(role not in class_ids for role in ROLES):
        raise ValueError('Prespecified role missing from frozen ontology')
    rows = [r for r in manifest['rows'] if r['split'] == 'evaluation']
    if len(rows) != 87:
        raise ValueError('Frozen evaluation count changed')
    cases = []
    for row in rows:
        case = row['case']
        if (root/'data'/row['staged_expert_mask']).exists():
            raise ValueError('Evaluation label accessible: '+case)
        prediction_path = root/'results/unet-v1/predictions'/f'{case}.npz'
        prediction_receipt = root/'results/unet-v1/predictions'/f'{case}.json'
        rec = json.loads(prediction_receipt.read_text())
        if rec['status'] != 'PREDICTION_NO_SCORE' or rec['prediction_sha256'] != sha(prediction_path):
            raise ValueError('U-Net prediction identity differs: '+case)
        with np.load(prediction_path, allow_pickle=False) as z:
            prediction = z['prediction']
        w, h = row['true_working_size_wh']
        if prediction.shape != (h, w):
            raise ValueError('U-Net prediction frame differs: '+case)
        start = time.monotonic()
        prompts = {}
        for role in ROLES:
            segments, count = label(prediction == class_ids[role], structure=np.ones((3, 3), np.uint8))
            slices = find_objects(segments)
            components = []
            for number, piece in enumerate(slices, 1):
                if piece is None:
                    continue
                area = int(np.count_nonzero(segments[piece] == number))
                if area < MIN_AREA:
                    continue
                ys, xs = piece
                pad = int(np.ceil(BOX_PAD_FRACTION * max(xs.stop-xs.start, ys.stop-ys.start)))
                box = [max(0, xs.start-pad), max(0, ys.start-pad),
                       min(w, xs.stop+pad), min(h, ys.stop+pad)]
                components.append({'predicted_component_pixels': area, 'box_xyxy': box})
            components.sort(key=lambda x: (-x['predicted_component_pixels'], x['box_xyxy']))
            prompts[role] = components[:MAX_COMPONENTS]
        cases.append({'case': case, 'dataset': row['dataset'],
                      'image_sha256': row['image_sha256'],
                      'unet_prediction_sha256': rec['prediction_sha256'],
                      'unet_prediction_receipt_sha256': sha(prediction_receipt),
                      'prompt_build_seconds': time.monotonic()-start,
                      'boxes': prompts})
    payload = {'status': 'MEDSAM_PROMPTS_FROZEN_NO_EVAL_LABELS',
               'method': 'U-Net-guided MedSAM boundary refinement, not standalone MedSAM',
               'roles': ROLES, 'min_component_area_px': MIN_AREA,
               'max_components_per_role': MAX_COMPONENTS,
               'box_pad_fraction_of_larger_side': BOX_PAD_FRACTION,
               'human_prompt_seconds': 0, 'reference_masks_read': False,
               'input_manifest_sha256': sha(manifest_path),
               'unet_model_receipt_sha256': sha(model_path),
               'unet_bundle_receipt_sha256': sha(complete_path),
               'cases': cases}
    partial = target.with_name(target.name+'.partial')
    partial.write_text(json.dumps(payload, indent=2)+'\n')
    os.replace(partial, target)
    counts = {role: sum(len(c['boxes'][role]) for c in cases) for role in ROLES}
    print(json.dumps({'status': payload['status'], 'images': len(cases), 'box_counts': counts}))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
