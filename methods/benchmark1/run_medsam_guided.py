"""MedSAM boundary refinement from frozen U-Net boxes, without test labels."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import transform
import torch
from torch.nn import functional as F
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / 'sources/MedSAM'))
from segment_anything import sam_model_registry

ROLES = ('tumor', 'stroma', 'necrosis')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path: Path, value: dict) -> None:
    partial = path.with_name(path.name+'.partial')
    partial.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(partial, path)


@torch.no_grad()
def one_image(model, image: np.ndarray, boxes: dict) -> dict:
    h, w = image.shape[:2]
    scaled = transform.resize(image, (1024, 1024), order=3,
                              preserve_range=True, anti_aliasing=True).astype(np.uint8)
    scaled = (scaled - scaled.min()) / max(1, int(scaled.max())-int(scaled.min()))
    tensor = torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1))).float().unsqueeze(0).cuda()
    embedding = model.image_encoder(tensor)
    predictions = {}
    for role in ROLES:
        merged = np.zeros((h, w), dtype=bool)
        for item in boxes[role]:
            box = torch.tensor(item['box_xyxy'], dtype=torch.float32, device='cuda')
            box = (box / torch.tensor([w, h, w, h], device='cuda')) * 1024
            sparse, dense = model.prompt_encoder(points=None, boxes=box[None, None, :], masks=None)
            logits, _ = model.mask_decoder(image_embeddings=embedding,
                image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense, multimask_output=False)
            up = F.interpolate(torch.sigmoid(logits), size=(h, w), mode='bilinear', align_corners=False)
            merged |= up[0, 0].cpu().numpy() > 0.5
        predictions[role] = merged.astype(np.uint8)
    return predictions


def main() -> None:
    gate_path = ROOT/'MEDSAM_GPU_ADMITTED.json'
    gate = json.loads(gate_path.read_text())
    prompts_path = ROOT/'data/MEDSAM_PROMPTS_FROZEN.json'
    checkpoint = ROOT/'models/medsam_vit_b.pth'
    if gate['status'] != 'ADMITTED' or gate['prompts_sha256'] != sha(prompts_path):
        raise ValueError('MedSAM GPU admission or prompt identity differs')
    if gate['checkpoint_sha256'] != sha(checkpoint) or gate['runner_sha256'] != sha(Path(__file__)):
        raise ValueError('MedSAM source/checkpoint differs')
    prompts = json.loads(prompts_path.read_text())
    if prompts['status'] != 'MEDSAM_PROMPTS_FROZEN_NO_EVAL_LABELS' or len(prompts['cases']) != 87:
        raise ValueError('Frozen image-only prompts are incomplete')
    out = ROOT/'results/medsam-guided-v1'
    if out.exists():
        raise FileExistsError('No in-place MedSAM GPU retry')
    out.mkdir()
    Image.MAX_IMAGE_PIXELS = None
    torch.set_num_threads(4)
    atomic(out/'BUNDLE_STATE.json', {'status': 'RUNNING',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'admission_sha256': sha(gate_path)})
    try:
        model = sam_model_registry['vit_b'](checkpoint=str(checkpoint)).cuda().eval()
        rows = []
        manifest = json.loads((ROOT/'data/INPUTS_FROZEN.json').read_text())
        membership = {r['case']: r for r in manifest['rows'] if r['split'] == 'evaluation'}
        for index, item in enumerate(prompts['cases'], 1):
            case = item['case']
            row = membership[case]
            if (ROOT/'data'/row['staged_expert_mask']).exists():
                raise ValueError('Evaluation label accessible: '+case)
            image_path = ROOT/'data'/row['staged_image']
            if sha(image_path) != item['image_sha256']:
                raise ValueError('Image identity differs: '+case)
            with Image.open(image_path) as source:
                image = np.asarray(source.convert('RGB'))
            w, h = row['true_working_size_wh']
            image = image[:h, :w]
            start = time.monotonic()
            prediction = one_image(model, image, item['boxes'])
            path = out/f'{case}.npz'
            partial = out/f'{case}.npz.partial'
            with partial.open('wb') as stream:
                np.savez_compressed(stream, **prediction)
            os.replace(partial, path)
            rec = {'status': 'MEDSAM_GUIDED_PREDICTION_NO_SCORE', 'case': case,
                   'prediction_sha256': sha(path), 'image_sha256': item['image_sha256'],
                   'prompts_sha256': sha(prompts_path),
                   'box_counts': {role: len(item['boxes'][role]) for role in ROLES},
                   'inference_seconds': time.monotonic()-start,
                   'human_prompt_seconds': 0}
            receipt = out/f'{case}.json'
            atomic(receipt, rec)
            rows.append({'case': case, 'prediction_sha256': rec['prediction_sha256'],
                         'receipt_sha256': sha(receipt)})
            print(json.dumps({'index': index, 'case': case,
                              'box_counts': rec['box_counts'],
                              'inference_seconds': rec['inference_seconds']}), flush=True)
        atomic(out/'BUNDLE_COMPLETE.json', {'status': 'GPU_WORK_COMPLETE_PENDING_INDEPENDENT_AUDIT',
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'admission_sha256': sha(gate_path), 'prediction_count': len(rows),
            'predictions': rows})
    except BaseException as error:
        atomic(out/'BUNDLE_FAILED.json', {'status': 'FAILED_LATCHED_NO_IN_PLACE_RETRY',
            'error': repr(error), 'failed_utc': datetime.now(timezone.utc).isoformat(),
            'admission_sha256': sha(gate_path)})
        raise


if __name__ == '__main__':
    main()
