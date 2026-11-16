#!/usr/bin/env python3
"""Bounded, source-only VISTA-PATH inference on CAM test-slide tiles.

The released thirteen text prompts and 1024-to-512 image preprocessing are
unchanged. Each predicted 1024-pixel crop is sampled at 4-um pixel centres;
tile probabilities are retained before global overlap blending and Otsu.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, AutoImageProcessor

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE = Path(__file__).resolve().parent
ADAPTER = ROOT / 'migration/external-benchmark-wsss-camelyon-2026-11-13-v24/vista-adapter'
sys.path.insert(0, str(ADAPTER))
from inference_utils import (BASE_MODEL_NAME, MASK2FORMER_NAME, encode_class_texts,
                             load_model, set_seed)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def save_array(path, value):
    temp = path.with_name(path.name + '.partial')
    with temp.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def infer(slide, start, stop, output):
    freeze_path = HERE / 'VISTA_SOURCE_FROZEN.json'
    freeze = json.loads(freeze_path.read_text())
    if sha(__file__) != freeze['tile_source_sha256'] or sha(ADAPTER / 'inference_utils.py') != freeze['vista_utils_sha256']:
        raise RuntimeError('Frozen VISTA source differs')
    checkpoint = Path(freeze['checkpoint_path'])
    if sha(checkpoint) != freeze['checkpoint_sha256']:
        raise RuntimeError('VISTA checkpoint differs')
    if slide not in freeze['slides']:
        raise ValueError('Slide outside frozen pair')
    settings = freeze['settings']
    row = freeze['slides'][slide]
    if sha(row['calibrated_state_path']) != row['calibrated_state_sha256']:
        raise RuntimeError('Calibrated image provenance changed')
    if sha(row['support_receipt_path']) != row['support_receipt_sha256']:
        raise RuntimeError('H&E support provenance changed')
    if not (0 <= start < stop <= len(row['selected_tiles'])) or stop-start > freeze['maximum_tiles_per_child']:
        raise ValueError('Tile range outside frozen bound')
    output = Path(output)
    if output.exists():
        raise FileExistsError('No in-place inference retry')
    output.mkdir(parents=True)
    set_seed(42)
    torch.set_num_threads(4)
    class Args:
        checkpoint_file = str(checkpoint)
        num_queries = settings['num_queries']
        m2f_image_size = settings['m2f_image_size']
        tune_mode = 'freeze'
        d_model = 512
        nhead = 8
        num_layers = 4
        bbx_random = settings['bbx_random']
    model = load_model(Args(), torch.device('cuda'))
    processor = CLIPProcessor.from_pretrained(BASE_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(
        MASK2FORMER_NAME, do_resize=False, do_reduce_labels=False, ignore_index=255)
    names = freeze['ordered_class_names']
    tokens = encode_class_texts(processor, {i+1:name for i,name in enumerate(names)}, range(1,len(names)+1))
    t0 = time.monotonic()
    with rasterio.open(row['calibrated_image_path']) as image:
        if (image.height, image.width, image.count) != (*row['calibrated_shape_yx'], 3):
            raise RuntimeError('Calibrated RGB geometry differs')
        with torch.no_grad():
            for selected_index in range(start, stop):
                tile = row['selected_tiles'][selected_index]
                y, x = tile['origin_yx_native']
                patch = np.moveaxis(image.read(window=Window(x,y,1024,1024)),0,-1)
                if patch.shape != (1024,1024,3):
                    raise RuntimeError('Incomplete native RGB crop')
                rgb512 = cv2.resize(patch,(512,512),interpolation=cv2.INTER_LINEAR)
                values = image_processor(images=rgb512,return_tensors='pt')['pixel_values'].cuda()
                probs4 = np.empty((len(names),128,128),dtype=np.float32)
                for i in range(1,len(names)+1):
                    ids,attn=tokens[i]
                    box=torch.tensor([[0.,0.,512.,512.]],device='cuda')
                    prediction=model(values,ids[None].cuda(),attn[None].cuda(),labels=None,box=box)
                    p=F.softmax(prediction['logits'],dim=1)[0,1].float().cpu().numpy()
                    full=cv2.resize(p,(1024,1024),interpolation=cv2.INTER_LINEAR)
                    probs4[i-1]=full[4::8,4::8]
                folder=output/f'tile-{selected_index:04d}'
                folder.mkdir(exist_ok=False)
                path=folder/'probabilities-13x128x128.npy'
                save_array(path,probs4)
                receipt={'status':'CAM_VISTA_SOURCE_ONLY_NATIVE_TILE_COMPLETE',
                         'slide':slide,'selected_tile_index':selected_index,
                         'origin_yx_native':[y,x],'shape_yx_native':[1024,1024],
                         'origin_yx_4um':[y//8,x//8],'shape_yx_4um':[128,128],
                         'prompt_names':names,'checkpoint_sha256':freeze['checkpoint_sha256'],
                         'calibrated_image_sha256':row['calibrated_image_sha256'],
                         'support_receipt_sha256':row['support_receipt_sha256'],
                         'source_sha256':sha(__file__),'freeze_sha256':sha(freeze_path),
                         'probabilities_path':str(path),'probabilities_sha256':sha(path),
                         'probability_min':float(probs4.min()),'probability_max':float(probs4.max()),
                         'expert_reference_opened':False}
                atomic_json(folder/'TILE_COMPLETE.json',receipt)
                print(json.dumps({'slide':slide,'tile':selected_index,'seconds':round(time.monotonic()-t0,2)}),flush=True)
    atomic_json(output/'RANGE_COMPLETE.json',{'status':'CAM_VISTA_TILE_RANGE_COMPLETE',
        'slide':slide,'start':start,'stop':stop,'tiles':stop-start,
        'source_sha256':sha(__file__),'freeze_sha256':sha(freeze_path),
        'seconds':time.monotonic()-t0,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_cuda_bytes':int(torch.cuda.max_memory_allocated()),'expert_reference_opened':False})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--slide',choices=('tumor_084','tumor_091','normal_019'),required=True)
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--stop',type=int,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    infer(args.slide,args.start,args.stop,args.output)


if __name__=='__main__':
    main()
