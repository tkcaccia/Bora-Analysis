"""Source-only prompted MedSAM tissue regions on one CAM test tile range."""
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
import rasterio
from rasterio.windows import Window
from PIL import Image
import torch
from torch.nn import functional as F

ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora')
DATA=Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
HERE=Path(__file__).resolve().parent
SOURCE=DATA/'comparators-2026-11-07-v1/sources/MedSAM'
PROMPTS=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v120/MEDSAM_PROMPTS_FROZEN.json'
PROMPT_AUDIT=ROOT/'migration/bora-camelyon-five-slide-protocol-2026-11-14-v121/PROMPT_AUDIT_VERIFIED.json'
sys.path.insert(0,str(SOURCE))
from segment_anything import sam_model_registry


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def tree_sha():
    h=hashlib.sha256()
    for path in sorted((SOURCE/'segment_anything').rglob('*.py')):
        h.update(str(path.relative_to(SOURCE)).encode())
        h.update(bytes.fromhex(sha(path)))
    return h.hexdigest()


def atomic(path,value):
    partial=path.with_name(path.name+'.partial')
    partial.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(partial,path)


@torch.inference_mode()
def infer_tile(model,image,boxes):
    height,width=image.shape[:2]
    scaled=np.asarray(Image.fromarray(image).resize((1024,1024),Image.Resampling.BICUBIC),dtype=np.float32)
    scaled=(scaled-scaled.min())/max(1.0,float(scaled.max()-scaled.min()))
    tensor=torch.from_numpy(np.ascontiguousarray(scaled.transpose(2,0,1))).unsqueeze(0).cuda()
    embedding=model.image_encoder(tensor)
    merged=np.zeros((height,width),dtype=bool)
    for item in boxes:
        box=torch.tensor(item['box_xyxy_native_crop'],dtype=torch.float32,device='cuda')
        box*=torch.tensor([1024/width,1024/height,1024/width,1024/height],device='cuda')
        sparse,dense=model.prompt_encoder(points=None,boxes=box[None,None,:],masks=None)
        logits,_=model.mask_decoder(image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,
            multimask_output=False)
        probability=F.interpolate(torch.sigmoid(logits),size=(height,width),
                                  mode='bilinear',align_corners=False)
        merged|=probability[0,0].cpu().numpy()>.5
    return merged[4::8,4::8].astype(np.uint8)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--slide',choices=('tumor_084','tumor_091','normal_019'),required=True)
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--stop',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    freeze_path=HERE/'MEDSAM_SOURCE_FROZEN.json'
    freeze=json.loads(freeze_path.read_text())
    if sha(__file__)!=freeze['source_sha256'] or sha(PROMPTS)!=freeze['prompt_manifest_sha256']:
        raise RuntimeError('Frozen source or prompt manifest changed')
    if sha(PROMPT_AUDIT)!=freeze['prompt_audit_sha256'] or tree_sha()!=freeze['medsam_source_tree_sha256']:
        raise RuntimeError('Independent prompt audit or MedSAM source differs')
    prompts=json.loads(PROMPTS.read_text())
    if sha(prompts['model_checkpoint_path'])!=freeze['checkpoint_sha256']:
        raise RuntimeError('MedSAM checkpoint differs')
    slide=next(s for s in prompts['slides'] if s['slide']==args.slide)
    if not (0<=args.start<args.stop<=slide['prompted_tiles'] and args.stop-args.start<=freeze['maximum_tiles_per_child']):
        raise ValueError('Tile range exceeds frozen bound')
    if args.output.exists():
        raise FileExistsError('No in-place MedSAM tile retry')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    started=time.monotonic()
    try:
        model=sam_model_registry['vit_b'](checkpoint=prompts['model_checkpoint_path']).cuda().eval()
        with rasterio.open(slide['calibrated_image_path']) as source:
            height,width=slide['shape_yx_4um']
            if (source.height,source.width,source.count)!=(height*8,width*8,3):
                raise RuntimeError('Calibrated source geometry differs')
            for index in range(args.start,args.stop):
                began=time.monotonic()
                tile=slide['tiles'][index]
                y0,x0=tile['origin_yx_4um'];th,tw=tile['shape_yx_4um']
                crop=np.moveaxis(source.read(window=Window(x0*8,y0*8,tw*8,th*8)),0,-1)
                if crop.shape!=(th*8,tw*8,3):
                    raise RuntimeError('Incomplete H&E crop')
                mask=infer_tile(model,crop,tile['boxes'])
                if mask.shape!=(th,tw) or np.any((mask!=0)&(mask!=1)):
                    raise RuntimeError('MedSAM sampled mask geometry differs')
                folder=args.output/f'tile-{index:04d}'
                folder.mkdir(exist_ok=False)
                path=folder/'metastasis-4um.npy'
                temporary=path.with_name(path.name+'.partial')
                with temporary.open('wb') as handle:
                    np.save(handle,mask,allow_pickle=False)
                    handle.flush();os.fsync(handle.fileno())
                os.replace(temporary,path)
                record={'status':'CAM_TEST_MEDSAM_PROMPTED_TILE_COMPLETE_SOURCE_ONLY',
                        'slide':args.slide,'prompt_tile_index':index,
                        'origin_yx_4um':[y0,x0],'shape_yx_4um':[th,tw],
                        'mask_path':str(path),'mask_sha256':sha(path),
                        'metastasis_pixels_4um':int(mask.sum()),
                        'box_count':len(tile['boxes']),'boxes':tile['boxes'],
                        'checkpoint_sha256':freeze['checkpoint_sha256'],
                        'prompt_manifest_sha256':freeze['prompt_manifest_sha256'],
                        'source_sha256':sha(__file__),
                        'medsam_source_tree_sha256':freeze['medsam_source_tree_sha256'],
                        'seconds':time.monotonic()-began,
                        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'peak_cuda_bytes':int(torch.cuda.max_memory_allocated()),
                        'test_reference_opened':False}
                atomic(folder/'TILE_COMPLETE.json',record)
                print(json.dumps({'slide':args.slide,'tile':index,'seconds':record['seconds']}),flush=True)
        torch.cuda.synchronize()
        atomic(args.output/'RANGE_COMPLETE.json',{'status':'CAM_TEST_MEDSAM_TILE_RANGE_COMPLETE',
            'slide':args.slide,'start':args.start,'stop':args.stop,'tiles':args.stop-args.start,
            'source_sha256':sha(__file__),'freeze_sha256':sha(freeze_path),
            'seconds':time.monotonic()-started,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'peak_cuda_bytes':int(torch.cuda.max_memory_allocated()),
            'test_reference_opened':False})
    except BaseException as error:
        atomic(args.output/'FAILED.json',{'status':'FAILED_LATCHED_PRESERVE_PARTIAL',
            'slide':args.slide,'start':args.start,'stop':args.stop,
            'error':repr(error),'test_reference_opened':False})
        raise


if __name__=='__main__':
    main()
