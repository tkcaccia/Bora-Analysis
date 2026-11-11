"""Bounded, image-only inference on context-overlapped full-slide tiles."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import torch

HERE = Path(__file__).resolve().parent
assert len(sys.argv) == 2, 'Pass one frozen slide slug'
assert sys.argv[1] != 'pm016', 'PM016 is excluded by current project policy'
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark2-unified-wsi-panels-2026-11-11-v18/full-wsi-comparators') / sys.argv[1]
B1 = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/comparators-2026-11-07-v1')
U = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/comparators-2026-11-07-v3')
D = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/comparators-deeplabv3plus-2026-11-08-v1')
PY = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/experiments/virchow2-inverse-2026-10-16/.venv/bin/python')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def dense(model, array):
    scores = np.zeros((24, 1024, 1024), np.float32)
    count = np.zeros((1024, 1024), np.uint8)
    for y in (0, 384, 512):
        for x in (0, 384, 512):
            a = array[y:y+512, x:x+512]
            tensor = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float()[None].cuda()/255
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(tensor)[0].float().cpu().numpy()
            scores[:, y:y+512, x:x+512] += logits
            count[y:y+512, x:x+512] += 1
    assert np.all(count)
    return (scores/count[None]).argmax(axis=0).astype(np.uint8)+1


def save_mask(method, row, mask, meta):
    out = DATA/method
    out.mkdir(exist_ok=True)
    path = out/f'{row["slug"]}.png'
    assert not path.exists()
    Image.fromarray(mask).save(path)
    atomic(out/f'{row["slug"]}.json', {'status': 'PREDICTION_NO_EXPERT_SCORE',
        'method': method, 'slug': row['slug'], 'field_sha256': row['field_sha256'],
        'prediction_sha256': sha(path), 'shape_yx': list(mask.shape), **meta})
    print(method, row['slug'], flush=True)


def main():
    gate = json.loads((DATA/'GPU_ADMITTED.json').read_text())
    assert gate['status'] == 'ADMITTED' and gate['runner_sha256'] == sha(__file__)
    assert gate['vista_adapter_sha256'] == sha(HERE/'vista-adapter/inference_utils.py')
    for name, expected in gate['plip_model_files_sha256'].items():
        assert sha(Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark2-unified-wsi-panels-2026-11-11-v18/plip-complete-tokenizer')/name) == expected
    assert gate['gpu_uuid'] in subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True)
    manifest = json.loads((DATA/'FIELDS_FROZEN.json').read_text())
    assert manifest['status'] == 'FROZEN_FULL_SLIDE_TILES' and len(manifest['fields']) > 0
    assert gate['fields_sha256'] == sha(DATA/'FIELDS_FROZEN.json')
    for row in manifest['fields']:
        assert sha(row['field']) == row['field_sha256']
    assert not (DATA/'GPU_STARTED.json').exists()
    atomic(DATA/'GPU_STARTED.json', {'status':'RUNNING', 'started':time.time(), 'pid':os.getpid(),
        'gate_sha256':sha(DATA/'GPU_ADMITTED.json')})
    torch.set_num_threads(4)
    try:
        recovered = gate.get('recovered_methods', [])
        if recovered:
            assert sys.argv[1] == 'visium_hd_human_prostate_cancer_ffpe'
            assert recovered == ['unet', 'deeplabv3plus', 'medsam']
            assert sha(DATA/'REUSED_METHODS_VERIFIED.json') == gate['reused_receipt_sha256']
        for method, code, cls, checkpoint in (
            ('unet', U/'sources/train_unet_balanced.py', 'UNet', U/'results/unet-balanced-v3/all397-final.pt'),
            ('deeplabv3plus', D/'sources/train_deeplabv3plus.py', 'DeepLabV3Plus', D/'results/deeplabv3plus-v1/all397-final.pt')):
            if method in recovered:
                continue
            assert sha(checkpoint) == gate['checkpoints'][method]
            module = load(code, method+'_module')
            model = getattr(module, cls)(classes=24).cuda().eval()
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            for row in manifest['fields']:
                array = np.asarray(Image.open(row['field']).convert('RGB'))
                start = time.monotonic()
                pred = dense(model, array)
                save_mask(method, row, pred, {'model_sha256':gate['checkpoints'][method],
                                              'seconds':time.monotonic()-start})
            del model
            torch.cuda.empty_cache()
        # MedSAM uses image-only U-Net connected-component boxes, at most
        # three per role. It is a prompted boundary result, not a full classifier.
        if 'medsam' not in recovered:
            checkpoint = B1/'models/medsam_vit_b.pth'
            assert sha(checkpoint) == gate['checkpoints']['medsam']
            sys.path.insert(0, str(B1/'sources/MedSAM'))
            from segment_anything import sam_model_registry
            from skimage.transform import resize
            from torch.nn import functional as F
            model = sam_model_registry['vit_b'](checkpoint=str(checkpoint)).cuda().eval()
            for row in manifest['fields']:
                array = np.asarray(Image.open(row['field']).convert('RGB'))
                guided = np.asarray(Image.open(DATA/'unet'/f'{row["slug"]}.png'))
                input_hash = sha(DATA/'unet'/f'{row["slug"]}.png')
                scaled = resize(array, (1024,1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
                scaled = (scaled-scaled.min())/max(1, int(scaled.max())-int(scaled.min()))
                tensor = torch.from_numpy(np.ascontiguousarray(scaled.transpose(2,0,1))).float()[None].cuda()
                start = time.monotonic()
                with torch.no_grad():
                    embedding = model.image_encoder(tensor)
                    output = np.zeros((1024,1024), np.uint8)
                    boxes = {}
                    for name, code in [('tumor',1),('stroma',3),('necrosis',5)]:
                        labels, count = ndimage.label(guided == code)
                        sizes = np.bincount(labels.ravel(), minlength=count+1)
                        parts = sorted([(int(sizes[i]),i) for i in range(1,count+1)], reverse=True)
                        selected = []
                        for area, i in parts[:3]:
                            if area < 256: continue
                            yy,xx = np.where(labels==i)
                            selected.append([int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)])
                        boxes[name] = selected
                        for box in selected:
                            b = torch.tensor(box, dtype=torch.float32, device='cuda')[None,None]
                            sparse,dense_prompt = model.prompt_encoder(points=None, boxes=b, masks=None)
                            logits,_ = model.mask_decoder(image_embeddings=embedding,
                                image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
                                dense_prompt_embeddings=dense_prompt, multimask_output=False)
                            binary = F.interpolate(torch.sigmoid(logits), size=(1024,1024), mode='bilinear', align_corners=False)[0,0].cpu().numpy()>.5
                            output[binary] = code
                save_mask('medsam',row,output,{'model_sha256':gate['checkpoints']['medsam'],
                    'unet_prompt_mask_sha256':input_hash,'boxes':boxes,'seconds':time.monotonic()-start})
            del model
            torch.cuda.empty_cache()
        # VISTA-PATH processes every frozen full-slide field, prompt-free.
        assert sha(B1/'models/vista_pytorch_model.bin') == gate['checkpoints']['vista_path']
        env = dict(os.environ, PYTHONPATH=f'{B1}/deps-vista:{B1}/sources/VISTA-PATH',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        names = ('tumor','stroma','lymphocytic_infiltrate','necrosis_or_debris','blood',
            'metaplasia_NOS','fat','plasma_cells','other_immune_infiltrate','lymphatics',
            'blood_vessel','normal_acinus_or_duct','dcis')
        out = DATA/'vista_path'
        out.mkdir(exist_ok=False)
        cmd = [str(PY),str(HERE/'vista-adapter/inference.py'),
            '--checkpoint_file',str(B1/'models/vista_pytorch_model.bin'),
            '--image_dir',str(DATA/'fields'),'--infer_vis_dir',str(out),
            '--class_names',*names,'--bbx_random','1','--num_queries','20',
            '--m2f_image_size','512','--crop_size','1024','--overlap','128',
            '--fg_thresh','0.05','--batch_size','1','--num_workers','0','--save_mask']
        subprocess.run(cmd,check=True,timeout=21600,env=env)
        for row in manifest['fields']:
            path = out/f'{row["slug"]}.png'
            assert path.exists() and Image.open(path).size == (1024,1024)
            atomic(out/f'{row["slug"]}.json',{'status':'PREDICTION_NO_EXPERT_SCORE',
                'method':'vista_path','slug':row['slug'],'field_sha256':row['field_sha256'],
                'prediction_sha256':sha(path),'model_sha256':gate['checkpoints']['vista_path']})
        atomic(DATA/'GPU_COMPLETE.json',{'status':'COMPLETE_PENDING_INDEPENDENT_AUDIT',
            'finished':time.time(),'tile_count':len(manifest['fields']),'method_count':4,
            'recovered_methods':recovered})
    except BaseException as exc:
        atomic(DATA/'GPU_FAILED.json',{'status':'FAILED_LATCHED_NO_IN_PLACE_RETRY',
            'failed':time.time(),'error':repr(exc)})
        raise


if __name__ == '__main__':
    main()
