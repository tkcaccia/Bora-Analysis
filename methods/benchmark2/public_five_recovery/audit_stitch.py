"""Independently verify one terminal slide and assemble complete class TIFFs."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyvips
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark2-unified-wsi-panels-2026-11-11-v18/full-wsi-comparators')
METHODS=('unet','deeplabv3plus','medsam','vista_path')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()


def main(slug):
    assert slug != 'pm016', 'PM016 is excluded by current project policy'
    out=ROOT/slug
    target=out/'FULL_SLIDE_VERIFIED.json'
    assert not target.exists()
    gate=json.loads((out/'GPU_ADMITTED.json').read_text())
    runner='run_gpu.py'
    assert gate['runner_sha256']==sha(HERE/runner)
    manifest=json.loads((out/'FIELDS_FROZEN.json').read_text())
    assert gate['fields_sha256']==sha(out/'FIELDS_FROZEN.json')
    assert json.loads((out/'GPU_COMPLETE.json').read_text())['status']=='COMPLETE_PENDING_INDEPENDENT_AUDIT'
    assert not (out/'GPU_FAILED.json').exists()
    unit=f'bora-unified-wsi-comparators-{slug}-v18.service'
    state=subprocess.check_output(['systemctl','--user','show',unit,
        '--property=ActiveState','--property=Result','--property=ExecMainStatus','--property=ControlGroup'],text=True)
    props=dict(line.split('=',1) for line in state.splitlines())
    assert props['ActiveState']=='inactive' and props['Result']=='success' and props['ExecMainStatus']=='0',props
    cgroup=props['ControlGroup']
    proc=Path('/sys/fs/cgroup',cgroup.lstrip('/'),'cgroup.procs') if cgroup else None
    assert proc is None or not proc.exists() or not proc.read_text().strip()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    W,H=manifest['slide_wh']; nx=(W+895)//896; ny=(H+895)//896
    assert len(manifest['fields'])==nx*ny
    stitched=[]
    for method in METHODS:
        tiles=[]
        for row in manifest['fields']:
            path=out/method/f'{row["slug"]}.png'
            rec=json.loads((out/method/f'{row["slug"]}.json').read_text())
            assert rec['field_sha256']==row['field_sha256']==sha(row['field'])
            assert rec['prediction_sha256']==sha(path)
            assert rec['model_sha256']==gate['checkpoints'][method]
            with Image.open(path) as im:
                assert im.mode=='L' and im.size==(1024,1024)
                low,high=im.getextrema()
                if method in ('unet','deeplabv3plus'):
                    assert 1<=low<=high<=24
                elif method=='medsam':
                    assert 0<=low<=high<=5
                else:
                    assert 0<=low<=high<=13
            tiles.append(pyvips.Image.new_from_file(str(path),access='sequential').crop(64,64,896,896))
        full=pyvips.Image.arrayjoin(tiles,across=nx).crop(0,0,W,H)
        assert full.width==W and full.height==H and full.bands==1
        dest=out/f'{method}__full-slide-labels.tif'
        assert not dest.exists()
        full.tiffsave(str(dest),tile=True,tile_width=512,tile_height=512,
            compression='zstd',predictor='horizontal',bigtiff=True)
        check=pyvips.Image.new_from_file(str(dest),access='random')
        assert (check.width,check.height,check.bands)==(W,H,1)
        stitched.append(dict(method=method,path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size))
        print('stitched',method,flush=True)
    target.write_text(json.dumps(dict(status='VERIFIED_FULL_SLIDE_PREDICTIONS_NO_EXPERT_ACCURACY',
        verified_utc=datetime.now(timezone.utc).isoformat(),slug=slug,slide_wh=[W,H],
        tile_count=len(manifest['fields']),methods=stitched,service=props,
        assigned_gpu_contexts_after_exit=[],scope_sha256=manifest['scope_sha256'],
        limitations=['No independent expert whole-slide masks or accuracy scores.',
        'MedSAM is U-Net box prompted for tumor, stroma and necrosis only.',
        'VISTA-PATH uses its own semantic vocabulary.']),indent=2)+'\n')
    print('VERIFIED',slug,flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2
    main(sys.argv[1])
