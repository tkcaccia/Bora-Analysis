"""Create fixed 1024 px context tiles covering one frozen WSI at 896 px stride."""
import hashlib
import json
import sys
from pathlib import Path

import pyvips

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora-Data/benchmark2-full-wsi-comparators-2026-11-10-v2')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(slug):
    scope = json.loads((HERE/'SCOPE.json').read_text())
    item = next(r for r in scope['slides'] if r['slug'] == slug)
    out = ROOT/slug
    assert not out.exists(), 'Never alter staged inputs in place'
    out.mkdir(parents=True)
    fields = out/'fields'
    fields.mkdir()
    slide = pyvips.Image.new_from_file(item['path'], access='random')
    assert (slide.width,slide.height)==(item['width'],item['height'])
    padded = slide.embed(64,64,slide.width+128,slide.height+128,extend='white')
    rows=[]
    for ty in range(item['tiles_y']):
        for tx in range(item['tiles_x']):
            name=f'{slug}__y{ty:04d}x{tx:04d}'
            path=fields/f'{name}.png'
            padded.crop(tx*896,ty*896,1024,1024).pngsave(str(path),compression=6)
            rows.append(dict(slug=name, field=str(path),field_sha256=sha(path),
                             core_xywh=[tx*896,ty*896,
                                min(896,item['width']-tx*896),
                                min(896,item['height']-ty*896)]))
        print('staged row',ty+1,'of',item['tiles_y'],flush=True)
    (out/'FIELDS_FROZEN.json').write_text(json.dumps(dict(
        status='FROZEN_FULL_SLIDE_TILES',scope_sha256=sha(HERE/'SCOPE.json'),
        source_sha256=item['sha256'],slide_wh=[item['width'],item['height']],
        core=896,margin=64,fields=rows),indent=2)+'\n')
    print('staged',len(rows),'tiles',flush=True)


if __name__=='__main__':
    assert len(sys.argv)==2
    main(sys.argv[1])
