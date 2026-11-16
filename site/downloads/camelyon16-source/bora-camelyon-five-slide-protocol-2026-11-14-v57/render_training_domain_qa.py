"""Render H&E, official training mask and source-only support at matched overview."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window

DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
MASKS = DATA / 'external-camelyon16-five-slide-2026-11-14-v51/training-class-masks'
SUPPORT = DATA / 'external-camelyon16-pair-2026-11-13-v54/he-support'
CAL = {
    'normal_093': DATA / 'external-camelyon16-pair-2026-11-13-v48/calibrated/normal_093',
    'tumor_093': DATA / 'external-camelyon16-pair-2026-11-13-v49/calibrated/tumor_093',
}
OUT = Path(__file__).parent

def blend(rgb, mask, color, strength=.58):
    result = rgb.copy()
    result[mask] = (rgb[mask].astype(np.float32) * (1 - strength) +
                    np.asarray(color, np.float32) * strength).astype(np.uint8)
    return result

def main(slide):
    if slide not in CAL:
        raise ValueError('Only CAM training slides are admitted')
    output = OUT / f'{slide}-training-domain-qa.jpg'
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((MASKS / slide / 'STATE.json').read_text())
    cal = json.loads((CAL[slide] / 'STATE.json').read_text())
    assert state['status'] == cal['status'] == 'COMPLETE'
    he = np.asarray(Image.open(CAL[slide] / 'he-overview.jpg').convert('RGB'))
    hh, ww = he.shape[:2]
    h, w = state['unpadded_shape_yx']
    with rasterio.open(state['class_mask_path']) as src:
        labels = src.read(1, window=Window(0, 0, w, h),
                          out_shape=(hh, ww), resampling=Resampling.nearest)
    geo = json.loads((SUPPORT / slide / 'tissue-area.geojson').read_text())
    support = rasterize(((f['geometry'], 1) for f in geo['features']),
                        out_shape=(hh, ww),
                        transform=Affine.scale(w / ww, h / hh),
                        fill=0, dtype=np.uint8).astype(bool)
    official = blend(he, labels == 1, (64, 185, 88))
    official = blend(official, labels == 2, (234, 67, 72), .8)
    he_support = blend(he, support, (18, 190, 215), .52)
    missing = blend(he, (labels == 1) & ~support, (242, 142, 18), .83)
    missing = blend(missing, (labels == 2) & ~support, (245, 36, 53), .9)
    panels = [he, official, he_support, missing]
    titles = ['Original H&E', 'Official: normal green, tumor red',
              'H&E-only accepted support: cyan',
              'Official tissue outside support: orange']
    image = Image.new('RGB', (ww * 2, (hh + 28) * 2), 'white')
    draw = ImageDraw.Draw(image)
    for i, panel in enumerate(panels):
        x = (i % 2) * ww
        y = (i // 2) * (hh + 28)
        draw.text((x + 8, y + 7), titles[i], fill=(0, 0, 0))
        image.paste(Image.fromarray(panel), (x, y + 28))
    image.save(output, quality=89)
    print(json.dumps({'slide': slide, 'path': str(output),
                      'overview_shape_yx': [hh, ww],
                      'official_normal_pixels_outside_support': int(((labels == 1) & ~support).sum()),
                      'official_tumor_pixels_outside_support': int(((labels == 2) & ~support).sum())}))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: render_training_domain_qa.py tumor_093|normal_093')
    main(sys.argv[1])
