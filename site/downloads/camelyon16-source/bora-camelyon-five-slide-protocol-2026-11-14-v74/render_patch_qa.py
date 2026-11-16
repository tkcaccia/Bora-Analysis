"""Local QA montage of frozen CAM pair training patches; never publish."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyvips
import rasterio
from rasterio.windows import Window

ROOT = Path('/mnt/sata_ssd/ChatGPT/Bora/migration')
DATA = Path('/mnt/sata_ssd/ChatGPT/Bora-Data')
SOURCE = ROOT / 'bora-camelyon-five-slide-protocol-2026-11-14-v72/PATCH_INDEX_FROZEN.json'
OUT = Path(__file__).parent / 'training-patch-qa-local.jpg'
CAL = {'tumor_093': DATA / 'external-camelyon16-pair-2026-11-13-v49/calibrated/tumor_093/calibrated-0p5-padded.tif',
       'normal_093': DATA / 'external-camelyon16-pair-2026-11-13-v48/calibrated/normal_093/calibrated-0p5-padded.tif'}
MASK = DATA / 'external-camelyon16-five-slide-2026-11-14-v58/training-labels'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    a = json.loads(SOURCE.read_text())
    samples = []
    for slide, cls in [('tumor_093', 2), ('tumor_093', 1), ('normal_093', 1)]:
        fit = [p for p in a['patches'] if p['slide'] == slide and p['official_class_id'] == cls and p['split'] == 'inner_fit']
        valid = [p for p in a['patches'] if p['slide'] == slide and p['official_class_id'] == cls and p['split'] == 'inner_validation']
        samples.extend(fit[:2] + valid[:1])
    montage = Image.new('RGB', (512 * 3, 540 * 3), 'white')
    draw = ImageDraw.Draw(montage)
    for index, row in enumerate(samples):
        slide, x, y = row['slide'], row['x_center_0p5'], row['y_center_0p5']
        he = pyvips.Image.new_from_file(str(CAL[slide]), access='random')
        crop = he.crop(x - 256, y - 256, 512, 512)
        rgb = np.frombuffer(crop.write_to_memory(), dtype=np.uint8).reshape(512, 512, 3).copy()
        with rasterio.open(MASK / slide / 'training-classes-in-support.tif') as label:
            mask = label.read(1, window=Window(x - 256, y - 256, 512, 512))
        rgb[mask == 1] = (rgb[mask == 1] * .55 + np.array([55, 205, 82]) * .45).astype(np.uint8)
        rgb[mask == 2] = (rgb[mask == 2] * .25 + np.array([230, 45, 48]) * .75).astype(np.uint8)
        col, row_id = index % 3, index // 3
        montage.paste(Image.fromarray(rgb), (col * 512, row_id * 540 + 28))
        draw.text((col * 512 + 8, row_id * 540 + 6),
                  f"{slide} class {row['official_class_id']} {row['split']}", fill=(0, 0, 0))
    montage.save(OUT, quality=88)
    receipt = {'status': 'LOCAL_VISUAL_QA_READY', 'patch_index_sha256': sha(SOURCE),
               'montage_path': str(OUT), 'montage_sha256': sha(OUT),
               'test_reference_opened': False, 'online_publication': False}
    (Path(__file__).parent / 'PATCH_QA_READY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
