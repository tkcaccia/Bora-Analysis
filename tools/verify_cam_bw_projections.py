#!/usr/bin/env python3
"""Require two published Bora response maps for every scored CAM test slide."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


site = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('site')
cam = site / 'camelyon16'
inventory = json.loads((site / 'PUBLIC_RELEASE_VERIFIED.json').read_text())['file_sha256']
count = 0
for score in sorted(cam.glob('*-scores.json')):
    data = json.loads(score.read_text())
    if data.get('training_slides_bora') != ['tumor_093']:
        continue
    slide = score.name.removesuffix('-scores.json')
    page = (cam / f'{slide}.html').read_text()
    if page.count('id="bora-bw-projections"') != 1:
        raise ValueError(f'{slide}: missing BW projection section')
    if page.count(f'{slide}.html#bora-bw-projections') != 0:
        raise ValueError(f'{slide}: incorrect self-link')
    for cls in (1, 2):
        relative = f'camelyon16/bw-projections/{slide}-class-{cls:02d}.webp'
        path = site / relative
        if not path.exists() or path.stat().st_size < 100 or \
                hashlib.sha256(path.read_bytes()).hexdigest() != inventory.get(relative):
            raise ValueError(f'{slide}: missing or unverified {relative}')
        head = path.read_bytes()[:12]
        if head[:4] != b'RIFF' or head[8:] != b'WEBP':
            raise ValueError(f'{slide}: invalid WebP')
        if len(re.findall(re.escape(relative.removeprefix('camelyon16/')), page)) != 1:
            raise ValueError(f'{slide}: image is not shown exactly once')
    index = (cam / 'index.html').read_text()
    if index.count(f'href="{slide}.html#bora-bw-projections"') != 1:
        raise ValueError(f'{slide}: gallery BW link missing')
    count += 1
if count < 1:
    raise ValueError('No current one-slide CAM scores found')
print(f'BW projections verified for {count} current scored test slides ({count * 2} maps)')
