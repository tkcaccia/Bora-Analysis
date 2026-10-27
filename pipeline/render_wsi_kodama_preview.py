#!/usr/bin/env python3
"""Render an annotation-free WSI tile-group preview at thumbnail scale."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

PALETTE = np.asarray([
    [37, 118, 184], [233, 117, 39], [55, 161, 100], [157, 86, 190],
    [230, 181, 39], [204, 68, 111], [59, 177, 186], [139, 120, 62],
    [134, 136, 156], [196, 87, 79], [100, 178, 82], [84, 113, 197],
    [188, 113, 163], [85, 152, 121], [218, 145, 107], [118, 164, 212],
    [151, 120, 205], [207, 156, 71], [110, 142, 80], [221, 104, 157],
], np.uint8)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thumbnail',type=Path,required=True)
    p.add_argument('--tile-groups',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--flat-output',type=Path)
    a=p.parse_args()
    image=Image.open(a.thumbnail).convert('RGB')
    labels=np.load(a.tile_groups,allow_pickle=False)
    if labels.ndim!=2 or labels.dtype.kind not in 'ui' or labels.max()>len(PALETTE):
        raise ValueError('Unsupported tile group grid')
    colors=np.full((*labels.shape,3),255,np.uint8)
    valid=labels>0
    colors[valid]=PALETTE[labels[valid]-1]
    mask=Image.fromarray(colors).resize(image.size,Image.Resampling.NEAREST)
    # Display support follows only the H&E thumbnail. It does not enter KODAMA
    # or the feature-space assignment, and is not an expert tissue mask.
    tissue=np.min(np.asarray(image),axis=2)<235
    alpha=Image.fromarray((tissue*190).astype(np.uint8))
    result=Image.composite(Image.blend(image,mask,.65),image,alpha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.save(a.output,quality=91)
    if a.flat_output:
        flat=Image.composite(mask,Image.new('RGB',image.size,'white'),alpha)
        a.flat_output.parent.mkdir(parents=True,exist_ok=True)
        flat.save(a.flat_output,quality=91)

if __name__=='__main__':main()
