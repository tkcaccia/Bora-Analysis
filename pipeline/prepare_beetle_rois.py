#!/usr/bin/env python3
"""Prepare capped, calibrated BEETLE tissue ROIs without GPU inference.

The source mask is used only to select informative fixed evaluation fields.
It is never provided to Virchow2 or Bora inference. Missing sparse TIFF tiles
are read as zero by a small native-grid tiled reader, avoiding full WSI loads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pyvips
import tifffile

from build_bora_case import _coarse


ROI = 3072
TISSUE_CODES = (2, 3, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spacing(page) -> tuple[float, float]:
    if str(page.tags["ResolutionUnit"].value) != "RESUNIT.CENTIMETER":
        raise ValueError("BEETLE slide lacks centimetre resolution unit")
    result = []
    for name in ("XResolution", "YResolution"):
        numerator, denominator = page.tags[name].value
        result.append(10000 * denominator / numerator)
    if any(not .4 <= x <= .6 for x in result):
        raise ValueError("BEETLE source physical spacing outside verified range")
    return tuple(result)


def sparse_crop(page, filehandle, x: int, y: int, width: int, height: int) -> np.ndarray:
    if not page.is_tiled or page.samplesperpixel != 1 or page.bitspersample != 8:
        raise ValueError("Expected sparse 8-bit tiled tissue-mask TIFF")
    if x < 0 or y < 0 or x + width > page.imagewidth or y + height > page.imagelength:
        raise ValueError("Native crop outside mask")
    tw, th = page.tilewidth, page.tilelength
    cols = math.ceil(page.imagewidth / tw)
    result = np.zeros((height, width), np.uint8)
    for ty in range(y // th, math.ceil((y + height) / th)):
        for tx in range(x // tw, math.ceil((x + width) / tw)):
            index = ty * cols + tx
            count = page.databytecounts[index]
            if not count:
                continue
            filehandle.seek(page.dataoffsets[index])
            tile, position, shape = page.decode(filehandle.read(count), index)
            if tile is None or position[2:4] != (ty * th, tx * tw) or tile.shape != (1, th, tw, 1):
                raise ValueError("Sparse mask tile decoding differs from declared geometry")
            xx0, yy0 = max(x, tx * tw), max(y, ty * th)
            xx1, yy1 = min(x + width, (tx + 1) * tw), min(y + height, (ty + 1) * th)
            result[yy0-y:yy1-y, xx0-x:xx1-x] = tile[0, yy0-ty*th:yy1-ty*th,
                                                      xx0-tx*tw:xx1-tx*tw, 0]
    return result


def candidates(low: np.ndarray, shape: tuple[int, int], factor_xy: tuple[float, float]) -> list[dict]:
    # The deterministic grid samples every annotated 96-pixel low-resolution bin.
    ys, xs = np.where(low[::4, ::4] >= 2)
    if not len(xs):
        return []
    row_bins, col_bins = (ys * 4) // 96, (xs * 4) // 96
    bins = np.unique(np.stack([row_bins, col_bins], axis=1), axis=0)
    height, width = shape
    found = []
    for by, bx in bins:
        x = min(max(0, round((int(bx) * 96 + 48) * factor_xy[0] - ROI // 2)), width - ROI)
        y = min(max(0, round((int(by) * 96 + 48) * factor_xy[1] - ROI // 2)), height - ROI)
        lx0, ly0 = int(x / factor_xy[0]), int(y / factor_xy[1])
        lx1, ly1 = int(math.ceil((x + ROI) / factor_xy[0])), int(math.ceil((y + ROI) / factor_xy[1]))
        patch = low[ly0:ly1, lx0:lx1]
        counts = {code:int(np.count_nonzero(patch == code)) for code in TISSUE_CODES}
        represented = sorted((n for n in counts.values() if n >= 20), reverse=True)
        if len(represented) < 2:
            continue
        found.append(dict(x=x, y=y, low_counts=counts,
                          score=(represented[1], sum(represented), represented[0])))
    unique = {}
    for item in found:
        unique[(item["x"], item["y"])] = item
    return sorted(unique.values(), key=lambda a:(a["score"], -a["y"], -a["x"]), reverse=True)


def eligible(mask: np.ndarray, mpp_xy: tuple[float, float], case_id: str) -> dict | None:
    wh = (round(mask.shape[1] * mpp_xy[0] / .5), round(mask.shape[0] * mpp_xy[1] / .5))
    try:
        grid, _ = _coarse(mask, (wh[1], wh[0]), case_id, (0, 1, 255))
    except ValueError:
        return None
    counts = Counter(int(v) for v in np.unique(mask) if v in TISSUE_CODES)
    return dict(physical_grid_tiles=int(np.count_nonzero(grid)),
                represented_anonymous_groups=int(len(np.unique(grid[grid > 0]))),
                expert_source_codes=sorted(counts),
                code_pixel_counts={str(c):int(np.count_nonzero(mask == c)) for c in TISSUE_CODES},
                working_shape_yx=[wh[1], wh[0]])


def plan(source: Path) -> tuple[list[dict], list[dict]]:
    slide_candidates, rejected = [], []
    for mask_path in sorted((source / "masks").glob("*.tif")):
        image_path = source / "images" / mask_path.name
        if not image_path.is_file():
            rejected.append(dict(slide=mask_path.stem, reason="missing_image"));continue
        with tifffile.TiffFile(mask_path) as mask_file, tifffile.TiffFile(image_path) as image_file:
            page = mask_file.pages[0]
            native_shape = (page.imagelength, page.imagewidth)
            if image_file.series[0].shape[:2] != native_shape or min(native_shape) < ROI:
                rejected.append(dict(slide=mask_path.stem, reason="geometry_or_size"));continue
            mpp = spacing(image_file.pages[0])
            level = mask_file.series[0].levels[min(4, len(mask_file.series[0].levels)-1)]
            low = level.asarray()
            factor = (native_shape[1]/low.shape[1], native_shape[0]/low.shape[0])
            accepted = []
            for candidate in candidates(low, native_shape, factor):
                if any(abs(candidate["x"] - q["x"]) < ROI and abs(candidate["y"] - q["y"]) < ROI
                       for q in accepted):
                    continue
                mask = sparse_crop(page, mask_file.filehandle, candidate["x"], candidate["y"], ROI, ROI)
                info = eligible(mask, mpp, f"{mask_path.stem}_{candidate['x']}_{candidate['y']}")
                if info is None:
                    continue
                accepted.append(dict(slide=mask_path.stem, image=str(image_path), mask=str(mask_path),
                                     mpp_xy=mpp, native_shape_yx=native_shape, **candidate, **info))
                if len(accepted) == 3:
                    break
            if accepted:
                slide_candidates.append(accepted)
            else:
                rejected.append(dict(slide=mask_path.stem, reason="no_two_group_3072px_field"))
    selected = [rows[0] for rows in slide_candidates]
    if len(selected) > 20:
        selected = sorted(selected, key=lambda x:(x["score"],x["slide"]), reverse=True)[:20]
    else:
        extras = [item for rows in slide_candidates for item in rows[1:]]
        extras.sort(key=lambda x:(x["score"],x["slide"]), reverse=True)
        for item in extras:
            if len(selected) == 20: break
            selected.append(item)
    if len(selected) != 20:
        raise ValueError(f"Only {len(selected)} eligible BEETLE ROIs; do not force uncalibrated or one-group images")
    return selected, rejected


def extract(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Use a fresh versioned BEETLE ROI output")
    selected, rejected = plan(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".beetle-", dir=output.parent) as temp:
        folder = Path(temp) / "cohort";folder.mkdir()
        images=folder/"images";masks=folder/"masks";images.mkdir();masks.mkdir()
        rows=[]
        for index,item in enumerate(selected,1):
            name=f"beetle_{index:02d}_{item['slide']}"
            with tifffile.TiffFile(item["mask"]) as f:
                mask=sparse_crop(f.pages[0],f.filehandle,item["x"],item["y"],ROI,ROI)
            p=images/(name+".png");q=masks/(name+".png")
            image=pyvips.Image.new_from_file(item["image"],page=0,access="random")
            crop=image.crop(item["x"],item["y"],ROI,ROI)
            if crop.bands != 3 or crop.format != "uchar":
                raise ValueError("BEETLE image must be uint8 RGB")
            array=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(ROI,ROI,3)
            Image.fromarray(array).save(p,optimize=True)
            Image.fromarray(mask).save(q,optimize=True)
            rows.append(dict(case_id=name,source_slide=item["slide"],source_image=item["image"],
                source_mask=item["mask"],native_crop_xywh=[item["x"],item["y"],ROI,ROI],
                source_mpp_xy=item["mpp_xy"],working_shape_yx=item["working_shape_yx"],
                image=str(p.relative_to(folder)),mask=str(q.relative_to(folder)),
                image_sha256=sha256(p),mask_sha256=sha256(q),
                source_label_counts=item["code_pixel_counts"],
                input_coarse_groups=item["represented_anonymous_groups"],
                physical_grid_tiles=item["physical_grid_tiles"]))
        manifest=dict(format="bora-beetle-calibrated-roi-v1",source="BEETLE capped 20 WSI image/mask pairs",
            fixed_selection="3072-pixel nonoverlapping source windows, two annotated tissue codes, at least16 eligible coarse tiles",
            source_images_train_bora=False,source_images_train_virchow2_here=False,
            public_tissue_codes={"2":"non-invasive epithelium","3":"invasive epithelium","4":"necrosis"},
            code_1="provider other; ignored for expert tissue score",code_0="unannotated",
            representative_roi_selection_uses_expert_mask=True,
            cases=rows,rejected_slides=rejected)
        (folder/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        os.rename(folder,output)
    return dict(cases=len(rows),source_slides=len({x["source_slide"] for x in rows}),
                output=str(output),rejected_slides=len(rejected))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--plan-only",action="store_true")
    a=p.parse_args()
    if a.plan_only:
        selected,rejected=plan(a.source)
        print(json.dumps(dict(cases=len(selected),slides=len({x["slide"] for x in selected}),
                              selection=[dict(slide=x["slide"],x=x["x"],y=x["y"],codes=x["expert_source_codes"],
                                              tiles=x["physical_grid_tiles"]) for x in selected],
                              rejected=rejected),indent=2))
    else:
        print(json.dumps(extract(a.source,a.output),indent=2))


if __name__=="__main__":main()
