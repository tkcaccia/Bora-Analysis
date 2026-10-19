"""Create one physically calibrated, annotation-derived Bora test field.

The reference is used only to construct the deliberately coarse anonymous tile
input and held out for scoring. This is a conditional refinement experiment,
not an independent tissue-discovery evaluation.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyvips
import tifffile
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box, shape


LABELS = {
    "invasive carcinoma": 1,
    "intratumoral stroma": 2,
    "desmoplastic submucosa": 2,
}


def crop_rgb(image, x, y, w, h):
    region = image.crop(x, y, w, h)
    if region.format != "uchar" or region.bands != 3:
        raise ValueError("Expected 8-bit RGB source")
    return np.ndarray((h, w, 3), dtype=np.uint8, buffer=region.write_to_memory()).copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--x", type=int, default=48000)
    p.add_argument("--y", type=int, default=12000)
    p.add_argument("--size", type=int, default=2048)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    encoder_revision = "3158645804b69e3f3bc4439d4116edddf0840a72"
    encoder_sha256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"

    source = pyvips.Image.new_from_file(str(a.image), access="random")
    if a.x < 0 or a.y < 0 or a.x+a.size > source.width or a.y+a.size > source.height:
        raise ValueError("ROI lies outside source")
    native_mpp = np.array([0.27381215039690188, 0.2738132569747595])
    native = crop_rgb(source, a.x, a.y, a.size, a.size)
    work_shape = np.rint(a.size*native_mpp[::-1]/.5).astype(int)
    work = cv2.resize(native, tuple(work_shape[::-1]), interpolation=cv2.INTER_AREA)
    work_mpp = native_mpp*a.size/work_shape[::-1]
    # Threshold only the H&E slide, not the expert annotation.
    support_native = np.min(native, axis=2) < 235
    support = cv2.resize(support_native.astype(np.uint8), tuple(work_shape[::-1]),
                         interpolation=cv2.INTER_AREA) > .3
    roi = box(a.x, a.y, a.x+a.size, a.y+a.size)
    shapes = []
    for f in json.loads(a.annotation.read_text())["features"]:
        if f["geometry"]["type"] not in ("Polygon", "MultiPolygon"):
            continue
        name = f.get("properties", {}).get("classification", {}).get("name")
        if name not in LABELS:
            continue
        geom = shape(f["geometry"])
        if geom.intersects(roi):
            shapes.append((geom.intersection(roi), LABELS[name]))
    transform = Affine(a.size/work_shape[1], 0, a.x, 0, a.size/work_shape[0], a.y)
    reference = rasterize(shapes, out_shape=tuple(work_shape), transform=transform,
                          fill=0, dtype=np.uint8)
    coarse = np.zeros(tuple(work_shape), np.uint8)
    for y in range(0, work_shape[0], 90):
        for x in range(0, work_shape[1], 90):
            region = reference[y:y+90, x:x+90]
            ids, counts = np.unique(region[region > 0], return_counts=True)
            if len(ids):
                coarse[y:y+90, x:x+90] = ids[np.argmax(counts)]
    # Avoid a spurious tile assignment when the expert drew only a tiny object.
    tile_counts = {str(i): int(np.sum(coarse[::90, ::90] == i)) for i in (1, 2)}
    if min(map(int, tile_counts.values())) < 2:
        raise ValueError(f"Insufficient class-specific tiles: {tile_counts}")
    tifffile.imwrite(a.output/"original.tif", work, photometric="rgb")
    tifffile.imwrite(a.output/"coarse.tif", coarse)
    tifffile.imwrite(a.output/"tissue.tif", support.astype(np.uint8))
    tifffile.imwrite(a.output/"native_tissue.tif", support_native.astype(np.uint8))
    tifffile.imwrite(a.output/"reference.working.tif", reference)
    tifffile.imwrite(a.output/"reference.native.tif", rasterize(shapes,
        out_shape=(a.size,a.size), transform=Affine(1,0,a.x,0,1,a.y), fill=0, dtype=np.uint8))
    variant_shapes = {name:[round(int(work_shape[0])/factor),round(int(work_shape[1])/factor)]
                      for name,factor in (("scale_1",2),("scale_2",4))}
    manifest = dict(case="visium_hd_colon_roi_48000_12000",image="original.tif",
        coarse="coarse.tif",tissue="tissue.tif",native_tissue="native_tissue.tif",
        tokens="features/base.npy",mpp_xy=work_mpp.tolist(),
        native_mpp_xy=native_mpp.tolist(),palette=[[50,110,235],[255,165,80]],
        variants={name:dict(features=f"features/{name}.npy",
            shape_yx=variant_shapes[name],
            mpp_xy=(native_mpp*a.size/np.asarray(variant_shapes[name][::-1])).tolist())
            for name in ("scale_1","scale_2")},
        encoder_revision=encoder_revision,encoder_sha256=encoder_sha256)
    (a.output/"case.json").write_text(json.dumps(manifest,indent=2)+"\n")
    report=dict(source_image=str(a.image),annotation=str(a.annotation),native_origin_xy=[a.x,a.y],
        native_shape_yx=[a.size,a.size],native_mpp_xy=native_mpp.tolist(),
        working_shape_yx=work_shape.tolist(),working_mpp_xy=work_mpp.tolist(),
        expert_labels_used=LABELS,reference_pixels={str(i):int(np.sum(reference==i)) for i in (1,2)},
        coarse_tile_counts=tile_counts,reference_role="coarse-input construction and conditional evaluation")
    (a.output/"preparation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
