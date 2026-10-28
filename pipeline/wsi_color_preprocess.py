#!/usr/bin/env python3
"""Optional slide-coordinate color-drift correction before Virchow2.

Fits exterior glass, never tissue. Existing sources and feature caches are not
modified. The model is an empirical RGB shading correction, not a calibrated
scanner flat field or H&E stain normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import HuberRegressor

VERSION = "bora-glass-plane-v1"
POLICY = dict(grid=24, minimum_cells=48, minimum_span_xy=.75,
              minimum_field_span_fraction=.015, minimum_cv_improvement=.25,
              maximum_gain_change=.05, minimum_glass_pixels_per_cell=200,
              maximum_clipped_glass_fraction=.05)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def glass_mask(rgb):
    """Bright, low-chroma, exterior-connected pixels separated from tissue."""
    a = np.asarray(rgb, np.float32)
    blurred = ndi.gaussian_filter(a, sigma=(1., 1., 0.))
    tissue = (np.ptp(blurred, axis=2) > 25) | (blurred.min(2) < 180)
    near_tissue = ndi.binary_dilation(tissue, structure=np.ones((7, 7)))
    candidate = (np.ptp(a, axis=2) < 18) & (a.min(2) > 150) & ~near_tissue
    labels, _ = ndi.label(candidate, structure=np.ones((3, 3)))
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return np.isin(labels, border[border > 0])


def fit_glass_plane(rgb, mask=None):
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("Expected uint8 RGB thumbnail")
    glass = glass_mask(a) if mask is None else np.asarray(mask, bool)
    if glass.shape != a.shape[:2]:
        raise ValueError("Glass mask dimensions differ")
    h, w = glass.shape
    cells = []
    n = POLICY["grid"]
    for row in range(n):
        for col in range(n):
            y0, y1 = round(row*h/n), round((row+1)*h/n)
            x0, x1 = round(col*w/n), round((col+1)*w/n)
            selected = glass[y0:y1, x0:x1]
            if selected.sum() < POLICY["minimum_glass_pixels_per_cell"]:
                continue
            # Mean coordinates of the actual retained pixels, not cell centres.
            yy, xx = np.nonzero(selected)
            cells.append(dict(x=float((x0+xx.mean())/(w-1)-.5),
                              y=float((y0+yy.mean())/(h-1)-.5),
                              rgb=np.median(a[y0:y1, x0:x1][selected], axis=0).tolist(),
                              fold=int((row >= n//2)*2 + (col >= n//2))))
    model = dict(format=VERSION, policy=POLICY, thumbnail_shape_yx=[h, w],
                 glass_pixels=int(glass.sum()), glass_cells=len(cells),
                 coordinates="x/(slide_width-1)-0.5, y/(slide_height-1)-0.5",
                 rgb_space="decoded RGB values; no ICC transformation",
                 dark_field_available=False, channels=[], applied=False,
                 validation="four held-out spatial quadrants; no tissue or labels used")
    if not cells:
        return dict(model, decision="SKIP", reason="No usable exterior glass"), glass, cells
    x = np.array([[r['x'], r['y']] for r in cells])
    z = np.array([r['rgb'] for r in cells])
    folds = np.array([r['fold'] for r in cells])
    clipped = np.mean(np.any(a[glass] >= 254, axis=1))
    model["clipped_glass_fraction"] = float(clipped)
    sufficient = (len(cells) >= POLICY['minimum_cells'] and
                  np.all(np.ptp(x, axis=0) >= POLICY['minimum_span_xy']) and
                  all(np.sum(folds == f) >= 8 for f in range(4)) and
                  clipped <= POLICY['maximum_clipped_glass_fraction'])
    if not sufficient:
        return dict(model, decision="SKIP", reason="Insufficient distributed, unclipped exterior glass"), glass, cells
    corners = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    for c, name in enumerate('RGB'):
        pred = np.zeros(len(z)); constant = np.zeros(len(z))
        for f in range(4):
            train = folds != f; test = ~train
            estimator = HuberRegressor(epsilon=1.35, alpha=0., max_iter=1000).fit(x[train], z[train, c])
            pred[test] = estimator.predict(x[test])
            constant[test] = np.median(z[train, c])
        error = float(np.median(abs(z[:, c]-pred)))
        baseline = float(np.median(abs(z[:, c]-constant)))
        estimator = HuberRegressor(epsilon=1.35, alpha=0., max_iter=1000).fit(x, z[:, c])
        coefficients = [float(estimator.intercept_), *map(float, estimator.coef_)]
        field = estimator.predict(corners); anchor = float(np.median(z[:, c]))
        span = float(np.ptp(field)/anchor)
        gains = anchor/field
        apply = (baseline > .25 and error <= (1-POLICY['minimum_cv_improvement'])*baseline
                 and span >= POLICY['minimum_field_span_fraction']
                 and float(np.max(abs(gains-1))) <= POLICY['maximum_gain_change'])
        model['channels'].append(dict(channel=name, apply=bool(apply),
            coefficients_intercept_x_y=coefficients, anchor=anchor,
            spatial_cv_mae=error, constant_cv_mae=baseline,
            predicted_field_span_fraction=span, corner_gain_range=[float(gains.min()), float(gains.max())]))
    model['applied'] = any(c['apply'] for c in model['channels'])
    model['decision'] = 'APPLY' if model['applied'] else 'SKIP'
    model['reason'] = ('Distributed glass supports a mild spatial correction' if model['applied']
                       else 'No channel passes the fixed effect-size, validation and gain gates')
    return model, glass, cells


def correct_region(rgb, model, origin_xy=(0, 0), slide_shape_yx=None):
    """Same field for whole slide or arbitrary crops; no crop-local fitting."""
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("Expected uint8 RGB")
    if not model['applied']:
        return a.copy()
    h, w = slide_shape_yx or a.shape[:2]
    ox, oy = origin_xy
    if min(ox, oy) < 0 or ox+a.shape[1] > w or oy+a.shape[0] > h:
        raise ValueError("Crop outside slide coordinate system")
    x = (np.arange(a.shape[1], dtype=np.float64)+ox)/(w-1)-.5
    y = (np.arange(a.shape[0], dtype=np.float64)+oy)/(h-1)-.5
    corrected = a.copy()
    for c, record in enumerate(model['channels']):
        if not record['apply']:
            continue
        b, bx, by = record['coefficients_intercept_x_y']
        field = b + bx*x[None, :] + by*y[:, None]
        corrected[..., c] = np.rint(np.clip(a[..., c]*(record['anchor']/field), 0, 255)).astype(np.uint8)
    return corrected


def _array(image):
    return np.ndarray(buffer=image.write_to_memory(), dtype=np.uint8,
                      shape=(image.height, image.width, image.bands)).copy()


def run(args):
    import pyvips
    from PIL import Image
    pyvips.cache_set_max_mem(64 << 20); pyvips.cache_set_max(0)
    source = args.image.resolve(); out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; source data are immutable")
    if (not 1 <= args.threads <= 4 or min(args.mpp) <= 0 or not all(map(math.isfinite, args.mpp))
            or not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0):
        raise ValueError("Expected 1..4 CPU threads and positive finite verified mpp")
    os.environ['VIPS_CONCURRENCY'] = str(args.threads)
    if shutil.disk_usage(out.parent).free < args.minimum_free_gib*1024**3:
        raise RuntimeError("Storage reserve unavailable")
    start = time.monotonic(); source_sha = sha256(source)
    if args.source_sha256 and source_sha != args.source_sha256:
        raise ValueError("Source checksum changed")
    raw = pyvips.Image.new_from_file(str(source), access='random')
    if raw.format != 'uchar' or raw.bands != 3 or raw.interpretation not in ('srgb', 'rgb'):
        raise ValueError("Expected scanner uint8 RGB; alpha/other encodings need explicit preparation")
    if min(raw.width, raw.height) < 256:
        raise ValueError("Expected a WSI with at least 256 pixels per axis")
    if args.apply and shutil.disk_usage(out.parent).free < args.minimum_free_gib*1024**3 + 8*raw.width*raw.height:
        raise RuntimeError("Lossless pyramid staging would violate the storage reserve")
    out.mkdir(); atomic_json(out/'STATE.json', dict(status='RUNNING', pid=os.getpid(), CPU_only=True))
    preview = _array(raw.resize(2048/raw.width, kernel='lanczos3'))
    model, glass, cells = fit_glass_plane(preview)
    model.update(source_sha256=source_sha, implementation_sha256=sha256(Path(__file__)),
                 source_shape_yx=[raw.height, raw.width],
                 source_mpp_xy=list(args.mpp), source=source.name,
                 source_unchanged=True, fit_uses_expert_annotations=False)
    atomic_json(out/'COLOR_MODEL.json', model); atomic_json(out/'GLASS_CELLS.json', {'cells':cells})
    Image.fromarray(preview).save(out/'original-overview.png')
    Image.fromarray(correct_region(preview, model)).save(out/'corrected-overview.png')
    Image.fromarray(glass.astype(np.uint8)*255).save(out/'exterior-glass.png')
    result_image = source
    checks = []
    if args.apply and model['applied']:
        # Evaluate one smooth global field lazily; no independent tile equalization.
        xyz = pyvips.Image.xyz(raw.width, raw.height).cast('double')
        x = xyz[0]/(raw.width-1)-.5; y = xyz[1]/(raw.height-1)-.5
        bands = []
        for c, record in enumerate(model['channels']):
            if record['apply']:
                b, bx, by = record['coefficients_intercept_x_y']
                band = (raw[c].cast('double')*(record['anchor']/(b+bx*x+by*y))).rint().cast('uchar')
            else:
                band = raw[c]
            bands.append(band)
        image = bands[0].bandjoin(bands[1:]).copy(interpretation='srgb',
                         xres=1000/args.mpp[0], yres=1000/args.mpp[1])
        # Explicit OME physical spacing; source pixel coordinate system is retained.
        ome = ('<?xml version="1.0" encoding="UTF-8"?>'
          '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
          '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8" '
          f'SizeX="{raw.width}" SizeY="{raw.height}" SizeC="3" SizeZ="1" SizeT="1" '
          f'PhysicalSizeX="{args.mpp[0]:.16g}" PhysicalSizeXUnit="µm" '
          f'PhysicalSizeY="{args.mpp[1]:.16g}" PhysicalSizeYUnit="µm" Interleaved="true">'
          '<Channel ID="Channel:0:0" SamplesPerPixel="3"/><TiffData IFD="0" PlaneCount="1"/>'
          '</Pixels></Image></OME>')
        image.set_type(pyvips.GValue.gstr_type, 'image-description', ome)
        result_image = out/'corrected.ome.tif'
        temp = out/'corrected.partial.tif'
        image.tiffsave(str(temp), tile=True, tile_width=256, tile_height=256,
                       pyramid=True, subifd=True, bigtiff=True, compression='deflate',
                       resunit='cm')
        temp.replace(result_image)
        restored = pyvips.Image.new_from_file(str(result_image), access='random')
        if (restored.width,restored.height,restored.bands,restored.format) != (raw.width,raw.height,3,'uchar'):
            raise ValueError("Output geometry/encoding changed")
        # Independent NumPy equation verifies lossless pixels at fixed spatial locations.
        for fy in (.02,.25,.5,.75,.98):
            for fx in (.02,.5,.98):
                sx = min(raw.width-256, round(fx*raw.width)); sy = min(raw.height-256, round(fy*raw.height))
                original = _array(raw.crop(sx,sy,256,256))
                expected = correct_region(original, model, (sx,sy), (raw.height,raw.width))
                actual = _array(restored.crop(sx,sy,256,256))
                if not np.array_equal(actual,expected):
                    raise ValueError(f"Independent pixel equation differs at {sx},{sy}")
                checks.append(dict(x=sx,y=sy,width=256,height=256,exact_pixel_equivalence=True))
        corrected_preview = _array(restored.resize(2048/raw.width, kernel='lanczos3'))
        Image.fromarray(corrected_preview).save(out/'corrected-overview.png')
    if sha256(source) != source_sha:
        raise ValueError("Source changed during processing")
    complete = dict(status='COMPLETE',format=VERSION, implementation_sha256=sha256(Path(__file__)),
        decision=model['decision'],
        correction_written=bool(args.apply and model['applied']), source_sha256=source_sha,
        input_image=str(source), output_image=str(result_image), output_sha256=sha256(result_image),
        shape_yx=[raw.height,raw.width], mpp_xy=list(args.mpp), coordinates_unchanged=True,
        compression='lossless deflate' if result_image != source else 'source unchanged',
        ICC_transform_applied=False, channels_corrected=[r['channel'] for r in model['channels'] if r['apply']],
        cache_policy='Corrected image requires new Virchow2 features and all downstream fits; never reuse old features',
        pixel_equivalence_checks=checks, elapsed_seconds=time.monotonic()-start,
        peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
        CPU_only=True,
        files_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='STATE.json'})
    atomic_json(out/'PREPROCESS_COMPLETE.json', complete)
    atomic_json(out/'STATE.json',dict(status='COMPLETE',CPU_only=True))
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mpp',type=float,nargs=2,required=True,metavar=('X','Y'))
    p.add_argument('--source-sha256');p.add_argument('--apply',action='store_true')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--minimum-free-gib',type=float,default=80)
    args=p.parse_args()
    output_preexisted = args.output.exists()
    try:
        print(json.dumps(run(args),indent=2))
    except BaseException as error:
        if not output_preexisted and args.output.is_dir():
            atomic_json(args.output/'STATE.json',dict(status='STOPPED',error=repr(error)))
        raise


if __name__=='__main__':
    main()
