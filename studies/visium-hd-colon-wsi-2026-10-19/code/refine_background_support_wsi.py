"""Remove low-stain scanner/glass blocks after Bora, preserving v1 evidence.

Only the original H&E and existing classified mask determine support. The
expert GeoJSON is never opened. The new result reuses vector checkpoints only
where its source native raster is byte-identical to the audited v1 source.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
from time import monotonic

import cv2
import numpy as np
import pyvips

from bora.wsi import ome_mpp


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-um", type=float, default=140.2)
    parser.add_argument("--minimum-stained-fraction", type=float, default=.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.minimum_stained_fraction < 1:
        raise ValueError("Stain fraction must be between zero and one")
    if args.block_um <= 0:
        raise ValueError("Positive physical block size required")
    source = args.source
    if not (source / "VECTOR_COMPLETE.json").exists():
        raise FileNotFoundError("Audited native source export required")
    report = json.loads((source / "CLASSIFIED_COMPLETE.json").read_text())
    h, w = report["native_shape_yx"]
    shape_xy, mpp = ome_mpp(args.image)
    if shape_xy != (w, h) or not np.allclose(mpp, report["native_mpp_xy"], rtol=0, atol=1e-9):
        raise ValueError("Source image and mask calibration disagree")
    block = max(32, int(round(args.block_um / float(np.mean(mpp)))))
    args.output.mkdir(parents=True)
    image = pyvips.Image.new_from_file(str(args.image), access="random")
    old = np.memmap(source / "classified.background.dat", mode="r", dtype=np.uint8,
                    shape=(h, w))
    output_path = args.output / "classified.background.dat"
    with output_path.open("wb") as stream:
        stream.truncate(h*w)
    new = np.memmap(output_path, mode="r+", dtype=np.uint8, shape=(h, w))
    started = monotonic()
    changed = []
    removed = Counter()
    low_support_blocks = 0
    examined_blocks = 0
    index = 0
    for y in range(0, h, 4096):
        for x in range(0, w, 4096):
            ch, cw = min(4096, h-y), min(4096, w-x)
            before = np.asarray(old[y:y+ch, x:x+cw])
            target = before.copy()
            if np.any(before):
                crop = image.crop(x, y, cw, ch)
                rgb = np.ndarray((ch, cw, 3), dtype=np.uint8,
                                 buffer=crop.write_to_memory())
                saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
                stained = (saturation > 20) & (np.min(rgb, axis=2) < 220)
                for by in range(0, ch, block):
                    for bx in range(0, cw, block):
                        sy, sx = min(ch, by+block), min(cw, bx+block)
                        area = target[by:sy, bx:sx]
                        if not np.any(area):
                            continue
                        examined_blocks += 1
                        fraction = float(np.mean(stained[by:sy, bx:sx]))
                        if fraction >= args.minimum_stained_fraction:
                            continue
                        low_support_blocks += 1
                        values, counts = np.unique(area[area > 0], return_counts=True)
                        for value, count in zip(values, counts):
                            removed[int(value)] += int(count)
                        area[:] = 0
            new[y:y+ch, x:x+cw] = target
            if np.any(before != target):
                changed.append(index)
            index += 1
    new.flush()
    old_background = json.loads((source / "BACKGROUND_COMPLETE.json").read_text())
    additional = sum(removed.values())
    summary = {
        "source_image": str(args.image), "native_shape_yx": [h, w],
        "native_mpp_xy": list(map(float, mpp)),
        "windows": index, "changed_window_indices": changed,
        "changed_windows": len(changed), "block_um": args.block_um,
        "block_pixels": block, "minimum_stained_fraction": args.minimum_stained_fraction,
        "stain_definition": "HSV saturation >20/255 and minimum RGB <220",
        "examined_predicted_blocks": examined_blocks,
        "low_support_blocks_removed": low_support_blocks,
        "additional_removed_pixels": additional,
        "additional_removed_by_class": {str(k): v for k, v in sorted(removed.items())},
        "previous_pixel_background_removed_pixels": old_background["removed_pixels"],
        "expert_annotation_read": False,
        "seconds": monotonic()-started}
    for name in ("CLASSIFIED_COMPLETE.json", "CLEANED_COMPLETE.json", "EVALUATION_RAW.json"):
        shutil.copy2(source / name, args.output / name)
    new_background = dict(old_background)
    new_background["rule"] = (old_background["rule"] + "; then reject predicted pixels "
        f"in physical {args.block_um:g}-um blocks with "
        f"<{100*args.minimum_stained_fraction:g}% H&E stain support")
    new_background["removed_pixels"] = old_background["removed_pixels"] + additional
    new_background["block_support_audit"] = summary
    atomic_json(args.output / "BACKGROUND_COMPLETE.json", new_background)
    # Reuse the audited final mask and immutable vector checkpoints only for
    # windows with an unchanged source mask. New vector work updates the copy.
    shutil.copyfile(source / "classified.final.dat", args.output / "classified.final.dat")
    vectors = args.output / "vector-windows"
    vectors.mkdir()
    changed_set = set(changed)
    for i in range(index):
        if i in changed_set:
            continue
        for suffix in ("json", "geojson"):
            filename = f"{i:03d}.{suffix}"
            os.link(source / "vector-windows" / filename, vectors / filename)
    atomic_json(args.output / "BLOCK_BACKGROUND_COMPLETE.json", summary)
    print(json.dumps({"changed_windows": len(changed),
                      "additional_removed_pixels": additional,
                      "reused_vector_windows": index-len(changed),
                      "seconds": round(summary["seconds"], 2)}), flush=True)


if __name__ == "__main__":
    main()
