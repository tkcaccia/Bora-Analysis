"""One bounded Chiamaka Virchow2 feature case; no queue, child jobs, or retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from PIL import Image
import tifffile


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.source))
    import torch
    from experiments.virchow2_inverse.gpu_ideas import load_model
    from experiments.virchow2_multi.run import encode

    case = json.loads((a.case/"case.json").read_text())
    source = tifffile.imread(a.case/"original.tif")
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB input")
    if not np.allclose(case["mpp_xy"], .5, atol=.01):
        raise ValueError("Virchow2 requires calibrated ~0.5 um/pixel input")
    output = a.case/"features"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.4)
    print(json.dumps(dict(stage="start",pid=os.getpid(),case=str(a.case),
        gpu=torch.cuda.get_device_name(0),image_sha256=sha256(a.case/"original.tif"))),flush=True)
    model = load_model(a.model)
    files = {}
    for name, factor in (("base",1),("scale_1",2),("scale_2",4)):
        start = monotonic()
        image = (source if factor == 1 else np.asarray(Image.fromarray(source).resize(
            (round(source.shape[1]/factor),round(source.shape[0]/factor)),Image.Resampling.LANCZOS)))
        values, metadata = encode(model,image)
        expected = tuple((np.asarray(image.shape[:2])+13)//14)
        if values.shape != (*expected,1280) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} token grid {values.shape}; expected {expected}")
        temp = output/(name+".tmp.npy")
        np.save(temp,values)
        dest = output/(name+".npy")
        os.replace(temp,dest)
        files[name] = dict(shape=list(values.shape),image_shape_yx=list(image.shape[:2]),
            sha256=sha256(dest),seconds=monotonic()-start,**metadata)
        print(json.dumps(dict(stage="checkpoint",name=name,seconds=files[name]["seconds"])),flush=True)
    report = dict(case=str(a.case),checkpoint_sha256=sha256(a.model/"model.safetensors"),
        source_sha256=sha256(a.case/"original.tif"),files=files)
    (output/"FEATURES_COMPLETE.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(stage="complete",report=str(output/"FEATURES_COMPLETE.json"))),flush=True)


if __name__ == "__main__":
    main()
