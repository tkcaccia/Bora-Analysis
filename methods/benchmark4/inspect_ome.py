#!/usr/bin/env python3
"""Record measured OME X/Y spacing for one downloaded 10x H&E image."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import tifffile


def mpp_from_ome(path: Path) -> tuple[float, float]:
    with tifffile.TiffFile(path) as tif:
        xml = tif.ome_metadata
    if not xml:
        raise ValueError("The image has no OME microscopy metadata")
    pixels = next((e for e in ET.fromstring(xml).iter()
                   if e.tag.rsplit("}", 1)[-1] == "Pixels"), None)
    if pixels is None:
        raise ValueError("OME Pixels element is missing")
    values = []
    for axis in ("X", "Y"):
        value = float(pixels.attrib[f"PhysicalSize{axis}"])
        unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm")
        factor = {"µm": 1.0, "um": 1.0, "micrometer": 1.0,
                  "nm": 0.001, "mm": 1000.0}.get(unit)
        if factor is None or not (0 < value * factor < 100):
            raise ValueError(f"Unsupported OME spacing/unit on {axis}: {value} {unit}")
        values.append(value * factor)
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    image = args.image.resolve(strict=True)
    x, y = mpp_from_ome(image)
    target = image.parent / "image_metadata.json"
    if target.exists():
        raise FileExistsError(f"Preserve the existing metadata: {target}")
    target.write_text(json.dumps({"image": str(image),
                                  "source": "OME Pixels PhysicalSizeX/Y",
                                  "mpp_x": x, "mpp_y": y}, indent=2) + "\n")
    print(f"{target}: {x:.9f} × {y:.9f} µm/pixel")


if __name__ == "__main__":
    main()
