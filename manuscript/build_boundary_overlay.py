"""Render the WSSS4LUAD manuscript comparison with visible green borders."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import binary_dilation
from benchmark.runtime import BORA_ROOT


ROOT = Path("/mnt/sata_ssd/Bora-benchmark/results/bora-pilot-independent-2026-10-16")
CASE = ROOT / "wsss4luad/error_0/tile_90"
OUT = BORA_ROOT / "docs/figures/wsss4luad_90_overlay_green.png"


def internal_border(labels: np.ndarray) -> np.ndarray:
    border = np.zeros(labels.shape, dtype=bool)
    border[:-1] |= (labels[:-1] != labels[1:]) & (labels[:-1] > 0) & (labels[1:] > 0)
    border[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] > 0) & (labels[:, 1:] > 0)
    return border


def main() -> None:
    image = tifffile.imread(ROOT / "wsss4luad/image.tif")
    masks = (
        tifffile.imread(ROOT / "wsss4luad/normalized_truth.tif"),
        tifffile.imread(CASE / "coarse.tif"),
        np.squeeze(tifffile.imread(CASE / "refined.ome.tif")),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")
    axes[0].set_title("H&E")
    for ax, labels, name in zip(axes[1:], masks, ("Reference", "Tile baseline", "Bora")):
        border = internal_border(labels)
        # A dark-green one-pixel halo keeps the light-green trace legible on both
        # pale gaps and saturated H&E. Dilation changes only this display image.
        overlay = np.zeros((*border.shape, 4), dtype=np.float32)
        overlay[binary_dilation(border, iterations=1)] = (0.05, 0.27, 0.10, 1.0)
        overlay[border] = (0.58, 1.0, 0.50, 1.0)
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
