"""Common tissue support and class colors for matched public WSI panels.

This is display-only. It never writes to or alters the saved prediction TIFFs.
The caller supplies the exact final Bora support raster aligned to the H&E
view and the existing 25-row RGB tissue-role palette.
"""

from __future__ import annotations

import numpy as np


# VISTA-PATH native IDs 0..13 to the established 24-role Bora palette.
VISTA_TO_SHARED = np.array(
    (0, 1, 3, 12, 5, 8, 15, 11, 14, 13, 10, 9, 6, 2), dtype=np.uint8
)


def map_vista(labels: np.ndarray) -> np.ndarray:
    """Map predeclared matching tissue names, preserving unclassified zero."""
    if labels.ndim != 2 or labels.min() < 0 or labels.max() > 13:
        raise ValueError("VISTA labels must be a 2-D array with IDs 0..13")
    return VISTA_TO_SHARED[labels]


def overlay(
    he: np.ndarray, labels: np.ndarray, support: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """Blend class color on Bora support, leaving glass and zeros as H&E."""
    if he.ndim != 3 or he.shape[2] != 3 or he.dtype != np.uint8:
        raise ValueError("H&E must be uint8 RGB")
    if labels.shape != he.shape[:2] or support.shape != labels.shape:
        raise ValueError("All input frames must align")
    if palette.shape != (25, 3) or palette.dtype != np.uint8:
        raise ValueError("Palette must have RGB rows for background plus 24 roles")
    if labels.min() < 0 or labels.max() > 24:
        raise ValueError("Class labels must be in 0..24")
    color = palette[labels]
    combined = np.rint(
        .45 * he.astype(np.float32) + .55 * color.astype(np.float32)
    ).astype(np.uint8)
    combined[(labels == 0) | ~support] = he[(labels == 0) | ~support]
    return combined
