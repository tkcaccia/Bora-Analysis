"""Small OME header contract tests; no whole-slide bytes are read."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from methods.benchmark4.inspect_ome import mpp_from_ome


class TestOmeSpacing(unittest.TestCase):
    def test_measured_xy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.ome.tif"
            tifffile.imwrite(path, np.zeros((8, 9), dtype=np.uint8), ome=True,
                             metadata={"PhysicalSizeX": 0.55,
                                       "PhysicalSizeXUnit": "µm",
                                       "PhysicalSizeY": 0.57,
                                       "PhysicalSizeYUnit": "µm"})
            self.assertEqual(mpp_from_ome(path), (0.55, 0.57))

    def test_unlabelled_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.tif"
            tifffile.imwrite(path, np.zeros((8, 9), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "no OME"):
                mpp_from_ome(path)


if __name__ == "__main__":
    unittest.main()
