#!/usr/bin/env python3
"""Benchmark wrapper for Bora's reusable reviewed-support materializer.

The pixel geometry and exact comparison live in the Bora package. This file
only exposes the same source-bound procedure as a command for benchmark runs.
"""

import argparse
import json

from bora.r_tissue_support import materialize_reviewed_tissue_support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibrated_he", help="Calibrated uint8 RGB H&E TIFF")
    parser.add_argument("reviewed_geojson", help="Reviewed pixel-edge tissue geometry")
    parser.add_argument("output_dir", help="New output directory")
    parser.add_argument("--minimum-free-gib", type=float, default=100,
                        help="Bora-volume reserve after projected mask (default: 100)")
    args = parser.parse_args()
    receipt = materialize_reviewed_tissue_support(
        args.calibrated_he, args.reviewed_geojson, args.output_dir,
        minimum_free_gib=args.minimum_free_gib)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
