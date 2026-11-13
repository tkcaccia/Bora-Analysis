"""Verify the three frozen, published Benchmark 3 GeoJSON outputs.

This checks saved results. It does not rerun Virchow2 or Bora inference.
Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


CASES = {
    "breast_giotto": "breast",
    "colorectal_seurat": "colorectal",
    "tonsil_spatialexperiment": "tonsil",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_one(key: str, slug: str, record: dict, archive: Path | None) -> dict:
    if archive is None:
        with urlopen(record["download"], timeout=120) as response:
            compressed = response.read()
    else:
        path = archive / slug / (slug + "-Visium-Bora-final.geojson.gz")
        compressed = path.read_bytes()
    assert len(compressed) == record["compressed_bytes"], (key, "compressed size")
    assert sha256(compressed) == record["compressed_sha256"], (key, "compressed SHA256")
    raw = gzip.decompress(compressed)
    assert len(raw) == record["geojson_bytes"], (key, "GeoJSON size")
    assert sha256(raw) == record["global_geojson_sha256"], (key, "GeoJSON SHA256")
    geo = json.loads(raw)
    assert geo["type"] == "FeatureCollection", key
    features = geo["features"]
    assert len(features) == record["group_count"], (key, "group count")
    ids = [f["properties"]["value"] for f in features]
    assert len(ids) == len(set(ids)), (key, "duplicate group ID")
    assert all(f["geometry"]["type"] in ("Polygon", "MultiPolygon") for f in features)
    return {"case": key, "groups": len(ids), "geojson_sha256": sha256(raw), "status": "verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True,
                        help="Downloaded site/visium/FULLSECTION_PROVENANCE.json")
    parser.add_argument("--archive", type=Path,
                        help="Local archive/visium-fullsection-2026-11-10-v1; omit to download pinned outputs")
    args = parser.parse_args()
    records = json.loads(args.provenance.read_text())
    assert set(records) == set(CASES), "Unexpected provenance case list"
    for key, slug in CASES.items():
        result = verify_one(key, slug, records[key], args.archive)
        print(json.dumps(result, sort_keys=True), flush=True)
    print("VERIFIED 3/3 published GeoJSON outputs")


if __name__ == "__main__":
    main()
