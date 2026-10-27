"""Create auditable per-image and size-band tables from completed resource runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def size_band(megapixels: float) -> str:
    if megapixels < 2:
        return "<2 MP"
    if megapixels < 16:
        return "2-<16 MP"
    if megapixels < 100:
        return "16-<100 MP"
    if megapixels < 1000:
        return "100-<1000 MP"
    return "WSI >=1000 MP"


def percentile(values: list[float], q: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    records = []
    failures = []
    for path in args.run:
        data = json.loads(path.read_text())
        for record in data["records"]:
            if record["status"] == "complete" and record.get("cleanup_verified"):
                records.append(record)
            else:
                failures.append({"run": str(path), "id": record.get("id"),
                                 "method": record.get("method"),
                                 "error": record.get("error")})
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["method"], record["input_sha256"]), []).append(record)
    image_rows = []
    for (method, input_hash), repetitions in sorted(groups.items()):
        reference = repetitions[0]
        dimensions = {r["native_pixels"] for r in repetitions}
        spacings = {tuple(r["mpp_xy"]) for r in repetitions}
        if len(dimensions) != 1 or len(spacings) != 1:
            raise ValueError(f"inconsistent size/spacing for {method} {input_hash}")
        image_rows.append({
            "method": method, "input": reference["input"],
            "input_sha256": input_hash, "repeats": len(repetitions),
            "native_megapixels": reference["native_megapixels"],
            "tissue_megapixels": reference["tissue_megapixels"],
            "tissue_area_mm2": reference["tissue_area_mm2"],
            "mpp_x": reference["mpp_xy"][0], "mpp_y": reference["mpp_xy"][1],
            "size_band": size_band(reference["native_megapixels"]),
            "median_cold_seconds": statistics.median(r["wall_seconds_cold"] for r in repetitions),
            "max_host_rss_mb": max(r["peak_process_tree_rss_mb"] for r in repetitions),
            "max_owned_gpu_mb": max((r["peak_owned_gpu_memory_mb"] or 0) for r in repetitions),
        })
    columns = ["method", "input", "input_sha256", "repeats", "native_megapixels",
               "tissue_megapixels", "tissue_area_mm2", "mpp_x", "mpp_y",
               "size_band", "median_cold_seconds", "max_host_rss_mb", "max_owned_gpu_mb"]
    write_csv(args.output_dir / "per_image.csv", image_rows, columns)
    band_rows = []
    by_band: dict[tuple[str, str], list[dict]] = {}
    for row in image_rows:
        by_band.setdefault((row["method"], row["size_band"]), []).append(row)
    for (method, band), items in sorted(by_band.items()):
        seconds = [x["median_cold_seconds"] for x in items]
        band_rows.append({"method": method, "size_band": band, "images": len(items),
                          "median_native_megapixels": statistics.median(x["native_megapixels"] for x in items),
                          "median_cold_seconds": statistics.median(seconds),
                          "p95_cold_seconds": percentile(seconds, .95),
                          "max_host_rss_mb": max(x["max_host_rss_mb"] for x in items),
                          "max_owned_gpu_mb": max(x["max_owned_gpu_mb"] for x in items)})
    write_csv(args.output_dir / "by_method_and_size.csv", band_rows,
              ["method", "size_band", "images", "median_native_megapixels",
               "median_cold_seconds", "p95_cold_seconds", "max_host_rss_mb",
               "max_owned_gpu_mb"])
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"{len(image_rows)} method-image groups, {len(failures)} failures")


if __name__ == "__main__":
    main()
