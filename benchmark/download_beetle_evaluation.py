"""Fetch the original BEETLE evaluation ROIs without the 147-GB WSI archive.

Only the 170 PNG members under images/evaluation/rois/ are downloaded. The
sequestered reference masks are neither available nor inferred by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import time
import zipfile
import zlib

import requests


RECORD = "https://zenodo.org/records/16812932"
ARCHIVE_URL = f"{RECORD}/files/images.zip?download=1"
OVERVIEW_URL = f"{RECORD}/files/data_overview.csv?download=1"
ARCHIVE_SIZE = 147_212_330_281
ARCHIVE_MD5 = "7bfd8524615fc4914b9998b8bfc80f9e"
OVERVIEW_MD5 = "fa3093bec3a80e7f3464dcc7cbc36fad"
MEMBER_PREFIX = "images/evaluation/rois/"
ROI_NAME = re.compile(r"patient[0-9]+_wsi[0-9]+_roi[0-9]+\.png")


def get_with_retry(session: requests.Session, url: str, headers: dict | None = None) -> requests.Response:
    for attempt in range(5):
        try:
            response = session.get(url, headers=headers, timeout=90)
            if response.status_code in (429, 500, 502, 503, 504):
                response.close()
                time.sleep(min(30, 2**attempt))
                continue
            response.raise_for_status()
            return response
        except (requests.RequestException, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("Zenodo did not return a successful response")


class RangeFile(io.RawIOBase):
    def __init__(self, session: requests.Session):
        self.session = session
        head = session.head(ARCHIVE_URL, timeout=30)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        if self.size != ARCHIVE_SIZE:
            raise ValueError(f"Archive size changed: {self.size}")
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        position = (offset, self.position + offset, self.size + offset)[whence]
        if position < 0:
            raise ValueError("Negative ZIP seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        end = min(self.position + size - 1, self.size - 1)
        response = get_with_retry(
            self.session, ARCHIVE_URL,
            {"Range": f"bytes={self.position}-{end}"},
        )
        expected_range = f"bytes {self.position}-{end}/{self.size}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
            raise ValueError("Server failed to honor the exact byte range")
        content = response.content
        response.close()
        if len(content) != end - self.position + 1:
            raise IOError("Short ZIP byte-range response")
        self.position = end + 1
        return content


def png_dimensions(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("Invalid PNG header")
    return struct.unpack(">II", data[16:24])


def download(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    overview = get_with_retry(session, OVERVIEW_URL).content
    if hashlib.md5(overview).hexdigest() != OVERVIEW_MD5:
        raise ValueError("BEETLE overview checksum mismatch")
    overview_path = destination / "data_overview.csv"
    overview_path.write_bytes(overview)
    rows = list(csv.DictReader(io.StringIO(overview.decode("utf-8-sig"))))
    development = {row["patient_id"] for row in rows if row["split"] == "development"}
    evaluation_rows = [row for row in rows if row["split"] == "evaluation"]
    evaluation = {row["patient_id"] for row in evaluation_rows}
    if len(evaluation_rows) != 54 or len(evaluation) != 54 or development & evaluation:
        raise ValueError("Unexpected BEETLE evaluation patient split")
    by_wsi = {row["name"]: row for row in evaluation_rows}

    inventory = []
    archive = RangeFile(session)
    with zipfile.ZipFile(archive) as zip_file:
        members = sorted(
            (info for info in zip_file.infolist()
             if info.filename.startswith(MEMBER_PREFIX) and not info.is_dir()),
            key=lambda item: item.filename,
        )
        if len(members) != 170:
            raise ValueError(f"Expected 170 evaluation ROI PNGs, found {len(members)}")
        for index, info in enumerate(members, 1):
            name = info.filename[len(MEMBER_PREFIX):]
            if not ROI_NAME.fullmatch(name):
                raise ValueError(f"Unexpected archive member: {info.filename}")
            wsi_name = name.rsplit("_roi", 1)[0]
            row = by_wsi.get(wsi_name)
            if row is None:
                raise ValueError(f"No official overview row for {name}")
            output = destination / "rois" / name
            output.parent.mkdir(exist_ok=True)
            existing_valid = False
            if output.exists() and output.stat().st_size == info.file_size:
                data = output.read_bytes()
                existing_valid = zlib.crc32(data) == info.CRC
                if not existing_valid:
                    data = zip_file.read(info)
            else:
                data = zip_file.read(info)
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                raise ValueError(f"ZIP CRC/size mismatch: {name}")
            dimensions = png_dimensions(data)
            if not existing_valid:
                temporary = output.with_suffix(".png.part")
                temporary.write_bytes(data)
                os.replace(temporary, output)
            inventory.append({
                "filename": name, "path": f"rois/{name}",
                "patient_id": row["patient_id"], "wsi_id": row["wsi_id"],
                "source_center": row["source"], "scanner": row["scanner"],
                "specimen_type": row["specimen_type"],
                "width": dimensions[0], "height": dimensions[1],
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "zip_crc32": f"{info.CRC:08x}",
            })
            if index % 20 == 0 or index == len(members):
                print(f"Verified {index}/{len(members)} ROIs", flush=True)

    report = {
        "dataset": "BEETLE evaluation ROIs", "zenodo_record": RECORD,
        "archive_url": ARCHIVE_URL, "archive_size_bytes": ARCHIVE_SIZE,
        "archive_published_md5_not_verified": ARCHIVE_MD5,
        "download_scope": "170 original evaluation ROI PNG members only; full ZIP and WSIs not downloaded",
        "source_overview_md5_verified": OVERVIEW_MD5,
        "patient_count": len(evaluation), "image_count": len(inventory),
        "independence": "Official evaluation patient IDs and centers are disjoint from official development set; local Bora-training overlap requires a separate source-ID audit.",
        "calibration": "Provider reports approximately 0.5 micrometres per pixel; exact per-ROI spacing is not embedded in the PNGs.",
        "annotation": "Ground-truth masks are sequestered on the BEETLE Grand Challenge; no local quality scores can be computed from this acquisition.",
        "images": inventory,
    }
    manifest = destination / "acquisition.json"
    temporary = manifest.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = download(args.destination)
    print(json.dumps({"image_count": result["image_count"], "patient_count": result["patient_count"]}))
