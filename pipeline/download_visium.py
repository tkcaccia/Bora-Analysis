"""Resumable 10x source downloader; no model or benchmark inference is launched.

Usage: python download.py --root /data/visium
Each dataset is limited to 10 GB of original image bytes. Files are checksummed
after transfer and the spatial archive is inspected for unsafe paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
from urllib.parse import urljoin

import requests

MANIFEST = Path(__file__).with_name("visium_datasets.json")
IMAGE_CAP = 10_000_000_000


def save_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> dict:
    head = requests.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers["Content-Length"])
    part = target.with_name(target.name + ".part")
    if target.exists() and target.stat().st_size == total:
        return {"bytes": total, "sha256": sha256(target), "url": url}
    if target.exists():
        raise RuntimeError(f"Existing file has wrong size: {target}")
    offset = part.stat().st_size if part.exists() else 0
    if offset > total:
        raise RuntimeError(f"Partial file exceeds source length: {part}")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as resp:
        resp.raise_for_status()
        if offset and resp.status_code != 206:
            raise RuntimeError(f"Server did not honor range for {url}")
        with part.open("ab" if offset else "wb") as out:
            for chunk in resp.iter_content(chunk_size=8 << 20):
                if chunk:
                    out.write(chunk)
    if part.stat().st_size != total:
        raise RuntimeError(f"Incomplete transfer: {part}")
    digest = sha256(part)
    os.replace(part, target)
    return {"bytes": total, "sha256": digest, "url": url}


def extract_spatial(archive: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for item in members:
            dest = (directory / item.name).resolve()
            if not dest.is_relative_to(directory.resolve()) or item.issym() or item.islnk() or item.isdev():
                raise RuntimeError(f"Unsafe spatial archive member: {item.name}")
        tar.extractall(directory, members=members)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=list(json.loads(MANIFEST.read_text())) + ["all"], default="all")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    status_file = args.root / "download_status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    for key, spec in manifest.items():
        if args.dataset not in ("all", key):
            continue
        folder = args.root / key
        folder.mkdir(exist_ok=True)
        status.setdefault(key, {})
        for suffix in spec["files"]:
            name = f"{spec['sample']}_{suffix}"
            url = urljoin(spec["base_url"], name)
            if suffix.startswith("tissue_image"):
                head = requests.head(url, timeout=60, allow_redirects=True)
                head.raise_for_status()
                if int(head.headers["Content-Length"]) > IMAGE_CAP:
                    raise RuntimeError(f"Image exceeds 10 GB cap: {url}")
            status[key][name] = download(url, folder / name)
            status[key][name]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save_json(status_file, status)
        extract_spatial(folder / f"{spec['sample']}_spatial.tar.gz", folder)
        status[key]["spatial_extracted"] = True
        matrix_archive = folder / f"{spec['sample']}_filtered_feature_bc_matrix.tar.gz"
        if matrix_archive.exists():
            extract_spatial(matrix_archive, folder)
            status[key]["matrix_extracted"] = True
        save_json(status_file, status)
        print(f"COMPLETE {key}", flush=True)


if __name__ == "__main__":
    main()
