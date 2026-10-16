"""Acquire pinned public files or import an approved local ZIP; never store data in Git."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/sata_ssd/Bora-benchmark")
LOCK = Path(__file__).with_name("sources.lock.json")


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(path, entry):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    if entry.get("size") is not None and path.stat().st_size != entry["size"]:
        raise ValueError(f"Size mismatch: {path}")
    for algorithm in ("sha256", "md5"):
        if entry.get(algorithm) and digest(path, algorithm) != entry[algorithm]:
            raise ValueError(f"{algorithm} mismatch: {path}")
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "sha256": digest(path)}


def fetch(entry, destination, reserve_gib=20):
    destination = Path(destination)
    if destination.exists():
        return verify(destination, entry)
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Resume a known-size partial when the server honors byte ranges.
    for attempt in range(3):
        try:
            start = partial.stat().st_size if partial.exists() and entry.get("size") else 0
            if start == entry.get("size"):
                result = verify(partial, entry)
                partial.replace(destination)
                result["path"] = str(destination.resolve())
                return result
            headers = {"User-Agent": "Bora-benchmark/0.1"}
            if start:
                headers["Range"] = f"bytes={start}-"
            request = urllib.request.Request(entry["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("Download returned an HTML/login page")
                if start and response.status == 206:
                    expected = f"bytes {start}-"
                    if not response.headers.get("Content-Range", "").startswith(expected):
                        raise ValueError("Server returned the wrong resumed byte range")
                    mode = "ab"
                elif response.status == 200:
                    mode = "wb"
                else:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                with partial.open(mode) as out:
                    while chunk := response.read(8 * 1024 * 1024):
                        if shutil.disk_usage(destination.parent).free < len(chunk) + reserve_gib * 2**30:
                            raise ValueError("Insufficient space while streaming download")
                        out.write(chunk)
            result = verify(partial, entry)
            partial.replace(destination)
            result["path"] = str(destination.resolve())
            return result
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def ranged_fetch(entry, destination, workers):
    """Download large known-size files in resumable, checked HTTP byte ranges."""
    destination = Path(destination)
    if destination.exists():
        return verify(destination, entry)
    size, chunk_size = entry["size"], 64 * 2**20
    parts = destination.with_name(destination.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [(lo, min(size - 1, lo + chunk_size - 1)) for lo in range(0, size, chunk_size)]
    previous = destination.with_name(destination.name + ".part")
    # A stopped sequential transfer can seed complete chunks, without discarding it.
    if previous.exists():
        with previous.open("rb") as seed:
            for lo, hi in ranges:
                chunk = parts / str(lo)
                if hi >= previous.stat().st_size:
                    break
                if not chunk.exists():
                    seed.seek(lo)
                    chunk.write_bytes(seed.read(hi - lo + 1))

    def transfer(bounds):
        lo, hi = bounds
        chunk = parts / str(lo)
        if chunk.exists() and chunk.stat().st_size == hi - lo + 1:
            return chunk
        for attempt in range(3):
            try:
                request = urllib.request.Request(entry["url"], headers={
                    "User-Agent": "Bora-benchmark/0.1", "Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    expected = f"bytes {lo}-{hi}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the exact byte range")
                    with chunk.open("wb") as out:
                        shutil.copyfileobj(response, out, length=2**20)
                if chunk.stat().st_size != hi - lo + 1:
                    raise ValueError("Truncated HTTP range")
                print(f"Downloaded range {lo}-{hi}", flush=True)
                return chunk
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(executor.map(transfer, ranges))
    assembled = destination.with_name(destination.name + ".assembled")
    with assembled.open("wb") as out:
        for chunk in chunks:
            with chunk.open("rb") as handle:
                shutil.copyfileobj(handle, out, length=8 * 2**20)
    result = verify(assembled, entry)
    assembled.replace(destination)
    result["path"] = str(destination.resolve())
    return result


def extract_zip(archive, destination, reserve_gib=20):
    """Reject traversal, links, collisions and inadequate disk space before extraction."""
    destination = Path(destination).resolve()
    marker = destination / ".bora-extraction.json"
    archive_hash = digest(archive)
    if marker.exists():
        if json.loads(marker.read_text())["archive_sha256"] != archive_hash:
            raise ValueError("Extraction exists for a different archive; select a new root")
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        names = set()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or "\\" in member.filename:
                raise ValueError("Unsafe ZIP path")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ZIP symlinks are unsupported")
            if member.filename in names:
                raise ValueError("Duplicate ZIP member")
            names.add(member.filename)
            if target.exists() and not member.is_dir():
                # Interrupted extraction must be inspected, never overwritten silently.
                raise ValueError(f"Extraction collision: {target}")
        needed = sum(member.file_size for member in members) + reserve_gib * 2**30
        if shutil.disk_usage(destination).free < needed:
            raise ValueError("Insufficient free space for extraction and reserved capacity")
        zipped.extractall(destination)
    marker.write_text(json.dumps({"archive_sha256": archive_hash, "members": len(members)}, indent=2) + "\n")


def extract_tar(archive, destination, reserve_gib=20):
    """Extract regular files and directories only; shard-specific markers allow paired TARs."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_hash = digest(archive)
    marker = destination / f".bora-tar-{archive_hash}.json"
    if marker.exists():
        return
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        names = set()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or "\\" in member.name:
                raise ValueError("Unsafe TAR path")
            if not (member.isfile() or member.isdir()):
                raise ValueError("TAR links/devices are unsupported")
            if member.name in names or (target.exists() and member.isfile()):
                raise ValueError("TAR duplicate or extraction collision")
            names.add(member.name)
        if shutil.disk_usage(destination).free < sum(m.size for m in members) + reserve_gib * 2**30:
            raise ValueError("Insufficient space for TAR extraction and reserved capacity")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out, length=8 * 2**20)
    marker.write_text(json.dumps({"archive_sha256": archive_hash, "members": len(members)}, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    catalog = json.loads(LOCK.read_text())
    parser.add_argument("dataset", nargs="?", choices=tuple(catalog))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--list-datasets", action="store_true",
                        help="List pinned sources and access requirements without downloading")
    parser.add_argument("--plan", action="store_true", help="Show sources without downloading")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check every pinned source file already on disk; never download or extract")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--local-archive", type=Path, help="Import a lawfully obtained ZIP for an access-controlled dataset")
    parser.add_argument("--reserve-gib", type=float, default=20)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1,
                        help="Parallel byte-range requests for large HTTPS files (1-8)")
    args = parser.parse_args(argv)
    if args.list_datasets:
        print(json.dumps({name: {"access": entry["access"],
                                 "files": len(entry.get("files", [])),
                                 "canonical_source": entry.get("canonical_source")}
                          for name, entry in catalog.items()}, indent=2))
        return
    if args.dataset is None:
        parser.error("select a dataset or use --list-datasets")
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be non-negative")
    source = catalog[args.dataset]
    if args.plan:
        print(json.dumps(source, indent=2))
        return
    if args.verify_only:
        if args.extract or args.local_archive:
            parser.error("--verify-only cannot be combined with --extract or --local-archive")
        if not source.get("files"):
            parser.error("this source has no pinned downloadable files; inspect its access instructions")
        raw = args.root.resolve() / "raw" / args.dataset
        records = []
        failures = []
        for entry in source.get("files", []):
            try:
                records.append(verify(raw / entry["path"], entry))
            except ValueError as exc:
                failures.append(str(exc))
        print(json.dumps({"dataset": args.dataset, "verified": records,
                          "failures": failures, "access": source["access"]}, indent=2))
        if failures:
            raise SystemExit(1)
        return
    raw = args.root.resolve() / "raw" / args.dataset
    raw.mkdir(parents=True, exist_ok=True)
    metadata = args.root.resolve() / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    if args.local_archive:
        archive = args.local_archive.resolve()
        if not zipfile.is_zipfile(archive):
            raise ValueError("Local input must be a ZIP archive")
        records = [verify(archive, {})]
        source = {"access": "user-supplied approved archive", "dataset": args.dataset,
                  "canonical_source": source["canonical_source"]}
        archives = [archive]
    else:
        if source["access"] == "manual":
            raise ValueError(source["instructions"])
        required = sum(f.get("size", 0) for f in source["files"] if not (raw / f["path"]).exists())
        if args.workers > 1:
            required *= 2  # Chunks and assembled archive coexist.
        if shutil.disk_usage(raw).free < required + args.reserve_gib * 2**30:
            raise ValueError("Insufficient space for download and reserved capacity")
        if source["access"] == "huggingface":
            # The installed hf command provides resumable Xet downloads without copying tokens.
            hf = Path(sys.executable).with_name("hf")
            if not hf.exists():
                raise ValueError("Install benchmark/requirements.txt in this Python environment")
            selected = [f["path"] for f in source["files"]] if source.get("files_only") else []
            subprocess.run([str(hf), "download", source["repo_id"], *selected, "--repo-type", "dataset",
                            "--revision", source["revision"], "--local-dir", str(raw),
                            "--max-workers", "2"], check=True)
            records = [verify(raw / f["path"], f) for f in source["files"]]
        else:
            records = [(ranged_fetch(f, raw / f["path"], args.workers)
                        if args.workers > 1 and f.get("size", 0) > 128 * 2**20 else fetch(f, raw / f["path"], args.reserve_gib))
                       for f in source["files"]]
        archives = [raw / f["path"] for f in source["files"] if f["path"].endswith((".zip", ".tar", ".tgz", ".tar.gz"))]
    report = {"observed_at": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
              "source": source, "files": records, "extracted": False}
    report_path = metadata / f"{args.dataset}-acquisition.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.extract:
        for archive in archives:
            destination = raw / "extracted"
            if archive.suffix == ".zip" and len(archives) > 1:
                destination /= archive.stem
            if archive.suffix == ".zip":
                extract_zip(archive, destination, args.reserve_gib)
            else:
                extract_tar(archive, destination, args.reserve_gib)
        report["extracted"] = bool(archives)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(records)} file(s); provenance: {report_path}")


if __name__ == "__main__":
    main()
