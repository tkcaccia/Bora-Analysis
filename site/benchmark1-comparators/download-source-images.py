#!/usr/bin/env python3
"""Download an original Benchmark 1 provider file, with a pinned checksum.

This retrieves provider archives, not the resized images displayed by Bora.
Run --list first; only the explicitly requested file is downloaded.
"""

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen


BCSS_REVISION = "502d4a3fbc77dbaca6f4664c19e2379ff077d418"
BCSS_SHA256 = (
    "a5a53bc4b53dce4a1a25ef40a00b9c2bd7c4b46a606e919ee9d6c54a9b363228",
    "f2afdd04c6d20d925a1a520f47ba519ae5ff71eb2ef75d5fc0da890ce367eea7",
    "7f6c829e371502607c91de688bc964cf4178395920e392f9cfceedaad9b79c4b",
    "3e3ed641a46a7b7cba6055c86d82ace56eeb010812d21f5a55457506f358b647",
    "559e604a859aa3379b09e4da0804d4fc9eaeb4e72a91b7c94d9a972882818eea",
    "8109a198404cfcff33ec34791704aceffb02905d56dd7cfd46084f3bc8dc31e2",
    "a5f03f0bf706b4ffbe43441eb0f337415a7aa25db90e7ff964c7987422820767",
    "55d39f2ccc5ae95a1d4703743199737b1792c03a5642fdfddd8316af94bdf5ad",
    "f3ea6f21aba26659e509ccd06b9c36a14d6f2e09e4db2b8d9c5415956a8bcd01",
    "4e412a29b8fe7f87c4d9bbbc2198f0992058b91388acdcd7901f6c72ed7aef9f",
)


def source_files():
    files = {}
    for i, digest in enumerate(BCSS_SHA256):
        name = f"train-{i:05d}-of-00010.parquet"
        files[("bcss", name)] = (
            f"https://huggingface.co/datasets/MedOtter/BCSS/resolve/"
            f"{BCSS_REVISION}/data/{name}", "sha256", digest
        )
    zenodo = {
        ("tiger", "roi-level-annotations.zip"):
            (6014422, "51d65ecc2837a98ff5ceee17fdedd7bd"),
        ("gcss", "GCSS.zip"):
            (8176354, "a1dfe099ca66475bef4229ad76434112"),
        ("ignite", "images.zip"):
            (17735903, "1a8bf1f245d1c7c7405f8f869b04cacc"),
        ("peso", "peso_testset_png.zip"):
            (5137717, "c212aa96a7f79eb177ce1e4bfedb1ac1"),
    }
    for (dataset, name), (record, digest) in zenodo.items():
        files[(dataset, name)] = (
            f"https://zenodo.org/records/{record}/files/{name}?download=1",
            "md5", digest
        )
    return files


def download(url, destination, algorithm, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = checksum(destination, algorithm)
        if actual == expected:
            print(f"Already verified: {destination}")
            return
        raise SystemExit(f"Existing file has wrong {algorithm}: {destination}")
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise SystemExit(f"Remove or resolve existing partial download first: {temporary}")
    digest = hashlib.new(algorithm)
    request = Request(url, headers={"User-Agent": "Bora-source-download/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("xb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
                digest.update(block)
            out.flush()
            os.fsync(out.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"{algorithm} mismatch: expected {expected}, got {actual}")
        temporary.replace(destination)
        print(f"Verified {algorithm}: {destination}")
    except Exception:
        print(f"Partial file retained for inspection: {temporary}")
        raise


def checksum(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show pinned source files")
    parser.add_argument("--dataset", choices=("bcss", "tiger", "gcss", "ignite", "peso"))
    parser.add_argument("--file", help="one file name from --list")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark1-sources"))
    args = parser.parse_args()
    files = source_files()
    if args.list:
        for (dataset, name), (url, algorithm, digest) in files.items():
            print(f"{dataset:7} {name:35} {algorithm}:{digest}\n        {url}")
        return
    if not args.dataset or not args.file:
        parser.error("specify --dataset and --file, or use --list")
    key = (args.dataset, args.file)
    if key not in files:
        parser.error("unknown source file; run --list")
    url, algorithm, digest = files[key]
    download(url, args.output_dir / args.dataset / args.file, algorithm, digest)


if __name__ == "__main__":
    main()
