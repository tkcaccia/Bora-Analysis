#!/usr/bin/env python3
"""Create a nonduplicating index under a user-selected Bora data root."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_LAYOUT = {
    "01_public_tissue_fields": "benchmark_catalog",
    "02_whole_slide_images": "wsi_benchmark",
    "03_visium_gene_clustering": "visium-spot-benchmark",
    "04_private_development": "private_benchmarks",
    "source_datasets": "datasets",
    "preprocessing_checkpoints": "preprocessing",
}


def organize(root: Path, layout: dict[str, str]) -> dict:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    view = root / "BENCHMARKS"
    view.mkdir(exist_ok=True)
    linked: dict[str, str] = {}
    absent: dict[str, str] = {}
    for label, source in layout.items():
        if not label or label in {".", ".."} or "/" in label:
            raise ValueError(f"Invalid category label: {label!r}")
        relative = Path(source)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError(f"Source must be a relative subdirectory: {source!r}")
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Source escapes data root: {source!r}")
        if not target.is_dir():
            absent[label] = source
            continue
        link = view / label
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"Existing link points elsewhere: {link}")
        elif link.exists():
            raise FileExistsError(f"A file blocks the category link: {link}")
        else:
            link.symlink_to(os.path.relpath(target, view))
        linked[label] = source
    readme = view / "README.md"
    lines = ["# Bora benchmark data", "", "Category links point to source directories without copying images.", ""]
    lines += [f"- `{label}/` → `{source}/`" for label, source in linked.items()]
    lines += ["", "Source images, weights and feature caches remain outside Git.", ""]
    if not readme.exists():
        readme.write_text("\n".join(lines))
    return {"data_root": str(root), "linked": linked, "absent": absent}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("BORA_DATA_ROOT", ".data")))
    parser.add_argument("--layout", type=Path,
                        help="Optional JSON object mapping category names to relative subdirectories")
    args = parser.parse_args()
    layout = json.loads(args.layout.read_text()) if args.layout else DEFAULT_LAYOUT
    if not isinstance(layout, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in layout.items()):
        raise ValueError("Layout must be a JSON object of string names and relative paths")
    print(json.dumps(organize(args.root, layout), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
