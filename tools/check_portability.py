#!/usr/bin/env python3
"""Fail when public text files contain a personal workstation or mount path."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".json", ".html", ".js", ".css",
                 ".sh", ".r", ".rmd", ".txt", ".yaml", ".yml",
                 ".toml", ".tsv", ".csv"}
FORBIDDEN = (
    re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    re.compile(r"/mnt/sata_ssd/"),
    re.compile(r"/media/user/"),
    re.compile(r"137\.158\.\d+\.\d+"),
    re.compile(r"\bChiamaka\b"),
    re.compile(r"\bStefano remote (?:PC|computer)\b", re.IGNORECASE),
)


def candidate_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, check=True,
    )
    return [ROOT / part.decode() for part in result.stdout.split(b"\0") if part]


def main() -> int:
    bad: list[str] = []
    checked = 0
    for path in candidate_paths():
        if path.resolve() == Path(__file__).resolve() or not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20_000_000:
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        checked += 1
        for number, line in enumerate(content.splitlines(), 1):
            if any(pattern.search(line) for pattern in FORBIDDEN):
                bad.append(f"{path.relative_to(ROOT)}:{number}")
    if bad:
        print("Personal host paths or identifiers in public text files:", file=sys.stderr)
        print("\n".join(bad), file=sys.stderr)
        return 1
    print(f"Portability scan passed: {checked} public text files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
