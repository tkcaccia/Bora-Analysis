"""Fail-fast, cold-process resource measurements for WSI segmentation methods.

The manifest contains explicit command arrays and immutable input identities.
This runner never submits to a scheduler and must only be started when the
declared local GPU is available. It stops the entire manifest on one failure.

Example: python benchmark/size_resource_benchmark.py --manifest runs.json \
    --output results.json --allow-gpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def proc_table() -> dict[int, tuple[int, int, int, int]]:
    """PID -> (parent PID, process group, startticks, RSS bytes)."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            # The command field may contain spaces or parentheses.
            fields = stat[stat.rfind(")") + 2:].split()
            ppid = int(fields[1])
            process_group = int(fields[2])
            startticks = int(fields[19])
            rss_pages = int((entry / "statm").read_text().split()[1])
            found[int(entry.name)] = (ppid, process_group, startticks,
                                      rss_pages * page_size)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return found


def owned_pids(table: dict[int, tuple[int, int, int, int]],
               seen: dict[int, int], root: int) -> set[int]:
    current = {pid for pid, tick in seen.items()
               if pid in table and table[pid][2] == tick}
    if root in table and root not in seen:
        seen[root] = table[root][2]
        current.add(root)
    while True:
        children = {pid for pid, (ppid, group, tick, _) in table.items()
                    if (ppid in current or group == root) and pid not in current}
        if not children:
            break
        for pid in children:
            seen[pid] = table[pid][2]
        current.update(children)
    return current


def stop_owned(seen: dict[int, int], root: int) -> None:
    """Bounded termination of only identities observed in our process group/tree."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        table = proc_table()
        remaining = owned_pids(table, seen, root)
        if not remaining:
            return
        for pid in remaining:
            if pid in table and table[pid][2] == seen.get(pid):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not owned_pids(proc_table(), seen, root):
                return
            time.sleep(0.5)


def owned_gpu_memory_mb(pids: set[int], required: bool) -> float | None:
    if not required:
        return None
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=3, check=True,
    )
    total = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) in pids:
            match = re.search(r"\d+(?:\.\d+)?", parts[1])
            if match:
                total += float(match.group())
    return total


def require_case(case: dict) -> None:
    required = ("id", "method", "command", "cwd", "input", "input_sha256",
                "native_width", "native_height", "mpp_x", "mpp_y",
                "tissue_pixels", "outputs", "timeout_seconds", "gpu")
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError(f"case missing {missing}")
    if not isinstance(case["command"], list) or not case["command"] or not all(
        isinstance(part, str) for part in case["command"]
    ):
        raise ValueError("command must be a nonempty argv array; shell is never used")
    if any(float(case[key]) <= 0 for key in ("native_width", "native_height",
                                          "mpp_x", "mpp_y", "timeout_seconds")):
        raise ValueError("dimensions, spacing and timeout must be positive")
    if not 0 <= int(case["tissue_pixels"]) <= int(case["native_width"]) * int(case["native_height"]):
        raise ValueError("tissue_pixels outside native image")
    if not case["outputs"]:
        raise ValueError("list at least one required output")


def check_input(case: dict) -> None:
    path = Path(case["input"])
    if not path.is_file() or digest(path) != case["input_sha256"]:
        raise ValueError(f"missing or changed input for {case['id']}: {path}")


def run_case(case: dict, log_dir: Path, gpu_allowed: bool,
             sample_seconds: float) -> dict:
    require_case(case)
    check_input(case)
    if case["gpu"] and not gpu_allowed:
        raise ValueError("GPU case requires --allow-gpu after admission and cleanup checks")
    for name in case["outputs"]:
        if Path(name).exists():
            raise FileExistsError(f"cold output already exists: {name}")
    cwd = Path(case["cwd"])
    if not cwd.is_dir():
        raise ValueError(f"missing cwd: {cwd}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (case["id"] + ".log")
    command_hash = hashlib.sha256(json.dumps(case["command"]).encode()).hexdigest()
    start = time.monotonic()
    peak_rss = 0
    peak_gpu = 0.0
    samples = 0
    seen: dict[int, int] = {}
    failure = None
    with log_path.open("wb") as log:
        process = subprocess.Popen(case["command"], cwd=cwd, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                table = proc_table()
                pids = owned_pids(table, seen, process.pid)
                peak_rss = max(peak_rss, sum(table[p][3] for p in pids if p in table))
                gpu_mb = owned_gpu_memory_mb(pids, bool(case["gpu"]))
                if gpu_mb is not None:
                    peak_gpu = max(peak_gpu, gpu_mb)
                samples += 1
                if process.poll() is not None:
                    break
                if time.monotonic() - start > case["timeout_seconds"]:
                    raise TimeoutError(f"{case['id']} exceeded bounded time")
                time.sleep(sample_seconds)
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        failure += "; parent termination unverified"
            stop_owned(seen, process.pid)
        elapsed = time.monotonic() - start
        exit_code = process.poll()
    # Bounded verification: a finished parent alone is not cleanup proof.
    remaining = []
    remaining_gpu = 0.0
    for _ in range(10):
        table = proc_table()
        remaining = sorted(owned_pids(table, seen, process.pid))
        try:
            remaining_gpu = owned_gpu_memory_mb(set(seen), bool(case["gpu"])) or 0.0
        except (subprocess.SubprocessError, OSError) as exc:
            failure = failure or f"cleanup telemetry unknown: {exc}"
            break
        if not remaining and remaining_gpu == 0:
            break
        time.sleep(1)
    if remaining:
        failure = failure or f"owned descendants remained after parent exit: {remaining}"
        stop_owned(seen, process.pid)
        remaining = sorted(owned_pids(proc_table(), seen, process.pid))
    if remaining or remaining_gpu:
        failure = failure or f"cleanup unverified: pids={remaining}, gpu_mb={remaining_gpu}"
    if exit_code != 0:
        failure = failure or f"exit status {exit_code}"
    missing = [path for path in case["outputs"] if not Path(path).is_file()]
    if missing:
        failure = failure or f"missing outputs: {missing}"
    return {
        "id": case["id"], "method": case["method"], "input": case["input"],
        "input_sha256": case["input_sha256"], "command_sha256": command_hash,
        "native_pixels": int(case["native_width"]) * int(case["native_height"]),
        "native_megapixels": int(case["native_width"]) * int(case["native_height"]) / 1e6,
        "tissue_megapixels": int(case["tissue_pixels"]) / 1e6,
        "tissue_area_mm2": (int(case["tissue_pixels"]) * float(case["mpp_x"])
                            * float(case["mpp_y"]) / 1e6),
        "mpp_xy": [case["mpp_x"], case["mpp_y"]],
        "wall_seconds_cold": elapsed,
        "seconds_per_tissue_megapixel": elapsed / max(int(case["tissue_pixels"]) / 1e6, 1e-9),
        "peak_process_tree_rss_mb": peak_rss / 1e6,
        "peak_owned_gpu_memory_mb": peak_gpu if case["gpu"] else None,
        "sample_interval_seconds": sample_seconds, "samples": samples,
        "exit_code": exit_code, "cleanup_verified": not remaining and not remaining_gpu and not
        (failure and "cleanup telemetry" in failure),
        "status": "complete" if failure is None else "failed",
        "error": failure, "log": str(log_path), "outputs": case["outputs"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not Path("/proc/self/stat").exists():
        parser.error("This runner requires Linux /proc")
    if args.sample_seconds < 1.0 and args.allow_gpu:
        parser.error("GPU telemetry must not exceed 1 Hz")
    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get("cases", [])
    if not cases or len({c.get("id") for c in cases}) != len(cases):
        parser.error("cases must be nonempty with unique ids")
    output = {"manifest_sha256": digest(args.manifest), "hardware": os.uname().nodename,
              "records": [], "complete": False}
    if args.output.exists():
        parser.error("output already exists; use a new run directory")
    checkpoint_dir = args.output.with_suffix("")
    for case in cases:
        try:
            result = run_case(case, checkpoint_dir / "logs", args.allow_gpu,
                              args.sample_seconds)
        except BaseException as exc:
            result = {"id": case.get("id"), "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"}
        output["records"].append(result)
        atomic_json(args.output, output)
        if result["status"] != "complete":
            return 1
    output["complete"] = True
    atomic_json(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
