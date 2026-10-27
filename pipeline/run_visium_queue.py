#!/usr/bin/env python3
"""Fail-fast GPU-host Visium encoder queue, held until WSI encoding completes.

This launcher runs on a configured GPU host. It does not submit scheduler jobs.
Each 1024-tile unit has its own result and GPU cleanup receipt. Any failed
unit stops the entire queue; partial output is preserved for investigation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def gpu_pids() -> list[int]:
    proc = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                           "--format=csv,noheader"], text=True,
                          capture_output=True, timeout=15, check=True)
    return [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


def session_members(sid: int) -> list[int]:
    members = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat.read_text()
            fields = text[text.rfind(")") + 2:].split()
            if int(fields[3]) == sid and fields[0] != "Z":
                members.append(int(stat.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def verified_cleanup(sid: int) -> dict:
    checks = []
    for _ in range(2):
        checks.append(dict(session_members=session_members(sid), gpu_pids=gpu_pids()))
        time.sleep(2)
    result = dict(at=now(), unit_session=sid, checks=checks,
                  owned_session_absent=all(not c["session_members"] for c in checks),
                  gpu_contexts_absent=all(not c["gpu_pids"] for c in checks))
    if not result["owned_session_absent"] or not result["gpu_contexts_absent"]:
        raise RuntimeError(f"GPU cleanup uncertain after session {sid}: {result}")
    return result


def stop_session(process: subprocess.Popen, grace: int = 20) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def wait_for_wsi(wsi_scratch: Path, expected_units: int, deadline: float) -> None:
    launch = json.loads((wsi_scratch.parent / "virchow2-wsi-v1-launch.json").read_text())
    pid, ticks = int(launch["pid"]), str(launch["startticks"])
    while time.monotonic() < deadline:
        complete = wsi_scratch / "QUEUE_COMPLETE.json"
        if complete.is_file():
            proof = json.loads(complete.read_text())
            if proof.get("units") != expected_units or proof.get("cases") != 5:
                raise RuntimeError("WSI predecessor completion differs from its frozen scope")
            state = json.loads((wsi_scratch / "STATE.json").read_text())
            if state.get("stopped") or state.get("active"):
                raise RuntimeError("WSI predecessor did not end cleanly")
            return
        state_path = wsi_scratch / "STATE.json"
        if state_path.exists() and json.loads(state_path.read_text()).get("stopped"):
            raise RuntimeError("WSI predecessor stopped; Visium queue cannot start")
        try:
            actual = Path(f"/proc/{pid}/stat").read_text().split()[21]
        except FileNotFoundError as exc:
            raise RuntimeError("WSI predecessor disappeared without completion") from exc
        if actual != ticks:
            raise RuntimeError("WSI predecessor PID was replaced")
        time.sleep(60)
    raise TimeoutError("Waiting for WSI predecessor exceeded the finite deadline")

def stage(source: Path, destination: Path, expected_sha: str) -> None:
    if destination.exists():
        raise FileExistsError("Staging image already exists; inspect before a new queue")
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + 80 * (1 << 30):
        raise OSError("Staging would violate the 80 GiB local disk reserve")
    h = hashlib.sha256()
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        for block in iter(lambda: input_file.read(8 << 20), b""):
            output_file.write(block)
            h.update(block)
        output_file.flush()
        os.fsync(output_file.fileno())
    if h.hexdigest() != expected_sha:
        raise ValueError("Staged WSI differs from its verified source checksum")


def run(args: argparse.Namespace) -> dict:
    queue_path = args.queue.resolve()
    queue_sha = sha256(queue_path)
    queue = json.loads(queue_path.read_text())
    if queue.get("format") != "bora-virchow2-visium-queue-v1" or len(queue["cases"]) != 3:
        raise ValueError("Expected the three-case frozen Visium queue")
    if not os.path.ismount(args.mount):
        raise RuntimeError("Configured data mount is absent")
    if not args.worker.is_file() or not args.python.is_file() or not args.model_dir.is_dir():
        raise FileNotFoundError("Encoder runtime, worker or model is absent")
    if args.check_only:
        return dict(queue_sha256=queue_sha, total_units=queue["total_units"],
                    wsi_complete=(args.wsi_scratch / "QUEUE_COMPLETE.json").is_file(),
                    gpu_contexts=gpu_pids())
    scratch = args.scratch.resolve()
    if scratch.exists():
        raise FileExistsError("Use a fresh WSI supervisor scratch directory")
    scratch.mkdir(parents=True)
    stage_dir = scratch / "staged"
    stage_dir.mkdir()
    state_path = scratch / "STATE.json"
    state = dict(created_utc=now(), queue_sha256=queue_sha,
                 worker_sha256=sha256(args.worker), completed=[], active=None,
                 stopped=None)
    atomic_json(state_path, state)
    deadline = time.monotonic() + args.deadline_hours * 3600
    try:
        wait_for_wsi(args.wsi_scratch, args.wsi_units, deadline)
        if gpu_pids():
            raise RuntimeError("Another GPU context is present at WSI queue admission")
        for case in queue["cases"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("Visium queue exceeded its finite campaign deadline")
            if sha256(queue_path) != queue_sha or sha256(args.worker) != state["worker_sha256"]:
                raise RuntimeError("Frozen queue or worker changed")
            source = Path(case["image"])
            staged = stage_dir / (case["case"] + source.suffix)
            stage(source, staged, case["image_sha256"])
            try:
                for index, unit in enumerate(case["units"]):
                    if time.monotonic() >= deadline or gpu_pids():
                        raise RuntimeError("Deadline or foreign GPU context before next Visium unit")
                    unit_id = f"{case['case']}/unit-{index:03d}"
                    output = args.mount / "preprocessing/virchow2-visium-v1/features" / unit_id
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if output.exists():
                        raise FileExistsError("Existing unit output requires a versioned recovery")
                    cmd = [str(args.python), str(args.worker),
                           "--image", str(staged), "--source-image", str(source),
                           "--plan", case["plan_dir"], "--model-dir", str(args.model_dir),
                           "--output", str(output), "--image-sha256", case["image_sha256"],
                           "--first-tile", str(unit["first_tile"]),
                           "--last-tile", str(unit["last_tile"]),
                           "--max-unit-tiles", str(queue["unit_tiles"]),
                           "--max-seconds", str(args.unit_timeout_seconds - 120)]
                    log = scratch / f"{case['case']}-unit-{index:03d}.log"
                    with log.open("w") as log_file:
                        proc = subprocess.Popen(cmd, stdout=log_file,
                                                stderr=subprocess.STDOUT,
                                                start_new_session=True)
                        state["active"] = dict(unit=unit_id, pid=proc.pid,
                                               startticks=Path(f"/proc/{proc.pid}/stat").read_text().split()[21],
                                               log=str(log))
                        atomic_json(state_path, state)
                        try:
                            code = proc.wait(timeout=args.unit_timeout_seconds)
                        except subprocess.TimeoutExpired:
                            stop_session(proc)
                            raise TimeoutError("Visium unit timed out; no next unit will launch")
                    cleanup = verified_cleanup(proc.pid)
                    if code != 0:
                        raise RuntimeError(f"Visium unit {unit_id} exited {code}; see {log}")
                    feature = output / "FEATURES_COMPLETE.json"
                    if not feature.is_file():
                        raise RuntimeError("Visium unit exited without its feature completion receipt")
                    result = json.loads(feature.read_text())
                    if (result["first_tile"], result["last_tile"]) != (
                            unit["first_tile"], unit["last_tile"]):
                        raise RuntimeError("Feature receipt belongs to a different tile unit")
                    atomic_json(output / "GPU_CLEANUP.json", cleanup)
                    state["completed"].append(unit_id)
                    state["active"] = None
                    atomic_json(state_path, state)
                    print(json.dumps(dict(completed=unit_id,
                                          of=queue["total_units"])), flush=True)
            finally:
                if staged.exists() and not gpu_pids():
                    staged.unlink()
        if len(state["completed"]) != queue["total_units"]:
            raise RuntimeError("Visium queue ended before all units completed")
        final = dict(completed_utc=now(), queue_sha256=queue_sha,
                     units=len(state["completed"]), cases=len(queue["cases"]),
                     tile_images_retained=False)
        atomic_json(scratch / "QUEUE_COMPLETE.json", final)
        return final
    except BaseException as exc:
        state["stopped"] = dict(at=now(), reason=repr(exc),
                                no_automatic_retry=True)
        atomic_json(state_path, state)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--wsi-scratch", type=Path, required=True)
    p.add_argument("--wsi-units", type=int, default=54)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--unit-timeout-seconds", type=int, default=3600)
    p.add_argument("--deadline-hours", type=float, default=72)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if not 300 <= args.unit_timeout_seconds <= 7200 or not 1 <= args.deadline_hours <= 168:
        raise ValueError("Unit or campaign deadline outside allowed bounds")
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
