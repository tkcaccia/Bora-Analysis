#!/usr/bin/env python3
"""Run calibrated Visium spot encoding after the full-image queue completes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_visium_queue import (atomic_json, gpu_pids, now,
                              sha256, stage, stop_session, verified_cleanup)


def wait_for_visium(scratch: Path, expected_units: int, deadline: float) -> None:
    launch = json.loads((scratch.parent / "virchow2-visium-v1-launch.json").read_text())
    pid, ticks = int(launch["pid"]), str(launch["startticks"])
    while time.monotonic() < deadline:
        complete = scratch / "QUEUE_COMPLETE.json"
        if complete.is_file():
            proof = json.loads(complete.read_text())
            state = json.loads((scratch / "STATE.json").read_text())
            if (proof.get("units") != expected_units or proof.get("cases") != 3 or
                    state.get("stopped") or state.get("active")):
                raise RuntimeError("Visium predecessor did not finish its frozen scope cleanly")
            return
        state_path = scratch / "STATE.json"
        if state_path.exists() and json.loads(state_path.read_text()).get("stopped"):
            raise RuntimeError("Visium predecessor stopped; spot queue cannot start")
        try:
            actual = Path(f"/proc/{pid}/stat").read_text().split()[21]
        except FileNotFoundError as exc:
            raise RuntimeError("Visium predecessor disappeared without completion") from exc
        if actual != ticks:
            raise RuntimeError("Visium predecessor PID was replaced")
        time.sleep(60)
    raise TimeoutError("Waiting for Visium predecessor exceeded the finite deadline")


def run(args: argparse.Namespace) -> dict:
    queue_path = args.queue.resolve()
    queue_sha = sha256(queue_path)
    queue = json.loads(queue_path.read_text())
    if queue.get("format") != "bora-virchow2-spot-queue-v1" or len(queue["cases"]) != 3:
        raise ValueError("Expected the three-case frozen spot queue")
    if not os.path.ismount(args.mount):
        raise RuntimeError("Configured data mount is absent")
    if not args.worker.is_file() or not args.python.is_file() or not args.model_dir.is_dir():
        raise FileNotFoundError("Encoder runtime, worker or model is absent")
    if args.check_only:
        return dict(queue_sha256=queue_sha, total_units=queue["total_units"],
                    visium_complete=(args.visium_scratch / "QUEUE_COMPLETE.json").is_file(),
                    gpu_contexts=gpu_pids())
    scratch = args.scratch.resolve()
    if scratch.exists():
        raise FileExistsError("Use a fresh spot supervisor scratch directory")
    scratch.mkdir(parents=True)
    stage_dir = scratch / "staged"
    stage_dir.mkdir()
    state_path = scratch / "STATE.json"
    state = dict(created_utc=now(), queue_sha256=queue_sha,
                 worker_sha256=sha256(args.worker), completed=[], active=None, stopped=None)
    atomic_json(state_path, state)
    deadline = time.monotonic() + args.deadline_hours * 3600
    try:
        wait_for_visium(args.visium_scratch, args.visium_units, deadline)
        if gpu_pids():
            raise RuntimeError("Another GPU context is present at spot queue admission")
        for case in queue["cases"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("Spot queue exceeded its finite campaign deadline")
            if sha256(queue_path) != queue_sha or sha256(args.worker) != state["worker_sha256"]:
                raise RuntimeError("Frozen queue or worker changed")
            source = Path(case["image"])
            staged = stage_dir / (case["case"] + source.suffix)
            stage(source, staged, case["image_sha256"])
            try:
                for index, unit in enumerate(case["units"]):
                    if time.monotonic() >= deadline or gpu_pids():
                        raise RuntimeError("Deadline or foreign GPU context before next spot unit")
                    unit_id = f"{case['case']}/unit-{index:03d}"
                    output = args.mount / "preprocessing/virchow2-spot-v1/features" / unit_id
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
                            raise TimeoutError("Spot unit timed out; no next unit will launch")
                    cleanup = verified_cleanup(proc.pid)
                    if code != 0:
                        raise RuntimeError(f"Spot unit {unit_id} exited {code}; see {log}")
                    feature = output / "FEATURES_COMPLETE.json"
                    if not feature.is_file():
                        raise RuntimeError("Spot unit exited without its feature completion receipt")
                    result = json.loads(feature.read_text())
                    if (result["first_tile"], result["last_tile"]) != (
                            unit["first_tile"], unit["last_tile"]):
                        raise RuntimeError("Feature receipt belongs to a different spot unit")
                    atomic_json(output / "GPU_CLEANUP.json", cleanup)
                    state["completed"].append(unit_id)
                    state["active"] = None
                    atomic_json(state_path, state)
                    print(json.dumps(dict(completed=unit_id, of=queue["total_units"])), flush=True)
            finally:
                if staged.exists() and not gpu_pids():
                    staged.unlink()
        if len(state["completed"]) != queue["total_units"]:
            raise RuntimeError("Spot queue ended before all units completed")
        final = dict(completed_utc=now(), queue_sha256=queue_sha,
                     units=len(state["completed"]), cases=len(queue["cases"]),
                     tile_images_retained=False)
        atomic_json(scratch / "QUEUE_COMPLETE.json", final)
        return final
    except BaseException as exc:
        state["stopped"] = dict(at=now(), reason=repr(exc), no_automatic_retry=True)
        atomic_json(state_path, state)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--visium-scratch", type=Path, required=True)
    p.add_argument("--visium-units", type=int, default=32)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--unit-timeout-seconds", type=int, default=3600)
    p.add_argument("--deadline-hours", type=float, default=120)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if not 300 <= args.unit_timeout_seconds <= 7200 or not 1 <= args.deadline_hours <= 168:
        raise ValueError("Unit or campaign deadline outside allowed bounds")
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
