#!/usr/bin/env python3
"""Fail-fast GPU-host queue for fixed Bora RGB context features.

The queue waits for the registered spot queue, uses one GPU case at a time,
and requires a two-check GPU cleanup receipt before advancing. It does not
submit scheduler jobs or retry a failed case.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_visium_queue import (atomic_json, gpu_pids, now, sha256, stage,
                              stop_session, verified_cleanup)


def wait_for_spots(scratch: Path, deadline: float) -> None:
    launch = json.loads((scratch.parent / "virchow2-spot-v1-launch.json").read_text())
    pid, ticks = int(launch["pid"]), str(launch["startticks"])
    while time.monotonic() < deadline:
        complete = scratch / "QUEUE_COMPLETE.json"
        if complete.is_file():
            proof = json.loads(complete.read_text())
            state = json.loads((scratch / "STATE.json").read_text())
            if (proof.get("units") != 19 or proof.get("cases") != 3 or
                    state.get("stopped") or state.get("active")):
                raise RuntimeError("Spot predecessor did not complete cleanly")
            return
        state_path = scratch / "STATE.json"
        if state_path.exists() and json.loads(state_path.read_text()).get("stopped"):
            raise RuntimeError("Spot predecessor stopped; no multiscale GPU case may start")
        try:
            actual = Path(f"/proc/{pid}/stat").read_text().split()[21]
        except FileNotFoundError as exc:
            raise RuntimeError("Spot predecessor disappeared without completion") from exc
        if actual != ticks:
            raise RuntimeError("Spot predecessor PID was replaced")
        time.sleep(60)
    raise TimeoutError("Waiting for spot predecessor exceeded finite deadline")


def base_dir(root: Path, case_id: str) -> Path:
    found = [root / "preprocessing" / f"virchow2-{version}" / "features" / case_id
             for version in ("v1", "v2", "v3", "v5")]
    complete = [p for p in found if (p / "FEATURES_COMPLETE.json").is_file()]
    if len(complete) != 1:
        raise ValueError(f"Expected one complete base feature source for {case_id}; found {len(complete)}")
    if not (complete[0] / "GPU_CLEANUP.json").is_file():
        raise ValueError("Base feature GPU cleanup is unverified: " + case_id)
    return complete[0]


def run(args: argparse.Namespace) -> dict:
    queue_path = args.queue.resolve()
    queue_sha = sha256(queue_path)
    queue = json.loads(queue_path.read_text())
    if queue.get("format") != "bora-virchow2-multiscale-queue-v1" or queue.get("total_cases") != 1060:
        raise ValueError("Expected frozen 1060-case multiscale queue")
    if not os.path.ismount(args.mount):
        raise RuntimeError("Configured data mount is absent")
    if not args.worker.is_file() or not args.python.is_file() or not args.model_dir.is_dir():
        raise FileNotFoundError("Worker, Python runtime or encoder checkpoint is absent")
    if args.check_only:
        return dict(queue_sha256=queue_sha, total_cases=len(queue["cases"]),
                    spot_complete=(args.spot_scratch / "QUEUE_COMPLETE.json").is_file(),
                    gpu_contexts=gpu_pids())
    scratch = args.scratch.resolve()
    if scratch.exists():
        raise FileExistsError("Use a fresh multiscale supervisor scratch directory")
    scratch.mkdir(parents=True)
    stage_dir = scratch / "staged"
    stage_dir.mkdir()
    state_path = scratch / "STATE.json"
    state = dict(created_utc=now(), queue_sha256=queue_sha,
                 worker_sha256=sha256(args.worker), completed=[], active=None, stopped=None)
    atomic_json(state_path, state)
    deadline = time.monotonic() + args.deadline_hours * 3600
    try:
        wait_for_spots(args.spot_scratch, deadline)
        if gpu_pids():
            raise RuntimeError("Another GPU context is present at multiscale admission")
        for item in queue["cases"]:
            if time.monotonic() >= deadline or gpu_pids():
                raise RuntimeError("Campaign deadline or foreign GPU context before next case")
            if sha256(queue_path) != queue_sha or sha256(args.worker) != state["worker_sha256"]:
                raise RuntimeError("Frozen queue or worker changed")
            case = item["case_id"]
            base = base_dir(args.mount, case)
            base_receipt = json.loads((base / "FEATURES_COMPLETE.json").read_text())
            source = Path(item["source_image"])
            expected_sha = json.loads((base / "INPUTS.json").read_text())["image_sha256"]
            staged = stage_dir / (case + source.suffix)
            stage(source, staged, expected_sha)
            try:
                output = args.mount / "preprocessing/full-rerun-2026-10-27/multiscale-v1/features" / case
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    raise FileExistsError("Existing case output requires a versioned recovery")
                cmd = [str(args.python), str(args.worker), "--case-id", case,
                       "--source", str(source), "--staged-source", str(staged),
                       "--base-dir", str(base), "--source-mpp-xy",
                       *map(str, item["source_mpp_xy"]), "--model-dir", str(args.model_dir),
                       "--output", str(output), "--max-seconds", str(args.unit_timeout_seconds - 120)]
                log = scratch / (case + ".log")
                with log.open("w") as stream:
                    proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                            start_new_session=True, stdin=subprocess.DEVNULL)
                    state["active"] = dict(case=case, pid=proc.pid,
                                           startticks=Path(f"/proc/{proc.pid}/stat").read_text().split()[21],
                                           log=str(log), base_receipt_sha256=sha256(base / "FEATURES_COMPLETE.json"))
                    atomic_json(state_path, state)
                    try:
                        code = proc.wait(timeout=args.unit_timeout_seconds)
                    except subprocess.TimeoutExpired:
                        stop_session(proc)
                        raise TimeoutError("Multiscale case timed out; no next case will launch")
                cleanup = verified_cleanup(proc.pid)
                if code != 0:
                    raise RuntimeError(f"Multiscale case {case} exited {code}; see {log}")
                marker = output / "FEATURES_COMPLETE.json"
                if not marker.is_file():
                    raise RuntimeError("Worker exited without completion receipt")
                result = json.loads(marker.read_text())
                if result["case_id"] != case or result["base_receipt_sha256"] != sha256(base / "FEATURES_COMPLETE.json"):
                    raise RuntimeError("Feature receipt belongs to different inputs")
                if result["source_sha256"] != expected_sha:
                    raise RuntimeError("Source image changed during feature extraction")
                atomic_json(output / "GPU_CLEANUP.json", cleanup)
                state["completed"].append(case)
                state["active"] = None
                atomic_json(state_path, state)
                print(json.dumps(dict(completed=case, of=len(queue["cases"]))), flush=True)
            finally:
                if staged.exists() and not gpu_pids():
                    staged.unlink()
        if len(state["completed"]) != len(queue["cases"]):
            raise RuntimeError("Multiscale queue ended before all cases completed")
        final = dict(completed_utc=now(), queue_sha256=queue_sha,
                     cases=len(state["completed"]), tile_images_retained=False)
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
    p.add_argument("--spot-scratch", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--unit-timeout-seconds", type=int, default=1800)
    p.add_argument("--deadline-hours", type=float, default=120)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if not 300 <= a.unit_timeout_seconds <= 3600 or not 1 <= a.deadline_hours <= 168:
        raise ValueError("Finite queue limits outside audited range")
    print(json.dumps(run(a), indent=2), flush=True)


if __name__ == "__main__":
    main()
