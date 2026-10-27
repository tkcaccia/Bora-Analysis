#!/usr/bin/env python3
"""Run the frozen CPU study after its sequential GPU feature receipts.

This controller never invokes a GPU worker. It waits on an existing immutable
multiscale queue and stops if that queue fails. CPU phases are individually
bounded and have no automatic retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from run_primary_cpu import atomic_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live(pid: int, ticks: str) -> bool:
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().split()[21] == str(ticks)
    except FileNotFoundError:
        return False


def wait_prefix(args, queue: dict, required: int, deadline: float) -> None:
    launch = json.loads(args.gpu_launch.read_text())
    if digest(args.gpu_queue) != launch["queue_sha256"]:
        raise ValueError("Multiscale queue changed after launch")
    expected = [c["case_id"] for c in queue["cases"][:required]]
    while time.monotonic() < deadline:
        state = json.loads((args.gpu_scratch / "STATE.json").read_text())
        if state.get("stopped"):
            raise RuntimeError("Feature queue stopped; CPU evaluation will not proceed")
        completed = state["completed"]
        if completed[:required] != expected[:len(completed[:required])]:
            raise ValueError("Feature checkpoint order or identity changed")
        if len(completed) >= required:
            for case_id in expected:
                output = args.root / "preprocessing/full-rerun-2026-10-27/multiscale-v1/features" / case_id
                if not (output / "FEATURES_COMPLETE.json").is_file() or not (output / "GPU_CLEANUP.json").is_file():
                    raise ValueError("Feature/GPU cleanup receipt missing for " + case_id)
            return
        if not live(int(launch["pid"]), str(launch["startticks"])):
            raise RuntimeError("Feature queue PID disappeared before required prefix completed")
        time.sleep(60)
    raise TimeoutError("Finite wait for GPU features exceeded")


def run_phase(args, phase: str, log: Path, timeout_hours: float) -> None:
    command = [sys.executable, str(Path(__file__).with_name("run_primary_cpu.py")), phase,
               "--root", str(args.root), "--output", str(args.output),
               "--label-map", str(args.label_map),
               "--case-timeout-seconds", "1800"]
    with log.open("x") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_hours * 3600)
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            raise TimeoutError(f"CPU phase {phase} exceeded finite deadline") from exc
    if code:
        raise RuntimeError(f"CPU phase {phase} exited {code}; see {log}")


def run(args) -> dict:
    queue = json.loads(args.gpu_queue.read_text())
    if (queue["total_cases"] != 1060 or len(queue["cases"]) != 1060
            or any(c["split"] != "train" for c in queue["cases"][:488])
            or any(c["split"] != "test" for c in queue["cases"][488:548])):
        raise ValueError("Frozen primary feature prefix differs")
    if args.check_only:
        launch = json.loads(args.gpu_launch.read_text())
        return dict(feature_queue_sha256=digest(args.gpu_queue), train_prefix=488,
                    test_prefix=548, cpu_output=str(args.output),
                    feature_controller_live=live(launch["pid"], launch["startticks"]))
    if args.output.exists():
        raise FileExistsError("Use a fresh CPU study output")
    args.output.mkdir(parents=True)
    state_path = args.output / "CONTROLLER_STATE.json"
    state = dict(format="bora-primary-cpu-controller-v1", created_utc=now(),
                 gpu_queue_sha256=digest(args.gpu_queue),
                 label_map_sha256=digest(args.label_map), phase="wait_train",
                 stopped=None)
    atomic_json(state_path, state)
    deadline = time.monotonic() + args.deadline_hours * 3600
    try:
        wait_prefix(args, queue, 488, deadline)
        state["phase"] = "prepare"; atomic_json(state_path, state)
        run_phase(args, "prepare", args.output / "train-prep.log", 48)
        state["phase"] = "fit"; atomic_json(state_path, state)
        run_phase(args, "fit", args.output / "model-fit.log", 2)
        state["phase"] = "wait_test"; atomic_json(state_path, state)
        wait_prefix(args, queue, 548, deadline)
        state["phase"] = "evaluate"; atomic_json(state_path, state)
        run_phase(args, "evaluate", args.output / "test-eval.log", 24)
        state["phase"] = "aggregate"; atomic_json(state_path, state)
        command = [sys.executable, str(Path(__file__).with_name("aggregate_primary_scores.py")),
                   "--root", str(args.root), "--run", str(args.output),
                   "--output", str(args.output / "primary-summary.json")]
        with (args.output / "aggregate.log").open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=3600)
        state["phase"] = "complete"; atomic_json(state_path, state)
        final = dict(format="bora-primary-full-rerun-complete-v1", at_utc=now(),
                     gpu_queue_sha256=digest(args.gpu_queue),
                     primary_summary_sha256=digest(args.output / "primary-summary.json"),
                     model_sha256=digest(args.output / "model.npz"),
                     test_images=60)
        atomic_json(args.output / "PRIMARY_COMPLETE.json", final)
        return final
    except BaseException as exc:
        state["stopped"] = dict(at_utc=now(), reason=repr(exc), no_automatic_retry=True)
        atomic_json(state_path, state)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gpu-queue", type=Path, required=True)
    p.add_argument("--gpu-scratch", type=Path, required=True)
    p.add_argument("--gpu-launch", type=Path, required=True)
    p.add_argument("--label-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-hours", type=float, default=120)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if not 1 <= a.deadline_hours <= 168:
        raise ValueError("Finite deadline must be 1..168 hours")
    print(json.dumps(run(a), indent=2), flush=True)


if __name__ == "__main__":
    main()
