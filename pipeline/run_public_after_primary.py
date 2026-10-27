#!/usr/bin/env python3
"""Generate public previews and a manuscript Results draft after primary completion.

This CPU-only waiter cannot start an encoder, Bora inference or a GPU queue.
It stops if the immutable primary controller fails or disappears unexpectedly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from run_primary_cpu import atomic_json
from build_bora_case import sha256


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def live(pid: int, ticks: str) -> bool:
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return False
    fields = path.read_text().split()
    return fields[21] == str(ticks) and fields[2] != "Z"


def run(args) -> dict:
    launch = json.loads(args.primary_launch.read_text())
    if not launch.get("no_gpu_descendants") or not launch.get("no_automatic_retry"):
        raise ValueError("Primary launch receipt lacks CPU-only fail-stop declaration")
    if args.output.exists():
        raise FileExistsError("Use a fresh reporting output")
    args.output.mkdir(parents=True)
    state_file = args.output / "STATE.json"
    state = dict(format="bora-public-after-primary-v1", created_utc=now(),
                 primary_pid=launch["pid"], primary_startticks=launch["startticks"],
                 phase="wait_primary", stopped=None, no_gpu_execution=True)
    atomic_json(state_file, state)
    deadline = time.monotonic() + args.deadline_hours * 3600
    try:
        while time.monotonic() < deadline:
            receipt = args.primary / "PRIMARY_COMPLETE.json"
            if receipt.is_file():
                done = json.loads(receipt.read_text())
                if done.get("test_images") != 60 or sha256(args.primary / "primary-summary.json") != done["primary_summary_sha256"]:
                    raise ValueError("Primary completion receipt or summary differs")
                break
            controller = args.primary / "CONTROLLER_STATE.json"
            if controller.is_file() and json.loads(controller.read_text()).get("stopped"):
                raise RuntimeError("Primary CPU controller stopped without complete results")
            if not live(launch["pid"], launch["startticks"]):
                raise RuntimeError("Primary CPU controller disappeared before completion")
            time.sleep(60)
        else:
            raise TimeoutError("Finite wait for primary results exceeded")
        scripts = Path(__file__).resolve().parent
        state["phase"] = "viewer"; atomic_json(state_file, state)
        viewer = args.output / "viewer"
        with (args.output / "viewer.log").open("x") as log:
            subprocess.run([sys.executable, str(scripts / "build_full_rerun_viewer.py"),
                "--root", str(args.root), "--run", str(args.primary), "--output", str(viewer)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=4 * 3600)
        if json.loads((viewer / "BUILD_COMPLETE.json").read_text())["cases"] != 60:
            raise ValueError("Viewer did not contain every primary case")
        state["phase"] = "manuscript_results"; atomic_json(state_file, state)
        manuscript_script = scripts / "build_full_rerun_results.py"
        results = args.output / "manuscript-results.md"
        with (args.output / "manuscript-results.log").open("x") as log:
            subprocess.run([sys.executable, str(manuscript_script),
                "--root", str(args.root), "--run", str(args.primary), "--output", str(results)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        state["phase"] = "complete"; atomic_json(state_file, state)
        complete = dict(format="bora-public-cpu-preparation-complete-v1", at_utc=now(),
            primary_receipt_sha256=sha256(args.primary / "PRIMARY_COMPLETE.json"),
            viewer_receipt_sha256=sha256(viewer / "BUILD_COMPLETE.json"),
            manuscript_results_sha256=sha256(results),
            publication_requires_visual_and_scientific_review=True)
        atomic_json(args.output / "REPORTS_COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        state["stopped"] = dict(at_utc=now(), reason=repr(exc), no_automatic_retry=True)
        atomic_json(state_file, state)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--primary-launch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-hours", type=float, default=120)
    a = p.parse_args()
    if not 1 <= a.deadline_hours <= 168:
        raise ValueError("Finite deadline must be 1..168 hours")
    print(json.dumps(run(a), indent=2))


if __name__ == "__main__":
    main()
