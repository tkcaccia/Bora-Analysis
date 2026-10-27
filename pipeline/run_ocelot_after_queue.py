#!/usr/bin/env python3
"""Wait for verified full context queue, then run the OCELOT CPU control once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from build_bora_case import sha256
from run_primary_cpu import atomic_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def live(pid: int, ticks: str) -> bool:
    path = Path(f"/proc/{pid}/stat")
    if not path.exists(): return False
    fields = path.read_text().split()
    return fields[21] == str(ticks) and fields[2] != "Z"


def run(args) -> dict:
    if args.output.exists():
        raise FileExistsError("Use a new versioned external output")
    launch = json.loads(args.gpu_launch.read_text())
    if sha256(args.gpu_queue) != launch["queue_sha256"]:
        raise ValueError("Frozen multiscale queue changed")
    queue = json.loads(args.gpu_queue.read_text())
    expected = [row["case_id"] for row in queue["cases"]]
    if len(queue["cases"]) != 1060 or any(x["split"] != "external_tumor_only" for x in queue["cases"][548:]):
        raise ValueError("OCELOT feature suffix changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    status = args.output.parent / "OCELOT_WAITER_STATE.json"
    if status.exists():
        raise FileExistsError("OCELOT waiter already launched")
    state = dict(format="bora-ocelot-cpu-waiter-v1", at_utc=now(), phase="wait_features",
                 multiscale_queue_sha256=launch["queue_sha256"], stopped=None,
                 no_gpu_execution=True, no_automatic_retry=True)
    atomic_json(status, state)
    deadline = time.monotonic() + args.deadline_hours * 3600
    try:
        while time.monotonic() < deadline:
            complete = args.gpu_scratch / "QUEUE_COMPLETE.json"
            if complete.is_file():
                proof = json.loads(complete.read_text())
                checkpoint = json.loads((args.gpu_scratch / "STATE.json").read_text())
                if (proof.get("cases") != 1060 or proof.get("queue_sha256") != launch["queue_sha256"]
                        or checkpoint.get("stopped") or checkpoint.get("active")
                        or checkpoint.get("completed") != expected):
                    raise ValueError("Multiscale completion proof differs")
                break
            checkpoint = json.loads((args.gpu_scratch / "STATE.json").read_text())
            if checkpoint.get("stopped"):
                raise RuntimeError("Multiscale GPU queue stopped; no external CPU scoring")
            if not live(launch["pid"], launch["startticks"]):
                raise RuntimeError("Multiscale GPU controller disappeared before completion")
            time.sleep(60)
        else:
            raise TimeoutError("Finite wait for complete context queue exceeded")
        while time.monotonic() < deadline:
            if (args.primary / "FIT_COMPLETE.json").is_file():
                break
            state_file = args.primary / "CONTROLLER_STATE.json"
            if state_file.is_file() and json.loads(state_file.read_text()).get("stopped"):
                raise RuntimeError("Primary CPU fit stopped")
            primary_launch = json.loads(args.primary_launch.read_text())
            if not live(primary_launch["pid"], primary_launch["startticks"]):
                raise RuntimeError("Primary CPU controller disappeared before model fit")
            time.sleep(60)
        else:
            raise TimeoutError("Finite wait for primary model exceeded")
        state["phase"] = "score_ocelot"; atomic_json(status, state)
        command = [sys.executable, str(Path(__file__).with_name("run_ocelot_external_cpu.py")),
                   "--root", str(args.root), "--primary", str(args.primary),
                   "--label-map", str(args.label_map), "--output", str(args.output)]
        with (args.output.parent / "ocelot-external.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=48 * 3600)
        state["phase"] = "complete"; atomic_json(status, state)
        return json.loads((args.output / "EXTERNAL_COMPLETE.json").read_text())
    except BaseException as exc:
        state["stopped"] = dict(at_utc=now(), reason=repr(exc), no_automatic_retry=True)
        atomic_json(status, state)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "primary", "primary-launch", "gpu-queue", "gpu-scratch", "gpu-launch", "label-map", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--deadline-hours", type=float, default=168)
    a = p.parse_args()
    if not 1 <= a.deadline_hours <= 168:
        raise ValueError("Finite wait must be 1..168 hours")
    print(json.dumps(run(a), indent=2))


if __name__ == "__main__":
    main()
