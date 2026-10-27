#!/usr/bin/env python3
"""Checkpointed CPU stages for the frozen BCSS/TIGER/IGNITE benchmark.

No GPU or scheduler command is invoked. Run each phase only after required
Virchow2 receipts exist. A failed command stops the phase without a retry.
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
from datetime import datetime, timezone

from build_bora_case import sha256


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def checked_command(parts: list[str], log: Path, seconds: int) -> None:
    with log.open("x") as stream:
        proc = subprocess.Popen(parts, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            raise TimeoutError("CPU case deadline exceeded: " + str(log)) from exc
    if code != 0:
        raise RuntimeError(f"CPU case exited {code}; see {log}")


def frozen_rows(root: Path, phase: str) -> list[dict]:
    base = root / "preprocessing/full-rerun-2026-10-27"
    split_path = base / "split-v1.json"
    split = json.loads(split_path.read_text())
    exclusions = json.loads((base / "preflight-exclusions-v1.json").read_text())
    if exclusions["split_sha256"] != sha256(split_path):
        raise ValueError("Exclusions do not match frozen split")
    excluded = {x["case_id"] for x in exclusions["excluded"]}
    if phase == "prepare":
        rows = [x for x in split["assignments"] if x["split"] == "train"
                and x["case_id"] not in excluded]
        if len(rows) != 476:
            raise ValueError("Training cohort differs")
    elif phase == "evaluate":
        rows = [x for x in split["assignments"] if x["split"] == "test"]
        if len(rows) != 60:
            raise ValueError("Test cohort differs")
    else:
        rows = []
    return sorted(rows, key=lambda r: (r["dataset"], r["case_id"]))


def run(args: argparse.Namespace) -> dict:
    root, output = args.root.resolve(), args.output.resolve()
    label_map = args.label_map.resolve()
    script_dir = Path(__file__).resolve().parent
    if not label_map.is_file():
        raise FileNotFoundError(label_map)
    if args.phase in ("fit", "evaluate") and not (output / "model.npz").is_file():
        if args.phase == "evaluate":
            raise FileNotFoundError("Frozen training model is missing")
    if args.phase == "prepare":
        rows = frozen_rows(root, args.phase)
        parent = output / "prepared"
        case_parent = output / "cases"
        parent.mkdir(parents=True, exist_ok=True)
        case_parent.mkdir(parents=True, exist_ok=True)
        logs = output / "logs"; logs.mkdir(exist_ok=True)
        state_file = output / "TRAIN_PREP_STATE.json"
        if state_file.exists():
            raise FileExistsError("Use a fresh output directory or an audited versioned recovery")
        state = dict(format="bora-primary-train-prep-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                     phase="prepare", expected_cases=len(rows), completed=[], active=None,
                     stopped=None, bora_release_version="0.2.1")
        atomic_json(state_file, state)
        try:
            for row in rows:
                case_id = row["case_id"]
                state["active"] = case_id; atomic_json(state_file, state)
                case = case_parent / case_id
                checked_command([sys.executable, str(script_dir / "build_bora_case.py"),
                    "--root", str(root), "--case-id", case_id,
                    "--label-map", str(label_map), "--output", str(case)],
                    logs / (case_id + "-build.log"), args.case_timeout_seconds)
                checked_command([sys.executable, "-m", "bora.cli", "prepare",
                    str(case / "case.json"), "--output", str(parent / case_id)],
                    logs / (case_id + "-prepare.log"), args.case_timeout_seconds)
                state["completed"].append(case_id); state["active"] = None
                atomic_json(state_file, state)
            atomic_json(output / "TRAIN_PREP_COMPLETE.json", dict(cases=len(rows),
                        state_sha256=sha256(state_file), completed_utc=datetime.now(timezone.utc).isoformat()))
            return dict(phase="prepare", completed=len(rows))
        except BaseException as exc:
            state["stopped"] = dict(reason=repr(exc), at_utc=datetime.now(timezone.utc).isoformat(),
                                    no_automatic_retry=True)
            atomic_json(state_file, state)
            raise
    if args.phase == "fit":
        if not (output / "TRAIN_PREP_COMPLETE.json").is_file():
            raise ValueError("Training preparation is incomplete")
        manifest = output / "training.json"
        checked_command([sys.executable, str(script_dir / "build_training_manifest.py"),
            "--root", str(root), "--cases", str(output / "cases"),
            "--prepared", str(output / "prepared"), "--output", str(manifest)],
            output / "fit-manifest.log", args.case_timeout_seconds)
        checked_command([sys.executable, "-m", "bora.cli", "train",
            str(manifest), "--output", str(output / "model.npz")],
            output / "fit.log", args.case_timeout_seconds)
        atomic_json(output / "FIT_COMPLETE.json", dict(model_sha256=sha256(output / "model.npz"),
                    manifest_sha256=sha256(manifest), at_utc=datetime.now(timezone.utc).isoformat()))
        return dict(phase="fit", model_sha256=sha256(output / "model.npz"))
    if not (output / "FIT_COMPLETE.json").is_file():
        raise ValueError("Classifier fit is incomplete")
    if sha256(output / "model.npz") != json.loads((output / "FIT_COMPLETE.json").read_text())["model_sha256"]:
        raise ValueError("Classifier artifact changed")
    rows = frozen_rows(root, "evaluate")
    results = output / "test-results"; results.mkdir(parents=True, exist_ok=True)
    cases = output / "cases"; cases.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"; logs.mkdir(exist_ok=True)
    state_file = output / "TEST_STATE.json"
    if state_file.exists():
        raise FileExistsError("Use a fresh evaluation output or audited recovery")
    state = dict(format="bora-primary-test-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                 expected_cases=len(rows), completed=[], active=None, stopped=None,
                 model_sha256=sha256(output / "model.npz"))
    atomic_json(state_file, state)
    try:
        for row in rows:
            case_id = row["case_id"]
            state["active"] = case_id; atomic_json(state_file, state)
            case = cases / case_id
            checked_command([sys.executable, str(script_dir / "build_bora_case.py"),
                "--root", str(root), "--case-id", case_id,
                "--label-map", str(label_map), "--output", str(case)],
                logs / (case_id + "-build.log"), args.case_timeout_seconds)
            result = results / case_id
            checked_command([sys.executable, "-m", "bora.cli", "refine",
                str(case / "case.json"), "--model", str(output / "model.npz"),
                "--output", str(result)],
                logs / (case_id + "-refine.log"), args.case_timeout_seconds)
            checked_command([sys.executable, str(script_dir / "score_bora_case.py"),
                "--case", str(case), "--result", str(result),
                "--output", str(result / "score.json")],
                logs / (case_id + "-score.log"), args.case_timeout_seconds)
            state["completed"].append(case_id); state["active"] = None
            atomic_json(state_file, state)
        atomic_json(output / "TEST_COMPLETE.json", dict(cases=len(rows),
                    state_sha256=sha256(state_file), completed_utc=datetime.now(timezone.utc).isoformat()))
        return dict(phase="evaluate", completed=len(rows))
    except BaseException as exc:
        state["stopped"] = dict(reason=repr(exc), at_utc=datetime.now(timezone.utc).isoformat(),
                                no_automatic_retry=True)
        atomic_json(state_file, state)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "fit", "evaluate"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label-map", type=Path, required=True)
    p.add_argument("--case-timeout-seconds", type=int, default=1800)
    args = p.parse_args()
    if not 300 <= args.case_timeout_seconds <= 7200:
        raise ValueError("Per-case CPU deadline must be finite, 300..7200 seconds")
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
