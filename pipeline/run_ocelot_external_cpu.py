#!/usr/bin/env python3
"""Score OCELOT's expert tumor positive control after the frozen primary fit.

OCELOT's broad negative label is automatic and cannot support multiclass IoU.
This CPU-only command requires completed context features and the exact primary
model; it reports tumor recall and tumor calls on the broad negative region.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from build_bora_case import sha256
from run_primary_cpu import atomic_json, checked_command


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(args) -> dict:
    root, primary, output = args.root.resolve(), args.primary.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new versioned external output")
    if not (primary / "FIT_COMPLETE.json").is_file():
        raise ValueError("Frozen primary classifier has not fitted")
    fit = json.loads((primary / "FIT_COMPLETE.json").read_text())
    model = primary / "model.npz"
    if sha256(model) != fit["model_sha256"]:
        raise ValueError("Primary model changed")
    split = json.loads((root / "preprocessing/full-rerun-2026-10-27/split-v1.json").read_text())
    rows = sorted((x for x in split["assignments"] if x["split"] == "external_tumor_only"),
                  key=lambda x: x["case_id"])
    if len(rows) != 512 or {r["dataset"] for r in rows} != {"ocelot"}:
        raise ValueError("Frozen OCELOT cohort changed")
    script_dir = Path(__file__).resolve().parent
    output.mkdir(parents=True)
    cases, results, logs = (output / name for name in ("cases", "results", "logs"))
    for path in (cases, results, logs): path.mkdir()
    state_file = output / "STATE.json"
    state = dict(format="bora-ocelot-external-cpu-v1", created_utc=now(),
                 expected=512, completed=[], active=None, stopped=None,
                 model_sha256=fit["model_sha256"], no_gpu_execution=True)
    atomic_json(state_file, state)
    import sys
    try:
        for row in rows:
            case_id = row["case_id"]
            feature = root / "preprocessing/full-rerun-2026-10-27/multiscale-v1/features" / case_id
            if not (feature / "FEATURES_COMPLETE.json").is_file() or not (feature / "GPU_CLEANUP.json").is_file():
                raise ValueError("Incomplete context feature or cleanup receipt: " + case_id)
            state["active"] = case_id; atomic_json(state_file, state)
            case = cases / case_id
            checked_command([sys.executable, str(script_dir / "build_bora_case.py"),
                "--root", str(root), "--case-id", case_id,
                "--label-map", str(args.label_map), "--output", str(case)],
                logs / (case_id + "-build.log"), args.case_timeout_seconds)
            result = results / case_id
            checked_command([sys.executable, "-m", "bora.cli", "refine", str(case / "case.json"),
                "--model", str(model), "--output", str(result)],
                logs / (case_id + "-refine.log"), args.case_timeout_seconds)
            checked_command([sys.executable, str(script_dir / "score_bora_case.py"),
                "--case", str(case), "--result", str(result),
                "--output", str(result / "score.json")],
                logs / (case_id + "-score.log"), args.case_timeout_seconds)
            score = json.loads((result / "score.json").read_text())
            if score["endpoint"] != "expert_tumor_recall_plus_automatic_broad_negative_control":
                raise ValueError("OCELOT endpoint changed")
            state["completed"].append(case_id); state["active"] = None
            atomic_json(state_file, state)
        values = [json.loads((results / r["case_id"] / "score.json").read_text()) for r in rows]
        recalls = np.asarray([v["final"]["expert_tumor_recall"] for v in values], float)
        negatives = np.asarray([v["final"]["weak_negative_tumor_fraction"] for v in values], float)
        final = dict(format="bora-ocelot-secondary-summary-v1", cases=512,
            model_sha256=fit["model_sha256"], endpoint=values[0]["endpoint"],
            mean_expert_tumor_recall=float(recalls.mean()),
            mean_automatic_broad_negative_tumor_fraction=float(negatives.mean()),
            median_expert_tumor_recall=float(np.median(recalls)),
            median_automatic_broad_negative_tumor_fraction=float(np.median(negatives)),
            multiclass_iou_applicable=False, independent_heldout_claim=False,
            per_case_score_sha256={r["case_id"]:sha256(results/r["case_id"]/"score.json") for r in rows})
        atomic_json(output / "summary.json", final)
        atomic_json(output / "EXTERNAL_COMPLETE.json", dict(at_utc=now(),
            summary_sha256=sha256(output / "summary.json"), cases=512))
        return final
    except BaseException as exc:
        state["stopped"] = dict(at_utc=now(), reason=repr(exc), no_automatic_retry=True)
        atomic_json(state_file, state)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label-map", type=Path, required=True)
    p.add_argument("--case-timeout-seconds", type=int, default=1800)
    a = p.parse_args()
    if not 300 <= a.case_timeout_seconds <= 7200:
        raise ValueError("Finite per-case deadline required")
    result = run(a)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
