#!/usr/bin/env python3
"""Summarize recorded Bora runtime on the complete 60-field cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_bora_case import sha256


def stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if a.shape != (60,) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected 60 finite nonnegative timing values")
    return dict(mean_seconds=float(a.mean()), median_seconds=float(np.median(a)),
                p95_seconds=float(np.quantile(a, .95)), maximum_seconds=float(a.max()))


def summarize(root: Path, run: Path) -> dict:
    complete = json.loads((run / "PRIMARY_COMPLETE.json").read_text())
    summary_file = run / "primary-summary.json"
    if sha256(summary_file) != complete["primary_summary_sha256"]:
        raise ValueError("Primary summary changed")
    summary = json.loads(summary_file.read_text())
    if summary["fields"] != 60 or len(summary["score_sha256"]) != 60:
        raise ValueError("Incomplete evaluation")
    fields = ("pipeline_wall_seconds_excluding_model_load", "classifier_seconds",
              "export_seconds")
    measured = {name:[] for name in fields}
    recorded_substages = {}
    for case_id, score_sha in summary["score_sha256"].items():
        result = run / "test-results" / case_id
        score_file = result / "score.json"
        if sha256(score_file) != score_sha:
            raise ValueError("Case score changed: " + case_id)
        score = json.loads(score_file.read_text())
        report_file = result / "report.json"
        if sha256(report_file) != score["result_report_sha256"]:
            raise ValueError("Case report changed: " + case_id)
        report = json.loads(report_file.read_text())
        for name in fields:
            measured[name].append(float(report[name]))
        for name, value in report.get("timing_seconds", {}).items():
            recorded_substages.setdefault(name, []).append(float(value))
    return dict(format="bora-primary-runtime-summary-v1", fields=60,
                model_sha256=summary["model_sha256"],
                timings={name:stats(values) for name, values in measured.items()},
                recorded_substages={name:stats(values) for name, values in recorded_substages.items()
                                    if len(values)==60},
                excludes_virchow2_extraction=True,
                excludes_cold_model_load=True,
                timing_scope="released Bora CLI report, no upstream image/feature preparation")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = summarize(args.root.resolve(), args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(value["timings"], indent=2))


if __name__ == "__main__":
    main()
