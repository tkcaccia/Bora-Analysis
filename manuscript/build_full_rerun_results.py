#!/usr/bin/env python3
"""Draft CMPB Results and score tables only from complete frozen receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from build_bora_case import sha256


def f(value: float | None) -> str:
    return "not estimable" if value is None else f"{value:.3f}"


def build(root: Path, run: Path) -> str:
    done = json.loads((run / "PRIMARY_COMPLETE.json").read_text())
    summary_file = run / "primary-summary.json"
    if sha256(summary_file) != done["primary_summary_sha256"]:
        raise ValueError("Primary summary changed after completion")
    summary = json.loads(summary_file.read_text())
    split = root / "preprocessing/full-rerun-2026-10-27/split-v1.json"
    if sha256(split) != summary["split_sha256"] or summary["fields"] != 60:
        raise ValueError("Frozen split or complete cohort differs")
    if len(summary["score_sha256"]) != 60:
        raise ValueError("Incomplete per-case score manifest")
    for case_id, digest in summary["score_sha256"].items():
        if sha256(run / "test-results" / case_id / "score.json") != digest:
            raise ValueError("Missing or changed per-case score: " + case_id)
    overall = summary["overall"]
    lines = ["# Results from the completed calibrated Bora rerun", "",
        "This text is generated from verified outputs and requires scientific and editorial review before insertion into the Word article. The 60 fields are a reused development evaluation, not an untouched independent test.", "",
        "## Primary multiclass tissue benchmark", "",
        "| Dataset | Evaluated fields | Mean final macro IoU | Mean expert-decoded coarse IoU | Mean paired change | Whole-group accuracy |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in ("bcss", "tiger", "ignite"):
        row = summary["per_dataset"][name]
        iou = row["macro_iou"]
        lines.append(f"| {name.upper()} | 20 | {f(iou['mean'])} | {f(iou['oracle_expert_decoded_coarse_mean'])} | {f(iou['paired_mean_difference'])} | {f(row['whole_group_accuracy'])} |")
    iou = overall["macro_iou"]
    lines.append(f"| Equal-image overall | 60 | {f(iou['mean'])} | {f(iou['oracle_expert_decoded_coarse_mean'])} | {f(iou['paired_mean_difference'])} | {f(overall['whole_group_accuracy'])} |")
    lines += ["", f"Source-group bootstrap 95% interval for the equal-image mean macro intersection-over-union: {f(summary['source_group_bootstrap']['mean_macro_iou_95pct'][0])}–{f(summary['source_group_bootstrap']['mean_macro_iou_95pct'][1])} ({summary['source_group_bootstrap']['source_groups']} known source groups).",
        "", "The expert-decoded coarse score uses the hidden annotation-origin class key and is a privileged geometry diagnostic. It does not estimate deployable tissue identification accuracy.",
        "", "## Tissue identity and interface outcomes", "",
        f"Whole-group accuracy was {f(overall['whole_group_accuracy'])} across {overall['whole_group_count']} eligible groups; class-balanced accuracy was {f(overall['whole_group_balanced_accuracy'])}. The mean tumor–stroma boundary F1 at 8 µm was {f(overall['tumor_stroma_boundary_f1_at_8um']['mean'])} across {overall['tumor_stroma_boundary_f1_at_8um']['images']} images with a defined expert interface.",
        "", "| Input tiles in refined group | Eligible groups | Whole-group accuracy |",
        "|---|---:|---:|"]
    for name in ("1-4", "5-9", "10-19", "20+"):
        b = overall["tile_count_bins"][name]
        lines.append(f"| {name} | {b['groups']} | {f(b['accuracy'])} |")
    lines += ["", "## Secondary and scale demonstrations", "",
        "OCELOT is reported separately because its broad non-cancer region was generated automatically. Its expert tumor recall and tumor call fraction on the broad automatic negative region should be inserted only from `EXTERNAL_COMPLETE.json` and `summary.json`. The five whole-slide examples and three Visium sections require their own completed provenance and figures. They must not be pooled with the 60-field multiclass score.",
        "", "## Evidence identifiers", "",
        f"- Bora model SHA256: `{summary['model_sha256']}`",
        f"- Frozen split SHA256: `{summary['split_sha256']}`",
        f"- Summary SHA256: `{sha256(summary_file)}`",
        f"- Complete receipt SHA256: `{sha256(run / 'PRIMARY_COMPLETE.json')}`", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new versioned manuscript output")
    content = build(args.root.resolve(), args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(args.output)


if __name__ == "__main__":
    main()
