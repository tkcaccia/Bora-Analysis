"""Profile the cached Bora pipeline with stage timing and process peak RSS.

This profiles image/input loading, model loading and the released Bora `run`
function. It does not infer Virchow2 tokens; report encoder time and GPU memory
separately. Use native image and token files produced by the same encoder run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if platform.system() == "Darwin" else 1024)


def resident_mb() -> float:
    value = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(value.strip()) / 1024


def run_worker(args: argparse.Namespace) -> None:
    if args.native_core:
        # Only needed for source checkouts that have not built the bundled core.
        spec = importlib.util.spec_from_file_location("bora._vendor.fastpls._core", args.native_core)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    from bora.classifier import NeighborContextClassifier
    from bora.pipeline import load_case, run
    import bora

    source_root = Path(bora.__file__).resolve().parents[2]
    source_commit = None
    if (source_root / ".git").exists():
        source_commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()

    baseline_peak = peak_rss_mb()
    start = perf_counter()
    case = load_case(args.manifest[0])
    load_case_seconds = perf_counter() - start
    after_case_peak = peak_rss_mb()
    after_case_resident = resident_mb()

    start = perf_counter()
    model = NeighborContextClassifier.load(args.model)
    model_load_seconds = perf_counter() - start
    after_model_peak = peak_rss_mb()
    after_model_resident = resident_mb()

    start = perf_counter()
    report = run(case, model, args.folder)
    run_wall = perf_counter() - start
    after_run_peak = peak_rss_mb()
    after_run_resident = resident_mb()
    stage = report["timing_seconds"]
    recorded = sum(stage.values()) + report["classifier_seconds"] + report["export_seconds"]
    result = {
        "case": args.manifest[0].stem.replace(".case", ""),
        "shape_yx": list(case.image.shape[:2]),
        "working_mpp_xy": case.mpp_xy.tolist(),
        "native_mpp_xy": case.native_mpp_xy.tolist(),
        "input_load_seconds": load_case_seconds,
        "model_load_refit_seconds": model_load_seconds,
        "rgb_conversion_seconds": stage["rgb_conversion"],
        "appearance_refinement_seconds": stage["refinement"],
        "group_pooling_context_seconds": stage["pooling_and_context"],
        "fastpls_predict_seconds": report["classifier_seconds"],
        "geojson_mask_export_seconds": report["export_seconds"],
        "other_run_seconds": max(0., report["pipeline_wall_seconds_excluding_model_load"] - recorded),
        "run_wall_seconds": run_wall,
        "total_wall_seconds": load_case_seconds + model_load_seconds + run_wall,
        "rss_after_import_mb": baseline_peak,
        "rss_after_case_load_mb": after_case_resident,
        "rss_after_model_load_mb": after_model_resident,
        "rss_after_run_mb": after_run_resident,
        "peak_rss_after_case_load_mb": after_case_peak,
        "peak_rss_after_model_load_mb": after_model_peak,
        "peak_rss_total_mb": after_run_peak,
        "model_classes": len(model.names_),
        "group_count": len(report["groups"]),
        "source_version": __import__("bora").__version__,
        "source_commit": source_commit,
        "rgb_cache_supplied": case.rgb is not None,
        "gpu_used": False,
        "virchow2_inference_included": False,
    }
    print(json.dumps(result, separators=(",", ":")))


def run_driver(args: argparse.Namespace) -> None:
    records = []
    for manifest in args.manifest:
        for repeat in range(args.repeats):
            with tempfile.TemporaryDirectory(prefix="bora-pipeline-resource-") as temporary:
                command = [sys.executable]
                if args.no_site:
                    command.append("-S")
                command += [str(Path(__file__).resolve()), "--worker",
                           "--manifest", str(manifest.resolve()), "--model", str(args.model.resolve()),
                           "--folder", str(Path(temporary) / "output")]
                if args.native_core:
                    command += ["--native-core", str(args.native_core.resolve())]
                process = subprocess.run(command, capture_output=True, text=True,
                                         timeout=args.timeout_seconds, check=False)
                if process.returncode:
                    raise RuntimeError(f"{manifest.name} failed:\n{process.stderr}")
                result = json.loads(process.stdout)
                result["repeat"] = repeat + 1
                records.append(result)
                print(f"{result['case']} {repeat + 1}/{args.repeats}: "
                      f"{result['total_wall_seconds']:.3f}s, "
                      f"{result['peak_rss_total_mb']:.1f} MB", flush=True)
    fields = ("input_load_seconds", "model_load_refit_seconds", "rgb_conversion_seconds",
              "appearance_refinement_seconds", "group_pooling_context_seconds",
              "fastpls_predict_seconds", "geojson_mask_export_seconds", "other_run_seconds",
              "run_wall_seconds", "total_wall_seconds", "peak_rss_total_mb")
    summary = {field: {"median": statistics.median(r[field] for r in records),
                       "maximum": max(r[field] for r in records)} for field in fields}
    output = {
        "scope": f"{len(args.manifest)} ROI cases with cached Virchow2 features and a pretrained Neighbor context model; no new encoder inference",
        "timing_contract": "Fresh process per case/repeat; load_case and model refit timed separately; run stage timers from Bora report; imports excluded; peak RSS includes imports",
        "memory_contract": "Peak RSS is the whole process high-water mark; stage-end resident values are observations, not stage-specific peaks",
        "hardware": platform.platform(),
        "repeats_per_case": args.repeats,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--native-core", type=Path)
    parser.add_argument("--no-site", action="store_true", help="Suppress site imports for an isolated source checkout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    if args.worker:
        if len(args.manifest) != 1 or args.folder is None:
            parser.error("Worker requires one manifest and output folder")
        run_worker(args)
    else:
        if args.output is None or args.repeats < 1:
            parser.error("Driver requires output and positive repeats")
        run_driver(args)


if __name__ == "__main__":
    main()
