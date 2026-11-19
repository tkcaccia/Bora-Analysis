#!/usr/bin/env python3
"""Refresh Benchmark 3's index tables from independently audited one-slide scores.

Run after a new method score JSON has passed its independent audit and before
refreshing PUBLIC_RELEASE_VERIFIED.json. --check is a read-only release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path


SITE = Path(__file__).resolve().parents[1]
CAM = SITE / "site" / "camelyon16"
INDEX = CAM / "index.html"
SECTION_ID = "one-slide-bora-results"
METHODS = (
    ("bora", "Bora"),
    ("unet", "Class-aware U-Net"),
    ("deeplabv3plus", "DeepLabV3+"),
    ("medsam", "MedSAM with U-Net boxes"),
    ("vista_path", "VISTA-PATH"),
)
# From the frozen 26-slide cohort; order is the existing gallery order.
SOURCE_SELECTED = (
    "tumor_038", "tumor_087", "normal_004", "normal_116", "tumor_005",
    "tumor_096", "tumor_101", "normal_021", "normal_103", "normal_123",
    "tumor_031", "tumor_016", "tumor_019", "tumor_078", "tumor_073",
    "normal_016", "normal_054", "normal_045", "normal_148", "normal_156",
)
PREVIOUSLY_INSPECTED = (
    "tumor_084", "tumor_091", "normal_019", "tumor_075", "normal_108",
    "normal_150",
)
COHORT = SOURCE_SELECTED + PREVIOUSLY_INSPECTED
SHA = re.compile(r"[0-9a-f]{64}\Z")
SECTION = re.compile(r'<section id="one-slide-bora-results">.*?</section>', re.DOTALL)
PROGRESS = re.compile(r'<section aria-label="Progress">.*?</section>', re.DOTALL)
PARTIAL = re.compile(r'<section><h2>How to read partial results</h2>.*?</section>', re.DOTALL)


def valid_hash(value: object) -> bool:
    return isinstance(value, str) and bool(SHA.fullmatch(value))


def read_audited_scores() -> dict[str, dict]:
    results = {}
    for slide in COHORT:
        path = CAM / f"{slide}-scores.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        assert data.get("slide") == slide, path
        assert data.get("training_slides_bora") == ["tumor_093"], path
        assert str(data.get("status", "")).startswith("INDEPENDENTLY_AUDITED"), path
        assert valid_hash(data.get("score_audit_sha256")), path
        if "official_positive_pixels" in data:
            assert (data["official_positive_pixels"] > 0) == slide.startswith("tumor_"), path
        assert set(data.get("methods", {})) <= {key for key, _ in METHODS}, path
        for key, method in data["methods"].items():
            assert valid_hash(method.get("score_receipt_sha256")), (path, key)
            native = method["metrics"]["native"]
            shared = method["metrics"]["shared_support"]
            assert native["tp"] + native["fp"] + native["fn"] + native["tn"] == method["metrics"]["scored_known_pixels"], (path, key)
            assert shared["tp"] + shared["fp"] + shared["fn"] + shared["tn"] == method["metrics"]["scored_known_pixels"], (path, key)
            assert (native["reference_positive_pixels"] > 0) == slide.startswith("tumor_"), (path, key)
            if "official_positive_pixels" in data:
                assert native["reference_positive_pixels"] == data["official_positive_pixels"], (path, key)
            if slide.startswith("tumor_"):
                assert all(native.get(k) is not None for k in ("iou", "dice", "balanced_accuracy")), (path, key)
            else:
                assert native.get("iou") is None and native["reference_positive_pixels"] == 0, (path, key)
        results[slide] = data
    unexpected = sorted(
        p.name for p in CAM.glob("*-scores.json")
        if re.fullmatch(r"(?:tumor|normal)_\d{3}-scores\.json", p.name)
        and p.stem.removesuffix("-scores") not in COHORT
    )
    assert not unexpected, f"Score files outside frozen cohort: {unexpected}"
    return results


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def cell(slide: str, method: dict | None) -> str:
    if method is None:
        return '<td class="score-pending">Pending</td>'
    native = method["metrics"]["native"]
    shared = method["metrics"]["shared_support"]
    if slide.startswith("tumor_"):
        primary = f"IoU <strong>{pct(native['iou'])}</strong>"
        secondary = f"Dice {pct(native['dice'])} · balanced accuracy {pct(native['balanced_accuracy'])}"
        sensitivity = f"Shared-support IoU {pct(shared['iou'])}"
    else:
        specificity = native.get("specificity")
        if specificity is None:
            specificity = native["tn"] / (native["tn"] + native["fp"])
        else:
            assert abs(specificity - native["tn"] / (native["tn"] + native["fp"])) < 1e-12
        primary = f"False-positive <strong>{native['false_positive_tissue_area_mm2']:.3f} mm²</strong>"
        secondary = f"Specificity {pct(specificity)}"
        sensitivity = f"Shared-support FP {shared['false_positive_tissue_area_mm2']:.3f} mm²"
    return f'<td>{primary}<small>{secondary}</small><small>{sensitivity}</small></td>'


def table(slides: list[str], scores: dict[str, dict], title: str, anchor: str) -> str:
    headers = "".join(f"<th scope=\"col\">{html.escape(name)}</th>" for _, name in METHODS)
    rows = []
    for slide in slides:
        data = scores.get(slide, {})
        methods = data.get("methods", {})
        if not methods:
            continue
        scope = "Previously inspected" if slide in PREVIOUSLY_INSPECTED else "Source selected"
        cells = "".join(cell(slide, methods.get(key)) for key, _ in METHODS)
        rows.append(
            f'<tr><th scope="row"><a href="{slide}.html">{slide}</a>'
            f'<small>{scope}</small></th>{cells}</tr>'
        )
    return (f'<h3 id="{anchor}">{title} ({len(rows)} scored slides)</h3>'
            '<div class="table-wrap"><table class="method-score-table">'
            f'<thead><tr><th scope="col">Slide</th>{headers}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def render(scores: dict[str, dict]) -> str:
    positive = [s for s in COHORT if s.startswith("tumor_")]
    benign = [s for s in COHORT if s.startswith("normal_")]
    nmethods = sum(len(d["methods"]) for d in scores.values())
    pending = [s for s in COHORT if s not in scores or len(scores[s]["methods"]) < len(METHODS)]
    return (
        f'<section id="{SECTION_ID}"><h2>Current one-slide method performance</h2>'
        f'<p><strong>{nmethods}/130 audited method results across {len(scores)}/26 slides.</strong> '
        'All Bora, U-Net and DeepLabV3+ CAM fits use tumor_093 only; MedSAM uses boxes from the current U-Net and VISTA-PATH uses a fixed generic tumor prompt. '
        'These methods have different upstream supervision and prompt dependencies. Native predictions are primary; shared-support values are sensitivity analyses. '
        'The source-selected slides and six previously inspected slides are labeled separately. All official references are now visible, so this is a descriptive, reference-visible comparison, not a blind challenge-test estimate.</p>'
        '<p>Scroll either table horizontally on narrow screens to see all five methods.</p>'
        + table(positive, scores, "Metastasis-positive reference · native performance", "positive-method-scores")
        + '<p>Positive-slide cells report native metastasis IoU, Dice and balanced accuracy, followed by shared-support IoU. Higher is better for these metrics.</p>'
        + table(benign, scores, "Benign reference · native false-positive burden", "benign-method-scores")
        + '<p>Benign-slide cells report native false-positive area on scored known tissue pixels and specificity, followed by shared-support false-positive area. Lower false-positive area is better. IoU, Dice and balanced accuracy are undefined without reference metastasis. The normal_019 scoring domain uses a post hoc corrected tissue support; see its slide page.</p>'
        + (f'<p><strong>{len(pending)} slides still have at least one pending method:</strong> '
           + ", ".join(html.escape(s) for s in pending) + '. Pending values are omitted until independently audited.</p>' if pending else '')
        + '</section>'
    )


def render_progress(scores: dict[str, dict]) -> str:
    bora = sum("bora" in d["methods"] for d in scores.values())
    complete = sum(len(d["methods"]) == len(METHODS) for d in scores.values())
    return (
        '<section aria-label="Progress"><h2>Current progress</h2><div class="status-grid">'
        '<article><strong>26/26</strong>Source slides identified</article>'
        '<article><strong>26/26</strong>Source identities verified</article>'
        f'<article><strong>{bora}/26</strong>One-slide Bora scores audited</article>'
        f'<article><strong>{complete}/26</strong>Complete current five-method comparisons</article>'
        '</div><p>The method tables below show each independently audited current-pipeline score. '
        'Pending methods have no reported value; earlier model fits are excluded. '
        'Native predictions are primary and shared-support results are secondary.</p></section>'
    )


def render_partial(scores: dict[str, dict]) -> str:
    complete = sum(len(d["methods"]) == len(METHODS) for d in scores.values())
    return (
        '<section><h2>How to read partial results</h2>'
        f'<p>{complete} slides have complete current five-method scores. '
        'Remaining method comparisons are pending. All expert views are public, '
        'so later scoring is reference-visible.</p></section>'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the index table is stale")
    args = parser.parse_args()
    scores = read_audited_scores()
    original = INDEX.read_text()
    assert len(SECTION.findall(original)) == 1, "Expected one method-results section"
    assert len(PROGRESS.findall(original)) == 1, "Expected one progress section"
    assert len(PARTIAL.findall(original)) == 1, "Expected one partial-results note"
    updated = SECTION.sub(lambda _: render(scores), original)
    updated = PROGRESS.sub(lambda _: render_progress(scores), updated)
    updated = PARTIAL.sub(lambda _: render_partial(scores), updated)
    if args.check:
        assert updated == original, "Benchmark 3 method table is stale; run refresh_cam_method_tables.py"
        print(f"Current: {len(scores)} scored slides and {sum(len(d['methods']) for d in scores.values())} methods")
    else:
        INDEX.write_text(updated)
        print(f"Updated: {len(scores)} scored slides and {sum(len(d['methods']) for d in scores.values())} methods")


if __name__ == "__main__":
    main()
