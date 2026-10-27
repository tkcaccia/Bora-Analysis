#!/usr/bin/env python3
"""Build the public 60-field Bora viewer from complete, checksum-verified results.

This command only reads completed CPU outputs and public source images. It
cannot run Virchow2 or Bora, and it refuses partial benchmark results.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import tifffile

from build_bora_case import sha256


MAX_SIDE = 1400


def palette(ids: list[int], *, semantic: bool) -> dict[int, tuple[int, int, int]]:
    colors = [(65, 105, 225), (225, 78, 68), (51, 163, 121), (211, 151, 44),
              (139, 94, 197), (32, 167, 191), (195, 92, 147), (115, 130, 58)]
    return {int(code): colors[(int(code) - (1 if semantic else 0)) % len(colors)] for code in ids}


def small(image: Image.Image) -> Image.Image:
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return image


def render(image: Image.Image, mask: np.ndarray, colors: dict[int, tuple[int, int, int]],
           output: Path, alpha: float = .38) -> None:
    original = small(image.convert("RGB"))
    if mask.shape != (image.height, image.width):
        mask = np.asarray(Image.fromarray(mask).resize((image.width, image.height),
                                                        Image.Resampling.NEAREST))
    labels = np.asarray(Image.fromarray(mask).resize(original.size, Image.Resampling.NEAREST))
    rgb = np.asarray(original, dtype=np.float32).copy()
    for label, color in colors.items():
        selected = labels == label
        rgb[selected] = rgb[selected] * (1 - alpha) + np.asarray(color) * alpha
    Image.fromarray(np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)).save(
        output, format="WEBP", quality=84, method=6)


def source_index(root: Path) -> dict[str, dict]:
    path = root / "management/tissue_mask_eligibility_2026-10-27.tsv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len({r["case_id"] for r in rows}):
        raise ValueError("Duplicate source inventory case IDs")
    return {r["case_id"]: r for r in rows}


def build(root: Path, run: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("Viewer destination exists; use a new versioned path")
    complete = json.loads((run / "PRIMARY_COMPLETE.json").read_text())
    summary_path = run / "primary-summary.json"
    if sha256(summary_path) != complete["primary_summary_sha256"]:
        raise ValueError("Summary changed after completion")
    summary = json.loads(summary_path.read_text())
    if complete["test_images"] != summary["fields"] or summary["fields"] != 60:
        raise ValueError("Expected the complete 60-field public evaluation")
    split_path = root / "preprocessing/full-rerun-2026-10-27/split-v1.json"
    if sha256(split_path) != summary["split_sha256"]:
        raise ValueError("Frozen split changed")
    rows = [r for r in json.loads(split_path.read_text())["assignments"] if r["split"] == "test"]
    if len(rows) != 60 or {r["dataset"] for r in rows} != {"bcss", "tiger", "ignite"}:
        raise ValueError("Unexpected public evaluation cohort")
    sources = source_index(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bora-viewer-", dir=destination.parent) as temp:
        folder = Path(temp) / "full-rerun"
        folder.mkdir()
        assets = folder / "assets"; assets.mkdir()
        gallery = []
        vector_bytes = 0
        for row in rows:
            case_id = row["case_id"]
            case = run / "cases" / case_id
            result = run / "test-results" / case_id
            score_path = result / "score.json"
            if sha256(score_path) != summary["score_sha256"][case_id]:
                raise ValueError("Missing or changed final score: " + case_id)
            score = json.loads(score_path.read_text())
            audit = json.loads((case / "CASE_COMPLETE.json").read_text())
            report = json.loads((result / "report.json").read_text())
            if score["result_report_sha256"] != sha256(result / "report.json"):
                raise ValueError("Score does not belong to final result: " + case_id)
            for name, digest in report["output_sha256"].items():
                if sha256(result / name) != digest:
                    raise ValueError("Output changed after Bora completion: " + case_id + "/" + name)
            source = root / sources[case_id]["source_image"]
            if sha256(source) != audit["source_sha256"]:
                raise ValueError("Original image differs: " + case_id)
            target = tifffile.imread(case / "expert-target-private.tif")
            groups = tifffile.imread(result / "groups.ome.tif")
            classified = tifffile.imread(result / "mask.ome.tif")
            coarse = np.load(case / "anonymous-tiles.npy", allow_pickle=False)
            if target.shape != groups.shape or target.shape != classified.shape:
                raise ValueError("Mask geometry differs: " + case_id)
            with Image.open(source) as original_file:
                original = original_file.convert("RGB")
            if original.size != (target.shape[1], target.shape[0]):
                raise ValueError("Original image geometry differs: " + case_id)
            place = assets / case_id; place.mkdir()
            small(original.copy()).save(place / "original.webp", format="WEBP", quality=88, method=6)
            semantic = palette(list(audit["canonical_role_ids"].values()), semantic=True)
            present_codes = {int(x) for x in np.unique(target)} | {int(x) for x in np.unique(classified)}
            present_roles = {name:code for name, code in audit["canonical_role_ids"].items()
                             if code in present_codes}
            anonymous = palette([int(x) for x in np.unique(coarse) if x], semantic=False)
            render(original, target, semantic, place / "expert.webp")
            grid = np.repeat(np.repeat(coarse, 90, axis=0), 90, axis=1)
            work_shape = audit["working_shape_yx"]
            grid = grid[:work_shape[0], :work_shape[1]]
            render(original, grid, anonymous, place / "input-tiles.webp")
            render(original, groups, palette([int(x) for x in np.unique(groups) if x], semantic=False),
                   place / "refined-groups.webp")
            render(original, classified, semantic, place / "final-mask.webp")
            with Image.open(result / "converted-rgb.png") as rgb:
                small(rgb.convert("RGB")).save(place / "converted-rgb.webp", format="WEBP", quality=86, method=6)
            vector = result / "mask.geojson"
            with vector.open("rb") as source_stream, gzip.open(place / "mask.geojson.gz", "wb", compresslevel=9) as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
            vector_bytes += (place / "mask.geojson.gz").stat().st_size
            gallery.append(dict(case_id=case_id, dataset=row["dataset"],
                native_shape_yx=audit["native_shape_yx"], native_mpp_xy=audit["native_mpp_xy"],
                role_ids=present_roles,
                role_colors={name:semantic[code] for name, code in present_roles.items()},
                images={name:f"assets/{case_id}/{name}.webp"
                    for name in ("original", "expert", "input-tiles", "converted-rgb", "refined-groups", "final-mask")},
                geojson_gz=f"assets/{case_id}/mask.geojson.gz", score=score,
                source_group=row["global_source_group"],
                unannotated_tissue_processed=True))
        data = dict(format="bora-public-full-rerun-viewer-v1", version="Bora 0.2.1",
            title="Bora calibrated tissue refinement", cohort="BCSS, TIGER and IGNITE",
            note="Development evaluation of expert-derived anonymous coarse tiles. This is not independent final testing.",
            summary=summary, cases=gallery)
        (folder / "data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        (folder / "index.html").write_text(HTML)
        (folder / "viewer.js").write_text(JS)
        (folder / "viewer.css").write_text(CSS)
        (folder / "BUILD_COMPLETE.json").write_text(json.dumps(dict(cases=60,
            summary_sha256=sha256(summary_path), total_compressed_vector_bytes=vector_bytes,
            data_sha256=sha256(folder / "data.json")), indent=2) + "\n")
        os.rename(folder, destination)
    return dict(cases=60, output=str(destination), compressed_vector_bytes=vector_bytes)


HTML = """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Bora calibrated tissue refinement</title><link rel=\"stylesheet\" href=\"viewer.css\"></head><body><header><h1>Bora calibrated tissue refinement</h1><p>Original H&amp;E, expert annotation, anonymous coarse tiles, Virchow2 evidence, refined groups and final classified mask.</p><p id=\"note\"></p></header><main><nav><label>Dataset <select id=\"dataset\"><option value=\"all\">All</option><option value=\"bcss\">BCSS</option><option value=\"tiger\">TIGER</option><option value=\"ignite\">IGNITE</option></select></label><label>Image <select id=\"case\"></select></label></nav><section id=\"score\"></section><section id=\"panels\" class=\"panels\"></section><section><h2>Tissue labels</h2><div id=\"legend\"></div><p><a id=\"vector\">Download native GeoJSON (gzip)</a></p></section></main><script src=\"viewer.js\"></script></body></html>"""

JS = """let data, chosen;const ids=['original','expert','input-tiles','converted-rgb','refined-groups','final-mask'];const titles=['Original H&E','Expert tissue annotation','Anonymous input tile mask','Virchow2 converted RGB','Bora refined anonymous groups','Bora final tissue mask'];const select=document.querySelector('#case'),dataset=document.querySelector('#dataset');fetch('data.json').then(r=>r.json()).then(d=>{data=d;document.querySelector('#note').textContent=d.note;dataset.onchange=populate;select.onchange=draw;populate();});function populate(){const cases=data.cases.filter(c=>dataset.value==='all'||c.dataset===dataset.value);select.replaceChildren(...cases.map(c=>{const o=document.createElement('option');o.value=c.case_id;o.textContent=c.dataset.toUpperCase()+' · '+c.case_id.split('__')[1];return o}));const wanted=decodeURIComponent(location.hash.slice(1));if(cases.some(c=>c.case_id===wanted))select.value=wanted;draw()}function draw(){chosen=data.cases.find(c=>c.case_id===select.value);if(!chosen)return;location.hash=encodeURIComponent(chosen.case_id);const score=chosen.score.final,coarse=chosen.score.oracle_expert_decoded_coarse;document.querySelector('#score').innerHTML='<h2>'+chosen.dataset.toUpperCase()+' · '+chosen.native_shape_yx.join(' × ')+' px · '+chosen.native_mpp_xy.join(' × ')+' µm/px</h2><p>Final macro IoU <strong>'+score.macro_iou.toFixed(3)+'</strong> · expert-decoded coarse IoU '+coarse.macro_iou.toFixed(3)+' · whole-group accuracy '+(chosen.score.group_accuracy===null?'not evaluable':chosen.score.group_accuracy.toFixed(3))+'</p><p>Coarse comparison uses the hidden expert identity key for diagnosis; it is not a deployable classifier.</p>';const panels=document.querySelector('#panels');panels.replaceChildren(...ids.map((id,i)=>{const article=document.createElement('article'),h=document.createElement('h3'),img=document.createElement('img');h.textContent=titles[i];img.src=chosen.images[id];img.alt=titles[i]+' for '+chosen.case_id;img.loading=i<2?'eager':'lazy';article.append(h,img);return article}));const legend=document.querySelector('#legend');legend.replaceChildren(...Object.entries(chosen.role_ids).map(([name,id])=>{const item=document.createElement('span'),chip=document.createElement('i');chip.style.background='rgb('+chosen.role_colors[name].join(',')+')';item.append(chip,document.createTextNode(name.replaceAll('_',' ')));return item}));document.querySelector('#vector').href=chosen.geojson_gz}window.addEventListener('hashchange',()=>{if(!data)return;const id=decodeURIComponent(location.hash.slice(1));if(data.cases.some(c=>c.case_id===id)){select.value=id;draw()}});"""

CSS = """*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;color:#19313b;background:#f7f9f9;margin:0}header,main{max-width:1600px;margin:auto;padding:24px}header{padding-bottom:0}h1{font-size:2rem;margin:.2em 0}h2{font-size:1.25rem}nav{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}label{display:grid;gap:5px}select{font:inherit;padding:8px;max-width:min(84vw,650px)}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:16px}article{background:white;border:1px solid #dce5e7;border-radius:10px;padding:12px}article h3{margin:0 0 8px;font-size:1rem}article img{display:block;width:100%;height:auto;max-height:600px;object-fit:contain}section{margin:16px 0}#legend{display:flex;flex-wrap:wrap;gap:8px 18px}#legend span{display:inline-flex;align-items:center;gap:6px}#legend i{width:16px;height:16px;border:1px solid #64757b;display:inline-block}a{color:#13607b}@media(max-width:450px){.panels{grid-template-columns:1fr}header,main{padding:14px}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.run.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
