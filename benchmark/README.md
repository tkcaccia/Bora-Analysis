# Bora tissue-mask benchmark

The current public result set is the 116-image Neighbor context study (BCSS 42, TIGER 26,
GCSS 24, IGNITE 24); see the [public viewer](https://tkcaccia.github.io/Bora-Analysis/)
and [current organization guide](../docs/benchmark_organization.md). This
directory also preserves the earlier acquisition and controlled-refinement
experiments described below. Those older cohort sizes and results must not be
substituted for the current classifier evaluation. Experimental Virchow2,
conversion, classifier and geometry code is preserved under
`history/bora-development/experiments/`.
The images that trained the manuscript fastPLS model are listed in
[the model-source guide](../docs/MANUSCRIPT_MODEL_DATA.md).

This package downloads and inventories pathology images with **tissue-region** masks, builds reference-derived tile masks, runs Bora, and scores its raster and GeoJSON outputs. Acquisition code is in `download.py`; dataset preparation is in `prepare.py`, `prepare_additional.py`, and `prepare_requested.py`.

The primary multiclass study uses BCSS (breast), TIGER (breast), and WSSS4LUAD (lung). Separate tissue-region analyses use RINGS **tumour** masks (prostate) and native CAMELYON16 tumour windows (lymph node). The processed CoCaHis-VISTA subset and SICAPv2 are catalogued as potential tissue-region sources, but their mask semantics are not yet verified for manuscript scoring. PAIP2019 and DigestPath2019 require provider-approved acquisitions. This repository does not acquire or evaluate cell or gland-instance datasets.

Original files and SHA256 inventories are stored outside Git at `/mnt/sata_ssd/Bora-benchmark`. Each record in `prepared/*.jsonl` gives the paired paths relative to that data root. Read `labels.json` and `resolution.json` before interpreting an annotation. Zero is ignored in BCSS and TIGER but denotes tumour in WSSS4LUAD; the benchmark normalizes these source IDs before making tile masks.

## Acquisition

From the Bora-Analysis repository root:

```bash
python -m pip install -r benchmark/requirements.txt
python -m benchmark.download --list-datasets
python -m benchmark.download bcss --plan
python -m benchmark.download tiger --verify-only
python -m benchmark.download bcss
python -m benchmark.download tiger --extract --workers 8
python -m benchmark.prepare bcss
python -m benchmark.prepare tiger
python -m benchmark.download wsss4luad
python -m benchmark.prepare_additional wsss4luad
python -m benchmark.prepare_additional rings
python -m benchmark.download camelyon16-subset
python -m benchmark.prepare_requested camelyon16-subset
```

RINGS tumour images and masks are prepared from the existing local tissue-only extraction. The [publisher's RINGS dataset](https://data.mendeley.com/datasets/h8bdwrtnr5/1) distributes tumour and gland annotations together, so this repository does not provide an archive downloader for RINGS. The mixed archives and extracted gland files were removed from the local data root. `--verify-only` checks existing pinned files without changing them. Source hashes and URLs for downloadable tissue-only datasets are in `sources.lock.json`. Raw archives and full-resolution data remain outside Git. Only the explicitly packaged, attributed viewer previews and GeoJSONs are published in `site/`.

## Evaluation and figures

Bora is installed separately from [tkcaccia/Bora](https://github.com/tkcaccia/Bora). Set `BORA_ROOT` when it is not a sibling checkout of Bora-Analysis. The commands below are historical experiments requiring their original pinned Bora runtime; current Bora uses the Neighbor context pipeline.

```bash
export BORA_ROOT=/path/to/Bora
PYTHONPATH=. python -m benchmark.run_bora_pilot
PYTHONPATH=. python -m benchmark.run_tiger_expansion
PYTHONPATH=. python -m benchmark.run_bcss_expansion
PYTHONPATH=. python -m benchmark.run_additional_datasets --dataset rings-tumor
PYTHONPATH=. python -m benchmark.run_camelyon_extension
PYTHONPATH=. python -m manuscript.build_mask_comparison
PYTHONPATH=. python -m manuscript.build_additional_mask_atlas
PYTHONPATH=. python -m manuscript.build_mia_pilot_docx
PYTHONPATH=. python -m manuscript.build_mia_working_preview
```

The principal manuscript analysis comprises 63 distinct ROIs: 12 BCSS, 15 TIGER, and 36 WSSS4LUAD. The separate RINGS and CAMELYON16 experiments use three images each and are not pooled with the primary multiclass endpoint. The figure generator shows reference, tile and refined masks over identical H&E coordinates with 42% opacity. The GeoJSON audit rasterizes unsimplified polygons back to pixels and repeats conversion to check deterministic bytes.

The manuscript and generated images remain in the Bora checkout under `docs/` and `output/pdf/`; only their generation code lives here.
