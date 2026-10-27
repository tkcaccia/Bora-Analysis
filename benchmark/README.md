# Bora tissue-mask benchmark

The current public result set is the 116-image Neighbor context study (BCSS 42, TIGER 26,
GCSS 24, IGNITE 24); see the [public viewer](https://tkcaccia.github.io/Bora-Analysis/)
and [current organization guide](../docs/benchmark_organization.md). This
directory retains public acquisition and source-annotation metadata. The
116-image scores are exploratory and should not be substituted for a new,
patient-disjoint confirmation.
The images that trained the manuscript fastPLS model are listed in
[the model-source guide](../docs/MANUSCRIPT_MODEL_DATA.md).
The separate [Visium HD colon whole-slide test](../studies/visium-hd-colon-wsi-2026-10-19/README.md)
reports complete-slide scale and native raster/vector fidelity alongside a
documented normal-gland classification failure. Its single, partially annotated
slide is not part of the 116-image cohort estimate.

This package downloads and inventories pathology images with **tissue-region** masks, builds reference-derived tile masks, runs Bora, and scores its raster and GeoJSON outputs. Acquisition code is in `download.py`; dataset preparation is in `prepare.py`, `prepare_additional.py`, and `prepare_requested.py`.

The frozen 27 October 2026 full rerun uses BCSS and TIGER (breast carcinoma) and IGNITE (lung carcinoma) for a 476-field classifier training set and a 60-field development evaluation, with 20 fields from each source. OCELOT's 512 mixed-organ fields have a separate tumor-positive endpoint because its broad negative is not expert verified. Twenty calibrated tissue fields from 12 clinical-center patients in BEETLE's **development** release are prepared as another secondary analysis; they are not the sequestered official evaluation set. [The current protocol](../pipeline/full_benchmark_protocol.json) and [Methods draft](../manuscript/FULL_RERUN_INTEGRATION_2026-10-27.md) give the exact splits, label rules and limitations. These new scores are pending. WSSS4LUAD, RINGS, CAMELYON16, GleasonXAI, PESO, GCSS and other acquired sources remain outside the frozen primary mean unless their tissue mask semantics, physical scale and source identities satisfy the protocol. This repository does not evaluate cell or gland-instance masks as tissue ground truth.

Set `BORA_DATA_ROOT` to a writable directory outside Git, or pass `--root`
to the acquisition commands. Original files and SHA256 inventories are stored
there. Virchow2 encoding can run on a separate authorized GPU host. Each record
in `prepared/*.jsonl` gives paired paths relative to its recorded data root.
Read `labels.json` and `resolution.json` before interpreting an annotation.
Zero is ignored in BCSS and TIGER but denotes tumour in WSSS4LUAD, so source
IDs must be normalized before scoring.

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

## Current preprocessing and Bora run

Use [the pipeline guide](../pipeline/README.md) to organize source data,
produce calibrated nonoverlapping Virchow2 tile plans, encode spatial tokens
without keeping tile image files, and invoke the released Bora 0.2.1
prepare/train/refine commands. The downloaded expert masks are reference
annotations: they are not passed to the Virchow2 encoder or to a target case
at inference. The fixed training workflow reads expert masks only after
anonymous group descriptors have been prepared.

The public viewer's 116-image result is an exploratory evaluation with
expert-derived coarse tile input. Do not interpret this as independent
annotation-free segmentation accuracy. For any new benchmark, lock patient
and parent-slide disjoint training/test partitions, harmonize tissue classes,
and regenerate all figures and tables from the new completed receipts.
