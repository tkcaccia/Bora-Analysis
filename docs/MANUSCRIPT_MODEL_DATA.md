# Images used to train the manuscript tissue classifier

The manuscript's Neighbor context fastPLS model was trained on **967 refined
whole-group examples from 349 annotated image fields**. This is a training
cohort, not the 116-image exploratory evaluation shown in the public viewer.
The exact source fields are listed in
[`manuscript_model_training_sources.json`](manuscript_model_training_sources.json).
Each record has its provider source ID, source image/annotation paths or ZIP
members, source group, selected native crop, native and working µm/px, class
mapping, and source/prepared-image checksums. The manifest also records the
SHA256 of the frozen registry from which these rows were exported. It contains
no full-resolution images, training descriptors or model artifact.

| Source release | Training fields | Where to obtain original images and annotations |
| --- | ---: | --- |
| [BCSS](https://github.com/PathologyDataScience/BCSS) | 90 | [Pinned MedOtter/BCSS mirror](../benchmark/sources.lock.json), revision `502d4a3fbc77dbaca6f4664c19e2379ff077d418` |
| [TIGER](https://zenodo.org/records/6014422) | 46 | Provider `roi-level-annotations.zip`; exact ROI members in the manifest |
| [GCSS](https://zenodo.org/records/8176354) | 56 | Provider `GCSS.zip`, including `images.zip`, `labels.zip` and ROI coordinates |
| [IGNITE v2](https://zenodo.org/records/17735903) | 40 | Provider `images.zip`, `annotations.zip`, label map and data overview; selected ZIP members in the manifest |
| [OCELOT v1.0.1](https://zenodo.org/records/8417503) | 36 | Provider image/annotation release; acquisition code pins the version and checksum |
| [PUMA](https://zenodo.org/records/15050523) | 25 | Tissue GeoJSON plus original ROI and context TIFF archives; PUMA was excluded from evaluation but remains in training |
| [PESO](https://zenodo.org/records/5137717) | 24 | Provider image, ground-truth and region archives |
| [GleasonXAI](https://doi.org/10.6084/m9.figshare.27301845.v1) | 20 | Pinned Figshare image archive, explanation table and label mapping |
| [BEETLE v1](https://zenodo.org/records/16812932) | 7 | Provider development WSI `images.zip` and `annotations.zip`; provider reports a roughly 147-GB image archive |
| [CAMELYON16](https://registry.opendata.aws/camelyon/) | 5 | Three pinned positive training WSIs and masks in the repository's bounded subset |

This table lists original releases. Several training fields are calibrated
**crops of those images**, not provider files by themselves. The manifest's
`native_crop_xywh`, per-axis spacing, hashes and source-group identifiers are
needed to reproduce the exact selected fields. For GCSS, source-slide
calibration was checked against public GDC metadata and provider ROI bounds;
do not assign the WSI's µm/px directly to its distributed PNG. For PESO,
original mask and image grids have a documented factor-of-two relationship.
The exact crop and spacing records remain in the source manifest. The current
public data workflow is described in [`pipeline/`](../pipeline/README.md).

## List or export the exact source IDs

From the Bora-Analysis repository root:

```bash
python tools/list_manuscript_model_sources.py --dataset tiger
python tools/list_manuscript_model_sources.py --csv manuscript-training-sources.csv
```

The CSV includes each field's source ID, original source path/ZIP member,
native crop and SHA256. The JSON has the complete class and resolution
metadata. All ten training dataset counts above are checked against the JSON.

## Acquire the provider releases

Set a data root with enough space for the selected archives. The commands
below download **provider data only**; they do not train Bora or run an
encoder. Inspect each provider's current access and reuse terms before
downloading or sharing derivatives. The pinned catalogues preserve original
file sizes and provider checksums.

```bash
DATA=/path/to/Bora-benchmark
python -m pip install -r benchmark/requirements.txt
python -m benchmark.download bcss --root "$DATA"
python -m benchmark.prepare bcss --root "$DATA"
python -m benchmark.download tiger --root "$DATA" --extract --workers 8
python -m benchmark.prepare tiger --root "$DATA"
python -m benchmark.download camelyon16-subset --root "$DATA" --workers 4
python -m benchmark.prepare_requested camelyon16-subset --root "$DATA"
```

The exact selected fields are the rows in the manuscript source manifest,
not every item in these provider releases.

The checksum-aware source downloader covers GCSS, PESO, IGNITE
metadata/annotations, PUMA tissue GeoJSON and a small GleasonXAI source
catalogue. It retains partial transfers and verifies pinned file identities:

```bash
python pipeline/download_catalog.py --root "$DATA" \
  --catalog pipeline/tissue_sources.lock.json
python pipeline/download_catalog.py --root "$DATA" \
  --catalog docs/manuscript_model_extra_catalog.json \
  --receipt-name manuscript-gleasonxai-acquisition.json
python pipeline/download_ocelot.py --data-root "$DATA"
```

The larger IGNITE, PUMA and BEETLE archives are provider releases and require
space and access checks before acquisition. The source manifest identifies
the exact image/annotation members required for model reproduction.
BEETLE's
development WSI and annotation archives should be obtained from its pinned
Zenodo v1 record; the public evaluation masks are sequestered and were **not**
used to train this model. The exact seven BEETLE WSI source IDs are in the
manifest. Do not download a newer dataset version and assume byte identity.

These commands fetch original releases. Recreating the **exact manuscript
model** also requires applying the recorded crop/calibration/ontology rules,
extracting the pinned Virchow2 spatial tokens, supplying the anonymous tile
clusters, then running the current Bora `prepare` and `train` commands. The
frozen source hashes are preserved here. Large derived token caches and the
training model are not redistributed; the exact historical model cannot be
recreated from the small source catalogue alone.
The Virchow2 checkpoint used for feature extraction was
[`paige-ai/Virchow2`](https://huggingface.co/paige-ai/Virchow2) at revision
`3158645804b69e3f3bc4439d4116edddf0840a72`, with checkpoint SHA256
`8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b`.
Access to that checkpoint follows its provider terms.

For a **new user's own model**, no manuscript dataset is required. Follow the
[Bora training guide](https://github.com/tkcaccia/Bora/blob/main/docs/WORKFLOW.md):
prepare each calibrated image and its tile clusters/tokens; in a training
manifest supply the corresponding expert TIFF mask or native-coordinate
GeoJSON and map source annotation values to tissue class IDs; run `bora train`;
then run `bora refine` on a separate target image using the saved model.

The training cohort includes sources with noncommercial or attribution terms;
see the [data reuse notice](../web/DATA_LICENSES.md). The model NPZ contains
training descriptors and labels and must be shared only when their underlying
data terms permit it.
