# Visium HD colon whole-slide demonstration

[Open the whole-slide viewer](https://tkcaccia.github.io/Bora-Analysis/visium-hd-colon/).
The source is a publicly released [10x Genomics Visium HD human colorectal
cancer H&E image](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4).
It measures 75,250 × 48,740 native pixels. Its OME pixel spacing is
0.27381215 µm in X and 0.27381326 µm in Y; the scanner objective is 20×.
The 11,098,424,743-byte source image has SHA-256
`beae7ecf17befbc8291c7228151674c858be8d6227333f25e08f5ef6868d93ab`.
The [download script](../../pipeline/download_colon_wsi.sh) verifies the
provider image. A writable data root is supplied by the user; no local machine
path is assumed.

The historical demonstration processed the entire image in 228 bounded
windows and exported a native mask and GeoJSON. A separately supplied,
partial GeoJSON annotation was withheld from image encoding and inference and
used for restricted scoring only. Its 382 features do not cover the whole
slide. The source and target choice had been inspected during development,
so this demonstration is exploratory and cannot establish independent
whole-slide accuracy.

The [generic pipeline guide](../../pipeline/README.md) documents image
acquisition, calibrated Virchow2 tile planning, feature extraction and Bora
inference. The public viewer contains attributed display previews. Large
source images, encoder weights, feature caches and native-resolution output
files remain outside Git. The viewer's previews are not calibrated model
inputs.
