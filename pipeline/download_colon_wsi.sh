#!/usr/bin/env bash
set -euo pipefail

destination=${1:?Usage: download_colon_wsi.sh DESTINATION_DIRECTORY}
mkdir -p "$destination"
image="$destination/Visium_HD_Human_Colon_Cancer_tissue_image.btf"
url='https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Human_Colon_Cancer/Visium_HD_Human_Colon_Cancer_tissue_image.btf'

if [[ ! -f "$image" ]]; then
  curl --fail --location --output "$image.part" "$url"
  mv "$image.part" "$image"
fi

expected_bytes=11098424743
actual_bytes=$(wc -c < "$image" | tr -d ' ')
if [[ "$actual_bytes" != "$expected_bytes" ]]; then
  printf 'Unexpected BTF byte count: %s (expected %s)\n' "$actual_bytes" "$expected_bytes" >&2
  exit 1
fi
expected_sha256=beae7ecf17befbc8291c7228151674c858be8d6227333f25e08f5ef6868d93ab
if command -v sha256sum >/dev/null 2>&1; then
  actual_sha256=$(sha256sum "$image" | cut -d' ' -f1)
else
  actual_sha256=$(shasum -a 256 "$image" | cut -d' ' -f1)
fi
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  printf 'Unexpected BTF SHA-256: %s (expected %s)\n' "$actual_sha256" "$expected_sha256" >&2
  exit 1
fi
printf '%s\n' "$image"
