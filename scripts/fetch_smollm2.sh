#!/usr/bin/env bash
# Fetch SmolLM2 (bf16 weights) and its tokenizer files from Hugging Face into
# data/smollm2-<size>/ (D67). Size 135m (default, ~272 MB) or 360m (~727 MB, the scale check).
# Resumable (curl -C -). Run from the repository root.
set -euo pipefail
size=${1:-135m}
# The revisions this project was developed against (2026-10-08), and their weights' sha256.
case "$size" in
  135m) rev=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
        sha=80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1 ;;
  360m) rev=f8027fd0eaeea54caa13c31d31b9fdc459c38b49
        sha=7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f ;;
  *) echo "usage: $0 [135m | 360m]" >&2; exit 2 ;;
esac
dir=data/smollm2-$size
mkdir -p "$dir"
base=https://huggingface.co/HuggingFaceTB/SmolLM2-${size^^}/resolve/$rev
for f in config.json vocab.json merges.txt tokenizer.json model.safetensors; do
  echo "fetching $f"
  curl -fL --retry 3 -C - -o "$dir/$f" "$base/$f"
done
echo "$sha  $dir/model.safetensors" | sha256sum -c -
