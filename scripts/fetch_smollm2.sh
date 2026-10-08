#!/usr/bin/env bash
# Fetch SmolLM2-135M (bf16 weights) and its tokenizer files from Hugging Face into
# data/smollm2-135m/ (D67). About 272 MB; resumable (curl -C -). Run from the repository root.
set -euo pipefail
dir=data/smollm2-135m
mkdir -p "$dir"
# The revision this project was developed against (2026-10-08).
rev=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
base=https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/$rev
for f in config.json vocab.json merges.txt tokenizer.json model.safetensors; do
  echo "fetching $f"
  curl -fL --retry 3 -C - -o "$dir/$f" "$base/$f"
done
echo "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1  $dir/model.safetensors" \
  | sha256sum -c -
