#!/usr/bin/env bash
# Fetch the GPT-2 124M checkpoint and tokenizer files from Hugging Face into data/gpt2/ (D10).
# About 550 MB; resumable (curl -C -). Run from the repository root.
set -euo pipefail
dir=data/gpt2
mkdir -p "$dir"
base=https://huggingface.co/openai-community/gpt2/resolve/main
for f in config.json vocab.json merges.txt tokenizer.json model.safetensors; do
  echo "fetching $f"
  curl -fL --retry 3 -C - -o "$dir/$f" "$base/$f"
done
# The checkpoint this project was developed against (2026-10-04).
echo "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707  $dir/model.safetensors" \
  | sha256sum -c -
