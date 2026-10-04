#!/usr/bin/env bash
# Plant one bug at a time in FILE, run the tests, restore.
# Usage: scripts/mutate.sh FILE MUTATIONS   (MUTATIONS: one sed expression per line)
# Set MUTATE_TEST_ARGS (e.g. "--test gpu_ops") to run only the tests that can see the mutant.
# Prefer line-addressed expressions with @ as the delimiter: 30s@old@new@
set -u
export TMPDIR="$PWD/target/tmp"
file="$1"
cp "$file" "$TMPDIR/mutate.bak"
while IFS= read -r expr; do
  [ -z "$expr" ] && continue
  sed -i "$expr" "$file"
  if cmp -s "$file" "$TMPDIR/mutate.bak"; then
    echo "NO-OP    $expr"
    continue
  fi
  if timeout 300 cargo test -q ${MUTATE_TEST_ARGS:-} >"$TMPDIR/mutate.log" 2>&1; then
    echo "SURVIVED $expr"
  else
    echo "caught   $expr"
  fi
  cp "$TMPDIR/mutate.bak" "$file"
done < "$2"
cargo test -q 2>&1 | grep -E 'test result' | head -n 1
