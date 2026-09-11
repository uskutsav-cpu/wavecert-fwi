#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-data/external/marmousi/mar_big_117_567.bin}"
mkdir -p "$(dirname "$OUT")"
curl -L --fail --retry 3 \
  'https://raw.githubusercontent.com/DeepWave-KAUST/Siamese_FWI-pub/main/data/mar_big_117_567.bin' \
  -o "$OUT"
echo "saved $OUT"
