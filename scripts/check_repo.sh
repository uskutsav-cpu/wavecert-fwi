#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m compileall -q src tests
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m wavecert demo --output /tmp/wavecert-check >/dev/null
echo "WaveCert-FWI checks passed."
