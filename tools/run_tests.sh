#!/usr/bin/env bash
# Run the unit test suite with the repo venv on any machine (Mac or Ubuntu/Jetson).
# Jetson torch 2.5 nv24.8 dlopens libcusparseLt.so.0 from pip nvidia-cusparselt-cu12, which is not on the loader
# path unless .venv/bin/activate was sourced -> put it on LD_LIBRARY_PATH here (2026-10-03, Plan 40 phase 2).
# A site-packages .pth ctypes preload was tried and aborts the suite (free(): invalid pointer) -> not used.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for d in .venv/lib/python3*/site-packages/nvidia/cusparselt/lib; do
  [[ -d "$d" ]] && export LD_LIBRARY_PATH="$PWD/$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
exec "$PY" -m unittest discover -s tests "$@"
