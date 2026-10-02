#!/usr/bin/env bash
# Plan 38 E-LIP-B: LivePortrait (KwaiVGI, code MIT; insightface/buffalo models non-commercial) reenactment of the
# FIXED avatar on Apple MPS (with CPU fallback). Read-only use of web/gemma_chat/avatar.jpg (copied to /tmp).
# Usage: tools/plan38_liveportrait_run.sh <tag> [driving=d0.mp4] [extra LivePortrait args, e.g. --animation_region lip]
# Needs ~/plan38_cache/LivePortrait (+ pretrained_weights) and ~/plan38_cache/venv_lp. Output: runs/plan38/lipsync/<tag>/
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; TAG=${1:?tag}; DRV=${2:-d0.mp4}; shift; shift || true
O="$ROOT/runs/plan38/lipsync/$TAG"; mkdir -p "$O"; cp "$ROOT/web/gemma_chat/avatar.jpg" /tmp/p38_avatar.jpg
cd ~/plan38_cache/LivePortrait; export PYTORCH_ENABLE_MPS_FALLBACK=1
L0=$(uptime | sed 's/.*averages: //'); t0=$(python3 -c 'import time;print(time.time())')
../venv_lp/bin/python inference.py -s /tmp/p38_avatar.jpg -d "assets/examples/driving/$DRV" -o "$O" "$@" > "$O/log.txt" 2>&1; ec=$?
t1=$(python3 -c 'import time;print(time.time())')
echo "{\"tag\":\"$TAG\",\"driving\":\"$DRV\",\"args\":\"$*\",\"exit\":$ec,\"wall_s\":$(python3 -c "print(round($t1-$t0,2))"),\"load_before\":\"$L0\",\"load_after\":\"$(uptime | sed 's/.*averages: //')\"}" | tee "$O/timing.json"
