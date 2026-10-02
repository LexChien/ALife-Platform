#!/usr/bin/env bash
# Plan 38: start a SEPARATE llama-server (port 8091 default) so the gemma_web daemon (:8080) is untouched.
# usage: tools/plan38_llm_server.sh [model.gguf] [port] [ctx] [n_parallel]
# Reasoning OFF (-rea off): measured 0/6 thought leaks vs 5/6 meta-narration with --reasoning-budget 0 alone
# (runs/plan38/llm/reasoning_modes.json).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/gemma/gemma.gguf}"
PORT="${2:-8091}"
CTX="${3:-16384}"
NP="${4:-2}"
BIN="$ROOT/third_party/llama.cpp/build-plan38/bin/llama-server"
exec "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -ngl 99 -c "$CTX" --jinja \
  -rea off -np "$NP" --cache-reuse 256 --metrics
