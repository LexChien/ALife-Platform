#!/usr/bin/env bash
# Self-watching runner for Plan 44 M4-M7 on the Jetson (run inside tmux session `cell_sim`).
# Restarts the orchestrator on failure (completed stages are skipped via stage_*.done), max 3 attempts.
# Author / 作者: Grok Bot ALife Project - Andreeva, 2026-11-23 (Asia/Taipei).
set -u
RUN="$1"; shift
PY="${CELL_SIM_PY:-$HOME/venvs/cell_sim/bin/python}"
cd "$(dirname "$0")/.."
mkdir -p "$RUN"
for attempt in 1 2 3; do
  echo "$(date '+%F %T') attempt $attempt" >> "$RUN/supervisor.log"
  env -u LD_LIBRARY_PATH "$PY" tools/run_cell_m4_m7.py --run "$RUN" "$@" >> "$RUN/orchestrator.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') exit $rc" >> "$RUN/supervisor.log"
  [ $rc -eq 0 ] && break
  sleep 10
done
env -u LD_LIBRARY_PATH "$PY" -m cell.report "$RUN" --verify >> "$RUN/supervisor.log" 2>&1
echo "$(date '+%F %T') supervisor finished" >> "$RUN/supervisor.log"
