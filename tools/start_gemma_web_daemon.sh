#!/usr/bin/env bash
# Start DigiClone gemma_web in a detached tmux session so it survives
# remote/agent Shell process-group cleanup (plain nohup often dies).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SESSION="${GEMMA_WEB_TMUX_SESSION:-gemma_web}"
HOST="${GEMMA_WEB_HOST:-127.0.0.1}"
PORT="${GEMMA_WEB_PORT:-8080}"
# Plan 38: DigiClone JARVIS config (resident llama-server + cognitive loop + realtime WS on PORT+1).
# Previous llama-cli config: GEMMA_WEB_CONFIG=configs/genai/gemma_llama_cpp.yaml
CONFIG="${GEMMA_WEB_CONFIG:-configs/genai/digiclone_jarvis.yaml}"
# macOS: Metal offload (same GGUF, same llama.cpp, ~4x faster). Override with GEMMA_WEB_PROFILE.
if [[ "$(uname -s)" == "Darwin" ]]; then DEFAULT_PROFILE=mac_metal; else DEFAULT_PROFILE=cpu_smoke; fi
PROFILE="${GEMMA_WEB_PROFILE:-$DEFAULT_PROFILE}"
PY="${ROOT}/.venv/bin/python"
# 2026-10-03: one log per start (never truncated) + exit record (code/signal/time/tail) in gemma_web_exits.log.
# runs/live_engine/gemma_web.log is a symlink to the current start's log (same path as before for readers).
LOGDIR="${ROOT}/runs/live_engine/gemma_web_logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${LOGDIR}/gemma_web_${STAMP}.log"
EXITS="${ROOT}/runs/live_engine/gemma_web_exits.log"
PIDFILE="${ROOT}/runs/live_engine/gemma_web.pid"

mkdir -p runs/live_engine "$LOGDIR"

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/api/health" 2>/dev/null | grep -q gemma_web; then
  echo "already_up host=${HOST} port=${PORT}"
  curl -fsS --max-time 2 "http://${HOST}:${PORT}/api/health"
  exit 0
fi

# free port if something else holds it
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "stopping listener on :${PORT}"
  PIDS=$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t || true)
  for p in $PIDS; do kill "$p" 2>/dev/null || true; done
  sleep 1
fi

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true
pkill -f 'apps/gemma_web.py' 2>/dev/null || true
sleep 1

# Plan 38 J1: make sure the resident llama-server (tmux gemma_llm, :8091) is up before the web app adopts it
if grep -q "backend: llama_server" "$CONFIG" 2>/dev/null; then
  "$PY" -m genai.llm.server_manager ensure >/dev/null 2>&1 || echo "WARN: llama-server ensure failed (web falls back to llama-cli)" >&2
fi

: > "$LOG"
[[ -L runs/live_engine/gemma_web.log || ! -e runs/live_engine/gemma_web.log ]] || mv runs/live_engine/gemma_web.log "${LOGDIR}/gemma_web_before_${STAMP}.log"
ln -sfn "gemma_web_logs/gemma_web_${STAMP}.log" runs/live_engine/gemma_web.log
# No exec: the wrapper shell records how the app ended (rc>128 => killed by signal rc-128, e.g. 137 = SIGKILL).
tmux new-session -d -s "$SESSION" -e "PATH=$PATH" "cd '$ROOT' && '$PY' -u apps/gemma_web.py --config '$CONFIG' --profile '$PROFILE' --host '$HOST' --port '$PORT' >>'$LOG' 2>&1; rc=\$?; { echo \"\$(date '+%F %T %z') start=${STAMP} exit rc=\$rc signal=\$(( rc > 128 ? rc - 128 : 0 )) log=${LOG#${ROOT}/}\"; tail -5 '$LOG' | sed 's/^/    | /'; } >> '$EXITS'"

ok=0
for i in $(seq 1 180); do  # Plan 38: bge-m3 + whisper + TTS + ack warm-up need up to ~40 s under load
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/health" 2>/dev/null | grep -q gemma_web; then
    ok=1
    break
  fi
  sleep 0.5
done
if [[ "$ok" != 1 ]]; then
  echo "FAIL: gemma_web did not become healthy" >&2
  tail -80 "$LOG" >&2 || true
  exit 1
fi
PID=$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t | head -1)
echo "$PID" > "$PIDFILE"
echo "started session=${SESSION} pid=${PID} url=http://127.0.0.1:${PORT}/"
curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/api/health"
echo
