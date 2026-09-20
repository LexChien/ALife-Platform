#!/usr/bin/env bash
# Start the ORIGINAL ALife Prototype (gemma_web UI + viz_life_history CLI).
# Version: 2026-09-20-thinking — latest binds active run; records thinking pulses
# Version: 2026-09-20 — gemma_web :8080, demo_latest CLI tip, kills fake prototype
set -euo pipefail

detect_root() {
  if [[ -n "${ALIFE_ROOT:-}" && -d "${ALIFE_ROOT}" ]]; then
    cd "${ALIFE_ROOT}" && pwd; return 0
  fi
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${here}/apps/gemma_web.py" ]]; then echo "${here}"; return 0; fi
  if [[ -f "${here}/../apps/gemma_web.py" ]]; then cd "${here}/.." && pwd; return 0; fi
  case "$(uname -s)" in
    Darwin) for c in "$HOME/Documents/AI/ALife-Platform" "$HOME/Documents/ALife-Platform"; do
      [[ -f "$c/apps/gemma_web.py" ]] && { echo "$c"; return 0; }; done ;;
    Linux) for c in "$HOME/Documents/ALife-Platform" "$HOME/Documents/AI/ALife-Platform"; do
      [[ -f "$c/apps/gemma_web.py" ]] && { echo "$c"; return 0; }; done ;;
  esac
  echo "ERROR: cannot find ALife-Platform with apps/gemma_web.py" >&2
  exit 1
}

ROOT="$(detect_root)"
cd "${ROOT}"
mkdir -p runs/live_engine

echo "=== ALife ORIGINAL stack ==="
echo "os=$(uname -s) root=${ROOT}"

# Free fake prototype if occupying 8080
if lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then
  if curl -sS http://127.0.0.1:8080/api/health 2>/dev/null | grep -q gemma_web; then
    echo "gemma_web already on :8080"
  else
    echo "stopping non-gemma process on :8080"
    # best-effort
    pkill -f 'apps.prototype_web' 2>/dev/null || true
    pkill -f 'uvicorn apps.prototype_web' 2>/dev/null || true
    sleep 1
  fi
fi

if ! curl -sS http://127.0.0.1:8080/api/health 2>/dev/null | grep -q gemma_web; then
  nohup ./tools/run_gemma_web \
    --config configs/genai/gemma_llama_cpp.yaml \
    --profile cpu_smoke \
    --host 127.0.0.1 \
    --port 8080 \
    >runs/live_engine/gemma_web.log 2>&1 &
  echo "gemma_web started pid=$!"
  disown $! 2>/dev/null || true
  sleep 2
fi

# Point latest -> ACTIVE gemma run (so CLI records live thinking), not frozen demo.
ACTIVE_RUN=""
if curl -sS http://127.0.0.1:8080/api/health 2>/dev/null | grep -q gemma_web; then
  ACTIVE_RUN="$(curl -sS http://127.0.0.1:8080/api/health 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin).get("run_dir",""))' 2>/dev/null || true)"
fi
if [[ -n "${ACTIVE_RUN}" && -f "${ACTIVE_RUN}/live_engine/live_evolution.json" ]]; then
  RUN_ID="$(basename "${ACTIVE_RUN}")"
  ln -sfn "${RUN_ID}" runs/chat_gemma_web/latest
  LIVE="${ROOT}/runs/chat_gemma_web/latest/live_engine/live_evolution.json"
else
  LATEST_JSON="$(ls -1dt runs/chat_gemma_web/*/live_engine/live_evolution.json 2>/dev/null | head -1 || true)"
  if [[ -n "${LATEST_JSON}" ]]; then
    RUN_ID="$(basename "$(dirname "$(dirname "${LATEST_JSON}")")")"
    # skip symlink names
    if [[ "${RUN_ID}" == "latest" || "${RUN_ID}" == "demo_latest" ]]; then
      RUN_ID="$(basename "$(dirname "$(dirname "$(readlink -f "${LATEST_JSON}" 2>/dev/null || echo "${LATEST_JSON}")")")")"
    fi
    ln -sfn "${RUN_ID}" runs/chat_gemma_web/latest
    LIVE="${ROOT}/runs/chat_gemma_web/latest/live_engine/live_evolution.json"
  else
    LIVE="${ROOT}/runs/live_engine/live_evolution.json"
  fi
fi

echo "=== ready ==="
echo "Website : http://127.0.0.1:8080/"
echo "Health  : http://127.0.0.1:8080/api/health"
echo "CLI log :"
echo "  ./.venv/bin/python tools/viz_life_history.py -f ${LIVE}"
echo "  # -f = follow/watch mode (original CLI)"
curl -sS -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8080/ || true
