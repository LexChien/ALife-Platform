#!/bin/zsh
# yaying v6 memory guard: the Mac froze at ~14:20 when two video models ran at once. Every 5 s read the macOS
# free-memory percentage; if it drops below MIN_FREE (default 12) kill v6 generation jobs by PID (never services).
MIN_FREE=${MIN_FREE:-12}
LOG=${LOG:-runs/yaying_clone/demo_v6/memguard.log}
while true; do
  f=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  if [[ -n "$f" && "$f" -lt "$MIN_FREE" ]]; then
    for pid in $(pgrep -f 'tools/yaying_v6_(emv3|wan22|sonic|lipsync)\.py'); do
      echo "$(date '+%F %T') free=${f}% < ${MIN_FREE}% -> kill $pid ($(ps -o command= -p $pid | cut -c1-120))" >> "$LOG"
      kill "$pid"
    done
  fi
  sleep 5
done
