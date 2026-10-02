#!/usr/bin/env bash
# Plan 38: run the CPU-side experiment batch sequentially (one at a time to limit contention).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/plan38_cache/venv/bin/python
L=runs/plan38/batch.log
echo "=== batch start $(date '+%F %T') load: $(uptime | sed 's/.*averages: //')" >> $L
for step in "${@:-stt vad embed lip guard}"; do :; done
run() { echo "--- $1 start $(date '+%T') load: $(uptime | sed 's/.*averages: //')" >> $L; shift; "$@" >> $L 2>&1; echo "--- exit $? $(date '+%T')" >> $L; }
for s in ${STEPS:-stt vad embed lip guard}; do
  case $s in
    stt) run stt $PY -u tools/plan38_stt_bench.py ;;
    vad) run vad $PY -u tools/plan38_vad_wake_bench.py vad wake ;;
    embed) run embed $PY -u tools/plan38_embed_bench.py ;;
    lip) mkdir -p runs/plan38/lipsync; say -v Meijia -o /tmp/p38_lip.aiff "好的，Lex。我已經把今天的實驗結果整理好了，存活率比昨天高了百分之十二。" && afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/p38_lip.aiff runs/plan38/lipsync/meijia_sample.wav
         run lip $PY -u tools/plan38_lipsync_warp.py runs/plan38/lipsync/meijia_sample.wav meijia ;;
    guard) EX=""; [ -f runs/plan38/lipsync/warp_meijia.mp4 ] && EX="lipsync_warp=runs/plan38/lipsync/warp_meijia.mp4"
           for f in runs/plan38/lipsync/liveportrait_*.mp4; do [ -f "$f" ] && EX="$EX $(basename $f .mp4)=$f"; done
           run guard $PY -u tools/plan38_appearance_guard.py --extra $EX ;;
    tts) run tts $PY -u tools/plan38_tts_bench.py ;;
  esac
done
echo "=== batch end $(date '+%F %T')" >> $L
