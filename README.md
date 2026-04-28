# ALife-Platform

`ALife-Platform` is a local-first research and integration repository that combines:

- `ASAL`: artificial life search and evaluation
- `Digital Clone`: persona, memory, retrieval, and consistency workflows
- `GenAI`: text/image/voice generation pipelines
- shared platform services for config, runtime, logging, artifacts, and evaluation

The repository is no longer planning-only. It contains runnable CLI entrypoints, evaluation paths, local run artifacts, and a working local `Gemma 4 + llama.cpp` integration path for `GenAI`.

## Current Scope

What works now:

- shared config loading with profile overrides
- shared runtime metadata and run summaries
- runnable CLI entrypoints for `ASAL`, `Digital Clone`, and `GenAI`
- evaluation entrypoint at `apps/eval_cli.py`
- OpenCLIP-based foundation model path for ASAL
- local `Gemma 4 GGUF` execution through `llama.cpp` subprocess integration

What is still incomplete:

- many subsystems are still baseline-quality, not final research-quality
- `ASAL` substrate/search coverage is still narrow
- `Digital Clone` evaluation is still lightweight compared with a full long-term identity system
- `GenAI` image/voice adapters are still dummy paths
- Gemma 4 text output currently works, but output cleaning can still be improved

## Repository Layout

```text
apps/                 Unified CLI entrypoints
configs/              YAML configs for ASAL / Clone / GenAI
core/                 Shared config, runtime, logging, tracking, storage
digital_clone/        Persona, memory, prompt building, evaluation
docs/                 Status, plans, architecture documents
foundation_models/    OpenCLIP and other FM adapters
genai/                Multimodal generation and LLM adapter layer
log/                  Project work logs
models/               Local model directory skeleton
research/asal_engine/ ASAL engine and imported ASAL history
runs/                 Local run outputs and validation artifacts
tests/                Unit and integration-style tests
tools/                Utility scripts such as CUDA / llama.cpp checks
```

## Main Entry Points

Run a subsystem:

```bash
./.venv/bin/python apps/asal_cli.py --config configs/asal/target_cell.yaml
./.venv/bin/python apps/clone_cli.py --config configs/clone/clone_baseline.yaml
./.venv/bin/python apps/genai_cli.py --config configs/genai/genai_baseline.yaml
```

Run evaluation:

```bash
./.venv/bin/python apps/eval_cli.py asal
./.venv/bin/python apps/eval_cli.py clone
./.venv/bin/python apps/eval_cli.py genai
```

Each CLI writes outputs under `runs/` and a normalized `summary.json` for the run.

## Environment Setup

Create and install the local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want to run tests:

```bash
./.venv/bin/python -m unittest discover -s tests
```

## Config Profiles

Configs support:

- `defaults`
- `profiles`
- `active_profile`
- `--profile <name>`
- `ALIFE_PROFILE=<name>`

Example:

```bash
./.venv/bin/python apps/genai_cli.py \
  --config configs/genai/gemma_llama_cpp.yaml \
  --profile cpu_smoke
```

The loader merges:

1. `defaults`
2. top-level config body
3. selected profile override

## ASAL

Primary files:

- [apps/asal_cli.py](/home/lexchien/Documents/ALife-Platform/apps/asal_cli.py)
- [research/asal_engine/engine.py](/home/lexchien/Documents/ALife-Platform/research/asal_engine/engine.py)
- [configs/asal/target_cell.yaml](/home/lexchien/Documents/ALife-Platform/configs/asal/target_cell.yaml)
- [configs/asal/target_cell_eval.yaml](/home/lexchien/Documents/ALife-Platform/configs/asal/target_cell_eval.yaml)

Run baseline search:

```bash
./.venv/bin/python apps/asal_cli.py --config configs/asal/target_cell.yaml
```

Run evaluation:

```bash
./.venv/bin/python apps/eval_cli.py asal \
  --config configs/asal/target_cell_eval.yaml
```

Jetson note:

- on `aarch64`, `apps/asal_cli.py` can re-exec into a known-good interpreter when the current Python cannot see CUDA
- this is meant to avoid accidentally using an incompatible system PyTorch wheel

## Digital Clone

Primary files:

- [apps/clone_cli.py](/home/lexchien/Documents/ALife-Platform/apps/clone_cli.py)
- [digital_clone/engine.py](/home/lexchien/Documents/ALife-Platform/digital_clone/engine.py)
- [digital_clone/memory/store.py](/home/lexchien/Documents/ALife-Platform/digital_clone/memory/store.py)
- [digital_clone/decision/policy.py](/home/lexchien/Documents/ALife-Platform/digital_clone/decision/policy.py)
- [configs/clone/clone_baseline.yaml](/home/lexchien/Documents/ALife-Platform/configs/clone/clone_baseline.yaml)
- [configs/clone/clone_gemma_llama_cpp.yaml](/home/lexchien/Documents/ALife-Platform/configs/clone/clone_gemma_llama_cpp.yaml)

Run baseline:

```bash
./.venv/bin/python apps/clone_cli.py \
  --config configs/clone/clone_baseline.yaml
```

Run evaluation:

```bash
./.venv/bin/python apps/eval_cli.py clone
./.venv/bin/python tools/run_clone_eval.py \
  --config configs/clone/clone_eval_dataset.yaml
```

Current state:

- retrieval and prompt construction exist
- shared LLM adapter integration exists
- evaluation is usable, but still not equivalent to a production-grade long-term identity system

## GenAI

Primary files:

- [apps/genai_cli.py](/home/lexchien/Documents/ALife-Platform/apps/genai_cli.py)
- [genai/multimodal/engine.py](/home/lexchien/Documents/ALife-Platform/genai/multimodal/engine.py)
- [genai/llm/adapter.py](/home/lexchien/Documents/ALife-Platform/genai/llm/adapter.py)
- [genai/llm/backends/llama_cpp.py](/home/lexchien/Documents/ALife-Platform/genai/llm/backends/llama_cpp.py)
- [configs/genai/genai_baseline.yaml](/home/lexchien/Documents/ALife-Platform/configs/genai/genai_baseline.yaml)
- [configs/genai/gemma_llama_cpp.yaml](/home/lexchien/Documents/ALife-Platform/configs/genai/gemma_llama_cpp.yaml)

Run dummy baseline:

```bash
./.venv/bin/python apps/genai_cli.py \
  --config configs/genai/genai_baseline.yaml
```

Run local Gemma 4 path:

```bash
./.venv/bin/python apps/genai_cli.py \
  --config configs/genai/gemma_llama_cpp.yaml \
  --profile cpu_smoke
```

Run GenAI evaluation:

```bash
./.venv/bin/python apps/eval_cli.py genai
./.venv/bin/python tools/run_genai_eval.py \
  --config configs/genai/genai_eval_dataset.yaml
```

## Local Gemma 4 + llama.cpp

The repository now supports a local `Gemma 4 GGUF` path through `llama.cpp`.

Current runtime mode:

- backend: `llama_cpp`
- driver: `subprocess`
- CLI: `third_party/llama.cpp/build/bin/llama-completion`
- model path: `models/gemma/gemma.gguf`

Important:

- the repository tracks the `models/` directory skeleton only
- the actual `.gguf` file is local-only
- `third_party/llama.cpp/` is also local-only

Check the runtime:

```bash
./.venv/bin/python tools/check_llama_cpp.py \
  --config configs/genai/gemma_llama_cpp.yaml
```

Expected success indicators:

- `exists: True`
- `cli_available: True`
- `driver: subprocess`
- `ok: True`

Run a direct single-turn Gemma chat without image/audio generation:

```bash
./.venv/bin/python tools/chat_gemma.py \
  --config configs/genai/gemma_llama_cpp.yaml \
  --profile cpu_smoke \
  --prompt "你是誰？請用繁體中文兩句話介紹你自己。" \
  --context "不要列點，不要輸出思考過程。" \
  --max-tokens 96 \
  --temperature 0.2 \
  --save-run
```

Run a local web chat UI for Gemma with text input, browser microphone input,
and browser speech playback:

```bash
./tools/run_gemma_web \
  --config configs/genai/gemma_llama_cpp.yaml \
  --profile cpu_smoke \
  --host 127.0.0.1 \
  --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

Notes:

- the backend still uses the local `Gemma 4 + llama.cpp` path already configured in the repo
- `configs/genai/gemma_llama_cpp.yaml` now uses `tools/llama_completion`, which auto-selects a usable local `llama.cpp` binary path
- on this macOS workspace, use `python3` instead of `./.venv/bin/python` because the checked-in `.venv` is a Linux ARM environment and cannot execute locally
- if your active `python3` cannot import `yaml`, use `./tools/run_gemma_web ...`, which auto-selects a usable interpreter with `PyYAML`
- microphone input and spoken reply use the browser Web Speech API
- microphone input now performs a browser permission preflight; the first click should trigger a microphone permission prompt
- if voice input still fails, confirm you are on `localhost`, using Chrome/Edge, and that microphone access is allowed for the page
- the web runtime now exposes config-driven `voice` and `avatar` sections from `configs/genai/gemma_llama_cpp.yaml`
- the UI now includes avatar state display and a voice auto-submit toggle
- browser mic support works best in Chromium-based browsers and requires microphone permission
- web chat transcripts are stored under `runs/chat_gemma_web/...`

Analyze archived real Gemma runs for extractor/fallback hit rates:

```bash
./.venv/bin/python tools/report_genai_postprocess.py \
  runs/genai/20260422-190853 \
  runs/genai/20260422-191022 \
  runs/genai/20260423-182527 \
  --outdir runs/genai/postprocess_report
```

## CUDA / OpenCLIP Verification

Check the CUDA environment and OpenCLIP device placement:

```bash
./.venv/bin/python tools/check_cuda.py
```

This verifies:

- Python executable
- PyTorch import
- CUDA availability
- CUDA tensor execution
- OpenCLIP adapter device
- OpenCLIP model parameter device

## Runs And Artifacts

The repository stores run outputs under `runs/`, for example:

- `runs/asal/...`
- `runs/digital_clone/...`
- `runs/genai/...`

Typical files include:

- `summary.json`
- generated images
- evaluation reports
- subsystem output payloads such as `genai_output.json`

## Logs And Historical Notes

Project work logs are under:

- [log/](/home/lexchien/Documents/ALife-Platform/log)

Imported historical ASAL records remain under:

- [research/asal_engine/history/](/home/lexchien/Documents/ALife-Platform/research/asal_engine/history)

Useful references:

- [docs/STATUS.md](/home/lexchien/Documents/ALife-Platform/docs/STATUS.md)
- [log/2026-04-22/llm_phase_a_execution.md](/home/lexchien/Documents/ALife-Platform/log/2026-04-22/llm_phase_a_execution.md)
- [log/2026-04-22/openclip_device_auto_fix.md](/home/lexchien/Documents/ALife-Platform/log/2026-04-22/openclip_device_auto_fix.md)

## Practical Status

Best short summary:

- platform wiring is real
- CLI entrypoints are real
- evaluation entrypoints are real
- local Gemma 4 inference is real
- many research and product layers are still baseline-grade

If you want the current detailed project state, start with:

- [docs/STATUS.md](/home/lexchien/Documents/ALife-Platform/docs/STATUS.md)
