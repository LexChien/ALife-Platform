# ALife-Platform

`ALife-Platform` is a local-first artificial-life and generative-AI integration
repository.

It currently combines:

- `ASAL`: artificial-life search, morphology scoring, narrative phase control,
  and exported organism artifacts
- `Digital Clone`: persona, short-term memory, retrieval, prompt construction,
  and consistency evaluation
- `GenAI`: local LLM integration, multimodal generation scaffolding, and
  evaluation artifacts
- `Gemma Web`: a browser prototype that connects ASAL progress, Digital Clone
  memory, local Gemma inference, and voice interaction into an early artificial
  life console

This repository is not only planning material. It contains runnable CLI
entrypoints, tests, local web UI code, and tracked `runs/` artifacts that show
actual execution state.

## Current State

Working now:

- ASAL run artifacts are available under `runs/asal/`.
- The latest ALife web prototype reads ASAL run summaries and serves organism
  images/GIFs through `/api/life` and `/artifacts/asal/...`.
- Local Gemma inference is wired through `llama.cpp` subprocess integration.
- The Gemma web UI supports text chat, browser microphone capture, browser
  speech playback, visible avatar state, and ASAL progress panels.
- Digital Clone memory is used in the web prototype for short local interaction
  continuity.
- Test coverage exists for LLM cleanup, web service behavior, and ASAL life
  state indexing.

Still incomplete:

- The visible body is currently ASAL organism artifacts, not a finished
  human-like avatar.
- Digital Clone is still a lightweight local memory/persona layer, not a
  production identity system.
- Voice input depends on browser/macOS runtime support.
- The ASAL narrative organism is a validated prototype, not a final artificial
  life model.

## Public Repository Rules

`runs/` artifacts are part of the public project state. If a run is used for
validation, demo behavior, or handoff of current progress, it should be tracked
and pushed.

`docs/` and `log/` are local planning and handoff areas. They are not part of
the public GitHub tracking scope unless the user explicitly names a specific
file as an exception.

Do not publish local-only runtime payloads such as:

- `.venv/`
- `.chroma_db/`
- `models/**/*.gguf`
- `third_party/llama.cpp/` build outputs
- `server.log`
- `__pycache__/` and `.pycache/`

## Repository Layout

```text
apps/                  CLI and local server entrypoints
configs/               YAML configs for ASAL, Digital Clone, and GenAI
context/               Small tracked project context files
core/                  Shared config, runtime, logging, storage, tracking
digital_clone/         Persona, memory, retrieval, prompt, evaluation modules
evaluation/            Shared evaluation helpers
foundation_models/     Foundation model adapters
genai/                 LLM, multimodal, image, voice, and web service modules
model_specs/           Model metadata/specification files
models/                Local model directory skeleton; large weights stay local
research/asal_engine/  ASAL engine, substrates, scoring, visualization
runs/                  Tracked run outputs and validation/demo artifacts
tests/                 Unit and integration-style tests
tools/                 Runtime helpers and local utility scripts
web/gemma_chat/        Browser UI for the ALife/Gemma prototype
```

## Environment

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On the current macOS workspace, `/usr/bin/python3` is often the safer runtime
for the local web server because an existing checked-in `.venv` may not match
the host architecture.

## Run Tests

Targeted validation used by the current web prototype:

```bash
/usr/bin/python3 -m unittest \
  tests.test_life_state \
  tests.test_gemma_web_service \
  tests.test_chat_gemma \
  tests.test_llm_adapter
```

Run the broader test suite:

```bash
python -m unittest discover -s tests
```

Check frontend syntax:

```bash
node --check web/gemma_chat/app.js
```

## ASAL

ASAL is the artificial-life search and artifact layer.

Run a baseline ASAL search:

```bash
python apps/asal_cli.py --config configs/asal/target_cell.yaml
```

Run the narrative cell-fusion profile:

```bash
python apps/asal_cli.py \
  --config configs/asal/target_cell_fusion_narrative.yaml \
  --profile cpu_tiny
```

Run ASAL evaluation:

```bash
python apps/eval_cli.py asal \
  --config configs/asal/target_cell_eval.yaml
```

Important outputs:

- `runs/asal/<run_id>/summary.json`
- `runs/asal/<run_id>/best.gif`
- `runs/asal/<run_id>/best.png`
- `runs/asal/<run_id>/narrative_summary.json`
- `runs/asal/<run_id>/phase_keyframes/*.png`

The current ALife web prototype uses these artifacts as the visible organism
body and project-progress source.

## Digital Clone

Run the baseline clone workflow:

```bash
python apps/clone_cli.py \
  --config configs/clone/clone_baseline.yaml
```

Run clone evaluation:

```bash
python apps/eval_cli.py clone
python tools/run_clone_eval.py \
  --config configs/clone/clone_eval_dataset.yaml
```

Current role in the web prototype:

- stores short local interaction memory
- provides retrieved context to the Gemma request
- does not claim full long-term identity or production-grade memory

## GenAI And Gemma

Run dummy GenAI baseline:

```bash
python apps/genai_cli.py \
  --config configs/genai/genai_baseline.yaml
```

Run local Gemma path:

```bash
python apps/genai_cli.py \
  --config configs/genai/gemma_llama_cpp.yaml \
  --profile cpu_smoke
```

Run direct Gemma chat:

```bash
python tools/chat_gemma.py \
  --config configs/genai/gemma_llama_cpp.yaml \
  --profile cpu_smoke \
  --max-tokens 160 \
  --temperature 0.2 \
  --save-run
```

Local model expectations:

- config path: `configs/genai/gemma_llama_cpp.yaml`
- CLI shim: `tools/llama_completion`
- default local model path: `models/gemma/gemma.gguf`
- `models/gemma/gemma.gguf` is local-only and must not be committed
- `third_party/llama.cpp/` build output is local-only and must not be committed

## ALife Gemma Web Prototype

Start the local server:

```bash
./tools/run_gemma_web \
  --config configs/genai/gemma_llama_cpp.yaml \
  --profile cpu_smoke \
  --host 127.0.0.1 \
  --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

Useful API endpoints:

- `GET /api/health`
- `GET /api/life`
- `POST /api/chat`
- `POST /api/reset`
- `POST /api/transcribe`
- `GET /artifacts/asal/<run_id>/<asset_path>`

The UI currently shows:

- ASAL organism GIF/image artifact
- `birth`, `split`, and `fusion` phase cards
- recent ASAL runs
- pending project tasks from tracked context files
- Gemma chat with ASAL/Digital Clone context
- browser voice controls and speech playback

Web server artifacts are stored under:

```text
runs/chat_gemma_web/<timestamp>/
```

## Runs And Artifacts

Tracked run folders are intentional. They are the evidence trail for what the
system has actually produced.

Current public artifact categories:

- `runs/asal/`: ASAL organism images, GIFs, summaries, narrative scores
- `runs/digital_clone/`: clone evaluation outputs
- `runs/genai/`: GenAI/Gemma outputs and reports
- `runs/chat_gemma/`: direct Gemma chat outputs
- `runs/chat_gemma_web/`: local web server metadata, sessions, and voice
  transcription artifacts

Typical files:

- `summary.json`
- `server_meta.json`
- `best.gif`
- `best.png`
- `narrative_summary.json`
- `trajectory_morphology.json`
- `genai_output.json`
- `report.md`

## Runtime Notes

Config files support:

- `defaults`
- `profiles`
- `active_profile`
- `--profile <name>`
- `ALIFE_PROFILE=<name>`

Config merge order:

1. `defaults`
2. top-level config body
3. selected profile override

The local Gemma web profile currently keeps vector DB disabled for browser
prototype memory so the server can run without requiring a persistent ChromaDB
setup.

## Practical Summary

This repository currently demonstrates an early artificial-life AI shell:

1. ASAL generates and scores organism artifacts.
2. `runs/` stores those artifacts as public validation state.
3. Gemma reads the ASAL progress context.
4. Digital Clone contributes short local memory.
5. The browser UI exposes the combined system through text, voice, and visible
   organism state.

The prototype is usable as a local research console, but it must not be
described as a completed human avatar or finished artificial-life system.
