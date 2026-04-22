# alife-platform

This is a practical integration prototype for `alife-platform` that combines:

- `research/asal_engine/` for ASAL-style artificial life search
- `core/` for shared runtime, config, artifacts, registry, logging
- `foundation_models/` for reusable FM backends
- `digital_clone/` for identity / memory / consistency baseline
- `genai/` for generation adapters
- `apps/` as unified CLI entrypoints

## Goals

1. Preserve ASAL as a research engine
2. Move common runtime concerns into platform core
3. Create a future-ready structure for:
   - Digital Clone
   - Artificial Life / ASAL
   - Generative AI

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python apps/asal_cli.py --config configs/asal/target_cell.yaml
python apps/clone_cli.py --config configs/clone/clone_baseline.yaml
python apps/genai_cli.py --config configs/genai/genai_baseline.yaml
```

On Jetson-class hosts, `apps/asal_cli.py` will automatically re-exec into
`ALIFE_ASAL_PYTHON` or `/home/lexchien/Documents/ASAL/.venv/bin/python` when
the current interpreter cannot see CUDA. This avoids accidentally running ASAL
with an incompatible desktop PyTorch wheel such as `torch 2.11 + cu130`.

## Status

This repository is a Phase 0 / Phase 1 prototype:
- Core runtime works
- ASAL engine skeleton works
- Foundation model registry works
- Digital clone baseline works
- GenAI baseline works

What still needs real implementation:
- Replace random/tiny stubs with actual OpenCLIP / TinyVLM / GGUF
- Import full ASAL logic into `research/asal_engine/`
- Add trained NCA / Lenia / MAP-Elites
- Add long-term memory store and real consistency evaluation

## Integrated ASAL history

This prototype now includes imported ASAL project history:
- `research/asal_engine/AGENTS.md`
- `research/asal_engine/README_IMPORTED.md`
- `research/asal_engine/ASAL_Architecture.mmd`
- `research/asal_engine/history/log/`
- `docs/WORKLOG_INDEX.md`
- `tools/query_worklog.py`

Search historical notes:
```bash
python tools/query_worklog.py jetson
python tools/query_worklog.py cuda openclip
python tools/query_worklog.py runtime summary --limit 10
```

The tool searches both root project logs under `log/` and imported ASAL
historical logs under `research/asal_engine/history/log/`.
