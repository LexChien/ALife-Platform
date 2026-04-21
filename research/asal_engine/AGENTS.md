# Repository Guidelines

## Project Structure & Module Organization
`run_asal.py` is the main entry point that wires together optimisation loops, CLIP embeddings, and artifact persistence. Core modules sit in `search/optim.py` (μ+λ evolutionary search), `scores.py` (fitness functions for target, open-ended, and illumination runs), and `substrates/boids.py` (baseline simulator implementing `reset`, `step`, `render`). Embedding helpers live under `fm/`, visual diagnostics under `viz/atlas.py`, and the Mermaid overview in `ASAL_Architecture.mmd`. Experiment outputs land in `runs/<timestamp>/` with a `runs/latest` symlink—prune large GIF/MP4s before pushing.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment aligned with `requirements.txt`.
- `pip install -r requirements.txt`: pull CLIP, NumPy, UMAP, and plotting dependencies.
- `python run_asal.py --mode target --prompt "a biological cell" --steps 400 --iters 80 --pop 32`: baseline supervised search; swap `--mode` to `openended` or `illuminate` for other objectives.
- `python viz/atlas.py --run runs/latest`: build the UMAP atlas for the newest run to sanity-check novelty and diversity.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, `snake_case` symbols, and module-level constants in ALL_CAPS. Keep functions side-effect free when possible and thread randomness through explicit `seed` arguments. Document public methods with short docstrings, mirror the substrate API when adding new simulators, and sort imports as standard library, third-party, then local. Run `black` or `ruff` locally if added to your toolchain, but do not commit their configs without consensus.

## Testing Guidelines
No automated suite exists yet; vet changes by executing each affected CLI mode and inspecting `runs/<timestamp>/summary.json`, `best.png`, and generated animations. When introducing deterministic utilities, add `pytest` modules under `tests/` (e.g. `tests/test_scores.py`) and assert against fixed seeds so reviewers can reproduce locally. Summarise manual test commands and observed best scores in your PR body.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects under ~50 characters following a `area: summary` pattern (e.g. `substrates: add lenia prototype`), with focused diffs per commit. Pull requests should outline purpose, behavioural changes, reproduction commands, and attach representative artifacts or screenshots/GIFs. Reference issues where applicable, confirm lint/tests executed, and exclude bulky `runs/` outputs unless deliberately sharing exemplars.

## Security & Configuration Tips
Avoid embedding credentials or API tokens; prefer environment variables and `.env` (gitignored). CLIP weights may be unavailable in restricted environments—note the fallback to deterministic random embeddings so reviewers understand metric fidelity. Archive or clean sizeable media in `runs/` before pushing to keep history lean.
