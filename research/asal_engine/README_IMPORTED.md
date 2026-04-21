# ASAL-MVP: Automated Search for Artificial Life

ASAL-MVP is a minimal, runnable skeleton inspired by *Automating the Search for Artificial Life with Foundation Models (ASAL)*. It couples a pluggable simulation substrate with evolutionary search strategies and CLIP embeddings to explore visual behaviours.

## Highlights
- **Three optimisation modes**: supervised target matching, open-ended novelty, and illumination diversity (Eq. 2–4 from the paper).
- **Pluggable substrates**: ships with a Boids baseline and a simple API for custom environments (`reset`, `step`, `render`).
- **VLM-backed scoring**: uses OpenCLIP when available; deterministically falls back to random projections when weights are missing.
- **Artifact-first workflow**: runs are archived under `runs/<timestamp>/` with GIF/MP4 captures and summaries for inspection.

## Repository Layout
- `run_asal.py` – CLI entry point orchestrating optimisation loops and artifact export.
- `search/optim.py` – (μ+λ) evolutionary search routine shared by all modes.
- `scores.py` – scoring functions implementing Eq. 2–4.
- `substrates/boids.py` – reference flocking simulator (`reset`, `step`, `render`).
- `fm/clip_embedder.py` – CLIP wrappers and deterministic fallbacks for image/text embeddings.
- `viz/atlas.py` – builds a UMAP "Simulation Atlas" from stored embeddings.
- `ASAL_Architecture.mmd` – Mermaid diagram summarising system flow.

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Python 3.9+ and a GPU-enabled PyTorch install are recommended but not required. If OpenCLIP weights cannot be downloaded (e.g., offline machines), the code switches to seeded random embeddings so experiments remain reproducible.

## Running Experiments
```bash
# Supervised target search (Eq. 2)
python run_asal.py --mode target --prompt "a biological cell" --steps 400 --iters 80 --pop 32

# Open-ended novelty search (Eq. 3)
python run_asal.py --mode openended --steps 600 --iters 60 --pop 32

# Illumination / quality-diversity search (Eq. 4)
python run_asal.py --mode illuminate --steps 400 --iters 80 --pop 128 --keep 64
```
Results are stored in `runs/<timestamp>/` with a `runs/latest` symlink for convenience. Inspect `summary.json`, `best.png`, and GIF/MP4 animations to evaluate outcomes.

## Visualising a Run
```bash
python viz/atlas.py --run runs/latest
```
Generates a UMAP projection of final embeddings for qualitative diversity assessment.

## Extending the Project
1. Copy `substrates/boids.py` as a starting point for new simulators; implement `reset`, `step`, and `render`.
2. Update scoring logic in `scores.py` or add new functions for custom objectives.
3. Document changes and follow the contributor guide in `AGENTS.md` for style, testing, and PR expectations.

## Contributing & Support
- Refer to `AGENTS.md` for coding standards, testing expectations, and PR conventions.
- File issues or start discussions to propose new substrates, scoring strategies, or visualisation ideas.

## License
License information has not been provided yet. Please clarify the intended license before distributing builds or datasets.
