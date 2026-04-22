# TASKS

- [ ] Replace `foundation_models/tiny_vlm_stub.py` with real TinyVLM backend
- [x] Import full ASAL `run_asal.py` logic into `research/asal_engine/`
- [ ] Connect `core/artifacts.py` to ASAL engine outputs
- [x] Add substrate registry for Boids / NCA / Lenia / ReactionDiffusion
- [x] Normalize output summary schema across ASAL / Digital Clone / GenAI
- [x] Add log-aware developer workflow using `tools/query_worklog.py`
- [x] Implement Phase 0 Missing Foundational Modules (RuntimeManager, ChromaDB Shared Storage, MLflow Experiment Tracking)
- [x] Upgrade Digital Clone Memory to use ChromaDB Vector Store
- [x] Upgrade ASAL configs to use OpenCLIP instead of random embedder
