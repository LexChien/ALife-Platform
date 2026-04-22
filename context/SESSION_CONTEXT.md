# SESSION_CONTEXT

## Integrated repository intent
This monorepo now combines:
- platform/core runtime scaffold
- research/asal_engine skeleton
- imported ASAL docs and architecture
- historical ASAL work logs for searchable project continuity

## Where to look first
- `research/asal_engine/README_IMPORTED.md`
- `research/asal_engine/AGENTS.md`
- `research/asal_engine/ASAL_Architecture.mmd`
- `docs/WORKLOG_INDEX.md`
- `research/asal_engine/history/log/`

## Historical continuity rule
Before changing ASAL runtime or repository workflow, search past logs:
```bash
python tools/query_worklog.py jetson
python tools/query_worklog.py cuda openclip
python tools/query_worklog.py runtime summary --limit 10
python tools/query_worklog.py git runs --format json
```

Default search roots now include both:
- `log/`
- `research/asal_engine/history/log/`

## Integration principle
- keep ASAL as research engine
- keep platform `core/` as shared runtime
- move reusable FM/runtime/artifact code upward
- preserve work logs as first-class engineering memory
- **Phase 1 active:** rely on real algorithms (OpenCLIP, ChromaDB, MLflow) rather than stubs.
