# DECISIONS

- ASAL is preserved as a research engine under `research/asal_engine/`
- platform runtime remains in `core/`
- imported historical logs are kept under `research/asal_engine/history/log/`
- searchable work log index is generated into `docs/WORKLOG_INDEX.md`
- no historical log files are modified; only indexed
- Owned-model strategy: Focus strictly on LoRA / SFT specialization (using PEFT/QLoRA) for specialized personas, planners, and judges instead of pre-training base models from scratch.
