# GenAI Postprocess Real-Run Report

This report summarizes extractor and fallback behavior on actual saved Gemma runs.

## Aggregate

- Runs analyzed: `3`
- Reasoning visible in final text: `33.33%`
- Runtime leak-detected flag: `0.00%`
- Extractor applied rate: `33.33%`
- Heuristic first-pass rate: `0.00%`
- Heuristic fallback applied rate: `33.33%`
- Heuristic fallback skipped rate: `0.00%`
- Extractor error rate: `0.00%`

## Per Run

| Run | Reasoning Visible | Leak Detected | Heuristic First Pass | Extractor Applied | Fallback Applied | Fallback Skipped | Error |
|---|---:|---:|---:|---:|---:|---:|---|
| 20260422-190853 | no | no | no | no | no | no |  |
| 20260422-191022 | yes | no | no | no | no | no |  |
| 20260423-182527 | no | no | no | yes | yes | no |  |

## Notes

- Older runs may predate postprocess instrumentation; missing flags are treated as false.
- `reasoning_visible_in_text` inspects the final emitted text, not the internal runtime flags.
