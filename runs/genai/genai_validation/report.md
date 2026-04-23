# GenAI Validation Report

## Hypothesis

The current GenAI baseline preserves prompt/context formatting and emits multimodal artifacts more consistently than a naive text-only baseline.

## Setup

- Config: `configs/genai/genai_eval_dataset.yaml`
- Cases: `2`
- Metrics: expected prefix, prompt reflection, context reflection, required substrings, reasoning leakage absence, audio presence, image presence
- Comparator: naive baseline without multimodal artifact generation

## Results

| Case | GenAI Score | Baseline Score | Improvement | GenAI Pass | Baseline Pass |
|---|---:|---:|---:|---:|---:|
| contextual_generation | 1.000 | 0.571 | +0.429 | yes | no |
| prompt_only_generation | 1.000 | 0.571 | +0.429 | yes | no |

## Aggregate

- GenAI mean score: `1.000`
- Baseline mean score: `0.571`
- Mean improvement: `+0.429`
- GenAI pass rate: `100.00%`
- Baseline pass rate: `0.00%`
- Win rate: `100.00%`
- Non-negative rate: `100.00%`
- Reasoning leak rate: `0.00%`
- Extractor applied rate: `0.00%`
- Heuristic fallback rate: `0.00%`
- Heuristic first-pass rate: `0.00%`
- Extractor error rate: `0.00%`

## Postprocess Analysis

This section isolates the reasoning-leak handling path so the team can tell whether a poor answer came from the model itself or from aggressive cleanup.

- Reasoning leak detection hit rate: `0.00%`
- Extractor second-pass hit rate: `0.00%`
- Heuristic fallback applied rate: `0.00%`
- Heuristic first-pass rescue rate: `0.00%`
- Heuristic fallback skipped rate: `0.00%`
- Extractor error rate: `0.00%`

| Case | Leak Detected | Heuristic First Pass | Extractor Applied | Fallback Applied | Fallback Skipped | Extractor Error |
|---|---:|---:|---:|---:|---:|---|
| contextual_generation | no | no | no | no | no |  |
| prompt_only_generation | no | no | no | no | no |  |

## Interpretation

This is still a baseline validation for formatting and artifact existence rather than generation quality. It is sufficient to close the minimal evaluation-layer gap for the GenAI path.
