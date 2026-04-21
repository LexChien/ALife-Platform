# ASAL Target Search Validation Report

## Hypothesis

Under the same evaluation budget, evolutionary search achieves higher best target similarity than random search on the current reaction-diffusion substrate.

## Setup

- Config: `configs/asal/target_cell_eval.yaml`
- Seeds: `0, 1, 2`
- Trials: `3`
- Metric: `best_score` from `supervised_target_score`
- Baseline: random search with matched evaluation budget

## Results

| Seed | Evolution Best | Random Best | Improvement |
|---|---:|---:|---:|
| 0 | 0.015113 | 0.016795 | -0.001682 |
| 1 | 0.022023 | 0.022722 | -0.000700 |
| 2 | 0.027573 | 0.015131 | +0.012442 |

## Aggregate

- Evolution mean best: `0.021570`
- Random mean best: `0.018216`
- Mean improvement: `+0.003353`
- Median improvement: `-0.000700`
- Win rate: `33.33%`
- Non-negative rate: `33.33%`

## Interpretation

This validates only the current minimal pipeline and reward setup. It does not yet validate a stronger biological-life theory, because the reward model is still a stub foundation model and the substrate/search stack is still a reduced baseline.
