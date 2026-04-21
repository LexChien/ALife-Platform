# ASAL Target Search Validation Report

## Hypothesis

Under the same evaluation budget, evolutionary search achieves higher best target similarity than random search on the current reaction-diffusion substrate.

## Setup

- Config: `configs/asal/target_cell_eval.yaml`
- Seeds: `0, 1, 2, 3, 4, 5, 6, 7`
- Trials: `8`
- Metric: `best_score` from `supervised_target_score`
- Baseline: random search with matched evaluation budget

## Results

| Seed | Evolution Best | Random Best | Improvement |
|---|---:|---:|---:|
| 0 | 0.015113 | 0.016795 | -0.001682 |
| 1 | 0.022023 | 0.022722 | -0.000700 |
| 2 | 0.027573 | 0.015131 | +0.012442 |
| 3 | 0.025147 | 0.024915 | +0.000232 |
| 4 | 0.016209 | 0.015348 | +0.000861 |
| 5 | 0.017539 | 0.020444 | -0.002905 |
| 6 | 0.030632 | 0.022588 | +0.008044 |
| 7 | 0.023921 | 0.018755 | +0.005166 |

## Aggregate

- Evolution mean best: `0.022270`
- Random mean best: `0.019587`
- Mean improvement: `+0.002682`
- Median improvement: `+0.000546`
- Win rate: `62.50%`
- Non-negative rate: `62.50%`

## Interpretation

This validates only the current minimal pipeline and reward setup. It does not yet validate a stronger biological-life theory, because the reward model is still a stub foundation model and the substrate/search stack is still a reduced baseline.
