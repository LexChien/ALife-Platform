# ASAL Target Search Validation Report

## Hypothesis

Under the same evaluation budget, evolutionary search achieves higher best target similarity than random search on the current reaction-diffusion substrate.

## Setup

- Config: `configs/asal/target_cell_eval.yaml`
- Seeds: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19`
- Trials: `20`
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
| 8 | 0.024207 | 0.027531 | -0.003324 |
| 9 | 0.030244 | 0.026053 | +0.004191 |
| 10 | 0.018899 | 0.017814 | +0.001085 |
| 11 | 0.018901 | 0.026426 | -0.007525 |
| 12 | 0.018750 | 0.021879 | -0.003129 |
| 13 | 0.025514 | 0.020588 | +0.004926 |
| 14 | 0.017398 | 0.017900 | -0.000502 |
| 15 | 0.018808 | 0.019046 | -0.000238 |
| 16 | 0.019751 | 0.017565 | +0.002186 |
| 17 | 0.022416 | 0.018727 | +0.003689 |
| 18 | 0.021420 | 0.022269 | -0.000849 |
| 19 | 0.025577 | 0.019068 | +0.006510 |

## Aggregate

- Evolution mean best: `0.022002`
- Random mean best: `0.020578`
- Mean improvement: `+0.001424`
- Improvement std: `0.004475`
- Median improvement: `+0.000546`
- 95% bootstrap CI of mean improvement: `[-0.000466, +0.003410]`
- Win rate: `55.00%`
- Non-negative rate: `55.00%`

## Interpretation

This validates only the current minimal pipeline and reward setup. It does not yet validate a stronger biological-life theory, because the reward model is still a stub foundation model and the substrate/search stack is still a reduced baseline.
