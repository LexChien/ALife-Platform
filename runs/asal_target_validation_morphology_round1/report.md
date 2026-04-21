# ASAL Target Search Validation Report

## Hypothesis

Under the same evaluation budget, evolutionary search achieves higher best target similarity than random search on the current reaction-diffusion substrate.

## Setup

- Config: `configs/asal/target_cell_morphology_eval.yaml`
- Seeds: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19`
- Trials: `20`
- Metric: `best_score` from `supervised_target_score`
- Baseline: random search with matched evaluation budget

## Results

| Seed | Evolution Best | Random Best | Improvement |
|---|---:|---:|---:|
| 0 | 0.789499 | 0.788949 | +0.000551 |
| 1 | 0.789488 | 0.789223 | +0.000266 |
| 2 | 0.789725 | 0.788854 | +0.000872 |
| 3 | 0.789060 | 0.788490 | +0.000570 |
| 4 | 0.790626 | 0.788974 | +0.001652 |
| 5 | 0.789968 | 0.789300 | +0.000668 |
| 6 | 0.789605 | 0.789224 | +0.000381 |
| 7 | 0.790090 | 0.789221 | +0.000868 |
| 8 | 0.789587 | 0.788393 | +0.001194 |
| 9 | 0.789589 | 0.788875 | +0.000713 |
| 10 | 0.789682 | 0.789261 | +0.000421 |
| 11 | 0.789969 | 0.788405 | +0.001564 |
| 12 | 0.789814 | 0.789550 | +0.000265 |
| 13 | 0.789174 | 0.789519 | -0.000345 |
| 14 | 0.789079 | 0.788910 | +0.000169 |
| 15 | 0.790889 | 0.788940 | +0.001949 |
| 16 | 0.789295 | 0.789917 | -0.000621 |
| 17 | 0.789065 | 0.789168 | -0.000103 |
| 18 | 0.790393 | 0.789923 | +0.000470 |
| 19 | 0.789978 | 0.789120 | +0.000858 |

## Aggregate

- Evolution mean best: `0.789729`
- Random mean best: `0.789111`
- Mean improvement: `+0.000618`
- Improvement std: `0.000625`
- Median improvement: `+0.000560`
- 95% bootstrap CI of mean improvement: `[+0.000358, +0.000890]`
- Win rate: `85.00%`
- Non-negative rate: `85.00%`

## Interpretation

This validates only the current minimal pipeline and reward setup. It does not yet validate a stronger biological-life theory, because the reward model is still a stub foundation model and the substrate/search stack is still a reduced baseline.
