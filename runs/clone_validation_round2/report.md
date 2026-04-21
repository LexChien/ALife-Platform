# Digital Clone Validation Report

## Hypothesis

The current Digital Clone baseline preserves persona tag, tone, required principles, response format, and input echo better than a naive echo baseline across a fixed test set.

## Setup

- Config: `configs/clone/clone_eval_dataset.yaml`
- Cases: `8`
- Metrics: persona tag, tone match, principle coverage, response prefix, input echo, forbidden absence, consistency threshold
- Comparator: naive echo baseline without persona/tone/principle formatting

## Results

| Case | Clone Score | Baseline Score | Improvement | Clone Pass | Baseline Pass |
|---|---:|---:|---:|---:|---:|
| phase0_scope | 1.000 | 0.250 | +0.750 | yes | no |
| memory_planning | 1.000 | 0.375 | +0.625 | yes | no |
| identity_question | 1.000 | 0.375 | +0.625 | yes | no |
| user_intent | 1.000 | 0.375 | +0.625 | yes | no |
| architecture_question | 1.000 | 0.375 | +0.625 | yes | no |
| concise_request | 1.000 | 0.375 | +0.625 | yes | no |
| clone_identity | 1.000 | 0.375 | +0.625 | yes | no |
| eval_question | 1.000 | 0.375 | +0.625 | yes | no |

## Aggregate

- Clone mean score: `1.000`
- Baseline mean score: `0.359`
- Mean improvement: `+0.641`
- Clone pass rate: `100.00%`
- Baseline pass rate: `0.00%`
- Win rate: `100.00%`
- Non-negative rate: `100.00%`

## Interpretation

This provides stronger evidence than self-checking alone because the current clone output is compared against a weaker baseline. It still does not prove long-term memory quality, drift resistance, or open-ended identity coherence because the memory and consistency modules remain stubs.
