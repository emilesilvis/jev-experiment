# Classification task comparison

Differences are Jev minus GPT. Brackets show paired 95% bootstrap intervals.

| Dataset | Examples | Jev accuracy | GPT accuracy | Accuracy difference | Jev macro-F1 | GPT macro-F1 | F1 difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| AG News topic classification | 400 | 86.8% | 77.0% | +9.8 pp [+6.5, +13.0] | 0.866 | 0.756 | +11.0 pp [+7.5, +14.6] |
| SNIPS user intent classification | 350 | 96.9% | 90.0% | +6.9 pp [+4.6, +9.1] | 0.969 | 0.924 | +4.5 pp [+2.7, +6.6] |
| TREC coarse question classification | 500 | 91.6% | 77.0% | +14.6 pp [+11.0, +18.2] | 0.910 | 0.788 | +12.2 pp [+7.4, +16.4] |

## Equal-weight summary of these datasets

- accuracy: Jev 91.7%, GPT 81.3%; mean difference +10.4 pp [+8.6, +12.2]. Median task difference +9.8 pp; range +6.9 to +14.6 pp.
- macro f1: Jev 91.5%, GPT 82.3%; mean difference +9.3 pp [+7.1, +11.2]. Median task difference +11.0 pp; range +4.5 to +12.2 pp.

## Operational results

Latency includes all attempts, including retries. Costs are estimates from saved usage and configured rates.

| Dataset | Model | Final failures | Failed attempts | Latency median / p95 (s) | Total cost (USD) |
|---|---|---:|---:|---:|---:|
| AG News topic classification | jev | 0/400 | 0 | 0.600 / 0.768 | $0.007501 |
| AG News topic classification | llm | 2/400 | 2 | 0.984 / 1.286 | $0.019205 |
| SNIPS user intent classification | jev | 1/350 | 1 | 0.605 / 0.767 | $0.007493 |
| SNIPS user intent classification | llm | 22/350 | 22 | 1.042 / 1.443 | $0.023624 |
| TREC coarse question classification | jev | 0/500 | 0 | 0.604 / 0.784 | $0.010803 |
| TREC coarse question classification | llm | 0/500 | 0 | 0.941 / 1.313 | $0.031430 |

## Valid-response diagnostic

These post-hoc accuracies use only examples where both providers returned valid responses. The selected subset excludes failures and does not replace the full end-to-end scores above.

| Dataset | Both valid / all | Jev accuracy on subset | GPT accuracy on subset |
|---|---:|---:|---:|
| AG News topic classification | 398/400 | 86.9% | 77.4% |
| SNIPS user intent classification | 327/350 | 97.2% | 96.3% |
| TREC coarse question classification | 500/500 | 91.6% | 77.0% |

## Paired outcomes

| Dataset | Both correct | Jev only | GPT only | Neither correct |
|---|---:|---:|---:|---:|
| AG News topic classification | 302 | 45 | 6 | 47 |
| SNIPS user intent classification | 314 | 25 | 1 | 10 |
| TREC coarse question classification | 373 | 85 | 12 | 30 |

## Sampling and uncertainty

- AG News topic classification: paired, class-stratified example bootstrap. Class support: Sci/Tech: 100, World: 100, Business: 100, Sports: 100.
- SNIPS user intent classification: paired, class-stratified example bootstrap. Class support: AddToPlaylist: 50, BookRestaurant: 50, GetWeather: 50, PlayMusic: 50, RateBook: 50, SearchCreativeWork: 50, SearchScreeningEvent: 50.
- TREC coarse question classification: paired example bootstrap. Class support: NUM: 113, LOC: 81, HUM: 65, DESC: 138, ENTY: 94, ABBR: 9.

5,000 bootstrap draws; seed 20260918. Conditional on these selected datasets and saved model runs; does not measure variation from dataset choice or repeated model calls.

- Each selected dataset has equal aggregate weight, regardless of sample size.
- Bootstrap draws use the same examples or clusters for both models. Aggregate intervals resample within each fixed dataset independently, then average task differences.
- Class-balanced results describe balanced benchmark samples, not production prevalence.
- Failures are incorrect outcomes; missing outcomes are rejected. Retry attempts contribute to cost and latency; only each example's final outcome contributes to accuracy.
- The primary scores measure end-to-end performance, including valid probability output. The shared-valid-response subset is a post-hoc diagnostic selected by response validity; it cannot establish classification accuracy on failed outputs or replace the full scores.
- Latency median and p95 describe individual attempts, including failed attempts; p95 uses linear interpolation. Concurrent suite load can affect timing.
- Confidence curves are descriptive, include whole tied-score groups, and do not imply calibration. Coverage budgets select the largest attainable coverage below each budget.
- Public benchmarks may overlap model training data. This suite does not establish a universal winner, and the earlier Flutter baseline is not pooled into its aggregate.

Full per-class metrics, identities, confidence curves, and coverage budgets are in `comparison.json`.

Statistical guidance: [Dror et al. (2018), The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing](https://aclanthology.org/P18-1128/).
