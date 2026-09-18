# Multi-task classification comparison

Protocol fixed before the new benchmark predictions. This is a small,
descriptive comparison of `jev-1.13.0` and `gpt-4o-mini-2024-07-18` on selected
public English classification datasets. It does not establish a universal
ranking or production accuracy.

## Tasks and samples

| Dataset | Task | Planned sample | Sampling |
| --- | --- | --- | --- |
| AG News | Four news topics | 400 | Seeded 100 per class from the test split |
| TREC | Six expected answer types | 500 | Full official test split, natural class frequencies |
| SNIPS | Seven user intents | 350 | Seeded 50 per intent from the held-out split |

The snapshots are authoritative for the final retained counts and contain
source URLs/revisions, content hashes, original split, exact duplicate handling,
sampling seed, and label descriptions. Duplicate removal or insufficient source
examples must be reported before prediction, never silently replaced by training
data. Model inputs contain text only. The Flutter-specific sanitizer is not used.

Use the original training/development material only if an adapter or taxonomy
needs clarification; do not tune prompts against benchmark predictions. The
task instructions and descriptions are identical across providers, frozen in
each dataset and copied into each run's identity. There is no few-shot training,
prompt search, or confidence-threshold tuning on these test outcomes.

The existing Flutter run influenced this experiment's design and is preserved
separately. It is not included in the three-task aggregate.

## Execution and reproducibility

- Same text and candidate descriptions for both providers on every example.
- Pinned model versions; OpenAI temperature zero and strict structured output.
- One initial call per provider per example. No hidden retries. A failure stops
  that dataset, preserves its attempt, and can be resumed explicitly.
- Provider order alternates by example. At most three datasets run concurrently;
  concurrency is recorded. Timing reflects this execution environment.
- Dataset bytes, provider settings, task rubric, and inference code hashes are
  checked before resuming. Preserve snapshots and raw predictions.
- Both predicted labels and complete probability distributions are saved.
  Invalid responses are recorded as failures, not converted into guesses.

The plan is 1,250 texts and 2,500 initial model calls. Repeated calls measure
response variability and must never be treated as new independent examples.
This first study uses one completed prediction per model per text.

## Analysis fixed before outcomes

Report each task first: accuracy, macro-F1, class supports and errors, paired
correctness counts (both correct, only Jev correct, only GPT correct, neither),
failures, median/p95 request latency, and estimated token cost.

For accuracy and macro-F1 differences, subtract GPT from Jev. Compute 95%
percentile intervals using 5,000 paired bootstrap samples with a fixed seed:
resample the same examples for both models. Preserve class counts for the
class-balanced samples; use ordinary paired resampling for the naturally
distributed TREC sample. Where group identifiers identify related inputs,
resample whole groups. Record the actual method in the report.

These intervals describe sampling uncertainty under the resampling assumptions,
conditional on the saved model outputs. They do not capture model-training
variation, all API nondeterminism, label errors, training-data contamination,
or performance on tasks absent from this suite. TREC's rare classes can have
particularly uncertain F1 estimates.

Give each task equal weight in the overall mean. Also show individual task
differences and their range; do not let the largest dataset dominate. With only
three deliberately selected tasks, do not present a pooled significance test
or confidence claim about all possible classification tasks.

Confidence analysis is descriptive. Compare accuracy-versus-coverage curves,
retaining whole groups of tied confidence values. Equal numeric confidence
thresholds are not equivalent between providers. Selecting an operating
threshold for deployment would require separate development data.

## Known limits and references

Public benchmarks may have appeared in either model's training data. Exact
deduplication does not remove all paraphrases or articles about the same event;
residual dependence can make item-bootstrap intervals too narrow. Balanced
samples describe balanced benchmark performance, not natural production traffic.
Cost estimates use configured token prices and omit cache discounts or other
billing adjustments. Latency includes network and service conditions.

- [AG News classification benchmark paper](https://arxiv.org/abs/1509.01626)
- [TREC taxonomy and source datasets](https://cogcomp.seas.upenn.edu/Data/QA/QC/)
- [SNIPS author-maintained benchmark](https://github.com/sonos/nlu-benchmark)
- [Paired evaluation and statistical testing in NLP](https://aclanthology.org/P18-1128/)
