# Jev vs an LLM: classification experiments

A small learning experiment: fetch 200 real GitHub issues (50 per team), ask
Jev and an OpenAI model who owns each issue, then score their answers. The
multi-task extension compares news topics, question types, and user intents.
Plain Python, one dependency (`requests`), no training or framework.

Read [the plain-language findings report](REPORT.md) for a blog-ready account of
the results, their limitations, and an explanatory image comparing Jev's
documented decision interface with typical LLM text generation.

## Compare different classification tasks

The benchmark suite uses AG News (400 test articles, 100 per class), TREC
(its 500-question coarse-label test set), and SNIPS (350 held-out requests,
50 per intent). Each frozen snapshot includes its task instructions, label
descriptions, sampling rules, and source provenance. The existing Flutter
results remain a separate exploratory baseline.

After setting up the environment and API keys as described below:

```sh
source .venv/bin/activate
set -a
source .env
set +a

python benchmark_data.py
python run_benchmarks.py --workers 3
python compare.py
```

The first command downloads public datasets and freezes reproducible samples
under `data-benchmarks/`. The runner makes 2,500 billable API calls on the first
complete run, saving each response under `results-benchmarks/<dataset>/`.
It can run three datasets concurrently, with sequential calls inside each
dataset and alternating provider order. Invalid completed model responses count
as incorrect and the dataset continues. Request errors stop that dataset; the
others can finish. Rerun the same command after resolving a request error to
reuse saved outcomes, including final invalid responses. Use the same worker
count when resuming. A file lock
prevents two processes writing predictions into the same results directory.

`compare.py` needs no API keys or network. It writes
`results-benchmarks/comparison.json` and `comparison.md`, containing scores,
paired uncertainty intervals, latency/cost summaries, and an equally weighted
summary across tasks. Its defaults use 5,000 bootstrap samples and a fixed
random seed. The original Flutter benchmark is excluded from this aggregate.

The partial initial attempt is archived in `results-benchmarks-attempt1/`.
It stopped on malformed probability responses. The final suite starts fresh
under the failure-counting policy documented in the protocol; no responses
from the initial attempt are reused in its statistics.

Read [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) for the fixed study design,
what the uncertainty intervals mean, and the limitations of this comparison.
On macOS/Linux, the runner uses the standard-library `fcntl` file lock.

## Run it

The original Flutter ownership experiment:

Requires Python 3.10+ and API keys for [TypeSafe/Jev](https://typesafe.ai) and
[OpenAI](https://platform.openai.com/api-keys). GitHub authentication is optional
for this public repository, but helps if you hit its unauthenticated rate limit.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your keys. `JEV_API_KEY` is your TypeSafe API key. Then:

```sh
set -a
source .env
set +a

python fetch_issues.py
python predict.py
python evaluate.py
```

If `data/issues.json` is already present, skip the fetch step to reuse that snapshot.

`.env` is loaded by your shell, not automatically by Python. Prediction makes
400 sequential, billable API calls at the default dataset size. There are no
automatic retries. A failed call is saved and stops the run; fix the cause (or
wait after a rate limit) and rerun `python predict.py` to resume. Successful
predictions are reused. Do not run two prediction processes in the same folder.

For a tiny first run, set `ISSUES_PER_TEAM=2`, `DATA_DIR=data-small`, and
`RESULTS_DIR=results-small` before fetching. That produces eight issues and
16 model calls. Use the same directory settings for all three commands.

## Read the code in this order

| File | What to learn |
| --- | --- |
| `fetch_issues.py` | GitHub pagination, balanced selection, input cleaning |
| `providers.py` | Two explicit HTTP requests and response validation |
| `predict.py` | Timing, saving predictions, resuming an interrupted run |
| `evaluate.py` | Accuracy, macro-F1, confusion matrices, confidence filtering |
| `common.py` | Shared team descriptions and small HTTP/JSON helpers |

## What is being compared?

The four possible answers are `team-framework`, `team-tool`, `team-ios`, and
`team-android`. An issue qualifies if it has **exactly one of these four labels**;
other labels are allowed. Pull requests are excluded. Open and closed issues
are included. Selection takes the newest-created qualifying issues per team
within an inclusive UTC creation-date window, by default the last 180 days.
A shortage fails explicitly; widen `SINCE_DATE` or lower `ISSUES_PER_TEAM`.

We fetch the issue title, issue body, and labels through GitHub's REST API.
"Description" means the issue's own body, never its comments or discussion.
The API returns the body's **current revision**, which can have been edited
since the issue was opened; this does not recover the first-ever revision.

Cleaning removes HTML comments, URLs, issue references, team label tokens,
explicit label/ownership metadata, Flutter team mentions, and common structured
label tokens. It keeps useful bug vocabulary such as Android, iOS, widgets,
and build errors. These fixed rules do not depend on the answer label. The
cleaned title and description are joined and truncated to `MAX_INPUT_CHARS`
(12,000 by default); **exactly this same string goes to both providers**.
Labels, issue numbers, timestamps, and other saved metadata are never passed
to the classifiers. Both receive the same candidate teams and descriptions.

The defaults are pinned model versions: `jev-1.13.0` and
`gpt-4o-mini-2024-07-18`. You can change `JEV_MODEL` and `LLM_MODEL` in `.env`.
The LLM must support Chat Completions, strict JSON-schema outputs, and
`temperature=0`. Its answer is a probability per team; the largest wins (ties
use the order above). Jev returns a choice and distribution directly.

## Files and reproducibility

- `data/issues.json`: original title/body/labels, target, cleaned `model_input`,
  creation/update times, selection window, and cleaning settings.
- `results/predictions.json`: dataset/code hashes, prompts, model settings, and
  every attempt's prediction, probabilities, confidence, latency, token usage,
  optional estimated USD cost, response/model identifiers, or error.
- `results/report.json`: all metrics, including per-team scores and threshold
  accuracy/coverage. `evaluate.py` also prints the main tables.

The fetcher refuses to overwrite a snapshot. Keep `issues.json` to reproduce
the exact inputs: GitHub labels/bodies can change, so refetching the same date
range is not sufficient. The predictor refuses to mix different datasets,
code, or model/price settings in one results file. Choose a new `RESULTS_DIR`
when comparing another model or prompt. The private repository preserves the
frozen datasets, predictions, reports, and archived initial benchmark attempt.
Credentials, local environments, download caches, draft datasets, locks, and
small smoke-test runs are excluded. API outputs can still vary across runs even
with pinned models and temperature zero. Scoring saved results is deterministic
and requires no API credentials or network.

## How to read the results

Accuracy is correct answers divided by **all sampled issues**. Macro-F1 is the
mean of the four per-team F1 scores. Missing and failed predictions count as
incorrect and appear in an extra confusion-matrix column; they are also counted
separately. `accuracy_on_predictions` excludes them and is reported separately.
After a retry, scoring uses the latest attempt, while latency/cost totals retain
all attempts. Avoid comparing incomplete runs as model-quality results.

At each confidence threshold (default `0,0.5,0.7,0.9`), the report gives accuracy
on accepted predictions and **coverage**, the fraction of all issues accepted.
No accepted predictions means accuracy is `null`, not zero.

For Jev, threshold confidence is the selected team's probability. Jev's separate
native confidence score is preserved as `provider_confidence`. For the LLM,
probabilities are **self-reported estimates**, not token probabilities or
calibrated confidence. Equal thresholds therefore need not mean equal certainty.

Latency is wall-clock time for each HTTP call plus local response processing;
it includes network overhead. Cost is `null` unless both input/output prices
are supplied for that provider in `.env` and token usage is available. Set a
free token rate explicitly to `0`. Estimates use tokens × configured prices;
they do not account for cache discounts, special billing, or failed requests
whose usage was unavailable. Partial known cost is reported separately from
the total, which stays `null` when any attempt's cost is unknown.

This is a descriptive experiment, not a definitive benchmark. Team labels are
imperfect ground truth, the balanced sample differs from real issue traffic,
cleaning is heuristic, long descriptions may be truncated, and either model
may have seen public issues during training. Inspect a few `model_input` fields
and mistakes before interpreting small score differences. There is no training
split or threshold tuning on this evaluation set.

## Check the mechanics without API calls

```sh
python -m unittest -v
```

Tests cover filtering/cleaning, API request formats, malformed predictions,
hand-calculated metrics, and resuming/scoring a run using mocked model replies.
They do not measure either model's actual accuracy.

API references: [GitHub issues](https://docs.github.com/en/rest/issues/issues#list-repository-issues),
[Jev API](https://docs.typesafe.ai/api),
[Jev confidence](https://docs.typesafe.ai/confidence),
[Jev models/pricing](https://docs.typesafe.ai/models), and
[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
