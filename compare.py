"""Compare frozen classification runs offline with paired bootstrap intervals."""

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import write_json
from evaluate import percentile, score


PROVIDERS = ("jev", "llm")
SAMPLING_METHODS = ("class_balanced", "natural")


def load_run(data_path, predictions_path):
    """Reject changed snapshots and incomplete runs; final failures remain outcomes."""
    data_bytes = Path(data_path).read_bytes()
    dataset = json.loads(data_bytes)
    results = json.loads(Path(predictions_path).read_text())
    identity = results["identity"]
    if identity["dataset_sha256"] != hashlib.sha256(data_bytes).hexdigest():
        raise ValueError("Predictions belong to a different dataset snapshot.")
    task = dataset["task"]
    labels = task["labels"]
    if not isinstance(labels, dict) or not labels:
        raise ValueError("Dataset task must specify label descriptions.")
    if identity["configuration"].get("task") != task:
        raise ValueError("Prediction task/prompt configuration does not match the dataset.")
    metadata = dataset["metadata"]
    if not isinstance(metadata.get("dataset_id"), str) or not metadata["dataset_id"]:
        raise ValueError("Dataset metadata must identify the dataset.")
    if metadata.get("sampling") not in SAMPLING_METHODS:
        raise ValueError("Dataset sampling must be 'class_balanced' or 'natural'.")
    issues = dataset["issues"]
    numbers = [issue["number"] for issue in issues]
    if not issues or len(set(numbers)) != len(numbers):
        raise ValueError("Dataset must have unique example numbers.")
    if any(issue["target"] not in labels for issue in issues):
        raise ValueError("Dataset has an unknown target label.")
    if any(not isinstance(issue.get("model_input"), str) or not issue["model_input"].strip()
           for issue in issues):
        raise ValueError("Dataset must have nonempty frozen model inputs.")
    grouped = ["group_id" in issue for issue in issues]
    if any(grouped) and not all(grouped):
        raise ValueError("Provide group_id for every example or for none.")
    if any(grouped) and any(not isinstance(issue["group_id"], (str, int))
                            or isinstance(issue["group_id"], bool) for issue in issues):
        raise ValueError("Group IDs must be strings or integers.")
    rows = results["predictions"]
    latest = {provider: {} for provider in PROVIDERS}
    for row in rows:
        provider, number = row["provider"], row["number"]
        if provider not in PROVIDERS or number not in set(numbers):
            raise ValueError("Prediction refers to an unknown provider or example.")
        if number in latest[provider] and not latest[provider][number].get("error"):
            raise ValueError("Duplicate prediction after a successful outcome.")
        if not row.get("error") and row.get("prediction") not in labels:
            raise ValueError("An unsuccessful outcome must record an explicit error.")
        if row.get("prediction") is not None and row["prediction"] not in labels:
            raise ValueError("Prediction has an unknown label.")
        for field in ("latency_seconds", "cost_usd"):
            value = row.get(field)
            if value is None and field == "cost_usd":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"Prediction {field} must be finite and nonnegative.")
        confidence = row.get("confidence")
        if confidence is not None and (isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Prediction confidence must be between 0 and 1.")
        latest[provider][number] = row
    for provider in PROVIDERS:
        if set(latest[provider]) != set(numbers):
            raise ValueError(f"Incomplete run: {provider} has missing outcomes.")
        settings = identity["configuration"].get(provider)
        if not isinstance(settings, dict) or not settings.get("model"):
            raise ValueError(f"Missing model settings for {provider}.")
    return {"dataset": dataset, "results": results, "latest": latest}


def coverage_curve(issues, latest):
    """Thresholds include whole confidence ties; failures stay in the denominator."""
    truth = {issue["number"]: issue["target"] for issue in issues}
    tied = defaultdict(list)
    for number, row in latest.items():
        if row.get("prediction") is not None and not row.get("error") \
                and row.get("confidence") is not None:
            tied[row["confidence"]].append(row["prediction"] == truth[number])
    curve = [{"threshold": None, "selected": 0, "coverage": 0.0, "accuracy": None}]
    selected = correct = 0
    for confidence in sorted(tied, reverse=True):
        selected += len(tied[confidence])
        correct += sum(tied[confidence])
        curve.append({"threshold": confidence, "selected": selected,
                      "coverage": selected / len(issues), "accuracy": correct / selected})
    return curve


def _metric_pair(targets, predictions, indices, label_count):
    """Accuracy and macro-F1 for a resample, counting failures as missed labels."""
    support = [0] * label_count
    predicted = [0] * label_count
    true_positive = [0] * label_count
    for index in indices:
        target, prediction = targets[index], predictions[index]
        support[target] += 1
        if prediction >= 0:
            predicted[prediction] += 1
            true_positive[target] += prediction == target
    accuracy = sum(true_positive) / len(indices)
    f1 = sum(2 * tp / (actual + guessed) if actual + guessed else 0.0
             for tp, actual, guessed in zip(true_positive, support, predicted)) / label_count
    return accuracy, f1


def _sampling_units(issues, sampling):
    if "group_id" in issues[0]:
        groups = defaultdict(list)
        for index, issue in enumerate(issues):
            groups[issue["group_id"]].append(index)
        # Preserve both strata and clusters when clusters do not cross labels.
        if sampling == "class_balanced" and all(
                len({issues[index]["target"] for index in group}) == 1
                for group in groups.values()):
            strata = defaultdict(list)
            for group in groups.values():
                strata[issues[group[0]]["target"]].append(group)
            return list(strata.values()), "paired, class-stratified cluster bootstrap"
        return [list(groups.values())], "paired cluster bootstrap (class counts may vary)"
    if sampling == "class_balanced":
        strata = defaultdict(list)
        for index, issue in enumerate(issues):
            strata[issue["target"]].append([index])
        return list(strata.values()), "paired, class-stratified example bootstrap"
    return [[[index] for index in range(len(issues))]], "paired example bootstrap"


def bootstrap_differences(issues, latest, labels, sampling, repetitions, rng):
    label_index = {label: index for index, label in enumerate(labels)}
    targets = [label_index[issue["target"]] for issue in issues]
    predictions = {}
    for provider in PROVIDERS:
        predictions[provider] = [
            label_index.get(latest[provider][issue["number"]].get("prediction"), -1)
            if not latest[provider][issue["number"]].get("error") else -1
            for issue in issues
        ]
    strata, method = _sampling_units(issues, sampling)
    draws = {"accuracy": [], "macro_f1": []}
    for _ in range(repetitions):
        indices = [index for stratum in strata for _ in range(len(stratum))
                   for index in rng.choice(stratum)]
        metrics = [_metric_pair(targets, predictions[p], indices, len(labels)) for p in PROVIDERS]
        for index, metric in enumerate(draws):
            draws[metric].append(metrics[0][index] - metrics[1][index])
    return draws, method


def _interval(draws):
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def compare_runs(runs, bootstrap=5000, seed=20260918):
    """Equal-weight summary of selected tasks; dataset choice is held fixed."""
    if not runs or type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("Provide at least one dataset and a positive bootstrap count.")
    ids, hashes = set(), set()
    shared_settings = shared_execution = shared_code = None
    rng = random.Random(seed)
    datasets, bootstrap_draws = [], []
    for run in runs:
        dataset, results, latest = run["dataset"], run["results"], run["latest"]
        dataset_id = dataset["metadata"]["dataset_id"]
        snapshot_hash = results["identity"]["dataset_sha256"]
        if dataset_id in ids or snapshot_hash in hashes:
            raise ValueError("Duplicate dataset in comparison.")
        ids.add(dataset_id)
        hashes.add(snapshot_hash)
        settings = {key: value for key, value in results["identity"]["configuration"].items()
                    if key != "task"}
        if shared_settings is not None and settings != shared_settings:
            raise ValueError("Model settings differ across datasets.")
        execution = results["identity"].get("execution")
        code = results["identity"].get("code_sha256")
        if datasets and code != shared_code:
            raise ValueError("Prediction code differs across datasets.")
        if datasets and execution != shared_execution:
            raise ValueError("Execution settings differ across datasets.")
        shared_settings = settings
        shared_execution, shared_code = execution, code
        labels = dataset["task"]["labels"]
        issues = dataset["issues"]
        metrics = {}
        paired = {"both_correct": 0, "jev_only_correct": 0,
                  "llm_only_correct": 0, "neither_correct": 0}
        for issue in issues:
            correct = [not latest[p][issue["number"]].get("error")
                       and latest[p][issue["number"]].get("prediction") == issue["target"]
                       for p in PROVIDERS]
            key = ("both_correct" if all(correct) else "jev_only_correct" if correct[0]
                   else "llm_only_correct" if correct[1] else "neither_correct")
            paired[key] += 1
        for provider in PROVIDERS:
            attempts = [row for row in results["predictions"] if row["provider"] == provider]
            metrics[provider] = score(issues, attempts, [0, 0.5, 0.7, 0.9], labels=labels)
            metrics[provider]["failure_rate"] = metrics[provider]["failed"] / len(issues)
            metrics[provider]["failed_attempts"] = sum(bool(row.get("error")) for row in attempts)
            metrics[provider]["accuracy_coverage_curve"] = coverage_curve(issues, latest[provider])
            metrics[provider]["coverage_budgets"] = [
                {"maximum_coverage": budget, **max(
                    (point for point in metrics[provider]["accuracy_coverage_curve"]
                     if point["coverage"] <= budget), key=lambda point: point["coverage"])}
                for budget in (0.25, 0.5, 0.75, 1.0)
            ]
        draws, method = bootstrap_differences(
            issues, latest, labels, dataset["metadata"]["sampling"], bootstrap, rng)
        bootstrap_draws.append(draws)
        shared_valid = [issue for issue in issues if all(
            not latest[p][issue["number"]].get("error")
            and latest[p][issue["number"]].get("prediction") in labels for p in PROVIDERS)]
        shared_accuracy = {
            p: (sum(latest[p][issue["number"]]["prediction"] == issue["target"]
                    for issue in shared_valid) / len(shared_valid)) if shared_valid else None
            for p in PROVIDERS
        }
        datasets.append({
            "dataset_id": dataset_id, "name": dataset["task"]["name"],
            "metadata": dataset["metadata"], "identity": results["identity"],
            "examples": len(issues), "class_support": dict(Counter(i["target"] for i in issues)),
            "models": metrics, "paired_outcomes": paired, "resampling": method,
            "shared_valid_response_diagnostic": {
                "examples": len(shared_valid), "coverage": len(shared_valid) / len(issues),
                "accuracy_by_model": shared_accuracy,
                "difference_jev_minus_llm": (shared_accuracy["jev"] - shared_accuracy["llm"])
                                           if shared_valid else None,
            },
            "differences_jev_minus_llm": {
                metric: {"estimate": metrics["jev"][metric] - metrics["llm"][metric],
                         "ci95": _interval(draws[metric])}
                for metric in draws
            },
        })
    aggregate = {}
    for metric in ("accuracy", "macro_f1"):
        effects = [item["differences_jev_minus_llm"][metric]["estimate"] for item in datasets]
        average_draws = [statistics.mean(draws[metric][i] for draws in bootstrap_draws)
                         for i in range(bootstrap)]
        aggregate[metric] = {
            "mean_by_model": {provider: statistics.mean(item["models"][provider][metric]
                                                        for item in datasets)
                              for provider in PROVIDERS},
            "mean_difference": statistics.mean(effects), "ci95": _interval(average_draws),
            "median_difference": statistics.median(effects),
            "range_difference": [min(effects), max(effects)],
            "datasets_favoring_jev": sum(effect > 0 for effect in effects),
            "datasets_favoring_llm": sum(effect < 0 for effect in effects),
            "datasets_tied": sum(effect == 0 for effect in effects),
        }
    return {
        "bootstrap": {"repetitions": bootstrap, "seed": seed, "interval": "95% percentile",
                      "direction": "Jev minus LLM",
                      "scope": "Conditional on these selected datasets and saved model runs; "
                               "does not measure variation from dataset choice or repeated model calls."},
        "model_settings": shared_settings, "execution": shared_execution,
        "prediction_code_sha256": shared_code, "datasets": datasets,
        "equal_weight_task_summary": aggregate,
        "notes": [
            "Each selected dataset has equal aggregate weight, regardless of sample size.",
            "Bootstrap draws use the same examples or clusters for both models. Aggregate intervals "
            "resample within each fixed dataset independently, then average task differences.",
            "Class-balanced results describe balanced benchmark samples, not production prevalence.",
            "Failures are incorrect outcomes; missing outcomes are rejected. Retry attempts contribute "
            "to cost and latency; only each example's final outcome contributes to accuracy.",
            "The primary scores measure end-to-end performance, including valid probability output. "
            "The shared-valid-response subset is a post-hoc diagnostic selected by response validity; "
            "it cannot establish classification accuracy on failed outputs or replace the full scores.",
            "Latency median and p95 describe individual attempts, including failed attempts; "
            "p95 uses linear interpolation. Concurrent suite load can affect timing.",
            "Confidence curves are descriptive, include whole tied-score groups, and do not imply "
            "calibration. Coverage budgets select the largest attainable coverage below each budget.",
            "Public benchmarks may overlap model training data. This suite does not establish a "
            "universal winner, and the earlier Flutter baseline is not pooled into its aggregate.",
        ],
    }


def _percent(value):
    return f"{100 * value:.1f}%"


def _effect(value):
    estimate = value.get("estimate", value.get("mean_difference"))
    lower, upper = value["ci95"]
    return f"{100 * estimate:+.1f} pp [{100 * lower:+.1f}, {100 * upper:+.1f}]"


def render_markdown(report):
    lines = ["# Classification task comparison", "",
             "Differences are Jev minus GPT. Brackets show paired 95% bootstrap intervals.", "",
             "| Dataset | Examples | Jev accuracy | GPT accuracy | Accuracy difference | Jev macro-F1 | GPT macro-F1 | F1 difference |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in report["datasets"]:
        jev, llm = (item["models"][provider] for provider in PROVIDERS)
        diff = item["differences_jev_minus_llm"]
        lines.append(f"| {item['name']} | {item['examples']} | {_percent(jev['accuracy'])} | "
                     f"{_percent(llm['accuracy'])} | {_effect(diff['accuracy'])} | "
                     f"{jev['macro_f1']:.3f} | {llm['macro_f1']:.3f} | {_effect(diff['macro_f1'])} |")
    lines.extend(["", "## Equal-weight summary of these datasets", ""])
    for metric, summary in report["equal_weight_task_summary"].items():
        lines.append(f"- {metric.replace('_', ' ')}: Jev {_percent(summary['mean_by_model']['jev'])}, "
                     f"GPT {_percent(summary['mean_by_model']['llm'])}; mean difference {_effect(summary)}. "
                     f"Median task difference {100 * summary['median_difference']:+.1f} pp; "
                     f"range {100 * summary['range_difference'][0]:+.1f} to "
                     f"{100 * summary['range_difference'][1]:+.1f} pp.")
    lines.extend(["", "## Operational results", "",
                  "Latency includes all attempts, including retries. Costs are estimates from saved usage and configured rates.", "",
                  "| Dataset | Model | Final failures | Failed attempts | Latency median / p95 (s) | Total cost (USD) |",
                  "|---|---|---:|---:|---:|---:|"])
    for item in report["datasets"]:
        for provider in PROVIDERS:
            metrics = item["models"][provider]
            cost = metrics["total_cost_usd"]
            cost_text = f"${cost:.6f}" if cost is not None else "unknown"
            lines.append(f"| {item['name']} | {provider} | {metrics['failed']}/{item['examples']} | "
                         f"{metrics['failed_attempts']} | {metrics['median_latency_seconds']:.3f} / "
                         f"{metrics['p95_latency_seconds']:.3f} | {cost_text} |")
    lines.extend(["", "## Valid-response diagnostic", "",
                  "These post-hoc accuracies use only examples where both providers returned valid responses. "
                  "The selected subset excludes failures and does not replace the full end-to-end scores above.", "",
                  "| Dataset | Both valid / all | Jev accuracy on subset | GPT accuracy on subset |",
                  "|---|---:|---:|---:|"])
    for item in report["datasets"]:
        diagnostic = item["shared_valid_response_diagnostic"]
        accuracy = diagnostic["accuracy_by_model"]
        formatted = {p: _percent(accuracy[p]) if accuracy[p] is not None else "n/a" for p in PROVIDERS}
        lines.append(f"| {item['name']} | {diagnostic['examples']}/{item['examples']} | "
                     f"{formatted['jev']} | {formatted['llm']} |")
    lines.extend(["", "## Paired outcomes", "",
                  "| Dataset | Both correct | Jev only | GPT only | Neither correct |",
                  "|---|---:|---:|---:|---:|"])
    for item in report["datasets"]:
        counts = item["paired_outcomes"]
        lines.append(f"| {item['name']} | {counts['both_correct']} | {counts['jev_only_correct']} | "
                     f"{counts['llm_only_correct']} | {counts['neither_correct']} |")
    lines.extend(["", "## Sampling and uncertainty", ""])
    for item in report["datasets"]:
        support = ", ".join(f"{label}: {count}" for label, count in item["class_support"].items())
        lines.append(f"- {item['name']}: {item['resampling']}. Class support: {support}.")
    lines.extend(["", f"{report['bootstrap']['repetitions']:,} bootstrap draws; seed "
                  f"{report['bootstrap']['seed']}. {report['bootstrap']['scope']}", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    lines.extend(["", "Full per-class metrics, identities, confidence curves, and coverage budgets are in `comparison.json`.", "",
                  "Statistical guidance: [Dror et al. (2018), The Hitchhiker’s Guide to Testing Statistical "
                  "Significance in Natural Language Processing](https://aclanthology.org/P18-1128/).", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data-benchmarks"))
    parser.add_argument("--results-root", type=Path, default=Path("results-benchmarks"))
    parser.add_argument("--datasets", nargs="+", help="Dataset directory names; default: all data snapshots")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    names = args.datasets or sorted(path.parent.name for path in args.data_root.glob("*/issues.json"))
    if len(set(names)) != len(names):
        raise ValueError("Duplicate dataset requested.")
    if any(Path(name).name != name or name in (".", "..") for name in names):
        raise ValueError("Dataset names must be direct directory names.")
    runs = [load_run(args.data_root / name / "issues.json",
                     args.results_root / name / "predictions.json") for name in names]
    report = compare_runs(runs, bootstrap=args.bootstrap, seed=args.seed)
    write_json(args.results_root / "comparison.json", report)
    (args.results_root / "comparison.md").write_text(render_markdown(report))
    print(f"Compared {len(runs)} datasets. Saved {args.results_root / 'comparison.md'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from None
