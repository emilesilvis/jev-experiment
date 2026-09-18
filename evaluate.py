"""Score saved predictions offline. No model calls and no API keys needed."""

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from common import TEAMS, write_json


def percentile(values, probability):
    """Linearly interpolated empirical percentile (including the endpoints)."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def score(issues, attempts, thresholds, labels=None):
    """All issues stay in the denominator, including missing/failed predictions."""
    labels = list(TEAMS if labels is None else labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Labels must be nonempty and unique.")
    truth = {issue["number"]: issue["target"] for issue in issues}
    if not truth or len(truth) != len(issues) or any(t not in labels for t in truth.values()):
        raise ValueError("Dataset must have unique issue numbers and valid targets.")
    latest = {}
    for row in attempts:
        if row["number"] not in truth:
            raise ValueError("Prediction refers to an issue outside this dataset.")
        if row.get("prediction") is not None and row["prediction"] not in labels:
            raise ValueError("Unknown predicted label.")
        confidence = row.get("confidence")
        if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1.")
        latest[row["number"]] = row
    valid = {number: row for number, row in latest.items()
             if row.get("prediction") in labels and not row.get("error")}
    columns = labels + ["missing/error"]
    matrix = [[0] * len(columns) for _ in labels]
    for number, target in truth.items():
        prediction = valid[number]["prediction"] if number in valid else "missing/error"
        matrix[labels.index(target)][columns.index(prediction)] += 1
    correct = sum(matrix[i][i] for i in range(len(labels)))
    per_team = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        per_team[label] = {
            "support": sum(matrix[i]),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    threshold_scores = []
    for threshold in thresholds:
        selected = [(number, row) for number, row in valid.items()
                    if row.get("confidence") is not None and row["confidence"] >= threshold]
        threshold_scores.append({
            "threshold": threshold,
            "selected": len(selected),
            "coverage": len(selected) / len(truth),
            "accuracy": (sum(row["prediction"] == truth[number] for number, row in selected)
                         / len(selected)) if selected else None,
        })
    latencies = [row["latency_seconds"] for row in attempts]
    costs = [row["cost_usd"] for row in attempts if row.get("cost_usd") is not None]
    return {
        "issues": len(truth),
        "predicted": len(valid),
        "failed": len(latest) - len(valid),
        "missing": len(truth) - len(latest),
        "accuracy": correct / len(truth),
        "accuracy_on_predictions": correct / len(valid) if valid else None,
        "macro_f1": statistics.mean(t["f1"] for t in per_team.values()),
        "per_class": per_team,
        # Retain the old key so existing Flutter report consumers still work.
        "per_team": per_team,
        "confusion_matrix": {"rows_actual": labels, "columns_predicted": columns, "counts": matrix},
        "confidence_thresholds": threshold_scores,
        "attempts": len(attempts),
        "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds": percentile(latencies, 0.95),
        "total_latency_seconds": sum(latencies),
        "cost_known_attempts": len(costs),
        "known_cost_usd": sum(costs) if costs else None,
        "total_cost_usd": sum(costs) if attempts and len(costs) == len(attempts) else None,
    }


def main():
    dataset_bytes = (Path(os.getenv("DATA_DIR", "data")) / "issues.json").read_bytes()
    results_dir = Path(os.getenv("RESULTS_DIR", "results"))
    results = json.loads((results_dir / "predictions.json").read_text())
    if hashlib.sha256(dataset_bytes).hexdigest() != results["identity"]["dataset_sha256"]:
        raise ValueError("Predictions belong to a different dataset snapshot.")
    thresholds = [float(x) for x in os.getenv("CONFIDENCE_THRESHOLDS", "0,0.5,0.7,0.9").split(",")]
    if not thresholds or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError("CONFIDENCE_THRESHOLDS must be comma-separated numbers from 0 to 1.")
    if any(r["provider"] not in ("jev", "llm") for r in results["predictions"]):
        raise ValueError("Unknown provider in predictions.")
    dataset = json.loads(dataset_bytes)
    labels = dataset.get("task", {}).get("labels", TEAMS)
    if "task" in dataset and results["identity"]["configuration"].get("task") != dataset["task"]:
        raise ValueError("Prediction task configuration does not match the dataset.")
    report = {"identity": results["identity"], "models": {}}
    for provider in ("jev", "llm"):
        rows = [r for r in results["predictions"] if r["provider"] == provider]
        metrics = score(dataset["issues"], rows, thresholds, labels=labels)
        report["models"][provider] = metrics
        print(f"\n{provider}: accuracy={metrics['accuracy']:.3f}, macro-F1={metrics['macro_f1']:.3f} "
              f"({metrics['predicted']}/{metrics['issues']} predictions)")
        print("Confusion matrix: rows=actual, columns=predicted")
        print(" " * 18 + " ".join(f"{c.removeprefix('team-'):>13}" for c in metrics["confusion_matrix"]["columns_predicted"]))
        for label, counts in zip(labels, metrics["confusion_matrix"]["counts"]):
            print(f"{label:18}" + " ".join(f"{n:13}" for n in counts))
        for item in metrics["confidence_thresholds"]:
            accuracy = f"{item['accuracy']:.3f}" if item["accuracy"] is not None else "n/a"
            print(f"  confidence >= {item['threshold']:.2f}: accuracy={accuracy}, "
                  f"coverage={item['coverage']:.1%} ({item['selected']} issues)")
    write_json(results_dir / "report.json", report)
    print(f"\nSaved {results_dir / 'report.json'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
