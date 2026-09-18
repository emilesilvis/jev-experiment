"""Send each frozen model input to Jev and the LLM; save after every call."""

import hashlib
import fcntl
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from common import TEAMS, validate_task, write_json
from providers import configuration, predict_jev, predict_llm, validate_config


def run(dataset_path, results_path, *, suite_workers=1, continue_invalid=False):
    """Run one immutable snapshot, preventing concurrent writes to its results."""
    dataset_path, results_path = Path(dataset_path), Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another prediction process is using {results_path}.") from None
        return _run(dataset_path, results_path, suite_workers=suite_workers,
                    continue_invalid=continue_invalid)


def _run(dataset_path, results_path, *, suite_workers, continue_invalid):
    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    task = dataset.get("task")
    if task is not None:
        validate_task(task)
    labels = task["labels"] if task is not None else TEAMS
    issues = dataset.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError("Dataset must contain nonempty issues/examples.")
    numbers = []
    for issue in issues:
        if (not isinstance(issue, dict) or type(issue.get("number")) is not int
                or issue.get("target") not in labels
                or not isinstance(issue.get("model_input"), str)
                or not issue["model_input"].strip()):
            raise ValueError("Every example needs an integer number, valid target, and nonempty input.")
        numbers.append(issue["number"])
    if len(set(numbers)) != len(numbers):
        raise ValueError("Dataset example numbers must be unique.")
    identity = {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "configuration": configuration(task),
        # Reject resuming after code changes; keep one experiment per results folder.
        "code_sha256": hashlib.sha256(b"".join(
            Path(__file__).with_name(name).read_bytes()
            for name in ("common.py", "providers.py", "predict.py")
        )).hexdigest(),
    }
    if task is not None:
        identity["execution"] = {
            "provider_order": "alternating", "suite_workers": suite_workers,
            "invalid_response_policy": "count_as_incorrect_and_continue" if continue_invalid else "stop",
        }
    if results_path.exists():
        results = json.loads(results_path.read_text())
        if results["identity"] != identity:
            raise ValueError("Dataset, code, or model settings changed. Choose a new RESULTS_DIR.")
    else:
        results = {
            "identity": identity,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "predictions": [],
        }
    validate_config(task)
    for row in results["predictions"]:
        if (row["number"] not in numbers or row["provider"] not in ("jev", "llm")
                or (not row.get("error") and row.get("prediction") not in labels)):
            raise ValueError("Saved results contain an invalid example, provider, or prediction.")
    completed = {(r["number"], r["provider"]) for r in results["predictions"]
                 if not r.get("error") or (continue_invalid and r.get("error_kind") == "invalid_response")}
    for index, issue in enumerate(issues):
        providers = [("jev", predict_jev), ("llm", predict_llm)]
        if task is not None and index % 2:
            providers.reverse()
        for provider, classify in providers:
            key = (issue["number"], provider)
            if key in completed:
                continue
            started = time.perf_counter()
            row = {"number": issue["number"], "provider": provider}
            try:
                # Only this frozen string crosses the provider boundary, never labels/IDs.
                row.update(classify(issue["model_input"], task=task) if task is not None
                           else classify(issue["model_input"]))
                row.setdefault("error", None)
                row["error_kind"] = "invalid_response" if row["error"] else None
            except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
                row.update(prediction=None, confidence=None, probabilities=None,
                           cost_usd=None, error=f"{type(exc).__name__}: {exc}",
                           error_kind="request_error")
            row["latency_seconds"] = time.perf_counter() - started
            row["recorded_at"] = datetime.now(timezone.utc).isoformat()
            # Keep failed attempts too, so their latency/cost are not silently discarded.
            results["predictions"].append(row)
            write_json(results_path, results)
            prefix = f"[{task['name']}] " if task is not None else ""
            print(f"{prefix}{len(completed) + 1}/{len(issues) * 2} {provider}: "
                  f"{row.get('prediction') or row['error']}", flush=True)
            if row["error"] and not (continue_invalid and row["error_kind"] == "invalid_response"):
                raise RuntimeError("Stopped after a failed call. Fix the cause and rerun; "
                                   "successful predictions will be reused.")
            completed.add(key)
    print(f"Saved {results_path}. Next: python evaluate.py")
    return results


def main():
    run(Path(os.getenv("DATA_DIR", "data")) / "issues.json",
        Path(os.getenv("RESULTS_DIR", "results")) / "predictions.json")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
