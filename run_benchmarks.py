"""Run the frozen benchmark tasks, with one writer per results directory."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import validate_task
from predict import run
from providers import validate_config


DATASETS = ("ag_news", "trec", "snips")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data-benchmarks"))
    parser.add_argument("--results-root", type=Path, default=Path("results-benchmarks"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3,
                        help="Maximum concurrent datasets; calls within each dataset are sequential.")
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Each dataset may be requested only once.")
    validate_config()
    jobs = []
    for name in args.datasets:
        dataset_path = args.data_root / name / "issues.json"
        if not dataset_path.exists():
            raise ValueError(f"Missing {dataset_path}; run benchmark_data.py first.")
        dataset = json.loads(dataset_path.read_text())
        validate_task(dataset.get("task"))
        if dataset.get("metadata", {}).get("dataset_id") != name:
            raise ValueError(f"Dataset identity does not match its directory: {name}.")
        jobs.append((name, dataset_path, args.results_root / name / "predictions.json"))
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run, source, target, suite_workers=args.workers,
                                   continue_invalid=True): name
                   for name, source, target in jobs}
        for future in as_completed(pending):
            name = pending[future]
            try:
                future.result()
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(name)
                print(f"{name} stopped: {error}", flush=True)
    if errors:
        raise RuntimeError("Incomplete datasets: " + ", ".join(errors)
                           + ". Fix the cause and rerun with the same settings to resume.")
    print("All requested datasets complete. Next: python compare.py")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
