"""Check that concurrent tasks keep rubrics and saved results isolated."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import write_json
import run_benchmarks


class SuiteRunnerTests(unittest.TestCase):
    def test_concurrent_tasks_with_same_example_id_keep_their_own_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = {}
            for name in run_benchmarks.DATASETS:
                task = {"name": name, "instructions": "Classify " + name,
                        "labels": {name + "_yes": "Yes", name + "_no": "No"}}
                tasks[name] = task
                write_json(root / "data" / name / "issues.json", {
                    "task": task, "metadata": {"dataset_id": name},
                    "issues": [{"number": 1, "target": name + "_yes",
                                "model_input": "Input for " + name}]})

            def classify(text, *, task):
                self.assertEqual(text, "Input for " + task["name"])
                return {"prediction": task["name"] + "_yes", "confidence": 1,
                        "cost_usd": 0, "error": None}

            with patch.dict(os.environ, {"JEV_API_KEY": "test", "OPENAI_API_KEY": "test"}, clear=True):
                with patch("predict.predict_jev", side_effect=classify):
                    with patch("predict.predict_llm", side_effect=classify):
                        with contextlib.redirect_stdout(io.StringIO()):
                            run_benchmarks.main(["--data-root", str(root / "data"),
                                                 "--results-root", str(root / "results"),
                                                 "--workers", "3"])
            for name, task in tasks.items():
                result = json.loads((root / "results" / name / "predictions.json").read_text())
                self.assertEqual(result["identity"]["configuration"]["task"], task)
                self.assertEqual(len(result["predictions"]), 2)
                self.assertEqual({row["prediction"] for row in result["predictions"]}, {name + "_yes"})
                self.assertEqual(result["identity"]["execution"]["suite_workers"], 3)


if __name__ == "__main__":
    unittest.main()
