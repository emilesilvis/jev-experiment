"""Exercise a saved run end-to-end with mocked providers and temporary files."""

import contextlib
import fcntl
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate
import predict
from common import TEAMS, write_json


class RunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.data = root / "data"
        self.results = root / "results"
        environment = patch.dict(os.environ, {
            "DATA_DIR": str(self.data), "RESULTS_DIR": str(self.results),
            "JEV_API_KEY": "test", "OPENAI_API_KEY": "test",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.team = next(iter(TEAMS))
        write_json(self.data / "issues.json", {"issues": [{
            "number": 123, "target": self.team, "labels": [self.team],
            "body": "UNSANITIZED BODY", "model_input": "Title: Widget crashes",
        }]})
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def answer(self):
        return {"prediction": self.team, "confidence": 0.8,
                "probabilities": {t: 0.8 if t == self.team else 0.2 / 3 for t in TEAMS},
                "cost_usd": None}

    def test_resume_after_failure_and_evaluate_without_keys(self):
        with patch("predict.predict_jev", return_value=self.answer()) as jev:
            with patch("predict.predict_llm", side_effect=RuntimeError("timeout")):
                with self.assertRaisesRegex(RuntimeError, "Stopped"):
                    predict.main()
            jev.assert_called_once_with("Title: Widget crashes")
        with patch("predict.predict_jev") as jev:
            with patch("predict.predict_llm", return_value=self.answer()) as llm:
                predict.main()
                jev.assert_not_called()
                llm.assert_called_once_with("Title: Widget crashes")
                predict.main()
                llm.assert_called_once()
        saved = json.loads((self.results / "predictions.json").read_text())
        self.assertEqual(len(saved["predictions"]), 3)
        with patch.dict(os.environ, {"JEV_API_KEY": "", "OPENAI_API_KEY": ""}):
            evaluate.main()
        report = json.loads((self.results / "report.json").read_text())
        self.assertEqual(report["models"]["jev"]["accuracy"], 1)
        self.assertEqual(report["models"]["llm"]["accuracy"], 1)
        self.assertEqual(report["models"]["llm"]["attempts"], 2)

    def test_changed_settings_or_dataset_cannot_mix_results(self):
        with patch("predict.predict_jev", return_value=self.answer()):
            with patch("predict.predict_llm", return_value=self.answer()):
                predict.main()
        with patch.dict(os.environ, {"LLM_MODEL": "different-model"}):
            with self.assertRaisesRegex(ValueError, "settings changed"):
                predict.main()
        with (self.data / "issues.json").open("a") as dataset:
            dataset.write("\n")
        with self.assertRaisesRegex(ValueError, "settings changed"):
            predict.main()
        with self.assertRaisesRegex(ValueError, "different dataset"):
            evaluate.main()

    def test_failed_answer_keeps_known_cost_and_stops(self):
        failed = {"prediction": None, "probabilities": None, "confidence": None,
                  "error": "LLM refused", "cost_usd": 0.001,
                  "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        with patch("predict.predict_jev", return_value=self.answer()):
            with patch("predict.predict_llm", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "Stopped"):
                    predict.main()
        saved = json.loads((self.results / "predictions.json").read_text())
        self.assertEqual(saved["predictions"][-1]["cost_usd"], 0.001)
        self.assertEqual(saved["predictions"][-1]["error"], "LLM refused")
        evaluate.main()
        report = json.loads((self.results / "report.json").read_text())
        self.assertEqual(report["models"]["llm"]["failed"], 1)
        self.assertEqual(report["models"]["llm"]["known_cost_usd"], 0.001)

    def test_custom_task_end_to_end_alternates_order_and_freezes_rubric(self):
        task = {"name": "Sentiment", "instructions": "Classify sentiment.",
                "labels": {"positive": "Favorable", "negative": "Unfavorable"}}
        dataset = {"task": task, "issues": [
            {"number": 1, "target": "positive", "model_input": "Good"},
            {"number": 2, "target": "negative", "model_input": "Bad"}]}
        write_json(self.data / "issues.json", dataset)
        calls = []

        def answer(provider):
            def classify(text, *, task):
                calls.append((provider, text, task))
                return {"prediction": "positive" if text == "Good" else "negative",
                        "confidence": 0.9, "cost_usd": 0.001}
            return classify

        with patch("predict.predict_jev", side_effect=answer("jev")):
            with patch("predict.predict_llm", side_effect=answer("llm")):
                predict.main()
        self.assertEqual([p for p, _, _ in calls], ["jev", "llm", "llm", "jev"])
        self.assertTrue(all(rubric == task for _, _, rubric in calls))
        saved = json.loads((self.results / "predictions.json").read_text())
        self.assertEqual(saved["identity"]["configuration"]["task"], task)
        evaluate.main()
        report = json.loads((self.results / "report.json").read_text())
        self.assertEqual(report["models"]["jev"]["accuracy"], 1)
        dataset["task"]["instructions"] = "A changed rubric"
        write_json(self.data / "issues.json", dataset)
        with patch("predict.predict_jev") as request:
            with self.assertRaisesRegex(ValueError, "settings changed"):
                predict.main()
            request.assert_not_called()

    def test_invalid_dataset_stops_before_provider_calls(self):
        write_json(self.data / "issues.json", {"issues": [
            {"number": 1, "target": self.team, "model_input": " "}]})
        with patch("predict.predict_jev") as request:
            with self.assertRaisesRegex(ValueError, "nonempty input"):
                predict.main()
            request.assert_not_called()

    def test_locked_results_prevent_duplicate_api_calls(self):
        self.results.mkdir(parents=True)
        with (self.results / "predictions.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch("predict.predict_jev") as request:
                with self.assertRaisesRegex(RuntimeError, "Another prediction process"):
                    predict.main()
                request.assert_not_called()

    def test_benchmark_invalid_responses_count_as_final_and_are_not_retried(self):
        task = {"name": "Sentiment", "instructions": "Classify sentiment.",
                "labels": {"positive": "Favorable", "negative": "Unfavorable"}}
        write_json(self.data / "issues.json", {"task": task, "issues": [
            {"number": 1, "target": "positive", "model_input": "Good"},
            {"number": 2, "target": "negative", "model_input": "Bad"}]})
        invalid = {"prediction": None, "confidence": None, "probabilities": None,
                   "error": "ValueError: Probabilities must sum to 1", "cost_usd": 0.001}
        valid = {"prediction": "negative", "confidence": 1, "cost_usd": 0.001}
        with patch("predict.predict_jev", side_effect=[invalid, valid]) as jev:
            with patch("predict.predict_llm", return_value=valid) as llm:
                for _ in range(2):
                    predict.run(self.data / "issues.json", self.results / "predictions.json",
                                continue_invalid=True)
                self.assertEqual(jev.call_count, 2)
                self.assertEqual(llm.call_count, 2)
        saved = json.loads((self.results / "predictions.json").read_text())
        self.assertEqual(len(saved["predictions"]), 4)
        self.assertEqual(saved["predictions"][0]["error_kind"], "invalid_response")
        evaluate.main()
        report = json.loads((self.results / "report.json").read_text())
        self.assertEqual(report["models"]["jev"]["accuracy"], 0.5)
        self.assertEqual(report["models"]["jev"]["failed"], 1)
        self.assertEqual(report["models"]["jev"]["total_cost_usd"], 0.002)

    def test_benchmark_request_errors_still_stop(self):
        with patch("predict.predict_jev", side_effect=RuntimeError("HTTP 429")) as request:
            with self.assertRaisesRegex(RuntimeError, "Stopped"):
                predict.run(self.data / "issues.json", self.results / "predictions.json",
                            continue_invalid=True)
            request.assert_called_once()
        saved = json.loads((self.results / "predictions.json").read_text())
        self.assertEqual(saved["predictions"][0]["error_kind"], "request_error")


if __name__ == "__main__":
    unittest.main()
