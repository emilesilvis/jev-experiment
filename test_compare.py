import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from compare import (bootstrap_differences, compare_runs, coverage_curve, load_run,
                     render_markdown)


class ComparisonTests(unittest.TestCase):
    def make_run(self, predictions_jev=None, predictions_llm=None, dataset_id="fixture",
                 targets=None, sampling="class_balanced"):
        targets = targets or ["a", "b", "a", "b"]
        predictions_jev = targets if predictions_jev is None else predictions_jev
        predictions_llm = targets if predictions_llm is None else predictions_llm
        task = {"name": dataset_id, "instructions": "Classify.", "labels": {"a": "A", "b": "B"}}
        dataset = {"task": task, "metadata": {"dataset_id": dataset_id, "sampling": sampling},
                   "issues": [{"number": i, "target": label, "model_input": f"Example {i}"}
                              for i, label in enumerate(targets)]}
        rows = []
        latest = {"jev": {}, "llm": {}}
        for provider, predictions in (("jev", predictions_jev), ("llm", predictions_llm)):
            for number, prediction in enumerate(predictions):
                row = {"number": number, "provider": provider, "prediction": prediction,
                       "confidence": 0.8 if prediction is not None else None,
                       "error": "timeout" if prediction is None else None,
                       "latency_seconds": 0.5, "cost_usd": 0.001}
                rows.append(row)
                latest[provider][number] = row
        identity = {"dataset_sha256": self.digest(dataset), "configuration": {
            "request_version": 2, "task": task,
            "jev": {"model": "jev-fixture"}, "llm": {"model": "gpt-fixture"}}}
        return {"dataset": dataset, "results": {"identity": identity, "predictions": rows},
                "latest": latest}

    @staticmethod
    def digest(dataset):
        return hashlib.sha256(json.dumps(dataset).encode()).hexdigest()

    def load_fixture(self, run):
        with tempfile.TemporaryDirectory() as directory:
            data_path, predictions_path = Path(directory) / "issues.json", Path(directory) / "predictions.json"
            data_path.write_text(json.dumps(run["dataset"]))
            predictions_path.write_text(json.dumps(run["results"]))
            return load_run(data_path, predictions_path)

    def test_paired_counts_and_hand_calculated_effects(self):
        run = self.make_run(["a", "b", "b", "a"], ["a", "a", "a", "a"])
        item = compare_runs([self.load_fixture(run)], bootstrap=100)["datasets"][0]
        self.assertEqual(item["paired_outcomes"], {
            "both_correct": 1, "jev_only_correct": 1, "llm_only_correct": 1, "neither_correct": 1})
        self.assertEqual(item["models"]["jev"]["accuracy"], 0.5)
        self.assertEqual(item["models"]["jev"]["macro_f1"], 0.5)
        self.assertAlmostEqual(item["models"]["llm"]["macro_f1"], 1 / 3)
        self.assertEqual(item["differences_jev_minus_llm"]["accuracy"]["estimate"], 0)
        self.assertAlmostEqual(item["differences_jev_minus_llm"]["macro_f1"]["estimate"], 1 / 6)

    def test_identical_models_have_zero_effect_and_zero_width_intervals(self):
        run = self.make_run(["a", "a", None, "b"], ["a", "a", None, "b"])
        report = compare_runs([run], bootstrap=300)
        for metric in ("accuracy", "macro_f1"):
            self.assertEqual(report["datasets"][0]["differences_jev_minus_llm"][metric],
                             {"estimate": 0.0, "ci95": [0.0, 0.0]})
            self.assertEqual(report["equal_weight_task_summary"][metric]["ci95"], [0.0, 0.0])

    def test_determinism_and_interval_contains_known_effect(self):
        run = self.make_run(predictions_llm=["b", "a", "a", "b"])
        first = compare_runs([run], bootstrap=600, seed=17)
        self.assertEqual(first, compare_runs([run], bootstrap=600, seed=17))
        effect = first["datasets"][0]["differences_jev_minus_llm"]["accuracy"]
        self.assertEqual(effect["estimate"], 0.5)
        self.assertLessEqual(effect["ci95"][0], 0.5)
        self.assertGreaterEqual(effect["ci95"][1], 0.5)

    def test_equal_task_weight_despite_unequal_sample_sizes(self):
        small = self.make_run(predictions_llm=["b", "a"], targets=["a", "b"], dataset_id="small")
        targets = ["a", "b"] * 5
        large = self.make_run(predictions_jev=["b", "a"] * 5, targets=targets, dataset_id="large")
        summary = compare_runs([small, large], bootstrap=100)["equal_weight_task_summary"]["accuracy"]
        self.assertEqual(summary["mean_by_model"], {"jev": 0.5, "llm": 0.5})
        self.assertEqual(summary["mean_difference"], 0)
        self.assertEqual(summary["range_difference"], [-1, 1])
        self.assertEqual(summary["ci95"], [0.0, 0.0])

    def test_failure_is_incorrect_and_retries_count_toward_cost_and_latency(self):
        run = self.make_run(predictions_jev=["a", "b", None, "b"])
        retry = copy.deepcopy(run["results"]["predictions"][0])
        retry.update(prediction=None, confidence=None, error="timeout", latency_seconds=2, cost_usd=None)
        run["results"]["predictions"].insert(0, retry)
        item = compare_runs([self.load_fixture(run)], bootstrap=100)["datasets"][0]
        metrics = item["models"]["jev"]
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["failure_rate"], 0.25)
        self.assertEqual(metrics["failed_attempts"], 2)
        self.assertEqual(metrics["total_latency_seconds"], 4)
        self.assertIsNone(metrics["total_cost_usd"])
        self.assertEqual(metrics["known_cost_usd"], 0.004)

    def test_confidence_ties_are_never_split(self):
        run = self.make_run(predictions_jev=["a", "a", "a", "b"])
        latest = run["latest"]["jev"]
        latest[0]["confidence"] = latest[1]["confidence"] = 0.9
        latest[2]["confidence"] = 0.8
        latest[3]["confidence"] = None
        curve = coverage_curve(run["dataset"]["issues"], latest)
        self.assertEqual([point["selected"] for point in curve], [0, 2, 3])
        self.assertEqual(curve[1]["accuracy"], 0.5)
        metrics = compare_runs([run], bootstrap=20)["datasets"][0]["models"]["jev"]
        self.assertEqual(metrics["coverage_budgets"][0]["selected"], 0)
        self.assertEqual(metrics["coverage_budgets"][-1]["coverage"], 0.75)

    def test_cluster_bootstrap_keeps_group_members_together(self):
        run = self.make_run(predictions_llm=["b", "b", "a", "a"], sampling="natural")
        # Every group has exactly one Jev-only win and one shared correct answer.
        for issue in run["dataset"]["issues"]:
            issue["group_id"] = issue["number"] // 2
        draws, method = bootstrap_differences(run["dataset"]["issues"], run["latest"],
                                             ["a", "b"], "natural", 100, random.Random(1))
        self.assertEqual(set(draws["accuracy"]), {0.5})
        self.assertIn("cluster", method)

    def test_natural_and_stratified_resampling_have_distinct_behavior(self):
        run = self.make_run(predictions_llm=["b", "b", "b", "b"])
        args = (run["dataset"]["issues"], run["latest"], ["a", "b"])
        stratified, _ = bootstrap_differences(*args, "class_balanced", 100, random.Random(1))
        natural, _ = bootstrap_differences(*args, "natural", 100, random.Random(1))
        self.assertEqual(set(stratified["accuracy"]), {0.5})
        self.assertGreater(len(set(natural["accuracy"])), 1)

    def test_changed_snapshot_prompt_and_incomplete_runs_are_rejected(self):
        for change, pattern in (
            (lambda run: run["dataset"]["issues"][0].update(model_input="changed"), "snapshot"),
            (lambda run: run["results"]["identity"]["configuration"].update(task={}), "configuration"),
            (lambda run: run["results"]["predictions"].pop(), "Incomplete"),
            (lambda run: run["results"]["predictions"].append(run["results"]["predictions"][0]), "Duplicate"),
            (lambda run: run["results"]["predictions"][0].update(prediction=None), "explicit error"),
        ):
            with self.subTest(pattern=pattern):
                run = self.make_run()
                change(run)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.load_fixture(run)

    def test_duplicate_datasets_and_different_model_settings_are_rejected(self):
        first, second = self.make_run(), self.make_run(dataset_id="second")
        with self.assertRaisesRegex(ValueError, "Duplicate dataset"):
            compare_runs([first, first], bootstrap=10)
        second["results"]["identity"]["configuration"]["llm"]["model"] = "other-model"
        with self.assertRaisesRegex(ValueError, "Model settings differ"):
            compare_runs([first, second], bootstrap=10)

    def test_different_prediction_code_or_execution_settings_are_rejected(self):
        for key, first_value, second_value, message in (
            ("code_sha256", "first", "second", "Prediction code differs"),
            ("execution", {"suite_workers": 1}, {"suite_workers": 3}, "Execution settings differ"),
        ):
            with self.subTest(key=key):
                first, second = self.make_run(), self.make_run(dataset_id="second")
                first["results"]["identity"][key] = first_value
                second["results"]["identity"][key] = second_value
                with self.assertRaisesRegex(ValueError, message):
                    compare_runs([first, second], bootstrap=10)

    def test_markdown_contains_methods_and_results(self):
        text = render_markdown(compare_runs([self.make_run()], bootstrap=100))
        self.assertIn("100.0%", text)
        self.assertIn("Equal-weight", text)
        self.assertIn("class-stratified", text)
        self.assertIn("Conditional on these selected datasets", text)
        self.assertIn("comparison.json", text)

    def test_shared_valid_diagnostic_excludes_either_models_failures(self):
        run = self.make_run(["a", "b", None, "b"], [None, "a", "a", "b"])
        item = compare_runs([run], bootstrap=20)["datasets"][0]
        diagnostic = item["shared_valid_response_diagnostic"]
        self.assertEqual(diagnostic["examples"], 2)
        self.assertEqual(diagnostic["accuracy_by_model"], {"jev": 1.0, "llm": 0.5})
        self.assertEqual(item["models"]["jev"]["accuracy"], 0.75)
        self.assertEqual(item["models"]["llm"]["accuracy"], 0.5)

    def test_no_shared_valid_responses_produce_null_diagnostic(self):
        run = self.make_run([None] * 4)
        report = compare_runs([run], bootstrap=20)
        diagnostic = report["datasets"][0]["shared_valid_response_diagnostic"]
        self.assertEqual(diagnostic["examples"], 0)
        self.assertIsNone(diagnostic["difference_jev_minus_llm"])
        self.assertIn("n/a", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
