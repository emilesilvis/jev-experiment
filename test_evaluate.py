import unittest

from common import TEAMS
from evaluate import score


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = list(TEAMS)
        self.issues = [{"number": i, "target": t} for i, t in enumerate(self.labels)]

    def row(self, number, prediction, confidence=0.8, error=None):
        return {"number": number, "prediction": prediction, "confidence": confidence,
                "error": error, "latency_seconds": 0.5, "cost_usd": None}

    def test_hand_calculated_metrics_and_empty_threshold(self):
        rows = [self.row(0, self.labels[0], 0.9), self.row(1, self.labels[0], 0.6),
                self.row(2, self.labels[2], 0.8), self.row(3, self.labels[3], 0.7)]
        report = score(self.issues, rows, [0, 0.8, 1])
        self.assertEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + 0 + 1 + 1) / 4)
        self.assertEqual(report["confusion_matrix"]["counts"][1], [1, 0, 0, 0, 0])
        self.assertEqual(report["confidence_thresholds"][1]["accuracy"], 1)
        self.assertEqual(report["confidence_thresholds"][1]["coverage"], 0.5)
        self.assertIsNone(report["confidence_thresholds"][2]["accuracy"])
        self.assertIsNone(report["total_cost_usd"])

    def test_failures_and_missing_predictions_stay_in_denominator(self):
        rows = [self.row(0, self.labels[0]), self.row(1, None, None, "timeout")]
        report = score(self.issues, rows, [0])
        self.assertEqual(report["accuracy"], 0.25)
        self.assertEqual(report["accuracy_on_predictions"], 1)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["missing"], 2)
        self.assertEqual(report["confusion_matrix"]["counts"][1][-1], 1)
        self.assertEqual(report["confidence_thresholds"][0]["coverage"], 0.25)

    def test_retry_uses_latest_prediction_but_all_attempt_latencies(self):
        rows = [self.row(0, None, None, "timeout"), self.row(0, self.labels[0])]
        rows[-1]["cost_usd"] = 0.01
        report = score(self.issues, rows, [0])
        self.assertEqual(report["predicted"], 1)
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(report["total_latency_seconds"], 1)
        self.assertEqual(report["known_cost_usd"], 0.01)
        self.assertIsNone(report["total_cost_usd"])

    def test_unknown_issue_or_prediction_is_rejected(self):
        for row in [self.row(99, self.labels[0]), self.row(0, "other")]:
            with self.assertRaises(ValueError):
                score(self.issues, [row], [0])

    def test_explicit_task_labels_and_latency_percentile(self):
        issues = [{"number": 0, "target": "news"}, {"number": 1, "target": "sport"}]
        rows = [self.row(0, "news"), self.row(1, "news")]
        rows[1]["latency_seconds"] = 1.5
        report = score(issues, rows, [0], labels={"news": "News", "sport": "Sport"})
        self.assertEqual(report["accuracy"], 0.5)
        self.assertAlmostEqual(report["macro_f1"], 1 / 3)
        self.assertEqual(report["per_class"]["sport"]["support"], 1)
        self.assertAlmostEqual(report["p95_latency_seconds"], 1.45)


if __name__ == "__main__":
    unittest.main()
