"""Exercise provider boundaries without spending API credits."""

import json
import math
import os
import unittest
from unittest.mock import patch

import providers
from common import TEAMS


class ProviderTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"JEV_API_KEY": "jev-secret", "OPENAI_API_KEY": "llm-secret"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.team = next(iter(TEAMS))
        self.probabilities = {team: 0.7 if team == self.team else 0.1 for team in TEAMS}

    def jev_response(self):
        return {
            "model": "jev-pinned", "id": "request-1",
            "answers": {"owner": {"type": "choice", "choice": self.team,
                                    "probabilities": self.probabilities, "confidence": 0.42}},
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

    def llm_response(self):
        return {
            "model": "llm-pinned", "id": "request-2",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.probabilities)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    def test_jev_request_and_confidence_are_explicit(self):
        with patch("providers.request_json", return_value=self.jev_response()) as request:
            result = providers.predict_jev("clean issue text")
        body = request.call_args.kwargs["json"]
        self.assertEqual(body["state"], "clean issue text")
        self.assertEqual(body["questions"]["owner"]["criteria"], TEAMS)
        self.assertEqual(body["questions"]["owner"]["instructions"], providers.INSTRUCTIONS)
        self.assertEqual(result["prediction"], self.team)
        self.assertEqual(result["confidence"], 0.7)
        self.assertEqual(result["provider_confidence"], 0.42)
        self.assertEqual(result["model"], "jev-pinned")
        self.assertIsNone(result["cost_usd"])

    def test_llm_request_has_strict_schema_and_same_text(self):
        with patch("providers.request_json", return_value=self.llm_response()) as request:
            result = providers.predict_llm("clean issue text")
        body = request.call_args.kwargs["json"]
        self.assertEqual(body["messages"][1]["content"], "clean issue text")
        schema = body["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(set(schema["schema"]["required"]), set(TEAMS))
        self.assertFalse(body["store"])
        self.assertEqual(result["prediction"], self.team)
        self.assertEqual(result["confidence_kind"], "self_reported_probability")
        self.assertIsNone(result["provider_confidence"])

    def test_invalid_distributions_fail_instead_of_becoming_predictions(self):
        bad = [float("nan"), float("inf"), -0.1, 1.1, True, "0.7", 0.2]
        for value in bad:
            with self.subTest(value=value):
                probabilities = dict(self.probabilities, **{self.team: value})
                with self.assertRaises(ValueError):
                    providers._probabilities(probabilities)
        for probabilities in ({}, {**self.probabilities, "unknown": 0.0}, []):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                providers._probabilities(probabilities)

    def test_jev_rejects_choice_disagreeing_with_probabilities(self):
        response = self.jev_response()
        response["answers"]["owner"]["choice"] = list(TEAMS)[1]
        with patch("providers.request_json", return_value=response):
            result = providers.predict_jev("issue")
        self.assertIsNone(result["prediction"])
        self.assertIn("highest-probability", result["error"])

    def test_llm_refusal_and_truncation_are_errors(self):
        for finish, refusal in (("length", None), ("stop", "Cannot classify")):
            response = self.llm_response()
            response["choices"][0]["finish_reason"] = finish
            response["choices"][0]["message"]["refusal"] = refusal
            with self.subTest(finish=finish, refusal=refusal):
                with patch("providers.request_json", return_value=response):
                    result = providers.predict_llm("issue")
                self.assertIsNone(result["prediction"])
                self.assertIsNone(result["probabilities"])
                self.assertIn("LLM refused", result["error"])

    def test_invalid_answers_preserve_available_usage_and_cost(self):
        jev = self.jev_response()
        jev["answers"]["owner"]["probabilities"] = {}
        llm = self.llm_response()
        llm["choices"][0]["message"]["content"] = "invalid JSON"
        truncated = self.llm_response()
        truncated["choices"][0]["finish_reason"] = "length"
        for prefix, predict, response in (("JEV", providers.predict_jev, jev),
                                          ("LLM", providers.predict_llm, llm),
                                          ("LLM", providers.predict_llm, truncated)):
            with self.subTest(provider=prefix, response=response):
                with patch.dict(os.environ, {f"{prefix}_INPUT_USD_PER_MILLION": "2",
                                             f"{prefix}_OUTPUT_USD_PER_MILLION": "3"}):
                    with patch("providers.request_json", return_value=response):
                        result = predict("issue")
                self.assertIsNone(result["prediction"])
                self.assertIsNone(result["confidence"])
                self.assertTrue(result["error"])
                self.assertEqual(result["usage"], response["usage"])
                self.assertEqual(result["model"], response["model"])
                self.assertEqual(result["response_id"], response["id"])
                self.assertAlmostEqual(result["cost_usd"], 0.00026)

    def test_transport_failure_still_raises(self):
        with patch("providers.request_json", side_effect=RuntimeError("HTTP 429")):
            for predict in (providers.predict_jev, providers.predict_llm):
                with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                    predict("issue")

    def test_cost_estimation_requires_rates_and_usage_and_allows_zero(self):
        for prefix, predict, response in (("JEV", providers.predict_jev, self.jev_response()),
                                           ("LLM", providers.predict_llm, self.llm_response())):
            with self.subTest(provider=prefix):
                with patch.dict(os.environ, {f"{prefix}_INPUT_USD_PER_MILLION": "2",
                                             f"{prefix}_OUTPUT_USD_PER_MILLION": "0"}):
                    with patch("providers.request_json", return_value=response):
                        self.assertTrue(math.isclose(predict("issue")["cost_usd"], 0.0002))
                    response["usage"] = {}
                    with patch("providers.request_json", return_value=response):
                        self.assertIsNone(predict("issue")["cost_usd"])

    def test_configuration_has_no_secrets_and_rejects_bad_rates(self):
        providers.validate_config()
        self.assertNotIn("secret", json.dumps(providers.configuration()))
        for value in ("nan", "inf", "-1", "nope"):
            with patch.dict(os.environ, {"JEV_INPUT_USD_PER_MILLION": value}):
                with self.assertRaises(ValueError):
                    providers.configuration()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), self.assertRaises(ValueError):
            providers.validate_config()

    def test_custom_task_uses_same_rubric_and_text_with_both_providers(self):
        task = {"name": "sentiment", "instructions": "Classify the text's sentiment.",
                "labels": {"positive": "Favorable opinion", "negative": "Unfavorable opinion"}}
        probabilities = {"positive": 0.8, "negative": 0.2}
        jev_response = self.jev_response()
        jev_response["answers"] = {"classification": {
            "type": "choice", "choice": "positive", "probabilities": probabilities,
            "confidence": 0.6}}
        llm_response = self.llm_response()
        llm_response["choices"][0]["message"]["content"] = json.dumps(probabilities)
        for classify, response in ((providers.predict_jev, jev_response),
                                   (providers.predict_llm, llm_response)):
            with self.subTest(provider=classify.__name__):
                with patch("providers.request_json", return_value=response) as request:
                    result = classify("I loved it.", task=task)
                self.assertEqual(result["prediction"], "positive")
                self.assertEqual(result["probabilities"], probabilities)
                body = request.call_args.kwargs["json"]
                if classify is providers.predict_jev:
                    question = body["questions"]["classification"]
                    self.assertEqual(body["state"], "I loved it.")
                    self.assertEqual(question["criteria"], task["labels"])
                    self.assertEqual(question["instructions"], task["instructions"])
                else:
                    self.assertEqual(body["messages"][1]["content"], "I loved it.")
                    self.assertIn(task["instructions"], body["messages"][0]["content"])
                    self.assertIn(json.dumps(task["labels"]), body["messages"][0]["content"])
                    schema = body["response_format"]["json_schema"]["schema"]
                    self.assertEqual(schema["required"], list(task["labels"]))
                self.assertNotIn("Flutter", json.dumps(body))
        config = providers.configuration(task)
        self.assertEqual(config["task"], task)
        task["labels"]["positive"] = "changed"
        self.assertEqual(config["task"]["labels"]["positive"], "Favorable opinion")

    def test_custom_task_rejects_unknown_response_label_and_invalid_rubric(self):
        with self.assertRaises(ValueError):
            providers._probabilities({"a": 1.0, "wrong": 0.0}, {"a": "A", "b": "B"})
        for task in ({}, {"name": "bad", "instructions": "test", "labels": {"a": "A"}}):
            with patch("providers.request_json") as request:
                with self.assertRaises(ValueError):
                    providers.predict_jev("text", task=task)
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
