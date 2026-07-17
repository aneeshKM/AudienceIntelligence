from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from audience_trend_miner.classification import (
    ArticleJudgment,
    DEFAULT_MODEL,
    GroqStructuredGenerator,
    classify_article,
)
from tests.test_trend_qualification import article


class ScriptedGenerator:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, object]]] = []

    def generate(self, prompt: str, schema: dict[str, object]) -> object:
        self.requests.append((prompt, schema))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def judgment(rejection_class: str = "accepted") -> dict[str, object]:
    accepted = rejection_class == "accepted"
    return {
        "supports_consumer_audience": accepted,
        "brand_safe": accepted,
        "rejection_class": rejection_class,
        "rationale": "Fixture judgment grounded in the supplied article.",
    }


class ArticleClassificationTest(unittest.TestCase):
    def test_groq_generator_requests_strict_judgment_and_returns_raw_content(self) -> None:
        raw_content = '{"supports_consumer_audience":true}'
        completions = MagicMock()
        completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw_content))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )
        schema: dict[str, object] = {"type": "object"}

        output = GroqStructuredGenerator(client=client).generate(
            "Classify this", schema
        )

        self.assertEqual(output, raw_content)
        completions.create.assert_called_once_with(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Classify this"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "article_judgment",
                    "strict": True,
                    "schema": schema,
                },
            },
            reasoning_effort="medium",
            max_completion_tokens=2048,
            stream=False,
        )

    def test_groq_generator_keeps_network_policy_under_application_control(self) -> None:
        with patch("audience_trend_miner.classification.Groq") as groq:
            GroqStructuredGenerator(api_key="secret")

        groq.assert_called_once_with(
            api_key="secret",
            max_retries=0,
            timeout=60,
        )

    def test_accepts_a_complete_schema_valid_commercial_judgment(self) -> None:
        generator = ScriptedGenerator(judgment())

        result = classify_article(
            article("Running shoes", 70_000, 140_000), generator, sleep=lambda _: None
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.judgment, ArticleJudgment(**judgment()))
        self.assertEqual(len(result.attempts), 1)
        self.assertTrue(result.attempts[0].validation_valid)
        prompt, schema = generator.requests[0]
        self.assertIn("Running shoes", prompt)
        self.assertFalse(schema["additionalProperties"])

    def test_rejects_every_required_non_commercial_or_unsafe_class(self) -> None:
        rejection_classes = (
            "tragedy",
            "violent_crime",
            "death_driven",
            "routine_politics",
            "isolated_news",
            "no_consumer_audience",
        )

        for rejection_class in rejection_classes:
            with self.subTest(rejection_class=rejection_class):
                result = classify_article(
                    article("News event", 70_000, 140_000),
                    ScriptedGenerator(judgment(rejection_class)),
                    sleep=lambda _: None,
                )
                self.assertFalse(result.accepted)
                self.assertEqual(result.judgment.rejection_class, rejection_class)

    def test_retries_cross_field_inconsistency_in_strict_judgment(self) -> None:
        inconsistent = judgment()
        inconsistent["brand_safe"] = False
        generator = ScriptedGenerator(inconsistent, judgment())

        result = classify_article(
            article("Running shoes", 70_000, 140_000),
            generator,
            sleep=lambda _: None,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].validation_valid)
        self.assertIn("accepted judgment must", result.attempts[0].error or "")

    def test_retries_invalid_output_then_recovers_without_accepting_partial_data(self) -> None:
        generator = ScriptedGenerator(
            '{"supports_consumer_audience": true',
            judgment(),
        )

        result = classify_article(
            article("Home espresso", 70_000, 140_000),
            generator,
            sleep=lambda _: None,
            jitter=lambda: 0,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].validation_valid)
        self.assertEqual(
            result.attempts[0].raw_output,
            '{"supports_consumer_audience": true',
        )
        self.assertTrue(result.attempts[1].validation_valid)

    def test_fails_closed_after_three_invalid_or_unavailable_attempts(self) -> None:
        generator = ScriptedGenerator(
            RuntimeError("temporarily unavailable"),
            {"brand_safe": True},
            {**judgment(), "unexpected": True},
        )

        result = classify_article(
            article("Consumer topic", 70_000, 140_000),
            generator,
            sleep=lambda _: None,
            jitter=lambda: 0,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.judgment)
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual(result.decision_reason, "exhausted_attempts")


if __name__ == "__main__":
    unittest.main()
