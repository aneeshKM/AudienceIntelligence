from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import json

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
    def test_groq_generator_uses_default_configurable_model_and_strict_schema(self) -> None:
        class Response:
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps(judgment())}}
                        ]
                    }
                ).encode()

        captured: list[object] = []

        ssl_context = MagicMock()

        def fake_urlopen(
            api_request: object, timeout: int, context: object
        ) -> Response:
            captured.extend((api_request, timeout, context))
            return Response()

        with (
            patch("audience_trend_miner.classification.request.urlopen", fake_urlopen),
            patch(
                "audience_trend_miner.classification.trusted_ssl_context",
                return_value=ssl_context,
            ),
        ):
            output = GroqStructuredGenerator(api_key="secret").generate(
                "Classify this", {"type": "object"}
            )

        request_body = json.loads(captured[0].data)
        self.assertEqual(json.loads(output), judgment())
        self.assertEqual(request_body["model"], DEFAULT_MODEL)
        self.assertTrue(request_body["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            captured[0].get_header("User-agent"),
            "AudienceTrendMiner/0.1 (https://github.com/aneeshKM/AudienceIntelligence)",
        )
        self.assertIs(captured[2], ssl_context)

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
