from __future__ import annotations

import json
import unittest

from audience_trend_miner.v2.cluster_adjudication import AdjudicationRequest
from audience_trend_miner.v2.cluster_adjudication.adapters import (
    LangChainGroqAdjudicationAdapter,
)
from audience_trend_miner.v2.shared import V2ContractError


class _FakeStructuredRunnable:
    def __init__(self, output: object) -> None:
        self.output = output
        self.messages: object = None

    def invoke(self, messages: object) -> object:
        self.messages = messages
        return self.output


class _FakeChatModel:
    def __init__(self, output: object) -> None:
        self.runnable = _FakeStructuredRunnable(output)
        self.schema: object = None
        self.options: dict[str, object] = {}

    def with_structured_output(self, schema: object, **options: object) -> _FakeStructuredRunnable:
        self.schema = schema
        self.options = options
        return self.runnable


class _InvalidStructuredChatModel:
    def with_structured_output(self, schema: object, **options: object) -> object:
        del schema, options
        raise ValueError("unsupported structured output schema")


class LangChainGroqAdjudicationAdapterTest(unittest.TestCase):
    def test_invalid_provider_contract_is_a_fatal_configuration_error(self) -> None:
        adapter = LangChainGroqAdjudicationAdapter(
            model="openai/gpt-oss-20b",
            chat_model=_InvalidStructuredChatModel(),
        )

        with self.assertRaisesRegex(
            V2ContractError, "Groq structured output configuration is invalid"
        ):
            adapter.invoke(
                AdjudicationRequest(
                    role="proposer",
                    prompt="proposer prompt",
                    members=(),
                )
            )

    def test_adapter_selects_supported_structured_output_contract_by_model(self) -> None:
        scenarios = (
            ("openai/gpt-oss-20b", "json_schema"),
            ("qwen/qwen3.6-27b", "function_calling"),
        )

        for model, expected_method in scenarios:
            with self.subTest(model=model):
                chat_model = _FakeChatModel(
                    {
                        "raw": object(),
                        "parsed": {"groups": [], "rejected": []},
                        "parsing_error": None,
                    }
                )
                adapter = LangChainGroqAdjudicationAdapter(
                    model=model, chat_model=chat_model
                )

                adapter.invoke(
                    AdjudicationRequest(
                        role="proposer",
                        prompt="proposer prompt",
                        members=(),
                    )
                )

                self.assertEqual(chat_model.schema["title"], "ClusterDecision")
                self.assertEqual(chat_model.options["method"], expected_method)

    def test_adapter_requests_provider_schema_without_exposing_disallowed_evidence(self) -> None:
        parsed = {
            "groups": [
                {
                    "name": "Home Air Purification",
                    "page_ids": [101, 102],
                    "rationale": "Shared products.",
                }
            ],
            "rejected": [],
        }
        chat_model = _FakeChatModel(
            {"raw": object(), "parsed": parsed, "parsing_error": None}
        )
        adapter = LangChainGroqAdjudicationAdapter(
            model="fixture/model", chat_model=chat_model
        )
        request = AdjudicationRequest(
            role="proposer",
            prompt="proposer prompt",
            members=(
                {
                    "page_id": 101,
                    "canonical_title": "Air purifier",
                    "lead": "Lead.",
                    "selected_categories": ["Air filters"],
                },
            ),
        )

        result = adapter.invoke(request)

        self.assertEqual(result, parsed)
        self.assertEqual(
            chat_model.options,
            {"method": "function_calling", "include_raw": True},
        )
        messages = chat_model.runnable.messages
        self.assertIsInstance(messages, list)
        model_payload = json.loads(messages[1].content)
        self.assertEqual(model_payload["members"], list(request.members))
        self.assertNotIn("traffic", json.dumps(model_payload).lower())
        self.assertNotIn("chain-of-thought", json.dumps(model_payload).lower())

    def test_schema_parse_failure_is_returned_for_deterministic_validation(self) -> None:
        raw_message = type("RawMessage", (), {"content": "not valid JSON"})()
        chat_model = _FakeChatModel(
            {
                "raw": raw_message,
                "parsed": None,
                "parsing_error": ValueError("invalid schema"),
            }
        )
        adapter = LangChainGroqAdjudicationAdapter(
            model="fixture/model", chat_model=chat_model
        )

        output = adapter.invoke(
            AdjudicationRequest(
                role="critic",
                prompt="critic prompt",
                members=(),
            )
        )

        self.assertEqual(output, "not valid JSON")


if __name__ == "__main__":
    unittest.main()
