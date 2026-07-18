from __future__ import annotations

import json
import unittest

from audience_trend_miner.v2.cluster_adjudication import AdjudicationRequest
from audience_trend_miner.v2.cluster_adjudication.adapters import (
    LangChainGroqAdjudicationAdapter,
)
from audience_trend_miner.v2.shared import V2ContractError


# Provide the fake structured runnable test double.
class _FakeStructuredRunnable:
    # Initialize the _FakeStructuredRunnable.
    def __init__(self, output: object) -> None:
        self.output = output
        self.messages: object = None

    # Return the scripted structured-provider response.
    def invoke(self, messages: object) -> object:
        self.messages = messages
        return self.output


# Provide the fake chat model test double.
class _FakeChatModel:
    # Initialize the _FakeChatModel.
    def __init__(self, output: object) -> None:
        self.runnable = _FakeStructuredRunnable(output)
        self.schema: object = None
        self.options: dict[str, object] = {}

    # Return with structured output.
    def with_structured_output(self, schema: object, **options: object) -> _FakeStructuredRunnable:
        self.schema = schema
        self.options = options
        return self.runnable


# Provide the invalid structured chat model test double.
class _InvalidStructuredChatModel:
    # Return with structured output.
    def with_structured_output(self, schema: object, **options: object) -> object:
        del schema, options
        raise ValueError("unsupported structured output schema")


# Provide the permission denied runnable test double.
class _PermissionDeniedRunnable:
    # Raise the scripted provider permission failure.
    def invoke(self, messages: object) -> object:
        del messages
        error = RuntimeError("model blocked at project level")
        error.status_code = 403
        raise error


# Provide the permission denied chat model test double.
class _PermissionDeniedChatModel:
    # Return with structured output.
    def with_structured_output(self, schema: object, **options: object) -> object:
        del schema, options
        return _PermissionDeniedRunnable()


# Group tests for lang chain groq adjudication adapter behavior.
class LangChainGroqAdjudicationAdapterTest(unittest.TestCase):
    # Verify: project blocked model is a fatal configuration error.
    def test_project_blocked_model_is_a_fatal_configuration_error(self) -> None:
        adapter = LangChainGroqAdjudicationAdapter(
            model="groq/compound-mini",
            chat_model=_PermissionDeniedChatModel(),
        )

        with self.assertRaisesRegex(
            V2ContractError, "unavailable to this project"
        ):
            adapter.invoke(
                AdjudicationRequest(
                    role="proposer",
                    prompt="proposer prompt",
                    members=(),
                )
            )

    # Verify: invalid provider contract is a fatal configuration error.
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

    # Verify: adapter selects supported structured output contract by model.
    def test_adapter_selects_supported_structured_output_contract_by_model(self) -> None:
        scenarios = (
            (
                "openai/gpt-oss-20b",
                {"method": "json_schema", "include_raw": True, "strict": True},
            ),
            (
                "qwen/qwen3.6-27b",
                {"method": "function_calling", "include_raw": True},
            ),
            (
                "groq/compound-mini",
                {"method": "json_mode", "include_raw": True},
            ),
        )

        for model, expected_options in scenarios:
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
                self.assertEqual(chat_model.options, expected_options)

    # Verify: compound json mode receives the exact schema in its prompt.
    def test_compound_json_mode_receives_the_exact_schema_in_its_prompt(self) -> None:
        parsed = {"approved": True, "challenges": []}
        chat_model = _FakeChatModel(
            {"raw": object(), "parsed": parsed, "parsing_error": None}
        )
        adapter = LangChainGroqAdjudicationAdapter(
            model="groq/compound-mini", chat_model=chat_model
        )

        result = adapter.invoke(
            AdjudicationRequest(
                role="critic",
                prompt="critic prompt",
                members=(),
            )
        )

        self.assertEqual(result, parsed)
        messages = chat_model.runnable.messages
        self.assertIn("Return only one JSON object", messages[0].content)
        self.assertIn('"approved":{"type":"boolean"}', messages[0].content)

    # Verify: adapter requests provider schema without exposing disallowed evidence.
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

    # Verify: schema parse failure is returned for deterministic validation.
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
