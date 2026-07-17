from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from audience_trend_miner.v2.cluster_adjudication.graph import (
    AdjudicationRequest,
)


DEFAULT_CLUSTER_MODEL = "openai/gpt-oss-20b"


DECISION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["groups", "rejected"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "page_ids", "rationale"],
                "properties": {
                    "name": {"type": "string"},
                    "page_ids": {"type": "array", "items": {"type": "integer"}},
                    "rationale": {"type": "string"},
                },
            },
        },
        "rejected": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_id", "reason"],
                "properties": {
                    "page_id": {"type": "integer"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}

CRITIQUE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["approved", "challenges"],
    "properties": {
        "approved": {"type": "boolean"},
        "challenges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["dimension", "message", "page_ids", "required_action"],
                "properties": {
                    "dimension": {
                        "enum": [
                            "semantic_coherence",
                            "commercial_meaning",
                            "brand_safety",
                            "naming",
                            "evidence_support",
                        ]
                    },
                    "message": {"type": "string"},
                    "page_ids": {"type": "array", "items": {"type": "integer"}},
                    "required_action": {"enum": ["revise", "reject"]},
                },
            },
        },
    },
}


class LangChainGroqAdjudicationAdapter:
    """Invoke one configured Groq chat model through the LangChain integration."""

    def __init__(self, *, model: str, chat_model: Any | None = None) -> None:
        self.model = model
        if chat_model is None:
            from langchain_groq import ChatGroq

            chat_model = ChatGroq(
                model=model,
                temperature=0,
                timeout=60,
                max_retries=0,
            )
        self._chat_model = chat_model

    def invoke(self, request: AdjudicationRequest) -> object:
        schema = CRITIQUE_SCHEMA if request.role == "critic" else DECISION_SCHEMA
        model_input = {
            "members": list(request.members),
            "proposal": request.proposal,
            "validation_errors": list(request.validation_errors),
            "critique": request.critique,
        }
        messages = [
            SystemMessage(content=request.prompt),
            HumanMessage(
                content=json.dumps(
                    model_input,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
        ]
        response = self._chat_model.with_structured_output(
            schema,
            method="json_schema",
            include_raw=True,
        ).invoke(messages)
        if not isinstance(response, dict):
            return response
        if response.get("parsing_error") is None:
            return response.get("parsed")
        raw = response.get("raw")
        return getattr(raw, "content", raw)


@dataclass(frozen=True)
class ProductionStageAdapterFactory:
    model: str = DEFAULT_CLUSTER_MODEL
    integration_name: str = "langchain-groq"

    def adapter_for(
        self, cluster_index: int, preliminary_cluster: dict[str, object]
    ) -> LangChainGroqAdjudicationAdapter:
        del cluster_index, preliminary_cluster
        return LangChainGroqAdjudicationAdapter(model=self.model)
