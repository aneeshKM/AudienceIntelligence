from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Protocol, cast

import jsonschema

from audience_trend_miner.v2.shared import V2ContractError


DEFAULT_NARRATIVE_MODEL = "openai/gpt-oss-120b"
MAX_NARRATIVE_ATTEMPTS = 3
NARRATIVE_PROMPT = """You write bounded commercial copy for one selected audience trend. Use only the supplied evidence. Return exactly the six requested fields. Do not return or alter direction, traffic, percentage change, coverage, confidence, or Impact Score. Do not claim causation, reader identity, income, intent, prediction, or future behavior. Do not provide hidden reasoning or chain-of-thought."""

NARRATIVE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name",
        "summary",
        "commercial_interpretation",
        "brand_categories",
        "buying_power_rating",
        "buying_power_rationale",
    ],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "summary": {"type": "string", "minLength": 1},
        "commercial_interpretation": {"type": "string", "minLength": 1},
        "brand_categories": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "buying_power_rating": {"enum": ["high", "medium", "low"]},
        "buying_power_rationale": {"type": "string", "minLength": 1},
    },
}

_PROHIBITED_CLAIMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "causation",
        re.compile(
            r"\b(caus(?:e|ed|es|ing)|because of|due to|led to|resulted in|"
            r"driv(?:e|es|en|ing)|prompt(?:s|ed|ing)|trigger(?:s|ed|ing)|explains?)\b",
            re.I,
        ),
    ),
    (
        "reader identity",
        re.compile(
            r"\b(readers?|viewers?|users?|homeowners?|shoppers?|buyers?|fans?|"
            r"enthusiasts?)\b|\baudience\s+(?:consists?|comprises?|is|are)\b",
            re.I,
        ),
    ),
    (
        "income",
        re.compile(
            r"\b(income|salary|wealthy|affluent|high[- ]net[- ]worth|"
            r"high[- ]earners?|low[- ]earners?|disposable income|earning power)\b",
            re.I,
        ),
    ),
    (
        "intent",
        re.compile(
            r"\b(intend(?:s|ed|ing)?|planning to (?:buy|purchase)|purchase intent|"
            r"shopping for|ready to (?:buy|purchase)|want(?:s|ed)? to (?:buy|purchase)|"
            r"seeking to (?:buy|purchase)|interested in (?:buying|purchasing))\b",
            re.I,
        ),
    ),
    (
        "prediction",
        re.compile(
            r"\b(will|forecast(?:s|ed|ing)?|predict(?:s|ed|ion|ing)?|likely to|"
            r"expected to|set to|poised to|going to|future demand)\b",
            re.I,
        ),
    ),
)

_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|"
    r"thousand|million|billion|percent|\d[\d,.]*%?|doubled|tripled|halved)"
)
_TRAFFIC_TERM = r"(?:traffic|views?|pageviews?|visits?|attention)"
_INVENTED_TRAFFIC = re.compile(
    rf"(?:\b{_TRAFFIC_TERM}\b[^.!?]{{0,50}}\b{_NUMBER_WORD}\b|"
    rf"\b{_NUMBER_WORD}\b[^.!?]{{0,50}}\b{_TRAFFIC_TERM}\b)",
    re.I,
)


@dataclass(frozen=True)
class NarrativeRequest:
    prompt: str
    evidence: dict[str, object]


class NarrativeAdapter(Protocol):
    model: str

    def invoke(self, request: NarrativeRequest) -> object: ...


class NarrativeAdapterFactory(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def integration_name(self) -> str: ...

    def adapter_for(
        self, cluster_index: int, cluster_id: str
    ) -> NarrativeAdapter: ...


@dataclass(frozen=True)
class NarrativeAttempt:
    attempt: int
    delivery_status: str
    validation_status: str
    output: object
    errors: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "delivery_status": self.delivery_status,
            "validation_status": self.validation_status,
            "output": deepcopy(self.output),
            "errors": list(self.errors),
        }


def generate_validated_narrative(
    adapter: NarrativeAdapter,
    evidence: dict[str, object],
) -> tuple[dict[str, object], tuple[NarrativeAttempt, ...]]:
    """Retry one isolated narrative until its generated fields are safe."""
    attempts: list[NarrativeAttempt] = []
    request = NarrativeRequest(prompt=NARRATIVE_PROMPT, evidence=deepcopy(evidence))
    for attempt_number in range(1, MAX_NARRATIVE_ATTEMPTS + 1):
        try:
            output = adapter.invoke(request)
        except Exception as error:
            attempts.append(
                NarrativeAttempt(
                    attempt_number,
                    "error",
                    "not_run",
                    None,
                    (f"{type(error).__name__}: {error}",),
                )
            )
            continue
        errors = narrative_validation_errors(output)
        status = "invalid" if errors else "valid"
        attempts.append(
            NarrativeAttempt(
                attempt_number,
                "delivered",
                status,
                deepcopy(output),
                errors,
            )
        )
        if not errors:
            return cast(dict[str, object], output), tuple(attempts)
    raise NarrativeExhausted(tuple(attempts))


class NarrativeExhausted(Exception):
    def __init__(self, attempts: tuple[NarrativeAttempt, ...]) -> None:
        super().__init__("narrative attempts exhausted")
        self.attempts = attempts


def narrative_validation_errors(output: object) -> tuple[str, ...]:
    validator = jsonschema.Draft202012Validator(NARRATIVE_SCHEMA)
    schema_errors = sorted(
        (error.message.lower() for error in validator.iter_errors(output)),
    )
    if schema_errors:
        return tuple(schema_errors)
    assert isinstance(output, dict)
    text = " ".join(_strings_in(output))
    claim_errors = tuple(
        f"prohibited {claim} claim"
        for claim, pattern in _PROHIBITED_CLAIMS
        if pattern.search(text)
    )
    traffic_errors = (
        ("prohibited invented traffic claim",)
        if _INVENTED_TRAFFIC.search(text)
        else ()
    )
    return (*claim_errors, *traffic_errors)


def _strings_in(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings_in(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings_in(item)]
    return []


class LangChainGroqNarrativeAdapter:
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

    def invoke(self, request: NarrativeRequest) -> object:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._chat_model.with_structured_output(
            NARRATIVE_SCHEMA,
            method="json_schema",
            include_raw=True,
        ).invoke(
            [
                SystemMessage(content=request.prompt),
                HumanMessage(
                    content=json.dumps(
                        request.evidence,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
            ]
        )
        if not isinstance(response, dict):
            return response
        if response.get("parsing_error") is None:
            return response.get("parsed")
        raw = response.get("raw")
        return getattr(raw, "content", raw)


@dataclass(frozen=True)
class ProductionNarrativeAdapterFactory:
    model: str = DEFAULT_NARRATIVE_MODEL
    integration_name: str = "langchain-groq"

    def adapter_for(
        self, cluster_index: int, cluster_id: str
    ) -> LangChainGroqNarrativeAdapter:
        del cluster_index, cluster_id
        return LangChainGroqNarrativeAdapter(model=self.model)


class _ScriptedNarrativeAdapter:
    def __init__(self, responses: list[object], *, model: str) -> None:
        self.model = model
        self._responses = deepcopy(responses)

    def invoke(self, request: NarrativeRequest) -> object:
        del request
        if not self._responses:
            raise RuntimeError("fixture narrative responses exhausted")
        response = self._responses.pop(0)
        if isinstance(response, dict) and set(response) == {"delivery_error"}:
            raise RuntimeError(str(response["delivery_error"]))
        return deepcopy(response)


@dataclass(frozen=True)
class FrozenNarrativeAdapterFactory:
    model: str
    _clusters: tuple[dict[str, object], ...]
    integration_name: str = "fixture"

    @classmethod
    def from_file(cls, path: Path) -> FrozenNarrativeAdapterFactory:
        try:
            fixture = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise V2ContractError("narrative fixture is unreadable") from error
        if (
            not isinstance(fixture, dict)
            or set(fixture) != {"schema_version", "model", "clusters"}
            or fixture["schema_version"] != "1.0"
            or not isinstance(fixture["model"], str)
            or not fixture["model"]
            or not isinstance(fixture["clusters"], list)
        ):
            raise V2ContractError("narrative fixture has an invalid shape")
        return cls(fixture["model"], tuple(deepcopy(fixture["clusters"])))

    def adapter_for(
        self, cluster_index: int, cluster_id: str
    ) -> _ScriptedNarrativeAdapter:
        try:
            fixture = self._clusters[cluster_index]
        except IndexError as error:
            raise V2ContractError("narrative fixture is missing a selected cluster") from error
        if (
            not isinstance(fixture, dict)
            or set(fixture) != {"cluster_id", "responses"}
            or fixture["cluster_id"] != cluster_id
            or not isinstance(fixture["responses"], list)
        ):
            raise V2ContractError("narrative fixture conflicts with selected clusters")
        return _ScriptedNarrativeAdapter(fixture["responses"], model=self.model)
