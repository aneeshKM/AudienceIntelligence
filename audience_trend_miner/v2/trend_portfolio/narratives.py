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
SUMMARY_TEMPLATE = (
    "Attention to {name} topics {direction_word} in the supplied comparison."
)
SUDDEN_SUMMARY_TEMPLATE = (
    "Attention to {name} topics was suddenly trending in the supplied comparison."
)
COMMERCIAL_INTERPRETATION_TEMPLATE = (
    "{category_text} brands may find the supplied topic group commercially relevant."
)
BUYING_POWER_RATIONALE_TEMPLATE = (
    "The {rating} rating is a qualitative assessment based on the supplied "
    "topics' relevance to {category_text}."
)
NARRATIVE_PROMPT = f"""You write bounded commercial copy for one selected audience trend. Use only the supplied evidence and return exactly the six requested fields. Copy source_cluster_name exactly as name. When suddenly_trending is true, use this summary exactly: '{SUDDEN_SUMMARY_TEMPLATE}' Otherwise use exactly: '{SUMMARY_TEMPLATE}' Choose only schema-listed brand categories and a buying-power rating. For commercial_interpretation use exactly: '{COMMERCIAL_INTERPRETATION_TEMPLATE}' For buying_power_rationale use exactly: '{BUYING_POWER_RATIONALE_TEMPLATE}' Do not return or alter deterministic facts, make any other claim, or provide hidden reasoning."""

_PORTFOLIO_SCHEMA_PATH = (
    Path(__file__).with_name("schemas") / "trend-portfolio.schema.json"
)
_PORTFOLIO_SCHEMA = json.loads(_PORTFOLIO_SCHEMA_PATH.read_text(encoding="utf-8"))
NARRATIVE_SCHEMA = cast(
    dict[str, object],
    _PORTFOLIO_SCHEMA["$defs"]["narrative"],
)
PROVIDER_NARRATIVE_SCHEMA = deepcopy(NARRATIVE_SCHEMA)
cast(
    dict[str, object],
    cast(dict[str, object], PROVIDER_NARRATIVE_SCHEMA["properties"])[
        "brand_categories"
    ],
).pop("uniqueItems", None)

_PROHIBITED_CLAIMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "causation",
        re.compile(
            r"\b(caus(?:e|ed|es|ing)|because of|due to|led to|resulted in|"
            r"driv(?:e|es|en|ing)|prompt(?:s|ed|ing)|trigger(?:s|ed|ing)|explains?|"
            r"spurr(?:ed|ing|s)|fuel(?:s|ed|ing)|spark(?:s|ed|ing)|attributable to|"
            r"stems? from|owing to)\b",
            re.I,
        ),
    ),
    (
        "reader identity",
        re.compile(
            r"\b(readers?|viewers?|users?|homeowners?|shoppers?|buyers?|purchasers?|"
            r"consumers|fans?|enthusiasts?)\b|"
            r"\b(?:people|individuals?|households?)\s+(?:are|who|with|seeking|planning)\b|"
            r"\baudience\s+(?:consists?|comprises?|is|are)\b",
            re.I,
        ),
    ),
    (
        "income",
        re.compile(
            r"\b(income|salary|wealthy|affluent|high[- ]net[- ]worth|"
            r"high[- ]earners?|low[- ]earners?|disposable income|earning power|"
            r"prosperous|well[- ]off|upper[- ]income)\b",
            re.I,
        ),
    ),
    (
        "intent",
        re.compile(
            r"\b(intend(?:s|ed|ing)?|planning to (?:buy|purchase)|purchase intent|"
            r"shopping for|ready to (?:buy|purchase)|want(?:s|ed)? to (?:buy|purchase)|"
            r"seek(?:s|ing)? to (?:buy|purchase|acquire)|"
            r"aim(?:s|ed|ing)? to (?:buy|purchase|acquire)|looking to (?:buy|purchase)|"
            r"interested in (?:buying|purchasing))\b",
            re.I,
        ),
    ),
    (
        "prediction",
        re.compile(
            r"\b(will|forecast(?:s|ed|ing)?|predict(?:s|ed|ion|ing)?|likely to|"
            r"expected to|set to|poised to|going to|future demand|anticipat(?:e|es|ed|ing)|"
            r"(?:should|could|may|might) (?:expand|grow|increase|rise|decline|fall|contract))\b",
            re.I,
        ),
    ),
)

_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|"
    r"thousand|million|billion|percent|\d[\d,.]*%?|doubled|tripled|halved|"
    r"surged|soared|spiked|plummeted|collapsed|record|unprecedented|massive|"
    r"dramatic|sharp|significant|substantial|slight|modest|high|low)"
)
_TRAFFIC_TERM = r"(?:traffic|views?|pageviews?|visits?|attention)"
_INVENTED_TRAFFIC = re.compile(
    rf"(?:\b{_TRAFFIC_TERM}\b[^.!?]{{0,50}}\b{_NUMBER_WORD}\b|"
    rf"\b{_NUMBER_WORD}\b[^.!?]{{0,50}}\b{_TRAFFIC_TERM}\b)",
    re.I,
)


# Carry code-owned facts into one isolated narrative generation call.
@dataclass(frozen=True)
class NarrativeRequest:
    prompt: str
    evidence: dict[str, object]


# Define the model boundary for audience narrative generation.
class NarrativeAdapter(Protocol):
    model: str

    # Invoke the configured backend and return its response.
    def invoke(self, request: NarrativeRequest) -> object: ...


# Define how the portfolio stage obtains per-audience narrative adapters.
class NarrativeAdapterFactory(Protocol):
    # Return the configured model name.
    @property
    def model(self) -> str: ...

    # Return the integration name.
    @property
    def integration_name(self) -> str: ...

    # Build the adapter for one stage work item.
    def adapter_for(
        self, cluster_index: int, cluster_id: str
    ) -> NarrativeAdapter: ...


# Record one generated narrative attempt and its validation errors.
@dataclass(frozen=True)
class NarrativeAttempt:
    attempt: int
    delivery_status: str
    validation_status: str
    output: object
    errors: tuple[str, ...]

    # Convert this value into artifact data.
    def record(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "delivery_status": self.delivery_status,
            "validation_status": self.validation_status,
            "output": deepcopy(self.output),
            "errors": list(self.errors),
        }


# Retry one isolated narrative until its generated fields are safe.
def generate_validated_narrative(
    adapter: NarrativeAdapter,
    evidence: dict[str, object],
) -> tuple[dict[str, object], tuple[NarrativeAttempt, ...]]:
    """Retry one isolated narrative until its generated fields are safe."""
    # Every attempt receives the same defensive copy of code-owned evidence.
    attempts: list[NarrativeAttempt] = []
    request = NarrativeRequest(prompt=NARRATIVE_PROMPT, evidence=deepcopy(evidence))
    # Delivery and content validation are recorded separately for auditability.
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
        errors = narrative_validation_errors(output, evidence)
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
        # The first valid result wins; invalid raw outputs remain attempt evidence only.
        if not errors:
            return cast(dict[str, object], output), tuple(attempts)
    raise NarrativeExhausted(tuple(attempts))


# Expose all bounded attempts when safe narrative generation fails.
class NarrativeExhausted(Exception):
    # Initialize the NarrativeExhausted.
    def __init__(self, attempts: tuple[NarrativeAttempt, ...]) -> None:
        super().__init__("narrative attempts exhausted")
        self.attempts = attempts


# Return deterministic validation errors for generated narrative fields.
def narrative_validation_errors(
    output: object,
    evidence: dict[str, object],
) -> tuple[str, ...]:
    # Structural schema errors take precedence because later checks assume exact fields.
    validator = jsonschema.Draft202012Validator(NARRATIVE_SCHEMA)
    schema_errors = sorted(
        (error.message.lower() for error in validator.iter_errors(output)),
    )
    if schema_errors:
        return tuple(schema_errors)
    assert isinstance(output, dict)
    # Remove the source name before claim scanning so names containing flagged words
    # do not create false positives.
    text = " ".join(_strings_in(output))
    source_name = evidence.get("source_cluster_name")
    if isinstance(source_name, str):
        text = text.replace(source_name, "")
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
    return (
        *claim_errors,
        *traffic_errors,
        *_bounded_template_errors(cast(dict[str, object], output), evidence),
    )


# Validate persisted narrative attempts through the owning module contract.
def validate_completed_narrative_evidence(
    record: dict[str, object],
    *,
    expected_model_input: dict[str, object],
    expected_model: str,
    expected_narrative: dict[str, object],
) -> None:
    """Validate persisted narrative attempts through the owning module contract."""
    # Prompt, input, model, numbering, and bounded attempt count are provenance facts.
    attempts = record.get("attempts")
    if (
        record.get("prompt") != NARRATIVE_PROMPT
        or record.get("model_input") != expected_model_input
        or record.get("model") != expected_model
        or not isinstance(attempts, list)
        or not attempts
        or len(attempts) > MAX_NARRATIVE_ATTEMPTS
    ):
        raise V2ContractError("Trend Portfolio narrative evidence is inconsistent")
    # Re-run validation against every delivered output instead of trusting status flags.
    for attempt_number, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict) or attempt.get("attempt") != attempt_number:
            raise V2ContractError("Trend Portfolio narrative evidence is inconsistent")
        delivery_status = attempt.get("delivery_status")
        validation_status = attempt.get("validation_status")
        output = attempt.get("output")
        errors = attempt.get("errors")
        if not isinstance(errors, list) or not all(
            isinstance(error, str) for error in errors
        ):
            raise V2ContractError("Trend Portfolio narrative evidence is inconsistent")
        if delivery_status == "error":
            consistent = validation_status == "not_run" and output is None and bool(errors)
        elif delivery_status == "delivered":
            expected_errors = narrative_validation_errors(output, expected_model_input)
            consistent = (
                validation_status == ("invalid" if expected_errors else "valid")
                and errors == list(expected_errors)
            )
        else:
            consistent = False
        if not consistent:
            raise V2ContractError("Trend Portfolio narrative evidence is inconsistent")
    # Only the final attempt may be valid, and it must equal the published narrative.
    final_attempt = cast(dict[str, object], attempts[-1])
    if (
        final_attempt["validation_status"] != "valid"
        or final_attempt["output"] != expected_narrative
        or any(
            cast(dict[str, object], attempt)["validation_status"] == "valid"
            for attempt in attempts[:-1]
        )
    ):
        raise V2ContractError("Trend Portfolio narrative evidence is inconsistent")


# Detect prohibited or invented claims in narrative copy.
def _bounded_template_errors(
    output: dict[str, object],
    evidence: dict[str, object],
) -> tuple[str, ...]:
    # Most generated prose is constrained to deterministic templates; the model mainly
    # selects bounded categories and a buying-power rating.
    name = evidence["source_cluster_name"]
    direction = evidence["direction"]
    categories = cast(list[str], output["brand_categories"])
    rating = output["buying_power_rating"]
    category_text = ", ".join(categories)
    suddenly_trending = evidence.get("suddenly_trending") is True
    direction_word = "rose" if direction == "robust_growth" else "declined"
    expected_summary = (
        SUDDEN_SUMMARY_TEMPLATE.format(name=name)
        if suddenly_trending
        else SUMMARY_TEMPLATE.format(name=name, direction_word=direction_word)
    )
    allowed_summaries = {expected_summary}
    if direction == "robust_growth":
        allowed_summaries.add(
            SUMMARY_TEMPLATE.format(name=name, direction_word="growing")
        )
    elif direction == "robust_shrinking":
        allowed_summaries.add(
            SUMMARY_TEMPLATE.format(name=name, direction_word="shrinking")
        )
    expected = {
        "name": name,
        "commercial_interpretation": COMMERCIAL_INTERPRETATION_TEMPLATE.format(
            category_text=category_text,
        ),
        "buying_power_rationale": BUYING_POWER_RATIONALE_TEMPLATE.format(
            rating=rating,
            category_text=category_text,
        ),
    }
    summary_errors = (
        ("summary is outside the bounded narrative template",)
        if output["summary"] not in allowed_summaries
        else ()
    )
    return (
        *summary_errors,
        *tuple(
        f"{field} is outside the bounded narrative template"
        for field, expected_value in expected.items()
        if output[field] != expected_value
        ),
    )


# Yield all strings nested inside a JSON-compatible value.
def _strings_in(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings_in(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings_in(item)]
    return []


# Generate structured narrative fields through Groq.
class LangChainGroqNarrativeAdapter:
    # Initialize the LangChainGroqNarrativeAdapter.
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

    # Invoke the configured backend and return its response.
    def invoke(self, request: NarrativeRequest) -> object:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._chat_model.with_structured_output(
            PROVIDER_NARRATIVE_SCHEMA,
            method="json_schema",
            include_raw=True,
            strict=True,
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


# Share production narrative model configuration across audiences.
@dataclass(frozen=True)
class ProductionNarrativeAdapterFactory:
    model: str = DEFAULT_NARRATIVE_MODEL
    integration_name: str = "langchain-groq"

    # Build the adapter for one stage work item.
    def adapter_for(
        self, cluster_index: int, cluster_id: str
    ) -> LangChainGroqNarrativeAdapter:
        del cluster_index, cluster_id
        return LangChainGroqNarrativeAdapter(model=self.model)


# Consume deterministic narrative responses for one audience.
class _ScriptedNarrativeAdapter:
    # Initialize the _ScriptedNarrativeAdapter.
    def __init__(self, responses: list[object], *, model: str) -> None:
        self.model = model
        self._responses = deepcopy(responses)

    # Invoke the configured backend and return its response.
    def invoke(self, request: NarrativeRequest) -> object:
        del request
        if not self._responses:
            raise RuntimeError("fixture narrative responses exhausted")
        response = self._responses.pop(0)
        if isinstance(response, dict) and set(response) == {"delivery_error"}:
            raise RuntimeError(str(response["delivery_error"]))
        return deepcopy(response)


# Load and match fixture narratives by source cluster identity.
@dataclass(frozen=True)
class FrozenNarrativeAdapterFactory:
    model: str
    _clusters: tuple[dict[str, object], ...]
    integration_name: str = "fixture"

    # Create an instance from file.
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

    # Build the adapter for one stage work item.
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
