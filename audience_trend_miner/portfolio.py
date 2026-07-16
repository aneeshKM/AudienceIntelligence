from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import random
import re
import time
from typing import Callable, Literal

import jsonschema

from audience_trend_miner.classification import StructuredGenerator
from audience_trend_miner.refinement import AcceptedAudience, ClusterRefinementResult
from audience_trend_miner.trends import trend_score
from audience_trend_miner.wikimedia import CanonicalArticle


PORTFOLIO_LIMIT = 10
AUDIENCE_ASSESSMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name", "description", "purchase_intent", "transaction_value",
        "category_breadth", "brand_safety", "brand_categories", "rationale",
        "name_is_targetable_group", "causal_claims_are_hypotheses",
        "rationale_avoids_wealth_inference",
    ],
    "properties": {
        "name": {"type": "string", "minLength": 3},
        "description": {"type": "string", "minLength": 1},
        "purchase_intent": {"type": "integer", "minimum": 1, "maximum": 3},
        "transaction_value": {"type": "integer", "minimum": 1, "maximum": 3},
        "category_breadth": {"type": "integer", "minimum": 1, "maximum": 3},
        "brand_safety": {"type": "integer", "minimum": 1, "maximum": 3},
        "brand_categories": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "rationale": {"type": "string", "minLength": 1},
        "name_is_targetable_group": {"const": True},
        "causal_claims_are_hypotheses": {"const": True},
        "rationale_avoids_wealth_inference": {"const": True},
    },
}


@dataclass(frozen=True)
class PortfolioAttempt:
    attempt: int
    raw_output: object | None
    validation_valid: bool
    error: str | None


@dataclass(frozen=True)
class BuyingPowerScores:
    purchase_intent: int
    transaction_value: int
    category_breadth: int
    brand_safety: int


@dataclass(frozen=True)
class PortfolioAudience:
    source_component_id: int
    page_ids: tuple[int, ...]
    name: str
    description: str
    previous_window_views: int
    current_window_views: int
    trend_score: float
    size_basis_points: int
    estimated_size_index: float
    potential_buying_power: Literal["high", "medium", "low"]
    buying_power_scores: BuyingPowerScores
    brand_categories: tuple[str, ...]
    buying_power_rationale: str


@dataclass(frozen=True)
class PortfolioAssessment:
    source_component_id: int
    prompt: str
    attempts: tuple[PortfolioAttempt, ...]


@dataclass(frozen=True)
class PortfolioResult:
    audiences: tuple[PortfolioAudience, ...]
    assessments: tuple[PortfolioAssessment, ...]


def build_portfolio(
    refinement: ClusterRefinementResult,
    articles: tuple[CanonicalArticle, ...],
    generator: StructuredGenerator,
    *,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> PortfolioResult:
    articles_by_id = {article.page_id: article for article in articles}
    ranked = sorted(
        refinement.accepted,
        key=lambda audience: (
            -_traffic(audience, articles_by_id)[2], audience.source_component_id
        ),
    )[:PORTFOLIO_LIMIT]
    basis_points = _allocate_basis_points(
        [_traffic(item, articles_by_id)[1] for item in ranked]
    )
    audiences: list[PortfolioAudience] = []
    assessments: list[PortfolioAssessment] = []
    for accepted, size_basis_points in zip(ranked, basis_points, strict=True):
        previous, current, score = _traffic(accepted, articles_by_id)
        prompt = _portfolio_prompt(accepted, articles_by_id, previous, current)
        payload, attempts = _generate_assessment(
            prompt, generator, sleep=sleep, jitter=jitter
        )
        assessments.append(
            PortfolioAssessment(accepted.source_component_id, prompt, attempts)
        )
        if payload is None:
            continue
        scores = BuyingPowerScores(
            payload["purchase_intent"], payload["transaction_value"],
            payload["category_breadth"], payload["brand_safety"],
        )
        audiences.append(
            PortfolioAudience(
                accepted.source_component_id,
                accepted.page_ids,
                payload["name"],
                payload["description"],
                previous,
                current,
                score,
                size_basis_points,
                size_basis_points / 100,
                _buying_power(scores),
                scores,
                tuple(payload["brand_categories"]),
                payload["rationale"],
            )
        )
    if len(audiences) != len(ranked):
        # Failed closed audiences must not leave the retained shares below 100%.
        retained = [item.current_window_views for item in audiences]
        reallocated = _allocate_basis_points(retained)
        audiences = [
            PortfolioAudience(
                **{
                    **item.__dict__,
                    "size_basis_points": points,
                    "estimated_size_index": points / 100,
                }
            )
            for item, points in zip(audiences, reallocated, strict=True)
        ]
    return PortfolioResult(tuple(audiences), tuple(assessments))


def _traffic(
    audience: AcceptedAudience,
    articles_by_id: dict[int, CanonicalArticle],
) -> tuple[int, int, float]:
    try:
        members = [articles_by_id[page_id] for page_id in audience.page_ids]
    except KeyError as error:
        raise ValueError(f"accepted audience references unknown page ID {error.args[0]}") from error
    previous = sum(item.previous_window_views for item in members)
    current = sum(item.current_window_views for item in members)
    return previous, current, trend_score(current, previous)


def _allocate_basis_points(traffic: list[int]) -> list[int]:
    if not traffic:
        return []
    total = sum(traffic)
    if total <= 0:
        raise ValueError("accepted audience traffic must be positive")
    quotas = [Fraction(value * 10_000, total) for value in traffic]
    allocated = [quota.numerator // quota.denominator for quota in quotas]
    remaining = 10_000 - sum(allocated)
    order = sorted(
        range(len(quotas)),
        key=lambda index: (-(quotas[index] - allocated[index]), index),
    )
    for index in order[:remaining]:
        allocated[index] += 1
    return allocated


def _generate_assessment(
    prompt: str,
    generator: StructuredGenerator,
    *,
    sleep: Callable[[float], None],
    jitter: Callable[[], float],
) -> tuple[dict[str, object] | None, tuple[PortfolioAttempt, ...]]:
    attempts: list[PortfolioAttempt] = []
    for attempt_number in range(1, 4):
        raw: object | None = None
        try:
            raw = generator.generate(prompt, AUDIENCE_ASSESSMENT_SCHEMA)
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            jsonschema.validate(parsed, AUDIENCE_ASSESSMENT_SCHEMA)
            assert isinstance(parsed, dict)
            word_count = len(str(parsed["name"]).split())
            if not 2 <= word_count <= 5:
                raise ValueError("audience name must contain two to five words")
            copy = f"{parsed['description']} {parsed['rationale']}"
            if re.search(
                r"\b(?:wealth|wealthy|income|affluent|disposable income)\b",
                copy,
                re.IGNORECASE,
            ):
                raise ValueError("portfolio copy must not infer reader wealth")
            if re.search(
                r"\b(?:caused|because|due to|driven by)\b",
                str(parsed["description"]),
                re.IGNORECASE,
            ) and not re.search(
                r"\b(?:may|might|could|suggest|hypothes|possibly|perhaps)\w*\b",
                str(parsed["description"]),
                re.IGNORECASE,
            ):
                raise ValueError("unsupported causal explanations must be hypotheses")
            attempts.append(PortfolioAttempt(attempt_number, raw, True, None))
            return parsed, tuple(attempts)
        except Exception as error:
            attempts.append(PortfolioAttempt(
                attempt_number, raw, False, f"{type(error).__name__}: {error}"
            ))
            if attempt_number < 3:
                sleep((2 ** (attempt_number - 1)) + jitter())
    return None, tuple(attempts)


def _buying_power(scores: BuyingPowerScores) -> Literal["high", "medium", "low"]:
    total = sum((scores.purchase_intent, scores.transaction_value,
                 scores.category_breadth, scores.brand_safety))
    if 10 <= total <= 12 and scores.brand_safety == 3:
        return "high"
    if 7 <= total <= 11 and scores.brand_safety >= 2:
        return "medium"
    return "low"


def _portfolio_prompt(
    audience: AcceptedAudience,
    articles_by_id: dict[int, CanonicalArticle],
    previous: int,
    current: int,
) -> str:
    members = "\n".join(
        f"- {articles_by_id[page_id].canonical_title}: "
        f"{articles_by_id[page_id].extract}"
        for page_id in audience.page_ids
    )
    return (
        "Create marketer-facing copy and assess buying power for this accepted audience. "
        "The name must identify a targetable group of people in two to five words, not "
        "merely label a topic. State observed traffic as fact, but frame unsupported "
        "causal explanations only as hypotheses (for example, 'suggesting'). Score "
        "purchase intent, typical transaction value, category breadth, and brand safety "
        "from 1 to 3. Name relevant brand categories and ground the rationale in those "
        "scores. Do not infer reader wealth or disposable income.\n\n"
        f"Refined name: {audience.name}\nRefinement rationale: {audience.rationale}\n"
        f"Observed previous-window traffic: {previous}\n"
        f"Observed current-window traffic: {current}\nSupporting articles:\n{members}"
    )
