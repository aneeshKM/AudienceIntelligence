from __future__ import annotations

from dataclasses import dataclass
import json
import random
import time
from typing import Callable, Literal

import jsonschema

from audience_trend_miner.classification import StructuredGenerator
from audience_trend_miner.clustering import CandidateComponent
from audience_trend_miner.wikimedia import CanonicalArticle


CLUSTER_REFINEMENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action", "audiences", "rejected_page_ids", "alternative_matches", "rationale"
    ],
    "properties": {
        "action": {"enum": ["validate", "split", "reject"]},
        "audiences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "page_ids", "rationale"],
                "properties": {
                    "name": {"type": "string", "minLength": 3},
                    "page_ids": {
                        "type": "array",
                        "minItems": 2,
                        "uniqueItems": True,
                        "items": {"type": "integer", "minimum": 1},
                    },
                    "rationale": {"type": "string", "minLength": 1},
                },
            },
        },
        "rejected_page_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "integer", "minimum": 1},
        },
        "alternative_matches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_id", "audience_name", "rationale"],
                "properties": {
                    "page_id": {"type": "integer", "minimum": 1},
                    "audience_name": {"type": "string", "minLength": 1},
                    "rationale": {"type": "string", "minLength": 1},
                },
            },
        },
        "rationale": {"type": "string", "minLength": 1},
    },
}

CLUSTER_SAFETY_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "materially_centered_on_tragedy",
        "materially_centered_on_violent_crime",
        "rationale",
    ],
    "properties": {
        "materially_centered_on_tragedy": {"type": "boolean"},
        "materially_centered_on_violent_crime": {"type": "boolean"},
        "rationale": {"type": "string", "minLength": 1},
    },
}


@dataclass(frozen=True)
class RefinementAttempt:
    attempt: int
    raw_output: object | None
    validation_valid: bool
    error: str | None


@dataclass(frozen=True)
class AlternativeMatch:
    page_id: int
    audience_name: str
    rationale: str


@dataclass(frozen=True)
class ProposedAudience:
    name: str
    page_ids: tuple[int, ...]
    rationale: str


@dataclass(frozen=True)
class ClusterSafetyAssessment:
    audience_name: str
    page_ids: tuple[int, ...]
    prompt: str
    safe: bool
    decision_reason: Literal["accepted", "safety_vetoed", "exhausted_attempts"]
    materially_centered_on_tragedy: bool | None
    materially_centered_on_violent_crime: bool | None
    rationale: str | None
    attempts: tuple[RefinementAttempt, ...]


@dataclass(frozen=True)
class AcceptedAudience:
    name: str
    page_ids: tuple[int, ...]
    rationale: str
    source_component_id: int
    safety: ClusterSafetyAssessment


@dataclass(frozen=True)
class ClusterRefinementDecision:
    component_id: int
    candidate_page_ids: tuple[int, ...]
    prompt: str
    action: Literal["validate", "split", "reject"]
    outcome: Literal[
        "accepted", "rejected", "safety_vetoed", "partially_accepted",
        "exhausted_attempts",
    ]
    rationale: str | None
    proposed_audiences: tuple[ProposedAudience, ...]
    rejected_page_ids: tuple[int, ...]
    alternative_matches: tuple[AlternativeMatch, ...]
    attempts: tuple[RefinementAttempt, ...]
    safety_assessments: tuple[ClusterSafetyAssessment, ...]


@dataclass(frozen=True)
class ClusterRefinementResult:
    accepted: tuple[AcceptedAudience, ...]
    decisions: tuple[ClusterRefinementDecision, ...]
    rejected_standalone_page_ids: tuple[int, ...]


def refine_candidate_clusters(
    components: tuple[CandidateComponent, ...],
    articles: tuple[CanonicalArticle, ...],
    generator: StructuredGenerator,
    *,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> ClusterRefinementResult:
    articles_by_id = {article.page_id: article for article in articles}
    if len(articles_by_id) != len(articles):
        raise ValueError("canonical articles must have unique page IDs")
    component_members = [page_id for item in components for page_id in item.page_ids]
    if len(component_members) != len(set(component_members)):
        raise ValueError("candidate components must have exclusive membership")
    if not set(component_members).issubset(articles_by_id):
        raise ValueError("candidate components reference unknown canonical articles")

    accepted: list[AcceptedAudience] = []
    decisions: list[ClusterRefinementDecision] = []
    for component in components:
        if not component.is_candidate_cluster:
            continue
        decision, component_accepted = _refine_component(
            component, articles_by_id, generator, sleep=sleep, jitter=jitter
        )
        decisions.append(decision)
        accepted.extend(component_accepted)

    accepted_members = [page_id for item in accepted for page_id in item.page_ids]
    if len(accepted_members) != len(set(accepted_members)):
        raise AssertionError("refinement produced duplicate accepted traffic membership")
    return ClusterRefinementResult(
        tuple(accepted),
        tuple(decisions),
        tuple(
            item.page_ids[0]
            for item in components
            if not item.is_candidate_cluster
        ),
    )


def _refine_component(
    component: CandidateComponent,
    articles_by_id: dict[int, CanonicalArticle],
    generator: StructuredGenerator,
    *,
    sleep: Callable[[float], None],
    jitter: Callable[[], float],
) -> tuple[ClusterRefinementDecision, tuple[AcceptedAudience, ...]]:
    prompt = _refinement_prompt(component, articles_by_id)
    parsed, attempts = _generate_with_retries(
        prompt,
        CLUSTER_REFINEMENT_SCHEMA,
        generator,
        validate=lambda candidate: _validate_membership(candidate, component.page_ids),
        sleep=sleep,
        jitter=jitter,
    )

    if parsed is None:
        return (
            ClusterRefinementDecision(
                component.component_id, component.page_ids, prompt,
                "reject", "exhausted_attempts", None, (), component.page_ids,
                (), tuple(attempts), (),
            ),
            (),
        )

    proposals = tuple(
        ProposedAudience(
            str(item["name"]), tuple(item["page_ids"]), str(item["rationale"])
        )
        for item in parsed["audiences"]
    )
    alternatives = tuple(AlternativeMatch(**item) for item in parsed["alternative_matches"])
    assessments = tuple(
        _assess_safety(proposal, articles_by_id, generator, sleep=sleep, jitter=jitter)
        for proposal in proposals
    )
    component_accepted = tuple(
        AcceptedAudience(
            proposal.name,
            proposal.page_ids,
            proposal.rationale,
            component.component_id,
            assessment,
        )
        for proposal, assessment in zip(proposals, assessments, strict=True)
        if assessment.safe
    )
    if not proposals:
        outcome = "rejected"
    elif not component_accepted:
        outcome = "safety_vetoed"
    elif len(component_accepted) != len(proposals):
        outcome = "partially_accepted"
    else:
        outcome = "accepted"
    safety_rejected = tuple(
        page_id
        for proposal, assessment in zip(proposals, assessments, strict=True)
        if not assessment.safe
        for page_id in proposal.page_ids
    )
    return (
        ClusterRefinementDecision(
            component.component_id,
            component.page_ids,
            prompt,
            parsed["action"],
            outcome,
            str(parsed["rationale"]),
            proposals,
            tuple(parsed["rejected_page_ids"]) + safety_rejected,
            alternatives,
            attempts,
            assessments,
        ),
        component_accepted,
    )


def _validate_membership(payload: dict[str, object], candidate_ids: tuple[int, ...]) -> None:
    action = payload["action"]
    audiences = payload["audiences"]
    rejected = payload["rejected_page_ids"]
    assert isinstance(audiences, list) and isinstance(rejected, list)
    groups = [item["page_ids"] for item in audiences]
    assigned = [page_id for group in groups for page_id in group]
    all_members = assigned + rejected
    if len(all_members) != len(set(all_members)) or set(all_members) != set(candidate_ids):
        raise ValueError("every candidate member must be assigned or rejected exactly once")
    if action == "validate" and (len(groups) != 1 or rejected):
        raise ValueError("validate must retain the complete component as one audience")
    if action == "split" and (not groups or (len(groups) == 1 and not rejected)):
        raise ValueError("split must create multiple audiences or reject a member")
    if action == "reject" and (groups or set(rejected) != set(candidate_ids)):
        raise ValueError("reject must reject the complete component")
    audience_names = {item["name"] for item in audiences}
    for alternative in payload["alternative_matches"]:
        if alternative["page_id"] in assigned:
            raise ValueError("alternative matches cannot contribute accepted traffic")
        if alternative["page_id"] not in candidate_ids:
            raise ValueError("alternative match references a foreign page")
        if alternative["audience_name"] not in audience_names:
            raise ValueError("alternative match references an unknown audience")


def _assess_safety(
    proposal: ProposedAudience,
    articles_by_id: dict[int, CanonicalArticle],
    generator: StructuredGenerator,
    *,
    sleep: Callable[[float], None],
    jitter: Callable[[], float],
) -> ClusterSafetyAssessment:
    prompt = _safety_prompt(proposal, articles_by_id)
    parsed, attempts = _generate_with_retries(
        prompt, CLUSTER_SAFETY_SCHEMA, generator, sleep=sleep, jitter=jitter
    )
    if parsed is not None:
        tragedy = bool(parsed["materially_centered_on_tragedy"])
        violent_crime = bool(parsed["materially_centered_on_violent_crime"])
        safe = not tragedy and not violent_crime
        return ClusterSafetyAssessment(
            proposal.name, proposal.page_ids, prompt, safe,
            "accepted" if safe else "safety_vetoed", tragedy, violent_crime,
            str(parsed["rationale"]), attempts,
        )
    return ClusterSafetyAssessment(
        proposal.name, proposal.page_ids, prompt, False, "exhausted_attempts",
        None, None, None, attempts,
    )


def _generate_with_retries(
    prompt: str,
    schema: dict[str, object],
    generator: StructuredGenerator,
    *,
    validate: Callable[[dict[str, object]], None] = lambda _: None,
    sleep: Callable[[float], None],
    jitter: Callable[[], float],
) -> tuple[dict[str, object] | None, tuple[RefinementAttempt, ...]]:
    attempts: list[RefinementAttempt] = []
    for attempt_number in range(1, 4):
        raw_output: object | None = None
        try:
            raw_output = generator.generate(prompt, schema)
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            jsonschema.validate(parsed, schema)
            if not isinstance(parsed, dict):
                raise ValueError("structured response must be an object")
            validate(parsed)
            attempts.append(RefinementAttempt(attempt_number, raw_output, True, None))
            return parsed, tuple(attempts)
        except Exception as error:
            attempts.append(
                RefinementAttempt(
                    attempt_number, raw_output, False, f"{type(error).__name__}: {error}"
                )
            )
            if attempt_number < 3:
                sleep((2 ** (attempt_number - 1)) + jitter())
    return None, tuple(attempts)


def _refinement_prompt(
    component: CandidateComponent, articles_by_id: dict[int, CanonicalArticle]
) -> str:
    evidence = "\n\n".join(
        _article_evidence(articles_by_id[page_id]) for page_id in component.page_ids
    )
    return (
        "Refine this similarity-based candidate component into defensible consumer "
        "audiences. Validate it only if every article describes one coherent audience; "
        "split semantic chains; otherwise reject members. Each audience needs at least "
        "two meaningfully distinct canonical articles. Assign every page exactly once. "
        "Alternative matches are audit notes only and cannot contribute traffic.\n\n"
        f"{evidence}"
    )


def _safety_prompt(
    proposal: ProposedAudience, articles_by_id: dict[int, CanonicalArticle]
) -> str:
    evidence = "\n\n".join(
        _article_evidence(articles_by_id[page_id]) for page_id in proposal.page_ids
    )
    return (
        "Apply a separate, non-negotiable cluster-level safety veto. Determine whether "
        "the audience is materially centered on tragedy or violent crime, including when "
        "commercially adjacent articles make it appear otherwise.\n\n"
        f"Proposed audience: {proposal.name}\n{evidence}"
    )


def _article_evidence(article: CanonicalArticle) -> str:
    return (
        f"Page ID: {article.page_id}\nTitle: {article.canonical_title}\n"
        f"Lead extract: {article.extract}\nCategories: {', '.join(article.categories)}\n"
        f"Current-window views: {article.current_window_views}"
    )
