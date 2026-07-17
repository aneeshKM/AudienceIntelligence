from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Literal, Protocol, Sequence, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from audience_trend_miner.v2.shared import V2ContractError


ALLOWED_PAGE_FIELDS = (
    "page_id",
    "canonical_title",
    "lead",
    "selected_categories",
)
class CritiqueDimension(StrEnum):
    SEMANTIC_COHERENCE = "semantic_coherence"
    COMMERCIAL_MEANING = "commercial_meaning"
    BRAND_SAFETY = "brand_safety"
    NAMING = "naming"
    EVIDENCE_SUPPORT = "evidence_support"


class RequiredAction(StrEnum):
    REVISE = "revise"
    REJECT = "reject"


FAIL_CLOSED_DIMENSIONS = frozenset(
    {CritiqueDimension.BRAND_SAFETY, CritiqueDimension.EVIDENCE_SUPPORT}
)

PROPOSER_PROMPT = """You are the Cluster Adjudication proposer. Use only the supplied Canonical Page evidence to form coherent, commercially meaningful, brand-safe Final Audience Clusters or reject Canonical Pages. Return decisions and concise evidence; do not provide hidden reasoning or chain-of-thought."""
CRITIC_PROMPT = """You are the independent Cluster Adjudication critic. Check semantic coherence, commercial meaning, brand safety, naming, and evidence support. Return approval or concise structured challenges only; do not provide hidden reasoning or chain-of-thought."""
REVISER_PROMPT = """You are the Cluster Adjudication reviser. Address the supplied validation errors and critic challenges once, without restoring rejected Canonical Pages. Return the final decisions and concise evidence; do not provide hidden reasoning or chain-of-thought."""


AdjudicationRole = Literal["proposer", "critic", "reviser"]


@dataclass(frozen=True)
class AdjudicationRequest:
    role: AdjudicationRole
    prompt: str
    members: tuple[dict[str, object], ...]
    proposal: object = None
    validation_errors: tuple[str, ...] = ()
    critique: object = None


@dataclass(frozen=True)
class CritiqueChallenge:
    dimension: CritiqueDimension
    message: str
    page_ids: tuple[int, ...]
    required_action: RequiredAction


@dataclass(frozen=True)
class CritiqueDecision:
    approved: bool
    challenges: tuple[CritiqueChallenge, ...]


class AdjudicationAdapter(Protocol):
    model: str

    def invoke(self, request: AdjudicationRequest) -> object: ...


@dataclass(frozen=True)
class ProviderAttempt:
    attempt: int
    delivery_status: Literal["delivered", "error"]
    error: str | None

    def record(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "delivery_status": self.delivery_status,
            "error": self.error,
        }


@dataclass(frozen=True)
class ModelStepRecord:
    role: AdjudicationRole
    status: Literal["completed", "exhausted"]
    validation_status: Literal["valid", "invalid", "not_run"]
    attempts: tuple[ProviderAttempt, ...]

    def record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "status": self.status,
            "validation_status": self.validation_status,
            "attempts": [attempt.record() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class AcceptedGroup:
    name: str
    rationale: str
    members: tuple[dict[str, object], ...]

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rationale": self.rationale,
            "members": deepcopy(list(self.members)),
        }


@dataclass(frozen=True)
class ClusterAdjudicationResult:
    accepted_groups: tuple[AcceptedGroup, ...]
    rejected_members: tuple[dict[str, object], ...]
    validation_status: str
    validation_errors: tuple[str, ...]
    steps: tuple[ModelStepRecord, ...] = ()

    def record(self) -> dict[str, object]:
        return {
            "accepted_groups": [group.record() for group in self.accepted_groups],
            "rejected_members": deepcopy(list(self.rejected_members)),
            "validation": {
                "status": self.validation_status,
                "errors": list(self.validation_errors),
            },
        }


class _GraphState(TypedDict, total=False):
    members: tuple[dict[str, object], ...]
    proposal: object
    proposal_result: ClusterAdjudicationResult
    critique: object
    critique_decision: CritiqueDecision | None
    revision: object
    result: ClusterAdjudicationResult


def execute_cluster_adjudication(
    preliminary_cluster: object,
    adapter: AdjudicationAdapter,
    *,
    step_progress: Callable[[ModelStepRecord], None] | None = None,
) -> ClusterAdjudicationResult:
    """Adjudicate one Preliminary Cluster with bounded critique and revision."""
    members = _model_visible_members(preliminary_cluster)
    steps: list[ModelStepRecord] = []

    def invoke(request: AdjudicationRequest) -> object:
        attempts: list[ProviderAttempt] = []
        for attempt_number in range(1, 4):
            try:
                output = adapter.invoke(request)
            except V2ContractError:
                raise
            except Exception as error:
                attempts.append(
                    ProviderAttempt(
                        attempt=attempt_number,
                        delivery_status="error",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            attempts.append(
                ProviderAttempt(
                    attempt=attempt_number,
                    delivery_status="delivered",
                    error=None,
                )
            )
            steps.append(
                ModelStepRecord(
                    role=request.role,
                    status="completed",
                    validation_status="not_run",
                    attempts=tuple(attempts),
                )
            )
            return output
        exhausted = ModelStepRecord(
            role=request.role,
            status="exhausted",
            validation_status="not_run",
            attempts=tuple(attempts),
        )
        steps.append(exhausted)
        if step_progress is not None:
            step_progress(exhausted)
        raise _DeliveryExhausted(request.role)

    def finish_validation(role: AdjudicationRole, status: Literal["valid", "invalid"]) -> None:
        previous = steps[-1]
        if previous.role != role:
            raise RuntimeError("adjudication step validation is out of order")
        completed = ModelStepRecord(
            role=previous.role,
            status=previous.status,
            validation_status=status,
            attempts=previous.attempts,
        )
        steps[-1] = completed
        if step_progress is not None:
            step_progress(completed)

    def propose(state: _GraphState) -> _GraphState:
        return {
            "proposal": invoke(
                AdjudicationRequest(
                    role="proposer",
                    prompt=PROPOSER_PROMPT,
                    members=state["members"],
                )
            )
        }

    def validate_proposal(state: _GraphState) -> _GraphState:
        result = _validated_result(state["members"], state["proposal"])
        finish_validation(
            "proposer", "valid" if result.validation_status == "valid" else "invalid"
        )
        return {"proposal_result": result}

    def critique(state: _GraphState) -> _GraphState:
        proposal_result = state["proposal_result"]
        model_output = invoke(
            AdjudicationRequest(
                role="critic",
                prompt=CRITIC_PROMPT,
                members=state["members"],
                proposal=deepcopy(state["proposal"]),
                validation_errors=proposal_result.validation_errors,
            )
        )
        parsed = _parse_critique(state["members"], model_output)
        finish_validation("critic", "valid" if parsed is not None else "invalid")
        return {
            "critique": model_output,
            "critique_decision": parsed,
        }

    def after_critique(state: _GraphState) -> str:
        decision = state["critique_decision"]
        if decision is None:
            return "reject"
        if (
            state["proposal_result"].validation_status == "valid"
            and decision.approved
        ):
            return "accept"
        return "revise"

    def accept(state: _GraphState) -> _GraphState:
        return {"result": state["proposal_result"]}

    def revise(state: _GraphState) -> _GraphState:
        proposal_result = state["proposal_result"]
        return {
            "revision": invoke(
                AdjudicationRequest(
                    role="reviser",
                    prompt=REVISER_PROMPT,
                    members=state["members"],
                    proposal=deepcopy(state["proposal"]),
                    validation_errors=proposal_result.validation_errors,
                    critique=deepcopy(state["critique"]),
                )
            )
        }

    def validate_revision(state: _GraphState) -> _GraphState:
        result = _validated_revision_result(
            state["members"],
            state["proposal"],
            cast(CritiqueDecision, state["critique_decision"]),
            state["revision"],
        )
        finish_validation(
            "reviser", "valid" if result.validation_status == "valid" else "invalid"
        )
        return {"result": result}

    def reject_invalid_critique(state: _GraphState) -> _GraphState:
        return {"result": _invalid_result(state["members"], ("invalid_critique",))}

    builder = StateGraph(_GraphState)
    builder.add_node("propose", propose)
    builder.add_node("validate_proposal", validate_proposal)
    builder.add_node("critique", critique)
    builder.add_node("accept", accept)
    builder.add_node("revise", revise)
    builder.add_node("validate_revision", validate_revision)
    builder.add_node("reject_invalid_critique", reject_invalid_critique)
    builder.add_edge(START, "propose")
    builder.add_edge("propose", "validate_proposal")
    builder.add_edge("validate_proposal", "critique")
    builder.add_conditional_edges(
        "critique",
        after_critique,
        {
            "accept": "accept",
            "revise": "revise",
            "reject": "reject_invalid_critique",
        },
    )
    builder.add_edge("accept", END)
    builder.add_edge("revise", "validate_revision")
    builder.add_edge("validate_revision", END)
    builder.add_edge("reject_invalid_critique", END)
    try:
        completed = builder.compile().invoke({"members": members})
        result = cast(ClusterAdjudicationResult, completed["result"])
    except _DeliveryExhausted as error:
        result = _invalid_result(members, (f"exhausted_delivery:{error.role}",))
    return ClusterAdjudicationResult(
        accepted_groups=result.accepted_groups,
        rejected_members=result.rejected_members,
        validation_status=result.validation_status,
        validation_errors=result.validation_errors,
        steps=tuple(steps),
    )


class _DeliveryExhausted(RuntimeError):
    def __init__(self, role: AdjudicationRole) -> None:
        super().__init__(f"{role} delivery exhausted")
        self.role = role


def _parse_critique(
    members: tuple[dict[str, object], ...], critique: object
) -> CritiqueDecision | None:
    if (
        not isinstance(critique, dict)
        or set(critique) != {"approved", "challenges"}
        or not isinstance(critique["approved"], bool)
        or not isinstance(critique["challenges"], list)
        or critique["approved"] != (not critique["challenges"])
    ):
        return None
    supplied_ids = {member["page_id"] for member in members}
    parsed_challenges: list[CritiqueChallenge] = []
    for challenge in critique["challenges"]:
        if not isinstance(challenge, dict) or set(challenge) != {
            "dimension",
            "message",
            "page_ids",
            "required_action",
        }:
            return None
        page_ids = challenge["page_ids"]
        dimension = challenge["dimension"]
        required_action = challenge["required_action"]
        try:
            parsed_dimension = CritiqueDimension(dimension)
            parsed_action = RequiredAction(required_action)
        except (TypeError, ValueError):
            return None
        if (
            not isinstance(challenge["message"], str)
            or not challenge["message"].strip()
            or not isinstance(page_ids, list)
            or not page_ids
            or any(
                not isinstance(page_id, int)
                or isinstance(page_id, bool)
                or page_id not in supplied_ids
                for page_id in page_ids
            )
            or len(page_ids) != len(set(page_ids))
        ):
            return None
        parsed_challenges.append(
            CritiqueChallenge(
                dimension=parsed_dimension,
                message=challenge["message"],
                page_ids=tuple(page_ids),
                required_action=parsed_action,
            )
        )
    return CritiqueDecision(
        approved=critique["approved"], challenges=tuple(parsed_challenges)
    )


def _validated_revision_result(
    members: tuple[dict[str, object], ...],
    proposal: object,
    critique: CritiqueDecision,
    revision: object,
) -> ClusterAdjudicationResult:
    result = _validated_result(members, revision)
    if result.validation_status != "valid":
        return result
    final_rejections = {
        cast(int, rejected["page_id"]) for rejected in result.rejected_members
    }
    errors: list[str] = []
    for page_id in _explicit_rejections(members, proposal):
        if page_id not in final_rejections:
            errors.append(f"resurrected_page_id:{page_id}")
    for page_id in _critic_required_rejections(critique):
        if page_id not in final_rejections:
            errors.append(f"critic_required_rejection_missing:{page_id}")
    if errors:
        return _invalid_result(members, tuple(dict.fromkeys(errors)))
    return result


def _explicit_rejections(
    members: tuple[dict[str, object], ...], proposal: object
) -> tuple[int, ...]:
    if not isinstance(proposal, dict) or not isinstance(proposal.get("rejected"), list):
        return ()
    supplied_ids = {member["page_id"] for member in members}
    rejected_ids: list[int] = []
    for rejection in proposal["rejected"]:
        if not isinstance(rejection, dict):
            continue
        page_id = rejection.get("page_id")
        reason = rejection.get("reason")
        if (
            isinstance(page_id, int)
            and not isinstance(page_id, bool)
            and page_id in supplied_ids
            and isinstance(reason, str)
            and reason.strip()
            and page_id not in rejected_ids
        ):
            rejected_ids.append(page_id)
    return tuple(rejected_ids)


def _critic_required_rejections(critique: CritiqueDecision) -> tuple[int, ...]:
    required: list[int] = []
    for challenge in critique.challenges:
        if (
            challenge.required_action is RequiredAction.REJECT
            or challenge.dimension in FAIL_CLOSED_DIMENSIONS
        ):
            for page_id in challenge.page_ids:
                if page_id not in required:
                    required.append(page_id)
    return tuple(required)


def _model_visible_members(
    preliminary_cluster: object,
) -> tuple[dict[str, object], ...]:
    if not isinstance(preliminary_cluster, dict):
        raise V2ContractError("Preliminary Cluster has an invalid shape")
    supplied_members = preliminary_cluster.get("members")
    if not isinstance(supplied_members, list) or not supplied_members:
        raise V2ContractError("Preliminary Cluster members are invalid")
    members: list[dict[str, object]] = []
    page_ids: set[int] = set()
    for supplied in supplied_members:
        if not isinstance(supplied, dict) or not all(
            field in supplied for field in ALLOWED_PAGE_FIELDS
        ):
            raise V2ContractError("Preliminary Cluster member is invalid")
        page_id = supplied["page_id"]
        canonical_title = supplied["canonical_title"]
        lead = supplied["lead"]
        selected_categories = supplied["selected_categories"]
        if (
            not isinstance(page_id, int)
            or isinstance(page_id, bool)
            or not isinstance(canonical_title, str)
            or not canonical_title
            or not isinstance(lead, str)
            or len(lead) > 600
            or not isinstance(selected_categories, list)
            or len(selected_categories) > 5
            or any(
                not isinstance(category, str) or not category
                for category in selected_categories
            )
            or len(selected_categories) != len(set(selected_categories))
        ):
            raise V2ContractError("Preliminary Cluster member is invalid")
        if page_id in page_ids:
            raise V2ContractError("Preliminary Cluster contains duplicate page IDs")
        page_ids.add(page_id)
        members.append(
            {field: deepcopy(supplied[field]) for field in ALLOWED_PAGE_FIELDS}
        )
    return tuple(members)


def _validated_result(
    members: tuple[dict[str, object], ...], proposal: object
) -> ClusterAdjudicationResult:
    if not isinstance(proposal, dict) or set(proposal) != {"groups", "rejected"}:
        return _invalid_result(members, ("invalid_proposal_shape",))
    groups = proposal.get("groups")
    rejected = proposal.get("rejected")
    if not isinstance(groups, list) or not isinstance(rejected, list):
        return _invalid_result(members, ("invalid_proposal_shape",))
    member_by_id: dict[int, dict[str, object]] = {
        cast(int, member["page_id"]): member for member in members
    }
    supplied_ids = set(member_by_id)
    errors: list[str] = []
    occurrences: dict[int, int] = {}
    parsed_groups: list[tuple[str, str, list[int]]] = []
    for group in groups:
        if not isinstance(group, dict) or set(group) != {
            "name",
            "page_ids",
            "rationale",
        }:
            return _invalid_result(members, ("invalid_proposal_shape",))
        page_ids = group.get("page_ids")
        name = group.get("name")
        rationale = group.get("rationale")
        if (
            not isinstance(page_ids, list)
            or not isinstance(name, str)
            or not name.strip()
            or not isinstance(rationale, str)
            or not rationale.strip()
            or any(
                not isinstance(page_id, int) or isinstance(page_id, bool)
                for page_id in page_ids
            )
        ):
            return _invalid_result(members, ("invalid_proposal_shape",))
        typed_page_ids = cast(list[int], page_ids)
        duplicate_ids = _duplicates(typed_page_ids)
        errors.extend(f"duplicate_page_id:{page_id}" for page_id in duplicate_ids)
        if len(set(typed_page_ids)) < 2:
            errors.append(
                "accepted_group_has_fewer_than_two_distinct_pages:" + name
            )
        for page_id in typed_page_ids:
            occurrences[page_id] = occurrences.get(page_id, 0) + 1
            if page_id not in supplied_ids:
                errors.append(f"unknown_page_id:{page_id}")
        parsed_groups.append((name, rationale, typed_page_ids))

    parsed_rejections: list[tuple[int, str]] = []
    for rejection in rejected:
        if not isinstance(rejection, dict) or set(rejection) != {"page_id", "reason"}:
            return _invalid_result(members, ("invalid_proposal_shape",))
        rejected_page_id = rejection.get("page_id")
        reason = rejection.get("reason")
        if (
            not isinstance(rejected_page_id, int)
            or isinstance(rejected_page_id, bool)
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            return _invalid_result(members, ("invalid_proposal_shape",))
        occurrences[rejected_page_id] = occurrences.get(rejected_page_id, 0) + 1
        if rejected_page_id not in supplied_ids:
            errors.append(f"unknown_page_id:{rejected_page_id}")
        parsed_rejections.append((rejected_page_id, reason))

    for page_id in member_by_id:
        count = occurrences.get(page_id, 0)
        if count == 0:
            errors.append(f"omitted_page_id:{page_id}")
        elif count > 1:
            errors.append(f"multiply_assigned_page_id:{page_id}")
    if errors:
        return _invalid_result(members, tuple(dict.fromkeys(errors)))

    accepted_groups = tuple(
        AcceptedGroup(
            name=name,
            rationale=rationale,
            members=tuple(member_by_id[page_id] for page_id in page_ids),
        )
        for name, rationale, page_ids in parsed_groups
    )
    rejected_members = tuple(
        {
            "page_id": page_id,
            "canonical_title": member_by_id[page_id]["canonical_title"],
            "reason": reason,
        }
        for page_id, reason in parsed_rejections
    )
    return ClusterAdjudicationResult(
        accepted_groups=accepted_groups,
        rejected_members=rejected_members,
        validation_status="valid",
        validation_errors=(),
    )


def _duplicates(values: Sequence[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    duplicates: list[int] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _invalid_result(
    members: tuple[dict[str, object], ...], errors: tuple[str, ...]
) -> ClusterAdjudicationResult:
    return ClusterAdjudicationResult(
        accepted_groups=(),
        rejected_members=tuple(
            {
                "page_id": member["page_id"],
                "canonical_title": member["canonical_title"],
                "reason": "invalid_adjudication",
            }
            for member in members
        ),
        validation_status="invalid",
        validation_errors=errors,
    )
