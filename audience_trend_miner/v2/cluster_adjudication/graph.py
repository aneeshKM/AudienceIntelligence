from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, Sequence, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from audience_trend_miner.v2.shared import V2ContractError


ALLOWED_PAGE_FIELDS = (
    "page_id",
    "canonical_title",
    "lead",
    "selected_categories",
)


class ProposalAdapter(Protocol):
    def invoke(
        self,
        model_input: Sequence[dict[str, object]],
        config: object = None,
        **kwargs: object,
    ) -> object: ...


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
    result: ClusterAdjudicationResult


def execute_cluster_adjudication(
    preliminary_cluster: object,
    proposal_adapter: ProposalAdapter,
) -> ClusterAdjudicationResult:
    """Propose and deterministically validate one isolated Preliminary Cluster."""
    members = _model_visible_members(preliminary_cluster)

    def propose(state: _GraphState) -> _GraphState:
        return {"proposal": proposal_adapter.invoke(state["members"])}

    def validate(state: _GraphState) -> _GraphState:
        return {"result": _validated_result(state["members"], state["proposal"])}

    builder = StateGraph(_GraphState)
    builder.add_node("propose", propose)
    builder.add_node("validate", validate)
    builder.add_edge(START, "propose")
    builder.add_edge("propose", "validate")
    builder.add_edge("validate", END)
    completed = builder.compile().invoke({"members": members})
    return cast(ClusterAdjudicationResult, completed["result"])


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
