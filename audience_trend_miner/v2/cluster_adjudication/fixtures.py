from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

from audience_trend_miner.v2.cluster_adjudication.graph import (
    AdjudicationRequest,
    AdjudicationRole,
)
from audience_trend_miner.v2.shared import V2ContractError


@dataclass(frozen=True)
class FrozenAdjudicationCall:
    role: AdjudicationRole
    prompt: str
    model: str


class FrozenAdjudicationAdapter:
    """Return deterministic role outputs while recording bounded model calls."""

    def __init__(
        self,
        proposal: object,
        critique: object,
        revision: object = None,
        *,
        model: str = "fixture/cluster-model",
    ) -> None:
        self.model = model
        self._outputs = {
            "proposer": deepcopy(proposal),
            "critic": deepcopy(critique),
            "reviser": deepcopy(revision),
        }
        self.calls: list[FrozenAdjudicationCall] = []

    def invoke(self, request: AdjudicationRequest) -> object:
        self.calls.append(
            FrozenAdjudicationCall(
                role=request.role,
                prompt=request.prompt,
                model=self.model,
            )
        )
        return deepcopy(self._outputs[request.role])


class ScriptedAdjudicationAdapter:
    """Play a sequence of delivered outputs and delivery errors by graph role."""

    def __init__(
        self,
        responses: dict[AdjudicationRole, list[object]],
        *,
        model: str,
    ) -> None:
        self.model = model
        self._responses = deepcopy(responses)

    def invoke(self, request: AdjudicationRequest) -> object:
        responses = self._responses[request.role]
        if not responses:
            raise RuntimeError(f"fixture {request.role} responses exhausted")
        response = responses.pop(0)
        if isinstance(response, dict) and set(response) == {"delivery_error"}:
            raise RuntimeError(str(response["delivery_error"]))
        return deepcopy(response)


@dataclass(frozen=True)
class FrozenStageAdapterFactory:
    model: str
    _clusters: tuple[dict[str, object], ...]
    integration_name: str = "fixture"

    @classmethod
    def from_file(cls, path: Path) -> FrozenStageAdapterFactory:
        try:
            fixture = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise V2ContractError("adjudication stage fixture is unreadable") from error
        if (
            not isinstance(fixture, dict)
            or set(fixture) != {"schema_version", "model", "clusters"}
            or fixture["schema_version"] != "1.0"
            or not isinstance(fixture["model"], str)
            or not fixture["model"]
            or not isinstance(fixture["clusters"], list)
        ):
            raise V2ContractError("adjudication stage fixture has an invalid shape")
        return cls(fixture["model"], tuple(deepcopy(fixture["clusters"])))

    def adapter_for(
        self, cluster_index: int, preliminary_cluster: dict[str, object]
    ) -> ScriptedAdjudicationAdapter:
        try:
            fixture = self._clusters[cluster_index]
        except IndexError as error:
            raise V2ContractError("adjudication stage fixture is missing a cluster") from error
        if not isinstance(fixture, dict) or set(fixture) != {"page_ids", "responses"}:
            raise V2ContractError("adjudication stage fixture has an invalid cluster")
        members = preliminary_cluster.get("members")
        if not isinstance(members, list):
            raise V2ContractError("Preliminary Cluster members are invalid")
        page_ids = [member.get("page_id") for member in members if isinstance(member, dict)]
        if fixture["page_ids"] != page_ids:
            raise V2ContractError("adjudication fixture conflicts with Preliminary Cluster")
        responses = fixture["responses"]
        if (
            not isinstance(responses, dict)
            or set(responses) != {"proposer", "critic", "reviser"}
            or any(not isinstance(value, list) for value in responses.values())
        ):
            raise V2ContractError("adjudication stage fixture has invalid responses")
        return ScriptedAdjudicationAdapter(
            cast(dict[AdjudicationRole, list[object]], responses), model=self.model
        )


class FrozenProposalAdapter:
    """Return one deterministic proposal while recording its model-visible input."""

    def __init__(self, proposal: object) -> None:
        self.model = "fixture/proposal-model"
        self._proposal = deepcopy(proposal)
        self.model_inputs: list[list[dict[str, object]]] = []

    @classmethod
    def from_file(cls, path: Path) -> FrozenProposalAdapter:
        try:
            fixture = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise V2ContractError("adjudication fixture is unreadable") from error
        if (
            not isinstance(fixture, dict)
            or set(fixture) != {"schema_version", "proposal"}
            or fixture["schema_version"] != "1.0"
        ):
            raise V2ContractError("adjudication fixture has an invalid shape")
        return cls(fixture["proposal"])

    def invoke(
        self,
        request: AdjudicationRequest,
    ) -> object:
        if request.role == "critic":
            return {"approved": True, "challenges": []}
        if request.role == "reviser":
            return deepcopy(self._proposal)
        recorded = deepcopy(list(request.members))
        self.model_inputs.append(recorded)
        return deepcopy(self._proposal)
