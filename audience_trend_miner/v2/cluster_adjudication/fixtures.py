from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

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
