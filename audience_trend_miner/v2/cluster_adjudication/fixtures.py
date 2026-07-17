from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Sequence

from audience_trend_miner.v2.shared import V2ContractError


class FrozenProposalAdapter:
    """Return one deterministic proposal while recording its model-visible input."""

    def __init__(self, proposal: object) -> None:
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
        model_input: Sequence[dict[str, object]],
        config: object = None,
        **kwargs: object,
    ) -> object:
        del config, kwargs
        recorded = deepcopy(list(model_input))
        self.model_inputs.append(recorded)
        return deepcopy(self._proposal)
