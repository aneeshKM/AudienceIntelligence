from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from audience_trend_miner.v2.shared import ProgressEvent, ProgressSink, V2ContractError


StageAction = Callable[[ProgressSink], Path]


@dataclass(frozen=True)
class GlobalRunStages:
    """The five public module interfaces required for one global run."""

    wikimedia_evidence: StageAction
    semantic_audience_formation: StageAction
    cluster_adjudication: StageAction
    trend_portfolio: StageAction
    run_publication: StageAction


def execute_global_run(
    *,
    run_id: str,
    progress_sink: ProgressSink,
    stages: GlobalRunStages,
) -> Path:
    """Execute the five V2 modules in dependency order with one event sequence."""
    sequence = 0

    def emit(event: ProgressEvent) -> None:
        nonlocal sequence
        if event.run_id != run_id:
            raise V2ContractError("module event belongs to a different run")
        sequence += 1
        progress_sink(replace(event, sequence=sequence))

    stages.wikimedia_evidence(emit)
    stages.semantic_audience_formation(emit)
    stages.cluster_adjudication(emit)
    stages.trend_portfolio(emit)
    publication = stages.run_publication(emit)
    if not publication.is_dir():
        raise V2ContractError("Run Publication did not complete")
    return publication
