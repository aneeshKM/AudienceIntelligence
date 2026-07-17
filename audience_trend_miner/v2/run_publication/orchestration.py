from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Mapping

from audience_trend_miner.v2.shared import (
    ProgressEvent,
    ProgressSink,
    V2ContractError,
    atomic_write_json,
)


StageAction = Callable[[ProgressSink], Path]
GLOBAL_RUN_CONFIGURATION_NAME = "global-run.json"


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
    run_directory: Path,
    configuration: Mapping[str, str],
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

    try:
        _record_global_configuration(
            run_directory,
            run_id=run_id,
            configuration=configuration,
        )
    except V2ContractError as error:
        emit(_failure_event(run_id, "global-run", error))
        raise

    publication: Path | None = None
    ordered_stages = (
        ("wikimedia-evidence", stages.wikimedia_evidence),
        ("semantic-audience-formation", stages.semantic_audience_formation),
        ("cluster-adjudication", stages.cluster_adjudication),
        ("trend-portfolio", stages.trend_portfolio),
        ("run-publication", stages.run_publication),
    )
    for module, action in ordered_stages:
        try:
            publication = action(emit)
        except V2ContractError as error:
            emit(_failure_event(run_id, module, error))
            raise
    assert publication is not None
    if not publication.is_dir():
        raise V2ContractError("Run Publication did not complete")
    return publication


def _record_global_configuration(
    run_directory: Path,
    *,
    run_id: str,
    configuration: Mapping[str, str],
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "configuration": dict(configuration),
    }
    path = run_directory / GLOBAL_RUN_CONFIGURATION_NAME
    if not path.exists():
        try:
            atomic_write_json(path, record, refuse_replace=True)
        except FileExistsError:
            pass
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ContractError("recorded global run configuration is unreadable") from error
    if existing != record:
        raise V2ContractError("global run configuration conflicts with recorded facts")


def _failure_event(
    run_id: str,
    module: str,
    error: V2ContractError,
) -> ProgressEvent:
    return ProgressEvent(
        run_id=run_id,
        sequence=1,
        timestamp=datetime.now(timezone.utc).isoformat(),
        module=module,
        operation="failed",
        level="error",
        message=str(error),
    )
