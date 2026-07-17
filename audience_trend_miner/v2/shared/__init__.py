"""Public interface for shared V2 run and artifact contracts."""

from audience_trend_miner.v2.shared.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    PROGRESS_SCHEMA_VERSION,
    BoundedProgress,
    ProgressEvent,
    ProgressSink,
    V2ContractError,
    atomic_write_json,
    consume_artifact,
    execute_fixture_stage,
    human_progress_sink,
    json_progress_sink,
    record_run_configuration,
    validate_artifact,
    validate_identifier,
    validate_schema,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PROGRESS_SCHEMA_VERSION",
    "BoundedProgress",
    "ProgressEvent",
    "ProgressSink",
    "V2ContractError",
    "atomic_write_json",
    "consume_artifact",
    "execute_fixture_stage",
    "human_progress_sink",
    "json_progress_sink",
    "record_run_configuration",
    "validate_artifact",
    "validate_identifier",
    "validate_schema",
]
