from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping

import jsonschema


SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
RUN_CONFIGURATION_NAME = "run.json"
ARTIFACT_SCHEMA_VERSION = "2.0"
PROGRESS_SCHEMA_VERSION = "1.0"


class V2ContractError(ValueError):
    """A V2 run, progress, or artifact contract was violated."""


@dataclass(frozen=True)
class BoundedProgress:
    current: int
    total: int

    def __post_init__(self) -> None:
        if self.total <= 0 or not 0 <= self.current <= self.total:
            raise V2ContractError("progress must be bounded by a positive total")


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    sequence: int
    timestamp: str
    module: str
    operation: str
    level: str
    message: str
    progress: BoundedProgress | None = None
    schema_version: str = PROGRESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _safe_identifier(self.run_id, "run_id")
        _safe_identifier(self.module, "module")
        if self.sequence <= 0:
            raise V2ContractError("event sequence must be positive")
        if self.level not in {"info", "warning", "error"}:
            raise V2ContractError("event level is unsupported")

    def record(self) -> dict[str, object]:
        record = asdict(self)
        if self.progress is None:
            record.pop("progress")
        _validate("v2-progress-event.schema.json", record)
        return record


ProgressSink = Callable[[ProgressEvent], None]


def human_progress_sink(stream: object) -> ProgressSink:
    def render(event: ProgressEvent) -> None:
        suffix = (
            f" ({event.progress.current}/{event.progress.total})"
            if event.progress
            else ""
        )
        print(
            f"[{event.module}:{event.operation}] {event.message}{suffix}",
            file=stream,
            flush=True,
        )

    return _monotonic_sink(render)


def json_progress_sink(stream: object) -> ProgressSink:
    def render(event: ProgressEvent) -> None:
        print(
            json.dumps(event.record(), separators=(",", ":"), sort_keys=True),
            file=stream,
            flush=True,
        )

    return _monotonic_sink(render)


def _monotonic_sink(render: ProgressSink) -> ProgressSink:
    last_sequence_by_run: dict[str, int] = {}

    def emit(event: ProgressEvent) -> None:
        previous = last_sequence_by_run.get(event.run_id, 0)
        if event.sequence <= previous:
            raise V2ContractError("event sequence must increase monotonically")
        render(event)
        last_sequence_by_run[event.run_id] = event.sequence

    return emit


def execute_fixture_stage(
    *,
    run_id: str,
    configuration: Mapping[str, str],
    output_root: Path,
    fixture_path: Path,
    progress_sink: ProgressSink,
    consume_existing: bool = False,
    interrupt_before_completion: bool = False,
) -> Path:
    """Exercise the shared V2 stage boundary with deterministic domain data."""
    _safe_identifier(run_id, "run_id")
    fixture = _load_fixture(fixture_path)
    stage = fixture["stage"]
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    record_run_configuration(run_directory, run_id, configuration)
    artifact_path = run_directory / f"{stage}.json"
    if consume_existing:
        consume_artifact(artifact_path, run_id=run_id, stage=stage)
        return artifact_path

    _emit(progress_sink, run_id, 1, stage, "load", "loading fixture", 1, 2)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": stage,
        "status": "complete",
        "payload": fixture["payload"],
    }
    validate_artifact(artifact, run_id=run_id, stage=stage)
    _atomic_write_json(
        artifact_path,
        artifact,
        interrupt_before_replace=interrupt_before_completion,
    )
    _emit(
        progress_sink, run_id, 2, stage, "publish", "published artifact", 2, 2
    )
    return artifact_path


def record_run_configuration(
    run_directory: Path, run_id: str, configuration: Mapping[str, str]
) -> None:
    record = {"schema_version": "1.0", "run_id": run_id, "configuration": dict(configuration)}
    path = run_directory / RUN_CONFIGURATION_NAME
    if not path.exists():
        try:
            _atomic_write_json(path, record, refuse_replace=True)
        except FileExistsError:
            pass
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ContractError("recorded run configuration is unreadable") from error
    if existing != record:
        raise V2ContractError("run configuration conflicts with recorded facts")


def consume_artifact(path: Path, *, run_id: str, stage: str) -> dict[str, object]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise V2ContractError("artifact is absent") from error
    except json.JSONDecodeError as error:
        raise V2ContractError("artifact is invalid") from error
    if artifact.get("status") != "complete":
        raise V2ContractError("artifact is incomplete")
    validate_artifact(artifact, run_id=run_id, stage=stage)
    return artifact


def validate_artifact(
    artifact: object, *, run_id: str, stage: str
) -> None:
    try:
        _validate("v2-stage-artifact.schema.json", artifact)
    except jsonschema.ValidationError as error:
        raise V2ContractError(f"artifact is schema-invalid: {error.message}") from error
    assert isinstance(artifact, dict)
    if artifact["run_id"] != run_id or artifact["stage"] != stage:
        raise V2ContractError("artifact belongs to different run facts")


def _emit(
    sink: ProgressSink,
    run_id: str,
    sequence: int,
    module: str,
    operation: str,
    message: str,
    current: int,
    total: int,
) -> None:
    sink(
        ProgressEvent(
            run_id=run_id,
            sequence=sequence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=module,
            operation=operation,
            level="info",
            message=message,
            progress=BoundedProgress(current, total),
        )
    )


def _load_fixture(path: Path) -> dict[str, object]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ContractError("fixture is unreadable") from error
    if not isinstance(fixture, dict) or set(fixture) != {"schema_version", "stage", "payload"}:
        raise V2ContractError("fixture has an invalid shape")
    if fixture["schema_version"] != "1.0" or not isinstance(fixture["payload"], dict):
        raise V2ContractError("fixture has an incompatible schema")
    _safe_identifier(fixture["stage"], "stage")
    return fixture


def _safe_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise V2ContractError(f"{name} must be one safe path segment")
    return value


def _atomic_write_json(
    path: Path,
    content: object,
    *,
    interrupt_before_replace: bool = False,
    refuse_replace: bool = False,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if interrupt_before_replace:
            raise V2ContractError("fixture interrupted before artifact completion")
        if refuse_replace:
            os.link(temporary_path, path)
        else:
            os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate(schema_name: str, instance: object) -> None:
    schema = json.loads((SCHEMA_DIRECTORY / schema_name).read_text(encoding="utf-8"))
    jsonschema.validate(instance, schema)
