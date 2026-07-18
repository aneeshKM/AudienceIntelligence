from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from threading import Lock, Thread
from typing import Annotated, Literal, Sequence

from fastapi import (
    FastAPI,
    HTTPException,
    Path as ApiPath,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from audience_trend_miner.v2.run_publication import validate_completed_publication
from audience_trend_miner.v2.shared import (
    BoundedProgress,
    ProgressEvent,
    V2ContractError,
    atomic_write_json,
)


DEFAULT_HOST = "127.0.0.1"
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
EVENT_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
EVENT_HISTORY_NAME = "ui-events.jsonl"
RUN_STATE_NAME = "ui-state.json"
STATIC_DIRECTORY = Path(__file__).parent / "static"


# Validate the browser request used to start or resume a run.
class StartRunRequest(BaseModel):
    run_id: Annotated[str, Field(pattern=RUN_ID_PATTERN)]
    as_of: date


# Require explicit confirmation before cancellation.
class CancelRunRequest(BaseModel):
    confirmed: bool


# Track durable and live state for one pipeline run.
@dataclass
class _RunState:
    run_id: str
    as_of: date
    status: Literal["running", "succeeded", "failed", "cancelled"] = "running"
    exit_code: int | None = None
    failure: dict[str, str] | None = None

    # Build an HTTP response with the required security headers.
    def response(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "as_of": self.as_of.isoformat(),
            "status": self.status,
            "exit_code": self.exit_code,
            "failure": self.failure,
        }


# Pair a live event queue with the last replayed sequence.
@dataclass(frozen=True)
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[ProgressEvent]


# Maintain durable ordered history plus live subscribers for one run.
class _RunEventLog:
    """Durable ordered event history with independent live subscribers."""

    # Initialize the _RunEventLog.
    def __init__(self, path: Path, run_id: str) -> None:
        self._path = path
        self._run_id = run_id
        self._lock = Lock()
        self._events = self._load()
        self._subscribers: list[_Subscriber] = []

    # Publish an event to durable history and live subscribers.
    def publish(self, event: ProgressEvent) -> ProgressEvent:
        # The server assigns its own contiguous sequence independent of CLI sequences.
        with self._lock:
            normalized = ProgressEvent(
                run_id=event.run_id,
                sequence=len(self._events) + 1,
                timestamp=event.timestamp,
                module=event.module,
                operation=event.operation,
                level=event.level,
                message=event.message,
                progress=event.progress,
            )
            record = normalized.record()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Durability precedes live fan-out so reconnect can always fill a gap.
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(record, separators=(",", ":"), sort_keys=True)
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._events.append(normalized)
            # Dead event loops are pruned without interrupting other subscribers.
            connected_subscribers = []
            for subscriber in self._subscribers:
                try:
                    subscriber.loop.call_soon_threadsafe(
                        subscriber.queue.put_nowait, normalized
                    )
                except RuntimeError:
                    continue
                connected_subscribers.append(subscriber)
            self._subscribers = connected_subscribers
            return normalized

    # Subscribe a queue to run events and return replay history.
    def subscribe(
        self,
        after_sequence: int,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[list[ProgressEvent], _Subscriber]:
        subscriber = _Subscriber(loop=loop, queue=asyncio.Queue())
        with self._lock:
            snapshot = list(self._events)
            self._subscribers.append(subscriber)
        return (
            [event for event in snapshot if event.sequence > after_sequence],
            subscriber,
        )

    # Remove a live run-event subscriber.
    def unsubscribe(self, subscriber: _Subscriber) -> None:
        with self._lock:
            self._subscribers = [
                registered
                for registered in self._subscribers
                if registered is not subscriber
            ]

    # Load durable events from disk.
    def _load(self) -> list[ProgressEvent]:
        if not self._path.exists():
            return []
        events: list[ProgressEvent] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise V2ContractError("event history is unreadable") from error
        for expected_sequence, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
                event = _progress_event(record, expected_run_id=self._run_id)
            except (
                json.JSONDecodeError,
                TypeError,
                KeyError,
                V2ContractError,
            ) as error:
                raise V2ContractError("event history is invalid") from error
            if event.sequence != expected_sequence:
                raise V2ContractError("event history sequence is invalid")
            events.append(event)
        return events


# Own subprocess lifecycle, persistence, recovery, and event forwarding.
class _RunSupervisor:
    # Initialize the _RunSupervisor.
    def __init__(
        self,
        output_root: Path,
        cli_command: Sequence[str],
        cli_arguments: Sequence[str],
    ) -> None:
        self._output_root = output_root
        self._cli_command = tuple(cli_command)
        self._cli_arguments = tuple(cli_arguments)
        self._states: dict[str, _RunState] = {}
        self._event_logs: dict[str, _RunEventLog] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._unowned_process_ids: dict[str, int] = {}
        self._lock = Lock()

    # Start or resume an owned pipeline subprocess.
    def start(self, request: StartRunRequest) -> dict[str, object]:
        with self._lock:
            # Recover durable ownership before deciding whether a duplicate may start.
            existing = self._load_and_recover_state(request.run_id)
            if (
                existing is not None
                and existing.status == "running"
                or request.run_id in self._unowned_process_ids
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A process is already active for this run ID.",
                )
            # Build an argument array; shell execution is deliberately disabled.
            command = [
                *self._cli_command,
                "--run-id",
                request.run_id,
                "--as-of",
                request.as_of.isoformat(),
                "--output-dir",
                str(self._output_root),
                "--progress-format",
                "json",
                *self._cli_arguments,
            ]
            event_log = self._event_log(request.run_id)
            try:
                process = subprocess.Popen(
                    command,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as error:
                self._states[request.run_id] = _RunState(
                    run_id=request.run_id,
                    as_of=request.as_of,
                    status="failed",
                    failure={
                        "code": "cli_start_failed",
                        "message": "The run command could not be started.",
                    },
                )
                self._persist_state(self._states[request.run_id])
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The run command could not be started.",
                ) from error
            # Persist ownership before the monitor thread begins consuming output.
            state = _RunState(run_id=request.run_id, as_of=request.as_of)
            self._states[request.run_id] = state
            self._processes[request.run_id] = process
            self._persist_state(state, process=process)
            Thread(
                target=self._monitor_process,
                args=(state, process, event_log),
                name=f"v2-run-{request.run_id}",
                daemon=True,
            ).start()
            return state.response()

    # Return the current state for a run.
    def get(self, run_id: str) -> dict[str, object]:
        with self._lock:
            state = self._load_and_recover_state(run_id)
            if state is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Run state was not found.",
                )
            return state.response()

    # Cancel an owned running subprocess.
    def cancel(self, run_id: str, *, confirmed: bool) -> dict[str, object]:
        # Cancellation is intentionally two-step because completed stage work is retained.
        if not confirmed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cancellation must be explicitly confirmed.",
            )
        with self._lock:
            state = self._load_and_recover_state(run_id)
            if state is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Run state was not found.",
                )
            if state.status != "running":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The run is not active.",
                )
            # A restarted backend may observe but must not terminate an unowned PID.
            process = self._processes.get(run_id)
            if process is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The active process is not owned by this server.",
                )
            state.status = "cancelled"
            state.failure = None
            self._persist_state(state, process=process)
            process.terminate()
            return state.response()

    # Subscribe a queue to run events and return replay history.
    def subscribe(
        self,
        run_id: str,
        after_sequence: int,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[
        _RunEventLog,
        list[ProgressEvent],
        _Subscriber,
    ]:
        with self._lock:
            history_path = self._history_path(run_id)
            if run_id not in self._states and not history_path.is_file():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Run event history was not found.",
                )
            event_log = self._event_log(run_id)
        backlog, subscriber = event_log.subscribe(after_sequence, loop)
        return event_log, backlog, subscriber

    # Read subprocess events and persist terminal state.
    def _monitor_process(
        self,
        state: _RunState,
        process: subprocess.Popen[str],
        event_log: _RunEventLog,
    ) -> None:
        assert process.stdout is not None
        # Replace malformed CLI records with one safe public event instead of exposing
        # raw subprocess output.
        for source_sequence, line in enumerate(process.stdout, start=1):
            try:
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or type(record.get("sequence")) is not int
                    or record["sequence"] != source_sequence
                ):
                    raise V2ContractError("progress event sequence is out of order")
                event = _progress_event(record, expected_run_id=state.run_id)
            except (
                json.JSONDecodeError,
                TypeError,
                KeyError,
                V2ContractError,
            ):
                event = ProgressEvent(
                    run_id=state.run_id,
                    sequence=1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    module="ui",
                    operation="malformed-event",
                    level="error",
                    message="The run emitted a malformed progress event.",
                )
            event_log.publish(event)
        process.stdout.close()
        # Process exit alone is insufficient for success; publication must validate too.
        exit_code = process.wait()
        with self._lock:
            state.exit_code = exit_code
            if self._processes.get(state.run_id) is process:
                del self._processes[state.run_id]
            if state.status == "cancelled":
                self._persist_state(state)
                return
            if exit_code != 0:
                state.status = "failed"
                state.failure = {
                    "code": "cli_exit_nonzero",
                    "message": "The run command exited unsuccessfully.",
                }
                self._persist_state(state)
                return
            try:
                validate_completed_publication(
                    self.publication_directory(state.run_id),
                    run_id=state.run_id,
                    as_of_date=state.as_of,
                )
            except V2ContractError:
                state.status = "failed"
                state.failure = {
                    "code": "publication_incomplete",
                    "message": "The run did not produce a complete publication.",
                }
                self._persist_state(state)
                return
            state.status = "succeeded"
            self._persist_state(state)

    # Load and recover state.
    def _load_and_recover_state(self, run_id: str) -> _RunState | None:
        # Memory is authoritative while this backend is alive; disk supports restart.
        state = self._states.get(run_id)
        if state is not None:
            self._refresh_unowned_process(run_id, state)
            return state
        path = self._state_path(run_id)
        if not path.is_file():
            return None
        # Accept only the exact persisted shape before reconstructing process ownership.
        try:
            persisted = json.loads(path.read_text(encoding="utf-8"))
            if set(persisted) != {"schema_version", "state", "process_id"}:
                raise ValueError
            if persisted["schema_version"] != "1.0":
                raise ValueError
            record = persisted["state"]
            if not isinstance(record, dict) or set(record) != {
                "run_id",
                "as_of",
                "status",
                "exit_code",
                "failure",
            }:
                raise ValueError
            process_id = persisted["process_id"]
            if process_id is not None and (
                type(process_id) is not int or process_id <= 0
            ):
                raise ValueError
            state = _RunState(
                run_id=record["run_id"],
                as_of=date.fromisoformat(record["as_of"]),
                status=record["status"],
                exit_code=record["exit_code"],
                failure=record["failure"],
            )
            if state.run_id != run_id or state.status not in {
                "running",
                "succeeded",
                "failed",
                "cancelled",
            }:
                raise ValueError
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Run state is invalid.",
            ) from error
        self._states[run_id] = state
        if process_id is not None:
            self._unowned_process_ids[run_id] = process_id
        self._refresh_unowned_process(run_id, state)
        return state

    # Refresh an unowned process state from durable artifacts.
    def _refresh_unowned_process(self, run_id: str, state: _RunState) -> None:
        # A surviving PID stays unowned; a vanished PID turns a running state resumable.
        process_id = self._unowned_process_ids.get(run_id)
        if process_id is None or _process_is_alive(process_id):
            return
        del self._unowned_process_ids[run_id]
        if state.status == "running":
            state.status = "failed"
            state.failure = {
                "code": "backend_interrupted",
                "message": "The backend stopped while the run was active. Resume is available.",
            }
        self._persist_state(state)

    # Atomically persist supervisor state.
    def _persist_state(
        self,
        state: _RunState,
        *,
        process: subprocess.Popen[str] | None = None,
    ) -> None:
        path = self._state_path(state.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        process_id = None
        if process is not None and process.poll() is None:
            process_id = process.pid
        atomic_write_json(
            path,
            {
                "schema_version": "1.0",
                "state": state.response(),
                "process_id": process_id,
            },
        )

    # Return the event log for a run.
    def _event_log(self, run_id: str) -> _RunEventLog:
        event_log = self._event_logs.get(run_id)
        if event_log is None:
            event_log = _RunEventLog(self._history_path(run_id), run_id)
            self._event_logs[run_id] = event_log
        return event_log

    # Return the durable event-history path for a run.
    def _history_path(self, run_id: str) -> Path:
        return self._run_directory(run_id) / EVENT_HISTORY_NAME

    # Return the durable supervisor-state path for a run.
    def _state_path(self, run_id: str) -> Path:
        return self._run_directory(run_id) / RUN_STATE_NAME

    # Return the final publication directory for a run.
    def publication_directory(self, run_id: str) -> Path:
        return self._run_directory(run_id) / "publication"

    # Return a run directory confined beneath the configured output root.
    def _run_directory(self, run_id: str) -> Path:
        # Resolve before use to prevent run IDs from escaping the configured output root.
        candidate = self._output_root / run_id
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._output_root):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run state was not found.",
            )
        return candidate


# Normalize an incoming subprocess progress event.
def _progress_event(record: object, *, expected_run_id: str) -> ProgressEvent:
    # Exact keys prevent unknown subprocess data from reaching the browser contract.
    if not isinstance(record, dict):
        raise V2ContractError("progress event must be an object")
    expected_keys = {
        "schema_version",
        "run_id",
        "sequence",
        "timestamp",
        "module",
        "operation",
        "level",
        "message",
    }
    if "progress" in record:
        expected_keys.add("progress")
    if set(record) != expected_keys or record.get("schema_version") != "1.0":
        raise V2ContractError("progress event has an invalid shape")
    if record.get("run_id") != expected_run_id:
        raise V2ContractError("progress event belongs to a different run")
    # Require an aware ISO timestamp so events from different stages remain comparable.
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str):
        raise V2ContractError("progress event timestamp is invalid")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise V2ContractError("progress event timestamp is invalid") from error
    if parsed_timestamp.tzinfo is None:
        raise V2ContractError("progress event timestamp must include a timezone")
    # Progress is optional, but when present its bounds are validated by the domain type.
    progress_record = record.get("progress")
    progress = None
    if progress_record is not None:
        if not isinstance(progress_record, dict) or set(progress_record) != {
            "current",
            "total",
        }:
            raise V2ContractError("progress bounds have an invalid shape")
        progress = BoundedProgress(
            current=progress_record["current"], total=progress_record["total"]
        )
    # Rebuild the event rather than forwarding the source mapping directly.
    event = ProgressEvent(
        run_id=record["run_id"],
        sequence=record["sequence"],
        timestamp=record["timestamp"],
        module=record["module"],
        operation=record["operation"],
        level=record["level"],
        message=_public_event_message(
            module=record["module"],
            operation=record["operation"],
            level=record["level"],
            source_message=record["message"],
        ),
        progress=progress,
    )
    try:
        event.record()
    except Exception as error:
        raise V2ContractError("progress event is schema-invalid") from error
    return event


# Redact sensitive values from a public event message.
def _public_event_message(
    *,
    module: object,
    operation: object,
    level: object,
    source_message: object,
) -> str:
    if not isinstance(source_message, str) or not source_message:
        raise V2ContractError("progress event message is invalid")
    if (
        not isinstance(module, str)
        or EVENT_IDENTIFIER_PATTERN.fullmatch(module) is None
        or not isinstance(operation, str)
        or EVENT_IDENTIFIER_PATTERN.fullmatch(operation) is None
    ):
        raise V2ContractError("progress event identifiers are invalid")
    if not isinstance(level, str):
        raise V2ContractError("progress event level is invalid")
    presentation = {
        "info": "update",
        "warning": "warning",
        "error": "failure",
    }.get(level)
    if presentation is None:
        raise V2ContractError("progress event level is invalid")
    return (
        f"{module.replace('-', ' ').capitalize()}: "
        f"{operation.replace('-', ' ')} {presentation}."
    )


# Check whether a process ID is still alive.
def _process_is_alive(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# Create the loopback application's process-control API.
def create_app(
    *,
    output_root: Path,
    cli_command: Sequence[str] = ("audience-trend-miner",),
    cli_arguments: Sequence[str] = (),
) -> FastAPI:
    """Create the loopback application's process-control API."""
    supervisor = _RunSupervisor(output_root.resolve(), cli_command, cli_arguments)
    app = FastAPI(title="AudienceIntelligence V2")
    app.mount("/assets", StaticFiles(directory=STATIC_DIRECTORY), name="assets")

    # Serve the application index page.
    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    # Validate and start a pipeline run.
    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    def start_run(request: StartRunRequest) -> dict[str, object]:
        return supervisor.start(request)

    # Return one pipeline run state.
    @app.get("/api/runs/{run_id}")
    def get_run(
        run_id: Annotated[str, ApiPath(pattern=RUN_ID_PATTERN)],
    ) -> dict[str, object]:
        return supervisor.get(run_id)

    # Cancel one pipeline run.
    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(
        request: CancelRunRequest,
        run_id: Annotated[str, ApiPath(pattern=RUN_ID_PATTERN)],
    ) -> dict[str, object]:
        return supervisor.cancel(run_id, confirmed=request.confirmed)

    # Return the validated portfolio for a completed run.
    @app.get("/api/runs/{run_id}/portfolio", response_class=JSONResponse)
    def get_portfolio(
        run_id: Annotated[str, ApiPath(pattern=RUN_ID_PATTERN)],
    ) -> JSONResponse:
        publication_directory = supervisor.publication_directory(run_id)
        try:
            validate_completed_publication(publication_directory, run_id=run_id)
            portfolio = json.loads(
                (publication_directory / "portfolio.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError, V2ContractError) as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="A completed Audience Portfolio was not found.",
            ) from error
        return JSONResponse(portfolio)

    # Stream replayed and live events to one browser client.
    @app.websocket("/api/runs/{run_id}/events")
    async def stream_run_events(
        websocket: WebSocket,
        run_id: Annotated[str, ApiPath(pattern=RUN_ID_PATTERN)],
        after_sequence: int = 0,
    ) -> None:
        if after_sequence < 0:
            await websocket.close(
                code=1008, reason="after_sequence must not be negative"
            )
            return
        try:
            event_log, backlog, subscriber = supervisor.subscribe(
                run_id, after_sequence, asyncio.get_running_loop()
            )
        except HTTPException:
            await websocket.close(code=4404, reason="Run event history was not found.")
            return
        await websocket.accept()
        try:
            for event in backlog:
                await websocket.send_json(event.record())
            while True:
                event = await subscriber.queue.get()
                await websocket.send_json(event.record())
        except WebSocketDisconnect:
            pass
        finally:
            event_log.unsubscribe(subscriber)

    return app


# Serve the local application, bound to loopback unless explicitly changed.
def serve(
    *,
    output_root: Path,
    host: str = DEFAULT_HOST,
    port: int = 8000,
) -> None:
    """Serve the local application, bound to loopback unless explicitly changed."""
    import uvicorn

    uvicorn.run(create_app(output_root=output_root), host=host, port=port)
