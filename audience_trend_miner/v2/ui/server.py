from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from threading import Lock, Thread
from typing import Annotated, Literal, Sequence, cast

from fastapi import (
    FastAPI,
    HTTPException,
    Path as ApiPath,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

from audience_trend_miner.v2.run_publication import validate_completed_publication
from audience_trend_miner.v2.shared import BoundedProgress, ProgressEvent, V2ContractError


DEFAULT_HOST = "127.0.0.1"
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
EVENT_HISTORY_NAME = "ui-events.jsonl"


class StartRunRequest(BaseModel):
    run_id: Annotated[str, Field(pattern=RUN_ID_PATTERN)]
    as_of: date


@dataclass
class _RunState:
    run_id: str
    as_of: date
    status: Literal["running", "succeeded", "failed"] = "running"
    exit_code: int | None = None
    failure: dict[str, str] | None = None

    def response(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "as_of": self.as_of.isoformat(),
            "status": self.status,
            "exit_code": self.exit_code,
            "failure": self.failure,
        }


class _RunEventLog:
    """Durable ordered event history with independent live subscribers."""

    def __init__(self, path: Path, run_id: str) -> None:
        self._path = path
        self._run_id = run_id
        self._lock = Lock()
        self._events = self._load()
        self._subscribers: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, object]]]
        ] = []

    def publish(self, event: ProgressEvent) -> dict[str, object]:
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
            ).record()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(normalized, separators=(",", ":"), sort_keys=True)
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._events.append(normalized)
            connected_subscribers = []
            for loop, subscriber in self._subscribers:
                try:
                    loop.call_soon_threadsafe(subscriber.put_nowait, normalized)
                except RuntimeError:
                    continue
                connected_subscribers.append((loop, subscriber))
            self._subscribers = connected_subscribers
            return normalized

    def subscribe(
        self,
        after_sequence: int,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[list[dict[str, object]], asyncio.Queue[dict[str, object]]]:
        subscriber: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        with self._lock:
            snapshot = list(self._events)
            self._subscribers.append((loop, subscriber))
        return (
            [
                event
                for event in snapshot
                if cast(int, event["sequence"]) > after_sequence
            ],
            subscriber,
        )

    def unsubscribe(self, subscriber: asyncio.Queue[dict[str, object]]) -> None:
        with self._lock:
            self._subscribers = [
                registered
                for registered in self._subscribers
                if registered[1] is not subscriber
            ]

    def _load(self) -> list[dict[str, object]]:
        if not self._path.exists():
            return []
        events: list[dict[str, object]] = []
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
            events.append(event.record())
        return events


class _RunSupervisor:
    def __init__(self, output_root: Path, cli_command: Sequence[str]) -> None:
        self._output_root = output_root
        self._cli_command = tuple(cli_command)
        self._states: dict[str, _RunState] = {}
        self._event_logs: dict[str, _RunEventLog] = {}
        self._lock = Lock()

    def start(self, request: StartRunRequest) -> dict[str, object]:
        with self._lock:
            existing = self._states.get(request.run_id)
            if existing is not None and existing.status == "running":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A process is already active for this run ID.",
                )
            command = [
                *self._cli_command,
                "v2-run",
                "--run-id",
                request.run_id,
                "--as-of",
                request.as_of.isoformat(),
                "--output-dir",
                str(self._output_root),
                "--progress-format",
                "json",
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
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The run command could not be started.",
                ) from error
            state = _RunState(run_id=request.run_id, as_of=request.as_of)
            self._states[request.run_id] = state
            Thread(
                target=self._monitor_process,
                args=(state, process, event_log),
                name=f"v2-run-{request.run_id}",
                daemon=True,
            ).start()
            return state.response()

    def get(self, run_id: str) -> dict[str, object]:
        with self._lock:
            state = self._states.get(run_id)
            if state is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Run state was not found.",
                )
            return state.response()

    def subscribe(
        self,
        run_id: str,
        after_sequence: int,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[
        _RunEventLog,
        list[dict[str, object]],
        asyncio.Queue[dict[str, object]],
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

    def _monitor_process(
        self,
        state: _RunState,
        process: subprocess.Popen[str],
        event_log: _RunEventLog,
    ) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                record = json.loads(line)
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
        exit_code = process.wait()
        with self._lock:
            state.exit_code = exit_code
            if exit_code != 0:
                state.status = "failed"
                state.failure = {
                    "code": "cli_exit_nonzero",
                    "message": "The run command exited unsuccessfully.",
                }
                return
            try:
                validate_completed_publication(
                    self._output_root / state.run_id / "publication",
                    run_id=state.run_id,
                    as_of_date=state.as_of,
                )
            except V2ContractError:
                state.status = "failed"
                state.failure = {
                    "code": "publication_incomplete",
                    "message": "The run did not produce a complete publication.",
                }
                return
            state.status = "succeeded"

    def _event_log(self, run_id: str) -> _RunEventLog:
        event_log = self._event_logs.get(run_id)
        if event_log is None:
            event_log = _RunEventLog(self._history_path(run_id), run_id)
            self._event_logs[run_id] = event_log
        return event_log

    def _history_path(self, run_id: str) -> Path:
        return self._output_root / run_id / EVENT_HISTORY_NAME


def _progress_event(record: object, *, expected_run_id: str) -> ProgressEvent:
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
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str):
        raise V2ContractError("progress event timestamp is invalid")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise V2ContractError("progress event timestamp is invalid") from error
    if parsed_timestamp.tzinfo is None:
        raise V2ContractError("progress event timestamp must include a timezone")
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
    event = ProgressEvent(
        run_id=record["run_id"],
        sequence=record["sequence"],
        timestamp=record["timestamp"],
        module=record["module"],
        operation=record["operation"],
        level=record["level"],
        message=record["message"],
        progress=progress,
    )
    try:
        event.record()
    except Exception as error:
        raise V2ContractError("progress event is schema-invalid") from error
    return event


def create_app(
    *,
    output_root: Path,
    cli_command: Sequence[str] = ("audience-trend-miner",),
) -> FastAPI:
    """Create the loopback application's process-control API."""
    supervisor = _RunSupervisor(output_root.resolve(), cli_command)
    app = FastAPI(title="AudienceIntelligence V2")

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    def start_run(request: StartRunRequest) -> dict[str, object]:
        return supervisor.start(request)

    @app.get("/api/runs/{run_id}")
    def get_run(
        run_id: Annotated[str, ApiPath(pattern=RUN_ID_PATTERN)],
    ) -> dict[str, object]:
        return supervisor.get(run_id)

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
                await websocket.send_json(event)
            while True:
                event = await subscriber.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            event_log.unsubscribe(subscriber)

    return app


def serve(
    *,
    output_root: Path,
    host: str = DEFAULT_HOST,
    port: int = 8000,
) -> None:
    """Serve the local application, bound to loopback unless explicitly changed."""
    import uvicorn

    uvicorn.run(create_app(output_root=output_root), host=host, port=port)
