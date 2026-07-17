from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import subprocess
from threading import Lock, Thread
from typing import Annotated, Literal, Sequence

from fastapi import FastAPI, HTTPException, Path as ApiPath, status
from pydantic import BaseModel, Field

from audience_trend_miner.v2.run_publication import validate_completed_publication
from audience_trend_miner.v2.shared import V2ContractError


DEFAULT_HOST = "127.0.0.1"
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


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


class _RunSupervisor:
    def __init__(self, output_root: Path, cli_command: Sequence[str]) -> None:
        self._output_root = output_root
        self._cli_command = tuple(cli_command)
        self._states: dict[str, _RunState] = {}
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
            try:
                process = subprocess.Popen(
                    command,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as error:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="The run command could not be started.",
                ) from error
            state = _RunState(run_id=request.run_id, as_of=request.as_of)
            self._states[request.run_id] = state
            Thread(
                target=self._wait_for_process,
                args=(state, process),
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

    def _wait_for_process(
        self, state: _RunState, process: subprocess.Popen[bytes]
    ) -> None:
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
                )
            except V2ContractError:
                state.status = "failed"
                state.failure = {
                    "code": "publication_incomplete",
                    "message": "The run did not produce a complete publication.",
                }
                return
            state.status = "succeeded"


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
