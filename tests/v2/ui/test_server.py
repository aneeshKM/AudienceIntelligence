from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from audience_trend_miner.v2.ui import create_app, serve


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        if response.status_code == 200 and response.json()["status"] != "running":
            return response.json()
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


def _write_completed_publication(root: Path, run_id: str) -> None:
    publication = root / run_id / "publication"
    publication.mkdir(parents=True)
    windows = {
        "previous": {"start": "2026-07-03", "end": "2026-07-09"},
        "current": {"start": "2026-07-10", "end": "2026-07-16"},
    }
    products = {
        "portfolio.json": {
            "schema_version": "1.0",
            "run_id": run_id,
            "as_of_date": "2026-07-17",
            "nominal_windows": windows,
            "audience_portfolio": [],
            "completion": {"status": "complete", "empty": True},
        },
        "audit.json": {
            "schema_version": "1.0",
            "run_id": run_id,
            "stage_evidence": {
                "wikimedia-evidence": {},
                "semantic-audience-formation": {},
                "cluster-adjudication": {},
                "trend-portfolio": {},
            },
        },
    }
    for name, product in products.items():
        (publication / name).write_text(
            json.dumps(product, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    modules = {
        name: {
            "status": "complete",
            "artifact_schema_version": "2.0",
            "sha256": "sha256:" + "0" * 64,
        }
        for name in (
            "wikimedia-evidence",
            "semantic-audience-formation",
            "cluster-adjudication",
            "trend-portfolio",
        )
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "as_of_date": "2026-07-17",
        "nominal_windows": windows,
        "configuration_provenance": {},
        "modules": modules,
        "schemas": {
            "portfolio.json": "1.0",
            "audit.json": "1.0",
            "manifest.json": "1.0",
        },
        "published_artifacts": {
            name: {
                "schema_version": "1.0",
                "sha256": "sha256:"
                + hashlib.sha256((publication / name).read_bytes()).hexdigest(),
                "bytes": (publication / name).stat().st_size,
            }
            for name in products
        },
        "integrity": {
            "algorithm": "sha256",
            "encoding": "utf-8",
            "manifest_excludes_self": True,
        },
        "completion": {"status": "complete"},
    }
    (publication / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class RunServerTest(unittest.TestCase):
    def test_flushed_cli_events_are_forwarded_live_before_process_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_cli = root / "streaming_cli.py"
            fake_cli.write_text(
                "import json, sys, time\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "def emit(sequence, operation, message, current):\n"
                "    print(json.dumps({\n"
                "        'schema_version': '1.0', 'run_id': run_id,\n"
                "        'sequence': sequence, 'timestamp': '2026-07-17T12:00:00+00:00',\n"
                "        'module': 'wikimedia-evidence', 'operation': operation,\n"
                "        'level': 'info', 'message': message,\n"
                "        'progress': {'current': current, 'total': 2},\n"
                "    }), flush=True)\n"
                "emit(1, 'fetch', 'first day', 1)\n"
                "time.sleep(0.5)\n"
                "emit(2, 'fetch', 'second day', 2)\n"
                "time.sleep(0.5)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=root / "runs",
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                started = client.post(
                    "/api/runs",
                    json={"run_id": "live-run", "as_of": "2026-07-17"},
                )
                with client.websocket_connect(
                    "/api/runs/live-run/events?after_sequence=0"
                ) as websocket:
                    first = websocket.receive_json()
                    running = client.get("/api/runs/live-run").json()
                    second = websocket.receive_json()

            self.assertEqual(started.status_code, 202)
            self.assertEqual(running["status"], "running")
            self.assertEqual(first["message"], "first day")
            self.assertEqual(second["message"], "second day")
            self.assertEqual([first["sequence"], second["sequence"]], [1, 2])
            self.assertEqual(
                set(first),
                {
                    "schema_version",
                    "run_id",
                    "sequence",
                    "timestamp",
                    "module",
                    "operation",
                    "level",
                    "message",
                    "progress",
                },
            )

    def test_reconnect_replays_only_the_durable_gap_then_continues_live(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_cli = root / "reconnect_cli.py"
            fake_cli.write_text(
                "import json, sys, time\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "for sequence in range(1, 4):\n"
                "    print(json.dumps({\n"
                "        'schema_version': '1.0', 'run_id': run_id,\n"
                "        'sequence': sequence, 'timestamp': '2026-07-17T12:00:00+00:00',\n"
                "        'module': 'trend-portfolio', 'operation': 'qualify',\n"
                "        'level': 'info', 'message': f'event {sequence}',\n"
                "    }), flush=True)\n"
                "    time.sleep(0.25)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=root / "runs",
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                client.post(
                    "/api/runs",
                    json={"run_id": "reconnect-run", "as_of": "2026-07-17"},
                )
                with client.websocket_connect(
                    "/api/runs/reconnect-run/events?after_sequence=0"
                ) as websocket:
                    first = websocket.receive_json()
                time.sleep(0.3)
                with client.websocket_connect(
                    "/api/runs/reconnect-run/events?after_sequence=1"
                ) as websocket:
                    missing = websocket.receive_json()
                    live = websocket.receive_json()

            self.assertEqual(first["sequence"], 1)
            self.assertEqual(
                [
                    (missing["sequence"], missing["message"]),
                    (live["sequence"], live["message"]),
                ],
                [(2, "event 2"), (3, "event 3")],
            )

    def test_backend_restart_recovers_event_history_by_run_and_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            fake_cli = root / "history_cli.py"
            fake_cli.write_text(
                "import json, sys\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "for sequence in range(1, 4):\n"
                "    print(json.dumps({\n"
                "        'schema_version': '1.0', 'run_id': run_id,\n"
                "        'sequence': sequence, 'timestamp': '2026-07-17T12:00:00+00:00',\n"
                "        'module': 'run-publication', 'operation': 'write',\n"
                "        'level': 'info', 'message': f'event {sequence}',\n"
                "    }), flush=True)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            first_app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(fake_cli)),
            )
            with TestClient(first_app) as client:
                client.post(
                    "/api/runs",
                    json={"run_id": "recovered-run", "as_of": "2026-07-17"},
                )
                _wait_for_terminal(client, "recovered-run")

            recovered_app = create_app(output_root=output_root)
            with TestClient(recovered_app) as recovered_client:
                with recovered_client.websocket_connect(
                    "/api/runs/recovered-run/events?after_sequence=1"
                ) as websocket:
                    recovered = [websocket.receive_json(), websocket.receive_json()]

            self.assertEqual(
                [(event["sequence"], event["message"]) for event in recovered],
                [(2, "event 2"), (3, "event 3")],
            )

    def test_malformed_cli_events_are_safely_normalized_without_stopping_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_cli = root / "malformed_cli.py"
            fake_cli.write_text(
                "import json, sys\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "print('private malformed details', flush=True)\n"
                "print(json.dumps({\n"
                "    'schema_version': '1.0', 'run_id': run_id,\n"
                "    'sequence': 2, 'timestamp': '2026-07-17T12:00:00+00:00',\n"
                "    'module': 'wikimedia-evidence', 'operation': 'fetch',\n"
                "    'level': 'info', 'message': 'invalid bounds',\n"
                "    'progress': {'current': True, 'total': 2},\n"
                "}), flush=True)\n"
                "print(json.dumps({\n"
                "    'schema_version': '1.0', 'run_id': run_id,\n"
                "    'sequence': 3, 'timestamp': '2026-07-17T12:00:00+00:00',\n"
                "    'module': 'wikimedia-evidence', 'operation': 'fetch',\n"
                "    'level': 'info', 'message': 'valid event',\n"
                "}), flush=True)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=root / "runs",
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                client.post(
                    "/api/runs",
                    json={"run_id": "malformed-run", "as_of": "2026-07-17"},
                )
                terminal = _wait_for_terminal(client, "malformed-run")
                with client.websocket_connect(
                    "/api/runs/malformed-run/events?after_sequence=0"
                ) as websocket:
                    events = [
                        websocket.receive_json(),
                        websocket.receive_json(),
                        websocket.receive_json(),
                    ]

            self.assertEqual(terminal["status"], "failed")
            self.assertEqual([event["sequence"] for event in events], [1, 2, 3])
            self.assertEqual(
                [event["operation"] for event in events],
                ["malformed-event", "malformed-event", "fetch"],
            )
            self.assertEqual(events[2]["message"], "valid event")
            self.assertNotIn("private malformed details", json.dumps(events))

    def test_server_binds_to_loopback_by_default(self) -> None:
        with patch("uvicorn.run") as run:
            serve(output_root=Path("runs"))

        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")

    def test_start_validates_input_and_invokes_cli_as_an_argument_array(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invocation_path = root / "invocation.json"
            fake_cli = root / "fake_cli.py"
            fake_cli.write_text(
                "import json, pathlib, sys\n"
                f"pathlib.Path({str(invocation_path)!r}).write_text(json.dumps(sys.argv[1:]))\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=root / "runs",
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                invalid_date = client.post(
                    "/api/runs", json={"run_id": "safe-run", "as_of": "17-07-2026"}
                )
                unsafe_run = client.post(
                    "/api/runs", json={"run_id": "../escape", "as_of": "2026-07-17"}
                )
                started = client.post(
                    "/api/runs", json={"run_id": "safe-run", "as_of": "2026-07-17"}
                )
                terminal = _wait_for_terminal(client, "safe-run")

            self.assertEqual(invalid_date.status_code, 422)
            self.assertEqual(unsafe_run.status_code, 422)
            self.assertEqual(started.status_code, 202)
            self.assertEqual(
                json.loads(invocation_path.read_text(encoding="utf-8")),
                [
                    "v2-run",
                    "--run-id",
                    "safe-run",
                    "--as-of",
                    "2026-07-17",
                    "--output-dir",
                    str((root / "runs").resolve()),
                    "--progress-format",
                    "json",
                ],
            )
            self.assertEqual(
                terminal,
                {
                    "run_id": "safe-run",
                    "as_of": "2026-07-17",
                    "status": "failed",
                    "exit_code": 1,
                    "failure": {
                        "code": "cli_exit_nonzero",
                        "message": "The run command exited unsuccessfully.",
                    },
                },
            )
            self.assertNotIn(str(root), json.dumps(terminal))

    def test_process_ownership_survives_clients_and_rejects_only_duplicate_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_cli = root / "slow_cli.py"
            fake_cli.write_text(
                "import time\n"
                "time.sleep(0.3)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=root / "runs",
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as first_client:
                started = first_client.post(
                    "/api/runs", json={"run_id": "owned-run", "as_of": "2026-07-17"}
                )
            with TestClient(app) as reconnected_client:
                retained = reconnected_client.get("/api/runs/owned-run")
                duplicate = reconnected_client.post(
                    "/api/runs", json={"run_id": "owned-run", "as_of": "2026-07-17"}
                )
                different = reconnected_client.post(
                    "/api/runs", json={"run_id": "other-run", "as_of": "2026-07-17"}
                )
                owned_terminal = _wait_for_terminal(reconnected_client, "owned-run")
                other_terminal = _wait_for_terminal(reconnected_client, "other-run")

            self.assertEqual(started.status_code, 202)
            self.assertEqual(retained.status_code, 200)
            self.assertEqual(retained.json()["status"], "running")
            self.assertEqual(duplicate.status_code, 409)
            self.assertEqual(different.status_code, 202)
            self.assertEqual(owned_terminal["status"], "failed")
            self.assertEqual(other_terminal["status"], "failed")

    def test_zero_exit_without_completed_publication_is_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_cli = root / "successful_cli.py"
            fake_cli.write_text("raise SystemExit(0)\n", encoding="utf-8")
            app = create_app(
                output_root=root / "runs",
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                started = client.post(
                    "/api/runs", json={"run_id": "partial-run", "as_of": "2026-07-17"}
                )
                terminal = _wait_for_terminal(client, "partial-run")

            self.assertEqual(started.status_code, 202)
            self.assertEqual(
                terminal,
                {
                    "run_id": "partial-run",
                    "as_of": "2026-07-17",
                    "status": "failed",
                    "exit_code": 0,
                    "failure": {
                        "code": "publication_incomplete",
                        "message": "The run did not produce a complete publication.",
                    },
                },
            )

    def test_cli_start_failure_is_retained_as_sanitized_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            app = create_app(
                output_root=root / "runs",
                cli_command=(str(root / "missing-cli"),),
            )

            with TestClient(app) as client:
                failed_start = client.post(
                    "/api/runs",
                    json={"run_id": "unstarted-run", "as_of": "2026-07-17"},
                )
                retained = client.get("/api/runs/unstarted-run")

            self.assertEqual(failed_start.status_code, 503)
            self.assertEqual(
                retained.json(),
                {
                    "run_id": "unstarted-run",
                    "as_of": "2026-07-17",
                    "status": "failed",
                    "exit_code": None,
                    "failure": {
                        "code": "cli_start_failed",
                        "message": "The run command could not be started.",
                    },
                },
            )
            self.assertNotIn(str(root), json.dumps(retained.json()))

    def test_success_requires_zero_exit_and_valid_same_run_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            _write_completed_publication(output_root, "complete-run")
            fake_cli = root / "successful_cli.py"
            fake_cli.write_text("raise SystemExit(0)\n", encoding="utf-8")
            app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                started = client.post(
                    "/api/runs",
                    json={"run_id": "complete-run", "as_of": "2026-07-17"},
                )
                terminal = _wait_for_terminal(client, "complete-run")

            self.assertEqual(started.status_code, 202)
            self.assertEqual(
                terminal,
                {
                    "run_id": "complete-run",
                    "as_of": "2026-07-17",
                    "status": "succeeded",
                    "exit_code": 0,
                    "failure": None,
                },
            )

    def test_success_rejects_publication_for_a_different_as_of_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            _write_completed_publication(output_root, "stale-run")
            fake_cli = root / "successful_cli.py"
            fake_cli.write_text("raise SystemExit(0)\n", encoding="utf-8")
            app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                client.post(
                    "/api/runs",
                    json={"run_id": "stale-run", "as_of": "2026-07-18"},
                )
                terminal = _wait_for_terminal(client, "stale-run")

            self.assertEqual(terminal["status"], "failed")
            failure = terminal["failure"]
            self.assertIsInstance(failure, dict)
            assert isinstance(failure, dict)
            self.assertEqual(failure["code"], "publication_incomplete")


if __name__ == "__main__":
    unittest.main()
