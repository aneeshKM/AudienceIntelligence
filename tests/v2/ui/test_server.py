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

from audience_trend_miner import __main__ as cli
from audience_trend_miner.v2.ui import create_app, serve


# Return wait for terminal.
def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        if response.status_code == 200 and response.json()["status"] != "running":
            return response.json()
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


# Write completed publication.
def _write_completed_publication(
    root: Path,
    run_id: str,
    *,
    audiences: list[dict[str, object]] | None = None,
) -> None:
    publication = root / run_id / "publication"
    publication.mkdir(parents=True)
    windows = {
        "previous": {"start": "2026-07-03", "end": "2026-07-09"},
        "current": {"start": "2026-07-10", "end": "2026-07-16"},
    }
    audience_portfolio = audiences or []
    products = {
        "portfolio.json": {
            "schema_version": "1.0",
            "run_id": run_id,
            "as_of_date": "2026-07-17",
            "nominal_windows": windows,
            "audience_portfolio": audience_portfolio,
            "completion": {
                "status": "complete",
                "empty": not audience_portfolio,
            },
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


# Group tests for run server behavior.
class RunServerTest(unittest.TestCase):
    # Verify: ui launch loads dotenv without overriding exported values.
    def test_ui_launch_loads_dotenv_without_overriding_exported_values(self) -> None:
        with (
            patch.object(cli, "load_dotenv") as load_dotenv,
            patch.object(cli, "_v2_ui_main", return_value=0) as ui_main,
            patch.object(sys, "argv", ["audience-trend-miner", "v2-ui"]),
        ):
            self.assertEqual(cli.main(), 0)

        load_dotenv.assert_called_once_with(override=False)
        ui_main.assert_called_once_with([])

    # Verify: primary page exposes one semantic installer interface.
    def test_primary_page_exposes_one_semantic_installer_interface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(output_root=Path(temporary_directory) / "runs")

            with TestClient(app) as client:
                response = client.get("/")

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/html"))
            self.assertIn("<main", response.text)
            self.assertIn('id="run-form"', response.text)
            self.assertIn('id="new-run"', response.text)
            self.assertNotIn('id="run-id"', response.text)
            self.assertIn('id="run-status"', response.text)
            self.assertIn('aria-live="polite"', response.text)
            self.assertIn('src="/assets/app.js"', response.text)

    # Verify: primary page contains accessible progress and portfolio regions.
    def test_primary_page_contains_accessible_progress_and_portfolio_regions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(output_root=Path(temporary_directory) / "runs")

            with TestClient(app) as client:
                response = client.get("/")

            self.assertIn('id="progress-feed"', response.text)
            self.assertIn('id="follow-progress"', response.text)
            self.assertIn('id="cancel-run"', response.text)
            self.assertIn('aria-describedby="cancel-run-hint"', response.text)
            self.assertIn('id="portfolio"', response.text)
            self.assertIn('id="growing-audiences"', response.text)
            self.assertIn('id="shrinking-audiences"', response.text)
            self.assertIn(
                "No robust audience trends qualified for this run.", response.text
            )
            self.assertIn(
                "Wikipedia attention does not prove reader identity, intent, income, "
                "causation, or future behavior.",
                response.text,
            )

    # Verify: completed run exposes only the validated portfolio contract.
    def test_completed_run_exposes_only_the_validated_portfolio_contract(self) -> None:
        audience = {
            "cluster_id": "cluster-clean-air",
            "source_preliminary_cluster_id": "preliminary-clean-air",
            "direction": "robust_growth",
            "traffic": {
                window: {
                    "observed_total": total,
                    "observed_page_days": 14,
                    "successful_days": 7,
                    "conservative_observed_minimum": total,
                    "conservative_observed_maximum": total + 100,
                    "seven_day_equivalent": float(total),
                    "minimum": float(total),
                    "maximum": float(total + 100),
                }
                for window, total in (("previous", 100_000), ("current", 125_000))
            },
            "percentage_change": 25.0,
            "coverage": {"previous": 1.0, "current": 0.86},
            "confidence": "robust",
            "impact_score": 3.5,
            "narrative": {
                "name": "Clean-air households",
                "summary": "Attention expanded across practical air-quality topics.",
                "commercial_interpretation": "Home-health planning is becoming salient.",
                "brand_categories": ["Air purifiers"],
                "buying_power_rating": "medium",
                "buying_power_rationale": "Considered purchases span several price points.",
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "runs"
            _write_completed_publication(root, "portfolio-run", audiences=[audience])
            app = create_app(output_root=root)

            with TestClient(app) as client:
                response = client.get("/api/runs/portfolio-run/portfolio")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["audience_portfolio"], [audience])
            self.assertNotIn("stage_evidence", response.text)

    # Verify: flushed cli events are forwarded live before process exit.
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
            self.assertEqual(
                first["message"], "Wikimedia evidence: fetch update."
            )
            self.assertEqual(
                second["message"], "Wikimedia evidence: fetch update."
            )
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

    # Verify: reconnect replays only the durable gap then continues live.
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
                [
                    (2, "Trend portfolio: qualify update."),
                    (3, "Trend portfolio: qualify update."),
                ],
            )

    # Verify: backend restart recovers event history by run and sequence.
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
                [
                    (2, "Run publication: write update."),
                    (3, "Run publication: write update."),
                ],
            )

    # Verify: malformed cli events are safely normalized without stopping reader.
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
            self.assertEqual(
                events[2]["message"], "Wikimedia evidence: fetch update."
            )
            self.assertNotIn("private malformed details", json.dumps(events))

    # Verify: out of order cli sequence is not hidden by durable ordering.
    def test_out_of_order_cli_sequence_is_not_hidden_by_durable_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_cli = root / "out_of_order_cli.py"
            fake_cli.write_text(
                "import json, sys\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "def emit(message):\n"
                "    print(json.dumps({\n"
                "        'schema_version': '1.0', 'run_id': run_id,\n"
                "        'sequence': 2, 'timestamp': '2026-07-17T12:00:00+00:00',\n"
                "        'module': 'trend-portfolio', 'operation': 'qualify',\n"
                "        'level': 'info', 'message': message,\n"
                "    }), flush=True)\n"
                "emit('skipped sequence one')\n"
                "emit('second event')\n"
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
                    json={"run_id": "ordered-run", "as_of": "2026-07-17"},
                )
                _wait_for_terminal(client, "ordered-run")
                with client.websocket_connect(
                    "/api/runs/ordered-run/events?after_sequence=0"
                ) as websocket:
                    events = [websocket.receive_json(), websocket.receive_json()]

            self.assertEqual([event["sequence"] for event in events], [1, 2])
            self.assertEqual(events[0]["operation"], "malformed-event")
            self.assertEqual(
                events[1]["message"], "Trend portfolio: qualify update."
            )
            self.assertNotIn("skipped sequence one", json.dumps(events))

    # Verify: server binds to loopback by default.
    def test_server_binds_to_loopback_by_default(self) -> None:
        with patch("uvicorn.run") as run:
            serve(output_root=Path("runs"))

        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")

    # Verify: start validates input and invokes cli as an argument array.
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

    # Verify: run paths cannot escape the configured output root.
    def test_run_paths_cannot_escape_the_configured_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            output_root.mkdir()
            outside_root = root / "outside"
            _write_completed_publication(outside_root, "linked-run")
            (output_root / "linked-run").symlink_to(outside_root / "linked-run")
            app = create_app(output_root=output_root)

            with TestClient(app) as client:
                state = client.get("/api/runs/linked-run")
                portfolio = client.get("/api/runs/linked-run/portfolio")

            self.assertEqual(state.status_code, 404)
            self.assertEqual(portfolio.status_code, 404)
            self.assertNotIn("audience_portfolio", portfolio.text)

    # Verify: structured events redact credentials before durable storage.
    def test_structured_events_redact_credentials_before_durable_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            fake_cli = root / "secret_cli.py"
            fake_cli.write_text(
                "import json, sys\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "print(json.dumps({\n"
                "    'schema_version': '1.0', 'run_id': run_id, 'sequence': 1,\n"
                "    'timestamp': '2026-07-17T12:00:00+00:00',\n"
                "    'module': 'wikimedia-evidence', 'operation': 'fetch',\n"
                "    'level': 'warning',\n"
                "    'message': 'retry with Authorization: Bearer top-secret-token',\n"
                "}), flush=True)\n"
                "print(json.dumps({\n"
                "    'schema_version': '1.0', 'run_id': run_id, 'sequence': 2,\n"
                "    'timestamp': '2026-07-17T12:00:01+00:00',\n"
                "    'module': 'cluster-adjudication', 'operation': 'review',\n"
                "    'level': 'info', 'message': 'system prompt: private instructions',\n"
                "}), flush=True)\n"
                "print(json.dumps({\n"
                "    'schema_version': '1.0', 'run_id': run_id, 'sequence': 3,\n"
                "    'timestamp': '2026-07-17T12:00:02+00:00',\n"
                "    'module': 'trend-portfolio', 'operation': 'narrate',\n"
                "    'level': 'info', 'message': 'raw model response: private judgment',\n"
                "}), flush=True)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                client.post(
                    "/api/runs",
                    json={"run_id": "secret-run", "as_of": "2026-07-17"},
                )
                _wait_for_terminal(client, "secret-run")
                with client.websocket_connect(
                    "/api/runs/secret-run/events?after_sequence=0"
                ) as websocket:
                    events = [
                        websocket.receive_json(),
                        websocket.receive_json(),
                        websocket.receive_json(),
                    ]

            durable_history = (
                output_root / "secret-run" / "ui-events.jsonl"
            ).read_text(encoding="utf-8")
            serialized_events = json.dumps(events)
            self.assertEqual(
                [event["message"] for event in events],
                [
                    "Wikimedia evidence: fetch warning.",
                    "Cluster adjudication: review update.",
                    "Trend portfolio: narrate update.",
                ],
            )
            for sensitive_text in (
                "top-secret-token",
                "private instructions",
                "private judgment",
            ):
                self.assertNotIn(sensitive_text, serialized_events)
                self.assertNotIn(sensitive_text, durable_history)

    # Verify: process ownership survives clients and rejects only duplicate runs.
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

    # Verify: retry uses same run id and retains artifacts and event history.
    def test_retry_uses_same_run_id_and_retains_artifacts_and_event_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            invocations = root / "invocations.jsonl"
            fake_cli = root / "resumable_cli.py"
            fake_cli.write_text(
                "import json, pathlib, sys\n"
                f"invocations = pathlib.Path({str(invocations)!r})\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "output_root = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
                "run_directory = output_root / run_id\n"
                "run_directory.mkdir(parents=True, exist_ok=True)\n"
                "artifact = run_directory / 'completed-module.json'\n"
                "operation = 'resume' if artifact.exists() else 'publish'\n"
                "artifact.write_text('completed', encoding='utf-8')\n"
                "with invocations.open('a', encoding='utf-8') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "print(json.dumps({\n"
                "    'schema_version': '1.0', 'run_id': run_id, 'sequence': 1,\n"
                "    'timestamp': '2026-07-17T12:00:00+00:00',\n"
                "    'module': 'wikimedia-evidence', 'operation': operation,\n"
                "    'level': 'info', 'message': operation,\n"
                "}), flush=True)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                for _ in range(2):
                    started = client.post(
                        "/api/runs",
                        json={"run_id": "retry-run", "as_of": "2026-07-17"},
                    )
                    self.assertEqual(started.status_code, 202)
                    self.assertEqual(
                        _wait_for_terminal(client, "retry-run")["status"], "failed"
                    )
                with client.websocket_connect(
                    "/api/runs/retry-run/events?after_sequence=0"
                ) as websocket:
                    events = [websocket.receive_json(), websocket.receive_json()]

            recorded_invocations = [
                json.loads(line)
                for line in invocations.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(recorded_invocations[0], recorded_invocations[1])
            self.assertEqual(
                recorded_invocations[1][
                    recorded_invocations[1].index("--run-id") + 1
                ],
                "retry-run",
            )
            self.assertEqual(
                [event["operation"] for event in events], ["publish", "resume"]
            )
            self.assertEqual([event["sequence"] for event in events], [1, 2])
            self.assertEqual(
                (output_root / "retry-run" / "completed-module.json").read_text(
                    encoding="utf-8"
                ),
                "completed",
            )

    # Verify: confirmed cancellation stops only owned process and keeps artifacts.
    def test_confirmed_cancellation_stops_only_owned_process_and_keeps_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            fake_cli = root / "cancellable_cli.py"
            fake_cli.write_text(
                "import pathlib, sys, time\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "output_root = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
                "run_directory = output_root / run_id\n"
                "run_directory.mkdir(parents=True, exist_ok=True)\n"
                "(run_directory / 'completed-module.json').write_text('keep')\n"
                "time.sleep(3)\n"
                "(run_directory / 'ran-after-cancel.txt').write_text('unsafe')\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(fake_cli)),
            )

            with TestClient(app) as client:
                client.post(
                    "/api/runs",
                    json={"run_id": "cancel-run", "as_of": "2026-07-17"},
                )
                artifact = output_root / "cancel-run" / "completed-module.json"
                deadline = time.monotonic() + 2
                while not artifact.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                unconfirmed = client.post(
                    "/api/runs/cancel-run/cancel", json={"confirmed": False}
                )
                cancelled = client.post(
                    "/api/runs/cancel-run/cancel", json={"confirmed": True}
                )
                terminal = _wait_for_terminal(client, "cancel-run")

            self.assertEqual(unconfirmed.status_code, 400)
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(terminal["status"], "cancelled")
            self.assertEqual(terminal["failure"], None)
            self.assertEqual(artifact.read_text(encoding="utf-8"), "keep")
            time.sleep(0.1)
            self.assertFalse(
                (output_root / "cancel-run" / "ran-after-cancel.txt").exists()
            )

    # Verify: backend restart recovers terminal state and allows resume.
    def test_backend_restart_recovers_terminal_state_and_allows_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            fake_cli = root / "failed_cli.py"
            fake_cli.write_text("raise SystemExit(1)\n", encoding="utf-8")
            first_app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(fake_cli)),
            )
            with TestClient(first_app) as client:
                client.post(
                    "/api/runs",
                    json={"run_id": "restart-run", "as_of": "2026-07-17"},
                )
                original = _wait_for_terminal(client, "restart-run")

            recovered_app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(fake_cli)),
            )
            with TestClient(recovered_app) as client:
                recovered = client.get("/api/runs/restart-run")
                resumed = client.post(
                    "/api/runs",
                    json={"run_id": "restart-run", "as_of": "2026-07-17"},
                )
                retried = _wait_for_terminal(client, "restart-run")

            self.assertEqual(recovered.status_code, 200)
            self.assertEqual(recovered.json(), original)
            self.assertEqual(resumed.status_code, 202)
            self.assertEqual(retried["status"], "failed")

    # Verify: backend restart does not duplicate a surviving cli process.
    def test_backend_restart_does_not_duplicate_a_surviving_cli_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            slow_cli = root / "surviving_cli.py"
            slow_cli.write_text(
                "import time\n"
                "time.sleep(0.5)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            first_app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(slow_cli)),
            )
            with TestClient(first_app) as original_client:
                original_client.post(
                    "/api/runs",
                    json={"run_id": "surviving-run", "as_of": "2026-07-17"},
                )
                recovered_app = create_app(
                    output_root=output_root,
                    cli_command=(sys.executable, str(slow_cli)),
                )
                with TestClient(recovered_app) as recovered_client:
                    recovered = recovered_client.get("/api/runs/surviving-run")
                    duplicate = recovered_client.post(
                        "/api/runs",
                        json={
                            "run_id": "surviving-run",
                            "as_of": "2026-07-17",
                        },
                    )
                    _wait_for_terminal(original_client, "surviving-run")
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        after_exit = recovered_client.get(
                            "/api/runs/surviving-run"
                        ).json()
                        if after_exit["status"] != "running":
                            break
                        time.sleep(0.01)

            self.assertEqual(recovered.status_code, 200)
            self.assertEqual(recovered.json()["status"], "running")
            self.assertEqual(duplicate.status_code, 409)
            self.assertEqual(after_exit["status"], "failed")
            self.assertEqual(after_exit["failure"]["code"], "backend_interrupted")

    # Verify: zero exit without completed publication is failure.
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

    # Verify: cli start failure is retained as sanitized terminal state.
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

    # Verify: success requires zero exit and valid same run publication.
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

    # Verify: success rejects publication for a different as of date.
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
