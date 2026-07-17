from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import tempfile
from threading import Thread
import time
import unittest

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - exercised only without browser test extras
    sync_playwright = None  # type: ignore[assignment]

import uvicorn

from audience_trend_miner.v2.ui import create_app
from tests.v2.ui.test_server import _write_completed_publication


FIXTURES = Path(__file__).parents[1] / "run_publication" / "fixtures"
FORMATION_FIXTURES = (
    Path(__file__).parents[1]
    / "semantic_audience_formation"
    / "fixtures"
)


def _unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class _LiveServer:
    def __init__(self, app) -> None:
        self.port = _unused_loopback_port()
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_level="error",
            )
        )
        self.thread = Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self.server.started:
            raise AssertionError("test server did not start")
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *_args) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


@unittest.skipIf(sync_playwright is None, "install the browser test extra")
class BrowserWorkflowTest(unittest.TestCase):
    def test_failure_disconnect_resume_and_empty_publication_render(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            wrapper = root / "interrupt_once.py"
            wrapper.write_text(
                "import json, pathlib, subprocess, sys, time\n"
                "arguments = sys.argv[1:]\n"
                "run_id = arguments[arguments.index('--run-id') + 1]\n"
                "output_root = pathlib.Path(arguments[arguments.index('--output-dir') + 1])\n"
                "run_directory = output_root / run_id\n"
                "run_directory.mkdir(parents=True, exist_ok=True)\n"
                "attempt_path = run_directory / 'fixture-attempt.txt'\n"
                "attempt = int(attempt_path.read_text() or '0') + 1 if attempt_path.exists() else 1\n"
                "attempt_path.write_text(str(attempt))\n"
                "command = [sys.executable, '-m', 'audience_trend_miner', *arguments]\n"
                "if attempt > 1:\n"
                "    raise SystemExit(subprocess.run(command).returncode)\n"
                "process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)\n"
                "assert process.stdout is not None\n"
                "for line in process.stdout:\n"
                "    print(line, end='', flush=True)\n"
                "    event = json.loads(line)\n"
                "    time.sleep(0.08)\n"
                "    if event['module'] == 'semantic-audience-formation':\n"
                "        process.terminate()\n"
                "        process.wait()\n"
                "        raise SystemExit(1)\n"
                "raise SystemExit(process.wait())\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(wrapper)),
                cli_arguments=(
                    "--wikimedia-fixture",
                    str(FIXTURES / "global_wikimedia_evidence.json"),
                    "--embedding-fixture",
                    str(FORMATION_FIXTURES / "preliminary_cluster_embeddings.json"),
                    "--cluster-fixture",
                    str(FIXTURES / "global_cluster_decisions.json"),
                    "--narrative-fixture",
                    str(FIXTURES / "global_narratives.json"),
                    "--similarity-threshold",
                    "0.3",
                ),
            )

            with _LiveServer(app) as base_url, sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(base_url)
                page.get_by_label("As-of Date").fill("2026-07-17")
                page.get_by_label("Run ID").fill("browser-resume")
                page.get_by_label("As-of Date").press("Enter")
                page.locator(".event-item").first.wait_for()
                last_sequence_before_disconnect = page.evaluate("lastSequence")
                page.evaluate("streamWanted = false; socket.close()")
                page.wait_for_function("socket === null")
                history_path = output_root / "browser-resume" / "ui-events.jsonl"
                deadline = time.monotonic() + 5
                gap_event = None
                while time.monotonic() < deadline:
                    if history_path.is_file():
                        history_during_disconnect = [
                            json.loads(line)
                            for line in history_path.read_text(
                                encoding="utf-8"
                            ).splitlines()
                        ]
                        gap_event = next(
                            (
                                event
                                for event in history_during_disconnect
                                if event["sequence"] > last_sequence_before_disconnect
                            ),
                            None,
                        )
                        if gap_event is not None:
                            break
                    time.sleep(0.01)
                self.assertIsNotNone(gap_event)
                assert gap_event is not None
                self.assertEqual(page.evaluate("lastSequence"), last_sequence_before_disconnect)
                page.evaluate("connectEventStream(activeRun.id)")
                page.wait_for_function(
                    "document.querySelector('#run-status').textContent.includes('failed')",
                    timeout=10_000,
                )
                replayed = page.locator(
                    f'.event-item[data-sequence="{gap_event["sequence"]}"]'
                )
                replayed.wait_for(timeout=5_000)
                self.assertEqual(
                    replayed.locator(".event-message").text_content(),
                    gap_event["message"],
                )
                first_attempt_count = page.locator(".event-item").count()

                page.get_by_role("button", name="Retry or resume run").click()
                page.get_by_text(
                    "No robust audience trends qualified for this run.", exact=True
                ).wait_for(timeout=20_000)

                operations = page.locator(".event-operation").all_text_contents()
                modules = page.locator(".module-heading h3").all_text_contents()
                self.assertGreater(page.locator(".event-item").count(), first_attempt_count)
                self.assertIn("resume", operations)
                self.assertTrue(
                    {
                        "wikimedia evidence",
                        "semantic audience formation",
                        "cluster adjudication",
                        "trend portfolio",
                        "run publication",
                    }.issubset(set(modules)),
                    modules,
                )
                self.assertTrue(page.locator("#portfolio").is_visible())
                self.assertTrue(page.get_by_text("Evidence limitation").is_visible())
                browser.close()

            history = [
                json.loads(line)
                for line in (
                    output_root / "browser-resume" / "ui-events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["sequence"] for event in history],
                list(range(1, len(history) + 1)),
            )
            self.assertEqual(
                (output_root / "browser-resume" / "fixture-attempt.txt").read_text(),
                "2",
            )

    def test_mixed_directions_are_named_and_not_conveyed_only_by_color(self) -> None:
        def audience(cluster_id: str, direction: str, change: float):
            return {
                "cluster_id": cluster_id,
                "source_preliminary_cluster_id": f"preliminary-{cluster_id}",
                "direction": direction,
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
                "percentage_change": change,
                "coverage": {"previous": 1.0, "current": 0.86},
                "confidence": "robust",
                "impact_score": 3.5,
                "narrative": {
                    "name": f"{cluster_id} audience",
                    "summary": "A fixture-backed attention trend.",
                    "commercial_interpretation": "Planning interest is changing.",
                    "brand_categories": ["Home planning"],
                    "buying_power_rating": "medium",
                    "buying_power_rationale": "Purchases span several price points.",
                },
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            _write_completed_publication(
                output_root,
                "mixed-run",
                audiences=[
                    audience("growing", "robust_growth", 25.0),
                    audience("shrinking", "robust_shrinking", -20.0),
                ],
            )
            successful_cli = root / "successful_cli.py"
            successful_cli.write_text("raise SystemExit(0)\n", encoding="utf-8")
            app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(successful_cli)),
            )

            with _LiveServer(app) as base_url, sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(base_url)
                page.get_by_label("As-of Date").fill("2026-07-17")
                page.get_by_label("Run ID").fill("mixed-run")
                page.get_by_role("button", name="Start or resume run").click()
                page.get_by_text("↗ Growing", exact=True).wait_for(timeout=10_000)
                self.assertTrue(page.get_by_text("↘ Shrinking", exact=True).is_visible())
                self.assertTrue(
                    page.get_by_role("status").get_attribute("aria-live") == "polite"
                )
                browser.close()

    def test_cancellation_requires_confirmation_and_keeps_completed_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "runs"
            cancellable_cli = root / "cancellable_cli.py"
            cancellable_cli.write_text(
                "import pathlib, sys, time\n"
                "run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
                "root = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
                "run_directory = root / run_id\n"
                "run_directory.mkdir(parents=True, exist_ok=True)\n"
                "(run_directory / 'completed-module.json').write_text('keep')\n"
                "time.sleep(20)\n",
                encoding="utf-8",
            )
            app = create_app(
                output_root=output_root,
                cli_command=(sys.executable, str(cancellable_cli)),
            )

            with _LiveServer(app) as base_url, sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(base_url)
                page.get_by_label("As-of Date").fill("2026-07-17")
                page.get_by_label("Run ID").fill("browser-cancel")
                page.get_by_role("button", name="Start or resume run").click()
                cancel = page.get_by_role("button", name="Cancel run")
                cancel.wait_for()
                artifact = output_root / "browser-cancel" / "completed-module.json"
                deadline = time.monotonic() + 5
                while not artifact.is_file() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(
                    artifact.is_file(),
                    (
                        page.locator("#run-status").text_content(),
                        [str(path.relative_to(root)) for path in root.rglob("*")],
                    ),
                )
                dialogs: list[str] = []

                def confirm(dialog) -> None:
                    dialogs.append(dialog.message)
                    dialog.accept()

                page.once("dialog", confirm)
                cancel.click()
                page.wait_for_function(
                    "document.querySelector('#run-status').textContent.includes('cancelled')",
                    timeout=10_000,
                )
                self.assertIn("Completed artifacts", dialogs[0])
                self.assertTrue(artifact.is_file())
                browser.close()


if __name__ == "__main__":
    unittest.main()
