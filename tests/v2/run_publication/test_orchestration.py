from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


FIXTURES = Path(__file__).with_name("fixtures")
FORMATION_FIXTURES = (
    Path(__file__).parents[1]
    / "semantic_audience_formation"
    / "fixtures"
)


# Return global cli arguments.
def global_cli_arguments(
    output_root: Path,
    *,
    run_id: str = "global-run",
    extra: tuple[str, ...] = (),
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "audience_trend_miner",
        "--run-id",
        run_id,
        "--as-of",
        "2026-07-17",
        "--output-dir",
        str(output_root),
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
        "--progress-format",
        "json",
        *extra,
    ]


# Run global cli.
def run_global_cli(
    output_root: Path,
    *,
    run_id: str = "global-run",
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        global_cli_arguments(output_root, run_id=run_id, extra=extra),
        check=False,
        capture_output=True,
        text=True,
    )


# Return rewritten fixture.
def _rewritten_fixture(
    root: Path,
    source: Path,
    name: str,
    update,
) -> Path:
    fixture = json.loads(source.read_text(encoding="utf-8"))
    update(fixture)
    path = root / name
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path


# Group tests for global orchestration behavior.
class GlobalOrchestrationTest(unittest.TestCase):
    # Verify: global cli orders all modules and succeeds only after publication.
    def test_global_cli_orders_all_modules_and_succeeds_only_after_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)

            completed = run_global_cli(output_root)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            modules = [event["module"] for event in events]
            first_positions = {
                module: modules.index(module)
                for module in (
                    "wikimedia-evidence",
                    "semantic-audience-formation",
                    "cluster-adjudication",
                    "trend-portfolio",
                    "run-publication",
                )
            }
            self.assertEqual(
                list(first_positions),
                sorted(first_positions, key=lambda module: first_positions[module]),
            )
            self.assertEqual(
                [event["sequence"] for event in events],
                list(range(1, len(events) + 1)),
            )
            self.assertTrue(
                all(event["schema_version"] == "1.0" for event in events)
            )
            publication = output_root / "global-run" / "publication"
            self.assertEqual(
                {path.name for path in publication.iterdir()},
                {"portfolio.json", "audit.json", "manifest.json"},
            )

    # Verify: json events are flushed while the process is running.
    def test_json_events_are_flushed_while_the_process_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            process = subprocess.Popen(
                global_cli_arguments(Path(temporary_directory)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None

            first_line = process.stdout.readline()

            self.assertTrue(first_line)
            self.assertIsNone(process.poll(), "first event was buffered until process exit")
            first_event = json.loads(first_line)
            self.assertEqual(first_event["module"], "wikimedia-evidence")
            remaining_stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertTrue(remaining_stdout)

    # Verify: resume from every module boundary reuses completed work.
    def test_resume_from_every_module_boundary_reuses_completed_work(self) -> None:
        stage_names = (
            "wikimedia-evidence",
            "semantic-audience-formation",
            "cluster-adjudication",
            "trend-portfolio",
            "run-publication",
        )
        for first_incomplete in range(1, len(stage_names) + 1):
            with self.subTest(first_incomplete=first_incomplete):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    output_root = Path(temporary_directory)
                    initial = run_global_cli(output_root)
                    self.assertEqual(initial.returncode, 0, initial.stderr)
                    run_directory = output_root / "global-run"
                    for stage in stage_names[first_incomplete:4]:
                        (run_directory / f"{stage}.json").unlink()
                    if first_incomplete <= 4:
                        publication = run_directory / "publication"
                        publication.unlink()
                        for completed_directory in run_directory.glob(
                            ".publication.*.complete"
                        ):
                            shutil.rmtree(completed_directory)

                    resumed = run_global_cli(output_root)

                    self.assertEqual(resumed.returncode, 0, resumed.stderr)
                    events = [
                        json.loads(line) for line in resumed.stdout.splitlines()
                    ]
                    for completed_stage in stage_names[:first_incomplete]:
                        operations = [
                            event["operation"]
                            for event in events
                            if event["module"] == completed_stage
                        ]
                        self.assertEqual(
                            operations,
                            ["resume"],
                            f"{completed_stage} repeated work: {operations}",
                        )

    # Verify: resume rejects configuration drift at module boundaries.
    def test_resume_rejects_configuration_drift_at_module_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            initial = run_global_cli(output_root)
            self.assertEqual(initial.returncode, 0, initial.stderr)
            changed_cluster = _rewritten_fixture(
                output_root,
                FIXTURES / "global_cluster_decisions.json",
                "changed-cluster.json",
                lambda fixture: fixture["clusters"][0]["responses"]["proposer"][0][
                    "groups"
                ][0].update(rationale="Changed fixture response."),
            )
            changed_narrative = _rewritten_fixture(
                output_root,
                FIXTURES / "global_narratives.json",
                "changed-narrative.json",
                lambda fixture: fixture["clusters"].append(
                    {"cluster_id": "unused", "responses": []}
                ),
            )
            changed_embedding = _rewritten_fixture(
                output_root,
                FORMATION_FIXTURES / "preliminary_cluster_embeddings.json",
                "changed-embedding.json",
                lambda fixture: fixture["embeddings"].update(
                    {"Selected Categories: Other": [0, 1]}
                ),
            )
            cases = (
                ("wikimedia-evidence", ("--as-of", "2026-07-18")),
                (
                    "semantic-audience-formation",
                    ("--embedding-fixture", str(changed_embedding)),
                ),
                ("cluster-adjudication", ("--cluster-fixture", str(changed_cluster))),
                ("trend-portfolio", ("--narrative-fixture", str(changed_narrative))),
            )
            for module, extra in cases:
                with self.subTest(module=module):
                    resumed = run_global_cli(output_root, extra=extra)
                    self.assertNotEqual(resumed.returncode, 0)
                    self.assertIn("conflict", resumed.stderr)
                    self.assertNotIn("Traceback", resumed.stderr)

    # Verify: module failure stops downstream work and returns failure status.
    def test_module_failure_stops_downstream_work_and_returns_failure_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            failing_fixture = output_root / "failing-cluster.json"
            failing_fixture.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "model": "fixture/failing-cluster-model",
                        "clusters": [],
                    }
                ),
                encoding="utf-8",
            )

            failed = run_global_cli(
                output_root,
                extra=("--cluster-fixture", str(failing_fixture)),
            )

            self.assertEqual(failed.returncode, 1)
            self.assertIn("error:", failed.stderr)
            self.assertNotIn("Traceback", failed.stderr)
            modules = {
                json.loads(line)["module"] for line in failed.stdout.splitlines()
            }
            self.assertEqual(
                modules,
                {
                    "wikimedia-evidence",
                    "semantic-audience-formation",
                    "cluster-adjudication",
                },
            )
            failure_event = json.loads(failed.stdout.splitlines()[-1])
            self.assertEqual(failure_event["operation"], "failed")
            self.assertEqual(failure_event["level"], "error")
            self.assertFalse((output_root / "global-run" / "publication").exists())

    # Verify: invalid fixture returns a clean failure status.
    def test_invalid_fixture_returns_a_clean_failure_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            invalid_fixture = output_root / "invalid.json"
            invalid_fixture.write_text("{}", encoding="utf-8")

            failed = run_global_cli(
                output_root,
                extra=("--narrative-fixture", str(invalid_fixture)),
            )

            self.assertEqual(failed.returncode, 1)
            self.assertIn("error: narrative fixture", failed.stderr)
            self.assertNotIn("Traceback", failed.stderr)

    # Verify: global resume rejects an incompatible completed publication.
    def test_global_resume_rejects_an_incompatible_completed_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            initial = run_global_cli(output_root)
            self.assertEqual(initial.returncode, 0, initial.stderr)
            manifest_path = output_root / "global-run" / "publication" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run_id"] = "different-run"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            failed = run_global_cli(output_root)

            self.assertEqual(failed.returncode, 1)
            self.assertIn("collides with requested run", failed.stderr)
            failed_events = [
                json.loads(line) for line in failed.stdout.splitlines()
            ]
            self.assertEqual(failed_events[-1]["module"], "run-publication")
            self.assertEqual(failed_events[-1]["operation"], "failed")
            for module in (
                "wikimedia-evidence",
                "semantic-audience-formation",
                "cluster-adjudication",
                "trend-portfolio",
            ):
                self.assertEqual(
                    [
                        event["operation"]
                        for event in failed_events
                        if event["module"] == module
                    ],
                    ["resume"],
                )


if __name__ == "__main__":
    unittest.main()
