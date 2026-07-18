from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FIXTURE = Path(__file__).parents[1] / "fixtures" / "v2_contract_stage.json"


# Return invoke stage.
def invoke_stage(output_dir: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "audience_trend_miner",
            "v2-fixture-stage",
            "--run-id",
            "contract-run",
            "--as-of",
            "2026-07-16",
            "--output-dir",
            output_dir,
            "--fixture",
            str(FIXTURE),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


# Group tests for v2 stage cli contract behavior.
class V2StageCliContractTest(unittest.TestCase):
    # Verify: stage publishes a complete schema versioned artifact.
    def test_stage_publishes_a_complete_schema_versioned_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            completed = invoke_stage(output_dir, "--progress-format", "json")

            artifact_path = Path(output_dir) / "contract-run" / "fixture-stage.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            events = [json.loads(line) for line in completed.stdout.splitlines()]

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(artifact["schema_version"], "2.0")
            self.assertEqual(artifact["run_id"], "contract-run")
            self.assertEqual(artifact["stage"], "fixture-stage")
            self.assertEqual(artifact["status"], "complete")
            self.assertEqual(artifact["payload"], {"candidate_count": 2})
            self.assertEqual([event["sequence"] for event in events], [1, 2])
            self.assertEqual(events[0]["progress"], {"current": 1, "total": 2})
            self.assertEqual(events[1]["progress"], {"current": 2, "total": 2})
            self.assertTrue(all(event["schema_version"] == "1.0" for event in events))

    # Verify: human progress uses the same bounded events.
    def test_human_progress_uses_the_same_bounded_events(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            completed = invoke_stage(output_dir)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                completed.stdout.splitlines(),
                [
                    "[fixture-stage:load] loading fixture (1/2)",
                    "[fixture-stage:publish] published artifact (2/2)",
                ],
            )

    # Verify: stable run id rejects incompatible configuration.
    def test_stable_run_id_rejects_incompatible_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            first = invoke_stage(output_dir)
            conflicting = invoke_stage(output_dir, "--as-of", "2026-07-17")

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotEqual(conflicting.returncode, 0)
            self.assertIn("run configuration conflicts", conflicting.stderr)

    # Verify: incomplete artifact is not consumable.
    def test_incomplete_artifact_is_not_consumable(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            run_directory = Path(output_dir) / "contract-run"
            run_directory.mkdir(parents=True)
            (run_directory / "fixture-stage.json").write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "run_id": "contract-run",
                        "stage": "fixture-stage",
                        "status": "writing",
                        "payload": {},
                    }
                ),
                encoding="utf-8",
            )

            completed = invoke_stage(output_dir, "--consume-existing")

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("artifact is incomplete", completed.stderr)

    # Verify: interrupted publication does not expose an artifact.
    def test_interrupted_publication_does_not_expose_an_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            completed = invoke_stage(output_dir, "--interrupt-before-completion")

            artifact_path = Path(output_dir) / "contract-run" / "fixture-stage.json"
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(artifact_path.exists())
            self.assertEqual(
                list((Path(output_dir) / "contract-run").glob(".fixture-stage.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
