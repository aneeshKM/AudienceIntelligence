from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WIKIMEDIA_FIXTURE = Path(__file__).parents[1] / "fixtures" / "v2_wikimedia_evidence.json"
EMBEDDING_FIXTURE = (
    Path(__file__).with_name("fixtures") / "preliminary_cluster_embeddings.json"
)


def run_stage(
    output_dir: Path,
    run_id: str = "formation-run",
    *,
    embedding_fixture: Path | None = None,
    extra_arguments: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    arguments = [
        sys.executable,
        "-m",
        "audience_trend_miner",
        "v2-semantic-audience-formation",
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--progress-format",
        "json",
    ]
    if embedding_fixture is not None:
        arguments.extend(
            [
                "--embedding-fixture",
                str(embedding_fixture),
                "--similarity-threshold",
                "0.3",
            ]
        )
    arguments.extend(extra_arguments)
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def publish_wikimedia_evidence(output_dir: Path, run_id: str = "formation-run") -> Path:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "audience_trend_miner",
            "v2-wikimedia-evidence",
            "--run-id",
            run_id,
            "--as-of",
            "2026-07-17",
            "--output-dir",
            str(output_dir),
            "--fixture",
            str(WIKIMEDIA_FIXTURE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return output_dir / run_id / "wikimedia-evidence.json"


class SemanticAudienceFormationStageTest(unittest.TestCase):
    def test_stage_runs_production_embeddings_with_safe_configuration_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            publish_wikimedia_evidence(output_dir)
            fake_package_root = output_dir / "fake-package"
            fake_package = fake_package_root / "sentence_transformers"
            fake_package.mkdir(parents=True)
            configuration_log = output_dir / "encoder-configuration.jsonl"
            fake_package.joinpath("__init__.py").write_text(
                """\
import json
import os
from pathlib import Path

class SentenceTransformer:
    def __init__(self, model):
        self.model = model

    def encode(self, representations, *, batch_size, convert_to_numpy):
        record = {
            \"model\": self.model,
            \"batch_size\": batch_size,
            \"count\": len(representations),
        }
        with Path(os.environ[\"TEST_ENCODER_LOG\"]).open(
            \"a\", encoding=\"utf-8\"
        ) as log:
            log.write(json.dumps(record) + \"\\n\")
        return [[1.0, 0.0] for _representation in representations]
""",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["TEST_ENCODER_LOG"] = str(configuration_log)
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(fake_package_root), environment.get("PYTHONPATH")))
            )

            completed = run_stage(
                output_dir,
                extra_arguments=(
                    "--embedding-model",
                    "local/override-model",
                    "--embedding-batch-size",
                    "1",
                ),
                environment=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual(
                [event["operation"] for event in events],
                ["select-categories", "form-preliminary-clusters"],
            )
            self.assertIn("using model 'local/override-model'", events[1]["message"])
            self.assertIn("threshold 0.76", events[1]["message"])
            self.assertIn("16384-token stricter-boundary guard", events[1]["message"])
            configurations = [
                json.loads(line) for line in configuration_log.read_text().splitlines()
            ]
            self.assertEqual(
                configurations,
                [
                    {"model": "local/override-model", "batch_size": 1, "count": 2},
                    {"model": "local/override-model", "batch_size": 1, "count": 2},
                ],
            )
            self.assertNotIn("embedding_vectors", completed.stdout)
            self.assertNotIn("similarity_matrix", completed.stdout)

    def test_stage_forms_fixture_backed_preliminary_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            publish_wikimedia_evidence(output_dir)

            completed = run_stage(output_dir, embedding_fixture=EMBEDDING_FIXTURE)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual(
                [event["operation"] for event in events],
                ["select-categories", "form-preliminary-clusters"],
            )
            self.assertIn("formed 1 Preliminary Clusters", events[1]["message"])
            self.assertIn("discarded 0 singleton components", events[1]["message"])

    def test_stage_consumes_completed_wikimedia_evidence_without_reacquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            publish_wikimedia_evidence(output_dir)

            completed = run_stage(
                output_dir,
                extra_arguments=("--category-selection-only",),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["module"], "semantic-audience-formation")
            self.assertEqual(events[0]["operation"], "select-categories")
            self.assertIn("rule set 1.0", events[0]["message"])
            self.assertEqual(events[0]["progress"], {"current": 2, "total": 2})

            artifact = json.loads(
                (output_dir / "formation-run" / "wikimedia-evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                artifact["payload"]["provenance"]["category_visibility"],
                "non-hidden",
            )

    def test_stage_rejects_absent_incomplete_incompatible_and_mismatched_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            artifact_path = publish_wikimedia_evidence(output_dir)
            valid_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            scenarios = (
                ("absent", None, "artifact is absent"),
                ("incomplete", {**valid_artifact, "status": "writing"}, "artifact is incomplete"),
                ("incompatible", {**valid_artifact, "schema_version": "1.0"}, "schema-invalid"),
                (
                    "missing hidden-category guarantee",
                    {
                        **valid_artifact,
                        "payload": {
                            **valid_artifact["payload"],
                            "provenance": {
                                key: value
                                for key, value in valid_artifact["payload"][
                                    "provenance"
                                ].items()
                                if key != "category_visibility"
                            },
                        },
                    },
                    "schema-incompatible",
                ),
                ("mismatched", {**valid_artifact, "run_id": "another-run"}, "different run facts"),
            )

            for name, artifact, expected_error in scenarios:
                with self.subTest(name=name):
                    artifact_path.unlink(missing_ok=True)
                    if artifact is not None:
                        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

                    completed = run_stage(output_dir)

                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(expected_error, completed.stderr)


if __name__ == "__main__":
    unittest.main()
