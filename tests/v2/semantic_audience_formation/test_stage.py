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


# Run Semantic Audience Formation through its CLI boundary.
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


# Publish wikimedia evidence.
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


# Publish clustering contract evidence.
def publish_clustering_contract_evidence(output_dir: Path) -> None:
    artifact_path = publish_wikimedia_evidence(output_dir)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    page_facts = (
        (7, "Gamma Two", "Gamma"),
        (2, "Alpha Two", "Alpha"),
        (10, "Singleton", "Isolated"),
        (4, "Beta One", "Beta"),
        (9, "Boundary Two", "Boundary"),
        (1, "Alpha One", "Alpha"),
        (6, "Gamma One", "Gamma"),
        (5, "Beta Two", "Beta"),
        (8, "Boundary One", "Boundary"),
        (3, "Alpha Three", "Alpha"),
    )
    artifact["payload"]["canonical_pages"] = [
        {
            "page_id": page_id,
            "canonical_title": title,
            "lead": f"{title.lower()} lead.",
            "categories": [category],
            "aliases": [title],
            "observations": [],
        }
        for page_id, title, category in page_facts
    ]
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")


# Group tests for semantic audience formation stage behavior.
class SemanticAudienceFormationStageTest(unittest.TestCase):
    # Verify: stage resumes a compatible completed artifact without embedding.
    def test_stage_resumes_a_compatible_completed_artifact_without_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            publish_wikimedia_evidence(output_dir)
            fake_package_root = output_dir / "fake-package"
            fake_package = fake_package_root / "sentence_transformers"
            fake_package.mkdir(parents=True)
            embedding_log = output_dir / "embedding-calls.jsonl"
            fake_package.joinpath("__init__.py").write_text(
                """\
import json
import os
from pathlib import Path

class SentenceTransformer:
    def __init__(self, model):
        self.model = model

    def encode(self, representations, *, batch_size, convert_to_numpy):
        with Path(os.environ["TEST_EMBEDDING_LOG"]).open("a", encoding="utf-8") as log:
            log.write(json.dumps({"count": len(representations)}) + "\\n")
        return [[1.0, 0.0] for _representation in representations]
""",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["TEST_EMBEDDING_LOG"] = str(embedding_log)
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(fake_package_root), environment.get("PYTHONPATH")))
            )

            first = run_stage(output_dir, environment=environment)
            second = run_stage(output_dir, environment=environment)
            evidence_path = output_dir / "formation-run" / "wikimedia-evidence.json"
            changed_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            changed_evidence["payload"]["canonical_pages"][0]["lead"] = "Changed lead."
            evidence_path.write_text(json.dumps(changed_evidence), encoding="utf-8")
            incompatible = run_stage(output_dir, environment=environment)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertNotEqual(incompatible.returncode, 0)
            self.assertIn(
                "conflicts with requested configuration",
                incompatible.stderr,
            )
            self.assertEqual(len(embedding_log.read_text().splitlines()), 2)
            resumed_events = [
                json.loads(line) for line in second.stdout.splitlines()
            ]
            self.assertEqual(
                [event["operation"] for event in resumed_events],
                ["resume"],
            )
            self.assertEqual(resumed_events[0]["progress"], {"current": 1, "total": 1})

    # Verify: review cap defaults to ten accepts all and rejects invalid values.
    def test_review_cap_defaults_to_ten_accepts_all_and_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            default_output = root / "default"
            publish_wikimedia_evidence(default_output)
            completed = run_stage(default_output, embedding_fixture=EMBEDDING_FIXTURE)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            default_payload = json.loads(
                (default_output / "formation-run" / "semantic-audience-formation.json").read_text()
            )["payload"]
            self.assertEqual(default_payload["configuration"]["review_cap"], 10)

            all_output = root / "all"
            publish_wikimedia_evidence(all_output)
            environment = os.environ.copy()
            environment["AUDIENCE_TREND_MINER_MAX_LLM_CLUSTERS"] = "all"
            completed = run_stage(
                all_output,
                embedding_fixture=EMBEDDING_FIXTURE,
                environment=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            all_payload = json.loads(
                (all_output / "formation-run" / "semantic-audience-formation.json").read_text()
            )["payload"]
            self.assertEqual(all_payload["configuration"]["review_cap"], "all")

            for index, invalid_value in enumerate(("", "0", "-1", "1.5", "many")):
                with self.subTest(invalid_value=invalid_value):
                    invalid_output = root / f"invalid-{index}"
                    publish_wikimedia_evidence(invalid_output)
                    completed = run_stage(
                        invalid_output,
                        embedding_fixture=EMBEDDING_FIXTURE,
                extra_arguments=("--review-cap", invalid_value),
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(
                        "review cap must be a positive integer or 'all'",
                        completed.stderr,
                    )
                    self.assertFalse(
                        (
                            invalid_output
                            / "formation-run"
                            / "semantic-audience-formation.json"
                        ).exists()
                    )

    # Verify: stage publishes ranked capped minimal cluster evidence.
    def test_stage_publishes_ranked_capped_minimal_cluster_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            publish_clustering_contract_evidence(output_dir)

            completed = run_stage(
                output_dir,
                embedding_fixture=EMBEDDING_FIXTURE,
                extra_arguments=("--review-cap", "2"),
            )

            artifact_path = output_dir / "formation-run" / "semantic-audience-formation.json"
            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            payload = artifact["payload"]
            self.assertEqual(artifact["status"], "complete")
            self.assertEqual(payload["configuration"]["review_cap"], 2)
            self.assertEqual(
                payload["counts"],
                {
                    "eligible_clusters": 4,
                    "selected_clusters": 2,
                    "omitted_clusters": 2,
                    "discarded_singleton_components": 1,
                    "subdivided_components": 0,
                    "subdivisions_created": 0,
                    "singleton_subdivisions": 0,
                },
            )
            self.assertEqual(
                [
                    [member["page_id"] for member in cluster["members"]]
                    for cluster in payload["preliminary_clusters"]
                ],
                [[1, 2, 3], [4, 5]],
            )
            self.assertEqual(
                set(payload["preliminary_clusters"][0]["members"][0]),
                {"page_id", "canonical_title", "lead", "selected_categories"},
            )
            self.assertIsNone(payload["preliminary_clusters"][0]["subdivision"])
            serialized = json.dumps(artifact).lower()
            self.assertNotIn("traffic", serialized)
            self.assertNotIn("observation", serialized)
            self.assertNotIn("embedding_vectors", serialized)
            self.assertNotIn("similarity_matrix", serialized)

    # Verify: stage runs production embeddings with safe configuration progress.
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
                [
                    "select-categories",
                    "embed-representations",
                    "form-components",
                    "subdivide-components",
                    "rank-clusters",
                    "select-review-cap",
                    "publish",
                ],
            )
            self.assertIn("using model 'local/override-model'", events[1]["message"])
            self.assertIn("threshold 0.76", events[2]["message"])
            self.assertIn("16384-token stricter-boundary guard", events[3]["message"])
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

    # Verify: stage forms fixture backed preliminary clusters.
    def test_stage_forms_fixture_backed_preliminary_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            publish_wikimedia_evidence(output_dir)

            completed = run_stage(output_dir, embedding_fixture=EMBEDDING_FIXTURE)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual(
                [event["operation"] for event in events],
                [
                    "select-categories",
                    "embed-representations",
                    "form-components",
                    "subdivide-components",
                    "rank-clusters",
                    "select-review-cap",
                    "publish",
                ],
            )
            self.assertEqual(
                [event["progress"] for event in events],
                [
                    {"current": current, "total": 7}
                    for current in range(1, 8)
                ],
            )
            self.assertIn("formed 1 connected components", events[2]["message"])
            self.assertIn("discarded 0 singleton components", events[2]["message"])

    # Verify: stage consumes completed wikimedia evidence without reacquisition.
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

    # Verify: stage rejects absent incomplete incompatible and mismatched evidence.
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
