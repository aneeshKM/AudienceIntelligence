from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path


def fixture_payload() -> dict[str, object]:
    return {
        "discovery": {
            (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): ["Alias_A"]
            for offset in range(7)
        },
        "pageviews": {
            "Alias_A": [
                {
                    "date": (date(2026, 7, 1) + timedelta(days=offset)).isoformat(),
                    "views": 5 if offset < 7 else 10,
                }
                for offset in range(14)
            ]
        },
        "metadata": {
            "Alias_A": {
                "page_id": 42,
                "canonical_title": "Canonical A",
                "extract": "A useful lead.",
                "categories": ["Examples"],
            }
        },
    }


def classification_fixture(*responses: object) -> dict[str, object]:
    return {"responses": list(responses)}


def write_embedding_fixture(path: Path, embeddings: list[list[float]]) -> None:
    path.write_text(
        json.dumps(
            {
                "model": "sentence-transformers/all-mpnet-base-v2",
                "embeddings": embeddings,
            }
        )
    )


class CliWikimediaTest(unittest.TestCase):
    def test_cli_fixtures_cover_all_rejections_retry_recovery_and_exhaustion(self) -> None:
        rejection_classes = (
            "tragedy",
            "violent_crime",
            "death_driven",
            "routine_politics",
            "isolated_news",
            "no_consumer_audience",
        )
        aliases = [f"Alias_{page_id}" for page_id in range(1, 9)]
        payload = {
            "discovery": {
                (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): aliases
                for offset in range(7)
            },
            "pageviews": {
                alias: [
                    {
                        "date": (date(2026, 7, 1) + timedelta(days=offset)).isoformat(),
                        "views": 10_000 if offset < 7 else 20_000,
                    }
                    for offset in range(14)
                ]
                for alias in aliases
            },
            "metadata": {
                alias: {
                    "page_id": page_id,
                    "canonical_title": f"Article {page_id}",
                    "extract": "Fixture lead.",
                    "categories": ["Fixture"],
                }
                for page_id, alias in enumerate(aliases, start=1)
            },
        }
        responses: list[object] = [
            {
                "supports_consumer_audience": False,
                "brand_safe": False,
                "rejection_class": rejection_class,
                "rationale": "Rejected fixture.",
            }
            for rejection_class in rejection_classes
        ]
        responses.extend(
            [
                {"brand_safe": True},
                {
                    "supports_consumer_audience": True,
                    "brand_safe": True,
                    "rejection_class": "accepted",
                    "rationale": "Recovered fixture.",
                },
                {"brand_safe": True},
                {"supports_consumer_audience": True},
                {"unexpected": True},
            ]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            wikimedia_path = temporary_path / "wikimedia.json"
            wikimedia_path.write_text(json.dumps(payload))
            classification_path = temporary_path / "classification.json"
            classification_path.write_text(json.dumps({"responses": responses}))
            embedding_path = temporary_path / "embeddings.json"
            write_embedding_fixture(embedding_path, [[1.0, 0.0]])
            output_directory = temporary_path / "runs"
            environment = os.environ.copy()
            environment["DATABASE_URL"] = "postgresql://postgres:test@localhost:55432/audience_intelligence_test"
            environment["AUDIENCE_TREND_MINER_TEST_MODE"] = "1"
            environment["AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE"] = str(
                wikimedia_path
            )
            environment["AUDIENCE_TREND_MINER_CLASSIFICATION_FIXTURE"] = str(
                classification_path
            )
            environment["AUDIENCE_TREND_MINER_EMBEDDING_FIXTURE"] = str(
                embedding_path
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "--as-of",
                    "2026-07-16",
                    "--output-dir",
                    str(output_directory),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            audit = json.loads(
                (next(output_directory.iterdir()) / "audit.json").read_text()
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        decisions = audit["article_classifications"]
        self.assertEqual(
            [item["decision_reason"] for item in decisions[:6]],
            list(rejection_classes),
        )
        self.assertEqual(len(decisions[6]["attempts"]), 2)
        self.assertTrue(decisions[6]["accepted"])
        self.assertEqual(len(decisions[7]["attempts"]), 3)
        self.assertEqual(decisions[7]["decision_reason"], "exhausted_attempts")
        self.assertEqual(
            [item["canonical_title"] for item in audit["qualified_signals"]],
            ["Article 7"],
        )

    def test_cli_classifies_qualified_article_and_publishes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            payload = fixture_payload()
            for observation in payload["pageviews"]["Alias_A"]:
                observation["views"] = (
                    10_000 if observation["date"] < "2026-07-08" else 20_000
                )
            wikimedia_path = temporary_path / "wikimedia.json"
            wikimedia_path.write_text(json.dumps(payload))
            classification_path = temporary_path / "classification.json"
            classification_path.write_text(
                json.dumps(
                    classification_fixture(
                        {
                            "supports_consumer_audience": True,
                            "brand_safe": True,
                            "rejection_class": "accepted",
                            "rationale": "Commercial fixture.",
                        }
                    )
                )
            )
            embedding_path = temporary_path / "embeddings.json"
            write_embedding_fixture(embedding_path, [[1.0, 0.0]])
            output_directory = temporary_path / "runs"
            environment = os.environ.copy()
            environment["DATABASE_URL"] = "postgresql://postgres:test@localhost:55432/audience_intelligence_test"
            environment["AUDIENCE_TREND_MINER_TEST_MODE"] = "1"
            environment["AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE"] = str(
                wikimedia_path
            )
            environment["AUDIENCE_TREND_MINER_CLASSIFICATION_FIXTURE"] = str(
                classification_path
            )
            environment["AUDIENCE_TREND_MINER_EMBEDDING_FIXTURE"] = str(
                embedding_path
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "--as-of",
                    "2026-07-16",
                    "--output-dir",
                    str(output_directory),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            run_directory = next(output_directory.iterdir())
            audit = json.loads((run_directory / "audit.json").read_text())
            evidence_exists = (
                run_directory / "classification" / "article_judgments.json"
            ).is_file()
            clustering = json.loads(
                (run_directory / "clustering" / "candidate_clusters.json").read_text()
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(audit["qualified_signals"][0]["canonical_title"], "Canonical A")
        self.assertTrue(evidence_exists)
        self.assertEqual(clustering["components"][0]["page_ids"], [42])
        self.assertFalse(clustering["components"][0]["is_candidate_cluster"])

    def test_cli_acquires_attention_from_explicit_fixture_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            fixture_path = temporary_path / "wikimedia-fixture.json"
            fixture_path.write_text(json.dumps(fixture_payload()))
            output_directory = temporary_path / "runs"
            environment = os.environ.copy()
            environment["DATABASE_URL"] = "postgresql://postgres:test@localhost:55432/audience_intelligence_test"
            environment["AUDIENCE_TREND_MINER_TEST_MODE"] = "1"
            environment["AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL"] = ""
            environment["GROQ_API_KEY"] = "test-key"
            environment["AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE"] = str(fixture_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "--as-of",
                    "2026-07-16",
                    "--output-dir",
                    str(output_directory),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            run_directory = next(output_directory.iterdir())
            audit = json.loads((run_directory / "audit.json").read_text())

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(audit["raw_candidate_titles"], ["Alias_A"])
            self.assertEqual(
                audit["canonical_articles"][0]["aliases"][0]["daily_views"][0],
                {"date": "2026-07-01", "views": 5},
            )
            self.assertTrue(
                (run_directory / "wikimedia" / "metadata" / "Alias_A.json").is_file()
            )

    def test_cli_aborts_without_artifacts_when_discovery_fixture_exhausts_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            payload = fixture_payload()
            payload["transient_failures"] = {"discovery:2026-07-11": 3}
            fixture_path = temporary_path / "wikimedia-fixture.json"
            fixture_path.write_text(json.dumps(payload))
            output_directory = temporary_path / "runs"
            environment = os.environ.copy()
            environment["DATABASE_URL"] = "postgresql://postgres:test@localhost:55432/audience_intelligence_test"
            environment["AUDIENCE_TREND_MINER_TEST_MODE"] = "1"
            environment["AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE"] = str(fixture_path)
            environment["GROQ_API_KEY"] = "test-key"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "--as-of",
                    "2026-07-16",
                    "--output-dir",
                    str(output_directory),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output_directory.exists())
            self.assertIn("after 3 attempts", completed.stderr)

if __name__ == "__main__":
    unittest.main()
