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


class CliWikimediaTest(unittest.TestCase):
    def test_cli_acquires_attention_from_explicit_fixture_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            fixture_path = temporary_path / "wikimedia-fixture.json"
            fixture_path.write_text(json.dumps(fixture_payload()))
            output_directory = temporary_path / "runs"
            environment = os.environ.copy()
            environment["AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL"] = ""
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

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output_directory.exists())
            self.assertIn("after 3 attempts", completed.stderr)

    def test_cli_publishes_degraded_run_when_one_article_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            payload = fixture_payload()
            for titles in payload["discovery"].values():
                titles.append("Broken_Alias")
            payload["transient_failures"] = {"metadata:Broken_Alias": 3}
            payload["pageviews"]["Broken_Alias"] = payload["pageviews"]["Alias_A"]
            fixture_path = temporary_path / "wikimedia-fixture.json"
            fixture_path.write_text(json.dumps(payload))
            output_directory = temporary_path / "runs"
            environment = os.environ.copy()
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
            self.assertTrue(audit["degraded"])
            self.assertEqual(audit["failures"][0]["subject"], "Broken_Alias")
            self.assertEqual(
                [article["canonical_title"] for article in audit["canonical_articles"]],
                ["Canonical A"],
            )


if __name__ == "__main__":
    unittest.main()
