from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import json
import os
from datetime import datetime, timezone
from pathlib import Path

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:test@localhost:55432/audience_intelligence_test",
)


def empty_run_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL"] = ""
    environment["GROQ_API_KEY"] = "test-key"
    environment["DATABASE_URL"] = TEST_DATABASE_URL
    return environment


class CliRunContractTest(unittest.TestCase):
    def test_fixed_as_of_run_creates_timestamped_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "--as-of",
                    "2026-07-16",
                    "--output-dir",
                    output_dir,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=empty_run_environment(),
            )

            run_directories = list(Path(output_dir).iterdir())
            self.assertEqual(
                (completed.returncode, len(run_directories)),
                (0, 1),
                completed.stderr,
            )

    def test_manifest_records_supplied_date_and_complete_analysis_windows(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "--as-of",
                    "2026-07-16",
                    "--output-dir",
                    output_dir,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=empty_run_environment(),
            )
            run_directory = next(Path(output_dir).iterdir())
            manifest = json.loads((run_directory / "manifest.json").read_text())

            self.assertEqual(
                manifest,
                {
                    "as_of_argument": "2026-07-16",
                    "as_of": "2026-07-16",
                    "current_window": {
                        "start": "2026-07-08",
                        "end": "2026-07-14",
                    },
                    "previous_window": {
                        "start": "2026-07-01",
                        "end": "2026-07-07",
                    },
                    "configuration": {
                        "model": "openai/gpt-oss-120b",
                        "classification_mode": "live",
                        "wikimedia_mode": "disabled",
                        "database_host": "localhost",
                        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                        "similarity_threshold": "0.62",
                        "embedding_mode": "local",
                    },
                    "run_id": run_directory.name,
                },
                completed.stderr,
            )

    def test_omitted_as_of_uses_current_utc_date(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            utc_date_before = datetime.now(timezone.utc).date().isoformat()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "--output-dir",
                    output_dir,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=empty_run_environment(),
            )
            utc_date_after = datetime.now(timezone.utc).date().isoformat()
            run_directory = next(Path(output_dir).iterdir())
            manifest = json.loads((run_directory / "manifest.json").read_text())

            self.assertEqual(manifest["as_of_argument"], None)
            self.assertIn(manifest["as_of"], {utc_date_before, utc_date_after})
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
