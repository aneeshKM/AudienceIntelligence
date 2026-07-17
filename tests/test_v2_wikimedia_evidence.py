from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FIXTURE = Path(__file__).with_name("fixtures") / "v2_wikimedia_evidence.json"


class V2WikimediaEvidenceCliTest(unittest.TestCase):
    def test_fixture_stage_publishes_complete_country_evidence_without_zero_filling(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "v2-wikimedia-evidence",
                    "--run-id",
                    "evidence-run",
                    "--as-of",
                    "2026-07-17",
                    "--output-dir",
                    output_dir,
                    "--fixture",
                    str(FIXTURE),
                    "--progress-format",
                    "json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            artifact_path = Path(output_dir) / "evidence-run" / "wikimedia-evidence.json"
            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            payload = artifact["payload"]

            self.assertEqual(artifact["status"], "complete")
            self.assertEqual(payload["as_of_date"], "2026-07-17")
            self.assertEqual(
                payload["nominal_windows"],
                {
                    "previous": {"start": "2026-07-02", "end": "2026-07-08"},
                    "current": {"start": "2026-07-09", "end": "2026-07-15"},
                },
            )
            self.assertEqual(len(payload["nominal_days"]), 14)
            self.assertEqual(payload["coverage"], {"previous": 5, "current": 5})
            self.assertEqual(payload["candidate_universe"], ["Alias_A", "Alias_B", "Canonical_A"])
            self.assertEqual(
                [(page["page_id"], page["aliases"]) for page in payload["canonical_pages"]],
                [(42, ["Alias_A", "Canonical_A"]), (84, ["Alias_B"])],
            )
            self.assertEqual(payload["canonical_pages"][0]["lead"], "A long canonical lead used for semantic evidence.")
            self.assertEqual(payload["canonical_pages"][0]["categories"], ["Useful", "Visible"])
            self.assertEqual(payload["exclusions"]["metadata_pages_unavailable"], 0)
            self.assertEqual(payload["exclusions"]["main_page"], 0)
            self.assertEqual(payload["exclusions"]["internal_namespaces"], {})
            observations = payload["canonical_pages"][0]["observations"]
            self.assertEqual(
                observations,
                [
                    {"date": "2026-07-02", "views_ceil": 110},
                    {"date": "2026-07-05", "views_ceil": 130},
                    {"date": "2026-07-09", "views_ceil": 210},
                    {"date": "2026-07-11", "views_ceil": 245},
                    {"date": "2026-07-15", "views_ceil": 240},
                ],
            )
            self.assertNotIn("2026-07-03", {item["date"] for item in observations})
            self.assertEqual(payload["exclusions"]["non_en_wikipedia_records"], 2)
            self.assertEqual(payload["exclusions"]["unavailable_days"], ["2026-07-04", "2026-07-07", "2026-07-10", "2026-07-14"])
            self.assertEqual(payload["provenance"]["source"], "fixture:top-per-country/US")
            self.assertEqual(payload["completion"], {"status": "complete", "minimum_successful_days_per_window": 4})
            self.assertEqual(len(payload["daily_cutoffs"]), 10)
            cutoffs = {
                item["date"]: item["views_ceil"] for item in payload["daily_cutoffs"]
            }
            self.assertEqual(cutoffs["2026-07-02"], 90)
            self.assertEqual(cutoffs["2026-07-06"], None)

    def test_metadata_exclusions_and_unavailability_are_neutral_and_deterministic(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        fixture["daily_responses"]["2026-07-02"]["records"].extend(
            [
                {"project": "en.wikipedia", "article": "Main_Page", "views_ceil": 100},
                {"project": "en.wikipedia", "article": "Special:Search", "views_ceil": 100},
                {"project": "en.wikipedia", "article": "Unavailable", "views_ceil": 100},
            ]
        )
        fixture["canonical_pages"].update(
            {
                "Main_Page": {"page_id": 1, "canonical_title": "Main Page", "lead": "", "categories": []},
                "Special:Search": {"page_id": -1, "canonical_title": "Special:Search", "lead": "", "categories": []},
                "Unavailable": {"unavailable": True},
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture_path = root / "fixture.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "audience_trend_miner", "v2-wikimedia-evidence", "--run-id", "filtered", "--as-of", "2026-07-17", "--output-dir", str(root / "runs"), "--fixture", str(fixture_path)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((root / "runs" / "filtered" / "wikimedia-evidence.json").read_text())["payload"]
            self.assertEqual(payload["exclusions"]["metadata_pages_unavailable"], 1)
            self.assertEqual(payload["exclusions"]["main_page"], 1)
            self.assertEqual(payload["exclusions"]["internal_namespaces"], {"special": 1})
            self.assertEqual([page["page_id"] for page in payload["canonical_pages"]], [42, 84])

    def test_stage_rejects_an_effective_window_below_minimum_coverage(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        fixture["daily_responses"]["2026-07-05"] = {"error": "unavailable"}
        fixture["daily_responses"]["2026-07-06"] = {"error": "unavailable"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            fixture_path = temporary_path / "insufficient-coverage.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            output_dir = temporary_path / "runs"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "audience_trend_miner",
                    "v2-wikimedia-evidence",
                    "--run-id",
                    "low-coverage-run",
                    "--as-of",
                    "2026-07-17",
                    "--output-dir",
                    str(output_dir),
                    "--fixture",
                    str(fixture_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "previous Effective Window has 3 successful days; at least 4 are required",
                completed.stderr,
            )
            self.assertFalse(
                (output_dir / "low-coverage-run" / "wikimedia-evidence.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
