from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import jsonschema

from audience_trend_miner.publication import PublicationInput, publish_run
from audience_trend_miner.trends import TrendQualificationResult, qualify_trends
from audience_trend_miner.wikimedia import (
    AcquisitionFailure,
    AliasTraffic,
    AnalysisWindows,
    CanonicalArticle,
    DailyView,
    RawArtifact,
    WikimediaAttentionResult,
)


def publication_input(
    output_root: Path,
    *,
    attention: WikimediaAttentionResult = WikimediaAttentionResult((), (), ()),
    qualification: TrendQualificationResult = TrendQualificationResult((), (), ()),
) -> PublicationInput:
    return PublicationInput(
        output_root=output_root,
        started_at=datetime(2026, 7, 16, 17, 30, 45, 123456, tzinfo=timezone.utc),
        as_of_argument=date(2026, 7, 16),
        as_of=date(2026, 7, 16),
        windows=AnalysisWindows(
            previous_start=date(2026, 7, 1),
            previous_end=date(2026, 7, 7),
            current_start=date(2026, 7, 8),
            current_end=date(2026, 7, 14),
        ),
        attention=attention,
        qualification=qualification,
    )


class RunPublicationTest(unittest.TestCase):
    def test_publishes_complete_empty_run_from_finished_domain_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "runs"
            published = publish_run(publication_input(output_root))

            self.assertEqual(published.name, "20260716T173045123456Z")
            self.assertEqual(
                {path.name for path in published.iterdir()},
                {
                    "manifest.json",
                    "portfolio.json",
                    "audit.json",
                    "report.html",
                    "wikimedia",
                },
            )
            manifest = json.loads((published / "manifest.json").read_text())
            portfolio = json.loads((published / "portfolio.json").read_text())
            audit = json.loads((published / "audit.json").read_text())
            schema_root = Path(__file__).parents[1] / "audience_trend_miner" / "schemas"
            jsonschema.validate(
                portfolio,
                json.loads((schema_root / "portfolio.schema.json").read_text()),
            )
            jsonschema.validate(
                audit,
                json.loads((schema_root / "audit.schema.json").read_text()),
            )

        self.assertEqual(manifest["current_window"]["start"], "2026-07-08")
        self.assertEqual(portfolio["audiences"], [])
        self.assertEqual(audit["qualified_signals"], [])

    def test_publishes_degraded_qualified_signals_with_alias_lineage_and_evidence(self) -> None:
        daily_views = tuple(
            DailyView(
                date=date(2026, 7, 1 + offset),
                views=10_000 if offset < 7 else 20_000,
            )
            for offset in range(14)
        )
        alias = AliasTraffic("Signal_Alias", 70_000, 140_000, daily_views)
        article = CanonicalArticle(
            page_id=42,
            canonical_title="Commercial Signal",
            extract="A useful lead.",
            categories=("Examples",),
            previous_window_views=70_000,
            current_window_views=140_000,
            aliases=(alias,),
        )
        noise_alias = AliasTraffic("Main_Page", 70_000, 140_000, daily_views)
        noise = CanonicalArticle(
            page_id=1,
            canonical_title="Main Page",
            extract="Navigation.",
            categories=(),
            previous_window_views=70_000,
            current_window_views=140_000,
            aliases=(noise_alias,),
        )
        attention = WikimediaAttentionResult(
            raw_candidate_titles=("Broken_Alias", "Main_Page", "Signal_Alias"),
            canonical_articles=(noise, article),
            raw_artifacts=(
                RawArtifact("metadata/Signal_Alias.json", {"page_id": 42}),
            ),
            failures=(
                AcquisitionFailure("metadata", "Broken_Alias", 3, "unavailable"),
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            published = publish_run(
                publication_input(
                    Path(temporary_directory) / "runs",
                    attention=attention,
                    qualification=qualify_trends((noise, article)),
                )
            )
            audit = json.loads((published / "audit.json").read_text())
            report = (published / "report.html").read_text()

            self.assertTrue(
                (published / "wikimedia" / "metadata" / "Signal_Alias.json").is_file()
            )

        self.assertTrue(audit["degraded"])
        self.assertEqual(audit["qualified_signals"][0]["alias_titles"], ["Signal_Alias"])
        self.assertEqual(
            next(
                decision["reasons"]
                for decision in audit["decisions"]
                if decision["canonical_title"] == "Commercial Signal"
            ),
            ["all_qualification_gates_passed"],
        )
        self.assertIn("Qualified attention signals", report)
        self.assertIn("Rejected deterministic noise", report)
        self.assertIn("Main Page", report)
        self.assertIn("not yet accepted audiences", report)

    def test_failed_staging_leaves_no_completed_or_temporary_run(self) -> None:
        attention = WikimediaAttentionResult(
            raw_candidate_titles=(),
            canonical_articles=(),
            raw_artifacts=(
                RawArtifact("metadata", {}),
                RawArtifact("metadata/Signal.json", {}),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "runs"

            with self.assertRaises(OSError):
                publish_run(publication_input(output_root, attention=attention))

            self.assertTrue(output_root.is_dir())
            self.assertEqual(list(output_root.iterdir()), [])

    def test_schema_validation_failure_creates_no_output_root(self) -> None:
        invalid_attention = WikimediaAttentionResult(
            raw_candidate_titles=(),
            canonical_articles=(),
            raw_artifacts=(),
            failures=(
                AcquisitionFailure("metadata", "Broken_Alias", 4, "unavailable"),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "runs"

            with self.assertRaises(jsonschema.ValidationError):
                publish_run(
                    publication_input(output_root, attention=invalid_attention)
                )

            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
