from __future__ import annotations

import unittest
from datetime import date, timedelta

from audience_trend_miner.transformation import (
    AliasEvidenceInput,
    IncompletePageviewsEvidence,
    TerminalEvidenceFailure,
    form_wikimedia_attention,
    transform_alias,
)
from audience_trend_miner.wikimedia import (
    AliasEvidence,
    AliasEvidenceFailure,
    AnalysisWindows,
    DailyView,
    MetadataResponse,
)


WINDOWS = AnalysisWindows(
    date(2026, 7, 1), date(2026, 7, 7), date(2026, 7, 8), date(2026, 7, 14)
)


def daily(previous: int, current: int) -> tuple[DailyView, ...]:
    return tuple(
        DailyView(
            date(2026, 7, 1) + timedelta(days=offset),
            previous if offset < 7 else current,
        )
        for offset in range(14)
    )


def metadata(title: str, page_id: int = 42) -> MetadataResponse:
    return MetadataResponse(page_id, title, "Lead.", ("Examples",), {})


class WikimediaAttentionTransformationTest(unittest.TestCase):
    def test_transforms_complete_alias_evidence_and_forms_canonical_article(self) -> None:
        first = transform_alias(
            AliasEvidenceInput("Alias_A", daily(10, 20), metadata("Canonical A")),
            WINDOWS,
        )
        second = transform_alias(
            AliasEvidenceInput("Canonical_A", daily(1, 2), metadata("Canonical A")),
            WINDOWS,
        )

        self.assertIsInstance(first, AliasEvidence)
        result = form_wikimedia_attention(
            ("Alias_A", "Canonical_A"), (first, second)  # type: ignore[arg-type]
        )

        article = result.canonical_articles[0]
        self.assertEqual(
            (article.page_id, article.previous_window_views, article.current_window_views),
            (42, 77, 154),
        )
        self.assertEqual([item.raw_title for item in article.aliases], ["Alias_A", "Canonical_A"])

    def test_reports_incomplete_pageviews_without_fetching_or_retrying(self) -> None:
        result = transform_alias(
            AliasEvidenceInput("Alias_A", daily(1, 2)[:-1], metadata("Canonical A")),
            WINDOWS,
        )

        self.assertIsInstance(result, IncompletePageviewsEvidence)
        self.assertIn("complete dated observations", result.reason)

    def test_converts_terminal_alias_evidence_failure_to_degraded_result(self) -> None:
        alias = transform_alias(
            AliasEvidenceInput(
                "Broken_Alias",
                TerminalEvidenceFailure("pageviews", 3, "unavailable"),
                metadata("Broken"),
            ),
            WINDOWS,
        )

        self.assertIsInstance(alias, AliasEvidenceFailure)
        result = form_wikimedia_attention(("Broken_Alias",), (alias,))  # type: ignore[arg-type]
        self.assertTrue(result.degraded)
        self.assertEqual(result.failures[0].operation, "pageviews")

    def test_conflicting_canonical_titles_exclude_the_page_group(self) -> None:
        first = transform_alias(
            AliasEvidenceInput("First", daily(1, 2), metadata("First title")), WINDOWS
        )
        second = transform_alias(
            AliasEvidenceInput("Second", daily(1, 2), metadata("Second title")), WINDOWS
        )

        result = form_wikimedia_attention(
            ("First", "Second"), (first, second)  # type: ignore[arg-type]
        )

        self.assertEqual(result.canonical_articles, ())
        self.assertEqual(result.failures[0].operation, "canonicalization")


if __name__ == "__main__":
    unittest.main()
