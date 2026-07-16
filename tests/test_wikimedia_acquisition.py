from __future__ import annotations

import unittest
from datetime import date, timedelta

from audience_trend_miner.wikimedia import (
    AcquisitionFailure,
    AcquisitionSettings,
    AnalysisWindows,
    FixtureWikimediaAdapter,
    HttpWikimediaAdapter,
    IncompleteCandidateUniverseError,
    acquire_wikimedia_attention,
)


def complete_daily_views(previous: int, current: int) -> list[dict[str, object]]:
    return [
        {
            "date": (date(2026, 7, 1) + timedelta(days=offset)).isoformat(),
            "views": previous if offset < 7 else current,
        }
        for offset in range(14)
    ]


class CanonicalArticleAggregationTest(unittest.TestCase):
    def test_builds_one_canonical_article_with_dated_alias_traffic_and_evidence(self) -> None:
        adapter = FixtureWikimediaAdapter(
            discovery={
                (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): [
                    "Alias_A",
                    "Canonical_A",
                ]
                for offset in range(7)
            },
            pageviews={
                "Alias_A": complete_daily_views(10, 20),
                "Canonical_A": complete_daily_views(1, 2),
            },
            metadata={
                "Alias_A": {
                    "page_id": 42,
                    "canonical_title": "Canonical A",
                    "extract": "A useful lead.",
                    "categories": ["Examples"],
                },
                "Canonical_A": {
                    "page_id": 42,
                    "canonical_title": "Canonical A",
                    "extract": "A useful lead.",
                    "categories": ["Examples"],
                },
            },
        )

        result = acquire_wikimedia_attention(
            AnalysisWindows(
                previous_start=date(2026, 7, 1),
                previous_end=date(2026, 7, 7),
                current_start=date(2026, 7, 8),
                current_end=date(2026, 7, 14),
            ),
            adapter,
        )

        self.assertFalse(result.degraded)
        self.assertEqual(result.raw_candidate_titles, ("Alias_A", "Canonical_A"))
        self.assertEqual(len(result.canonical_articles), 1)
        article = result.canonical_articles[0]
        self.assertEqual(
            (
                article.page_id,
                article.canonical_title,
                article.previous_window_views,
                article.current_window_views,
            ),
            (42, "Canonical A", 77, 154),
        )
        self.assertEqual(
            [alias.raw_title for alias in article.aliases],
            ["Alias_A", "Canonical_A"],
        )
        self.assertEqual(
            article.aliases[0].daily_views[0].date,
            date(2026, 7, 1),
        )
        self.assertEqual(
            article.aliases[0].daily_views[-1].date,
            date(2026, 7, 14),
        )
        self.assertEqual(
            {artifact.name for artifact in result.raw_artifacts},
            {
                *(f"discovery/{date(2026, 7, 8) + timedelta(days=offset)}.json" for offset in range(7)),
                "pageviews/Alias_A.json",
                "pageviews/Canonical_A.json",
                "metadata/Alias_A.json",
                "metadata/Canonical_A.json",
            },
        )


class CandidateUniverseAcquisitionTest(unittest.TestCase):
    def test_discovery_exhaustion_aborts_the_candidate_universe_after_three_attempts(self) -> None:
        adapter = FixtureWikimediaAdapter(
            discovery={
                (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): []
                for offset in range(7)
            },
            pageviews={},
            metadata={},
            transient_failures={"discovery:2026-07-11": 3},
        )

        with self.assertRaises(IncompleteCandidateUniverseError) as raised:
            acquire_wikimedia_attention(
                AnalysisWindows(
                    previous_start=date(2026, 7, 1),
                    previous_end=date(2026, 7, 7),
                    current_start=date(2026, 7, 8),
                    current_end=date(2026, 7, 14),
                ),
                adapter,
            )

        self.assertEqual(
            (
                raised.exception.failure.operation,
                raised.exception.failure.subject,
                raised.exception.failure.attempts,
            ),
            ("discovery", "2026-07-11", 3),
        )


class AliasEvidenceAcquisitionTest(unittest.TestCase):
    def test_article_failure_degrades_and_preserves_the_healthy_candidate(self) -> None:
        adapter = FixtureWikimediaAdapter(
            discovery={
                (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): [
                    "Broken_Alias",
                    "Healthy_Alias",
                ]
                for offset in range(7)
            },
            pageviews={
                "Healthy_Alias": complete_daily_views(3, 6),
            },
            metadata={
                "Healthy_Alias": {
                    "page_id": 7,
                    "canonical_title": "Healthy Article",
                    "extract": "Healthy lead.",
                    "categories": [],
                }
            },
            transient_failures={"pageviews:Broken_Alias": 3},
        )

        result = acquire_wikimedia_attention(
            AnalysisWindows(
                previous_start=date(2026, 7, 1),
                previous_end=date(2026, 7, 7),
                current_start=date(2026, 7, 8),
                current_end=date(2026, 7, 14),
            ),
            adapter,
        )

        self.assertTrue(result.degraded)
        self.assertEqual(
            [article.canonical_title for article in result.canonical_articles],
            ["Healthy Article"],
        )
        self.assertEqual(
            result.failures,
            (
                AcquisitionFailure(
                    operation="pageviews",
                    subject="Broken_Alias",
                    attempts=3,
                    reason="scripted transient failure: pageviews:Broken_Alias",
                ),
            ),
        )

    def test_metadata_failure_preserves_pageviews_evidence_acquired_before_failure(self) -> None:
        adapter = FixtureWikimediaAdapter(
            discovery={
                (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): [
                    "Broken_Metadata"
                ]
                for offset in range(7)
            },
            pageviews={
                "Broken_Metadata": complete_daily_views(3, 6),
            },
            metadata={},
            transient_failures={"metadata:Broken_Metadata": 3},
        )

        result = acquire_wikimedia_attention(
            AnalysisWindows(
                previous_start=date(2026, 7, 1),
                previous_end=date(2026, 7, 7),
                current_start=date(2026, 7, 8),
                current_end=date(2026, 7, 14),
            ),
            adapter,
        )

        self.assertEqual(result.canonical_articles, ())
        self.assertEqual(result.failures[0].operation, "metadata")
        self.assertIn(
            "pageviews/Broken_Metadata.json",
            {artifact.name for artifact in result.raw_artifacts},
        )
        self.assertNotIn(
            "metadata/Broken_Metadata.json",
            {artifact.name for artifact in result.raw_artifacts},
        )

    def test_incomplete_daily_pageviews_are_rejected_as_an_article_failure(self) -> None:
        adapter = FixtureWikimediaAdapter(
            discovery={
                (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): [
                    "Incomplete_Alias"
                ]
                for offset in range(7)
            },
            pageviews={
                "Incomplete_Alias": complete_daily_views(3, 6)[:-1],
            },
            metadata={
                "Incomplete_Alias": {
                    "page_id": 9,
                    "canonical_title": "Incomplete Article",
                    "extract": "Incomplete lead.",
                    "categories": [],
                }
            },
        )

        result = acquire_wikimedia_attention(
            AnalysisWindows(
                previous_start=date(2026, 7, 1),
                previous_end=date(2026, 7, 7),
                current_start=date(2026, 7, 8),
                current_end=date(2026, 7, 14),
            ),
            adapter,
            AcquisitionSettings(base_backoff_seconds=0),
        )

        self.assertEqual(result.canonical_articles, ())
        self.assertEqual(
            (result.failures[0].operation, result.failures[0].attempts),
            ("pageviews", 3),
        )
        self.assertIn("complete dated observations", result.failures[0].reason)


class CanonicalArticleConflictTest(unittest.TestCase):
    def test_conflicting_canonical_titles_exclude_the_page_group_deterministically(self) -> None:
        adapter = FixtureWikimediaAdapter(
            discovery={
                (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): [
                    "First_Alias",
                    "Second_Alias",
                ]
                for offset in range(7)
            },
            pageviews={
                "First_Alias": complete_daily_views(1, 2),
                "Second_Alias": complete_daily_views(3, 4),
            },
            metadata={
                "First_Alias": {
                    "page_id": 42,
                    "canonical_title": "First Canonical Title",
                    "extract": "First lead.",
                    "categories": [],
                },
                "Second_Alias": {
                    "page_id": 42,
                    "canonical_title": "Second Canonical Title",
                    "extract": "Second lead.",
                    "categories": [],
                },
            },
        )

        result = acquire_wikimedia_attention(
            AnalysisWindows(
                previous_start=date(2026, 7, 1),
                previous_end=date(2026, 7, 7),
                current_start=date(2026, 7, 8),
                current_end=date(2026, 7, 14),
            ),
            adapter,
        )

        self.assertEqual(result.canonical_articles, ())
        self.assertEqual(
            (result.failures[0].operation, result.failures[0].subject),
            ("canonicalization", "42"),
        )
        self.assertIn("conflicting canonical titles", result.failures[0].reason)


class RecordingJsonTransport:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get_json(self, url: str) -> object:
        self.urls.append(url)
        if "/top/" in url:
            return {"items": [{"articles": [{"article": "Raw_Title"}]}]}
        if "/per-article/" in url:
            return {
                "items": [
                    {"timestamp": "2026070100", "views": 12},
                ]
            }
        return {
            "query": {
                "pages": [
                    {
                        "pageid": 123,
                        "title": "Canonical Title",
                        "extract": "Lead text.",
                        "categories": [{"title": "Category:Examples"}],
                    }
                ]
            }
        }


class HttpWikimediaAdapterTest(unittest.TestCase):
    def test_translates_wikimedia_operations_and_response_envelopes(self) -> None:
        transport = RecordingJsonTransport()
        adapter = HttpWikimediaAdapter(transport=transport)

        discovery = adapter.daily_top_pages(date(2026, 7, 8))
        pageviews = adapter.article_pageviews(
            "Raw/Title", date(2026, 7, 1), date(2026, 7, 14)
        )
        metadata = adapter.article_metadata("Raw/Title")

        self.assertEqual(discovery.titles, ("Raw_Title",))
        self.assertEqual(pageviews.daily_views[0].date, date(2026, 7, 1))
        self.assertEqual(
            (metadata.page_id, metadata.canonical_title, metadata.categories),
            (123, "Canonical Title", ("Examples",)),
        )
        self.assertEqual(
            transport.urls,
            [
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/2026/07/08",
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Raw%2FTitle/daily/2026070100/2026071400",
                "https://en.wikipedia.org/w/api.php?action=query&format=json&formatversion=2&redirects=1&prop=extracts%7Ccategories&exintro=1&explaintext=1&titles=Raw%2FTitle",
            ],
        )


if __name__ == "__main__":
    unittest.main()
