from __future__ import annotations

import unittest
from datetime import date, timedelta
import tempfile
from pathlib import Path
import json

from audience_trend_miner.v2.wikimedia_evidence.jobs import (
    CompletedEvidence,
    EvidenceJobStore,
    FailedEvidence,
)
from audience_trend_miner.resumable_wikimedia import acquire_resumable_wikimedia_attention
from audience_trend_miner.wikimedia import AnalysisWindows, FixtureWikimediaAdapter
from audience_trend_miner.v2.wikimedia_evidence import (
    acquire_country_days,
    execute_wikimedia_evidence,
)
from tests.postgres import test_database_url


DATABASE_URL = test_database_url()


class EvidenceJobStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = EvidenceJobStore(DATABASE_URL)
        cls.store.migrate()

    def setUp(self) -> None:
        self.store.clear_for_tests()

    def test_scheduling_is_idempotent_and_claim_is_atomic(self) -> None:
        self.store.schedule_alias_evidence("run-1", ("Alias_A",))
        self.store.schedule_alias_evidence("run-1", ("Alias_A",))

        claimed = self.store.claim(
            "fetcher-1", lease_seconds=60, operations=("pageviews",)
        )

        self.assertEqual((claimed.operation, claimed.subject), ("pageviews", "Alias_A"))
        self.assertIsNone(
            self.store.claim(
                "fetcher-2", lease_seconds=60, operations=("pageviews",)
            )
        )

    def test_resume_rejects_changed_effective_configuration(self) -> None:
        self.store.ensure_run("run-1", {"model": "model-a"})

        with self.assertRaisesRegex(ValueError, "does not match"):
            self.store.ensure_run("run-1", {"model": "model-b"})

    def test_publication_path_is_reserved_before_filesystem_exposure(self) -> None:
        self.store.ensure_run("run-1", {"model": "model-a"})
        self.store.reserve_publication_path("run-1", "/runs/one")

        with self.assertRaisesRegex(ValueError, "different path"):
            self.store.reserve_publication_path("run-1", "/runs/two")

        self.store.mark_publication_complete("run-1", "/runs/one")

    def test_expired_claim_is_recovered_and_attempt_is_incremented(self) -> None:
        self.store.schedule_alias_evidence("run-1", ("Alias_A",))
        first = self.store.claim(
            "fetcher-1", lease_seconds=0, operations=("metadata",)
        )
        recovered = self.store.claim(
            "fetcher-2", lease_seconds=60, operations=("metadata",)
        )

        self.assertEqual(first.attempts, 1)
        self.assertEqual(recovered.attempts, 2)
        self.assertEqual(recovered.claimed_by, "fetcher-2")

    def test_stale_worker_cannot_complete_after_lease_is_reclaimed(self) -> None:
        self.store.schedule_alias_evidence("run-1", ("Alias_A",))
        stale = self.store.claim(
            "fetcher-1", lease_seconds=0, operations=("metadata",)
        )
        current = self.store.claim(
            "fetcher-2", lease_seconds=60, operations=("metadata",)
        )

        with self.assertRaisesRegex(RuntimeError, "lease was lost"):
            self.store.complete(stale, {"stale": True})
        self.store.complete(current, {"current": True})

    def test_repeated_crashes_exhaust_bounded_claim_attempts(self) -> None:
        self.store.schedule_alias_evidence("run-1", ("Alias_A",))
        for attempt in range(1, 4):
            claimed = self.store.claim(
                f"fetcher-{attempt}", lease_seconds=0, operations=("metadata",)
            )
            self.assertEqual(claimed.attempts, attempt)

        self.assertIsNone(
            self.store.claim(
                "fetcher-4", lease_seconds=60, operations=("metadata",)
            )
        )
        result = self.store.results_at_barrier("run-1", ("metadata",))[0]
        self.assertIsInstance(result, FailedEvidence)
        self.assertEqual(result.attempts, 3)

    def test_jsonb_evidence_and_terminal_barrier_survive_resume(self) -> None:
        self.store.schedule_alias_evidence("run-1", ("Alias_A", "Alias_B"))
        completed = self.store.claim(
            "fetcher", lease_seconds=60, operations=("pageviews",)
        )
        self.store.complete(completed, {"items": [{"views": 42}]})
        failed = self.store.claim(
            "fetcher", lease_seconds=60, operations=("pageviews",)
        )
        self.store.fail(failed, "unavailable", terminal=True)

        results = self.store.results_at_barrier("run-1", ("pageviews",))
        successful = [item for item in results if isinstance(item, CompletedEvidence)]
        failures = [item for item in results if isinstance(item, FailedEvidence)]
        self.assertEqual(successful[0].evidence, {"items": [{"views": 42}]})
        self.assertEqual(failures[0].reason, "unavailable")

    def test_barrier_does_not_expose_partial_evidence(self) -> None:
        self.store.schedule_discovery("run-1", ("2026-07-08", "2026-07-09"))
        first = self.store.claim(
            "fetcher", lease_seconds=60, operations=("discovery",)
        )
        self.store.complete(first, {"titles": ["Alias_A"]})

        with self.assertRaisesRegex(RuntimeError, "before the run barrier"):
            self.store.results_at_barrier("run-1", ("discovery",))

    def test_resumed_acquisition_reuses_persisted_evidence(self) -> None:
        adapter = CountingAdapter(
            FixtureWikimediaAdapter(
                discovery={
                    (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): ["Alias_A"]
                    for offset in range(7)
                },
                pageviews={
                    "Alias_A": [
                        {
                            "date": (date(2026, 7, 1) + timedelta(days=offset)).isoformat(),
                            "views": 10 if offset < 7 else 20,
                        }
                        for offset in range(14)
                    ]
                },
                metadata={
                    "Alias_A": {
                        "page_id": 42,
                        "canonical_title": "Canonical A",
                        "extract": "Lead.",
                        "categories": [],
                    }
                },
            )
        )
        windows = AnalysisWindows(
            date(2026, 7, 1), date(2026, 7, 7), date(2026, 7, 8), date(2026, 7, 14)
        )

        first = acquire_resumable_wikimedia_attention(
            "stable-run", windows, adapter, self.store, workers=2
        )
        calls_after_first = adapter.calls
        second = acquire_resumable_wikimedia_attention(
            "stable-run", windows, adapter, self.store, workers=2
        )

        self.assertEqual(adapter.calls, calls_after_first)
        self.assertEqual(first.canonical_articles, second.canonical_articles)
        fetched = self.store.results_at_barrier(
            "stable-run", ("discovery", "pageviews", "metadata")
        )
        self.assertEqual(len(fetched), 9)
        self.assertFalse(any(item.operation == "transform" for item in fetched))

    def test_incomplete_pageviews_exhaust_as_fetching_degradation(self) -> None:
        adapter = FixtureWikimediaAdapter(
            discovery={
                (date(2026, 7, 8) + timedelta(days=offset)).isoformat(): ["Alias_A"]
                for offset in range(7)
            },
            pageviews={"Alias_A": []},
            metadata={
                "Alias_A": {
                    "page_id": 42,
                    "canonical_title": "Canonical A",
                    "extract": "Lead.",
                    "categories": [],
                }
            },
        )
        windows = AnalysisWindows(
            date(2026, 7, 1), date(2026, 7, 7), date(2026, 7, 8), date(2026, 7, 14)
        )

        result = acquire_resumable_wikimedia_attention(
            "incomplete-pageviews", windows, adapter, self.store, workers=2
        )

        self.assertTrue(result.degraded)
        self.assertEqual(result.failures[0].operation, "pageviews")
        pageviews = self.store.results_at_barrier(
            "incomplete-pageviews", ("pageviews",)
        )[0]
        self.assertIsInstance(pageviews, FailedEvidence)
        self.assertEqual(pageviews.attempts, 3)

    def test_country_day_jobs_retry_independently_report_progress_and_resume(self) -> None:
        adapter = CountryAdapter()
        progress = []

        first = acquire_country_days(
            "country-run",
            tuple(date(2026, 7, 1) + timedelta(days=offset) for offset in range(14)),
            adapter,
            self.store,
            progress.append,
            workers=2,
        )
        calls_after_first = adapter.calls
        second_progress = []
        second = acquire_country_days(
            "country-run",
            tuple(date(2026, 7, 1) + timedelta(days=offset) for offset in range(14)),
            adapter,
            self.store,
            second_progress.append,
            workers=2,
        )

        self.assertEqual(adapter.calls, calls_after_first)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 14)
        exhausted = next(item for item in first if item.subject == "2026-07-03")
        self.assertIsInstance(exhausted, FailedEvidence)
        self.assertEqual(exhausted.attempts, 3)
        successful = next(item for item in first if item.subject == "2026-07-01")
        self.assertEqual(
            successful.evidence["records"],
            [{"project": "en.wikipedia", "article": "Page_1", "views_ceil": 1}],
        )
        self.assertEqual(progress[-1].progress.current, 14)
        self.assertEqual(progress[-1].progress.total, 14)
        self.assertEqual([event.sequence for event in progress], list(range(1, 15)))
        self.assertEqual(second_progress[-1].progress.current, 14)

    def test_country_day_jobs_reject_a_window_below_coverage_threshold(self) -> None:
        adapter = CountryAdapter(
            unavailable={"2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"}
        )

        with self.assertRaisesRegex(
            ValueError, "previous Effective Window has 3 successful days"
        ):
            acquire_country_days(
                "low-country-coverage",
                tuple(
                    date(2026, 7, 1) + timedelta(days=offset)
                    for offset in range(14)
                ),
                adapter,
                self.store,
                lambda event: None,
                workers=2,
            )

    def test_production_stage_resolves_metadata_batches_publishes_and_resumes(self) -> None:
        adapter = ProductionCountryAdapter()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = execute_wikimedia_evidence(
                run_id="production-stage", as_of_date=date(2026, 7, 17),
                output_root=root, adapter=adapter, store=self.store,
                progress_sink=lambda event: None, workers=2,
            )
            calls = adapter.calls
            second = execute_wikimedia_evidence(
                run_id="production-stage", as_of_date=date(2026, 7, 17),
                output_root=root, adapter=adapter, store=self.store,
                progress_sink=lambda event: None, workers=2,
            )

            self.assertEqual(first, second)
            self.assertEqual(adapter.calls, calls)
            payload = json.loads(first.read_text())["payload"]
            self.assertEqual(payload["canonical_pages"][0]["lead"], "Lead")
            self.assertEqual(payload["canonical_pages"][0]["categories"], ["Visible"])
            self.assertEqual(payload["exclusions"]["metadata_pages_unavailable"], 0)


class CountryAdapter:
    def __init__(self, unavailable=None) -> None:
        self.calls = 0
        self.attempts = {}
        self.unavailable = unavailable or {"2026-07-03"}

    def daily_country_top_pages(self, day):
        from audience_trend_miner.wikimedia import (
            CountryPageviewRecord,
            CountryTopPagesResponse,
            WikimediaTransientError,
        )

        self.calls += 1
        day_text = day.isoformat()
        attempt = self.attempts.get(day_text, 0) + 1
        self.attempts[day_text] = attempt
        if day_text in self.unavailable:
            raise WikimediaTransientError("unavailable", retry_immediately=True)
        return CountryTopPagesResponse(
            (
                CountryPageviewRecord("en.wikipedia", f"Page_{day.day}", day.day),
                CountryPageviewRecord("de.wikipedia", "Andere", 100),
            ),
            {"day": day_text},
        )


class ProductionCountryAdapter(CountryAdapter):
    def __init__(self):
        super().__init__(unavailable=set())

    def metadata_batch(self, titles):
        from audience_trend_miner.wikimedia import MetadataBatchResponse, MetadataResponse

        self.calls += 1
        return MetadataBatchResponse(
            (MetadataResponse(42, "Canonical Page", "Lead", ("Visible",), {}),),
            {title: 42 for title in titles},
            (),
        )


class CountingAdapter:
    def __init__(self, wrapped: FixtureWikimediaAdapter) -> None:
        self.wrapped = wrapped
        self.calls = 0

    def daily_top_pages(self, day):
        self.calls += 1
        return self.wrapped.daily_top_pages(day)

    def article_pageviews(self, raw_title, start, end):
        self.calls += 1
        return self.wrapped.article_pageviews(raw_title, start, end)

    def article_metadata(self, raw_title):
        self.calls += 1
        return self.wrapped.article_metadata(raw_title)


if __name__ == "__main__":
    unittest.main()
