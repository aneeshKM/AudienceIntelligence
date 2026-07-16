from __future__ import annotations

import os
import unittest
from datetime import date, timedelta

from audience_trend_miner.evidence_jobs import (
    CompletedEvidence,
    EvidenceJobStore,
    FailedEvidence,
)
from audience_trend_miner.resumable_wikimedia import acquire_resumable_wikimedia_attention
from audience_trend_miner.wikimedia import AnalysisWindows, FixtureWikimediaAdapter


DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:test@localhost:55432/audience_intelligence_test",
)


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

    def test_incomplete_transformation_retries_completed_upstream_evidence(self) -> None:
        self.store.schedule_alias_evidence("run-1", ("Alias_A",))
        pageviews = self.store.claim(
            "fetcher", lease_seconds=60, operations=("pageviews",)
        )
        self.store.complete(pageviews, {"daily_views": []})
        self.assertIsNone(
            self.store.claim_ready_transformation(
                "transformer", lease_seconds=60, run_id="run-1"
            )
        )
        metadata = self.store.claim(
            "fetcher", lease_seconds=60, operations=("metadata",)
        )
        self.store.complete(metadata, {"page_id": 1})
        ready = self.store.claim_ready_transformation(
            "transformer", lease_seconds=60, run_id="run-1"
        )

        self.store.recover_incomplete_pageviews(
            ready.job, "dated observations incomplete"
        )

        reacquired = self.store.claim(
            "fetcher", lease_seconds=60, operations=("pageviews",)
        )
        self.assertEqual(reacquired.subject, "Alias_A")
        self.assertIsNone(
            self.store.claim_ready_transformation(
                "other-transformer", lease_seconds=60, run_id="run-1"
            )
        )

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
        transformed = self.store.results_at_barrier("stable-run", ("transform",))
        self.assertEqual(len(transformed), 1)
        self.assertIsInstance(transformed[0], CompletedEvidence)


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
