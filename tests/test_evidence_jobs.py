from __future__ import annotations

import os
import unittest
from datetime import date, timedelta

from audience_trend_miner.evidence_jobs import EvidenceJobStore
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

    def test_enqueue_is_idempotent_and_claim_is_atomic(self) -> None:
        self.store.enqueue("run-1", "pageviews", "Alias_A")
        self.store.enqueue("run-1", "pageviews", "Alias_A")

        claimed = self.store.claim("fetcher-1", lease_seconds=60)

        self.assertEqual((claimed.operation, claimed.subject), ("pageviews", "Alias_A"))
        self.assertIsNone(self.store.claim("fetcher-2", lease_seconds=60))

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
        self.store.enqueue("run-1", "metadata", "Alias_A")
        first = self.store.claim("fetcher-1", lease_seconds=0)
        recovered = self.store.claim("fetcher-2", lease_seconds=60)

        self.assertEqual(first.attempts, 1)
        self.assertEqual(recovered.attempts, 2)
        self.assertEqual(recovered.claimed_by, "fetcher-2")

    def test_stale_worker_cannot_complete_after_lease_is_reclaimed(self) -> None:
        self.store.enqueue("run-1", "metadata", "Alias_A")
        stale = self.store.claim("fetcher-1", lease_seconds=0)
        current = self.store.claim("fetcher-2", lease_seconds=60)

        with self.assertRaisesRegex(RuntimeError, "lease was lost"):
            self.store.complete(stale, {"stale": True})
        self.store.complete(current, {"current": True})

    def test_repeated_crashes_exhaust_bounded_claim_attempts(self) -> None:
        self.store.enqueue("run-1", "metadata", "Alias_A")
        for attempt in range(1, 4):
            claimed = self.store.claim(f"fetcher-{attempt}", lease_seconds=0)
            self.assertEqual(claimed.attempts, attempt)

        self.assertIsNone(self.store.claim("fetcher-4", lease_seconds=60))
        job = self.store.jobs("run-1")[0]
        self.assertEqual((job.status, job.attempts), ("failed", 3))

    def test_jsonb_evidence_and_terminal_barrier_survive_resume(self) -> None:
        self.store.enqueue("run-1", "pageviews", "Alias_A")
        self.store.enqueue("run-1", "pageviews", "Alias_B")
        completed = self.store.claim("fetcher", lease_seconds=60)
        self.store.complete(completed, {"items": [{"views": 42}]})
        failed = self.store.claim("fetcher", lease_seconds=60)
        self.store.fail(failed, "unavailable", terminal=True)

        self.assertTrue(self.store.run_jobs_terminal("run-1", "pageviews"))
        self.assertEqual(
            self.store.completed_evidence("run-1", "pageviews"),
            ((completed.subject, {"items": [{"views": 42}]}),),
        )

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
        self.assertEqual(
            [job.status for job in self.store.jobs("stable-run") if job.operation == "transform"],
            ["completed"],
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
