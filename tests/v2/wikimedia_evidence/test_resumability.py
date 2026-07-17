from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from audience_trend_miner.v2.wikimedia_evidence import (
    acquire_country_days,
    execute_wikimedia_evidence,
)
from audience_trend_miner.v2.wikimedia_evidence.adapters import (
    CountryPageviewRecord,
    CountryTopPagesResponse,
    MetadataBatchResponse,
    MetadataResponse,
    WikimediaTransientError,
)
from audience_trend_miner.v2.wikimedia_evidence.jobs import (
    CompletedEvidence,
    EvidenceJobStore,
    FailedEvidence,
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
        self.store.schedule_country_days("run-1", ("2026-07-01",))
        self.store.schedule_country_days("run-1", ("2026-07-01",))

        claimed = self.store.claim(
            "fetcher-1", lease_seconds=60, operations=("country-day",)
        )

        self.assertEqual(
            (claimed.operation, claimed.subject), ("country-day", "2026-07-01")
        )
        self.assertIsNone(
            self.store.claim(
                "fetcher-2", lease_seconds=60, operations=("country-day",)
            )
        )

    def test_resume_rejects_changed_effective_configuration(self) -> None:
        self.store.ensure_run("run-1", {"country": "US"})

        with self.assertRaisesRegex(ValueError, "does not match"):
            self.store.ensure_run("run-1", {"country": "CA"})

    def test_expired_claim_is_recovered_without_accepting_stale_completion(self) -> None:
        self.store.schedule_metadata_batches("run-1", ('["Alias_A"]',))
        stale = self.store.claim(
            "fetcher-1", lease_seconds=0, operations=("metadata-batch",)
        )
        current = self.store.claim(
            "fetcher-2", lease_seconds=60, operations=("metadata-batch",)
        )

        with self.assertRaisesRegex(RuntimeError, "lease was lost"):
            self.store.complete(stale, {"stale": True})
        self.store.complete(current, {"current": True})
        result = self.store.results_at_barrier("run-1", ("metadata-batch",))[0]
        self.assertIsInstance(result, CompletedEvidence)
        self.assertEqual(result.evidence, {"current": True})

    def test_country_day_jobs_retry_independently_report_progress_and_resume(self) -> None:
        adapter = CountryAdapter()
        progress = []
        days = tuple(date(2026, 7, 1) + timedelta(days=offset) for offset in range(14))

        first = acquire_country_days(
            "country-run", days, adapter, self.store, progress.append, workers=2
        )
        calls_after_first = adapter.calls
        resumed_progress = []
        second = acquire_country_days(
            "country-run", days, adapter, self.store, resumed_progress.append, workers=2
        )

        self.assertEqual(first, second)
        self.assertEqual(adapter.calls, calls_after_first)
        exhausted = next(item for item in first if item.subject == "2026-07-03")
        self.assertIsInstance(exhausted, FailedEvidence)
        self.assertEqual(exhausted.attempts, 3)
        self.assertEqual([event.sequence for event in progress], list(range(1, 15)))
        self.assertEqual(resumed_progress[-1].progress.current, 14)

    def test_production_stage_resolves_metadata_publishes_and_resumes(self) -> None:
        adapter = ProductionCountryAdapter()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = execute_wikimedia_evidence(
                run_id="production-stage",
                as_of_date=date(2026, 7, 17),
                output_root=root,
                adapter=adapter,
                store=self.store,
                progress_sink=lambda event: None,
                workers=2,
            )
            calls = adapter.calls
            second = execute_wikimedia_evidence(
                run_id="production-stage",
                as_of_date=date(2026, 7, 17),
                output_root=root,
                adapter=adapter,
                store=self.store,
                progress_sink=lambda event: None,
                workers=2,
            )

            self.assertEqual(first, second)
            self.assertEqual(adapter.calls, calls)
            payload = json.loads(first.read_text(encoding="utf-8"))["payload"]
            self.assertEqual(payload["canonical_pages"][0]["lead"], "Lead")
            self.assertEqual(payload["canonical_pages"][0]["categories"], ["Visible"])


class CountryAdapter:
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self.calls = 0
        self.attempts: dict[str, int] = {}
        self.unavailable = unavailable if unavailable is not None else {"2026-07-03"}

    def daily_country_top_pages(self, day: date) -> CountryTopPagesResponse:
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
    def __init__(self) -> None:
        super().__init__(unavailable=set())

    def metadata_batch(self, titles: tuple[str, ...]) -> MetadataBatchResponse:
        self.calls += 1
        return MetadataBatchResponse(
            (MetadataResponse(42, "Canonical Page", "Lead", ("Visible",), {}),),
            {title: 42 for title in titles},
            (),
        )


if __name__ == "__main__":
    unittest.main()
