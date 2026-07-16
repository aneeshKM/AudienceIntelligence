from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import random
import time

from audience_trend_miner.evidence_jobs import EvidenceJob, EvidenceJobStore
from audience_trend_miner.wikimedia import (
    AcquisitionFailure,
    AliasEvidence,
    AliasEvidenceFailure,
    AliasTraffic,
    AnalysisWindows,
    DailyView,
    IncompleteCandidateUniverseError,
    MetadataResponse,
    RawArtifact,
    WikimediaAdapter,
    WikimediaEvidence,
    WikimediaAttentionResult,
    WikimediaPermanentError,
    transform_wikimedia_attention,
)


def acquire_resumable_wikimedia_attention(
    run_id: str,
    windows: AnalysisWindows,
    adapter: WikimediaAdapter,
    store: EvidenceJobStore,
    *,
    workers: int = 4,
    configuration: dict[str, str] | None = None,
) -> WikimediaAttentionResult:
    store.migrate()
    store.ensure_run(run_id, configuration or {})
    day = windows.current_start
    while day <= windows.current_end:
        store.enqueue(run_id, "discovery", day.isoformat())
        day += timedelta(days=1)
    _drain(run_id, ("discovery",), windows, adapter, store, workers)
    discovery = _jobs_by_operation(store.jobs(run_id), "discovery")
    failed_discovery = [job for job in discovery if job.status == "failed"]
    if failed_discovery:
        job = failed_discovery[0]
        raise IncompleteCandidateUniverseError(
            AcquisitionFailure("discovery", job.subject, job.attempts, job.error or "failed")
        )
    titles = tuple(
        sorted({title for job in discovery for title in job.evidence["titles"]})
    )
    for title in titles:
        store.enqueue(run_id, "pageviews", title)
        store.enqueue(run_id, "metadata", title)
    fetch_workers = max(1, workers // 2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        fetching = executor.submit(
            _drain,
            run_id,
            ("pageviews", "metadata"),
            windows,
            adapter,
            store,
            fetch_workers,
        )
        transforming = executor.submit(
            _drain_transformations, run_id, titles, windows, store
        )
        fetching.result()
        transforming.result()
    evidence = _form_evidence(titles, store.jobs(run_id), windows)
    return transform_wikimedia_attention(evidence)


def _drain(run_id, operations, windows, adapter, store, workers) -> None:
    def work(worker_number: int) -> None:
        while not store.run_jobs_terminal(run_id, operations[0]) or any(
            not store.run_jobs_terminal(run_id, operation) for operation in operations[1:]
        ):
            job = store.claim(
                f"worker-{worker_number}",
                lease_seconds=60,
                run_id=run_id,
                operations=operations,
            )
            if job is None:
                time.sleep(0.01)
                continue
            try:
                store.complete(job, _fetch(job, windows, adapter))
            except Exception as error:
                terminal = isinstance(error, WikimediaPermanentError) or job.attempts >= 3
                store.fail(job, str(error), terminal=terminal)
                if not terminal and not getattr(error, "retry_immediately", False):
                    delay = 2 ** (job.attempts - 1)
                    time.sleep(delay + random.uniform(0, delay))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        tuple(executor.map(work, range(workers)))


def _fetch(job: EvidenceJob, windows: AnalysisWindows, adapter: WikimediaAdapter) -> dict:
    if job.operation == "discovery":
        response = adapter.daily_top_pages(date.fromisoformat(job.subject))
        return {"titles": list(response.titles), "raw": response.raw}
    if job.operation == "pageviews":
        response = adapter.article_pageviews(
            job.subject, windows.previous_start, windows.current_end
        )
        return {
            "daily_views": [
                {"date": item.date.isoformat(), "views": item.views}
                for item in response.daily_views
            ],
            "raw": response.raw,
        }
    response = adapter.article_metadata(job.subject)
    return {
        "page_id": response.page_id,
        "canonical_title": response.canonical_title,
        "extract": response.extract,
        "categories": list(response.categories),
        "raw": response.raw,
    }


def _drain_transformations(
    run_id: str,
    titles: tuple[str, ...],
    windows: AnalysisWindows,
    store: EvidenceJobStore,
) -> None:
    while True:
        jobs = store.jobs(run_id)
        indexed = {(job.operation, job.subject): job for job in jobs}
        for title in titles:
            dependencies = (
                indexed.get(("pageviews", title)),
                indexed.get(("metadata", title)),
            )
            if all(
                item is not None and item.status in {"completed", "failed"}
                for item in dependencies
            ):
                store.enqueue(run_id, "transform", title)
        transformations = _jobs_by_operation(store.jobs(run_id), "transform")
        if len(transformations) == len(titles) and all(
            job.status in {"completed", "failed"} for job in transformations
        ):
            return
        job = store.claim(
            "transformer",
            lease_seconds=60,
            run_id=run_id,
            operations=("transform",),
        )
        if job is None:
            time.sleep(0.01)
            continue
        try:
            store.complete(job, _transform_alias(job.subject, indexed, windows))
        except Exception as error:
            store.fail(job, str(error), terminal=job.attempts >= 3)


def _transform_alias(
    title: str,
    indexed: dict[tuple[str, str], EvidenceJob],
    windows: AnalysisWindows,
) -> dict:
    pageviews = indexed[("pageviews", title)]
    metadata = indexed[("metadata", title)]
    if pageviews.status != "completed" or metadata.status != "completed":
        failed = pageviews if pageviews.status == "failed" else metadata
        return {
            "kind": "failure",
            "operation": failed.operation,
            "attempts": failed.attempts,
            "reason": failed.error or "failed",
        }
    daily = tuple(
        DailyView(date.fromisoformat(item["date"]), item["views"])
        for item in pageviews.evidence["daily_views"]
    )
    expected = tuple(
        windows.previous_start + timedelta(days=offset)
        for offset in range((windows.current_end - windows.previous_start).days + 1)
    )
    if tuple(item.date for item in daily) != expected:
        raise ValueError("Pageviews evidence is not complete for both analysis windows")
    return {
        "kind": "alias",
        "daily_views": pageviews.evidence["daily_views"],
        "pageviews_raw": pageviews.evidence["raw"],
        "metadata": metadata.evidence,
    }


def _form_evidence(
    titles: tuple[str, ...], jobs: tuple[EvidenceJob, ...], windows: AnalysisWindows
) -> WikimediaEvidence:
    aliases = []
    artifacts = []
    for job in _jobs_by_operation(jobs, "discovery"):
        if job.status == "completed":
            artifacts.append(RawArtifact("discovery", job.subject, job.evidence["raw"]))
    for job in jobs:
        if job.status != "completed" or job.operation not in {"pageviews", "metadata"}:
            continue
        artifacts.append(RawArtifact(job.operation, job.subject, job.evidence["raw"]))
    for title in titles:
        transformed = next(
            job for job in jobs if job.operation == "transform" and job.subject == title
        )
        if transformed.status != "completed" or transformed.evidence["kind"] == "failure":
            failure = transformed.evidence if transformed.evidence else {}
            aliases.append(
                AliasEvidenceFailure(
                    AcquisitionFailure(
                        failure.get("operation", "canonicalization"),
                        title,
                        failure.get("attempts", transformed.attempts),
                        failure.get("reason", transformed.error or "failed"),
                    ),
                    (),
                )
            )
            continue
        payload = transformed.evidence
        metadata = payload["metadata"]
        daily = tuple(
            DailyView(date.fromisoformat(item["date"]), item["views"])
            for item in payload["daily_views"]
        )
        previous = sum(
            item.views for item in daily if windows.previous_start <= item.date <= windows.previous_end
        )
        current = sum(
            item.views for item in daily if windows.current_start <= item.date <= windows.current_end
        )
        alias_artifacts = ()
        aliases.append(
            AliasEvidence(
                AliasTraffic(title, previous, current, daily),
                MetadataResponse(
                    metadata["page_id"],
                    metadata["canonical_title"],
                    metadata["extract"],
                    tuple(metadata["categories"]),
                    metadata["raw"],
                ),
                alias_artifacts,
            )
        )
    return WikimediaEvidence(titles, tuple(aliases), tuple(artifacts))


def _jobs_by_operation(jobs: tuple[EvidenceJob, ...], operation: str) -> tuple[EvidenceJob, ...]:
    return tuple(job for job in jobs if job.operation == operation)
