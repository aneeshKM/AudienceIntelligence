from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import random
from threading import Event
import time

from audience_trend_miner.evidence_jobs import (
    CompletedEvidence,
    EvidenceJob,
    EvidenceJobStore,
    FailedEvidence,
    ReadyTransformation,
    TerminalEvidence,
)
from audience_trend_miner.transformation import (
    AliasEvidenceInput,
    IncompletePageviewsEvidence,
    TerminalEvidenceFailure,
    form_wikimedia_attention,
    transform_alias,
)
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
    WikimediaAttentionResult,
    WikimediaPermanentError,
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
    discovery_days = tuple(
        (windows.current_start + timedelta(days=offset)).isoformat()
        for offset in range((windows.current_end - windows.current_start).days + 1)
    )
    store.schedule_discovery(run_id, discovery_days)
    _drain(run_id, ("discovery",), windows, adapter, store, workers)
    discovery = store.results_at_barrier(run_id, ("discovery",))
    failed_discovery = [item for item in discovery if isinstance(item, FailedEvidence)]
    if failed_discovery:
        failure = failed_discovery[0]
        raise IncompleteCandidateUniverseError(
            AcquisitionFailure(
                "discovery", failure.subject, failure.attempts, failure.reason
            )
        )
    titles = tuple(
        sorted(
            {
                title
                for item in discovery
                if isinstance(item, CompletedEvidence)
                for title in item.evidence["titles"]
            }
        )
    )
    store.schedule_alias_evidence(run_id, titles)
    fetch_workers = max(1, workers // 2)
    transformations_done = Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        fetching = executor.submit(
            _drain,
            run_id,
            ("pageviews", "metadata"),
            windows,
            adapter,
            store,
            fetch_workers,
            transformations_done,
        )
        transforming = executor.submit(
            _drain_transformations,
            run_id,
            windows,
            store,
            transformations_done,
        )
        fetching.result()
        transforming.result()
    return _form_attention(
        titles,
        store.results_at_barrier(
            run_id, ("discovery", "pageviews", "metadata", "transform")
        ),
        windows,
    )


def _drain(
    run_id, operations, windows, adapter, store, workers, done: Event | None = None
) -> None:
    def work(worker_number: int) -> None:
        while (
            done is not None and not done.is_set()
        ) or not store.barrier_reached(run_id, operations):
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
    windows: AnalysisWindows,
    store: EvidenceJobStore,
    done: Event,
) -> None:
    try:
        while True:
            if store.barrier_reached(run_id, ("transform",)):
                return
            ready = store.claim_ready_transformation(
                "transformer",
                lease_seconds=60,
                run_id=run_id,
            )
            if ready is None:
                time.sleep(0.01)
                continue
            try:
                result = transform_alias(_alias_input(ready), windows)
                if isinstance(result, IncompletePageviewsEvidence):
                    store.recover_incomplete_pageviews(ready.job, result.reason)
                else:
                    store.complete(ready.job, _encode_transformation(result))
            except Exception as error:
                store.fail(
                    ready.job, str(error), terminal=ready.job.attempts >= 3
                )
    finally:
        done.set()


def _alias_input(ready: ReadyTransformation) -> AliasEvidenceInput:
    return _alias_input_from_evidence(
        ready.job.subject, ready.pageviews, ready.metadata
    )


def _alias_input_from_evidence(
    title: str, pageviews: TerminalEvidence, metadata: TerminalEvidence
) -> AliasEvidenceInput:
    pageviews_value = (
        tuple(
            DailyView(date.fromisoformat(item["date"]), item["views"])
            for item in pageviews.evidence["daily_views"]
        )
        if isinstance(pageviews, CompletedEvidence)
        else TerminalEvidenceFailure(
            "pageviews", pageviews.attempts, pageviews.reason
        )
    )
    metadata_value = (
        MetadataResponse(
            metadata.evidence["page_id"],
            metadata.evidence["canonical_title"],
            metadata.evidence["extract"],
            tuple(metadata.evidence["categories"]),
            {},
        )
        if isinstance(metadata, CompletedEvidence)
        else TerminalEvidenceFailure(
            "metadata", metadata.attempts, metadata.reason
        )
    )
    return AliasEvidenceInput(title, pageviews_value, metadata_value)


def _form_attention(
    titles: tuple[str, ...], evidence: tuple[TerminalEvidence, ...], windows: AnalysisWindows
) -> WikimediaAttentionResult:
    aliases: list[AliasEvidence | AliasEvidenceFailure] = []
    artifacts = []
    for item in evidence:
        if not isinstance(item, CompletedEvidence):
            continue
        if item.operation in {"discovery", "pageviews", "metadata"}:
            artifacts.append(
                RawArtifact(item.operation, item.subject, item.evidence["raw"])
            )
    indexed = {(item.operation, item.subject): item for item in evidence}
    for title in titles:
        transformed = indexed[("transform", title)]
        if isinstance(transformed, FailedEvidence):
            aliases.append(
                AliasEvidenceFailure(
                    AcquisitionFailure(
                        "canonicalization",
                        title,
                        transformed.attempts,
                        transformed.reason,
                    ),
                    (),
                )
            )
            continue
        payload = transformed.evidence
        if payload.get("version") == 2:
            aliases.append(_decode_transformation(title, payload))
            continue
        # Transitional decoder for transformation jobs written before this refactor.
        legacy = transform_alias(
            _alias_input_from_evidence(
                title,
                indexed[("pageviews", title)],
                indexed[("metadata", title)],
            ),
            windows,
        )
        if isinstance(legacy, IncompletePageviewsEvidence):
            raise ValueError(legacy.reason)
        aliases.append(legacy)
    result = form_wikimedia_attention(titles, tuple(aliases))
    return WikimediaAttentionResult(
        result.raw_candidate_titles,
        result.canonical_articles,
        tuple(artifacts),
        result.failures,
    )


def _encode_transformation(result: AliasEvidence | AliasEvidenceFailure) -> dict:
    if isinstance(result, AliasEvidenceFailure):
        return {
            "version": 2,
            "kind": "failure",
            "operation": result.failure.operation,
            "attempts": result.failure.attempts,
            "reason": result.failure.reason,
        }
    return {
        "version": 2,
        "kind": "alias",
        "traffic": {
            "previous_window_views": result.traffic.previous_window_views,
            "current_window_views": result.traffic.current_window_views,
            "daily_views": [
                {"date": item.date.isoformat(), "views": item.views}
                for item in result.traffic.daily_views
            ],
        },
        "metadata": {
            "page_id": result.metadata.page_id,
            "canonical_title": result.metadata.canonical_title,
            "extract": result.metadata.extract,
            "categories": list(result.metadata.categories),
        },
    }


def _decode_transformation(
    title: str, payload: dict
) -> AliasEvidence | AliasEvidenceFailure:
    if payload["kind"] == "failure":
        return AliasEvidenceFailure(
            AcquisitionFailure(
                payload["operation"], title, payload["attempts"], payload["reason"]
            ),
            (),
        )
    traffic = payload["traffic"]
    metadata = payload["metadata"]
    return AliasEvidence(
        AliasTraffic(
            title,
            traffic["previous_window_views"],
            traffic["current_window_views"],
            tuple(
                DailyView(date.fromisoformat(item["date"]), item["views"])
                for item in traffic["daily_views"]
            ),
        ),
        MetadataResponse(
            metadata["page_id"],
            metadata["canonical_title"],
            metadata["extract"],
            tuple(metadata["categories"]),
            {},
        ),
        (),
    )
