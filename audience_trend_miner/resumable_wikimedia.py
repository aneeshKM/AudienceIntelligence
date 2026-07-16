from __future__ import annotations

from datetime import date, timedelta

from audience_trend_miner.evidence_jobs import (
    CompletedEvidence,
    EvidenceJob,
    EvidenceJobExecution,
    EvidenceJobStore,
    FailedEvidence,
    TerminalEvidence,
)
from audience_trend_miner.transformation import (
    AliasEvidenceInput,
    TerminalEvidenceFailure,
    TerminalWikimediaEvidence,
    transform_wikimedia_attention,
)
from audience_trend_miner.wikimedia import (
    AcquisitionFailure,
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
    """Preferred interface: fetch terminal evidence, then transform it in memory."""
    evidence = fetch_terminal_wikimedia_evidence(
        run_id,
        windows,
        adapter,
        store,
        workers=workers,
        configuration=configuration,
    )
    return transform_wikimedia_attention(evidence, windows)


def fetch_terminal_wikimedia_evidence(
    run_id: str,
    windows: AnalysisWindows,
    adapter: WikimediaAdapter,
    store: EvidenceJobStore,
    *,
    workers: int = 4,
    configuration: dict[str, str] | None = None,
) -> TerminalWikimediaEvidence:
    """Fetch and persist complete terminal evidence for one Candidate Universe."""
    run_facts = {
        **(configuration or {}),
        "evidence_max_attempts": "3",
        "evidence_lease_seconds": "60",
        "evidence_workers": str(workers),
        "evidence_backoff": "exponential-jitter-v1",
    }
    store.ensure_run(run_id, run_facts)
    execution = EvidenceJobExecution(store)
    discovery_days = tuple(
        (windows.current_start + timedelta(days=offset)).isoformat()
        for offset in range((windows.current_end - windows.current_start).days + 1)
    )
    store.schedule_discovery(run_id, discovery_days)
    execution.drain(
        run_id,
        ("discovery",),
        lambda job: _fetch(job, windows, adapter),
        workers=workers,
        is_terminal_error=lambda error: isinstance(error, WikimediaPermanentError),
    )
    discovery = store.results_at_barrier(run_id, ("discovery",))
    failed_discovery = tuple(
        item for item in discovery if isinstance(item, FailedEvidence)
    )
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
    execution.drain(
        run_id,
        ("pageviews", "metadata"),
        lambda job: _fetch(job, windows, adapter),
        workers=workers,
        is_terminal_error=lambda error: isinstance(error, WikimediaPermanentError),
    )
    evidence = store.results_at_barrier(
        run_id, ("discovery", "pageviews", "metadata")
    )
    return _terminal_projection(titles, evidence)


def _fetch(
    job: EvidenceJob,
    windows: AnalysisWindows,
    adapter: WikimediaAdapter,
) -> dict[str, object]:
    if job.operation == "discovery":
        response = adapter.daily_top_pages(date.fromisoformat(job.subject))
        return {"titles": list(response.titles), "raw": response.raw}
    if job.operation == "pageviews":
        response = adapter.article_pageviews(
            job.subject, windows.previous_start, windows.current_end
        )
        _validate_pageviews(response.daily_views, windows)
        return {
            "daily_views": [
                {"date": item.date.isoformat(), "views": item.views}
                for item in response.daily_views
            ],
            "raw": response.raw,
        }
    if job.operation == "metadata":
        response = adapter.article_metadata(job.subject)
        return {
            "page_id": response.page_id,
            "canonical_title": response.canonical_title,
            "extract": response.extract,
            "categories": list(response.categories),
            "raw": response.raw,
        }
    raise ValueError(f"unsupported Evidence Job operation: {job.operation}")


def _validate_pageviews(
    daily_views: tuple[DailyView, ...], windows: AnalysisWindows
) -> None:
    expected_dates = tuple(
        windows.previous_start + timedelta(days=offset)
        for offset in range((windows.current_end - windows.previous_start).days + 1)
    )
    if tuple(item.date for item in daily_views) != expected_dates:
        raise ValueError(
            "Pageviews evidence must contain complete dated observations "
            f"from {windows.previous_start} through {windows.current_end}"
        )


def _terminal_projection(
    titles: tuple[str, ...], evidence: tuple[TerminalEvidence, ...]
) -> TerminalWikimediaEvidence:
    indexed = {(item.operation, item.subject): item for item in evidence}
    aliases = tuple(
        _alias_input(
            title,
            indexed[("pageviews", title)],
            indexed[("metadata", title)],
        )
        for title in titles
    )
    artifacts = tuple(
        RawArtifact(item.operation, item.subject, item.evidence["raw"])
        for item in evidence
        if isinstance(item, CompletedEvidence)
        and item.operation in {"discovery", "pageviews", "metadata"}
    )
    return TerminalWikimediaEvidence(titles, aliases, artifacts)


def _alias_input(
    title: str,
    pageviews: TerminalEvidence,
    metadata: TerminalEvidence,
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
        else TerminalEvidenceFailure("metadata", metadata.attempts, metadata.reason)
    )
    return AliasEvidenceInput(title, pageviews_value, metadata_value)
