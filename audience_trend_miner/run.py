from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from audience_trend_miner.classification import (
    ArticleClassificationResult,
    FixtureStructuredGenerator,
    GroqStructuredGenerator,
    StructuredGenerator,
    classify_articles,
)
from audience_trend_miner.configuration import EffectiveRunConfiguration, load_run_configuration
from audience_trend_miner.publication import PublicationInput, publish_run
from audience_trend_miner.evidence_jobs import EvidenceJobStore
from audience_trend_miner.resumable_wikimedia import acquire_resumable_wikimedia_attention
from audience_trend_miner.trends import qualify_trends
from audience_trend_miner.wikimedia import (
    AnalysisWindows,
    FixtureWikimediaAdapter,
    HttpWikimediaAdapter,
    WikimediaAdapter,
    WikimediaAttentionResult,
)


def execute_run(
    as_of_argument: date | None,
    output_directory: Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Sequence attention acquisition, qualification, and Run Publication."""
    configuration = load_run_configuration()
    started_at = datetime.now(timezone.utc)
    as_of = as_of_argument or started_at.date()
    current_end = as_of - timedelta(days=2)
    current_start = current_end - timedelta(days=6)
    previous_end = current_start - timedelta(days=1)
    windows = AnalysisWindows(
        previous_start=previous_end - timedelta(days=6),
        previous_end=previous_end,
        current_start=current_start,
        current_end=current_end,
    )

    attention = WikimediaAttentionResult((), (), ())
    adapter = _selected_wikimedia_adapter(configuration)
    effective_run_id = run_id or started_at.strftime("%Y%m%dT%H%M%S%fZ")
    recorded_run_facts = {
        **configuration.safe_provenance(),
        "as_of": as_of.isoformat(),
        "previous_window": (
            f"{windows.previous_start.isoformat()}/{windows.previous_end.isoformat()}"
        ),
        "current_window": (
            f"{windows.current_start.isoformat()}/{windows.current_end.isoformat()}"
        ),
    }
    job_store = EvidenceJobStore(configuration.database_url)
    job_store.migrate()
    job_store.ensure_run(effective_run_id, recorded_run_facts)
    if adapter is not None:
        attention = acquire_resumable_wikimedia_attention(
            effective_run_id,
            windows,
            adapter,
            job_store,
            configuration=recorded_run_facts,
        )
    qualification = qualify_trends(attention.canonical_articles)
    classification = ArticleClassificationResult((), (), ())
    if qualification.qualified:
        generator = _selected_structured_generator(configuration)
        classification = classify_articles(
            tuple(decision.article for decision in qualification.qualified),
            generator,
            sleep=(lambda _: None)
            if isinstance(generator, FixtureStructuredGenerator)
            else time.sleep,
        )

    publication_path = str((output_directory / effective_run_id).resolve())
    job_store.reserve_publication_path(effective_run_id, publication_path)
    published = publish_run(
        PublicationInput(
            output_root=output_directory,
            started_at=started_at,
            as_of_argument=as_of_argument,
            as_of=as_of,
            windows=windows,
            attention=attention,
            qualification=qualification,
            classification=classification,
            configuration=configuration.safe_provenance(),
            run_id=effective_run_id,
        )
    )
    job_store.mark_publication_complete(effective_run_id, str(published.resolve()))
    return published


def _selected_wikimedia_adapter(
    configuration: EffectiveRunConfiguration,
) -> WikimediaAdapter | None:
    if configuration.wikimedia_fixture:
        return FixtureWikimediaAdapter.from_file(configuration.wikimedia_fixture)
    if configuration.wikimedia_base_url == "":
        return None
    return (
        HttpWikimediaAdapter(rest_base_url=configuration.wikimedia_base_url)
        if configuration.wikimedia_base_url
        else HttpWikimediaAdapter()
    )


def _selected_structured_generator(
    configuration: EffectiveRunConfiguration,
) -> StructuredGenerator:
    if configuration.classification_fixture:
        return FixtureStructuredGenerator.from_file(configuration.classification_fixture)
    return GroqStructuredGenerator(
        api_key=configuration.groq_api_key,
        model=configuration.model,
    )
