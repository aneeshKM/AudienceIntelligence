from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from audience_trend_miner.publication import PublicationInput, publish_run
from audience_trend_miner.trends import qualify_trends
from audience_trend_miner.wikimedia import (
    AnalysisWindows,
    FixtureWikimediaAdapter,
    HttpWikimediaAdapter,
    WikimediaAdapter,
    WikimediaAttentionResult,
    acquire_wikimedia_attention,
)


def execute_run(as_of_argument: date | None, output_directory: Path) -> Path:
    """Sequence attention acquisition, qualification, and Run Publication."""
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
    adapter = _selected_wikimedia_adapter()
    if adapter is not None:
        attention = acquire_wikimedia_attention(windows, adapter)
    qualification = qualify_trends(attention.canonical_articles)

    return publish_run(
        PublicationInput(
            output_root=output_directory,
            started_at=started_at,
            as_of_argument=as_of_argument,
            as_of=as_of,
            windows=windows,
            attention=attention,
            qualification=qualification,
        )
    )


def _selected_wikimedia_adapter() -> WikimediaAdapter | None:
    fixture_path = os.environ.get("AUDIENCE_TREND_MINER_WIKIMEDIA_FIXTURE")
    rest_base_url = os.environ.get("AUDIENCE_TREND_MINER_WIKIMEDIA_BASE_URL")
    if fixture_path:
        return FixtureWikimediaAdapter.from_file(Path(fixture_path))
    if rest_base_url == "":
        return None
    return (
        HttpWikimediaAdapter(rest_base_url=rest_base_url)
        if rest_base_url
        else HttpWikimediaAdapter()
    )
