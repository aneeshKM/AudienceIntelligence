from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from audience_trend_miner.wikimedia import (
    AcquisitionFailure,
    AliasEvidence,
    AliasEvidenceFailure,
    AliasTraffic,
    AnalysisWindows,
    CanonicalArticle,
    DailyView,
    MetadataResponse,
    RawArtifact,
    WikimediaAttentionResult,
)


@dataclass(frozen=True)
class TerminalEvidenceFailure:
    operation: str
    attempts: int
    reason: str


@dataclass(frozen=True)
class AliasEvidenceInput:
    raw_title: str
    pageviews: tuple[DailyView, ...] | TerminalEvidenceFailure
    metadata: MetadataResponse | TerminalEvidenceFailure


@dataclass(frozen=True)
class IncompletePageviewsEvidence:
    raw_title: str
    reason: str


@dataclass(frozen=True)
class TerminalWikimediaEvidence:
    """Complete, immutable fetching outcome for one Candidate Universe."""

    raw_candidate_titles: tuple[str, ...]
    aliases: tuple[AliasEvidenceInput, ...]
    raw_artifacts: tuple[RawArtifact, ...]


type AliasTransformation = (
    AliasEvidence | AliasEvidenceFailure | IncompletePageviewsEvidence
)


def transform_alias(
    evidence: AliasEvidenceInput, windows: AnalysisWindows
) -> AliasTransformation:
    """Transform one alias without depending on storage or fetching details."""
    if isinstance(evidence.pageviews, TerminalEvidenceFailure):
        return _failed_alias(evidence.raw_title, evidence.pageviews)
    if isinstance(evidence.metadata, TerminalEvidenceFailure):
        return _failed_alias(evidence.raw_title, evidence.metadata)
    expected_dates = tuple(
        windows.previous_start + timedelta(days=offset)
        for offset in range((windows.current_end - windows.previous_start).days + 1)
    )
    if tuple(item.date for item in evidence.pageviews) != expected_dates:
        return IncompletePageviewsEvidence(
            evidence.raw_title,
            "Pageviews evidence must contain complete dated observations "
            f"from {windows.previous_start} through {windows.current_end}",
        )
    previous_views = sum(
        item.views
        for item in evidence.pageviews
        if windows.previous_start <= item.date <= windows.previous_end
    )
    current_views = sum(
        item.views
        for item in evidence.pageviews
        if windows.current_start <= item.date <= windows.current_end
    )
    return AliasEvidence(
        AliasTraffic(
            evidence.raw_title, previous_views, current_views, evidence.pageviews
        ),
        evidence.metadata,
        (),
    )


def form_wikimedia_attention(
    raw_candidate_titles: tuple[str, ...],
    aliases: tuple[AliasEvidence | AliasEvidenceFailure, ...],
) -> WikimediaAttentionResult:
    """Form Canonical Articles after every Candidate Universe alias is terminal."""
    successful = tuple(item for item in aliases if isinstance(item, AliasEvidence))
    failures = [
        item.failure for item in aliases if isinstance(item, AliasEvidenceFailure)
    ]
    grouped: dict[int, list[AliasEvidence]] = {}
    for alias_evidence in successful:
        grouped.setdefault(alias_evidence.metadata.page_id, []).append(alias_evidence)
    articles: list[CanonicalArticle] = []
    for page_id in sorted(grouped):
        page_evidence = grouped[page_id]
        titles = {item.metadata.canonical_title for item in page_evidence}
        if len(titles) != 1:
            failures.append(
                AcquisitionFailure(
                    "canonicalization",
                    str(page_id),
                    1,
                    "aliases returned conflicting canonical titles: "
                    + ", ".join(sorted(titles)),
                )
            )
            continue
        metadata = page_evidence[0].metadata
        traffic = tuple(item.traffic for item in page_evidence)
        articles.append(
            CanonicalArticle(
                page_id,
                metadata.canonical_title,
                metadata.extract,
                metadata.categories,
                sum(alias.previous_window_views for alias in traffic),
                sum(alias.current_window_views for alias in traffic),
                traffic,
            )
        )
    return WikimediaAttentionResult(
        raw_candidate_titles, tuple(articles), (), tuple(failures)
    )


def transform_wikimedia_attention(
    evidence: TerminalWikimediaEvidence,
    windows: AnalysisWindows,
) -> WikimediaAttentionResult:
    """Synchronously transform terminal fetched evidence without database access."""
    aliases: list[AliasEvidence | AliasEvidenceFailure] = []
    for alias_input in evidence.aliases:
        transformed = transform_alias(alias_input, windows)
        if isinstance(transformed, IncompletePageviewsEvidence):
            raise ValueError(
                "terminal Pageviews evidence is incomplete: " + transformed.reason
            )
        aliases.append(transformed)
    result = form_wikimedia_attention(evidence.raw_candidate_titles, tuple(aliases))
    return WikimediaAttentionResult(
        result.raw_candidate_titles,
        result.canonical_articles,
        evidence.raw_artifacts,
        result.failures,
    )


def _failed_alias(
    raw_title: str, failure: TerminalEvidenceFailure
) -> AliasEvidenceFailure:
    return AliasEvidenceFailure(
        AcquisitionFailure(failure.operation, raw_title, failure.attempts, failure.reason),
        (),
    )
