from __future__ import annotations

from dataclasses import dataclass
import math

from audience_trend_miner.wikimedia import CanonicalArticle


MINIMUM_CURRENT_VIEWS = 100_000
MAX_GROWTH_LOG2 = 10.0
TECHNICAL_NAMESPACES = frozenset(
    {
        "category",
        "draft",
        "file",
        "help",
        "media",
        "mediawiki",
        "module",
        "portal",
        "special",
        "template",
        "timedtext",
        "user",
        "user talk",
        "wikipedia",
    }
)


@dataclass(frozen=True)
class QualificationGates:
    minimum_traffic: bool
    growth: bool
    positive_score: bool


@dataclass(frozen=True)
class TrendDecision:
    article: CanonicalArticle
    score: float
    gates: QualificationGates
    included: bool
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class TrendQualificationResult:
    qualified: tuple[TrendDecision, ...]
    rejected_noise: tuple[TrendDecision, ...]
    decisions: tuple[TrendDecision, ...]


def trend_score(current_views: int, previous_views: int) -> float:
    """Calculate the specified scale-and-acceleration score safely."""
    growth = math.log2((current_views + 1) / (previous_views + 1))
    return math.log(current_views + 1) * min(growth, MAX_GROWTH_LOG2)


def qualify_trends(
    articles: tuple[CanonicalArticle, ...],
) -> TrendQualificationResult:
    """Apply auditable gates and conservative deterministic noise filtering."""
    decisions: list[TrendDecision] = []
    for candidate in articles:
        score = trend_score(
            candidate.current_window_views,
            candidate.previous_window_views,
        )
        gates = QualificationGates(
            minimum_traffic=candidate.current_window_views >= MINIMUM_CURRENT_VIEWS,
            growth=(
                candidate.current_window_views > candidate.previous_window_views
            ),
            positive_score=score > 0,
        )
        exclusion_reason = deterministic_exclusion_reason(candidate.canonical_title)
        decisions.append(
            TrendDecision(
                article=candidate,
                score=score,
                gates=gates,
                included=(
                    exclusion_reason is None
                    and gates.minimum_traffic
                    and gates.growth
                    and gates.positive_score
                ),
                exclusion_reason=exclusion_reason,
            )
        )

    qualified = tuple(
        sorted(
            (decision for decision in decisions if decision.included),
            key=lambda decision: (-decision.score, decision.article.page_id),
        )
    )
    rejected_noise = tuple(
        decision for decision in decisions if decision.exclusion_reason is not None
    )
    return TrendQualificationResult(
        qualified=qualified,
        rejected_noise=rejected_noise,
        decisions=tuple(decisions),
    )


def deterministic_exclusion_reason(title: str) -> str | None:
    normalized = title.replace("_", " ").strip()
    if normalized.casefold() == "main page":
        return "main_page"
    namespace, separator, _ = normalized.partition(":")
    if separator and namespace.casefold() in TECHNICAL_NAMESPACES:
        return f"technical_namespace:{namespace.casefold()}"
    return None
