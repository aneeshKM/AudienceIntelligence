from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from audience_trend_miner.v2.trend_portfolio.traffic import ClusterTraffic


MINIMUM_CURRENT_VIEWS = 100_000
MAXIMUM_PORTFOLIO_SIZE = 10
MAXIMUM_CHANGE_OCTAVES = 10


@dataclass(frozen=True)
class PortfolioAudience:
    """One qualified cluster with its deterministic ranking fact."""

    traffic: ClusterTraffic
    impact_score: float


@dataclass(frozen=True)
class AudiencePortfolio:
    """The bounded collection of robust, qualified audience trends."""

    audiences: tuple[PortfolioAudience, ...]


def qualify_and_rank_portfolio(
    clusters: Iterable[ClusterTraffic],
) -> AudiencePortfolio:
    """Apply the V2 scale gate and rank robust directions together."""
    qualified = (
        PortfolioAudience(traffic=cluster, impact_score=_impact_score(cluster))
        for cluster in clusters
        if cluster.direction != "uncertain_direction"
        and cluster.current.minimum >= MINIMUM_CURRENT_VIEWS
    )
    ranked = sorted(
        qualified,
        key=lambda audience: (-audience.impact_score, audience.traffic.cluster_id),
    )
    return AudiencePortfolio(tuple(ranked[:MAXIMUM_PORTFOLIO_SIZE]))


def _impact_score(cluster: ClusterTraffic) -> float:
    previous_views = cluster.previous.seven_day_equivalent
    current_views = cluster.current.seven_day_equivalent
    scale = math.log1p(max(previous_views, current_views))
    change = abs(math.log2((current_views + 1) / (previous_views + 1)))
    return scale * min(change, MAXIMUM_CHANGE_OCTAVES)
