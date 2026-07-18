from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from audience_trend_miner.v2.trend_portfolio.traffic import ClusterTraffic


MAXIMUM_PORTFOLIO_SIZE = 10
MAXIMUM_CHANGE_OCTAVES = 10


# Combine a terminal cluster with its measured direction and impact.
@dataclass(frozen=True)
class AudienceTrend:
    """One accepted Final Audience Cluster and its deterministic ranking fact."""

    final_cluster_traffic: ClusterTraffic
    impact_score: float


# Hold the ranked trends that pass the portfolio qualification gate.
@dataclass(frozen=True)
class AudiencePortfolio:
    """The bounded collection of robust, qualified audience trends."""

    audience_trends: tuple[AudienceTrend, ...]


# Return the ranked portfolio plus audited exclusions.
@dataclass(frozen=True)
class PortfolioQualification:
    """Selected product data alongside complete traffic retained for audit."""

    portfolio: AudiencePortfolio
    audit_cluster_traffic: tuple[ClusterTraffic, ...]


# Apply the V2 scale gate and rank robust directions together.
def qualify_and_rank_portfolio(
    final_cluster_traffic: Iterable[ClusterTraffic],
) -> PortfolioQualification:
    """Apply the V2 scale gate and rank robust directions together."""
    # Retain every traffic record for audit even when its direction is not product-ready.
    audit_cluster_traffic = tuple(final_cluster_traffic)
    qualified = (
        AudienceTrend(
            final_cluster_traffic=traffic,
            impact_score=_impact_score(traffic),
        )
        for traffic in audit_cluster_traffic
        if traffic.direction
        in {"robust_growth", "robust_shrinking", "sudden_growth"}
    )
    # Impact combines attention scale and magnitude of change; cluster ID breaks ties.
    ranked = sorted(
        qualified,
        key=lambda trend: (
            -trend.impact_score,
            trend.final_cluster_traffic.cluster_id,
        ),
    )
    return PortfolioQualification(
        portfolio=AudiencePortfolio(tuple(ranked[:MAXIMUM_PORTFOLIO_SIZE])),
        audit_cluster_traffic=audit_cluster_traffic,
    )


# Calculate the symmetric impact score used for ranking.
def _impact_score(final_cluster_traffic: ClusterTraffic) -> float:
    # log1p keeps zero traffic defined and prevents large audiences dominating linearly.
    previous_traffic = final_cluster_traffic.previous.seven_day_equivalent
    current_traffic = final_cluster_traffic.current.seven_day_equivalent
    scale = math.log1p(max(previous_traffic, current_traffic))
    # Symmetric log-ratio scoring treats equal growth and shrinkage magnitudes alike.
    change = abs(
        math.log2((current_traffic + 1) / (previous_traffic + 1))
    )
    return scale * min(change, MAXIMUM_CHANGE_OCTAVES)
