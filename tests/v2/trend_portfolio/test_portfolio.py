from __future__ import annotations

import math
import unittest
from typing import cast

from audience_trend_miner.v2.trend_portfolio import (
    ClusterTraffic,
    Direction,
    WindowTraffic,
    qualify_and_rank_portfolio,
)


def _window(*, views: float, minimum: float | None = None) -> WindowTraffic:
    resolved_minimum = views if minimum is None else minimum
    return WindowTraffic(
        observed_total=int(views),
        observed_page_days=7,
        successful_days=7,
        conservative_observed_minimum=int(resolved_minimum),
        conservative_observed_maximum=int(views),
        seven_day_equivalent=views,
        minimum=resolved_minimum,
        maximum=views,
    )


def _trend(
    cluster_id: str,
    *,
    previous_views: float,
    current_views: float,
    current_minimum: float | None = None,
    direction: str = "robust_growth",
) -> ClusterTraffic:
    return ClusterTraffic(
        cluster_id=cluster_id,
        source_preliminary_cluster_id=f"source-{cluster_id}",
        name=cluster_id,
        rationale=f"{cluster_id} rationale.",
        member_page_ids=(1, 2),
        previous=_window(views=previous_views),
        current=_window(views=current_views, minimum=current_minimum),
        direction=cast(Direction, direction),
    )


class AudiencePortfolioTest(unittest.TestCase):
    def test_qualifies_robust_directions_and_ranks_by_symmetric_impact(self) -> None:
        growing = _trend(
            "growing",
            previous_views=100_000,
            current_views=200_000,
            current_minimum=100_000,
        )
        shrinking = _trend(
            "shrinking",
            previous_views=400_000,
            current_views=200_000,
            current_minimum=100_000,
            direction="robust_shrinking",
        )
        uncertain = _trend(
            "uncertain",
            previous_views=1,
            current_views=1_000_000,
            direction="uncertain_direction",
        )
        below_scale = _trend(
            "below-scale",
            previous_views=1,
            current_views=1_000_000,
            current_minimum=99_999,
        )

        malformed = _trend(
            "malformed",
            previous_views=1,
            current_views=1_000_000,
            direction="robust-ish",
        )

        qualification = qualify_and_rank_portfolio(
            (growing, uncertain, below_scale, shrinking, malformed)
        )
        portfolio = qualification.portfolio

        self.assertEqual(
            [
                trend.final_cluster_traffic.cluster_id
                for trend in portfolio.audience_trends
            ],
            ["shrinking", "growing"],
        )
        self.assertAlmostEqual(
            portfolio.audience_trends[0].impact_score,
            math.log(400_001) * abs(math.log2(200_001 / 400_001)),
        )
        self.assertAlmostEqual(
            portfolio.audience_trends[1].impact_score,
            math.log(200_001) * abs(math.log2(200_001 / 100_001)),
        )
        self.assertEqual(
            qualification.audit_cluster_traffic,
            (growing, uncertain, below_scale, shrinking, malformed),
        )

    def test_caps_change_and_selects_top_ten_with_stable_ties(self) -> None:
        tied = [
            _trend(
                f"cluster-{index:02d}",
                previous_views=0,
                current_views=1_048_575,
            )
            for index in range(12, 0, -1)
        ]

        portfolio = qualify_and_rank_portfolio(tied).portfolio

        self.assertEqual(
            [
                trend.final_cluster_traffic.cluster_id
                for trend in portfolio.audience_trends
            ],
            [f"cluster-{index:02d}" for index in range(1, 11)],
        )
        self.assertEqual(
            portfolio.audience_trends[0].impact_score,
            math.log(1_048_576) * 10,
        )

    def test_no_qualifying_cluster_produces_empty_portfolio(self) -> None:
        qualification = qualify_and_rank_portfolio(
            (
                _trend(
                    "uncertain",
                    previous_views=100_000,
                    current_views=100_000,
                    direction="uncertain_direction",
                ),
                _trend(
                    "too-small",
                    previous_views=1,
                    current_views=99_999,
                ),
            )
        )

        self.assertEqual(qualification.portfolio.audience_trends, ())


if __name__ == "__main__":
    unittest.main()
