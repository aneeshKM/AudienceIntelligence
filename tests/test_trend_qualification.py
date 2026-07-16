from __future__ import annotations

import math
import unittest

from audience_trend_miner.trends import qualify_trends, trend_score
from audience_trend_miner.wikimedia import AliasTraffic, CanonicalArticle


def article(
    title: str,
    previous: int,
    current: int,
    *,
    page_id: int = 1,
) -> CanonicalArticle:
    return CanonicalArticle(
        page_id=page_id,
        canonical_title=title,
        extract="",
        categories=(),
        previous_window_views=previous,
        current_window_views=current,
        aliases=(
            AliasTraffic(
                raw_title=title.replace(" ", "_"),
                previous_window_views=previous,
                current_window_views=current,
                daily_views=(),
            ),
        ),
    )


class TrendQualificationTest(unittest.TestCase):
    def test_score_uses_plus_one_baselines_and_caps_base_two_growth_at_ten(self) -> None:
        self.assertEqual(trend_score(1023, 0), math.log(1024) * 10)
        self.assertEqual(trend_score(2047, 0), math.log(2048) * 10)
        self.assertAlmostEqual(
            trend_score(200_001, 100_000),
            math.log(200_002) * math.log2(200_002 / 100_001),
        )

    def test_exhausts_traffic_growth_and_positive_score_boundaries(self) -> None:
        result = qualify_trends(
            (
                article("Below traffic", 1, 99_999, page_id=1),
                article("At traffic", 99_999, 100_000, page_id=2),
                article("No growth", 100_000, 100_000, page_id=3),
                article("Declining", 100_001, 100_000, page_id=4),
                article("Zero baseline", 0, 100_000, page_id=5),
            )
        )

        self.assertEqual(
            [candidate.article.canonical_title for candidate in result.qualified],
            ["Zero baseline", "At traffic"],
        )
        decisions = {item.article.canonical_title: item for item in result.decisions}
        self.assertFalse(decisions["Below traffic"].gates.minimum_traffic)
        self.assertFalse(decisions["No growth"].gates.growth)
        self.assertFalse(decisions["No growth"].gates.positive_score)
        self.assertFalse(decisions["Declining"].gates.growth)
        self.assertLess(decisions["Declining"].score, 0)

    def test_filters_only_explicit_navigation_noise_and_keeps_lists_and_indexes(self) -> None:
        result = qualify_trends(
            (
                article("Main Page", 10, 200_000, page_id=1),
                article("Special:Search", 10, 200_000, page_id=2),
                article("Wikipedia:Contents", 10, 200_000, page_id=3),
                article("List of films of 2026", 10, 200_000, page_id=4),
                article("Index of robotics articles", 10, 200_000, page_id=5),
            )
        )

        self.assertEqual(
            [candidate.article.canonical_title for candidate in result.qualified],
            ["List of films of 2026", "Index of robotics articles"],
        )
        rejected = {item.article.canonical_title: item for item in result.rejected_noise}
        self.assertEqual(rejected["Main Page"].exclusion_reason, "main_page")
        self.assertEqual(
            rejected["Special:Search"].exclusion_reason,
            "technical_namespace:special",
        )

    def test_equal_scores_are_ordered_by_page_id(self) -> None:
        result = qualify_trends(
            (
                article("Later page", 50_000, 100_000, page_id=20),
                article("Earlier page", 50_000, 100_000, page_id=10),
            )
        )

        self.assertEqual(
            [candidate.article.page_id for candidate in result.qualified],
            [10, 20],
        )


if __name__ == "__main__":
    unittest.main()
