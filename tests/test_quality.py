from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from audience_trend_miner.quality import (
    evaluate_frozen_fixture,
    verify_publication_quality,
)


FIXTURE = Path(__file__).with_name("fixtures") / "v1_quality_evaluation.json"


class FrozenEvaluationTest(unittest.TestCase):
    def test_v1_fixture_meets_all_frozen_quality_gates(self) -> None:
        result = evaluate_frozen_fixture(json.loads(FIXTURE.read_text()))

        self.assertEqual(result.commercial_relevance, 0.8)
        self.assertEqual(result.approved_top_five, 4)
        self.assertTrue(result.passed)

    def test_rejects_an_accepted_unsafe_or_incoherent_cluster(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        fixture["clusters"][0]["has_unrelated_member"] = True

        with self.assertRaisesRegex(ValueError, "unrelated member"):
            evaluate_frozen_fixture(fixture)


class PublicationQualityTest(unittest.TestCase):
    def test_verifies_exact_size_and_complete_final_membership_lineage(self) -> None:
        audit = {
            "canonical_articles": [
                {"page_id": 1, "current_window_views": 300, "aliases": [{"raw_title": "Alias_A"}]},
                {"page_id": 2, "current_window_views": 200, "aliases": [{"raw_title": "Alias_B"}]},
                {"page_id": 3, "current_window_views": 500, "aliases": [{"raw_title": "Alias_C"}]},
            ],
            "qualified_signals": [{"page_id": page_id} for page_id in (1, 2, 3)],
            "article_classifications": [
                {"page_id": page_id, "accepted": True} for page_id in (1, 2, 3)
            ],
            "candidate_clustering": {"components": [
                {"component_id": 1, "page_ids": [1, 2]},
                {"component_id": 2, "page_ids": [3]},
            ]},
            "cluster_refinement": {"accepted": [
                {"source_component_id": 1, "page_ids": [1, 2], "safety": {"safe": True}},
                {"source_component_id": 2, "page_ids": [3], "safety": {"safe": True}},
            ]},
            "portfolio_calculations": [
                {"source_component_id": 1, "page_ids": [1, 2], "size_basis_points": 5000, "estimated_size_index": 50.0},
                {"source_component_id": 2, "page_ids": [3], "size_basis_points": 5000, "estimated_size_index": 50.0},
            ],
        }
        portfolio = {"audiences": [
            {"source_component_id": 1, "page_ids": [1, 2], "estimated_size_index": 50.0},
            {"source_component_id": 2, "page_ids": [3], "estimated_size_index": 50.0},
        ]}

        result = verify_publication_quality(audit, portfolio)

        self.assertEqual(result.traced_page_ids, (1, 2, 3))
        self.assertEqual(result.total_size_basis_points, 10_000)

        broken = copy.deepcopy(audit)
        broken["canonical_articles"][0]["aliases"] = []
        with self.assertRaisesRegex(ValueError, "alias lineage"):
            verify_publication_quality(broken, portfolio)
