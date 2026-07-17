from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
import tempfile

from audience_trend_miner.classification import classify_articles
from audience_trend_miner.clustering import FrozenEmbeddingAdapter, form_candidate_clusters
from audience_trend_miner.portfolio import build_portfolio
from audience_trend_miner.publication import publish_run
from audience_trend_miner.refinement import refine_candidate_clusters
from audience_trend_miner.trends import qualify_trends
from audience_trend_miner.wikimedia import WikimediaAttentionResult
from tests.test_article_classification import ScriptedGenerator, judgment
from tests.test_publication import (
    _qualified_article,
    publication_input,
    refinement_audience,
    refinement_decision,
)

from audience_trend_miner.quality import (
    evaluate_frozen_fixture,
    verify_publication_quality,
)


FIXTURE = Path(__file__).with_name("fixtures") / "v1_quality_evaluation.json"


class FrozenEvaluationTest(unittest.TestCase):
    def test_v1_fixture_meets_all_frozen_quality_gates(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        articles = tuple(
            _qualified_article(page_id, f"Frozen article {page_id}")
            for page_id in range(1, 11)
        )
        qualification = qualify_trends(articles)
        classification = classify_articles(
            articles,
            ScriptedGenerator(*(judgment() for _ in articles)),
            sleep=lambda _: None,
        )
        embeddings = tuple(
            tuple(1.0 if dimension == (page_id - 1) // 2 else 0.0 for dimension in range(5))
            for page_id in range(1, 11)
        )
        clustering = form_candidate_clusters(articles, FrozenEmbeddingAdapter(embeddings))
        refinement_responses = []
        for cluster in fixture["clusters"]:
            component = clustering.components[cluster["component_id"] - 1]
            refinement_responses.extend((
                refinement_decision(
                    "validate",
                    [refinement_audience(cluster["name"], *component.page_ids)],
                    [],
                    alternative_matches=[],
                ),
                {"materially_centered_on_tragedy": False,
                 "materially_centered_on_violent_crime": False,
                 "rationale": "Frozen editor-labelled safe cluster."},
            ))
        refinement = refine_candidate_clusters(
            clustering.components,
            articles,
            ScriptedGenerator(*refinement_responses),
            sleep=lambda _: None,
        )
        portfolio_result = build_portfolio(
            refinement,
            articles,
            ScriptedGenerator(*(_portfolio_assessment(review["audience_name"])
                                for review in fixture["top_audience_editor_reviews"])),
            sleep=lambda _: None,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            published = publish_run(publication_input(
                Path(temporary_directory),
                attention=WikimediaAttentionResult(
                    tuple(article.aliases[0].raw_title for article in articles),
                    articles,
                    (),
                ),
                qualification=qualification,
                classification=classification,
                clustering=clustering,
                refinement=refinement,
                portfolio=portfolio_result,
            ))
            audit = json.loads((published / "audit.json").read_text())
            portfolio = json.loads((published / "portfolio.json").read_text())
        result = evaluate_frozen_fixture(fixture, audit, portfolio)

        self.assertEqual(result.commercial_relevance, 0.8)
        self.assertEqual(result.approved_top_five, 4)
        self.assertTrue(result.passed)

    def test_rejects_an_accepted_unsafe_or_incoherent_cluster(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        fixture["clusters"][0]["has_unrelated_member"] = True
        audit = {"article_classifications": [{"page_id": page_id, "accepted": True} for page_id in range(1, 11)],
                 "cluster_refinement": {"accepted": [{"source_component_id": component_id} for component_id in range(1, 6)]}}
        portfolio = {"audiences": [{"name": item["audience_name"]} for item in fixture["top_audience_editor_reviews"]]}

        with self.assertRaisesRegex(ValueError, "unrelated member"):
            evaluate_frozen_fixture(fixture, audit, portfolio)


def _portfolio_assessment(name: str) -> dict[str, object]:
    return {
        "name": name,
        "description": "Frozen traffic increased, suggesting growing consumer interest.",
        "purchase_intent": 2,
        "transaction_value": 2,
        "category_breadth": 2,
        "brand_safety": 3,
        "brand_categories": ["Consumer goods"],
        "rationale": "Frozen editor-labelled commercial audience.",
        "name_is_targetable_group": True,
        "causal_claims_are_hypotheses": True,
        "rationale_avoids_wealth_inference": True,
    }


class PublicationQualityTest(unittest.TestCase):
    def test_verifies_exact_size_and_complete_final_membership_lineage(self) -> None:
        audit = {
            "raw_candidate_titles": ["Alias_A", "Alias_B", "Alias_C"],
            "failures": [],
            "canonical_articles": [
                {"page_id": 1, "current_window_views": 300, "aliases": [{"raw_title": "Alias_A"}]},
                {"page_id": 2, "current_window_views": 200, "aliases": [{"raw_title": "Alias_B"}]},
                {"page_id": 3, "current_window_views": 500, "aliases": [{"raw_title": "Alias_C"}]},
            ],
            "qualified_signals": [{"page_id": page_id} for page_id in (1, 2, 3)],
            "decisions": [{"page_id": page_id, "outcome": "classified_signal"} for page_id in (1, 2, 3)],
            "article_classifications": [
                {"page_id": page_id, "accepted": True} for page_id in (1, 2, 3)
            ],
            "candidate_clustering": {"components": [
                {"component_id": 1, "page_ids": [1, 2], "is_candidate_cluster": True},
                {"component_id": 2, "page_ids": [3], "is_candidate_cluster": False},
            ]},
            "cluster_refinement": {"accepted": [
                {"source_component_id": 1, "page_ids": [1, 2], "safety": {"safe": True}},
                {"source_component_id": 2, "page_ids": [3], "safety": {"safe": True}},
            ], "decisions": [{"component_id": 1}], "rejected_standalone_page_ids": [3]},
            "portfolio_calculations": [
                {"source_component_id": 1, "page_ids": [1, 2], "size_basis_points": 5000, "estimated_size_index": 50.0},
                {"source_component_id": 2, "page_ids": [3], "size_basis_points": 5000, "estimated_size_index": 50.0},
            ],
            "portfolio_assessments": [
                {"source_component_id": 1}, {"source_component_id": 2}
            ],
        }
        portfolio = {"audiences": [
            {"source_component_id": 1, "page_ids": [1, 2], "estimated_size_index": 50.0},
            {"source_component_id": 2, "page_ids": [3], "estimated_size_index": 50.0},
        ]}

        result = verify_publication_quality(audit, portfolio)

        self.assertEqual(result.traced_page_ids, (1, 2, 3))
        self.assertEqual(result.total_size_basis_points, 10_000)
        self.assertEqual(result.scored_component_ids, (1, 2))

        broken = copy.deepcopy(audit)
        broken["canonical_articles"][0]["aliases"] = []
        broken["raw_candidate_titles"].remove("Alias_A")
        with self.assertRaisesRegex(ValueError, "alias lineage"):
            verify_publication_quality(broken, portfolio)
