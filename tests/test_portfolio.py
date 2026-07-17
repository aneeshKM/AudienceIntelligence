from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from audience_trend_miner.portfolio import build_portfolio
from audience_trend_miner.publication import publish_run
from audience_trend_miner.classification import classify_articles
from audience_trend_miner.clustering import FrozenEmbeddingAdapter, form_candidate_clusters
from audience_trend_miner.refinement import (
    AcceptedAudience,
    ClusterRefinementResult,
    ClusterSafetyAssessment,
    RefinementAttempt,
)
from audience_trend_miner.wikimedia import WikimediaAttentionResult
from audience_trend_miner.trends import qualify_trends
from tests.test_article_classification import ScriptedGenerator
from tests.test_article_classification import judgment
from tests.test_publication import _qualified_article, publication_input


def accepted(name: str, component_id: int, *page_ids: int) -> AcceptedAudience:
    safety = ClusterSafetyAssessment(
        name, tuple(page_ids), "safety prompt", True, "accepted",
        False, False, "safe", (RefinementAttempt(1, {}, True, None),),
    )
    return AcceptedAudience(
        name, tuple(page_ids), "Coherent audience.", component_id, safety
    )


def assessment(
    name: str,
    description: str,
    scores: tuple[int, int, int, int],
) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "purchase_intent": scores[0],
        "transaction_value": scores[1],
        "category_breadth": scores[2],
        "brand_safety": scores[3],
        "brand_categories": ["Sportswear", "Fitness technology"],
        "rationale": "The four component scores support this rating.",
        "name_is_targetable_group": True,
        "causal_claims_are_hypotheses": True,
        "rationale_avoids_wealth_inference": True,
    }


class PortfolioTransformationTest(unittest.TestCase):
    def test_builds_ranked_capped_portfolio_with_exact_size_and_buying_power(self) -> None:
        articles = tuple(
            _qualified_article(page_id, f"Article {page_id}")
            for page_id in range(1, 25)
        )
        refined = ClusterRefinementResult(
            tuple(
                accepted(f"Candidate {number}", number, number * 2 - 1, number * 2)
                for number in range(1, 13)
            ),
            (),
            (),
        )
        responses = [
            assessment(
                f"Active Lifestyle Group {number}",
                "Observed traffic rose from 140,000 to 280,000 views, suggesting increased interest.",
                (3, 3, 3, 3) if number == 1 else
                (3, 3, 3, 1) if number == 2 else (2, 2, 2, 2),
            )
            for number in range(1, 11)
        ]

        result = build_portfolio(
            refined,
            articles,
            ScriptedGenerator(*responses),
            sleep=lambda _: None,
        )
        self.assertEqual(len(result.audiences), 10)
        self.assertEqual([item.source_component_id for item in result.audiences], list(range(1, 11)))
        self.assertEqual(sum(item.size_basis_points for item in result.audiences), 10_000)
        self.assertEqual(result.audiences[0].estimated_size_index, 10.0)
        self.assertEqual(result.audiences[0].potential_buying_power, "high")
        self.assertEqual(result.audiences[1].potential_buying_power, "low")
        self.assertEqual(result.audiences[2].potential_buying_power, "medium")
        self.assertEqual(result.audiences[0].brand_categories, ("Sportswear", "Fitness technology"))
        self.assertIn("suggesting", result.audiences[0].description)

    def test_empty_refinement_produces_valid_empty_portfolio_without_model_calls(self) -> None:
        result = build_portfolio(
            ClusterRefinementResult((), (), ()),
            (),
            ScriptedGenerator(),
            sleep=lambda _: None,
        )
        self.assertEqual(result.audiences, ())
        self.assertEqual(result.assessments, ())

    def test_publishes_the_same_audience_data_in_json_and_html(self) -> None:
        articles = (_qualified_article(1, "Trail shoes"), _qualified_article(2, "Trail running"))
        refined = ClusterRefinementResult((accepted("Trail runners", 1, 1, 2),), (), ())
        result = build_portfolio(
            refined,
            articles,
            ScriptedGenerator(assessment(
                "Emerging Trail Runners",
                "Traffic rose from 140,000 to 280,000 views, suggesting growing interest.",
                (2, 3, 3, 3),
            )),
            sleep=lambda _: None,
        )
        qualification = qualify_trends(articles)
        classification = classify_articles(
            articles, ScriptedGenerator(judgment(), judgment()), sleep=lambda _: None
        )
        clustering = form_candidate_clusters(
            articles, FrozenEmbeddingAdapter(((1.0, 0.0), (1.0, 0.0)))
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            published = publish_run(publication_input(
                Path(temporary_directory),
                attention=WikimediaAttentionResult((), articles, ()),
                qualification=qualification,
                classification=classification,
                clustering=clustering,
                refinement=refined,
                portfolio=result,
            ))
            portfolio_text = (published / "portfolio.json").read_text()
            portfolio = json.loads(portfolio_text)
            audit = json.loads((published / "audit.json").read_text())
            report = (published / "report.html").read_text()

        audience = portfolio["audiences"][0]
        self.assertEqual(audience["source_component_id"], 1)
        self.assertEqual(audience["page_ids"], [1, 2])
        self.assertEqual(audit["quality_checks"]["traced_page_ids"], [1, 2])
        self.assertEqual(audit["quality_checks"]["total_size_basis_points"], 10_000)
        self.assertEqual(audience["estimated_size_index"], 100.0)
        self.assertIn(
            '"estimated_size_index": 100.00',
            portfolio_text,
        )
        for displayed in (
            audience["name"], audience["description"], "100.00",
            audience["potential_buying_power"].title(),
            *audience["brand_categories"], audience["buying_power_rationale"],
        ):
            self.assertIn(str(displayed), report)


if __name__ == "__main__":
    unittest.main()
