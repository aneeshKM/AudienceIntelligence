from __future__ import annotations

import unittest

from audience_trend_miner.clustering import CandidateComponent
from audience_trend_miner.refinement import refine_candidate_clusters
from tests.test_article_classification import ScriptedGenerator
from tests.test_publication import _qualified_article


def audience(name: str, *page_ids: int) -> dict[str, object]:
    return {
        "name": name,
        "page_ids": list(page_ids),
        "rationale": "These distinct articles describe one targetable audience.",
    }


def refinement(
    action: str,
    audiences: list[dict[str, object]],
    rejected_page_ids: list[int],
    *,
    alternative_matches: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "action": action,
        "audiences": audiences,
        "rejected_page_ids": rejected_page_ids,
        "alternative_matches": alternative_matches or [],
        "rationale": "Fixture refinement decision.",
    }


def safety(*, tragedy: bool = False, violent_crime: bool = False) -> dict[str, object]:
    return {
        "materially_centered_on_tragedy": tragedy,
        "materially_centered_on_violent_crime": violent_crime,
        "rationale": "Fixture cluster-level safety assessment.",
    }


class ClusterRefinementTest(unittest.TestCase):
    def test_validates_splits_rejects_and_vetoes_unsafe_audiences(self) -> None:
        articles = tuple(
            _qualified_article(page_id, title)
            for page_id, title in enumerate(
                (
                    "Running shoes", "Marathon training", "Home espresso",
                    "Coffee grinders", "Coffee shops", "Crime documentary",
                    "Victim biography", "Unrelated one", "Unrelated two",
                    "Standalone signal",
                ),
                start=1,
            )
        )
        components = (
            CandidateComponent(1, (1, 2), True),
            CandidateComponent(2, (3, 4, 5), True),
            CandidateComponent(3, (6, 7), True),
            CandidateComponent(4, (8, 9), True),
            CandidateComponent(5, (10,), False),
        )
        generator = ScriptedGenerator(
            refinement("validate", [audience("Endurance Runners", 1, 2)], []),
            safety(),
            refinement(
                "split",
                [audience("Home Coffee Brewers", 3, 4)],
                [5],
                alternative_matches=[{
                    "page_id": 5,
                    "audience_name": "Home Coffee Brewers",
                    "rationale": "Related, but not evidence for the same audience.",
                }],
            ),
            safety(),
            refinement("validate", [audience("True Crime Followers", 6, 7)], []),
            safety(violent_crime=True),
            refinement("reject", [], [8, 9]),
        )

        result = refine_candidate_clusters(
            components, articles, generator, sleep=lambda _: None
        )

        self.assertEqual(
            [(item.name, item.page_ids) for item in result.accepted],
            [("Endurance Runners", (1, 2)), ("Home Coffee Brewers", (3, 4))],
        )
        contributed = [page_id for item in result.accepted for page_id in item.page_ids]
        self.assertEqual(len(contributed), len(set(contributed)))
        self.assertEqual([item.action for item in result.decisions], ["validate", "split", "validate", "reject"])
        self.assertEqual(result.decisions[2].outcome, "safety_vetoed")
        self.assertEqual(result.rejected_standalone_page_ids, (10,))
        self.assertEqual(result.decisions[1].alternative_matches[0].page_id, 5)
        self.assertNotIn(5, contributed)
        self.assertEqual(len(generator.requests), 7)

    def test_retries_invalid_membership_then_fails_closed_with_complete_audit(self) -> None:
        articles = (
            _qualified_article(1, "One"),
            _qualified_article(2, "Two"),
            _qualified_article(3, "Three"),
        )
        components = (CandidateComponent(1, (1, 2, 3), True),)
        no_op_split = refinement(
            "split", [audience("Unchanged Audience", 1, 2, 3)], []
        )
        invalid = refinement(
            "split",
            [audience("First", 1, 2), audience("Second", 2, 3)],
            [],
        )

        result = refine_candidate_clusters(
            components,
            articles,
            ScriptedGenerator(no_op_split, invalid, invalid),
            sleep=lambda _: None,
            jitter=lambda: 0,
        )

        self.assertEqual(result.accepted, ())
        self.assertEqual(result.decisions[0].outcome, "exhausted_attempts")
        self.assertEqual(result.decisions[0].rejected_page_ids, (1, 2, 3))
        self.assertEqual(len(result.decisions[0].attempts), 3)
        self.assertTrue(all(not attempt.validation_valid for attempt in result.decisions[0].attempts))
        self.assertIn(
            "split must create multiple audiences or reject a member",
            result.decisions[0].attempts[0].error,
        )

    def test_retries_cluster_safety_failure_and_fails_closed(self) -> None:
        articles = (_qualified_article(1, "One"), _qualified_article(2, "Two"))
        components = (CandidateComponent(1, (1, 2), True),)
        generator = ScriptedGenerator(
            refinement("validate", [audience("Coherent Audience", 1, 2)], []),
            {"materially_centered_on_tragedy": False},
            RuntimeError("unavailable"),
            {"unexpected": True},
        )

        result = refine_candidate_clusters(
            components, articles, generator, sleep=lambda _: None, jitter=lambda: 0
        )

        self.assertEqual(result.accepted, ())
        assessment = result.decisions[0].safety_assessments[0]
        self.assertFalse(assessment.safe)
        self.assertEqual(assessment.decision_reason, "exhausted_attempts")
        self.assertEqual(len(assessment.attempts), 3)


if __name__ == "__main__":
    unittest.main()
