from __future__ import annotations

import math
from pathlib import Path
import unittest

from audience_trend_miner.clustering import (
    DEFAULT_EMBEDDING_MODEL,
    FrozenEmbeddingAdapter,
    form_candidate_clusters,
)
from tests.test_publication import _qualified_article


class CandidateClusteringTest(unittest.TestCase):
    def test_builds_auditable_graph_at_inclusive_threshold_and_retains_singletons(self) -> None:
        shoes = _qualified_article(1, "Running shoes")
        marathon = _qualified_article(2, "Marathon training")
        espresso = _qualified_article(3, "Home espresso")
        adapter = FrozenEmbeddingAdapter(
            (
                (1.0, 0.0),
                (0.62, math.sqrt(1 - 0.62**2)),
                (-1.0, 0.0),
            )
        )

        result = form_candidate_clusters(
            (shoes, marathon, espresso), adapter, threshold=0.62
        )

        self.assertEqual(len(adapter.embedded_texts), 3)
        self.assertEqual(result.model, DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(len(result.similarities), 3)
        self.assertAlmostEqual(result.similarities[0].cosine_similarity, 0.62)
        self.assertEqual([(edge.left_page_id, edge.right_page_id) for edge in result.edges], [(1, 2)])
        self.assertEqual(
            [
                (component.page_ids, component.is_candidate_cluster)
                for component in result.components
            ],
            [((1, 2), True), ((3,), False)],
        )
        self.assertEqual(
            result.representations[0].text,
            "Title: Running shoes\nLead extract: Fixture lead.\n"
            "Categories: Consumer topics",
        )
        self.assertEqual(adapter.embedded_texts[0], result.representations[0].text)
        self.assertEqual(result.representations[0].embedding, (1.0, 0.0))

    def test_similarity_below_threshold_does_not_create_an_edge(self) -> None:
        first = _qualified_article(10, "First")
        second = _qualified_article(11, "Second")

        result = form_candidate_clusters(
            (first, second),
            FrozenEmbeddingAdapter(
                ((1.0, 0.0), (0.619, math.sqrt(1 - 0.619**2)))
            ),
            threshold=0.62,
        )

        self.assertEqual(result.edges, ())
        self.assertEqual([item.page_ids for item in result.components], [(10,), (11,)])

    def test_loads_the_committed_frozen_embedding_fixture(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "clustering_embeddings.json"
        adapter = FrozenEmbeddingAdapter.from_file(fixture)

        result = form_candidate_clusters(
            (_qualified_article(20, "One"), _qualified_article(21, "Two")),
            adapter,
        )

        self.assertEqual(result.edges[0].cosine_similarity, 1.0)
        self.assertEqual(result.components[0].page_ids, (20, 21))


if __name__ == "__main__":
    unittest.main()
