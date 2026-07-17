from __future__ import annotations

from pathlib import Path
import unittest

from audience_trend_miner.v2.semantic_audience_formation import SelectedCategoryPage
from audience_trend_miner.v2.semantic_audience_formation.clustering import (
    form_preliminary_clusters,
)
from audience_trend_miner.v2.semantic_audience_formation.embeddings import (
    FrozenEmbeddingAdapter,
)


FIXTURE = (
    Path(__file__).with_name("fixtures") / "preliminary_cluster_embeddings.json"
)


def page(page_id: int, title: str, category: str) -> SelectedCategoryPage:
    return SelectedCategoryPage(
        page_id=page_id,
        canonical_title=title,
        lead=f"{title.lower()} lead.",
        selected_categories=(category,),
    )


class PreliminaryClusteringTest(unittest.TestCase):
    def test_forms_and_ranks_minimal_clusters_from_dual_representations(self) -> None:
        pages = (
            page(7, "Gamma Two", "Gamma"),
            page(2, "Alpha Two", "Alpha"),
            page(10, "Singleton", "Isolated"),
            page(4, "Beta One", "Beta"),
            page(9, "Boundary Two", "Boundary"),
            page(1, "Alpha One", "Alpha"),
            page(6, "Gamma One", "Gamma"),
            page(5, "Beta Two", "Beta"),
            page(8, "Boundary One", "Boundary"),
            page(3, "Alpha Three", "Alpha"),
        )

        artifact = form_preliminary_clusters(
            pages,
            FrozenEmbeddingAdapter.from_file(FIXTURE),
            threshold=0.3,
        )

        self.assertEqual(artifact.embedding_model, "deterministic-fixture")
        self.assertEqual(artifact.content_weight, 0.7)
        self.assertEqual(artifact.category_weight, 0.3)
        self.assertEqual(artifact.threshold, 0.3)
        self.assertEqual(artifact.singleton_count, 1)
        self.assertEqual(
            [cluster.page_ids for cluster in artifact.preliminary_clusters],
            [(1, 2, 3), (4, 5), (6, 7), (8, 9)],
        )
        self.assertEqual(
            [cluster.cohesion for cluster in artifact.preliminary_clusters],
            [1.0, 1.0, 1.0, 0.3],
        )
        record = artifact.record()
        retained_page_ids = {
            member["page_id"]
            for cluster in record["preliminary_clusters"]
            for member in cluster["members"]
        }
        self.assertEqual(retained_page_ids, set(range(1, 10)))
        self.assertNotIn("traffic", str(record).lower())


if __name__ == "__main__":
    unittest.main()
