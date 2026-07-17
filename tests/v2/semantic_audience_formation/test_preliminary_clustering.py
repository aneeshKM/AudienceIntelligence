from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from audience_trend_miner.v2.semantic_audience_formation import SelectedCategoryPage
from audience_trend_miner.v2.semantic_audience_formation.clustering import (
    SubdivisionPolicy,
    form_preliminary_clusters,
)
from audience_trend_miner.v2.semantic_audience_formation.embeddings import (
    FrozenEmbeddingAdapter,
)
from audience_trend_miner.v2.shared import V2ContractError


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
    def test_rejects_a_page_that_cannot_fit_the_guard_without_truncation(self) -> None:
        pages = (
            SelectedCategoryPage(1, "One", "x" * 500, ("Shared",)),
            page(2, "Two", "Shared"),
        )
        vectors = ((1.0, 0.0), (1.0, 0.0))

        with self.assertRaisesRegex(
            V2ContractError,
            "Canonical Page 1 exceeds the model-input token guard",
        ):
            form_preliminary_clusters(
                pages,
                SequentialEmbeddingAdapter([vectors, vectors]),
                threshold=0.75,
                subdivision_policy=SubdivisionPolicy(
                    max_input_tokens=250,
                    fixed_prompt_tokens=0,
                    stricter_threshold_step=0.1,
                ),
            )

    def test_subdivides_only_oversized_components_without_dropping_members(self) -> None:
        pages = tuple(
            page(index, f"Page {index}", "Shared") for index in range(1, 7)
        )
        pair_vectors = (
            (1.0, 0.0),
            (1.0, 0.0),
            (0.8, 0.6),
            (0.8, 0.6),
            (0.0, 1.0),
            (0.0, 1.0),
        )
        policy = SubdivisionPolicy(
            max_input_tokens=250,
            fixed_prompt_tokens=0,
            stricter_threshold_step=0.1,
        )

        artifact = form_preliminary_clusters(
            pages,
            SequentialEmbeddingAdapter([pair_vectors, pair_vectors]),
            threshold=0.75,
            subdivision_policy=policy,
        )

        clusters = [cluster.page_ids for cluster in artifact.preliminary_clusters]
        self.assertCountEqual(clusters, [(1, 2), (3, 4), (5, 6)])
        self.assertEqual(
            sorted(page_id for cluster in clusters for page_id in cluster),
            list(range(1, 7)),
        )
        self.assertEqual(artifact.singleton_count, 0)
        self.assertEqual(artifact.subdivision_policy.method, "stricter-boundary")
        self.assertEqual(
            artifact.subdivision_policy.token_estimation,
            "utf8-bytes-upper-bound",
        )
        repeated = form_preliminary_clusters(
            tuple(reversed(pages)),
            SequentialEmbeddingAdapter([pair_vectors, pair_vectors]),
            threshold=0.75,
            subdivision_policy=policy,
        )
        self.assertEqual(
            [cluster.page_ids for cluster in artifact.preliminary_clusters],
            [cluster.page_ids for cluster in repeated.preliminary_clusters],
        )

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

    def test_rejects_invalid_embeddings_before_similarity_is_used(self) -> None:
        pages = (page(1, "One", "Shared"), page(2, "Two", "Shared"))
        scenarios = (
            ([((), ()), ((1.0,), (1.0,))], "must be non-empty"),
            ([((1.0,), (float("nan"),)), ((1.0,), (1.0,))], "must be finite"),
            ([((0.0,), (0.0,)), ((1.0,), (1.0,))], "non-zero magnitude"),
            (
                [((1.0,), (1.0, 2.0)), ((1.0,), (1.0,))],
                "dimensionally consistent",
            ),
            (
                [((1.0,), (1.0,)), ((1.0, 0.0), (1.0, 0.0))],
                "dimensionally consistent",
            ),
        )

        for responses, expected_error in scenarios:
            with self.subTest(expected_error=expected_error):
                adapter = SequentialEmbeddingAdapter(responses)
                with self.assertRaisesRegex(V2ContractError, expected_error):
                    form_preliminary_clusters(pages, adapter, threshold=0.5)

    def test_handles_representative_candidate_universe_without_persisting_vectors(self) -> None:
        pages = tuple(page(index, f"Page {index}", "Shared") for index in range(1, 258))
        adapter = ConstantNumpyEmbeddingAdapter()

        artifact = form_preliminary_clusters(pages, adapter, threshold=0.9)

        self.assertEqual(adapter.batch_sizes, [257, 257])
        retained_page_ids = sorted(
            page_id
            for cluster in artifact.preliminary_clusters
            for page_id in cluster.page_ids
        )
        self.assertEqual(retained_page_ids, list(range(1, 258)))
        record = artifact.record()
        self.assertNotIn("embeddings", record)
        self.assertNotIn("similarity_matrix", record)

    def test_preserves_cosine_similarity_for_finite_extreme_vectors(self) -> None:
        pages = (page(1, "One", "Shared"), page(2, "Two", "Shared"))
        extreme_vectors = (
            ((1e308, 1e308), (1e308, 1e308)),
            ((5e-324, 5e-324), (5e-324, 5e-324)),
        )

        artifact = form_preliminary_clusters(
            pages,
            SequentialEmbeddingAdapter(list(extreme_vectors)),
            threshold=0.9,
        )

        self.assertEqual(
            [cluster.page_ids for cluster in artifact.preliminary_clusters],
            [(1, 2)],
        )
        self.assertAlmostEqual(artifact.preliminary_clusters[0].cohesion, 1.0)


class SequentialEmbeddingAdapter:
    model = "invalid-fixture"

    def __init__(self, responses: list[tuple[tuple[float, ...], ...]]) -> None:
        self._responses = iter(responses)

    def embed(
        self, _representations: tuple[str, ...]
    ) -> tuple[tuple[float, ...], ...]:
        return next(self._responses)


class ConstantNumpyEmbeddingAdapter:
    model = "representative-fixture"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, representations: tuple[str, ...]) -> np.ndarray:
        self.batch_sizes.append(len(representations))
        return np.tile(np.array((1.0, 0.5, 0.25)), (len(representations), 1))


if __name__ == "__main__":
    unittest.main()
