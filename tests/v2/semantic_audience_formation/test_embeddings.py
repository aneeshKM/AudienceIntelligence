from __future__ import annotations

import unittest

import numpy as np

from audience_trend_miner.v2.semantic_audience_formation.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingAdapter,
)
from audience_trend_miner.v2.shared import V2ContractError


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int, bool]] = []

    def encode(
        self,
        representations: tuple[str, ...],
        *,
        batch_size: int,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.calls.append((representations, batch_size, convert_to_numpy))
        return np.ones((len(representations), 3), dtype=float)


class ProductionEmbeddingAdapterTest(unittest.TestCase):
    def test_uses_documented_default_model_and_configured_batches(self) -> None:
        encoder = RecordingEncoder()
        loaded_models: list[str] = []

        adapter = SentenceTransformerEmbeddingAdapter(
            encoder_factory=lambda model: (loaded_models.append(model), encoder)[1]
        )
        vectors = adapter.embed(tuple(f"representation-{index}" for index in range(65)))

        self.assertEqual(adapter.model, DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(loaded_models, ["sentence-transformers/all-mpnet-base-v2"])
        self.assertEqual(len(vectors), 65)
        self.assertEqual(
            encoder.calls,
            [
                (
                    tuple(f"representation-{index}" for index in range(65)),
                    DEFAULT_EMBEDDING_BATCH_SIZE,
                    True,
                )
            ],
        )

    def test_uses_valid_model_and_batch_size_overrides(self) -> None:
        encoder = RecordingEncoder()

        adapter = SentenceTransformerEmbeddingAdapter(
            model="local/experiment-model",
            batch_size=7,
            encoder_factory=lambda _model: encoder,
        )
        adapter.embed(("one", "two"))

        self.assertEqual(adapter.model, "local/experiment-model")
        self.assertEqual(encoder.calls[0][1], 7)

    def test_rejects_invalid_model_and_batch_configuration(self) -> None:
        for model, batch_size, expected_error in (
            ("", 32, "embedding model must be non-empty"),
            ("   ", 32, "embedding model must be non-empty"),
            (DEFAULT_EMBEDDING_MODEL, 0, "embedding batch size must be positive"),
            (DEFAULT_EMBEDDING_MODEL, -1, "embedding batch size must be positive"),
        ):
            with self.subTest(model=model, batch_size=batch_size):
                with self.assertRaisesRegex(V2ContractError, expected_error):
                    SentenceTransformerEmbeddingAdapter(
                        model=model,
                        batch_size=batch_size,
                        encoder_factory=lambda _model: RecordingEncoder(),
                    )


if __name__ == "__main__":
    unittest.main()
