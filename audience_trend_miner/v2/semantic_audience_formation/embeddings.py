from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from audience_trend_miner.v2.shared import V2ContractError


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32


class SentenceEncoder(Protocol):
    def encode(
        self,
        representations: Sequence[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
    ) -> object: ...


def _load_sentence_transformer(model: str) -> SentenceEncoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


class SentenceTransformerEmbeddingAdapter:
    """Run configurable, batched local Sentence Transformer inference."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        *,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        encoder_factory: Callable[[str], SentenceEncoder] = _load_sentence_transformer,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise V2ContractError("embedding model must be non-empty")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise V2ContractError("embedding batch size must be positive")
        self.model = model.strip()
        self.batch_size = batch_size
        try:
            self._encoder = encoder_factory(self.model)
        except Exception as error:
            raise V2ContractError(
                f"embedding model {self.model!r} could not be loaded"
            ) from error

    def embed(
        self, representations: Sequence[str]
    ) -> Sequence[Sequence[float]] | NDArray[np.float64]:
        try:
            encoded = self._encoder.encode(
                tuple(representations),
                batch_size=self.batch_size,
                convert_to_numpy=True,
            )
            return cast(Sequence[Sequence[float]] | NDArray[np.float64], encoded)
        except V2ContractError:
            raise
        except Exception as error:
            raise V2ContractError("embedding inference failed") from error


class FrozenEmbeddingAdapter:
    """Resolve exact semantic representations from a deterministic fixture."""

    def __init__(
        self, model: str, embeddings: dict[str, tuple[float, ...]]
    ) -> None:
        self.model = model
        self._embeddings = embeddings

    @classmethod
    def from_file(cls, path: Path) -> FrozenEmbeddingAdapter:
        try:
            fixture = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise V2ContractError("embedding fixture is unreadable") from error
        if (
            not isinstance(fixture, dict)
            or fixture.get("schema_version") != "1.0"
            or not isinstance(fixture.get("model"), str)
            or not fixture["model"]
            or not isinstance(fixture.get("embeddings"), dict)
        ):
            raise V2ContractError("embedding fixture has an invalid shape")
        embeddings: dict[str, tuple[float, ...]] = {}
        for representation, vector in fixture["embeddings"].items():
            if not isinstance(representation, str) or not isinstance(vector, list):
                raise V2ContractError("embedding fixture has an invalid shape")
            try:
                embeddings[representation] = tuple(float(value) for value in vector)
            except (TypeError, ValueError) as error:
                raise V2ContractError("embedding fixture has an invalid vector") from error
        return cls(fixture["model"], embeddings)

    def embed(self, representations: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        try:
            return tuple(self._embeddings[text] for text in representations)
        except KeyError as error:
            raise V2ContractError(
                f"embedding fixture has no vector for representation {error.args[0]!r}"
            ) from error
