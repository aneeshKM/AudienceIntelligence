from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from audience_trend_miner.v2.shared import V2ContractError


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
