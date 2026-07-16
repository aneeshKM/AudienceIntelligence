from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Protocol, Sequence

from audience_trend_miner.wikimedia import CanonicalArticle


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.62


class EmbeddingAdapter(Protocol):
    model: str

    def embed(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class SemanticRepresentation:
    page_id: int
    canonical_title: str
    text: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class PairwiseSimilarity:
    left_page_id: int
    right_page_id: int
    cosine_similarity: float


@dataclass(frozen=True)
class SimilarityEdge:
    left_page_id: int
    right_page_id: int
    cosine_similarity: float


@dataclass(frozen=True)
class CandidateComponent:
    component_id: int
    page_ids: tuple[int, ...]
    is_candidate_cluster: bool


@dataclass(frozen=True)
class CandidateClusteringResult:
    model: str
    threshold: float
    representations: tuple[SemanticRepresentation, ...]
    similarities: tuple[PairwiseSimilarity, ...]
    edges: tuple[SimilarityEdge, ...]
    components: tuple[CandidateComponent, ...]


def form_candidate_clusters(
    articles: tuple[CanonicalArticle, ...],
    adapter: EmbeddingAdapter,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> CandidateClusteringResult:
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("similarity threshold must be between -1 and 1")
    if len({article.page_id for article in articles}) != len(articles):
        raise ValueError("canonical articles must have unique page IDs")

    representations = tuple(
        _represent(article, adapter) for article in articles
    )
    similarities: list[PairwiseSimilarity] = []
    edges: list[SimilarityEdge] = []
    adjacency = {article.page_id: set() for article in articles}
    for left_index, left in enumerate(representations):
        for right in representations[left_index + 1 :]:
            similarity = _cosine(left.embedding, right.embedding)
            pair = PairwiseSimilarity(left.page_id, right.page_id, similarity)
            similarities.append(pair)
            if similarity >= threshold:
                edge = SimilarityEdge(left.page_id, right.page_id, similarity)
                edges.append(edge)
                adjacency[left.page_id].add(right.page_id)
                adjacency[right.page_id].add(left.page_id)

    components: list[CandidateComponent] = []
    visited: set[int] = set()
    for page_id in adjacency:
        if page_id in visited:
            continue
        pending = [page_id]
        members: list[int] = []
        visited.add(page_id)
        while pending:
            member = pending.pop()
            members.append(member)
            for neighbour in adjacency[member]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        page_ids = tuple(sorted(members))
        components.append(
            CandidateComponent(len(components) + 1, page_ids, len(page_ids) > 1)
        )

    return CandidateClusteringResult(
        adapter.model,
        threshold,
        representations,
        tuple(similarities),
        tuple(edges),
        tuple(components),
    )


def semantic_text(article: CanonicalArticle) -> str:
    return (
        f"Title: {article.canonical_title}\n"
        f"Lead extract: {article.extract}\n"
        f"Categories: {', '.join(article.categories)}"
    )


def _represent(
    article: CanonicalArticle, adapter: EmbeddingAdapter
) -> SemanticRepresentation:
    text = semantic_text(article)
    embedding = tuple(float(value) for value in adapter.embed(text))
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError(f"invalid embedding for page ID {article.page_id}")
    return SemanticRepresentation(
        article.page_id, article.canonical_title, text, embedding
    )


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("embeddings must have non-zero magnitude")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


class SentenceTransformerEmbeddingAdapter:
    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model = model
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(model)

    def embed(self, text: str) -> Sequence[float]:
        return self._encoder.encode(text, convert_to_numpy=True).tolist()


class FrozenEmbeddingAdapter:
    def __init__(
        self,
        embeddings: Sequence[Sequence[float]],
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model = model
        self._embeddings = list(embeddings)
        self.embedded_texts: list[str] = []

    @classmethod
    def from_file(cls, path: os.PathLike[str]) -> FrozenEmbeddingAdapter:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload["embeddings"],
            model=payload.get("model", DEFAULT_EMBEDDING_MODEL),
        )

    def embed(self, text: str) -> Sequence[float]:
        index = len(self.embedded_texts)
        self.embedded_texts.append(text)
        return self._embeddings[index]
