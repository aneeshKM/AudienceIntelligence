from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol, Sequence

from audience_trend_miner.v2.semantic_audience_formation.categories import (
    SelectedCategoryPage,
)
from audience_trend_miner.v2.shared import V2ContractError


CONTENT_WEIGHT = 0.7
CATEGORY_WEIGHT = 0.3


class EmbeddingAdapter(Protocol):
    model: str

    def embed(
        self, representations: Sequence[str]
    ) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class PreliminaryCluster:
    members: tuple[SelectedCategoryPage, ...]
    cohesion: float

    @property
    def page_ids(self) -> tuple[int, ...]:
        return tuple(member.page_id for member in self.members)


@dataclass(frozen=True)
class PreliminaryClusterArtifact:
    embedding_model: str
    content_weight: float
    category_weight: float
    threshold: float
    preliminary_clusters: tuple[PreliminaryCluster, ...]
    singleton_count: int

    def record(self) -> dict[str, object]:
        """Return the minimal serializable evidence produced by fixture formation."""
        return asdict(self)


def form_preliminary_clusters(
    pages: Sequence[SelectedCategoryPage],
    embedding_adapter: EmbeddingAdapter,
    *,
    threshold: float,
) -> PreliminaryClusterArtifact:
    """Form and rank Preliminary Clusters without traffic evidence."""
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise V2ContractError("similarity threshold must be between -1 and 1")
    ordered_pages = tuple(sorted(pages, key=lambda page: page.page_id))
    content_representations = tuple(
        f"Title: {page.canonical_title}\nLead: {page.lead}" for page in ordered_pages
    )
    category_representations = tuple(
        "Selected Categories: "
        + (" | ".join(page.selected_categories) or "(none)")
        for page in ordered_pages
    )
    content_vectors = embedding_adapter.embed(content_representations)
    category_vectors = embedding_adapter.embed(category_representations)
    if len(content_vectors) != len(ordered_pages) or len(category_vectors) != len(
        ordered_pages
    ):
        raise V2ContractError("embedding count does not match Canonical Page count")

    similarities: dict[tuple[int, int], float] = {}
    neighbors = {index: set() for index in range(len(ordered_pages))}
    for left in range(len(ordered_pages)):
        for right in range(left + 1, len(ordered_pages)):
            combined_similarity = (
                CONTENT_WEIGHT
                * _cosine_similarity(content_vectors[left], content_vectors[right])
                + CATEGORY_WEIGHT
                * _cosine_similarity(category_vectors[left], category_vectors[right])
            )
            similarities[(left, right)] = combined_similarity
            if combined_similarity >= threshold:
                neighbors[left].add(right)
                neighbors[right].add(left)

    components: list[tuple[int, ...]] = []
    unseen = set(neighbors)
    while unseen:
        start = min(unseen)
        component: set[int] = set()
        pending = [start]
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(neighbors[index] - component)
        unseen -= component
        components.append(tuple(sorted(component)))

    singleton_count = sum(len(component) == 1 for component in components)
    clusters = [
        PreliminaryCluster(
            members=tuple(ordered_pages[index] for index in component),
            cohesion=_mean_pairwise_similarity(component, similarities),
        )
        for component in components
        if len(component) > 1
    ]
    clusters.sort(
        key=lambda cluster: (-cluster.cohesion, -len(cluster.members), cluster.page_ids)
    )
    return PreliminaryClusterArtifact(
        embedding_model=embedding_adapter.model,
        content_weight=CONTENT_WEIGHT,
        category_weight=CATEGORY_WEIGHT,
        threshold=threshold,
        preliminary_clusters=tuple(clusters),
        singleton_count=singleton_count,
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise V2ContractError("embeddings must be non-empty and dimensionally consistent")
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if not left_magnitude or not right_magnitude:
        raise V2ContractError("embeddings must have non-zero magnitude")
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_magnitude * right_magnitude
    )
    if not math.isfinite(similarity):
        raise V2ContractError("embedding similarity must be finite")
    return similarity


def _mean_pairwise_similarity(
    component: Sequence[int], similarities: dict[tuple[int, int], float]
) -> float:
    pair_values = [
        similarities[(component[left], component[right])]
        for left in range(len(component))
        for right in range(left + 1, len(component))
    ]
    return sum(pair_values) / len(pair_values)
