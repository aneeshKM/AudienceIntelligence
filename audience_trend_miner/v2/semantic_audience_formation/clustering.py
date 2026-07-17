from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

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
    ) -> Sequence[Sequence[float]] | NDArray[np.float64]: ...


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
    content_vectors = _validated_embedding_matrix(
        embedding_adapter.embed(content_representations),
        expected_count=len(ordered_pages),
    )
    category_vectors = _validated_embedding_matrix(
        embedding_adapter.embed(category_representations),
        expected_count=len(ordered_pages),
    )
    if content_vectors.shape[1] != category_vectors.shape[1]:
        raise V2ContractError("embeddings must be dimensionally consistent")

    content_similarity = content_vectors @ content_vectors.T
    category_similarity = category_vectors @ category_vectors.T
    combined_similarities = (
        CONTENT_WEIGHT * content_similarity + CATEGORY_WEIGHT * category_similarity
    )
    neighbors = {index: set() for index in range(len(ordered_pages))}
    edge_indices = np.argwhere(
        np.triu(combined_similarities >= threshold, k=1)
    )
    for left, right in edge_indices:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))

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
            cohesion=_mean_pairwise_similarity(component, combined_similarities),
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


def _validated_embedding_matrix(
    vectors: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    expected_count: int,
) -> NDArray[np.float64]:
    if expected_count == 0:
        if len(vectors) != 0:
            raise V2ContractError("embedding count does not match Canonical Page count")
        return np.empty((0, 0), dtype=float)
    try:
        matrix = np.asarray(vectors, dtype=float)
    except (TypeError, ValueError) as error:
        raise V2ContractError("embeddings must be dimensionally consistent") from error
    if matrix.ndim != 2 or matrix.shape[0] != expected_count:
        if matrix.ndim == 2 and matrix.shape[0] != expected_count:
            raise V2ContractError("embedding count does not match Canonical Page count")
        raise V2ContractError("embeddings must be dimensionally consistent")
    if matrix.shape[1] == 0:
        raise V2ContractError("embeddings must be non-empty")
    if not np.isfinite(matrix).all():
        raise V2ContractError("embeddings must be finite")
    scales = np.max(np.abs(matrix), axis=1)
    if np.any(scales == 0):
        raise V2ContractError("embeddings must have non-zero magnitude")
    scaled_matrix = matrix / scales[:, np.newaxis]
    magnitudes = np.linalg.norm(scaled_matrix, axis=1)
    return scaled_matrix / magnitudes[:, np.newaxis]


def _mean_pairwise_similarity(
    component: Sequence[int], similarities: NDArray[np.float64]
) -> float:
    component_similarities = similarities[np.ix_(component, component)]
    pair_values = component_similarities[np.triu_indices(len(component), k=1)]
    return float(np.mean(pair_values))
