from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
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
DEFAULT_SIMILARITY_THRESHOLD = 0.76
DEFAULT_MAX_MODEL_INPUT_TOKENS = 16_384
DEFAULT_FIXED_PROMPT_TOKENS = 2_048
DEFAULT_STRICTER_THRESHOLD_STEP = 0.02


@dataclass(frozen=True)
class SubdivisionPolicy:
    max_input_tokens: int = DEFAULT_MAX_MODEL_INPUT_TOKENS
    fixed_prompt_tokens: int = DEFAULT_FIXED_PROMPT_TOKENS
    stricter_threshold_step: float = DEFAULT_STRICTER_THRESHOLD_STEP
    method: str = field(default="stricter-boundary", init=False)
    token_estimation: str = field(default="utf8-bytes-upper-bound", init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_input_tokens, int)
            or isinstance(self.max_input_tokens, bool)
            or self.max_input_tokens <= 0
        ):
            raise V2ContractError("model-input token guard must be positive")
        if (
            not isinstance(self.fixed_prompt_tokens, int)
            or isinstance(self.fixed_prompt_tokens, bool)
            or self.fixed_prompt_tokens < 0
            or self.fixed_prompt_tokens >= self.max_input_tokens
        ):
            raise V2ContractError(
                "fixed prompt tokens must fit within the model-input token guard"
            )
        if (
            not math.isfinite(self.stricter_threshold_step)
            or not 0.01 <= self.stricter_threshold_step <= 1
        ):
            raise V2ContractError(
                "stricter threshold step must be between 0.01 and 1"
            )


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
    subdivision_policy: SubdivisionPolicy
    preliminary_clusters: tuple[PreliminaryCluster, ...]
    singleton_count: int

    def record(self) -> dict[str, object]:
        """Return the minimal serializable evidence produced by fixture formation."""
        return asdict(self)


def form_preliminary_clusters(
    pages: Sequence[SelectedCategoryPage],
    embedding_adapter: EmbeddingAdapter,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    subdivision_policy: SubdivisionPolicy = SubdivisionPolicy(),
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

    components = _connected_components(tuple(range(len(ordered_pages))), neighbors)

    singleton_count = sum(len(component) == 1 for component in components)
    reviewable_components = [
        subdivision
        for component in components
        if len(component) > 1
        for subdivision in _subdivide_component(
            component,
            combined_similarities,
            ordered_pages,
            subdivision_policy,
            threshold=threshold,
        )
    ]
    clusters = [
        PreliminaryCluster(
            members=tuple(ordered_pages[index] for index in component),
            cohesion=_mean_pairwise_similarity(component, combined_similarities),
        )
        for component in reviewable_components
    ]
    clusters.sort(
        key=lambda cluster: (-cluster.cohesion, -len(cluster.members), cluster.page_ids)
    )
    return PreliminaryClusterArtifact(
        embedding_model=embedding_adapter.model,
        content_weight=CONTENT_WEIGHT,
        category_weight=CATEGORY_WEIGHT,
        threshold=threshold,
        subdivision_policy=subdivision_policy,
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
    if len(component) == 1:
        return 1.0
    component_similarities = similarities[np.ix_(component, component)]
    pair_values = component_similarities[np.triu_indices(len(component), k=1)]
    return float(np.mean(pair_values))


def _connected_components(
    indices: Sequence[int], neighbors: dict[int, set[int]]
) -> list[tuple[int, ...]]:
    components: list[tuple[int, ...]] = []
    unseen = set(indices)
    while unseen:
        start = min(unseen)
        component: set[int] = set()
        pending = [start]
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend((neighbors[index] & unseen) - component)
        unseen -= component
        components.append(tuple(sorted(component)))
    return components


def _subdivide_component(
    component: tuple[int, ...],
    similarities: NDArray[np.float64],
    pages: Sequence[SelectedCategoryPage],
    policy: SubdivisionPolicy,
    *,
    threshold: float,
) -> list[tuple[int, ...]]:
    if _estimated_input_tokens(component, pages, policy) <= policy.max_input_tokens:
        return [component]
    for index in component:
        if _estimated_input_tokens((index,), pages, policy) > policy.max_input_tokens:
            raise V2ContractError(
                f"Canonical Page {pages[index].page_id} exceeds the model-input token guard"
            )

    stricter_threshold = min(1.0, threshold + policy.stricter_threshold_step)
    neighbors = {index: set() for index in component}
    component_similarities = similarities[np.ix_(component, component)]
    for local_left, local_right in np.argwhere(
        np.triu(component_similarities >= stricter_threshold, k=1)
    ):
        left = component[int(local_left)]
        right = component[int(local_right)]
        neighbors[left].add(right)
        neighbors[right].add(left)
    subdivisions = _connected_components(component, neighbors)
    if len(subdivisions) == 1:
        if stricter_threshold < 1.0:
            return _subdivide_component(
                component,
                similarities,
                pages,
                policy,
                threshold=stricter_threshold,
            )
        return _pack_indistinguishable_members(component, pages, policy)
    return [
        nested
        for subdivision in subdivisions
        for nested in _subdivide_component(
            subdivision,
            similarities,
            pages,
            policy,
            threshold=stricter_threshold,
        )
    ]


def _estimated_input_tokens(
    component: Sequence[int],
    pages: Sequence[SelectedCategoryPage],
    policy: SubdivisionPolicy,
) -> int:
    records = (
        json.dumps(
            {
                "page_id": pages[index].page_id,
                "canonical_title": pages[index].canonical_title,
                "lead": pages[index].lead,
                "selected_categories": pages[index].selected_categories,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        for index in component
    )
    record_sizes = [len(record) for record in records]
    return (
        policy.fixed_prompt_tokens
        + 2
        + sum(record_sizes)
        + max(0, len(record_sizes) - 1)
    )


def _pack_indistinguishable_members(
    component: tuple[int, ...],
    pages: Sequence[SelectedCategoryPage],
    policy: SubdivisionPolicy,
) -> list[tuple[int, ...]]:
    groups: list[tuple[int, ...]] = []
    current: tuple[int, ...] = ()
    for index in component:
        candidate = (*current, index)
        if (
            current
            and _estimated_input_tokens(candidate, pages, policy)
            > policy.max_input_tokens
        ):
            groups.append(current)
            current = (index,)
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups
