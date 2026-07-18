from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Callable, Protocol, Sequence

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


# Configure the token guard and stricter-boundary subdivision strategy.
@dataclass(frozen=True)
class SubdivisionPolicy:
    max_input_tokens: int = DEFAULT_MAX_MODEL_INPUT_TOKENS
    fixed_prompt_tokens: int = DEFAULT_FIXED_PROMPT_TOKENS
    stricter_threshold_step: float = DEFAULT_STRICTER_THRESHOLD_STEP
    method: str = field(default="stricter-boundary", init=False)
    token_estimation: str = field(default="utf8-bytes-upper-bound", init=False)

    # Validate the initialized SubdivisionPolicy.
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


# Define the embedding boundary used by semantic clustering.
class EmbeddingAdapter(Protocol):
    model: str

    # Generate embedding vectors for the supplied representations.
    def embed(
        self, representations: Sequence[str]
    ) -> Sequence[Sequence[float]] | NDArray[np.float64]: ...


# Represent one semantically connected group before adjudication.
@dataclass(frozen=True)
class PreliminaryCluster:
    members: tuple[SelectedCategoryPage, ...]
    cohesion: float | None
    source_component_page_ids: tuple[int, ...]
    subdivision_threshold: float | None

    # Return the cluster member page IDs.
    @property
    def page_ids(self) -> tuple[int, ...]:
        return tuple(member.page_id for member in self.members)


# Capture clustering output and its reproducibility metadata.
@dataclass(frozen=True)
class PreliminaryClusterArtifact:
    embedding_model: str
    content_weight: float
    category_weight: float
    threshold: float
    subdivision_policy: SubdivisionPolicy
    preliminary_clusters: tuple[PreliminaryCluster, ...]
    singleton_count: int
    subdivided_component_count: int
    subdivision_count: int
    singleton_subdivision_count: int

    # Return the minimal serializable evidence produced by fixture formation.
    def record(self) -> dict[str, object]:
        """Return the minimal serializable evidence produced by fixture formation."""
        return asdict(self)


# Form and rank Preliminary Clusters without traffic evidence.
def form_preliminary_clusters(
    pages: Sequence[SelectedCategoryPage],
    embedding_adapter: EmbeddingAdapter,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    subdivision_policy: SubdivisionPolicy = SubdivisionPolicy(),
    progress: Callable[[str, str], None] | None = None,
) -> PreliminaryClusterArtifact:
    """Form and rank Preliminary Clusters without traffic evidence."""
    # Stable page ordering makes vector rows, graph nodes, and output ties reproducible.
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
    # Embed content and category evidence separately so each signal keeps its weight.
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
    if progress is not None:
        progress(
            "embed-representations",
            "embedded content and category representations using model "
            f"{embedding_adapter.model!r}",
        )

    # Normalized dot products are cosine similarities; clipping absorbs float drift.
    content_similarity = np.clip(content_vectors @ content_vectors.T, -1.0, 1.0)
    category_similarity = np.clip(category_vectors @ category_vectors.T, -1.0, 1.0)
    combined_similarities = (
        CONTENT_WEIGHT * content_similarity + CATEGORY_WEIGHT * category_similarity
    )
    # An inclusive threshold turns the blended matrix into an undirected graph.
    neighbors = {index: set() for index in range(len(ordered_pages))}
    edge_indices = np.argwhere(
        np.triu(combined_similarities >= threshold, k=1)
    )
    for left, right in edge_indices:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))

    components = _connected_components(tuple(range(len(ordered_pages))), neighbors)

    singleton_count = sum(len(component) == 1 for component in components)
    if progress is not None:
        progress(
            "form-components",
            f"formed {len(components)} connected components at inclusive threshold "
            f"{threshold:g}; discarded "
            f"{singleton_count} singleton components",
        )
    subdivider = _ComponentSubdivider(
        combined_similarities,
        ordered_pages,
        subdivision_policy,
    )
    # Singletons have no audience relationship to adjudicate, while large components
    # must be subdivided until their complete evidence fits the model input guard.
    reviewable_components: list[
        tuple[ComponentSubdivision, tuple[int, ...]]
    ] = []
    subdivided_component_count = 0
    subdivision_count = 0
    for component in components:
        if len(component) == 1:
            continue
        subdivisions = subdivider.subdivide(component, threshold=threshold)
        reviewable_components.extend(
            (subdivision, component) for subdivision in subdivisions
        )
        if not (
            len(subdivisions) == 1
            and subdivisions[0].indices == component
            and subdivisions[0].similarity_threshold is None
        ):
            subdivided_component_count += 1
            subdivision_count += len(subdivisions)
    singleton_subdivision_count = sum(
        len(subdivision.indices) == 1
        for subdivision, _source_component in reviewable_components
    )
    if progress is not None:
        progress(
            "subdivide-components",
            f"subdivided {subdivided_component_count} oversized components into "
            f"{subdivision_count} subdivisions using the "
            f"{subdivision_policy.max_input_tokens}-token {subdivision_policy.method} guard",
        )
    # Preserve source-component provenance so later stages cannot move pages between
    # semantic components while revising a cluster.
    clusters = [
        PreliminaryCluster(
            members=tuple(
                ordered_pages[index] for index in subdivision.indices
            ),
            cohesion=_mean_pairwise_similarity(
                subdivision.indices, combined_similarities
            ),
            source_component_page_ids=tuple(
                ordered_pages[index].page_id for index in source_component
            ),
            subdivision_threshold=subdivision.similarity_threshold,
        )
        for subdivision, source_component in reviewable_components
    ]
    # Rank deterministically by cohesion, size, and finally page identity.
    clusters.sort(
        key=lambda cluster: (
            cluster.cohesion is None,
            -(cluster.cohesion or 0.0),
            -len(cluster.members),
            cluster.page_ids,
        )
    )
    if progress is not None:
        progress(
            "rank-clusters",
            f"ranked {len(clusters)} eligible Preliminary Clusters by whole-component cohesion",
        )
    return PreliminaryClusterArtifact(
        embedding_model=embedding_adapter.model,
        content_weight=CONTENT_WEIGHT,
        category_weight=CATEGORY_WEIGHT,
        threshold=threshold,
        subdivision_policy=subdivision_policy,
        preliminary_clusters=tuple(clusters),
        singleton_count=singleton_count,
        subdivided_component_count=subdivided_component_count,
        subdivision_count=subdivision_count,
        singleton_subdivision_count=singleton_subdivision_count,
    )


# Normalize and validate an embedding matrix.
def _validated_embedding_matrix(
    vectors: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    expected_count: int,
) -> NDArray[np.float64]:
    # Validate shape before numeric work so malformed adapters fail deterministically.
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
    # Scale before norm calculation to avoid overflow for large finite vectors.
    scales = np.max(np.abs(matrix), axis=1)
    if np.any(scales == 0):
        raise V2ContractError("embeddings must have non-zero magnitude")
    scaled_matrix = matrix / scales[:, np.newaxis]
    magnitudes = np.linalg.norm(scaled_matrix, axis=1)
    return scaled_matrix / magnitudes[:, np.newaxis]


# Calculate mean pairwise similarity for a component.
def _mean_pairwise_similarity(
    component: Sequence[int], similarities: NDArray[np.float64]
) -> float | None:
    if len(component) == 1:
        return None
    component_similarities = similarities[np.ix_(component, component)]
    pair_values = component_similarities[np.triu_indices(len(component), k=1)]
    return float(np.mean(pair_values))


# Find connected components in a similarity graph.
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


# Return accepted subgroups and rejected pages from one component split.
@dataclass(frozen=True)
class ComponentSubdivision:
    indices: tuple[int, ...]
    similarity_threshold: float | None


# Recursively fit connected components within the model input budget.
@dataclass(frozen=True)
class _ComponentSubdivider:
    similarities: NDArray[np.float64]
    pages: Sequence[SelectedCategoryPage]
    policy: SubdivisionPolicy

    # Recursively subdivide a component until it fits the model guard.
    def subdivide(
        self,
        component: tuple[int, ...],
        *,
        threshold: float,
        subdivision_threshold: float | None = None,
    ) -> list[ComponentSubdivision]:
        # A component that already fits remains intact and needs no threshold provenance.
        if self._estimated_input_tokens(component) <= self.policy.max_input_tokens:
            return [ComponentSubdivision(component, subdivision_threshold)]
        for index in component:
            if self._estimated_input_tokens((index,)) > self.policy.max_input_tokens:
                raise V2ContractError(
                    f"Canonical Page {self.pages[index].page_id} exceeds "
                    "the model-input token guard"
                )

        # Raising the edge threshold breaks weak semantic bridges without truncating
        # any page evidence.
        stricter_threshold = min(
            1.0,
            threshold + self.policy.stricter_threshold_step,
        )
        neighbors = {index: set() for index in component}
        component_similarities = self.similarities[np.ix_(component, component)]
        for local_left, local_right in np.argwhere(
            np.triu(component_similarities >= stricter_threshold, k=1)
        ):
            left = component[int(local_left)]
            right = component[int(local_right)]
            neighbors[left].add(right)
            neighbors[right].add(left)
        subdivisions = _connected_components(component, neighbors)
        # Recurse until the graph separates; exact duplicates at threshold 1.0 are
        # packed deterministically as the final lossless fallback.
        if len(subdivisions) == 1:
            if stricter_threshold < 1.0:
                return self.subdivide(
                    component,
                    threshold=stricter_threshold,
                    subdivision_threshold=stricter_threshold,
                )
            return [
                ComponentSubdivision(group, 1.0)
                for group in self._pack_indistinguishable_members(component)
            ]
        return [
            nested
            for subdivision in subdivisions
            for nested in self.subdivide(
                subdivision,
                threshold=stricter_threshold,
                subdivision_threshold=stricter_threshold,
            )
        ]

    # Estimate the model input tokens for a group of pages.
    def _estimated_input_tokens(self, component: Sequence[int]) -> int:
        records = (
            json.dumps(
                {
                    "page_id": self.pages[index].page_id,
                    "canonical_title": self.pages[index].canonical_title,
                    "lead": self.pages[index].lead,
                    "selected_categories": self.pages[index].selected_categories,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            for index in component
        )
        record_sizes = [len(record) for record in records]
        return (
            self.policy.fixed_prompt_tokens
            + 2
            + sum(record_sizes)
            + max(0, len(record_sizes) - 1)
        )

    # Pack inseparable members within the model token budget.
    def _pack_indistinguishable_members(
        self,
        component: tuple[int, ...],
    ) -> list[tuple[int, ...]]:
        groups: list[tuple[int, ...]] = []
        current: tuple[int, ...] = ()
        for index in component:
            candidate = (*current, index)
            if (
                current
                and self._estimated_input_tokens(candidate)
                > self.policy.max_input_tokens
            ):
                groups.append(current)
                current = (index,)
            else:
                current = candidate
        if current:
            groups.append(current)
        return groups
