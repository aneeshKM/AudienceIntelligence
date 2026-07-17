from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import jsonschema

from audience_trend_miner.v2.semantic_audience_formation.categories import (
    CATEGORY_RULE_SET_VERSION,
    CategorySelection,
    select_categories,
)
from audience_trend_miner.v2.semantic_audience_formation.clustering import (
    CATEGORY_WEIGHT,
    CONTENT_WEIGHT,
    EmbeddingAdapter,
    SubdivisionPolicy,
    form_preliminary_clusters,
)
from audience_trend_miner.v2.shared import (
    ARTIFACT_SCHEMA_VERSION,
    BoundedProgress,
    ProgressEvent,
    ProgressSink,
    V2ContractError,
    atomic_write_json,
    consume_artifact,
    validate_artifact,
    validate_schema,
)
from audience_trend_miner.v2.wikimedia_evidence import consume_wikimedia_evidence


STAGE = "semantic-audience-formation"
SCHEMA_PATH = Path(__file__).with_name("schemas") / "semantic-audience-formation.schema.json"
DEFAULT_MAX_LLM_CLUSTERS = 10
ReviewCap = int | Literal["all"]


def parse_review_cap(value: object) -> ReviewCap:
    """Parse the configured Cluster Adjudication review budget."""
    if value == "all":
        return "all"
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise V2ContractError("review cap must be a positive integer or 'all'")
    parsed = int(value)
    if parsed <= 0:
        raise V2ContractError("review cap must be a positive integer or 'all'")
    return parsed


def execute_category_selection(
    *,
    run_id: str,
    output_root: Path,
    progress_sink: ProgressSink,
    wikimedia_evidence_path: Path | None = None,
    event_progress: BoundedProgress | None = None,
) -> CategorySelection:
    """Validate Wikimedia Evidence and form deterministic Selected Categories."""
    evidence_path = wikimedia_evidence_path or (
        output_root / run_id / "wikimedia-evidence.json"
    )
    artifact = consume_wikimedia_evidence(evidence_path, run_id=run_id)
    payload = artifact["payload"]
    assert isinstance(payload, dict)
    canonical_pages = payload["canonical_pages"]
    assert isinstance(canonical_pages, list)
    selection = select_categories(canonical_pages)
    page_count = len(selection.pages)
    progress_sink(
        ProgressEvent(
            run_id=run_id,
            sequence=1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=STAGE,
            operation="select-categories",
            level="info",
            message=(
                "selected meaningful categories with rule set "
                f"{CATEGORY_RULE_SET_VERSION}"
            ),
            progress=event_progress
            or BoundedProgress(page_count, max(page_count, 1)),
        )
    )
    return selection


def execute_preliminary_clustering(
    *,
    run_id: str,
    output_root: Path,
    progress_sink: ProgressSink,
    embedding_adapter: EmbeddingAdapter,
    threshold: float,
    max_llm_clusters: ReviewCap = DEFAULT_MAX_LLM_CLUSTERS,
    wikimedia_evidence_path: Path | None = None,
    interrupt_before_completion: bool = False,
) -> Path:
    """Form, cap, and atomically publish Preliminary Cluster evidence."""
    if not (
        max_llm_clusters == "all"
        or (
            isinstance(max_llm_clusters, int)
            and not isinstance(max_llm_clusters, bool)
            and max_llm_clusters > 0
        )
    ):
        raise V2ContractError("review cap must be a positive integer or 'all'")
    subdivision_policy = SubdivisionPolicy()
    requested_configuration = _formation_configuration(
        embedding_model=embedding_adapter.model,
        threshold=threshold,
        subdivision_policy=subdivision_policy,
        review_cap=max_llm_clusters,
    )
    artifact_path = output_root / run_id / f"{STAGE}.json"
    if artifact_path.exists():
        completed = consume_artifact(artifact_path, run_id=run_id, stage=STAGE)
        try:
            validate_schema(SCHEMA_PATH, completed["payload"])
        except jsonschema.ValidationError as error:
            raise V2ContractError(
                f"Semantic Audience Formation is schema-incompatible: {error.message}"
            ) from error
        payload = completed["payload"]
        assert isinstance(payload, dict)
        if payload["configuration"] != requested_configuration:
            raise V2ContractError(
                "completed Semantic Audience Formation artifact conflicts with "
                "requested configuration"
            )
        progress_sink(
            ProgressEvent(
                run_id=run_id,
                sequence=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                module=STAGE,
                operation="resume",
                level="info",
                message="resumed compatible completed Semantic Audience Formation",
                progress=BoundedProgress(1, 1),
            )
        )
        return artifact_path

    selection = execute_category_selection(
        run_id=run_id,
        output_root=output_root,
        progress_sink=progress_sink,
        wikimedia_evidence_path=wikimedia_evidence_path,
        event_progress=BoundedProgress(1, 7),
    )
    sequence = 1

    def emit_formation_progress(operation: str, message: str) -> None:
        nonlocal sequence
        sequence += 1
        progress_sink(
            ProgressEvent(
                run_id=run_id,
                sequence=sequence,
                timestamp=datetime.now(timezone.utc).isoformat(),
                module=STAGE,
                operation=operation,
                level="info",
                message=message,
                progress=BoundedProgress(sequence, 7),
            )
        )

    result = form_preliminary_clusters(
        selection.pages,
        embedding_adapter,
        threshold=threshold,
        subdivision_policy=subdivision_policy,
        progress=emit_formation_progress,
    )
    selected_clusters = (
        result.preliminary_clusters
        if max_llm_clusters == "all"
        else result.preliminary_clusters[:max_llm_clusters]
    )
    eligible_count = len(result.preliminary_clusters)
    selected_count = len(selected_clusters)
    emit_formation_progress(
        "select-review-cap",
        f"selected {selected_count} of {eligible_count} eligible Preliminary Clusters "
        f"with review cap {max_llm_clusters!r}",
    )
    payload = {
        "configuration": requested_configuration,
        "counts": {
            "eligible_clusters": eligible_count,
            "selected_clusters": selected_count,
            "omitted_clusters": eligible_count - selected_count,
            "discarded_singleton_components": result.singleton_count,
            "subdivided_components": result.subdivided_component_count,
            "subdivisions_created": result.subdivision_count,
        },
        "preliminary_clusters": [
            {
                "cohesion": cluster.cohesion,
                "members": [
                    {
                        "page_id": member.page_id,
                        "canonical_title": member.canonical_title,
                        "lead": member.lead,
                        "selected_categories": list(member.selected_categories),
                    }
                    for member in cluster.members
                ],
            }
            for cluster in selected_clusters
        ],
        "completion": {"status": "complete"},
    }
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "complete",
        "payload": payload,
    }
    try:
        validate_schema(SCHEMA_PATH, payload)
    except jsonschema.ValidationError as error:
        raise V2ContractError(
            f"Semantic Audience Formation is schema-invalid: {error.message}"
        ) from error
    validate_artifact(artifact, run_id=run_id, stage=STAGE)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        artifact_path,
        artifact,
        interrupt_before_replace=interrupt_before_completion,
    )
    progress_sink(
        ProgressEvent(
            run_id=run_id,
            sequence=7,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=STAGE,
            operation="publish",
            level="info",
            message=(
                f"published {selected_count} of {eligible_count} eligible "
                "Preliminary Clusters"
            ),
            progress=BoundedProgress(7, 7),
        )
    )
    return artifact_path


def _formation_configuration(
    *,
    embedding_model: str,
    threshold: float,
    subdivision_policy: SubdivisionPolicy,
    review_cap: ReviewCap,
) -> dict[str, object]:
    return {
        "category_rule_set_version": CATEGORY_RULE_SET_VERSION,
        "embedding_model": embedding_model,
        "content_weight": CONTENT_WEIGHT,
        "category_weight": CATEGORY_WEIGHT,
        "similarity_threshold": threshold,
        "subdivision_policy": asdict(subdivision_policy),
        "review_cap": review_cap,
    }
