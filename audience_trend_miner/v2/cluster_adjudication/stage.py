from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Protocol, TypedDict, cast

import jsonschema

from audience_trend_miner.v2.cluster_adjudication.graph import (
    AdjudicationAdapter,
    ModelStepRecord,
    execute_cluster_adjudication,
)
from audience_trend_miner.v2.semantic_audience_formation.stage import (
    SCHEMA_PATH as FORMATION_SCHEMA_PATH,
)
from audience_trend_miner.v2.shared import (
    ARTIFACT_SCHEMA_VERSION,
    BoundedProgress,
    ProgressEvent,
    ProgressSink,
    V2ContractError,
    atomic_write_json,
    canonical_json_fingerprint,
    consume_artifact,
    validate_artifact,
    validate_schema,
)


STAGE = "cluster-adjudication"
SCHEMA_PATH = Path(__file__).with_name("schemas") / "cluster-adjudication.schema.json"


# Define how the stage obtains an adapter for each preliminary cluster.
class StageAdapterFactory(Protocol):
    # Identify the model recorded in the stage configuration.
    @property
    def model(self) -> str: ...

    # Identify the provider integration recorded in the stage configuration.
    @property
    def integration_name(self) -> str: ...

    # Build the adapter used to adjudicate one preliminary cluster.
    def adapter_for(
        self, cluster_index: int, preliminary_cluster: dict[str, object]
    ) -> AdjudicationAdapter: ...


# Store resumable adjudication output with its source-component identity.
class CompletedClusterRecord(TypedDict):
    preliminary_cluster_id: str
    final_audience_clusters: list[dict[str, object]]
    rejected_members: list[dict[str, object]]
    adjudication: dict[str, object]


# Adjudicate all preliminary clusters and atomically publish the stage artifact.
def execute_cluster_adjudication_stage(
    *,
    run_id: str,
    output_root: Path,
    progress_sink: ProgressSink,
    adapter_factory: StageAdapterFactory,
    semantic_formation_path: Path | None = None,
    interrupt_before_completion: bool = False,
) -> Path:
    """Adjudicate every selected Preliminary Cluster and publish atomically."""
    # The formation fingerprint binds this stage to one exact selected-cluster input.
    formation_path = semantic_formation_path or (
        output_root / run_id / "semantic-audience-formation.json"
    )
    formation = consume_artifact(
        formation_path, run_id=run_id, stage="semantic-audience-formation"
    )
    try:
        validate_schema(FORMATION_SCHEMA_PATH, formation["payload"])
    except jsonschema.ValidationError as error:
        raise V2ContractError(
            f"Semantic Audience Formation is schema-incompatible: {error.message}"
        ) from error
    formation_payload = formation["payload"]
    assert isinstance(formation_payload, dict)
    preliminary_clusters = formation_payload["preliminary_clusters"]
    assert isinstance(preliminary_clusters, list)
    typed_clusters = [cluster for cluster in preliminary_clusters if isinstance(cluster, dict)]
    if len(typed_clusters) != len(preliminary_clusters):
        raise V2ContractError("Semantic Audience Formation contains invalid clusters")
    _validate_exclusive_input_pages(typed_clusters)

    framework = {"name": "langgraph", "version": _package_version("langgraph")}
    configuration: dict[str, object] = {
        "model": adapter_factory.model,
        "framework": framework,
        "integration": {
            "name": adapter_factory.integration_name,
            "version": (
                "1.0"
                if adapter_factory.integration_name == "fixture"
                else _package_version(adapter_factory.integration_name)
            ),
        },
        "semantic_audience_formation_fingerprint": canonical_json_fingerprint(
            formation
        ),
    }
    artifact_path = output_root / run_id / f"{STAGE}.json"
    checkpoint_path = output_root / run_id / f".{STAGE}.checkpoint.json"
    sequence = 0
    total_clusters = len(typed_clusters)
    progress_total = max(total_clusters, 1)

    # Emit one ordered, bounded progress event for the current cluster.
    def emit(operation: str, message: str, cluster_number: int) -> None:
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
                progress=BoundedProgress(min(cluster_number, progress_total), progress_total),
            )
        )

    # Completed output is reusable only after schema, configuration, and membership checks.
    if artifact_path.exists():
        completed_artifact = consume_artifact(
            artifact_path, run_id=run_id, stage=STAGE
        )
        try:
            validate_schema(SCHEMA_PATH, completed_artifact["payload"])
        except jsonschema.ValidationError as error:
            raise V2ContractError(
                f"Cluster Adjudication is schema-incompatible: {error.message}"
            ) from error
        completed_payload = completed_artifact["payload"]
        assert isinstance(completed_payload, dict)
        if completed_payload["configuration"] != configuration:
            raise V2ContractError(
                "completed Cluster Adjudication artifact conflicts with requested configuration"
            )
        _validate_completed_payload(typed_clusters, completed_payload)
        emit("resume", "resumed compatible completed Cluster Adjudication", progress_total)
        return artifact_path

    final_clusters: list[dict[str, object]] = []
    rejected_members: list[dict[str, object]] = []
    adjudications: list[dict[str, object]] = []
    # Rehydrate already validated cluster records before requesting any new model work.
    completed_records = _load_checkpoint(
        checkpoint_path,
        run_id=run_id,
        configuration=configuration,
        preliminary_clusters=typed_clusters,
    )
    for completed_record in completed_records:
        final_clusters.extend(completed_record["final_audience_clusters"])
        rejected_members.extend(completed_record["rejected_members"])
        adjudications.append(completed_record["adjudication"])
        emit(
            "resume-cluster",
            f"resumed {completed_record['preliminary_cluster_id']}",
            len(adjudications),
        )

    # Each preliminary cluster is an isolated resumable unit of adjudication.
    for cluster_index in range(len(completed_records), total_clusters):
        preliminary_cluster = typed_clusters[cluster_index]
        cluster_number = cluster_index + 1
        preliminary_cluster_id = f"preliminary-cluster-{cluster_number:04d}"

        # Translate a graph model-step update into stage progress.
        def report_step(step: ModelStepRecord) -> None:
            emit(
                step.role,
                f"{preliminary_cluster_id} {step.role} {step.status} after "
                f"{len(step.attempts)} delivery attempt(s); validation "
                f"{step.validation_status}",
                cluster_number,
            )

        result = execute_cluster_adjudication(
            preliminary_cluster,
            adapter_factory.adapter_for(cluster_index, preliminary_cluster),
            step_progress=report_step,
        )
        # Assign stable global final-cluster IDs only after graph validation succeeds.
        completed_groups: list[dict[str, object]] = []
        for group in result.accepted_groups:
            completed_groups.append(
                {
                    "cluster_id": f"final-audience-cluster-{len(final_clusters) + len(completed_groups) + 1:04d}",
                    "source_preliminary_cluster_id": preliminary_cluster_id,
                    **group.record(),
                }
            )
        completed_rejections: list[dict[str, object]] = [
            {
                **member,
                "source_preliminary_cluster_id": preliminary_cluster_id,
            }
            for member in result.rejected_members
        ]
        completed_adjudication: dict[str, object] = {
            "preliminary_cluster_id": preliminary_cluster_id,
            "steps": [step.record() for step in result.steps],
            "validation": {
                "status": result.validation_status,
                "errors": list(result.validation_errors),
            },
        }
        new_checkpoint_record = CompletedClusterRecord(
            preliminary_cluster_id=preliminary_cluster_id,
            final_audience_clusters=completed_groups,
            rejected_members=completed_rejections,
            adjudication=completed_adjudication,
        )
        completed_records.append(new_checkpoint_record)
        final_clusters.extend(completed_groups)
        rejected_members.extend(completed_rejections)
        adjudications.append(completed_adjudication)
        # Checkpoint after every cluster so provider calls never need to be repeated.
        atomic_write_json(
            checkpoint_path,
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "configuration": configuration,
                "completed": completed_records,
            },
        )

    accepted_page_count = sum(
        len(cast(list[object], cluster["members"])) for cluster in final_clusters
    )
    # The final artifact separates product membership from auditable model-step evidence.
    payload = {
        "configuration": configuration,
        "counts": {
            "preliminary_clusters": total_clusters,
            "final_audience_clusters": len(final_clusters),
            "accepted_pages": accepted_page_count,
            "rejected_pages": len(rejected_members),
        },
        "final_audience_clusters": final_clusters,
        "rejected_members": rejected_members,
        "adjudications": adjudications,
        "completion": {"status": "complete"},
    }
    try:
        validate_schema(SCHEMA_PATH, payload)
    except jsonschema.ValidationError as error:
        raise V2ContractError(
            f"Cluster Adjudication is schema-invalid: {error.message}"
        ) from error
    _validate_completed_payload(typed_clusters, payload)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "complete",
        "payload": payload,
    }
    validate_artifact(artifact, run_id=run_id, stage=STAGE)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement is the completion boundary; the hidden checkpoint is then stale.
    atomic_write_json(
        artifact_path,
        artifact,
        interrupt_before_replace=interrupt_before_completion,
    )
    emit("publish", f"published {len(final_clusters)} Final Audience Clusters", progress_total)
    checkpoint_path.unlink(missing_ok=True)
    return artifact_path


# Ensure no page appears in more than one preliminary cluster.
def _validate_exclusive_input_pages(clusters: list[dict[str, object]]) -> None:
    page_ids: list[object] = []
    for cluster in clusters:
        members = cluster.get("members")
        assert isinstance(members, list)
        page_ids.extend(
            member.get("page_id") for member in members if isinstance(member, dict)
        )
    if len(page_ids) != len(set(page_ids)):
        raise V2ContractError("selected Preliminary Clusters contain duplicate page IDs")


# Verify output IDs, provenance, counts, and terminal page membership.
def _validate_completed_payload(
    preliminary_clusters: list[dict[str, object]],
    payload: dict[str, object],
) -> None:
    # Stable ordered IDs make missing, reordered, or injected records detectable.
    final_clusters = cast(list[dict[str, object]], payload["final_audience_clusters"])
    rejected_members = cast(list[dict[str, object]], payload["rejected_members"])
    adjudications = cast(list[dict[str, object]], payload["adjudications"])
    expected_sources = [
        f"preliminary-cluster-{index:04d}"
        for index in range(1, len(preliminary_clusters) + 1)
    ]
    if [item["preliminary_cluster_id"] for item in adjudications] != expected_sources:
        raise V2ContractError(
            "Cluster Adjudication provenance does not match Preliminary Clusters"
        )
    if [cluster["cluster_id"] for cluster in final_clusters] != [
        f"final-audience-cluster-{index:04d}"
        for index in range(1, len(final_clusters) + 1)
    ]:
        raise V2ContractError("Cluster Adjudication Final Audience Cluster IDs are invalid")
    output_sources = {
        cast(str, item["source_preliminary_cluster_id"])
        for item in [*final_clusters, *rejected_members]
    }
    if not output_sources.issubset(set(expected_sources)):
        raise V2ContractError(
            "Cluster Adjudication provenance does not match Preliminary Clusters"
        )
    # Validate terminal exclusivity independently within every source component.
    for source_id, preliminary_cluster in zip(
        expected_sources, preliminary_clusters, strict=True
    ):
        _validate_component_terminal_membership(
            preliminary_cluster,
            [
                cluster
                for cluster in final_clusters
                if cluster["source_preliminary_cluster_id"] == source_id
            ],
            [
                member
                for member in rejected_members
                if member["source_preliminary_cluster_id"] == source_id
            ],
            context="Cluster Adjudication",
        )
    accepted_page_count = sum(
        len(cast(list[dict[str, object]], cluster["members"]))
        for cluster in final_clusters
    )
    expected_counts = {
        "preliminary_clusters": len(preliminary_clusters),
        "final_audience_clusters": len(final_clusters),
        "accepted_pages": accepted_page_count,
        "rejected_pages": len(rejected_members),
    }
    if payload["counts"] != expected_counts:
        raise V2ContractError("Cluster Adjudication counts do not match membership")


# Ensure each input page is accepted or rejected exactly once.
def _validate_component_terminal_membership(
    preliminary_cluster: dict[str, object],
    final_clusters: list[dict[str, object]],
    rejected_members: list[dict[str, object]],
    *,
    context: str,
) -> None:
    supplied_members = cast(
        list[dict[str, object]], preliminary_cluster["members"]
    )
    supplied_ids = [member["page_id"] for member in supplied_members]
    terminal_ids: list[object] = []
    for final_cluster in final_clusters:
        members = final_cluster.get("members")
        if not isinstance(members, list) or any(
            not isinstance(member, dict) or "page_id" not in member
            for member in members
        ):
            raise V2ContractError(f"{context} terminal membership is invalid")
        terminal_ids.extend(member["page_id"] for member in members)
    if any("page_id" not in member for member in rejected_members):
        raise V2ContractError(f"{context} terminal membership is invalid")
    terminal_ids.extend(member["page_id"] for member in rejected_members)
    if (
        len(terminal_ids) != len(set(terminal_ids))
        or set(terminal_ids) != set(supplied_ids)
    ):
        raise V2ContractError(f"{context} terminal membership is not exclusive")


# Return an installed package version or an unknown marker.
def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


# Load and validate resumable records from a compatible checkpoint.
def _load_checkpoint(
    path: Path,
    *,
    run_id: str,
    configuration: dict[str, object],
    preliminary_clusters: list[dict[str, object]],
) -> list[CompletedClusterRecord]:
    # A checkpoint is trusted only when its run and complete effective configuration match.
    if not path.exists():
        return []
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ContractError("Cluster Adjudication checkpoint is unreadable") from error
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"schema_version", "run_id", "configuration", "completed"}
        or checkpoint["schema_version"] != "1.0"
        or checkpoint["run_id"] != run_id
        or checkpoint["configuration"] != configuration
        or not isinstance(checkpoint["completed"], list)
        or len(checkpoint["completed"]) > len(preliminary_clusters)
    ):
        raise V2ContractError(
            "Cluster Adjudication checkpoint conflicts with requested configuration"
        )
    completed = checkpoint["completed"]
    # Validate records in prefix order so resumption always continues at one clear index.
    for index, record in enumerate(completed, start=1):
        if (
            not isinstance(record, dict)
            or set(record) != {
                "preliminary_cluster_id",
                "final_audience_clusters",
                "rejected_members",
                "adjudication",
            }
            or record["preliminary_cluster_id"] != f"preliminary-cluster-{index:04d}"
            or not isinstance(record["final_audience_clusters"], list)
            or not isinstance(record["rejected_members"], list)
            or not isinstance(record["adjudication"], dict)
        ):
            raise V2ContractError("Cluster Adjudication checkpoint is invalid")
        typed_record = cast(dict[str, object], record)
        final_clusters = cast(
            list[dict[str, object]], typed_record["final_audience_clusters"]
        )
        rejected_members = cast(
            list[dict[str, object]], typed_record["rejected_members"]
        )
        if any(
            not isinstance(item, dict)
            or item.get("source_preliminary_cluster_id")
            != f"preliminary-cluster-{index:04d}"
            for item in [*final_clusters, *rejected_members]
        ):
            raise V2ContractError("Cluster Adjudication checkpoint provenance is invalid")
        adjudication = cast(dict[str, object], typed_record["adjudication"])
        if adjudication.get("preliminary_cluster_id") != record["preliminary_cluster_id"]:
            raise V2ContractError("Cluster Adjudication checkpoint provenance is invalid")
        _validate_component_terminal_membership(
            preliminary_clusters[index - 1],
            final_clusters,
            rejected_members,
            context="Cluster Adjudication checkpoint",
        )
    return cast(list[CompletedClusterRecord], completed)
