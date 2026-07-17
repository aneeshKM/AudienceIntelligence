from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import cast

import jsonschema

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
from audience_trend_miner.v2.trend_portfolio.narratives import (
    NARRATIVE_PROMPT,
    NarrativeAdapterFactory,
    NarrativeExhausted,
    generate_validated_narrative,
    narrative_validation_errors,
)
from audience_trend_miner.v2.trend_portfolio.portfolio import (
    AudienceTrend,
    qualify_and_rank_portfolio,
)
from audience_trend_miner.v2.trend_portfolio.traffic import (
    ClusterTraffic,
    attach_cluster_traffic,
)


STAGE = "trend-portfolio"
SCHEMA_PATH = Path(__file__).with_name("schemas") / "trend-portfolio.schema.json"


def execute_trend_portfolio_stage(
    *,
    run_id: str,
    output_root: Path,
    progress_sink: ProgressSink,
    adapter_factory: NarrativeAdapterFactory,
    wikimedia_evidence_path: Path | None = None,
    cluster_adjudication_path: Path | None = None,
    interrupt_before_completion: bool = False,
) -> Path:
    """Attach, qualify, narrate, and atomically publish one Audience Portfolio."""
    run_directory = output_root / run_id
    evidence_path = wikimedia_evidence_path or run_directory / "wikimedia-evidence.json"
    adjudication_path = (
        cluster_adjudication_path or run_directory / "cluster-adjudication.json"
    )
    evidence_artifact = consume_artifact(
        evidence_path, run_id=run_id, stage="wikimedia-evidence"
    )
    adjudication_artifact = consume_artifact(
        adjudication_path, run_id=run_id, stage="cluster-adjudication"
    )
    evidence_payload = cast(dict[str, object], evidence_artifact["payload"])
    adjudication_payload = cast(dict[str, object], adjudication_artifact["payload"])
    source_clusters = {
        cast(str, cluster["cluster_id"]): cluster
        for cluster in cast(
            list[dict[str, object]],
            adjudication_payload["final_audience_clusters"],
        )
    }
    run_facts: dict[str, object] = {
        "as_of_date": evidence_payload["as_of_date"],
        "nominal_windows": evidence_payload["nominal_windows"],
    }
    configuration: dict[str, object] = {
        "model": adapter_factory.model,
        "integration": {
            "name": adapter_factory.integration_name,
            "version": (
                "1.0"
                if adapter_factory.integration_name == "fixture"
                else _package_version(adapter_factory.integration_name)
            ),
        },
        "wikimedia_evidence_fingerprint": canonical_json_fingerprint(evidence_artifact),
        "cluster_adjudication_fingerprint": canonical_json_fingerprint(
            adjudication_artifact
        ),
    }
    artifact_path = run_directory / f"{STAGE}.json"
    checkpoint_path = run_directory / f".{STAGE}.checkpoint.json"
    failure_path = run_directory / f".{STAGE}.failure.json"
    sequence = 0

    def emit(operation: str, message: str, current: int, total: int) -> None:
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
                progress=BoundedProgress(current, total),
            )
        )

    traffic = attach_cluster_traffic(
        run_id=run_id,
        wikimedia_evidence_path=evidence_path,
        cluster_adjudication_path=adjudication_path,
    )
    qualification = qualify_and_rank_portfolio(traffic)
    selected = qualification.portfolio.audience_trends

    if artifact_path.exists():
        artifact = consume_artifact(artifact_path, run_id=run_id, stage=STAGE)
        _validate_payload(artifact["payload"])
        payload = cast(dict[str, object], artifact["payload"])
        if payload["configuration"] != configuration:
            raise V2ContractError(
                "completed Trend Portfolio conflicts with requested configuration"
            )
        _validate_completed_facts(
            payload,
            run_facts,
            qualification.audit_cluster_traffic,
            selected,
            source_clusters,
        )
        failure_path.unlink(missing_ok=True)
        emit("resume", "resumed compatible completed Audience Portfolio", 1, 1)
        return artifact_path

    emit("attachment", "attached traffic to terminal membership", 1, 1)
    emit("qualification", f"qualified {len(selected)} robust clusters", 1, 1)
    emit("ranking", f"ranked and selected {len(selected)} clusters", 1, 1)

    completed = _load_checkpoint(
        checkpoint_path,
        run_id=run_id,
        configuration=configuration,
        selected=selected,
        source_clusters=source_clusters,
    )
    total = max(len(selected), 1)
    for index in range(len(completed), len(selected)):
        trend = selected[index]
        cluster_id = trend.final_cluster_traffic.cluster_id
        source_cluster = source_clusters[cluster_id]
        evidence = _model_evidence(
            trend,
            cast(list[dict[str, object]], source_cluster["members"]),
        )
        try:
            narrative, attempts = generate_validated_narrative(
                adapter_factory.adapter_for(index, cluster_id),
                evidence,
            )
        except NarrativeExhausted as error:
            atomic_write_json(
                failure_path,
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "configuration": configuration,
                    "cluster_id": cluster_id,
                    "prompt": NARRATIVE_PROMPT,
                    "model_input": evidence,
                    "model": adapter_factory.model,
                    "attempts": [attempt.record() for attempt in error.attempts],
                },
            )
            raise V2ContractError(
                f"narrative validation exhausted for {cluster_id}: "
                + "; ".join(error.attempts[-1].errors)
            ) from error
        record: dict[str, object] = {
            "cluster_id": cluster_id,
            "portfolio_item": _portfolio_item(trend, narrative),
            "evidence": {
                "cluster_id": cluster_id,
                "prompt": NARRATIVE_PROMPT,
                "model_input": evidence,
                "model": adapter_factory.model,
                "attempts": [attempt.record() for attempt in attempts],
            },
        }
        completed.append(record)
        atomic_write_json(
            checkpoint_path,
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "configuration": configuration,
                "completed": completed,
            },
        )
        emit(
            "narrative",
            f"validated narrative for {cluster_id} after {len(attempts)} attempt(s)",
            index + 1,
            total,
        )

    payload = {
        "configuration": configuration,
        "run_facts": run_facts,
        "counts": {"qualified": len(selected), "narrated": len(completed)},
        "audience_portfolio": [record["portfolio_item"] for record in completed],
        "narrative_evidence": [record["evidence"] for record in completed],
        "audit_cluster_traffic": [
            _traffic_record(item) for item in qualification.audit_cluster_traffic
        ],
        "completion": {"status": "complete"},
    }
    _validate_payload(payload)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "complete",
        "payload": payload,
    }
    validate_artifact(artifact, run_id=run_id, stage=STAGE)
    run_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        artifact_path,
        artifact,
        interrupt_before_replace=interrupt_before_completion,
    )
    emit("publish", "published complete Audience Portfolio", total, total)
    checkpoint_path.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    return artifact_path


def _model_evidence(
    trend: AudienceTrend,
    members: list[dict[str, object]],
) -> dict[str, object]:
    traffic = trend.final_cluster_traffic
    return {
        "source_cluster_name": traffic.name,
        "source_cluster_rationale": traffic.rationale,
        "members": members,
        "direction": traffic.direction,
    }


def _portfolio_item(
    trend: AudienceTrend, narrative: dict[str, object]
) -> dict[str, object]:
    traffic = trend.final_cluster_traffic
    return {
        "cluster_id": traffic.cluster_id,
        "source_preliminary_cluster_id": traffic.source_preliminary_cluster_id,
        **_deterministic_facts(trend),
        "narrative": narrative,
    }


def _deterministic_facts(trend: AudienceTrend) -> dict[str, object]:
    traffic = trend.final_cluster_traffic
    previous = traffic.previous.seven_day_equivalent
    current = traffic.current.seven_day_equivalent
    percentage_change = None if previous == 0 else (current - previous) / previous * 100
    members = len(traffic.member_page_ids)
    return {
        "direction": traffic.direction,
        "traffic": {
            "previous": asdict(traffic.previous),
            "current": asdict(traffic.current),
        },
        "percentage_change": percentage_change,
        "coverage": {
            "previous": traffic.previous.observed_page_days
            / (traffic.previous.successful_days * members),
            "current": traffic.current.observed_page_days
            / (traffic.current.successful_days * members),
        },
        "confidence": "robust",
        "impact_score": trend.impact_score,
    }


def _traffic_record(traffic: ClusterTraffic) -> dict[str, object]:
    return {
        "cluster_id": traffic.cluster_id,
        "source_preliminary_cluster_id": traffic.source_preliminary_cluster_id,
        "name": traffic.name,
        "rationale": traffic.rationale,
        "member_page_ids": list(traffic.member_page_ids),
        "previous": asdict(traffic.previous),
        "current": asdict(traffic.current),
        "direction": traffic.direction,
    }


def _load_checkpoint(
    path: Path,
    *,
    run_id: str,
    configuration: dict[str, object],
    selected: tuple[AudienceTrend, ...],
    source_clusters: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ContractError("Trend Portfolio checkpoint is unreadable") from error
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"schema_version", "run_id", "configuration", "completed"}
        or checkpoint["schema_version"] != "1.0"
        or checkpoint["run_id"] != run_id
        or checkpoint["configuration"] != configuration
        or not isinstance(checkpoint["completed"], list)
        or len(checkpoint["completed"]) > len(selected)
    ):
        raise V2ContractError("Trend Portfolio checkpoint conflicts with requested configuration")
    expected_ids = [
        trend.final_cluster_traffic.cluster_id for trend in selected
    ]
    completed = cast(list[dict[str, object]], checkpoint["completed"])
    if any(
        not isinstance(record, dict)
        or set(record) != {"cluster_id", "portfolio_item", "evidence"}
        or record["cluster_id"] != expected_ids[index]
        for index, record in enumerate(completed)
    ):
        raise V2ContractError("Trend Portfolio checkpoint provenance is invalid")
    for trend, record in zip(selected, completed, strict=False):
        cluster_id = trend.final_cluster_traffic.cluster_id
        _validate_completed_record(
            trend,
            record,
            source_clusters[cluster_id],
            cast(str, configuration["model"]),
            conflict_message="Trend Portfolio checkpoint deterministic facts conflict",
        )
    return completed


def _validate_payload(payload: object) -> None:
    try:
        validate_schema(SCHEMA_PATH, payload)
    except jsonschema.ValidationError as error:
        raise V2ContractError(
            f"Trend Portfolio is schema-invalid: {error.message}"
        ) from error


def _validate_completed_facts(
    payload: dict[str, object],
    run_facts: dict[str, object],
    audit_traffic: tuple[ClusterTraffic, ...],
    selected: tuple[AudienceTrend, ...],
    source_clusters: dict[str, dict[str, object]],
) -> None:
    portfolio = cast(list[dict[str, object]], payload["audience_portfolio"])
    evidence = cast(list[dict[str, object]], payload["narrative_evidence"])
    expected_counts = {"qualified": len(selected), "narrated": len(selected)}
    if (
        payload["run_facts"] != run_facts
        or payload["counts"] != expected_counts
        or len(portfolio) != len(selected)
        or len(evidence) != len(selected)
        or payload["audit_cluster_traffic"]
        != [_traffic_record(item) for item in audit_traffic]
    ):
        raise V2ContractError("completed Trend Portfolio deterministic facts conflict")
    for trend, item, narrative_record in zip(
        selected, portfolio, evidence, strict=True
    ):
        cluster_id = trend.final_cluster_traffic.cluster_id
        _validate_completed_record(
            trend,
            {
                "cluster_id": cluster_id,
                "portfolio_item": item,
                "evidence": narrative_record,
            },
            source_clusters[cluster_id],
            cast(str, cast(dict[str, object], payload["configuration"])["model"]),
            conflict_message="completed Trend Portfolio deterministic facts conflict",
        )


def _validate_completed_record(
    trend: AudienceTrend,
    record: dict[str, object],
    source_cluster: dict[str, object],
    model: str,
    *,
    conflict_message: str,
) -> None:
    if set(record) != {"cluster_id", "portfolio_item", "evidence"}:
        raise V2ContractError(conflict_message)
    item = record["portfolio_item"]
    evidence = record["evidence"]
    if not isinstance(item, dict) or not isinstance(evidence, dict):
        raise V2ContractError(conflict_message)
    narrative = item.get("narrative")
    cluster_id = trend.final_cluster_traffic.cluster_id
    members = source_cluster.get("members")
    if not isinstance(narrative, dict) or not isinstance(members, list):
        raise V2ContractError(conflict_message)
    expected_item = _portfolio_item(trend, narrative)
    expected_input = _model_evidence(
        trend,
        cast(list[dict[str, object]], members),
    )
    attempts = evidence.get("attempts")
    if (
        record["cluster_id"] != cluster_id
        or item != expected_item
        or narrative_validation_errors(narrative)
        or set(evidence)
        != {"cluster_id", "prompt", "model_input", "model", "attempts"}
        or evidence["cluster_id"] != cluster_id
        or evidence["prompt"] != NARRATIVE_PROMPT
        or evidence["model_input"] != expected_input
        or evidence["model"] != model
        or not _attempts_are_consistent(attempts, narrative)
    ):
        raise V2ContractError(conflict_message)


def _attempts_are_consistent(attempts: object, narrative: dict[str, object]) -> bool:
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
        return False
    for number, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, dict)
            or set(attempt)
            != {"attempt", "delivery_status", "validation_status", "output", "errors"}
            or attempt["attempt"] != number
            or not isinstance(attempt["errors"], list)
            or any(not isinstance(error, str) for error in attempt["errors"])
        ):
            return False
        delivery = attempt["delivery_status"]
        validation = attempt["validation_status"]
        errors = attempt["errors"]
        if delivery == "error":
            if validation != "not_run" or attempt["output"] is not None or not errors:
                return False
        elif delivery == "delivered" and validation == "invalid":
            if not errors:
                return False
        elif delivery == "delivered" and validation == "valid":
            if errors or number != len(attempts) or attempt["output"] != narrative:
                return False
        else:
            return False
    return cast(dict[str, object], attempts[-1])["validation_status"] == "valid"


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"
