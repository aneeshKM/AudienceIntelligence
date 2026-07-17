from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import cast

import jsonschema

from audience_trend_miner.v2.shared import (
    BoundedProgress,
    ProgressEvent,
    ProgressSink,
    V2ContractError,
    atomic_write_json,
    canonical_json_fingerprint,
    consume_artifact,
    validate_identifier,
    validate_schema,
)


STAGE = "run-publication"
UPSTREAM_STAGES = (
    "wikimedia-evidence",
    "semantic-audience-formation",
    "cluster-adjudication",
    "trend-portfolio",
)
PACKAGE_DIRECTORY = Path(__file__).parent
SCHEMA_DIRECTORY = PACKAGE_DIRECTORY / "schemas"
V2_DIRECTORY = PACKAGE_DIRECTORY.parent
UPSTREAM_SCHEMAS = {
    "wikimedia-evidence": V2_DIRECTORY
    / "wikimedia_evidence/schemas/wikimedia-evidence.schema.json",
    "semantic-audience-formation": V2_DIRECTORY
    / "semantic_audience_formation/schemas/semantic-audience-formation.schema.json",
    "cluster-adjudication": V2_DIRECTORY
    / "cluster_adjudication/schemas/cluster-adjudication.schema.json",
    "trend-portfolio": V2_DIRECTORY
    / "trend_portfolio/schemas/trend-portfolio.schema.json",
}
FINAL_SCHEMAS = {
    "portfolio.json": SCHEMA_DIRECTORY / "portfolio.schema.json",
    "audit.json": SCHEMA_DIRECTORY / "audit.schema.json",
    "manifest.json": SCHEMA_DIRECTORY / "manifest.schema.json",
}
FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "chain_of_thought",
        "database_url",
        "hidden_reasoning",
        "password",
        "secret",
        "token",
    }
)


def execute_run_publication(
    *,
    run_id: str,
    output_root: Path,
    progress_sink: ProgressSink,
    upstream_paths: dict[str, Path] | None = None,
    interrupt_before_completion: bool = False,
    fail_after_artifact: int | None = None,
) -> Path:
    """Validate one completed run and atomically expose its final contract."""
    validate_identifier(run_id, "run_id")
    run_directory = output_root / run_id
    publication_directory = run_directory / "publication"
    paths = upstream_paths or {}
    artifacts = {
        stage: consume_artifact(
            paths.get(stage, run_directory / f"{stage}.json"),
            run_id=run_id,
            stage=stage,
        )
        for stage in UPSTREAM_STAGES
    }
    _validate_upstream(artifacts)
    _ensure_safe(artifacts)

    if os.path.lexists(publication_directory):
        _validate_existing_publication(publication_directory, run_id, artifacts)
        _emit(
            progress_sink,
            run_id,
            1,
            "resume",
            "resumed compatible completed Run Publication",
            1,
            1,
        )
        return publication_directory

    _emit(
        progress_sink,
        run_id,
        1,
        "validate",
        "validated four compatible upstream artifacts",
        1,
        3,
    )
    portfolio, audit = _assemble_products(run_id, artifacts)
    run_directory.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(prefix=".publication.", dir=run_directory)
    )
    completed_directory = staging_directory.with_name(
        f"{staging_directory.name}.complete"
    )
    completed_owned = False
    published = False
    try:
        products = {"portfolio.json": portfolio, "audit.json": audit}
        for index, (name, product) in enumerate(products.items(), start=1):
            validate_schema(FINAL_SCHEMAS[name], product)
            atomic_write_json(staging_directory / name, product)
            if fail_after_artifact == index:
                raise V2ContractError("publication write failed")

        manifest = _manifest(run_id, artifacts, portfolio, staging_directory)
        validate_schema(FINAL_SCHEMAS["manifest.json"], manifest)
        atomic_write_json(staging_directory / "manifest.json", manifest)
        if fail_after_artifact == 3:
            raise V2ContractError("publication write failed")
        _validate_staged_publication(staging_directory, run_id, artifacts)
        _emit(
            progress_sink,
            run_id,
            2,
            "stage",
            "staged and validated exact final artifact set",
            2,
            3,
        )
        if interrupt_before_completion:
            raise V2ContractError("publication interrupted before completion")
        os.rename(staging_directory, completed_directory)
        completed_owned = True
        try:
            os.symlink(
                completed_directory.name,
                publication_directory,
                target_is_directory=True,
            )
        except OSError as error:
            raise V2ContractError("publication collision prevented completion") from error
        published = True
        _sync_directory(run_directory)
    finally:
        _remove_staging_directory(staging_directory)
        if completed_owned and not published:
            _remove_staging_directory(completed_directory)

    _emit(
        progress_sink,
        run_id,
        3,
        "publish",
        "atomically published portfolio.json, audit.json, and manifest.json",
        3,
        3,
    )
    return publication_directory


def _validate_upstream(artifacts: dict[str, dict[str, object]]) -> None:
    for stage, artifact in artifacts.items():
        try:
            validate_schema(UPSTREAM_SCHEMAS[stage], artifact["payload"])
        except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as error:
            message = getattr(error, "message", str(error))
            raise V2ContractError(
                f"{stage} artifact is schema-incompatible: {message}"
            ) from error

    evidence = artifacts["wikimedia-evidence"]
    formation = artifacts["semantic-audience-formation"]
    adjudication = artifacts["cluster-adjudication"]
    trend = artifacts["trend-portfolio"]
    evidence_payload = _payload(evidence)
    formation_payload = _payload(formation)
    adjudication_payload = _payload(adjudication)
    trend_payload = _payload(trend)
    formation_configuration = _mapping(formation_payload["configuration"])
    adjudication_configuration = _mapping(adjudication_payload["configuration"])
    trend_configuration = _mapping(trend_payload["configuration"])

    if (
        formation_configuration["wikimedia_evidence_fingerprint"]
        != _semantic_evidence_fingerprint(evidence)
        or adjudication_configuration["semantic_audience_formation_fingerprint"]
        != canonical_json_fingerprint(formation)
        or trend_configuration["wikimedia_evidence_fingerprint"]
        != canonical_json_fingerprint(evidence)
        or trend_configuration["cluster_adjudication_fingerprint"]
        != canonical_json_fingerprint(adjudication)
    ):
        raise V2ContractError("upstream artifacts are incompatible")

    expected_run_facts = {
        "as_of_date": evidence_payload["as_of_date"],
        "nominal_windows": evidence_payload["nominal_windows"],
    }
    if trend_payload["run_facts"] != expected_run_facts:
        raise V2ContractError("upstream artifacts contain mismatched run facts")
    _validate_counts_and_membership(
        formation_payload, adjudication_payload, trend_payload
    )


def _validate_counts_and_membership(
    formation: dict[str, object],
    adjudication: dict[str, object],
    trend: dict[str, object],
) -> None:
    preliminary = _list_of_mappings(formation["preliminary_clusters"])
    formation_counts = _mapping(formation["counts"])
    if (
        cast(int, formation_counts["selected_clusters"]) != len(preliminary)
        or cast(int, formation_counts["eligible_clusters"])
        != cast(int, formation_counts["selected_clusters"])
        + cast(int, formation_counts["omitted_clusters"])
    ):
        raise V2ContractError("Semantic Audience Formation counts are inconsistent")

    final_clusters = _list_of_mappings(adjudication["final_audience_clusters"])
    rejected = _list_of_mappings(adjudication["rejected_members"])
    adjudications = _list_of_mappings(adjudication["adjudications"])
    adjudication_counts = _mapping(adjudication["counts"])
    accepted_pages = sum(
        len(_list_of_mappings(cluster["members"])) for cluster in final_clusters
    )
    expected_preliminary_ids = [
        f"preliminary-cluster-{index:04d}"
        for index in range(1, len(preliminary) + 1)
    ]
    expected_final_ids = [
        f"final-audience-cluster-{index:04d}"
        for index in range(1, len(final_clusters) + 1)
    ]
    if adjudication_counts != {
        "preliminary_clusters": len(preliminary),
        "final_audience_clusters": len(final_clusters),
        "accepted_pages": accepted_pages,
        "rejected_pages": len(rejected),
    } or [
        record["preliminary_cluster_id"] for record in adjudications
    ] != expected_preliminary_ids or [
        cluster["cluster_id"] for cluster in final_clusters
    ] != expected_final_ids:
        raise V2ContractError("Cluster Adjudication counts are inconsistent")
    for source_id, source_cluster in zip(
        expected_preliminary_ids, preliminary, strict=True
    ):
        expected_page_ids = [
            member["page_id"]
            for member in _list_of_mappings(source_cluster["members"])
        ]
        terminal_page_ids = [
            member["page_id"]
            for cluster in final_clusters
            if cluster["source_preliminary_cluster_id"] == source_id
            for member in _list_of_mappings(cluster["members"])
        ] + [
            member["page_id"]
            for member in rejected
            if member["source_preliminary_cluster_id"] == source_id
        ]
        if (
            len(terminal_page_ids) != len(set(terminal_page_ids))
            or set(terminal_page_ids) != set(expected_page_ids)
        ):
            raise V2ContractError(
                "Cluster Adjudication terminal provenance is inconsistent"
            )
    valid_sources = set(expected_preliminary_ids)
    if any(
        cluster["source_preliminary_cluster_id"] not in valid_sources
        for cluster in final_clusters
    ) or any(
        member["source_preliminary_cluster_id"] not in valid_sources
        for member in rejected
    ):
        raise V2ContractError("Cluster Adjudication terminal provenance is inconsistent")

    portfolio = _list_of_mappings(trend["audience_portfolio"])
    narratives = _list_of_mappings(trend["narrative_evidence"])
    traffic = _list_of_mappings(trend["audit_cluster_traffic"])
    trend_counts = _mapping(trend["counts"])
    if trend_counts != {"qualified": len(portfolio), "narrated": len(narratives)}:
        raise V2ContractError("Trend Portfolio counts are inconsistent")
    final_by_id = {cluster["cluster_id"]: cluster for cluster in final_clusters}
    portfolio_ids = [item["cluster_id"] for item in portfolio]
    traffic_ids = [record["cluster_id"] for record in traffic]
    if (
        len(portfolio_ids) != len(set(portfolio_ids))
        or [record["cluster_id"] for record in narratives] != portfolio_ids
        or len(traffic_ids) != len(set(traffic_ids))
        or set(traffic_ids) != set(final_by_id)
    ):
        raise V2ContractError("Trend Portfolio membership is inconsistent")
    for item in portfolio:
        source = final_by_id.get(item["cluster_id"])
        if source is None or (
            item["source_preliminary_cluster_id"]
            != source["source_preliminary_cluster_id"]
        ):
            raise V2ContractError("Trend Portfolio provenance is inconsistent")


def _assemble_products(
    run_id: str, artifacts: dict[str, dict[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    trend = _payload(artifacts["trend-portfolio"])
    run_facts = _mapping(trend["run_facts"])
    audience_portfolio = trend["audience_portfolio"]
    assert isinstance(audience_portfolio, list)
    portfolio: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "as_of_date": run_facts["as_of_date"],
        "nominal_windows": run_facts["nominal_windows"],
        "audience_portfolio": audience_portfolio,
        "completion": {
            "status": "complete",
            "empty": len(audience_portfolio) == 0,
        },
    }
    audit: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "stage_evidence": {
            stage: _audit_stage_evidence(stage, _payload(artifact))
            for stage, artifact in artifacts.items()
        },
    }
    return portfolio, audit


def _audit_stage_evidence(
    stage: str, payload: dict[str, object]
) -> dict[str, object]:
    evidence = deepcopy(payload)
    if stage != "trend-portfolio":
        return evidence
    for narrative in _list_of_mappings(evidence["narrative_evidence"]):
        for attempt in _list_of_mappings(narrative["attempts"]):
            attempt.pop("output", None)
    return evidence


def _manifest(
    run_id: str,
    artifacts: dict[str, dict[str, object]],
    portfolio: dict[str, object],
    staging_directory: Path,
) -> dict[str, object]:
    evidence = _payload(artifacts["wikimedia-evidence"])
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "as_of_date": portfolio["as_of_date"],
        "nominal_windows": portfolio["nominal_windows"],
        "configuration_provenance": {
            "wikimedia-evidence": evidence["provenance"],
            **{
                stage: _payload(artifacts[stage])["configuration"]
                for stage in UPSTREAM_STAGES[1:]
            },
        },
        "modules": _module_integrity(artifacts),
        "schemas": {name: "1.0" for name in FINAL_SCHEMAS},
        "published_artifacts": {
            name: {
                "schema_version": "1.0",
                "sha256": _file_fingerprint(staging_directory / name),
                "bytes": (staging_directory / name).stat().st_size,
            }
            for name in ("portfolio.json", "audit.json")
        },
        "integrity": {
            "algorithm": "sha256",
            "encoding": "utf-8",
            "manifest_excludes_self": True,
        },
        "completion": {"status": "complete"},
    }


def _validate_existing_publication(
    directory: Path,
    run_id: str,
    artifacts: dict[str, dict[str, object]],
) -> None:
    try:
        _validate_staged_publication(directory, run_id, artifacts)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError, V2ContractError) as error:
        raise V2ContractError(
            "existing publication collides with requested run"
        ) from error


def _validate_staged_publication(
    directory: Path,
    run_id: str,
    artifacts: dict[str, dict[str, object]],
) -> None:
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != set(
        FINAL_SCHEMAS
    ):
        raise V2ContractError("publication does not contain the exact artifact set")
    products: dict[str, dict[str, object]] = {}
    for name, schema in FINAL_SCHEMAS.items():
        loaded = json.loads((directory / name).read_text(encoding="utf-8"))
        validate_schema(schema, loaded)
        if not isinstance(loaded, dict) or loaded.get("run_id") != run_id:
            raise V2ContractError("published artifact belongs to different run facts")
        products[name] = loaded
    _ensure_safe(products)
    expected_portfolio, expected_audit = _assemble_products(run_id, artifacts)
    if (
        products["portfolio.json"] != expected_portfolio
        or products["audit.json"] != expected_audit
    ):
        raise V2ContractError("published artifacts are internally inconsistent")
    manifest = products["manifest.json"]
    if manifest["modules"] != _module_integrity(artifacts):
        raise V2ContractError("publication provenance does not match upstream artifacts")
    expected_manifest = _manifest(
        run_id, artifacts, expected_portfolio, directory
    )
    if manifest != expected_manifest:
        raise V2ContractError("manifest provenance is internally inconsistent")
    published = _mapping(manifest["published_artifacts"])
    for name in ("portfolio.json", "audit.json"):
        record = _mapping(published[name])
        if (
            record["sha256"] != _file_fingerprint(directory / name)
            or record["bytes"] != (directory / name).stat().st_size
        ):
            raise V2ContractError("published artifact integrity check failed")


def _module_integrity(
    artifacts: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        stage: {
            "status": "complete",
            "artifact_schema_version": artifact["schema_version"],
            "sha256": canonical_json_fingerprint(artifact),
        }
        for stage, artifact in artifacts.items()
    }


def _semantic_evidence_fingerprint(artifact: dict[str, object]) -> str:
    payload = _payload(artifact)
    pages = _list_of_mappings(payload["canonical_pages"])
    evidence = sorted(
        (
            {
                "page_id": page["page_id"],
                "canonical_title": page["canonical_title"],
                "lead": page["lead"],
                "categories": sorted(set(cast(list[str], page["categories"]))),
            }
            for page in pages
        ),
        key=lambda page: (page["page_id"], page["canonical_title"]),
    )
    return canonical_json_fingerprint(evidence)


def _ensure_safe(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in FORBIDDEN_KEYS:
                raise V2ContractError(
                    f"published artifacts contain prohibited field {key!r}"
                )
            _ensure_safe(nested)
    elif isinstance(value, list):
        for nested in value:
            _ensure_safe(nested)


def _payload(artifact: dict[str, object]) -> dict[str, object]:
    return _mapping(artifact["payload"])


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V2ContractError("artifact contains an invalid object")
    return value


def _list_of_mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise V2ContractError("artifact contains an invalid collection")
    return cast(list[dict[str, object]], value)


def _file_fingerprint(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _remove_staging_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _emit(
    sink: ProgressSink,
    run_id: str,
    sequence: int,
    operation: str,
    message: str,
    current: int,
    total: int,
) -> None:
    sink(
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
