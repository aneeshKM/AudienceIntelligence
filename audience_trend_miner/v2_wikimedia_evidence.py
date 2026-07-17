from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Mapping

from audience_trend_miner.v2_contracts import (
    ARTIFACT_SCHEMA_VERSION,
    BoundedProgress,
    ProgressEvent,
    ProgressSink,
    V2ContractError,
    _atomic_write_json,
    _safe_identifier,
    _validate,
    record_run_configuration,
    validate_artifact,
)


STAGE = "wikimedia-evidence"
MINIMUM_SUCCESSFUL_DAYS = 4


def execute_wikimedia_evidence_fixture(
    *,
    run_id: str,
    as_of_date: date,
    output_root: Path,
    fixture_path: Path,
    progress_sink: ProgressSink,
) -> Path:
    _safe_identifier(run_id, "run_id")
    fixture = _load_fixture(fixture_path)
    previous_start = as_of_date - timedelta(days=15)
    previous_end = as_of_date - timedelta(days=9)
    current_start = as_of_date - timedelta(days=8)
    current_end = as_of_date - timedelta(days=2)
    nominal_dates = tuple(
        previous_start + timedelta(days=offset) for offset in range(14)
    )

    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    record_run_configuration(
        run_directory,
        run_id,
        {"as_of": as_of_date.isoformat(), "wikimedia_mode": "fixture"},
    )

    daily_responses = fixture["daily_responses"]
    nominal_days: list[dict[str, object]] = []
    daily_cutoffs: list[dict[str, object]] = []
    candidate_titles: set[str] = set()
    observations_by_alias: dict[str, list[dict[str, object]]] = {}
    unavailable_days: list[str] = []
    coverage = {"previous": 0, "current": 0}
    non_en_wikipedia_records = 0

    for sequence, day in enumerate(nominal_dates, start=1):
        day_text = day.isoformat()
        response = daily_responses.get(day_text)
        window = "previous" if day <= previous_end else "current"
        if not isinstance(response, dict) or "error" in response:
            nominal_days.append(
                {"date": day_text, "window": window, "status": "unavailable"}
            )
            unavailable_days.append(day_text)
        else:
            records = response.get("records")
            if not isinstance(records, list):
                raise V2ContractError(f"fixture records are invalid for {day_text}")
            coverage[window] += 1
            nominal_days.append(
                {"date": day_text, "window": window, "status": "successful"}
            )
            en_records = [
                record
                for record in records
                if isinstance(record, dict) and record.get("project") == "en.wikipedia"
            ]
            non_en_wikipedia_records += len(records) - len(en_records)
            cutoff = min(
                (
                    int(record["views_ceil"])
                    for record in records
                    if isinstance(record, dict)
                ),
                default=None,
            )
            daily_cutoffs.append({"date": day_text, "views_ceil": cutoff})
            for record in en_records:
                title = str(record["article"])
                candidate_titles.add(title)
                observations_by_alias.setdefault(title, []).append(
                    {"date": day_text, "views_ceil": int(record["views_ceil"])}
                )
        progress_sink(
            ProgressEvent(
                run_id=run_id,
                sequence=sequence,
                timestamp=f"{day_text}T00:00:00+00:00",
                module=STAGE,
                operation="acquire",
                level="info",
                message=f"processed {day_text}",
                progress=BoundedProgress(sequence, 15),
            )
        )

    for window, successful_days in coverage.items():
        if successful_days < MINIMUM_SUCCESSFUL_DAYS:
            raise V2ContractError(
                f"{window} Effective Window has {successful_days} successful days; "
                f"at least {MINIMUM_SUCCESSFUL_DAYS} are required"
            )

    canonical_pages = _canonicalize(
        candidate_titles, observations_by_alias, fixture["canonical_pages"]
    )
    payload = {
        "as_of_date": as_of_date.isoformat(),
        "nominal_windows": {
            "previous": {
                "start": previous_start.isoformat(),
                "end": previous_end.isoformat(),
            },
            "current": {
                "start": current_start.isoformat(),
                "end": current_end.isoformat(),
            },
        },
        "nominal_days": nominal_days,
        "coverage": coverage,
        "candidate_universe": sorted(candidate_titles),
        "canonical_pages": canonical_pages,
        "daily_cutoffs": daily_cutoffs,
        "provenance": {
            "source": fixture["source"],
            "country": "US",
            "project": "en.wikipedia",
            "traffic_measure": "views_ceil",
        },
        "exclusions": {
            "non_en_wikipedia_records": non_en_wikipedia_records,
            "unavailable_days": unavailable_days,
        },
        "completion": {
            "status": "complete",
            "minimum_successful_days_per_window": MINIMUM_SUCCESSFUL_DAYS,
        },
    }
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "complete",
        "payload": payload,
    }
    _validate("v2-wikimedia-evidence.schema.json", payload)
    validate_artifact(artifact, run_id=run_id, stage=STAGE)
    artifact_path = run_directory / f"{STAGE}.json"
    _atomic_write_json(artifact_path, artifact)
    progress_sink(
        ProgressEvent(
            run_id=run_id,
            sequence=15,
            timestamp=f"{as_of_date.isoformat()}T00:00:00+00:00",
            module=STAGE,
            operation="publish",
            level="info",
            message="published complete Wikimedia Evidence",
            progress=BoundedProgress(15, 15),
        )
    )
    return artifact_path


def _canonicalize(
    candidate_titles: set[str],
    observations_by_alias: Mapping[str, list[dict[str, object]]],
    canonical_facts: object,
) -> list[dict[str, object]]:
    if not isinstance(canonical_facts, dict):
        raise V2ContractError("fixture canonical_pages are invalid")
    grouped: dict[int, dict[str, object]] = {}
    for alias in sorted(candidate_titles):
        facts = canonical_facts.get(alias)
        if not isinstance(facts, dict):
            raise V2ContractError(f"fixture has no canonical identity for {alias}")
        page_id = int(facts["page_id"])
        page = grouped.setdefault(
            page_id,
            {
                "page_id": page_id,
                "canonical_title": str(facts["canonical_title"]),
                "aliases": [],
                "observations": [],
            },
        )
        if page["canonical_title"] != str(facts["canonical_title"]):
            raise V2ContractError(f"canonical title conflict for page {page_id}")
        page["aliases"].append(alias)
        page["observations"].extend(observations_by_alias.get(alias, []))
    for page in grouped.values():
        observations_by_date: dict[str, int] = {}
        for observation in page["observations"]:
            observation_date = str(observation["date"])
            observations_by_date[observation_date] = (
                observations_by_date.get(observation_date, 0)
                + int(observation["views_ceil"])
            )
        page["observations"] = [
            {
                "date": observation_date,
                "views_ceil": observations_by_date[observation_date],
            }
            for observation_date in sorted(observations_by_date)
        ]
    return [grouped[page_id] for page_id in sorted(grouped)]


def _load_fixture(path: Path) -> dict[str, object]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V2ContractError("Wikimedia Evidence fixture is unreadable") from error
    if (
        not isinstance(fixture, dict)
        or fixture.get("schema_version") != "1.0"
        or not isinstance(fixture.get("source"), str)
        or not isinstance(fixture.get("daily_responses"), dict)
    ):
        raise V2ContractError("Wikimedia Evidence fixture has an invalid shape")
    return fixture
